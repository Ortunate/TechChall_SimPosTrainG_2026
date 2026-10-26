
#!/usr/bin/env bash
# Technical Challenge 2026 - WSL2 / AutoDL environment setup.
#
# Usage:
#   bash scripts/setup_env.sh base
#   bash scripts/setup_env.sh train
#   bash scripts/setup_env.sh vllm
#   bash scripts/setup_env.sh all
#
# base  : Install system tools, uv, and Python 3.12.
# train : Prepare the locked training environment (.venv).
# vllm  : Prepare the independent inference environment (.venv-vllm).
# all   : Prepare vLLM first, then the training environment.

set -Eeuo pipefail

PROFILE="${1:-base}"

case "$PROFILE" in
    base|train|vllm|all) ;;
    *)
        echo "Usage: $0 [base|train|vllm|all]" >&2
        exit 2
        ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

log() {
    printf '\n[setup] %s\n' "$*"
}

fail() {
    printf '[setup] ERROR: %s\n' "$*" >&2
    exit 1
}

# ------------------------------------------------------------
# 1. Load environment configuration.
# ------------------------------------------------------------

[[ -f "$ROOT/scripts/env.sh" ]] \
    || fail "Missing scripts/env.sh"

# Load package mirrors and cache paths from the shared script.
source "$ROOT/scripts/env.sh"

# Default mirror for ordinary Python packages.
VLLM_PYPI_INDEX="${UV_DEFAULT_INDEX:-https://mirror.nju.edu.cn/pypi/web/simple}"

# CUDA 12.9 PyTorch mirror for the independent vLLM environment.
VLLM_TORCH_INDEX="${VLLM_TORCH_INDEX:-https://mirrors.nju.edu.cn/pytorch/whl/cu129}"

# Apply conservative download settings if not already configured.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"

# Ensure large caches remain outside the AutoDL system disk.
if [[ -d /root/autodl-tmp && -w /root/autodl-tmp ]]; then
    export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
    export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/autodl-tmp/cache/uv}"
    export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/root/autodl-tmp/cache/uv-python}"
else
    export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
    export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
fi

mkdir -p "$HF_HOME" "$UV_CACHE_DIR"

if [[ -n "${UV_PYTHON_INSTALL_DIR:-}" ]]; then
    mkdir -p "$UV_PYTHON_INSTALL_DIR"
fi

log "Selected profile: $PROFILE"
log "Python package index: $VLLM_PYPI_INDEX"
log "vLLM PyTorch index: $VLLM_TORCH_INDEX"
log "HF_HOME: $HF_HOME"
log "UV_CACHE_DIR: $UV_CACHE_DIR"


# ------------------------------------------------------------
# 2. Validate operating system and GPU visibility.
# ------------------------------------------------------------

[[ -f /etc/os-release ]] \
    || fail "Cannot detect the operating system."

source /etc/os-release

[[ "${ID:-}" == "ubuntu" ]] \
    || fail "This script requires Ubuntu."

[[ "$(uname -m)" == "x86_64" ]] \
    || fail "This script requires x86_64."

# GPU access is optional during environment installation.
if command -v nvidia-smi >/dev/null 2>&1; then
    log "Checking accessible NVIDIA GPUs..."

    if ! nvidia-smi -L; then
        log "No accessible GPU. CPU-only installation may continue."
    fi
else
    log "nvidia-smi unavailable. GPU verification is deferred."
fi

# The CUDA Toolkit compiler is not required for prebuilt wheels.
if command -v nvcc >/dev/null 2>&1; then
    log "CUDA Toolkit detected:"
    nvcc --version | tail -n 1
else
    log "CUDA Toolkit compiler not found; this is acceptable."
fi


# ------------------------------------------------------------
# 3. Install missing Ubuntu system packages.
# ------------------------------------------------------------

packages=(
    ca-certificates
    curl
    git
    openssh-client
    tmux
    build-essential
)

missing=()

for package in "${packages[@]}"; do
    if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null \
        | grep -qx 'install ok installed'; then
        missing+=("$package")
    fi
done

if (( ${#missing[@]} > 0 )); then
    if (( EUID == 0 )); then
        admin=()
    else
        command -v sudo >/dev/null 2>&1 \
            || fail "sudo is required to install system packages."
        admin=(sudo)
    fi

    # APT requires /tmp to be writable by the unprivileged _apt user.
    if (( EUID == 0 )) \
        && id _apt >/dev/null 2>&1 \
        && command -v runuser >/dev/null 2>&1; then

        runuser -u _apt -- sh -c \
            'f=$(mktemp /tmp/apt.conf.XXXXXX) && rm -f "$f"' \
            || fail "APT cannot write to /tmp. Check permissions (1777) and disk space."
    fi

    log "Installing missing system packages: ${missing[*]}"

    "${admin[@]}" apt-get update
    "${admin[@]}" apt-get install -y "${missing[@]}"
else
    log "All required Ubuntu packages are installed."
fi


# ------------------------------------------------------------
# 4. Install uv and locate Python 3.12.
# ------------------------------------------------------------

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv..."

    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

command -v uv >/dev/null 2>&1 \
    || fail "uv installation failed."

log "uv version: $(uv --version)"

if ! uv python find 3.12 >/dev/null 2>&1; then
    log "Installing Python 3.12..."
    uv python install 3.12
fi

log "Python interpreter: $(uv python find 3.12)"


# ------------------------------------------------------------
# 5. Install the independent vLLM inference environment.
# ------------------------------------------------------------

if [[ "$PROFILE" == "vllm" || "$PROFILE" == "all" ]]; then

    [[ -f requirements/vllm.in ]] \
        || fail "Missing requirements/vllm.in"

    log "Preparing vLLM inference environment: .venv-vllm"

    # Keep inference dependencies separate from the training lockfile.
    if [[ ! -f .venv-vllm/pyvenv.cfg ]]; then
        uv venv --python 3.12 .venv-vllm
    fi

    # The vLLM prebuilt CUDA wheels currently target CUDA 12.9.
    #
    # --no-config:
    #   Ignore the training project's uv index configuration.
    #
    # --default-index:
    #   Resolve ordinary dependencies through the NJU PyPI mirror.
    #
    # --index:
    #   Give the NJU CUDA 12.9 PyTorch index higher priority.
    #
    # --only-binary vllm:
    #   Do not compile vLLM from source.
    #
    # UV_TORCH_BACKEND is explicitly unset because it would override
    # the custom PyTorch mirror, even without --torch-backend.

    env -u UV_TORCH_BACKEND uv pip install \
        --no-config \
        --python .venv-vllm/bin/python \
        --default-index "$VLLM_PYPI_INDEX" \
        --index "$VLLM_TORCH_INDEX" \
        --only-binary vllm \
        -r requirements/vllm.in

    log "Checking installed inference dependencies..."

    # Inspect installation metadata without requiring an active GPU.
    .venv-vllm/bin/python -c '
import importlib.metadata as metadata
import torch

print("vLLM:", metadata.version("vllm"))
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())
'

    mkdir -p outputs/environments

    # Record actual installed versions for experiment reproducibility.
    uv pip freeze \
        --python .venv-vllm/bin/python \
        > outputs/environments/vllm-freeze.txt

    log "vLLM environment installed."
    log "Dependency snapshot: outputs/environments/vllm-freeze.txt"
fi


# ------------------------------------------------------------
# 6. Synchronize the locked training environment.
# ------------------------------------------------------------

if [[ "$PROFILE" == "train" || "$PROFILE" == "all" ]]; then

    [[ -f pyproject.toml ]] \
        || fail "Missing pyproject.toml"

    [[ -f uv.lock ]] \
        || fail "Missing uv.lock. Generate and verify it first."

    log "Synchronizing training environment: .venv"

    # Preserve the exact dependency versions and sources in uv.lock.
    uv sync --locked

    log "Checking installed training dependencies..."

    .venv/bin/python -c '
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())
'

    mkdir -p outputs/environments

    uv pip freeze \
        --python .venv/bin/python \
        > outputs/environments/train-freeze.txt

    log "Training environment synchronized."
    log "Dependency snapshot: outputs/environments/train-freeze.txt"
fi


# ------------------------------------------------------------
# 7. Final instructions.
# ------------------------------------------------------------

log "Setup completed successfully: $PROFILE"

log "Training Python: $ROOT/.venv/bin/python"
log "Inference Python: $ROOT/.venv-vllm/bin/python"

log "Before running experiments in a new shell:"
log "  cd $ROOT"
log "  source scripts/env.sh"

log "GPU model loading must be verified after GPU allocation."
