#!/usr/bin/env bash
# Technical Challenge 2026 - Ubuntu 22.04 / WSL2 / AutoDL container setup
# Usage: bash scripts/setup_env.sh [train|vllm|all]
set -Eeuo pipefail

PROFILE="${1:-train}"
case "$PROFILE" in train|vllm|all) ;; *) echo "Usage: $0 [train|vllm|all]" >&2; exit 2;; esac
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

log() { printf '\n[setup] %s\n' "$*"; }
fail() { printf '[setup] ERROR: %s\n' "$*" >&2; exit 1; }

[[ -f /etc/os-release ]] || fail 'Only Ubuntu Linux is supported.'
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "ubuntu" ]] || fail "Expected Ubuntu; found ${ID:-unknown}."
[[ "$(uname -m)" == "x86_64" ]] || fail 'This setup targets x86_64.'

if [[ -d /root/autodl-tmp && -w /root/autodl-tmp ]]; then
    # AutoDL: put large caches and downloaded Python runtimes on the data disk.
    export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
    export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/autodl-tmp/cache/uv}"
    export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/root/autodl-tmp/cache/uv-python}"
else
    export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
    export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
fi
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"
if [[ -n "${UV_PYTHON_INSTALL_DIR:-}" ]]; then mkdir -p "$UV_PYTHON_INSTALL_DIR"; fi

# Do not install a Linux NVIDIA driver or CUDA toolkit inside WSL2/AutoDL containers.
if command -v nvidia-smi >/dev/null 2>&1; then
    log 'GPU driver check (driver-managed CUDA version, not toolkit version):'
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
elif [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
    log 'WSL GPU driver check:'
    /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
    log 'WARNING: nvidia-smi not found. CPU-only checks can run, but GPU tests require host GPU access.'
fi
if command -v nvcc >/dev/null 2>&1; then
    log "CUDA compiler detected: $(nvcc --version | tail -n 1)"
else
    log 'No nvcc compiler found. This is OK for precompiled PyTorch/vLLM wheels.'
fi

# System utilities: check before installation; apt operations run sequentially.
pkgs=(ca-certificates curl git openssh-client tmux build-essential)
missing=()
for p in "${pkgs[@]}"; do
    if ! dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -qx 'install ok installed'; then
        missing+=("$p")
    fi
done
if (( ${#missing[@]} > 0 )); then
    command -v apt-get >/dev/null 2>&1 || fail 'apt-get unavailable.'
    if (( EUID == 0 )); then admin=();
    else command -v sudo >/dev/null 2>&1 || fail 'sudo is required to install missing system packages.'; admin=(sudo); fi
    log "Installing missing Ubuntu packages: ${missing[*]}"
    "${admin[@]}" apt-get update
    "${admin[@]}" apt-get install -y "${missing[@]}"
else
    log 'All required Ubuntu system packages already installed.'
fi

if ! command -v uv >/dev/null 2>&1; then
    log 'Installing uv from the official installer...'
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || fail 'uv still not on PATH; check installation and ~/.local/bin.'
log "uv: $(uv --version)"
if ! uv python find 3.12 >/dev/null 2>&1; then
    log 'Installing Python 3.12 with uv...'
    uv python install 3.12
fi
uv python find 3.12

if [[ "$PROFILE" == train || "$PROFILE" == all ]]; then
    [[ -f pyproject.toml ]] || fail 'Missing pyproject.toml: declare train dependencies first.'
    log 'Synchronizing training/development environment: .venv/'
    if [[ -f uv.lock ]]; then
        uv sync --locked
    else
        log 'No uv.lock found; generating initial lockfile.'
        uv lock
        uv sync --locked
    fi
    .venv/bin/python -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, "GPU available:", torch.cuda.is_available())'
fi

if [[ "$PROFILE" == vllm || "$PROFILE" == all ]]; then
    [[ -f requirements/vllm.in ]] || fail 'Missing requirements/vllm.in.'
    log 'Preparing independent vLLM environment: .venv-vllm/'
    if [[ ! -f .venv-vllm/pyvenv.cfg ]]; then uv venv --python 3.12 .venv-vllm; fi
    uv pip install --python .venv-vllm/bin/python --torch-backend=auto -r requirements/vllm.in
    .venv-vllm/bin/python -c 'import vllm,torch; print("vLLM:",vllm.__version__,"PyTorch:",torch.__version__,"CUDA:",torch.version.cuda)'
    mkdir -p outputs/environments
    uv pip freeze --python .venv-vllm/bin/python > outputs/environments/vllm-freeze.txt
    log 'vLLM dependency snapshot: outputs/environments/vllm-freeze.txt'
fi

log 'Completed. Use .venv/bin/python for training, .venv-vllm/bin/python for vLLM.'
log "HF_HOME=$HF_HOME"
log 'Note: exported paths apply only to this setup process; use scripts/env.sh when running experiments.'
