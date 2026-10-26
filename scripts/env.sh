#!/usr/bin/env bash
# Source this file when running an experiment: source scripts/env.sh
if [[ -d /root/autodl-tmp && -w /root/autodl-tmp ]]; then
    export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
    export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/autodl-tmp/cache/uv}"
    export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/root/autodl-tmp/cache/uv-python}"
fi
