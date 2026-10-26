#!/usr/bin/env bash
# Source this file when running an experiment: source scripts/env.sh
if [[ -d /root/autodl-tmp && -w /root/autodl-tmp ]]; then
    export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
    export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/autodl-tmp/cache/uv}"
    export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/root/autodl-tmp/cache/uv-python}"
fi

# Use mainland China mirrors only inside AutoDL instances.
if [[ -d "/root/autodl-tmp" ]]; then
    # Python package index for uv.
    export UV_DEFAULT_INDEX="https://mirror.nju.edu.cn/pypi/web/simple"

    # Hugging Face model and dataset mirror.
    export HF_ENDPOINT="https://hf-mirror.com"

    # Network timeout for slow package downloads.
    export UV_HTTP_TIMEOUT=120
    export UV_CONCURRENT_DOWNLOADS=3
    
    # Keep large caches on the data disk.
    export HF_HOME="/root/autodl-tmp/cache/huggingface"
    export UV_CACHE_DIR="/root/autodl-tmp/cache/uv"

    mkdir -p "$HF_HOME" "$UV_CACHE_DIR"
fi