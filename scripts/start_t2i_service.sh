#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}/scripts:${PYTHONPATH:-}"

# ---- Service defaults (override via environment) ----
export T2I_MODEL_PATH="${T2I_MODEL_PATH:-/nas/models/t2i/qwen-image-2512/Qwen/Qwen-Image-2512}"
export T2I_CUDA_DEVICES="${T2I_CUDA_DEVICES:-0,1}"
export T2I_HOST="${T2I_HOST:-0.0.0.0}"
export T2I_PORT="${T2I_PORT:-8100}"
export T2I_MAX_QUEUE_SIZE="${T2I_MAX_QUEUE_SIZE:-32}"
export T2I_REQUEST_TIMEOUT_SECONDS="${T2I_REQUEST_TIMEOUT_SECONDS:-600}"
export T2I_LOG_DIR="${T2I_LOG_DIR:-${ROOT_DIR}/logs/t2i_service}"
export T2I_LOG_LEVEL="${T2I_LOG_LEVEL:-INFO}"

echo "Starting ViMax local T2I service"
echo "  model_path: ${T2I_MODEL_PATH}"
echo "  cuda_devices: ${T2I_CUDA_DEVICES}"
echo "  listen: ${T2I_HOST}:${T2I_PORT}"
echo "  log_dir: ${T2I_LOG_DIR}"

exec python -m t2i_service.server
