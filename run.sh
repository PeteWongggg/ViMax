#!/usr/bin/env bash
# ViMax local image-edit service launcher.
#
# Usage:
#   ./run.sh start    # 后台启动（默认）
#   ./run.sh stop     # 停止
#   ./run.sh restart  # 重启
#   ./run.sh status   # 查看状态
#
# 环境变量可在下方「配置区」修改，或在启动前 export 覆盖。

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# ===== 配置区（按需修改）=====
export T2I_MODEL_PATH="${T2I_MODEL_PATH:-/nas/models/i2i/qwen/Qwen/Qwen-Image-Edit-2511}"
export T2I_CUDA_DEVICES="1"
export T2I_DEVICE_MAP="cuda"
export T2I_PORT="${T2I_PORT:-8911}"
export T2I_HOST="${T2I_HOST:-0.0.0.0}"
export T2I_JOB_RETENTION_COUNT="${T2I_JOB_RETENTION_COUNT:-64}"
export T2I_LOG_DIR="${T2I_LOG_DIR:-${ROOT_DIR}/logs/t2i_service}"
# =============================

LOG_DIR="${T2I_LOG_DIR}"
STDOUT_LOG="${LOG_DIR}/stdout.log"
PID_FILE="${LOG_DIR}/service.pid"
START_SCRIPT="${ROOT_DIR}/scripts/start_t2i_service.sh"

mkdir -p "${LOG_DIR}"

is_running() {
  if [ ! -f "${PID_FILE}" ]; then
    return 1
  fi
  local pid
  pid="$(cat "${PID_FILE}")"
  if [ -z "${pid}" ]; then
    return 1
  fi
  kill -0 "${pid}" 2>/dev/null
}

start_service() {
  if is_running; then
    echo "Service already running | pid=$(cat "${PID_FILE}") | port=${T2I_PORT}"
    exit 0
  fi

  echo "Starting image-edit service..."
  echo "  model_path: ${T2I_MODEL_PATH}"
  echo "  cuda_devices: ${T2I_CUDA_DEVICES}"
  echo "  port: ${T2I_PORT}"
  echo "  stdout_log: ${STDOUT_LOG}"
  echo "  pid_file: ${PID_FILE}"

  nohup "${START_SCRIPT}" >> "${STDOUT_LOG}" 2>&1 &
  echo $! > "${PID_FILE}"
  sleep 2

  if is_running; then
    echo "Started | pid=$(cat "${PID_FILE}")"
    echo "Health: curl -s http://127.0.0.1:${T2I_PORT}/health"
  else
    echo "Failed to start. Check ${STDOUT_LOG}" >&2
    rm -f "${PID_FILE}"
    exit 1
  fi
}

stop_service() {
  if ! is_running; then
    echo "Service is not running"
    rm -f "${PID_FILE}"
    exit 0
  fi

  local pid
  pid="$(cat "${PID_FILE}")"
  echo "Stopping service | pid=${pid}"
  kill "${pid}" 2>/dev/null || true

  for _ in $(seq 1 30); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      rm -f "${PID_FILE}"
      echo "Stopped"
      return 0
    fi
    sleep 1
  done

  echo "Graceful stop timed out, sending SIGKILL"
  kill -9 "${pid}" 2>/dev/null || true
  rm -f "${PID_FILE}"
  echo "Stopped (forced)"
}

status_service() {
  if is_running; then
    echo "running | pid=$(cat "${PID_FILE}") | port=${T2I_PORT}"
    curl -fsS "http://127.0.0.1:${T2I_PORT}/health" 2>/dev/null | python3 -m json.tool || true
  else
    echo "not running"
    exit 1
  fi
}

restart_service() {
  stop_service || true
  start_service
}

case "${1:-start}" in
  start) start_service ;;
  stop) stop_service ;;
  restart) restart_service ;;
  status) status_service ;;
  *)
    echo "Usage: $0 {start|stop|restart|status}" >&2
    exit 1
    ;;
esac
