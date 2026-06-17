#!/usr/bin/env bash
# 必须用 bash 运行（不要用 sh）。若误用 sh，会自动切到 bash。
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

# ===== 只改这里 =====
PROMPT="基于当前人物肖像，生成一张健身照."
REF_IMAGES=(
  "input1.png"
)
# ====================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESPONSE_FILE="$(mktemp)"
trap 'rm -f "$RESPONSE_FILE"' EXIT

PAYLOAD="$(python3 - "$SCRIPT_DIR" "$PROMPT" "${REF_IMAGES[@]}" <<'PY'
import base64, json, sys
from pathlib import Path

script_dir = Path(sys.argv[1])
prompt = sys.argv[2]
ref_paths = sys.argv[3:]
refs_b64 = []
for raw in ref_paths:
    if not raw:
        continue
    path = Path(raw)
    if not path.is_absolute():
        path = script_dir / path
    if not path.exists():
        raise SystemExit(f"reference image not found: {path}")
    refs_b64.append(base64.b64encode(path.read_bytes()).decode("ascii"))

payload = {
    "prompt": prompt,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "true_cfg_scale": 4.0,
    "guidance_scale": 1.0,
    "seed": 42,
    "reference_images_b64": refs_b64,
}
print(json.dumps(payload, ensure_ascii=False))
PY
)"

HTTP_CODE="$(
  curl -sS --max-time 600 \
    -H "Content-Type: application/json" \
    -X POST "http://127.0.0.1:8100/v1/images/generations" \
    -d "$PAYLOAD" \
    -o "$RESPONSE_FILE" \
    -w "%{http_code}"
)"

python3 - "$HTTP_CODE" "$RESPONSE_FILE" "$SCRIPT_DIR/curl_test_output.png" <<'PY'
import base64
import json
import sys
from pathlib import Path

http_code, response_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
body = Path(response_path).read_text(encoding="utf-8")
try:
    data = json.loads(body)
except json.JSONDecodeError:
    print(f"HTTP {http_code}")
    print(body)
    raise SystemExit(1)

if http_code != "200" or "image_b64" not in data:
    print(f"HTTP {http_code}")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    raise SystemExit(1)

Path(output_path).write_bytes(base64.b64decode(data["image_b64"]))
summary = {k: v for k, v in data.items() if k != "image_b64"}
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"saved: {output_path}")
PY
