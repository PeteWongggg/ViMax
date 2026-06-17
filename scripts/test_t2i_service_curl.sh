#!/usr/bin/env bash
set -euo pipefail

# ===== 只改这里 =====
PROMPT="基于当前人物肖像，生成一张健身照."
REF_IMAGES=(
  "input1.png"
)
# ====================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PAYLOAD="$(python3 - "$PROMPT" "${REF_IMAGES[@]}" <<'PY'
import base64, json, sys
from pathlib import Path

prompt = sys.argv[1]
payload = {
    "prompt": prompt,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "true_cfg_scale": 4.0,
    "guidance_scale": 1.0,
    "seed": 42,
    "reference_images_b64": [
        base64.b64encode(Path(path).read_bytes()).decode("ascii")
        for path in sys.argv[2:]
        if path
    ],
}
print(json.dumps(payload, ensure_ascii=False))
PY
)"

curl -sS --max-time 600 \
  -H "Content-Type: application/json" \
  -X POST "http://127.0.0.1:8100/v1/images/generations" \
  -d "$PAYLOAD" \
| python3 -c "
import base64, json, sys
data = json.load(sys.stdin)
open('${SCRIPT_DIR}/curl_test_output.png', 'wb').write(base64.b64decode(data['image_b64']))
print('saved: ${SCRIPT_DIR}/curl_test_output.png')
"
