"""Quick client for local image-edit service smoke tests."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for local image-edit service")
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument(
        "--prompt",
        default=(
            "The magician bear is on the left, the alchemist bear is on the right, "
            "facing each other in the central park square."
        ),
    )
    parser.add_argument("--output", default="service_test.png")
    parser.add_argument("--reference-image", action="append", default=[], dest="reference_images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    args = parser.parse_args()

    payload = {
        "prompt": args.prompt,
        "seed": args.seed,
        "num_inference_steps": args.steps,
        "true_cfg_scale": args.true_cfg_scale,
        "guidance_scale": args.guidance_scale,
        "negative_prompt": " ",
        "reference_images_b64": [
            base64.b64encode(Path(path).read_bytes()).decode("ascii")
            for path in args.reference_images
        ],
    }
    url = f"{args.base_url.rstrip('/')}/v1/images/generations"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8", errors="replace"), file=sys.stderr)
        return 1

    image_b64 = body["image_b64"]
    with open(args.output, "wb") as handle:
        handle.write(base64.b64decode(image_b64))

    print(json.dumps({k: body[k] for k in body if k != "image_b64"}, ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
