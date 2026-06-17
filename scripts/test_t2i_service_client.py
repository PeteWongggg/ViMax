"""Quick client for local T2I service smoke tests."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for local T2I service")
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--prompt", default="一个成年人")
    parser.add_argument("--output", default="service_test.png")
    parser.add_argument("--size", default="1664x928")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    payload = {
        "prompt": args.prompt,
        "size": args.size,
        "seed": args.seed,
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
