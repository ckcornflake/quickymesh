"""
Smoke test for Gemini image generation.

Run from the repo root:
    python scripts/smoke_test_gemini.py

Optional flags:
    --prompt "a red sports car"     Override the test prompt
    --model  "gemini-2.5-flash-..."  Override the model (default: from defaults.yaml)
    --out    "test_output.png"       Override the output path
    --count  2                       Generate N images (default 1)

Exit code 0 = success, 1 = failure.
"""

import argparse
import sys
import time
from pathlib import Path

# Make sure repo root is on sys.path regardless of cwd
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from src.config import config
from src.workers.concept_art import GeminiConceptArtWorker
from src.image_utils import pad_to_square
from PIL import Image
import io


def parse_args():
    p = argparse.ArgumentParser(description="Smoke-test the Gemini image generation API.")
    p.add_argument("--prompt", default="a small ceramic teapot, plain white background",
                   help="Prompt to send to Gemini")
    p.add_argument("--model", default=None,
                   help="Override model name (default: from defaults.yaml / GEMINI_MODEL env var)")
    p.add_argument("--out", default="smoke_test_output", help="Output file stem (no extension)")
    p.add_argument("--count", type=int, default=1, help="Number of images to generate")
    return p.parse_args()


def main():
    args = parse_args()
    model = args.model or config.gemini_model

    print(f"Model  : {model}")
    print(f"Prompt : {args.prompt}")
    print(f"Count  : {args.count}")
    print()

    try:
        api_key = config.gemini_api_key
    except EnvironmentError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    worker = GeminiConceptArtWorker(api_key=api_key, model=model)

    for i in range(args.count):
        label = f"[{i+1}/{args.count}]"
        print(f"{label} Sending request to Gemini...", flush=True)
        t0 = time.perf_counter()

        try:
            raw_bytes = worker.generate_image(args.prompt)
        except Exception as e:
            print(f"{label} FAILED: {e}")
            sys.exit(1)

        elapsed = time.perf_counter() - t0
        print(f"{label} Response received in {elapsed:.1f}s — {len(raw_bytes)} bytes")

        # Decode and inspect
        try:
            img = Image.open(io.BytesIO(raw_bytes))
            print(f"{label} Image decoded: {img.size[0]}x{img.size[1]} {img.mode}")
        except Exception as e:
            print(f"{label} Could not decode image bytes: {e}")
            sys.exit(1)

        # Pad to 1024x1024 and save
        padded = pad_to_square(img, size=1024)
        stem = args.out if args.count == 1 else f"{args.out}_{i+1}"
        out_path = _REPO / f"{stem}.png"
        padded.save(str(out_path))
        print(f"{label} Saved (padded to 1024x1024): {out_path}")
        print()

    print("Smoke test PASSED.")


if __name__ == "__main__":
    main()
