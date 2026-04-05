#!/usr/bin/env python3
"""
test_comfyui.py — Submit a test workflow to the dockerized ComfyUI and download the result.

Usage:
    python docker/test_comfyui.py
    python docker/test_comfyui.py --host localhost --port 8190
    python docker/test_comfyui.py --health-only        # just check the server is up
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import requests

WORKFLOW_PATH = Path(__file__).parent / "workflows" / "docker_test_simple.json"
OUTPUT_DIR = Path(__file__).parent / "test_output"

POSITIVE_PROMPT = "a sleek sci-fi spaceship, concept art, clean white background, hard surface modeling, detailed hull"
NEGATIVE_PROMPT = "blurry, low quality, text, watermark, signature"


def wait_for_server(base_url: str, timeout: int = 120) -> bool:
    """Poll until ComfyUI is ready or timeout expires."""
    print(f"Waiting for ComfyUI at {base_url} ...", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/system_stats", timeout=3)
            if r.status_code == 200:
                stats = r.json()
                print(f"  Server ready. System: {stats.get('system', {})}")
                return True
        except requests.exceptions.ConnectionError:
            pass
        print("  Not ready yet, retrying in 3s ...", flush=True)
        time.sleep(3)
    print(f"ERROR: Server did not become ready within {timeout}s.")
    return False


def load_workflow(positive: str, negative: str) -> dict:
    """Load the test workflow and inject prompts."""
    raw = WORKFLOW_PATH.read_text(encoding="utf-8")
    raw = raw.replace("TEST_POSITIVE", positive)
    raw = raw.replace("TEST_NEGATIVE", negative)
    return json.loads(raw)


def queue_prompt(base_url: str, workflow: dict) -> str:
    """POST the workflow to /prompt and return the prompt_id."""
    client_id = str(uuid.uuid4())
    payload = {"prompt": workflow, "client_id": client_id}
    r = requests.post(f"{base_url}/prompt", json=payload, timeout=30)
    if not r.ok:
        print(f"  ERROR {r.status_code}: {r.text}")
        r.raise_for_status()
    prompt_id = r.json()["prompt_id"]
    print(f"  Queued prompt_id={prompt_id}")
    return prompt_id


def wait_for_completion(base_url: str, prompt_id: str, timeout: int = 300) -> dict:
    """Poll /history until the prompt finishes, then return its output data."""
    print(f"  Waiting for generation (timeout={timeout}s) ...", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{base_url}/history/{prompt_id}", timeout=10)
        r.raise_for_status()
        history = r.json()
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("completed"):
                print("  Generation complete.")
                return entry["outputs"]
            if status.get("status_str") == "error":
                msgs = status.get("messages", [])
                raise RuntimeError(f"ComfyUI reported error: {msgs}")
        time.sleep(2)
    raise TimeoutError(f"Generation did not complete within {timeout}s.")


def download_outputs(base_url: str, outputs: dict, out_dir: Path) -> list[Path]:
    """Download all SaveImage outputs to out_dir. Returns list of saved paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for node_id, node_output in outputs.items():
        for image_info in node_output.get("images", []):
            filename = image_info["filename"]
            subfolder = image_info.get("subfolder", "")
            img_type = image_info.get("type", "output")
            params = {"filename": filename, "subfolder": subfolder, "type": img_type}
            r = requests.get(f"{base_url}/view", params=params, timeout=30)
            r.raise_for_status()
            dest = out_dir / filename
            dest.write_bytes(r.content)
            print(f"  Saved: {dest}")
            saved.append(dest)
    return saved


def main():
    parser = argparse.ArgumentParser(description="Test dockerized ComfyUI")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8190)
    parser.add_argument("--health-only", action="store_true",
                        help="Only check server health, don't submit a workflow")
    parser.add_argument("--positive", default=POSITIVE_PROMPT)
    parser.add_argument("--negative", default=NEGATIVE_PROMPT)
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    if not wait_for_server(base_url):
        sys.exit(1)

    if args.health_only:
        print("Health check passed.")
        return

    print(f"\nLoading workflow: {WORKFLOW_PATH}")
    workflow = load_workflow(args.positive, args.negative)

    print("Submitting workflow ...")
    prompt_id = queue_prompt(base_url, workflow)

    outputs = wait_for_completion(base_url, prompt_id)

    print(f"\nDownloading output images -> {OUTPUT_DIR}/")
    saved = download_outputs(base_url, outputs, OUTPUT_DIR)

    if saved:
        print(f"\nDone. {len(saved)} image(s) saved to {OUTPUT_DIR}/")
    else:
        print("\nWARNING: No images found in output. Check ComfyUI logs.")


if __name__ == "__main__":
    main()
