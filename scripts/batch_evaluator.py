#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
from pathlib import Path

import openai

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enums.cue import Cue


def main():
    p = argparse.ArgumentParser(
        description="Send prebuilt sub‐batches to 4o, one at a time with confirmation"
    )
    p.add_argument("--cue", type=str, choices=[cue.value for cue in Cue], 
                       help="Evaluate responses for specific cue only")
    args = p.parse_args()

    client = openai.OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))
    cue = Cue(args.cue)
    batch_dir = Path("data/batches") / cue.value

    if not batch_dir.exists():
        print(f"❌ No directory for cue '{cue.value}': {batch_dir}")
        sys.exit(1)

    # find all sub‐batches
    batch_files = sorted(batch_dir.glob("batch_*.jsonl"))
    if not batch_files:
        print(f"❌ No batch_*.jsonl files under {batch_dir}")
        sys.exit(1)

    print(f"Found {len(batch_files)} sub-batches for cue '{cue.value}':")
    for bf in batch_files:
        print("  ", bf.name)

    print()
    for bf in batch_files:
        resp = input(f"\nSend batch '{bf.name}'? [y/N/q]: ").strip().lower()
        if resp in ("q", "quit", "exit"):
            print("Aborting further batches.")
            break
        if resp not in ("y", "yes"):
            print(f"⏭ Skipping {bf.name}")
            continue

        print(f"Uploading {bf.name} …")
        upload = client.files.create(
            file= open(bf, "rb"),
            purpose="batch"
        )
        file_id = upload.id
        print(" ↳ file_id =", file_id)

        metadata = {"description": f"Eval {cue.value} – {bf.name}"}

        print("Creating batch …")
        batch = client.batches.create(
            input_file_id     = file_id,
            endpoint          = "/v1/chat/completions",
            completion_window = "24h",
            metadata          = metadata
        )
        print(f"✅ Created batch: {batch.id}")

        # optional pause to avoid hammering
        time.sleep(1.0)

    print("\nAll done.")

if __name__ == "__main__":
    main()