#!/usr/bin/env python3
"""GSM8K Dataset Preparation using Official VERL Method"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Prepare GSM8K dataset using VERL")
    parser.add_argument("--local_dir", default="/data/kevinchu/gsm8k", help="Output directory")
    args = parser.parse_args()
    
    # Create directory
    os.makedirs(args.local_dir, exist_ok=True)
    
    # Find VERL script
    verl_script = "verl/examples/data_preprocess/gsm8k.py"
    if not os.path.exists(verl_script):
        print(f"❌ VERL script not found: {verl_script}")
        return 1
    
    # Run VERL preprocessing
    cmd = [sys.executable, verl_script, "--local_dir", args.local_dir]
    print(f"🔄 Running: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Dataset prepared!")
        print(f"📁 Files in: {args.local_dir}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 