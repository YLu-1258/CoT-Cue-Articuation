#!/usr/bin/env python3
"""
Add new vLLM instances using the existing vllm_serve.py script
"""
import os
import subprocess
import sys
import shutil

def add_instance_with_serve(config_file):
    """Add instances using vllm_serve.py by temporarily managing PID file"""
    
    original_pids = []
    pid_file = "vllm_pids.txt"
    backup_file = "vllm_pids_backup.txt"
    
    # Step 1: Backup existing PID file if it exists
    if os.path.exists(pid_file):
        print(f"📁 Backing up existing {pid_file}")
        shutil.copy(pid_file, backup_file)
        with open(pid_file, 'r') as f:
            original_pids = [line.strip() for line in f.readlines() if line.strip()]
        os.remove(pid_file)
    
    try:
        # Step 2: Start new instances using vllm_serve.py
        print(f"🚀 Starting new instances with config: {config_file}")
        cmd = ["python", "vllm_serve.py", "--config", config_file]
        result = subprocess.run(cmd, check=True)
        
        # Step 3: Read new PIDs
        new_pids = []
        if os.path.exists(pid_file):
            with open(pid_file, 'r') as f:
                new_pids = [line.strip() for line in f.readlines() if line.strip()]
        
        # Step 4: Merge old and new PIDs
        all_pids = original_pids + new_pids
        with open(pid_file, 'w') as f:
            for pid in all_pids:
                f.write(f"{pid}\n")
        
        print(f"✅ Successfully added {len(new_pids)} new instance(s)")
        print(f"📝 Updated {pid_file} with {len(all_pids)} total PIDs")
        
        # Clean up backup
        if os.path.exists(backup_file):
            os.remove(backup_file)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start instances: {e}")
        
        # Restore original PID file if something went wrong
        if os.path.exists(backup_file):
            print("🔄 Restoring original PID file")
            shutil.move(backup_file, pid_file)
        
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        
        # Restore original PID file if something went wrong
        if os.path.exists(backup_file):
            print("🔄 Restoring original PID file")
            shutil.move(backup_file, pid_file)
        
        sys.exit(1)

def main():
    if len(sys.argv) != 2:
        print("Usage: python add_with_serve.py <config_file>")
        print("Example: python add_with_serve.py vllm_serve_add_one.yaml")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        sys.exit(1)
    
    add_instance_with_serve(config_file)

if __name__ == "__main__":
    main() 