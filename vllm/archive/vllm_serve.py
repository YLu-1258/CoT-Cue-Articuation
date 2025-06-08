import argparse
import subprocess
import os
from typing import List
import signal
import time
import yaml
import torch

PID_FILE = "vllm_pids.txt"

def get_available_gpus():
    try:
        nvidia_smi = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'])
        memory_used = [int(x) for x in nvidia_smi.decode('utf-8').strip().split('\n')]
        return [i for i, mem in enumerate(memory_used) if mem == min(memory_used)]
    except:
        raise ValueError("No GPU Available")

def start_vllm_servers(models: List[str], ports: List[int], background: bool = True):
    """
    Start multiple vLLM servers in parallel with nohup
    
    Args:
        models (List[str]): List of model names to serve
        ports (List[int]): List of ports to serve models on
        background (bool): Whether to run in background with nohup
    """
    gpu_ids = get_available_gpus()[:len(models)]
    print("Using GPUs:", gpu_ids)
    
    # Store PIDs for later cleanup
    pids = []
    
    # Start each server
    processes = []
    for model, port, gpu_id in zip(models, ports, gpu_ids):
        # Prepare environment variables
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Include HF_TOKEN if available
        if 'HF_TOKEN' in os.environ:
            env['HF_TOKEN'] = os.environ['HF_TOKEN']
            print(f"Using HF_TOKEN for model: {model}")
        
        if background:
            # Use nohup for background execution
            log_file = f"vllm_server_{port}.log"
            cmd = [
                "nohup",
                "vllm",
                "serve",
                model,
                "--port",
                str(port)
            ]
            
            print(f"Starting server in background: CUDA_VISIBLE_DEVICES={gpu_id} vllm serve {model} --port {port}")
            print(f"Logs saved to: {log_file}")
            
            # Start process with nohup and redirect output
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid  # Create new process group
                )
        else:
            # Regular execution (for debugging)
            cmd = [
                "vllm",
                "serve",
                model,
                "--port",
                str(port)
            ]
            
            print(f"Starting server: CUDA_VISIBLE_DEVICES={gpu_id} vllm serve {model} --port {port}")
            
            process = subprocess.Popen(
                cmd,
                env=env,
                preexec_fn=os.setsid
            )
        
        processes.append(process)
        pids.append(process.pid)
        
        # Wait a bit between starting servers
        time.sleep(5)
    
    # Save PIDs to file for later cleanup
    with open(PID_FILE, 'w') as f:
        for pid in pids:
            f.write(f"{pid}\n")
    
    print(f"Process IDs saved to {PID_FILE}")
    return processes

def kill_all_servers():
    """Kill only the vLLM servers started by this script using saved PIDs"""
    print("Stopping vLLM servers started by this script...")
    
    if not os.path.exists(PID_FILE):
        print(f"No {PID_FILE} found - no servers to stop")
        return
    
    print(f"Reading PIDs from {PID_FILE}")
    with open(PID_FILE, 'r') as f:
        pids = [line.strip() for line in f.readlines() if line.strip()]
    
    if not pids:
        print("No PIDs found in file")
        os.remove(PID_FILE)
        return
    
    killed_count = 0
    for pid_str in pids:
        try:
            pid = int(pid_str)
            
            # Check if process exists and belongs to current user
            try:
                # Check if process exists
                os.kill(pid, 0)  # Send signal 0 to check if process exists
                print(f"Killing process {pid}")
                
                # Try graceful shutdown first
                os.kill(pid, signal.SIGTERM)
                time.sleep(2)
                
                # Check if still running
                try:
                    os.kill(pid, 0)  # Check if still exists
                    print(f"Force killing process {pid}")
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # Already dead from SIGTERM
                
                killed_count += 1
                
            except ProcessLookupError:
                print(f"Process {pid} not found (already stopped)")
            except PermissionError:
                print(f"Permission denied for process {pid} (not your process)")
                
        except ValueError:
            print(f"Invalid PID: {pid_str}")
    
    # Remove PID file
    os.remove(PID_FILE)
    print(f"Removed {PID_FILE}")
    
    if killed_count > 0:
        print(f"✓ Successfully stopped {killed_count} vLLM server(s)")
    else:
        print("No processes were running or needed to be stopped")

def cleanup_processes(processes):
    """Clean up all running processes (for non-background mode)"""
    for process in processes:
        try:
            # Kill the entire process group
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass

def main():
    parser = argparse.ArgumentParser(description="Start multiple vLLM servers")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml file containing model and port configurations")
    parser.add_argument("--debug", action="store_true", help="Debug mode (runs in foreground)")
    parser.add_argument("--kill", action="store_true", help="Kill all running vLLM servers and exit")
    args = parser.parse_args()
    
    # Kill mode
    if args.kill:
        kill_all_servers()
        return
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    if args.debug:
        print("Configuration loaded:")
        print(f"Models: {config['models']}")
        print(f"Ports: {config['ports']}")
    
    if len(config['models']) != len(config['ports']):
        raise ValueError("Number of models must match number of ports")
    
    # Check if already running
    if os.path.exists(PID_FILE):
        print(f"Warning: {PID_FILE} already exists. Run with --kill to stop existing servers first.")
        return
    
    try:
        background_mode = not args.debug
        processes = start_vllm_servers(config['models'], config['ports'], background=background_mode)
        
        if background_mode:
            print("\n✓ All servers started in background")
            print(f"✓ Process IDs saved to {PID_FILE}")
            print("✓ Logs saved to vllm_server_[port].log files")
            print("\nTo stop all servers later, run:")
            print(f"  python {__file__} --config {args.config} --kill")
            print("Or manually (only kills your processes):")
            print(f"  kill $(cat {PID_FILE})")
        else:
            print("Running in foreground mode (debug). Press Ctrl+C to stop...")
            # Keep script running until interrupted
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        if not background_mode:
            print("\nShutting down servers...")
            cleanup_processes(processes)
            print("Servers shut down.")

if __name__ == "__main__":
    main()
