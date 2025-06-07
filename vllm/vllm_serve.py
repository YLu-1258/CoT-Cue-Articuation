import argparse
import subprocess
import os
from typing import List
import signal
import time
import yaml
import torch

def get_available_gpus():
    try:
        nvidia_smi = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'])
        memory_used = [int(x) for x in nvidia_smi.decode('utf-8').strip().split('\n')]
        return [i for i, mem in enumerate(memory_used) if mem == min(memory_used)]
    except:
        raise ValueError("No GPU Available")

def start_vllm_servers(models: List[str], ports: List[int]):
    """
    Start multiple vLLM servers in parallel
    
    Args:
        models (List[str]): List of model names to serve
        ports (List[int]): List of ports to serve models on
        gpu_ids (List[int], optional): List of GPU IDs to use. If None, will use all available GPUs.
    """
    gpu_ids = get_available_gpus()[:len(models)]
    print("Using GPUs:", gpu_ids)
    
    # Start each server
    processes = []
    for model, port, gpu_id in zip(models, ports, gpu_ids):
        cmd = [
            f"CUDA_VISIBLE_DEVICES={gpu_id}",
            "vllm",
            "serve",
            model,
            "--port",
            str(port)
        ]
        
        # Join command components with spaces
        cmd_str = " ".join(cmd)
        print(f"Starting server: {cmd_str}")
        
        # Start process
        process = subprocess.Popen(
            cmd_str,
            shell=True,
            preexec_fn=os.setsid  # Create new process group
        )
        processes.append(process)
        
        # Wait a bit between starting servers
        time.sleep(5)
    
    return processes

def cleanup_processes(processes):
    """Clean up all running processes"""
    for process in processes:
        try:
            # Kill the entire process group
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass

def main():
    parser = argparse.ArgumentParser(description="Start multiple vLLM servers")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml file containing model and port configurations")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    if args.debug:
        print("Configuration loaded:")
        print(f"Models: {config['models']}")
        print(f"Ports: {config['ports']}")
    
    if len(config['models']) != len(config['ports']):
        raise ValueError("Number of models must match number of ports")
    
    try:
        processes = start_vllm_servers(config['models'], config['ports'])
        
        # Keep script running until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        cleanup_processes(processes)
        print("Servers shut down.")

if __name__ == "__main__":
    main()
