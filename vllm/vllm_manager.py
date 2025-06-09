#!/usr/bin/env python3
"""
vLLM Manager - Easy management of multiple vLLM instances
"""

import argparse
import subprocess
import os
import json
import time
import signal
from typing import List, Dict
from pathlib import Path

class VLLMManager:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.logs_dir = self.base_dir / "logs"
        self.configs_dir = self.base_dir / "configs"
        self.pid_file = self.base_dir / "vllm_pids.txt"
        
        # Ensure directories exist
        self.logs_dir.mkdir(exist_ok=True)
        self.configs_dir.mkdir(exist_ok=True)
    
    def get_available_gpus(self) -> List[int]:
        """Get available GPUs sorted by memory usage."""
        try:
            result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'])
            memory_used = [int(x) for x in result.decode('utf-8').strip().split('\n')]
            # Return GPU IDs sorted by memory usage (least used first)
            gpu_usage = [(i, mem) for i, mem in enumerate(memory_used)]
            gpu_usage.sort(key=lambda x: x[1])
            return [gpu[0] for gpu in gpu_usage]
        except:
            print("❌ Failed to get GPU information")
            return []
    
    def check_port_available(self, port: int) -> bool:
        """Check if a port is available."""
        try:
            result = subprocess.run(['ss', '-tuln'], capture_output=True, text=True)
            return f":{port}" not in result.stdout
        except:
            # Fallback method
            try:
                result = subprocess.run(['netstat', '-tuln'], capture_output=True, text=True)
                return f":{port}" not in result.stdout
            except:
                print(f"⚠️  Cannot check port {port} availability")
                return True
    
    def find_available_ports(self, start_port: int = 6000, count: int = 1) -> List[int]:
        """Find available ports starting from start_port."""
        available = []
        port = start_port
        while len(available) < count:
            if self.check_port_available(port):
                available.append(port)
            port += 1
            if port > start_port + 100:  # Safety limit
                break
        return available
    
    def start_instance(self, model: str, port: int, gpu_id: int) -> int:
        """Start a single vLLM instance."""
        log_file = self.logs_dir / f"vllm_server_{port}.log"
        
        # Prepare environment
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Command to run for Alex
        # cmd = [
        #     "bash", "-c",
        #     f"source /data/alexl/anaconda3/etc/profile.d/conda.sh && conda activate vllm && vllm serve {model} --port {port}"
        # ]

        cmd = [
            "bash", "-c",
            f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate vllm && vllm serve {model} --port {port}"
        ]
        # Using Kevin's vllm environment
        
        print(f"🚀 Starting {model} on GPU {gpu_id}, port {port}")
        print(f"📝 Logs: {log_file}")
        
        # Start process
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                ["nohup"] + cmd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid
            )
        
        # Save PID
        with open(self.pid_file, 'a') as f:
            f.write(f"{process.pid}\n")
        
        return process.pid
    
    def start_single_model(self, model: str, port: int = None, gpu_id: int = None):
        """Start a single instance of a specific model."""
        print(f"🎯 Starting single instance of {model}")
        
        # Get available resources
        available_gpus = self.get_available_gpus()
        available_ports = self.find_available_ports(6000, 1)
        
        # Use specified or auto-select GPU
        if gpu_id is not None:
            if gpu_id not in available_gpus:
                print(f"⚠️  GPU {gpu_id} not available. Available GPUs: {available_gpus}")
                return None
            selected_gpu = gpu_id
        else:
            if not available_gpus:
                print("❌ No GPUs available")
                return None
            selected_gpu = available_gpus[0]  # Use least loaded GPU
        
        # Use specified or auto-select port
        if port is not None:
            if not self.check_port_available(port):
                print(f"⚠️  Port {port} is not available")
                return None
            selected_port = port
        else:
            if not available_ports:
                print("❌ No ports available")
                return None
            selected_port = available_ports[0]
        
        print(f"📊 Using GPU {selected_gpu}, Port {selected_port}")
        
        # Start the instance
        pid = self.start_instance(model, selected_port, selected_gpu)
        
        print(f"✅ Started {model} on GPU {selected_gpu}, port {selected_port}")
        print(f"📋 PID: {pid}")
        print(f"💡 Use 'python vllm_manager.py status' to check startup progress")
        
        return pid
    
    def start_multiple(self, model: str, count: int, start_port: int = 6000):
        """Start multiple instances of the same model."""
        print(f"🎯 Starting {count} instances of {model}")
        
        # Get available resources
        available_gpus = self.get_available_gpus()
        available_ports = self.find_available_ports(start_port, count)
        
        if len(available_gpus) < count:
            print(f"⚠️  Only {len(available_gpus)} GPUs available, requested {count}")
            count = len(available_gpus)
        
        if len(available_ports) < count:
            print(f"⚠️  Only {len(available_ports)} ports available, requested {count}")
            count = len(available_ports)
        
        print(f"📊 Resources:")
        print(f"   GPUs: {available_gpus[:count]}")
        print(f"   Ports: {available_ports[:count]}")
        
        # Start instances
        pids = []
        for i in range(count):
            gpu_id = available_gpus[i]
            port = available_ports[i]
            pid = self.start_instance(model, port, gpu_id)
            pids.append(pid)
            time.sleep(2)  # Small delay between starts
        
        print(f"\n✅ Started {len(pids)} instances")
        print(f"📋 PIDs: {pids}")
        print(f"💡 Use 'python vllm_manager.py status' to check startup progress")
        
        return pids
    
    def stop_single(self, identifier: str):
        """Stop a single vLLM instance by PID or port."""
        if not self.pid_file.exists():
            print("📭 No managed instances found")
            return
        
        with open(self.pid_file, 'r') as f:
            pids = [line.strip() for line in f.readlines() if line.strip()]
        
        target_pid = None
        
        # Check if identifier is a PID
        if identifier.isdigit():
            target_pid = int(identifier)
            if str(target_pid) not in pids:
                print(f"⚠️  PID {target_pid} not found in managed instances")
                return
        else:
            # Check if identifier is a port (find PID by port)
            port = identifier.replace(':', '').replace('port', '')
            if port.isdigit():
                try:
                    result = subprocess.run(['ss', '-tlnp'], capture_output=True, text=True)
                    for line in result.stdout.split('\n'):
                        if f":{port}" in line and 'LISTEN' in line:
                            # Extract PID from ss output
                            parts = line.split()
                            for part in parts:
                                if 'pid=' in part:
                                    pid_part = part.split('pid=')[1].split(',')[0]
                                    if pid_part.isdigit():
                                        target_pid = int(pid_part)
                                        break
                            break
                    
                    if target_pid is None:
                        print(f"⚠️  No process found listening on port {port}")
                        return
                    
                    if str(target_pid) not in pids:
                        print(f"⚠️  Process on port {port} (PID {target_pid}) is not managed by this tool")
                        return
                        
                except Exception as e:
                    print(f"❌ Error finding process on port {port}: {e}")
                    return
            else:
                print(f"❌ Invalid identifier: {identifier}. Use PID number or port number.")
                return
        
        # Stop the process
        try:
            print(f"🛑 Stopping process {target_pid}...")
            os.kill(target_pid, signal.SIGTERM)
            time.sleep(1)
            try:
                os.kill(target_pid, 0)  # Check if still exists
                os.kill(target_pid, signal.SIGKILL)  # Force kill if needed
            except ProcessLookupError:
                pass
            print(f"✅ Stopped process {target_pid}")
            
            # Remove from PID file
            remaining_pids = [p for p in pids if p != str(target_pid)]
            if remaining_pids:
                with open(self.pid_file, 'w') as f:
                    for pid in remaining_pids:
                        f.write(f"{pid}\n")
            else:
                self.pid_file.unlink()
                print("🧹 No more managed instances, cleaned up PID file")
                
        except (ProcessLookupError):
            print(f"⚠️  Process {target_pid} not found")
        except PermissionError:
            print(f"❌ Permission denied for process {target_pid}")

    def stop_all(self):
        """Stop all managed vLLM instances."""
        if not self.pid_file.exists():
            print("📭 No instances to stop")
            return
        
        print("🛑 Stopping all vLLM instances...")
        
        with open(self.pid_file, 'r') as f:
            pids = [line.strip() for line in f.readlines() if line.strip()]
        
        stopped_count = 0
        for pid_str in pids:
            try:
                pid = int(pid_str)
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)
                try:
                    os.kill(pid, 0)  # Check if still exists
                    os.kill(pid, signal.SIGKILL)  # Force kill if needed
                except ProcessLookupError:
                    pass
                stopped_count += 1
                print(f"✅ Stopped process {pid}")
            except (ValueError, ProcessLookupError):
                print(f"⚠️  Process {pid_str} not found")
            except PermissionError:
                print(f"❌ Permission denied for process {pid_str}")
        
        # Remove PID file
        self.pid_file.unlink()
        print(f"🧹 Cleaned up PID file")
        print(f"✅ Stopped {stopped_count} instances")
    
    def status(self):
        """Show status of all instances with detailed information."""
        if not self.pid_file.exists():
            print("📭 No managed instances found")
            return
        
        with open(self.pid_file, 'r') as f:
            pids = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"📊 vLLM Instance Status ({len(pids)} managed)")
        print("=" * 80)
        
        running_count = 0
        for pid_str in pids:
            try:
                pid = int(pid_str)
                os.kill(pid, 0)  # Check if process exists
                
                # Get detailed process info
                try:
                    result = subprocess.run(['ps', '-p', str(pid), '-o', 'pid,cmd'], 
                                          capture_output=True, text=True)
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        cmd_line = lines[1]
                        # Extract model and port from command line
                        model = "unknown"
                        port = "unknown"
                        if "serve" in cmd_line:
                            parts = cmd_line.split()
                            for i, part in enumerate(parts):
                                if part == "serve" and i + 1 < len(parts):
                                    model = parts[i + 1].split('/')[-1]  # Get model name
                                elif part == "--port" and i + 1 < len(parts):
                                    port = parts[i + 1]
                        
                        print(f"✅ PID {pid:6} | Port {port:4} | Model: {model}")
                    else:
                        print(f"✅ PID {pid:6} | Details unavailable")
                except:
                    print(f"✅ PID {pid:6} | Running (details unavailable)")
                
                running_count += 1
            except ProcessLookupError:
                print(f"❌ PID {pid:6} | Not running")
            except ValueError:
                print(f"⚠️  Invalid PID: {pid_str}")
        
        print("=" * 80)
        print(f"📈 Summary: {running_count}/{len(pids)} instances running")
        
        # Show all active ports in 6000 range
        try:
            result = subprocess.run(['ss', '-tuln'], capture_output=True, text=True)
            ports = []
            for line in result.stdout.split('\n'):
                if ':60' in line and 'LISTEN' in line:
                    port = line.split(':')[1].split()[0]
                    if port.isdigit():
                        ports.append(port)
            if ports:
                print(f"🌐 All active ports: {', '.join(sorted(set(ports)))}")
        except:
            pass
    
    def list_logs(self):
        """List available log files."""
        log_files = list(self.logs_dir.glob("vllm_server_*.log"))
        if not log_files:
            print("📭 No log files found")
            return
        
        print(f"📝 Available log files:")
        for log_file in sorted(log_files):
            size = log_file.stat().st_size / 1024 / 1024  # MB
            print(f"   {log_file.name} ({size:.1f} MB)")
        
        print(f"\n💡 View logs with: tail -f logs/vllm_server_<port>.log")


def main():
    parser = argparse.ArgumentParser(description="vLLM Manager - Easy vLLM instance management")
    parser.add_argument("command", choices=['start', 'add', 'stop', 'stop-single', 'status', 'logs'], 
                       help="Command to execute")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                       help="Model to serve")
    parser.add_argument("--count", type=int, default=1,
                       help="Number of instances to start (for 'start' command)")
    parser.add_argument("--start-port", type=int, default=6000,
                       help="Starting port number")
    parser.add_argument("--port", type=int,
                       help="Specific port to use (for 'add' command)")
    parser.add_argument("--gpu", type=int,
                       help="Specific GPU to use (for 'add' command)")
    parser.add_argument("--pid", type=str,
                       help="PID or port number to stop (for stop-single command)")
    
    args = parser.parse_args()
    manager = VLLMManager()
    
    if args.command == "start":
        if args.count == 1:
            # Single instance - use the new single model function
            manager.start_single_model(args.model, args.port, args.gpu)
        else:
            # Multiple instances of same model
            manager.start_multiple(args.model, args.count, args.start_port)
    elif args.command == "add":
        # Add a single model instance
        manager.start_single_model(args.model, args.port, args.gpu)
    elif args.command == "stop":
        manager.stop_all()
    elif args.command == "stop-single":
        if not args.pid:
            print("❌ --pid argument required for stop-single command")
            print("Usage: python vllm_manager.py stop-single --pid <PID_or_PORT>")
            return
        manager.stop_single(args.pid)
    elif args.command == "status":
        manager.status()
    elif args.command == "logs":
        manager.list_logs()


if __name__ == "__main__":
    main() 