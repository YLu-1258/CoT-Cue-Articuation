#!/usr/bin/env python3
"""
Fast multi-GPU response generation for CoT cue articulation experiments.

This script maximizes throughput by:
- Using multiple VLLM instances across different GPUs/ports
- Dynamic load balancing across GPUs
- Async/concurrent processing with optimal thread pools
- Real-time progress tracking and performance monitoring
- Intelligent batching and retry mechanisms
"""

import argparse
import asyncio
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
import threading
from queue import Queue, Empty
import signal

import aiohttp
import uvloop  # High-performance event loop
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enums.cue import Cue


@dataclass
class GPUInstance:
    """Represents a GPU instance running VLLM."""
    port: int
    base_url: str = field(init=False)
    model_id: Optional[str] = None
    active_requests: int = 0
    total_requests: int = 0
    total_tokens: int = 0
    avg_response_time: float = 0.0
    last_response_time: float = 0.0
    error_count: int = 0
    is_healthy: bool = True
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def __post_init__(self):
        self.base_url = f"http://localhost:{self.port}/v1"
    
    @property
    def load_score(self) -> float:
        """Calculate load score for load balancing (lower = less loaded)."""
        if not self.is_healthy:
            return float('inf')
        
        # Combine active requests, average response time, and error rate
        base_score = self.active_requests
        time_penalty = min(self.avg_response_time / 1000, 5.0)  # Cap at 5s penalty
        error_penalty = self.error_count * 0.1
        
        return base_score + time_penalty + error_penalty
    
    def update_stats(self, response_time: float, tokens: int = 0, error: bool = False):
        """Thread-safe stats update."""
        with self.lock:
            self.total_requests += 1
            self.last_response_time = response_time
            
            if error:
                self.error_count += 1
                self.is_healthy = self.error_count < 10  # Mark unhealthy after 10 errors
            else:
                self.total_tokens += tokens
                # Exponential moving average for response time
                alpha = 0.1
                self.avg_response_time = alpha * response_time + (1 - alpha) * self.avg_response_time


@dataclass
class WorkItem:
    """A work item to be processed."""
    question_id: int
    entry: Dict[str, Any]
    cue: Cue
    attempts: int = 0
    max_attempts: int = 3


class FastResponseGenerator:
    """High-performance multi-GPU response generator."""
    
    def __init__(self, ports: List[int], output_dir: str, max_workers_per_gpu: int = 4):
        """
        Initialize the fast response generator.
        
        Args:
            ports: List of ports where VLLM instances are running
            output_dir: Directory to save responses
            max_workers_per_gpu: Number of concurrent workers per GPU
        """
        self.gpus = [GPUInstance(port) for port in ports]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers_per_gpu = max_workers_per_gpu
        self.total_workers = len(ports) * max_workers_per_gpu
        
        # Performance tracking
        self.stats = {
            'start_time': 0,
            'total_processed': 0,
            'total_errors': 0,
            'responses_per_second': 0.0,
            'estimated_completion': 0,
        }
        
        # Work management
        self.work_queue = Queue()
        self.retry_queue = Queue()
        self.completed_queue = Queue()
        
        # Progress tracking
        self.console = Console()
        self.progress_bar = None
        self.shutdown_event = threading.Event()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.console.print("\n🛑 Shutdown signal received. Finishing current work...")
        self.shutdown_event.set()
    
    async def _test_gpu_health(self, gpu: GPUInstance) -> bool:
        """Test if a GPU instance is healthy and get model info."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                # Test models endpoint
                async with session.get(f"{gpu.base_url}/models") as response:
                    if response.status == 200:
                        models_data = await response.json()
                        if models_data.get('data'):
                            gpu.model_id = models_data['data'][0]['id']
                            gpu.is_healthy = True
                            return True
        except Exception as e:
            self.console.print(f"❌ GPU {gpu.port} health check failed: {e}")
        
        gpu.is_healthy = False
        return False
    
    async def _initialize_gpus(self):
        """Initialize and health check all GPU instances."""
        self.console.print("🔍 Checking GPU instance health...")
        
        health_tasks = [self._test_gpu_health(gpu) for gpu in self.gpus]
        health_results = await asyncio.gather(*health_tasks, return_exceptions=True)
        
        healthy_gpus = sum(1 for gpu in self.gpus if gpu.is_healthy)
        
        if healthy_gpus == 0:
            raise RuntimeError("❌ No healthy GPU instances found!")
        
        self.console.print(f"✅ {healthy_gpus}/{len(self.gpus)} GPU instances are healthy")
        
        # Display GPU info in simple format
        for gpu in self.gpus:
            status = "🟢" if gpu.is_healthy else "🔴"
            model_name = gpu.model_id[:30] + "..." if gpu.model_id and len(gpu.model_id) > 30 else gpu.model_id or "Unknown"
            self.console.print(f"  {status} Port {gpu.port}: {model_name}")
    
    def _select_best_gpu(self) -> Optional[GPUInstance]:
        """Select the GPU with the lowest load score."""
        healthy_gpus = [gpu for gpu in self.gpus if gpu.is_healthy]
        if not healthy_gpus:
            return None
        
        return min(healthy_gpus, key=lambda g: g.load_score)
    
    async def _make_request(self, gpu: GPUInstance, prompt: str) -> str:
        """Make a request to a specific GPU instance."""
        start_time = time.time()
        
        with gpu.lock:
            gpu.active_requests += 1
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
                payload = {
                    "model": gpu.model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 1024
                }
                
                async with session.post(f"{gpu.base_url}/chat/completions", json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        
                        # Update stats
                        response_time = (time.time() - start_time) * 1000
                        tokens = len(content.split())  # Rough token estimate
                        gpu.update_stats(response_time, tokens, error=False)
                        
                        return content
                    else:
                        error_text = await response.text()
                        raise aiohttp.ClientError(f"HTTP {response.status}: {error_text}")
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            gpu.update_stats(response_time, error=True)
            raise e
        
        finally:
            with gpu.lock:
                gpu.active_requests -= 1
    
    async def _process_work_item(self, work_item: WorkItem) -> Optional[Dict]:
        """Process a single work item."""
        try:
            # Select best GPU
            gpu = self._select_best_gpu()
            if not gpu:
                raise RuntimeError("No healthy GPUs available")
            
            # Generate both responses
            unbiased_response = await self._make_request(gpu, work_item.entry["unbiased_question"])
            biased_response = await self._make_request(gpu, work_item.entry["biased_question"])
            
            return {
                "question_id": work_item.question_id,
                "unbiased_question": work_item.entry["unbiased_question"],
                "unbiased_response": unbiased_response,
                "biased_question": work_item.entry["biased_question"],
                "biased_response": biased_response,
                "correct_answer": work_item.entry["correct_answer"],
                "suggested_wrong_answer": work_item.entry["biased_answer"],
                "cue_type": work_item.entry["cue_type"],
                "gpu_port": gpu.port,
                "status": "success"
            }
        
        except Exception as e:
            work_item.attempts += 1
            if work_item.attempts < work_item.max_attempts:
                # Retry with exponential backoff
                await asyncio.sleep(2 ** work_item.attempts)
                return await self._process_work_item(work_item)
            else:
                return {
                    "question_id": work_item.question_id,
                    "error": str(e),
                    "attempts": work_item.attempts,
                    "status": "error"
                }
    
    def _get_stats_summary(self) -> str:
        """Get a summary of current stats for tqdm postfix."""
        elapsed = time.time() - self.stats['start_time'] if self.stats['start_time'] > 0 else 0.001
        processed = self.stats['total_processed']
        rate = processed / elapsed if elapsed > 0 else 0
        
        # GPU stats summary
        active_total = sum(gpu.active_requests for gpu in self.gpus if gpu.is_healthy)
        healthy_count = sum(1 for gpu in self.gpus if gpu.is_healthy)
        
        return f"Rate: {rate:.1f}/s, Active: {active_total}, GPUs: {healthy_count}/{len(self.gpus)}, Errors: {self.stats['total_errors']}"
    
    async def _worker(self, worker_id: int, work_queue: asyncio.Queue, results_queue: asyncio.Queue):
        """Async worker that processes work items."""
        while not self.shutdown_event.is_set():
            try:
                work_item = await asyncio.wait_for(work_queue.get(), timeout=1.0)
                result = await self._process_work_item(work_item)
                await results_queue.put(result)
                work_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.console.print(f"❌ Worker {worker_id} error: {e}")
    
    async def generate_responses_for_dataset(self, input_file: Path, cue: Cue, resume: bool = True) -> Path:
        """Generate responses for a dataset using all available GPUs."""
        output_file = self.output_dir / f"{cue.value}_responses.jsonl"
        
        self.console.print(Panel(f"[bold green]Generating responses for {cue.display_name}[/bold green]"))
        self.console.print(f"📥 Input: {input_file}")
        self.console.print(f"📤 Output: {output_file}")
        
        # Initialize GPUs
        await self._initialize_gpus()
        
        # Load and prepare data
        with open(input_file, 'r') as f:
            data_entries = [json.loads(line.strip()) for line in f if line.strip()]
        
        # Handle resume logic
        existing_ids = set()
        if resume and output_file.exists():
            with open(output_file, 'r') as f:
                existing_responses = [json.loads(line.strip()) for line in f if line.strip()]
                existing_ids = {resp.get("question_id", -1) for resp in existing_responses}
            self.console.print(f"📋 Found {len(existing_ids)} existing responses. Resuming...")
        
        # Prepare work items
        work_items = []
        for i, entry in enumerate(data_entries):
            if i not in existing_ids:
                work_items.append(WorkItem(i, entry, cue))
        
        total_work = len(work_items)
        if total_work == 0:
            self.console.print("✅ All responses already generated!")
            return output_file
        
        self.console.print(f"🔢 Processing {total_work} new items")
        
        # Set up async queues
        work_queue = asyncio.Queue()
        results_queue = asyncio.Queue()
        
        # Add work items to queue
        for item in work_items:
            await work_queue.put(item)
        
        # Start workers
        workers = []
        for i in range(self.total_workers):
            worker = asyncio.create_task(self._worker(i, work_queue, results_queue))
            workers.append(worker)
        
        # Track progress and write results
        self.stats['start_time'] = time.time()
        processed = 0
        
        # Create progress bar
        pbar = tqdm(total=total_work, desc="🚀 Generating responses", unit="resp")
        
        try:
            with open(output_file, 'a') as f:
                while processed < total_work and not self.shutdown_event.is_set():
                    try:
                        result = await asyncio.wait_for(results_queue.get(), timeout=1.0)
                        
                        # Write result immediately
                        f.write(json.dumps(result) + "\n")
                        f.flush()
                        
                        processed += 1
                        self.stats['total_processed'] = processed
                        
                        if result.get('status') == 'error':
                            self.stats['total_errors'] += 1
                        
                        # Update progress bar
                        pbar.update(1)
                        pbar.set_postfix_str(self._get_stats_summary())
                        
                    except asyncio.TimeoutError:
                        # Periodically update stats even without new results
                        pbar.set_postfix_str(self._get_stats_summary())
                        continue
        finally:
            pbar.close()
        
        # Cleanup workers
        for worker in workers:
            worker.cancel()
        
        self.console.print(f"✅ Response generation completed: {output_file}")
        return output_file
    
    async def generate_all_responses(self, data_dir: str = "data/prompts") -> Dict[Cue, Path]:
        """Generate responses for all available datasets."""
        data_path = Path(data_dir)
        results = {}
        
        for cue in Cue:
            input_file = data_path / f"{cue.value}.jsonl"
            if input_file.exists():
                output_file = await self.generate_responses_for_dataset(input_file, cue)
                results[cue] = output_file
                self.console.print()
            else:
                self.console.print(f"❌ Dataset not found: {input_file}")
        
        return results


async def main():
    """Main async function."""
    parser = argparse.ArgumentParser(description="Fast multi-GPU response generation")
    parser.add_argument("--ports", type=int, nargs='+', required=True,
                       help="List of ports where VLLM instances are running")
    parser.add_argument("--workers-per-gpu", type=int, default=4,
                       help="Number of concurrent workers per GPU (default: 4)")
    parser.add_argument("--cue", type=str, choices=[cue.value for cue in Cue],
                       help="Generate responses for specific cue only")
    parser.add_argument("--output-dir", type=str, default="data/responses/raw",
                       help="Output directory for responses")
    
    args = parser.parse_args()
    
    # Use uvloop for better performance
    uvloop.install()
    
    console = Console()
    console.print("[bold green]Fast Multi-GPU Response Generation[/bold green]")
    console.print("=" * 60)
    
    # Create generator
    generator = FastResponseGenerator(
        ports=args.ports,
        output_dir=args.output_dir,
        max_workers_per_gpu=args.workers_per_gpu
    )
    
    try:
        if args.cue:
            # Generate for specific cue
            cue = Cue(args.cue)
            input_file = Path("data/prompts") / f"{cue.value}.jsonl"
            
            if not input_file.exists():
                console.print(f"❌ Dataset not found: {input_file}")
                console.print("Run 'python scripts/generate_data.py' first")
                return 1
            
            await generator.generate_responses_for_dataset(input_file, cue)
        else:
            # Generate for all cues
            results = await generator.generate_all_responses()
            
            if results:
                console.print("\n🎉 [bold green]All responses generated successfully![/bold green]")
                for cue, output_file in results.items():
                    console.print(f"  {cue.display_name}: {output_file}")
            else:
                console.print("\n⚠️  No datasets found for response generation.")
                console.print("Run 'python scripts/generate_data.py' first")
                return 1
    
    except KeyboardInterrupt:
        console.print("\n🛑 Generation interrupted by user")
        return 1
    except Exception as e:
        console.print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 