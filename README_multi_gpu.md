# Fast Multi-GPU Response Generation

High-performance response generation script using multiple VLLM instances across different GPUs with simple progress tracking.

## Features

⚡ **Multi-GPU Processing**
- Dynamic load balancing across GPUs
- Async/await for maximum concurrency  
- Automatic retry with exponential backoff
- Health monitoring and failover

📊 **Simple Progress Tracking**
- Clean tqdm progress bar
- Real-time statistics in progress bar
- Per-GPU status display
- Performance metrics (rate, active requests, errors)

## Installation

```bash
# Install required packages
pip install aiohttp uvloop tqdm rich
```

## Usage

### Basic Usage
```bash
# Use multiple VLLM instances
python scripts/generate_responses_multi_gpu.py --ports 6001 6002 6003 6004
```

### Advanced Options
```bash
# Custom configuration
python scripts/generate_responses_multi_gpu.py \
    --ports 6001 6002 6003 6004 6005 \
    --workers-per-gpu 6 \
    --cue stanford_professor \
    --output-dir data/fast_responses
```

## Progress Display

The script shows a simple progress bar with key metrics:

```
🚀 Generating responses: 42%|████▎     | 420/1000 [02:15<03:45, 2.5resp/s] Rate: 3.1/s, Active: 8, GPUs: 4/4, Errors: 2
```

Shows:
- Progress percentage and bar
- Current/total items processed  
- Elapsed time and ETA
- Current processing rate
- Active requests across all GPUs
- Healthy GPUs count
- Total errors encountered

## Performance

Expected throughput:
- **Single GPU**: ~2-4 responses/sec
- **4 GPUs**: ~8-16 responses/sec  
- **8 GPUs**: ~16-32 responses/sec

Adjust `--workers-per-gpu` based on your GPU memory and model size. 