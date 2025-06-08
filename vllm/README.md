# vLLM Manager 🚀

Easy management of multiple vLLM instances with automatic GPU allocation and port management.

NEW SYSTEM DOES NOT USE CONFIGS.

## 🎯 Quick Start

### Start a Single Instance
```bash
python vllm_manager.py start --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
```

### Start Multiple Instances
```bash
python vllm_manager.py start --count 4 --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
```

### Add Different Models One by One
```bash
python vllm_manager.py add --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
python vllm_manager.py add --model "meta-llama/Llama-3.1-8B-Instruct"
python vllm_manager.py add --model "microsoft/DialoGPT-medium" --port 6006 --gpu 2
```

### Check Status
```bash
python vllm_manager.py status
```

### Stop All Instances
```bash
python vllm_manager.py stop
```

### Stop Single Instance
```bash
python vllm_manager.py stop-single --pid 6000        # Stop instance on port 6000
python vllm_manager.py stop-single --pid 123456      # Stop instance with PID 123456
```

## 📋 Manual Commands

### Using the Python Manager Directly
```bash
# Start multiple instances of same model
python vllm_manager.py start --count 4 --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Add individual models one by one
python vllm_manager.py add --model "meta-llama/Llama-3.1-8B-Instruct"
python vllm_manager.py add --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" --port 6005 --gpu 2

# Check status (shows model details)
python vllm_manager.py status

# Stop all
python vllm_manager.py stop

# Stop single instance (by port or PID)
python vllm_manager.py stop-single --pid 6000
python vllm_manager.py stop-single --pid 123456

# List log files
python vllm_manager.py logs
```

### Available Models
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (default)
- `meta-llama/Meta-Llama-3.1-8B-Instruct`
- Any Hugging Face model compatible with vLLM

## 📁 Directory Structure

```
vllm/
├── vllm_manager.py         # Main management script
├── logs/                   # Log files (auto-created)
│   └── vllm_server_*.log
├── configs/                # Old config files (archived)
└── archive/                # Old scripts (archived)
```

## 🔧 Features

- **Automatic GPU Detection**: Finds available GPUs sorted by memory usage
- **Port Management**: Automatically finds available ports starting from 6000
- **Process Tracking**: Keeps track of all started instances via PID file
- **Organized Logs**: All logs saved to `logs/vllm_server_<port>.log`
- **Easy Cleanup**: Single command to stop all managed instances
- **Status Monitoring**: Real-time status of all instances

## 📝 Logs

View logs in real-time:
```bash
# View specific instance log
tail -f logs/vllm_server_6000.log

# List all log files
python vllm_manager.py logs
```

## 🛠️ Troubleshooting

### Check What's Running
```bash
python vllm_manager.py status
```

### Kill All Processes (Emergency)
```bash
python vllm_manager.py stop
```

### Check GPU Usage
```bash
nvidia-smi
```

### Check Port Usage
```bash
ss -tuln | grep :60
```

## 🚀 Usage Examples

### For Response Generation (4 GPUs)
```bash
# Start instances
python vllm_manager.py start --count 4 --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Wait for startup and check status
python vllm_manager.py status

# Generate responses using all 4 GPUs
python scripts/generate_responses_multi_gpu.py --ports 6000 6001 6002 6003
```

### For Development (1 GPU)
```bash
python vllm_manager.py start --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# Use port 6000 in your client code
```

### Mixed Model Setup (Different models on different GPUs)
```bash
# Add different models one by one
python vllm_manager.py add --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
python vllm_manager.py add --model "meta-llama/Llama-3.1-8B-Instruct"
python vllm_manager.py add --model "microsoft/DialoGPT-medium"

# Check what's running
python vllm_manager.py status
# Output shows: Port 6000: Qwen, Port 6001: Llama, Port 6002: DialoGPT
```

## ⚡ Performance Notes

- Each instance uses 1 GPU
- Startup takes ~30-60 seconds per instance
- Check status regularly during startup
- Logs show when models are fully loaded and ready

## 🚀 Multi-GPU Response Generation

The vLLM manager integrates seamlessly with the multi-GPU response generation system for maximum speed.

### Step 1: Start Multiple vLLM Instances
```bash
# Start 4 instances across 4 GPUs
python vllm_manager.py start --count 4 --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Check that all instances are running
python vllm_manager.py status
# Output: Port 6000: Qwen, Port 6001: Qwen, Port 6002: Qwen, Port 6003: Qwen
```

### Step 2: Generate Responses with Multi-GPU Acceleration
```bash
# Generate responses using all 4 instances (dramatically faster!)
python scripts/generate_responses_multi_gpu.py --ports 6000 6001 6002 6003

# Generate for specific dataset only
python scripts/generate_responses_multi_gpu.py --ports 6000 6001 6002 6003 --cue stanford_professor

# Adjust parallelism (default: 4 workers per GPU)
python scripts/generate_responses_multi_gpu.py --ports 6000 6001 6002 6003 --workers-per-gpu 8
```

### Performance Benefits
- **4x Speed**: 4 GPUs = ~4x faster response generation
- **Load Balancing**: Work distributed evenly across all instances
- **Resume Support**: Automatically resumes if interrupted
- **Real-time Progress**: Shows completion rate, speed, and ETA

### Example Output
```
🚀 Using 4 vLLM instances across 4 ports
📊 Total entries: 1531
✅ Already processed: 193
🔄 To process: 1338
🔥 Total parallel workers: 16

✅ 1200/1338 (89.7%) | 5.2 resp/s | ETA: 4.4min
🎉 Response generation completed!
⏱️  Total time: 12.3 minutes
🚀 Average rate: 108.8 responses/minute
```

### Mixed Model Multi-GPU Setup
```bash
# Start different models on different GPUs
python vllm_manager.py add --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" --port 6000
python vllm_manager.py add --model "meta-llama/Llama-3.1-8B-Instruct" --port 6001

# Use both models for comparison
python scripts/generate_responses_multi_gpu.py --ports 6000 6001
```

### Complete Workflow Example
```bash
# 1. Start vLLM instances
python vllm_manager.py start --count 4 --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# 2. Verify all instances are ready
python vllm_manager.py status

# 3. Generate responses (much faster with 4 GPUs!)
python scripts/generate_responses_multi_gpu.py --ports 6000 6001 6002 6003

# 4. When done, stop all instances
python vllm_manager.py stop
```

## 🔄 Migration from Old System

The old YAML-based configuration files are preserved in `configs/` and old scripts in `archive/` for reference. 