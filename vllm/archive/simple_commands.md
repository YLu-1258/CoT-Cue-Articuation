# Simple vLLM Management

## Start Servers (Background with nohup)
```bash
# Start all servers in background (default mode)
python vllm_serve.py --config vllm_serve_config.yaml

# With HF token for Meta Llama models
export HF_TOKEN="your_hf_token_here"
python vllm_serve.py --config vllm_serve_config.yaml

# Debug mode (runs in foreground for testing)
python vllm_serve.py --config vllm_serve_config.yaml --debug
```

## Stop All Servers (Easy Way)
```bash
# Built-in kill function (recommended)
python vllm_serve.py --config vllm_serve_config.yaml --kill

# Or manual kill (only your processes)
kill $(cat vllm_pids.txt)
```

## Check Status
```bash
# See if processes are running
ps aux | grep vllm

# Check recent logs (separate log for each server)
tail -f vllm_server_6004.log
tail -f vllm_server_6005.log

# Check which ports are being used
netstat -tuln | grep -E ":(6004|6005)"
```

## What Happens When You Start
✅ **Automatic nohup**: Servers run in background  
✅ **Individual logs**: Each server gets its own log file  
✅ **PID tracking**: Process IDs saved to `vllm_pids.txt`  
✅ **Easy cleanup**: Built-in `--kill` option (only kills your processes)  
✅ **HF auth**: Automatically passes HF_TOKEN to servers  

## Files Created
- `vllm_pids.txt` - Process IDs for easy cleanup
- `vllm_server_6004.log` - Logs for server on port 6004  
- `vllm_server_6005.log` - Logs for server on port 6005

## Setup HF Authentication
```bash
# Method 1: Environment variable (temporary)
export HF_TOKEN="your_token_here"

# Method 2: Login with CLI (permanent)
pip install huggingface_hub
huggingface-cli login

# Method 3: Add to .bashrc (permanent)
echo 'export HF_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

## Quick Reference
- **Start**: `python vllm_serve.py --config vllm_serve_config.yaml`
- **Stop**: `python vllm_serve.py --config vllm_serve_config.yaml --kill`
- **Debug**: `python vllm_serve.py --config vllm_serve_config.yaml --debug`
- **Logs**: `tail -f vllm_server_[port].log` 