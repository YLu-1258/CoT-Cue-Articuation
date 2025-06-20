#!/bin/bash
# Complete VERL Qwen3-4B GSM8K Training Pipeline

echo "🚀 VERL Qwen3-4B GSM8K Training Pipeline"
echo "========================================"

# 1. Setup environment (if needed)
if [ ! -d "verl" ]; then
    echo "📦 Setting up environment..."
    bash setup_environment.sh
fi

# 2. Prepare GSM8K dataset
if [ ! -f "/data/kevinchu/gsm8k/train.parquet" ]; then
    echo "📊 Preparing GSM8K dataset..."
    python scripts/prepare_gsm8k_data.py --local_dir /data/kevinchu/gsm8k
fi

# 3. Setup Wandb (optional)
echo "🔗 Setup Wandb for monitoring? (y/n)"
read -r setup_wandb
if [ "$setup_wandb" = "y" ]; then
    wandb login
fi

# 4. Start training
echo "🚀 Starting GRPO training..."
bash scripts/run_qwen3_4b_grpo_cuda115.sh

echo "✅ Pipeline complete!"
echo "📄 Training log: verl_grpo_qwen3_4b.log" 