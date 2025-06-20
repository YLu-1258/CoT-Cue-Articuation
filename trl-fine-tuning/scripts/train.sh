#!/bin/bash
# TRL GRPO Training Launcher

set -e

echo "🚀 TRL GRPO Training for Qwen3-4B on GSM8K"
echo "============================================"

# Check if we're in the right directory
if [ ! -f "training/train_grpo.py" ]; then
    echo "❌ Error: Please run this script from the trl-fine-tuning directory"
    exit 1
fi

# Install requirements if needed
echo "📦 Checking requirements..."
if ! python -c "import trl" 2>/dev/null; then
    echo "📦 Installing requirements..."
    pip install -r requirements.txt
fi

# Set environment variables
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# Create outputs directory
mkdir -p outputs

# Run training
echo "🚀 Starting training..."
python training/train_grpo.py 2>&1 | tee training.log

echo "✅ Training completed!"
echo "📁 Model saved in: ./outputs/qwen_grpo_gsm8k"
echo "�� Log: training.log" 