#!/bin/bash
# Setup Wandb for training monitoring

echo "🔗 Setting up Weights & Biases for monitoring..."

# Check if wandb is installed
if ! python -c "import wandb" 2>/dev/null; then
    echo "📦 Installing wandb..."
    pip install wandb
fi

# Login to wandb
echo "Please login to Wandb (get your API key from https://wandb.ai/authorize)"
wandb login

# Test wandb connection
python -c "import wandb; wandb.init(project='test', mode='disabled'); print('✅ Wandb setup complete!')"

echo "✅ Wandb is ready!"
echo "Your training will automatically log to: https://wandb.ai"
echo "Look for project: 'qwen_grpo_gsm8k'" 