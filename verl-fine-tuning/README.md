# VERL Fine-tuning for Qwen3-4B on GSM8K

Minimal setup for fine-tuning Qwen3-4B using VERL GRPO on GSM8K math problems.

## 🚀 One-Command Start

```bash
cd verl-fine-tuning && bash train.sh
```

**Or step by step:**

```bash
cd verl-fine-tuning

# 1. Setup environment
bash setup_environment.sh

# 2. Prepare GSM8K dataset  
python scripts/prepare_gsm8k_data.py --local_dir /data/kevinchu/gsm8k

# 3. Start training
bash scripts/run_qwen3_4b_grpo_cuda115.sh
```

## 📊 Monitor Training

```bash
# Watch training progress
tail -f verl_grpo_qwen3_4b.log

# Setup Wandb (optional)
wandb login
```

## 🎯 What You Get

- **Built-in GSM8K reward function** (extracts "####" answers automatically)
- **Real-time metrics**: `val/test_score/openai/gsm8k` (target >0.8)
- **CUDA 11.5 optimized** (no FlashAttention required)
- **Memory efficient**: ~12-16GB VRAM
- **Training time**: ~6-8 hours on single A100

## 📁 Structure

```
verl-fine-tuning/
├── scripts/
│   ├── prepare_gsm8k_data.py      # Uses official VERL preprocessing
│   └── run_qwen3_4b_grpo_cuda115.sh # Official GRPO training script
├── setup_environment.sh           # One-time setup
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🔧 Key Features

- Uses **official VERL GRPO algorithm** (`algorithm.adv_estimator=grpo`)
- **vLLM backend** with multiple rollouts (`rollout.n=5`)
- **Advanced KL regularization** (`kl_loss_type=low_var_kl`)
- **Automatic answer extraction** and scoring for math problems
- **Built-in Wandb integration** for monitoring

Based on official VERL example: `verl/examples/grpo_trainer/run_qwen3-8b.sh` 