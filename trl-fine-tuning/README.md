# TRL GRPO Fine-tuning for Qwen3-4B on GSM8K

A clean, modular implementation of GRPO (Group Relative Policy Optimization) fine-tuning using TRL for Qwen models on GSM8K math reasoning tasks.

## 🎯 What This Does

- **Trains Qwen3-4B** (or Qwen3-1B fallback) using GRPO reinforcement learning
- **Binary reward system**: +1 for correct math answers, 0 for incorrect
- **GSM8K dataset**: Grade school math word problems
- **Memory efficient**: Runs on single GPU with ~12-16GB VRAM

## 📁 Project Structure

```
trl-fine-tuning/
├── config/
│   └── grpo_config.py       # Training hyperparameters
├── data/
│   └── gsm8k_loader.py      # Dataset loading and formatting
├── models/
│   └── reward_function.py   # Binary reward function (+1/0)
├── training/
│   └── train_grpo.py        # Main training script
├── evaluation/
│   └── evaluate.py          # Model evaluation on test set
├── scripts/
│   ├── train.sh             # Training launcher
│   └── test_setup.py        # Verify setup works
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Make sure you're in the finetune conda environment
conda activate finetune

# Install dependencies
cd trl-fine-tuning
pip install -r requirements.txt
```

### 2. Verify Setup

```bash
# Run comprehensive setup test
python scripts/test_setup.py
```

This will check:
- ✅ Required packages (TRL, transformers, etc.)
- ✅ Custom modules can be imported
- ✅ Model access (Qwen3-4B or Qwen3-1B)
- ✅ GSM8K dataset download
- ✅ Reward function works correctly

### 3. Start Training

```bash
# Simple one-command training
bash scripts/train.sh
```

**What happens during training:**
- Downloads Qwen3-4B (or falls back to Qwen3-1B)
- Loads 1000 GSM8K training examples
- Trains for 2 epochs with GRPO
- Saves model to `./outputs/qwen_grpo_gsm8k/`
- Logs everything to `training.log`

### 4. Evaluate Results

```bash
# Evaluate on 100 test examples
python evaluation/evaluate.py --model_path ./outputs/qwen_grpo_gsm8k --num_samples 100

# Or evaluate on more examples
python evaluation/evaluate.py --model_path ./outputs/qwen_grpo_gsm8k --num_samples 500
```

## ⚙️ How It Works

### Reward Function
The binary reward function (`models/reward_function.py`) gives:
- **+1.0** if the model's answer matches the ground truth
- **0.0** if wrong or no answer found

It extracts answers from multiple formats:
```python
"#### 42"           # GSM8K standard format
"The answer is 42"  # Natural language
"Answer: 42"        # Alternative format
```

### Training Configuration
Key settings in `config/grpo_config.py`:
- **2 epochs** (quick training)
- **Batch size 2** per device, 16 gradient accumulation
- **Learning rate 5e-7** with cosine scheduling
- **GRPO beta 0.01** (KL divergence penalty)
- **512 max tokens** for generation

### Dataset Format
GSM8K examples are formatted as:
```
Question: Janet's ducks lay 16 eggs per day...

Answer: Let me solve this step by step.
```

## 📊 Expected Results

### Training Progress
- **Start**: ~10-20% accuracy (untrained)
- **After 1 epoch**: ~40-60% accuracy
- **After 2 epochs**: ~60-80% accuracy

### Performance Metrics
- **Training time**: 2-4 hours on RTX 4090
- **Memory usage**: 12-16GB VRAM
- **Final accuracy**: 60-80% on GSM8K test set

### Real-time Monitoring
During training you'll see:
```
Current accuracy: 0.650 (13/20)
```

## 🔧 Customization

### Change Training Parameters

Edit `config/grpo_config.py`:
```python
def get_grpo_config(output_dir: str = "./outputs/qwen_grpo_gsm8k"):
    return GRPOConfig(
        num_train_epochs=5,           # Train longer
        per_device_train_batch_size=4, # Use more memory
        learning_rate=1e-6,           # Different learning rate
        beta=0.02,                    # Stronger KL penalty
    )
```

### Use More Training Data

Edit `training/train_grpo.py`:
```python
# Use more GSM8K examples
train_dataset, test_dataset = prepare_gsm8k_dataset(
    subset_size=5000,  # Use 5000 training examples
    test_size=1000     # Use 1000 test examples
)
```

### Modify Reward Function

Edit `models/reward_function.py` to implement custom rewards:
```python
def __call__(self, completions, **kwargs):
    rewards = []
    for completion in completions:
        # Your custom logic here
        if "step by step" in completion:
            reward = 1.0  # Reward step-by-step reasoning
        else:
            reward = 0.0
        rewards.append(reward)
    return rewards
```

## 🚨 Troubleshooting

### Memory Issues
```bash
# Edit config/grpo_config.py
per_device_train_batch_size=1,  # Reduce batch size
gradient_accumulation_steps=32, # Increase accumulation
```

### Model Download Issues
The script automatically tries:
1. `Qwen/Qwen3-4B` (preferred)
2. `Qwen/Qwen3-1B` (fallback)

If both fail, check your internet connection or HuggingFace access.

### Import Errors
```bash
# Make sure you're in the right directory
cd trl-fine-tuning

# Test imports
python scripts/test_setup.py
```

### Training Stuck
```bash
# Monitor GPU usage
nvidia-smi

# Check training log
tail -f training.log
```

## 📈 Monitoring Training

### Console Output
Real-time updates during training:
```
🚀 Starting TRL GRPO training for Qwen3-4B on GSM8K
✅ Using model: Qwen/Qwen3-4B
📊 Loading GSM8K dataset...
🔧 Initializing GRPO trainer...
Current accuracy: 0.450 (9/20)
```

### Weights & Biases
If you have wandb set up:
```bash
wandb login
# Training automatically logs to wandb
```

### Log Files
- `training.log`: Complete training output
- `evaluation_results_100.json`: Detailed evaluation results

## 🧪 Advanced Usage

### Run Tests Only
```bash
# Test specific components
python -c "from models.reward_function import GSM8KRewardFunction; print('Reward function works!')"
python -c "from data.gsm8k_loader import prepare_gsm8k_dataset; print('Data loader works!')"
```

### Custom Evaluation
```python
# Create your own evaluation script
from evaluation.evaluate import evaluate_model

accuracy = evaluate_model(
    model_path="./outputs/qwen_grpo_gsm8k",
    num_samples=50  # Quick test
)
print(f"Accuracy: {accuracy:.3f}")
```

### Multi-GPU Training
Edit `scripts/train.sh`:
```bash
export CUDA_VISIBLE_DEVICES=0,1  # Use multiple GPUs
```

Then edit `config/grpo_config.py`:
```python
per_device_train_batch_size=1,  # Reduce per-device batch size
gradient_accumulation_steps=32, # Increase accumulation
```

## 💡 Tips & Best Practices

### For Development
- Start with `subset_size=100` for quick testing
- Use `num_train_epochs=1` for initial runs
- Monitor `nvidia-smi` for memory usage

### For Production
- Increase `subset_size=5000` for better performance
- Train for `num_train_epochs=5-10`
- Save checkpoints frequently
- Use wandb for experiment tracking

### For Debugging
- Check `training.log` for errors
- Run `python scripts/test_setup.py` first
- Test components individually

## 🔗 References

- **TRL Documentation**: [GRPO Trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- **GSM8K Dataset**: [HuggingFace](https://huggingface.co/datasets/gsm8k)
- **Qwen Models**: [HuggingFace](https://huggingface.co/Qwen)
- **Original Paper**: [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300)

## 🤝 Contributing

Found an issue or want to improve something?
1. Test your changes with `python scripts/test_setup.py`
2. Make sure training still works with `bash scripts/train.sh`
3. Update this README if you add new features 