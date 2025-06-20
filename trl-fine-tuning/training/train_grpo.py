#!/usr/bin/env python3
"""
TRL GRPO Training for Qwen3-4B on GSM8K
Main training script using modular components
"""

import os
import sys
import logging
import argparse
from transformers import AutoTokenizer
from trl import GRPOTrainer

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.grpo_config import get_grpo_config, get_model_candidates
from data.gsm8k_loader import prepare_gsm8k_dataset
from models.reward_function import GSM8KRewardFunction

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def select_model():
    """Select the best available Qwen model."""
    model_candidates = get_model_candidates()
    
    for model_name in model_candidates:
        try:
            # Try to load tokenizer to check if model exists
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"✅ Using model: {model_name}")
            return model_name
        except Exception as e:
            logger.warning(f"❌ Model {model_name} not available: {e}")
            continue
    
    raise ValueError("No suitable Qwen model found!")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="GRPO Training with optional vLLM integration")
    parser.add_argument("--model", type=str, help="Model name to train (overrides auto-selection)")
    parser.add_argument("--no-vllm", action="store_true", help="Disable vLLM integration")
    parser.add_argument("--vllm-port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--vllm-host", type=str, default="0.0.0.0", help="vLLM server host")
    parser.add_argument("--output-dir", type=str, default="./outputs/qwen_grpo_gsm8k", help="Output directory")
    parser.add_argument("--subset-size", type=int, default=1000, help="Training dataset subset size")
    parser.add_argument("--test-size", type=int, default=200, help="Test dataset size")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting TRL GRPO training for Qwen on GSM8K")
    
    # Select model
    if args.model:
        model_name = args.model
        logger.info(f"✅ Using specified model: {model_name}")
    else:
        model_name = select_model()
    
    # Prepare dataset
    train_dataset, test_dataset = prepare_gsm8k_dataset(
        subset_size=args.subset_size, 
        test_size=args.test_size
    )
    
    # Initialize reward function
    reward_fn = GSM8KRewardFunction()
    
    # Get training configuration
    training_args = get_grpo_config(
        output_dir=args.output_dir
    )
    
    # Update vLLM server settings if using vLLM
    if not args.no_vllm:
        training_args.vllm_server_host = args.vllm_host
        training_args.vllm_server_port = args.vllm_port
        logger.info(f"🔗 vLLM server configured at {args.vllm_host}:{args.vllm_port}")
    else:
        logger.info("🚫 vLLM integration disabled")
    
    # Initialize trainer
    logger.info("🔧 Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    
    # Start training
    logger.info("🚀 Starting training...")
    trainer.train()
    
    # Save final model
    logger.info("💾 Saving final model...")
    trainer.save_model()
    
    # Final evaluation
    if test_dataset:
        logger.info("📊 Running final evaluation...")
        eval_results = trainer.evaluate()
        logger.info(f"Final evaluation results: {eval_results}")
    
    logger.info("✅ Training completed!")

if __name__ == "__main__":
    # Set environment variables for better performance
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    main() 