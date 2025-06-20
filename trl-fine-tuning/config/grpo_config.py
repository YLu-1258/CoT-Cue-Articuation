"""
GRPO Training Configuration
"""

from trl import GRPOConfig
import os

def get_grpo_config(output_dir: str = "./outputs/qwen_grpo_gsm8k"):
    """Get GRPO training configuration."""
    
    return GRPOConfig(
        output_dir=output_dir,
        
        # Training schedule
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # Reduced for memory efficiency
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        
        # Logging and saving
        logging_steps=5,
        save_steps=200,
        save_total_limit=2,
        log_completions=True,
        num_completions_to_print=3,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=100,
        
        # Optimization
        bf16=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        
        # Experiment tracking - WANDB ONLY
        run_name="qwen_grpo_gsm8k",
        report_to="wandb",
        
        # GRPO specific parameters
        beta=0.0,  # KL coefficient - set to 0.0 when using vLLM to disable reference model
        # Generation parameters for GRPOConfig
        max_completion_length=1024,  # Increased for complete math solutions
        num_generations=2,  # Reduced from default to speed up training
        temperature=0.8,
        top_p=0.95,
        
        # vLLM integration - TRL handles everything automatically
        use_vllm=True,
        
        # Reward function configuration
        reward_weights=[1.0],
        scale_rewards=True,
    )

def get_model_candidates():
    """Get list of candidate models to try."""
    return [
        "Qwen/Qwen3-1.7B",
    ] 