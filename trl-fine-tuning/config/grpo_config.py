"""
GRPO Training Configuration
"""

from trl import GRPOConfig
import os

def get_grpo_config(output_dir: str = "./outputs/qwen_grpo_gsm8k"):
    """Get GRPO training configuration."""
    
    #     output_dir=OUTPUT_DIR,
    # run_name=RUN_NAME,
    # learning_rate=5e-6,
    # adam_beta1=0.9,
    # adam_beta2=0.99,
    # weight_decay=0.1,
    # warmup_ratio=0.1,
    # lr_scheduler_type='cosine',
    # logging_steps=1,
    # bf16=True,
    # per_device_train_batch_size=2,  # Increased from 1
    # gradient_accumulation_steps=2,  # Reduced from 4
    # num_generations=8,  # Reduced from 16
    # max_prompt_length=256,
    # max_completion_length=786,
    # num_train_epochs=1,
    # save_steps=100,
    # max_grad_norm=0.1,
    # report_to="wandb",
    # log_on_each_node=False,
    
    return GRPOConfig(
        output_dir=output_dir,
        
        # Training schedule
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,  # Reduced for memory efficiency
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=0.1,
        
        # Logging and saving
        logging_steps=5,
        save_steps=100,
        log_completions=True,
        num_completions_to_print=3,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=100,
        
        # Optimization
        bf16=True,
        
        # Experiment tracking - WANDB ONLY
        run_name="qwen_grpo_gsm8k",
        report_to="wandb",
        
        # Generation parameters for GRPOConfig
        max_completion_length=786,  # Increased for complete math solutions
        num_generations=8,  # Reduced from default to speed up training
        
        # vLLM integration - TRL handles everything automatically
        use_vllm=True,
    )

def get_model_candidates():
    """Get list of candidate models to try."""
    return [
        "Qwen/Qwen3-1.7B",
    ] 