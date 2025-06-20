"""
GSM8K Dataset Loader and Preparation
"""

from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

def prepare_gsm8k_dataset(subset_size: int = 1000, test_size: int = 200):
    """Load and prepare GSM8K dataset for GRPO training."""
    logger.info("Loading GSM8K dataset...")
    
    # Load GSM8K from HuggingFace
    dataset = load_dataset("gsm8k", "main")
    
    def format_example(example):
        """Format GSM8K example for GRPO training."""
        # Extract ground truth answer
        answer_text = example['answer']
        ground_truth = float(answer_text.split('####')[-1].strip().replace(',', ''))
        
        return {
            "prompt": f"Question: {example['question']}\n\nSolve this step by step and provide your final answer in the format '#### [number]'.",
            "ground_truth": ground_truth,
        }
    
    # Prepare training set (use subset for faster training)
    train_dataset = dataset["train"].select(range(min(subset_size, len(dataset["train"]))))
    train_dataset = train_dataset.map(format_example)
    
    # Prepare test set for evaluation
    test_dataset = dataset["test"].select(range(min(test_size, len(dataset["test"]))))
    test_dataset = test_dataset.map(format_example)
    
    logger.info(f"Prepared {len(train_dataset)} training examples")
    logger.info(f"Prepared {len(test_dataset)} test examples")
    
    return train_dataset, test_dataset