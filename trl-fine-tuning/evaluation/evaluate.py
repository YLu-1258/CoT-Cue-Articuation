#!/usr/bin/env python3
"""
Evaluate trained GRPO model on GSM8K test set
"""

import os
import sys
import json
import torch
import logging
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.reward_function import GSM8KRewardFunction

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model_path: str, num_samples: int = 100):
    """Evaluate model on GSM8K test set."""
    logger.info(f"Loading model from: {model_path}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Setup generation pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Load GSM8K test set
    logger.info("Loading GSM8K test set...")
    dataset = load_dataset("gsm8k", "main")
    test_data = dataset["test"].select(range(num_samples))
    
    # Initialize reward function for answer extraction
    reward_fn = GSM8KRewardFunction()
    
    correct = 0
    total = 0
    results = []
    
    for i, example in enumerate(test_data):
        question = example['question']
        ground_truth_text = example['answer']
        ground_truth = float(ground_truth_text.split('####')[-1].strip().replace(',', ''))
        
        # Create prompt (consistent with training format)
        prompt = f"Question: {question}\n\nSolve this step by step and provide your final answer in the format '#### [number]'.\n\nAnswer:"
        
        # Generate response
        try:
            response = generator(prompt, max_new_tokens=512)[0]['generated_text']
            # Extract only the generated part
            generated_text = response[len(prompt):]
            
            # Extract predicted answer
            predicted = reward_fn.extract_answer(generated_text)
            
            # Check if correct
            is_correct = False
            if predicted is not None and abs(predicted - ground_truth) < 1e-6:
                is_correct = True
                correct += 1
            
            total += 1
            
            # Store result
            result = {
                "question": question,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "generated_text": generated_text,
                "is_correct": is_correct
            }
            results.append(result)
            
            if i % 10 == 0:
                accuracy = correct / total if total > 0 else 0
                logger.info(f"Progress: {i+1}/{num_samples}, Accuracy: {accuracy:.3f}")
                
        except Exception as e:
            logger.error(f"Error processing example {i}: {e}")
            total += 1
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "predicted": None,
                "generated_text": f"Error: {e}",
                "is_correct": False
            })
    
    # Final results
    accuracy = correct / total if total > 0 else 0
    logger.info(f"Final Results:")
    logger.info(f"Correct: {correct}/{total}")
    logger.info(f"Accuracy: {accuracy:.3f}")
    
    # Save detailed results
    output_file = f"evaluation_results_{num_samples}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "model_path": model_path,
            "num_samples": num_samples,
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "results": results
        }, f, indent=2)
    
    logger.info(f"Detailed results saved to: {output_file}")
    
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Evaluate GRPO trained model on GSM8K")
    parser.add_argument("--model_path", default="./outputs/qwen_grpo_gsm8k", 
                        help="Path to trained model")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of test samples to evaluate")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        return
    
    accuracy = evaluate_model(args.model_path, args.num_samples)
    print(f"\n🎯 Final Accuracy: {accuracy:.3f}")

if __name__ == "__main__":
    main() 