#!/usr/bin/env python3
"""
Generate and evaluate responses for pre-formatted GSM8K prompts using vLLM-hosted HuggingFace model.

This script:
1. Loads pre-formatted GSM8K prompts from JSONL files
2. Generates responses using a vLLM-hosted model
3. Evaluates answer correctness using GSM8K evaluation metrics
4. Saves results with detailed evaluation metrics
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional
import re

from tqdm import tqdm

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "verl"))

from llm.client import LLMClient
from verl.utils.reward_score.gsm8k import extract_solution, compute_score


def extract_answer_from_response(response: str) -> str:
    """Extract the final answer from model response in 'Therefore, the best answer is: (X)' format."""
    # Look for the specific format first
    pattern = r"Therefore,\s*the\s*best\s*answer\s*is:\s*\(?(\d+)\)?"
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Fallback to the GSM8K standard extraction methods
    extracted = extract_solution(response, method="flexible")
    if extracted is not None:
        return str(extracted).replace(",", "").replace("$", "")
    
    extracted = extract_solution(response, method="strict")
    if extracted is not None:
        return str(extracted).replace(",", "").replace("$", "")
    
    # Last resort: look for any number at the end
    numbers = re.findall(r'\b\d+\b', response)
    if numbers:
        return numbers[-1]
    
    return None


def extract_ground_truth_from_prompt_data(entry: Dict) -> str:
    """Extract ground truth from the prompt data structure."""
    # Try multiple possible locations for ground truth
    if "reward_model" in entry and "ground_truth" in entry["reward_model"]:
        return str(entry["reward_model"]["ground_truth"])
    
    if "extra_info" in entry and "answer" in entry["extra_info"]:
        return str(entry["extra_info"]["answer"])
    
    if "ground_truth" in entry:
        return str(entry["ground_truth"])
    
    # If we can't find it, return None
    return None


def extract_prompt_text(entry: Dict) -> str:
    """Extract the prompt text from the entry."""
    if "prompt" in entry and isinstance(entry["prompt"], list):
        # Extract content from the prompt list
        for item in entry["prompt"]:
            if isinstance(item, dict) and "content" in item:
                return item["content"]
    
    if "prompt" in entry and isinstance(entry["prompt"], str):
        return entry["prompt"]
    
    return ""


class GSM8KPromptsResponseGenerator:
    """Generates and evaluates responses for pre-formatted GSM8K prompts."""
    
    def __init__(
        self, 
        client: LLMClient,
        output_dir: str,
        max_workers: int = 4,
        temperature: float = 0.0,
        max_tokens: int = 1024
    ):
        """
        Initialize GSM8K prompts response generator.
        
        Args:
            client: LLM client for generating responses
            output_dir: Directory to save response files
            max_workers: Number of parallel workers for generation
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens to generate per response
        """
        self.client = client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def load_prompts_from_jsonl(self, file_path: Path) -> List[Dict]:
        """Load prompts from JSONL file."""
        print(f"Loading prompts from {file_path}...")
        
        prompts = []
        with open(file_path, 'r') as f:
            for idx, line in enumerate(f):
                if line.strip():
                    try:
                        entry = json.loads(line.strip())
                        
                        # Extract relevant information
                        prompt_text = extract_prompt_text(entry)
                        ground_truth = extract_ground_truth_from_prompt_data(entry)
                        
                        if prompt_text and ground_truth is not None:
                            prompts.append({
                                "question_id": entry.get("extra_info", {}).get("index", idx),
                                "prompt_text": prompt_text,
                                "ground_truth": str(ground_truth),
                                "original_entry": entry
                            })
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Skipping line {idx+1}: JSON decode error: {e}")
                        continue
        
        print(f"Loaded {len(prompts)} prompts from {file_path}")
        return prompts
    
    def generate_single_response(self, entry: Dict) -> Dict:
        """Generate response for a single prompt."""
        try:
            question_id = entry["question_id"]
            prompt_text = entry["prompt_text"]
            
            # Generate response
            response = self.client.prompt(
                user_message=prompt_text,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract predicted answer
            predicted_answer = extract_answer_from_response(response)
            
            # Evaluate correctness
            ground_truth = entry["ground_truth"]
            is_correct = False
            score = 0.0
            
            if predicted_answer is not None:
                # Try exact match first
                if predicted_answer == ground_truth:
                    is_correct = True
                    score = 1.0
                else:
                    # Try numerical comparison for cases like "42.0" vs "42"
                    try:
                        if float(predicted_answer) == float(ground_truth):
                            is_correct = True
                            score = 1.0
                    except (ValueError, TypeError):
                        pass
            
            # Use the official GSM8K scoring function as well
            official_score = compute_score(response, ground_truth, method="flexible")
            
            return {
                "question_id": question_id,
                "prompt_text": prompt_text,
                "response": response,
                "ground_truth": ground_truth,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "score": score,
                "official_score": official_score,
                "original_entry": entry["original_entry"],
                "status": "success"
            }
            
        except Exception as e:
            return {
                "question_id": entry.get("question_id", -1),
                "error": str(e),
                "status": "error"
            }
    
    def generate_responses(
        self, 
        data: List[Dict], 
        output_file: str,
        resume: bool = True
    ) -> Path:
        """
        Generate responses for GSM8K prompts.
        
        Args:
            data: List of prompt entries
            output_file: Name of output file
            resume: Whether to resume from existing partial results
            
        Returns:
            Path to output file with responses
        """
        output_path = self.output_dir / output_file
        
        print(f"Generating responses for GSM8K prompts...")
        print(f"Output: {output_path}")
        
        # Check existing responses if resuming
        existing_ids = set()
        if resume and output_path.exists():
            with open(output_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            resp = json.loads(line.strip())
                            if resp.get("status") == "success":
                                existing_ids.add(resp.get("question_id", -1))
                        except json.JSONDecodeError:
                            continue
            print(f"Found {len(existing_ids)} existing responses. Resuming...")
        
        # Prepare entries to process
        to_process = []
        for entry in data:
            if entry["question_id"] not in existing_ids:
                to_process.append(entry)
        
        print(f"Total entries: {len(data)}")
        print(f"Already processed: {len(existing_ids)}")
        print(f"To process: {len(to_process)}")
        
        if not to_process:
            print("No new entries to process.")
            return output_path
        
        # Test connection first
        print(self.client.test_connection())
        
        # Generate responses in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_id = {
                executor.submit(self.generate_single_response, entry): entry["question_id"]
                for entry in to_process
            }
            
            with open(output_path, "a") as f:
                completed = 0
                correct = 0
                
                for future in tqdm(
                    as_completed(future_to_id),
                    total=len(future_to_id),
                    desc="Generating responses",
                    unit="resp",
                ):
                    question_id = future_to_id[future]
                    
                    try:
                        result = future.result()
                        completed += 1
                        
                        if result.get("status") == "success" and result.get("is_correct"):
                            correct += 1
                        
                        # Write result immediately
                        f.write(json.dumps(result) + "\n")
                        f.flush()
                        
                        accuracy = correct / completed if completed > 0 else 0
                        tqdm.write(f"✅ Question {question_id+1} completed. Running accuracy: {accuracy:.3f}")
                        
                    except Exception as e:
                        tqdm.write(f"❌ Question {question_id+1} failed: {e}")
                        
                        # Write error result
                        error_result = {
                            "question_id": question_id,
                            "error": str(e),
                            "status": "error"
                        }
                        f.write(json.dumps(error_result) + "\n")
                        f.flush()
        
        print(f"✅ Response generation completed. Output: {output_path}")
        return output_path
    
    def evaluate_results(self, results_file: Path) -> Dict:
        """Evaluate the generated results and compute metrics."""
        print(f"\n📊 Evaluating results from {results_file}")
        
        total_questions = 0
        successful_responses = 0
        correct_answers = 0
        failed_responses = 0
        
        correct_examples = []
        incorrect_examples = []
        error_examples = []
        
        with open(results_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    result = json.loads(line.strip())
                    total_questions += 1
                    
                    if result.get("status") == "success":
                        successful_responses += 1
                        if result.get("is_correct"):
                            correct_answers += 1
                            correct_examples.append(result)
                        else:
                            incorrect_examples.append(result)
                    else:
                        failed_responses += 1
                        error_examples.append(result)
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON decode error on line {line_num}: {e}")
                    failed_responses += 1
        
        # Calculate metrics
        success_rate = successful_responses / total_questions if total_questions > 0 else 0
        accuracy = correct_answers / successful_responses if successful_responses > 0 else 0
        overall_accuracy = correct_answers / total_questions if total_questions > 0 else 0
        
        metrics = {
            "total_questions": total_questions,
            "successful_responses": successful_responses,
            "failed_responses": failed_responses,
            "correct_answers": correct_answers,
            "success_rate": success_rate,
            "accuracy": accuracy,
            "overall_accuracy": overall_accuracy
        }
        
        # Print results
        print("\n" + "="*50)
        print("📊 GSM8K PROMPTS EVALUATION RESULTS")
        print("="*50)
        print(f"Total questions:      {total_questions}")
        print(f"Successful responses: {successful_responses}")
        print(f"Failed responses:     {failed_responses}")
        print(f"Correct answers:      {correct_answers}")
        print()
        print(f"Success rate:         {success_rate:.3f} ({successful_responses}/{total_questions})")
        print(f"Accuracy (on valid):  {accuracy:.3f} ({correct_answers}/{successful_responses})")
        print(f"Overall accuracy:     {overall_accuracy:.3f} ({correct_answers}/{total_questions})")
        
        # Save detailed evaluation
        eval_file = results_file.parent / f"{results_file.stem}_evaluation.json"
        with open(eval_file, 'w') as f:
            json.dump({
                "metrics": metrics,
                "sample_correct": correct_examples[:5],  # First 5 correct examples
                "sample_incorrect": incorrect_examples[:5],  # First 5 incorrect examples
                "sample_errors": error_examples[:5]  # First 5 error examples
            }, f, indent=2)
        
        print(f"\n💾 Detailed evaluation saved to: {eval_file}")
        
        return metrics


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate and evaluate responses for GSM8K prompts using vLLM")
    parser.add_argument("--prompts-file", type=str, required=True, help="Path to JSONL file with prompts")
    parser.add_argument("--port", type=int, required=True, help="vLLM server port")
    parser.add_argument("--model-id", type=str, help="Specific model ID to use")
    parser.add_argument("--max-workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens to generate")
    parser.add_argument("--output-dir", type=str, default="data/gsm8k_prompts_responses", help="Output directory")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing results")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start fresh (don't resume)")
    parser.add_argument("--evaluate-only", type=str, help="Only evaluate existing results file")
    
    args = parser.parse_args()
    
    print("🚀 GSM8K Prompts Response Generation and Evaluation")
    print("=" * 50)
    
    # If only evaluating existing results
    if args.evaluate_only:
        results_file = Path(args.evaluate_only)
        if not results_file.exists():
            print(f"❌ Results file not found: {results_file}")
            sys.exit(1)
        
        generator = GSM8KPromptsResponseGenerator(None, args.output_dir, args.max_workers)
        generator.evaluate_results(results_file)
        return
    
    # Check prompts file exists
    prompts_file = Path(args.prompts_file)
    if not prompts_file.exists():
        print(f"❌ Prompts file not found: {prompts_file}")
        sys.exit(1)
    
    # Create LLM client
    try:
        client = LLMClient.local(port=args.port, model_id=args.model_id)
        print(f"📡 Connected to: {client}")
        print(client.test_connection())
    except Exception as e:
        print(f"❌ Failed to connect to vLLM server: {e}")
        print(f"Make sure the vLLM server is running on port {args.port}")
        print("Example: python -m vllm.entrypoints.openai.api_server --model YOUR_MODEL --port 6002")
        sys.exit(1)
    
    # Create response generator
    generator = GSM8KPromptsResponseGenerator(
        client=client,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Load prompts
    try:
        data = generator.load_prompts_from_jsonl(prompts_file)
        if not data:
            print("❌ No valid prompts found in the file")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to load prompts: {e}")
        sys.exit(1)
    
    # Generate responses
    output_file = f"{prompts_file.stem}_responses.jsonl"
    results_path = generator.generate_responses(data, output_file, resume=args.resume)
    
    # Evaluate results
    metrics = generator.evaluate_results(results_path)
    
    print("\n🎉 GSM8K prompts evaluation completed!")
    print(f"📁 Results saved to: {results_path}")
    print(f"🎯 Final accuracy: {metrics['overall_accuracy']:.3f}")


if __name__ == "__main__":
    main() 