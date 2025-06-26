#!/usr/bin/env python3
"""
Generate and evaluate GSM8K responses using vLLM-hosted HuggingFace model.

This script:
1. Loads the GSM8K dataset from HuggingFace
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

import datasets
from tqdm import tqdm

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "verl"))

from llm.client import LLMClient
from verl.utils.reward_score.gsm8k import extract_solution, compute_score


def extract_ground_truth_answer(answer_str: str) -> str:
    """Extract numeric answer from GSM8K answer string that ends with #### N."""
    solution = re.search(r"#### ([\-0-9\.\,]+)", answer_str)
    if solution is not None:
        return solution.group(1).replace(",", "").replace("$", "")
    # Fallback: try to find the last number in the string
    numbers = re.findall(r"[\-0-9\.\,]+", answer_str)
    if numbers:
        return numbers[-1].replace(",", "").replace("$", "")
    return "0"  # Default fallback


def format_gsm8k_prompt(question: str, include_thinking: bool = True) -> str:
    """Format GSM8K question with appropriate instruction."""
    if include_thinking:
        instruction = 'Let\'s think step by step and output the final answer after "####".'
    else:
        instruction = 'Please solve this step by step and provide your final numerical answer after "####".'
    
    return f"{question.strip()} {instruction}"


class GSM8KResponseGenerator:
    """Generates and evaluates responses for GSM8K dataset."""
    
    def __init__(
        self, 
        client: LLMClient,
        output_dir: str,
        max_workers: int = 4,
        temperature: float = 0.0,
        max_tokens: int = 1024
    ):
        """
        Initialize GSM8K response generator.
        
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
    
    def load_gsm8k_dataset(self, split: str = "test", max_examples: int = None) -> List[Dict]:
        """Load GSM8K dataset from HuggingFace."""
        print(f"Loading GSM8K {split} dataset from HuggingFace...")
        dataset = datasets.load_dataset("openai/gsm8k", "main")[split]
        
        processed_data = []
        for idx, example in enumerate(dataset):
            if max_examples is not None and idx >= max_examples:
                break
                
            question = example["question"]
            answer = example["answer"]
            ground_truth = extract_ground_truth_answer(answer)
            
            processed_data.append({
                "question_id": idx,
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "formatted_question": format_gsm8k_prompt(question)
            })
        
        print(f"Loaded {len(processed_data)} examples from GSM8K {split} set")
        return processed_data
    
    def generate_single_response(self, entry: Dict) -> Dict:
        """Generate response for a single GSM8K question."""
        try:
            question_id = entry["question_id"]
            formatted_question = entry["formatted_question"]
            
            # Generate response
            response = self.client.prompt(
                user_message=formatted_question,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract predicted answer
            predicted_answer = extract_solution(response, method="flexible")
            if predicted_answer is None:
                predicted_answer = extract_solution(response, method="strict")
            
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
                "question": entry["question"],
                "formatted_question": formatted_question,
                "response": response,
                "ground_truth": ground_truth,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "score": score,
                "official_score": official_score,
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
        Generate responses for GSM8K dataset.
        
        Args:
            data: List of GSM8K examples
            output_file: Name of output file
            resume: Whether to resume from existing partial results
            
        Returns:
            Path to output file with responses
        """
        output_path = self.output_dir / output_file
        
        print(f"Generating GSM8K responses...")
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
                    desc="Generating GSM8K responses",
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
        print("📊 GSM8K EVALUATION RESULTS")
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
    parser = argparse.ArgumentParser(description="Generate and evaluate GSM8K responses using vLLM")
    parser.add_argument("--port", type=int, required=True, help="vLLM server port")
    parser.add_argument("--model-id", type=str, help="Specific model ID to use")
    parser.add_argument("--max-workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--output-dir", type=str, default="data/gsm8k_responses", help="Output directory")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to use")
    parser.add_argument("--max-examples", type=int, default=200, help="Maximum number of examples to process")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing results")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start fresh (don't resume)")
    parser.add_argument("--evaluate-only", type=str, help="Only evaluate existing results file")
    
    args = parser.parse_args()
    
    print("🚀 GSM8K Response Generation and Evaluation")
    print("=" * 50)
    
    # If only evaluating existing results
    if args.evaluate_only:
        results_file = Path(args.evaluate_only)
        if not results_file.exists():
            print(f"❌ Results file not found: {results_file}")
            sys.exit(1)
        
        generator = GSM8KResponseGenerator(None, args.output_dir, args.max_workers)
        generator.evaluate_results(results_file)
        return
    
    # Create LLM client
    try:
        client = LLMClient.local(port=args.port, model_id=args.model_id)
        print(f"📡 Connected to: {client}")
        print(client.test_connection())
    except Exception as e:
        print(f"❌ Failed to connect to vLLM server: {e}")
        print(f"Make sure the vLLM server is running on port {args.port}")
        print("Example: python -m vllm.entrypoints.openai.api_server --model YOUR_MODEL --port 8000")
        sys.exit(1)
    
    # Create response generator
    generator = GSM8KResponseGenerator(
        client=client,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Load GSM8K dataset
    try:
        data = generator.load_gsm8k_dataset(split=args.split, max_examples=args.max_examples)
    except Exception as e:
        print(f"❌ Failed to load GSM8K dataset: {e}")
        sys.exit(1)
    
    # Generate responses
    output_file = f"gsm8k_{args.split}_responses.jsonl"
    results_path = generator.generate_responses(data, output_file, resume=args.resume)
    
    # Evaluate results
    metrics = generator.evaluate_results(results_path)
    
    print("\n🎉 GSM8K evaluation completed!")
    print(f"📁 Results saved to: {results_path}")
    print(f"🎯 Final accuracy: {metrics['overall_accuracy']:.3f}")


if __name__ == "__main__":
    main() 