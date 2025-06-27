#!/usr/bin/env python3
"""
Evaluate model responses for CoT cue articulation experiments.

This script evaluates model responses to determine:
1. Whether the model switched its answer due to the cue
2. Whether the model explicitly acknowledged the cue
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.client import LLMClient
from evaluation.evaluator import ModelEvaluator
from enums.cue import Cue


def main():
    """Evaluate responses for all datasets."""
    parser = argparse.ArgumentParser(description="Evaluate model responses")
    parser.add_argument("--port", type=int, help="Evaluator LLM server port")
    parser.add_argument("--model-id", type=str, help="Specific evaluator model ID to use")
    parser.add_argument("--max-workers", type=int, help="Number of parallel workers")
    parser.add_argument("--cue", type=str, choices=[cue.value for cue in Cue], 
                       help="Evaluate responses for specific cue only")
    parser.add_argument("--response-file", type=str, 
                       help="Custom response file to evaluate")
    parser.add_argument("--out", type=str, 
                        help="output directory for evaluation results")
    
    args = parser.parse_args()
    
    print("🚀 CoT Cue Articulation - Response Evaluation")
    print("=" * 50)
    
    # Create evaluator LLM client
    try:
        client = LLMClient.local(port=args.port, model_id=args.model_id)
        print(f"📡 Connected to evaluator: {client}")
        print(client.test_connection())
    except Exception as e:
        print(f"❌ Failed to connect to evaluator LLM server: {e}")
        print(f"Make sure the server is running on port {args.port}")
        sys.exit(1)
    
    # Create model evaluator with folder named after the actual model being used
    max_workers = args.max_workers if args.max_workers else 8  # Default to 8 if not specified
    model_name = client.model_id.replace('/', '_').replace(':', '_')  # Clean model name for folder
    evaluator = ModelEvaluator(client, f"data/model_evaluation/{model_name}", max_workers)
    if args.out:
        evaluator = ModelEvaluator(client, args.out, max_workers)
    
    # Evaluate responses
    if args.cue:
        # Evaluate for specific cue
        cue = Cue(args.cue)
        if args.response_file:
            responses_file = Path(args.response_file)
        else:
            responses_file = Path("data/responses/filtered") / f"{cue.value}_responses_filtered.jsonl"
        
        if not responses_file.exists():
            print(f"❌ Responses file not found: {responses_file}")
            print("Run 'python scripts/generate_responses.py' first")
            sys.exit(1)
        
        evaluation_file = evaluator.evaluate_responses(responses_file, cue)
        print(f"\n✅ Evaluation completed: {evaluation_file}")
    else:
        # Evaluate for all cues
        results = evaluator.evaluate_all_responses("data/responses/filtered")
        
        print("=" * 50)
        print("📊 EVALUATION SUMMARY:")
        
        for cue, evaluation_file in results.items():
            print(f"{cue.display_name}: {evaluation_file}")
        
        if results:
            print("\n🎉 All evaluations completed successfully!")
            print("Ready for analysis.")
        else:
            print("\n⚠️  No response files found for evaluation.")
            print("Run 'python scripts/generate_responses.py' first")


if __name__ == "__main__":
    main() 