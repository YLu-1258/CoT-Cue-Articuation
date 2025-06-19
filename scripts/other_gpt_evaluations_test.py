#!/usr/bin/env python3

import json
import sys
from pathlib import Path
import argparse
from typing import List
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from llm.client import LLMClient
from enums.cue import Cue
from evaluation.evaluator import ModelEvaluator  # You may need to write this

def extract_question_ids(jsonl_path: Path) -> List[int]:
    """Extract question_ids from a JSONL file."""
    question_ids = []
    with open(jsonl_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if "question_id" in data:
                    question_ids.append(data["question_id"])
            except json.JSONDecodeError:
                continue
    return question_ids

def main():
    parser = argparse.ArgumentParser(description="Prompt multiple models on selected question_ids")
    parser.add_argument("--cue", type=str, required=True, choices=[cue.value for cue in Cue], help="Cue to apply")
    parser.add_argument("--port", type=int, default=5000, help="LLM server port (same for all models)")
    parser.add_argument("--max-workers", type=int, default=8, help="Number of parallel workers")

    args = parser.parse_args()
    cue = Cue(args.cue)
    input_file = Path("data/model_evaluation/gpt-4o") / f"{cue.value}_evaluations.jsonl"

    # Extract question_ids
    question_ids = extract_question_ids(input_file)
    print(f"Found {len(question_ids)} question_ids in {input_file}")

    model_ids = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano"]
    max_workers = 16
    

    for model_id in model_ids:
        print(f"\n🚀 Prompting model: {model_id}")
        client = LLMClient.local(port=args.port, model_id=model_id)
        evaluator = ModelEvaluator(client, f"data/model_evaluation/{model_id}", max_workers)
        responses_file = Path("data/responses/filtered") / f"{cue.value}_responses_filtered.jsonl"
        proceed = input("Proceed with prompting? (Y/N)")
        if proceed.lower() == "y":
            evaluator.evaluate_question_ids(
                responses_file=responses_file,
                cue=cue,
                question_ids=question_ids
            )

if __name__ == "__main__":
    main()
