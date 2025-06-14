"""Model evaluation for cue articulation and response switching."""

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.client import LLMClient
from enums.cue import Cue
from evaluation.prompts import (
    EVALUATION_PROMPT_TEMPLATE,
    CUE_DESCRIPTIONS,
    CUE_SPECIFIC_CASES,
    EVALUATION_SYSTEM_PROMPT
)
import time


class ModelEvaluator:
    """Evaluates model responses for cue articulation"""
    
    def __init__(
        self,
        evaluator_client: LLMClient,
        output_dir: str,
        max_workers: int
    ):
        """
        Initialize model evaluator.
        
        Args:
            evaluator_client: LLM client to use for evaluation
            output_dir: Directory to save evaluation results
            max_workers: Number of parallel workers
        """
        self.client = evaluator_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
    
    def evaluate_responses(
        self,
        responses_file: Path,
        cue: Cue,
        resume: bool = True
    ) -> Path:
        """
        Evaluate model responses for cue articulation.
        
        Args:
            responses_file: Path to responses JSONL file
            cue: Cue type for naming output file
            resume: Whether to resume from existing evaluations
            
        Returns:
            Path to evaluation results file
        """
        output_file = self.output_dir / f"{cue.value}_evaluations.jsonl"
        
        print(f"Evaluating responses for {cue.display_name}")
        print(f"Input: {responses_file}")
        print(f"Output: {output_file}")
        
        # Load response data
        responses = self._load_jsonl(responses_file)
        total = len(responses)
        
        # Check existing evaluations if resuming
        existing_ids = set()
        if resume and output_file.exists():
            existing_evals = self._load_jsonl(output_file)
            existing_ids = {eval_result.get("question_id", -1) for eval_result in existing_evals}
            print(f"Found {len(existing_ids)} existing evaluations. Resuming...")
        
        # Prepare entries to process
        to_process = []
        for response in responses:
            question_id = response.get("question_id", -1)
            if question_id not in existing_ids and response.get("status") == "success":
                to_process.append(response)
        
        print(f"Total responses: {total}")
        print(f"Already evaluated: {len(existing_ids)}")
        print(f"To process: {len(to_process)}")
        
        if not to_process:
            print("No new responses to evaluate.")
            return output_file
        
        # Test evaluator connection
        print(self.client.test_connection())
        
        # Monitor GPU usage
        self._monitor_gpu_usage()
        
        # Evaluate in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_id = {
                executor.submit(self._evaluate_single_response, response, cue): response["question_id"]
                for response in to_process
            }
            
            with open(output_file, "a") as f:
                completed = 0
                for future in as_completed(future_to_id):
                    question_id = future_to_id[future]
                    
                    try:
                        result = future.result()
                        completed += 1
                        
                        # Write result immediately
                        f.write(json.dumps(result) + "\n")
                        f.flush()
                        
                        if result.get("status") == "success":
                            print(f"✅ Q{question_id}/{total} done "
                                  f"(ack_cue={result['acknowledged_cue']})")
                        else:
                            print(f"❌ Q{question_id}/{total} failed: {result.get('error')}")
                    
                    except Exception as e:
                        print(f"❌ Q{question_id}/{total} exception: {e}")
                        
                        error_result = {
                            "question_id": question_id,
                            "cue": cue.value,
                            "error": str(e),
                            "status": "error"
                        }
                        f.write(json.dumps(error_result) + "\n")
                        f.flush()
        
        print(f"✅ Evaluation completed. Output: {output_file}")
        return output_file
    

    def _extract_cot(self, response : str):
        marker = "</think>"
        idx = response.find(marker)
        if idx != -1:
            cot = response[:idx]      # all characters before the marker
        else:
            cot = response
        return cot
    
    def _evaluate_single_response(self, response: Dict, cue: Cue) -> Dict:
        """Evaluate a single response for cue articulation."""
        try:
            # Format evaluation prompt
            prompt = EVALUATION_PROMPT_TEMPLATE.format(
                cue_description=CUE_DESCRIPTIONS[cue.value],
                cue_specific_cases=CUE_SPECIFIC_CASES[cue.value],
                biased_response=self._extract_cot(response["biased_response"])
            )
            
            # Get evaluation from model
            system_prompt = EVALUATION_SYSTEM_PROMPT
            
            retries = 3
            backoff = 45
            raw = None
            for attempt in range(1, retries + 1):
                try:
                    raw = self.client.prompt(
                        prompt,
                        system_prompt=system_prompt,
                        temperature=0.0,
                        max_tokens=1000  # Increased for detailed analysis
                    ).strip().lower()
                    break
                except Exception as e:
                    print(f"⚠️ API error {e}, retrying in {backoff:.1f}s ({attempt}/{retries})...")
                    time.sleep(backoff)

            if raw is None:
                return {
                    "question_id": response.get("question_id", -1),
                    "cue": cue.value,
                    "error": "failed after retries",
                    "status": "error"
                }

            
            # Parse true/false from the response
            if "final answer: true" in raw.lower():
                acknowledged = "yes"  # Keep internal format as yes/no
            elif "final answer: false" in raw.lower():
                acknowledged = "no"
            elif "true" in raw.lower():
                acknowledged = "yes"  # Fallback to old method
            else:
                acknowledged = "no"
            
            return {
                "question_id": response["question_id"],
                "cue": cue.value,
                "acknowledged_cue": acknowledged,
                "raw_evaluation": raw,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "question_id": response["question_id"],
                "cue": cue.value,
                "error": str(e),
                "raw_response": raw if 'raw' in locals() else None,
                "status": "error"
            }
    
    def evaluate_all_responses(self, responses_dir: str = "data/responses") -> Dict[Cue, Path]:
        """Evaluate all response files."""
        responses_path = Path(responses_dir)
        results = {}
        
        for cue in Cue:
            # Check for filtered files first, then fall back to regular files
            responses_file = responses_path / f"{cue.value}_responses_filtered.jsonl"
            if not responses_file.exists():
                responses_file = responses_path / f"{cue.value}_responses.jsonl"
                
            if responses_file.exists():
                evaluation_file = self.evaluate_responses(responses_file, cue)
                results[cue] = evaluation_file
                print()
            else:
                print(f"❌ Response file not found: {responses_file}")
        
        return results
    
    def _monitor_gpu_usage(self):
        """Monitor and print GPU usage if available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    used, total = map(int, line.split(', '))
                    percentage = (used / total) * 100
                    print(f"🔧 GPU {i}: {used}MB / {total}MB ({percentage:.1f}%)")
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            # GPU monitoring is optional, fail silently
            pass
    
    def _load_jsonl(self, filepath: Path) -> List[Dict]:
        """Load JSONL file into list of dictionaries."""
        entries = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line.strip()))
        return entries 