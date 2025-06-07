"""Response generation for datasets."""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import sys
from pathlib import Path
from tqdm import tqdm

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.client import LLMClient
from enums.cue import Cue


class ResponseGenerator:
    """Generates model responses for biased and unbiased prompts."""
    
    def __init__(
        self, 
        client: LLMClient,
        output_dir: str,
        max_workers: int
    ):
        """
        Initialize response generator.
        
        Args:
            client: LLM client for generating responses
            output_dir: Directory to save response files
            max_workers: Number of parallel workers for generation
        """
        self.client = client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
    
    def generate_responses_for_dataset(
        self, 
        input_file: Path, 
        cue: Cue,
        resume: bool = True
    ) -> Path:
        """
        Generate model responses for a dataset.
        
        Args:
            input_file: Path to input JSONL dataset
            cue: Cue type for naming the output file
            resume: Whether to resume from existing partial results
            
        Returns:
            Path to output file with responses
        """
        output_file = self.output_dir / f"{cue.value}_responses.jsonl"
        
        print(f"Generating responses for {cue.display_name}")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
        
        # Load input data
        data_entries = self._load_jsonl(input_file)
        total = len(data_entries)
        
        # Check existing responses if resuming
        existing_ids = set()
        if resume and output_file.exists():
            existing_responses = self._load_jsonl(output_file)
            existing_ids = {resp.get("question_id", -1) for resp in existing_responses}
            print(f"Found {len(existing_ids)} existing responses. Resuming...")
        
        # Prepare entries to process
        to_process = []
        for i, entry in enumerate(data_entries):
            if i not in existing_ids:
                to_process.append((i, entry))
        
        print(f"Total entries: {total}")
        print(f"Already processed: {len(existing_ids)}")
        print(f"To process: {len(to_process)}")
        
        if not to_process:
            print("No new entries to process.")
            return output_file
        
        # Test connection first
        print(self.client.test_connection())
        
        # Generate responses in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_id = {
                executor.submit(self._generate_single_response, entry, question_id): question_id
                for question_id, entry in to_process
            }
            
            with open(output_file, "a") as f:
                completed = 0
                for future in tqdm(
                    as_completed(future_to_id),
                    total=len(future_to_id),
                    desc=f"Generating [{cue.display_name}]",
                    unit="resp",
                ):
                    question_id = future_to_id[future]
                    
                    try:
                        result = future.result()
                        completed += 1
                        
                        # Write result immediately
                        f.write(json.dumps(result) + "\n")
                        f.flush()
                        
                        tqdm.write(f"✅ Question {question_id+1}/{total} completed")
                        
                    except Exception as e:
                        tqdm.write(f"❌ Question {question_id+1}/{total} failed: {e}")
                        
                        # Write error result
                        error_result = {
                            "question_id": question_id,
                            "error": str(e),
                            "status": "error"
                        }
                        f.write(json.dumps(error_result) + "\n")
                        f.flush()
        
        print(f"✅ Response generation completed. Output: {output_file}")
        return output_file
    
    def _generate_single_response(self, entry: Dict, question_id: int) -> Dict:
        """Generate responses for a single data entry."""
        try:
            # Generate unbiased response
            unbiased_response = self.client.prompt(entry["unbiased_question"])
            
            # Generate biased response  
            biased_response = self.client.prompt(entry["biased_question"])
            
            return {
                "question_id": question_id,
                "unbiased_question": entry["unbiased_question"],
                "unbiased_response": unbiased_response,
                "biased_question": entry["biased_question"], 
                "biased_response": biased_response,
                "correct_answer": entry["correct_answer"],
                "suggested_wrong_answer": entry["biased_answer"],
                "cue_type": entry["cue_type"],
                "status": "success"
            }
            
        except Exception as e:
            return {
                "question_id": question_id,
                "error": str(e),
                "status": "error"
            }
    
    def generate_all_responses(self, data_dir: str = "data") -> Dict[Cue, Path]:
        """Generate responses for all available datasets."""
        data_path = Path(data_dir)
        results = {}
        
        for cue in Cue:
            input_file = data_path / f"{cue.value}.jsonl"
            if input_file.exists():
                output_file = self.generate_responses_for_dataset(input_file, cue)
                results[cue] = output_file
                print()
            else:
                print(f"❌ Dataset not found: {input_file}")
        
        return results
    
    def _load_jsonl(self, filepath: Path) -> List[Dict]:
        """Load JSONL file into list of dictionaries."""
        entries = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line.strip()))
        return entries 