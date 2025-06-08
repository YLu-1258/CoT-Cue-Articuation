#!/usr/bin/env python3
"""
Filter responses where unbiased answers match correct answers.

This script uses the CSV files with extracted answers to filter the JSONL response
files, keeping only responses where the unbiased answer matches the correct answer.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Set

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enums.cue import Cue


def load_correct_question_ids(csv_file: Path) -> Set[int]:
    """Load question IDs where biased answers are correct."""
    correct_ids = set()
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Check if biased answers match their correct answers
            if row['biased_match'] == 'True':
                question_id = int(row['question_id'])
                correct_ids.add(question_id)
    
    return correct_ids


def filter_responses(responses_file: Path, correct_ids: Set[int], output_file: Path) -> Dict[str, int]:
    """Filter responses to keep only those with correct unbiased answers."""
    stats = {"total": 0, "correct_unbiased": 0, "written": 0}
    
    with open(responses_file, 'r') as input_f, open(output_file, 'w') as output_f:
        for line in input_f:
            if line.strip():
                response = json.loads(line.strip())
                stats["total"] += 1
                
                question_id = response.get("question_id", -1)
                
                # Only keep responses where both answers were correct and response was successful
                if (question_id in correct_ids and 
                    response.get("status") == "success"):
                    stats["correct_unbiased"] += 1
                    output_f.write(line)
                    stats["written"] += 1
    
    return stats


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Filter responses where both unbiased and biased answers are correct")
    parser.add_argument("--cue", type=str, choices=[cue.value for cue in Cue],
                       help="Filter specific cue only")
    parser.add_argument("--responses-dir", type=str, default="data/responses",
                       help="Directory containing response files")
    parser.add_argument("--output-dir", type=str, default="data/responses/filtered",
                       help="Output directory for filtered responses")
    
    args = parser.parse_args()
    
    responses_dir = Path(args.responses_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔍 Filtering responses where both unbiased AND biased answers are correct...")
    print("=" * 70)
    
    if args.cue:
        # Filter specific cue
        cues_to_process = [Cue(args.cue)]
    else:
        # Filter all cues
        cues_to_process = list(Cue)
    
    total_stats = {"total": 0, "correct_unbiased": 0, "written": 0}
    
    for cue in cues_to_process:
        print(f"\n📊 Processing {cue.display_name}...")
        
        # Load CSV file with extracted answers
        csv_file = responses_dir / f"extracted_answers_{cue.value}.csv"
        responses_file = responses_dir / f"{cue.value}_responses.jsonl"
        output_file = output_dir / f"{cue.value}_responses_filtered.jsonl"
        
        if not csv_file.exists():
            print(f"❌ CSV file not found: {csv_file}")
            continue
        
        if not responses_file.exists():
            print(f"❌ Response file not found: {responses_file}")
            continue
        
        # Load correct question IDs
        correct_ids = load_correct_question_ids(csv_file)
        print(f"   Found {len(correct_ids)} questions with both answers correct")
        
        # Filter responses
        stats = filter_responses(responses_file, correct_ids, output_file)
        
        # Update totals
        for key in total_stats:
            total_stats[key] += stats[key]
        
        # Report stats
        both_correct_rate = (stats["correct_unbiased"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"   Total responses: {stats['total']}")
        print(f"   Both answers correct: {stats['correct_unbiased']} ({both_correct_rate:.1f}%)")
        print(f"   Written to filtered file: {stats['written']}")
        print(f"   📤 Output: {output_file}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("🎯 **Final Summary**")
    both_correct_rate = (total_stats["correct_unbiased"] / total_stats["total"] * 100) if total_stats["total"] > 0 else 0
    print(f"   Total responses processed: {total_stats['total']}")
    print(f"   Total with both answers correct: {total_stats['correct_unbiased']} ({both_correct_rate:.1f}%)")
    print(f"   Total written to filtered files: {total_stats['written']}")
    print(f"\n💡 Filtered files are in: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 