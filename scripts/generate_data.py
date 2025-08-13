#!/usr/bin/env python3
"""
Generate datasets for CoT cue articulation experiments.

This script creates biased and unbiased prompt datasets for testing
how language models respond to different types of cues.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_generation.generator import DataGenerator
from enums.cue import Cue


def main():
    """Generate and validate all datasets."""
    print("🚀 CoT Cue Articulation - Data Generation")
    print("=" * 50)
    dataset = input("What dataset do you want to generate? (mmlu/gsm8k/gsm8k-correctness) [default: mmlu]: ").strip() or "mmlu"
    rl = bool(input("Generate for RL setting? (y/n) [default: y]: ").strip().lower() in ["y", "yes", ""])
    if dataset == "gsm8k":
        split = input("What split do you want to generate? (train/test) [default: train]: ").strip() or "train"
        generator = DataGenerator(dataset_name=dataset, split=split)
    # Initialize data generator
    else:
        generator = DataGenerator(dataset_name=dataset, rl=rl)
    
    # Generate all datasets
    generated_files = generator.generate_all_datasets()
    
    # Validate all datasets
    validation_results = generator.validate_all_datasets()
    
    # Final summary
    print("=" * 50)
    print("📊 GENERATION SUMMARY:")
    
    for cue in Cue:
        file_path = generated_files.get(cue, "Not generated")
        validation = "✅ VALID" if validation_results.get(cue, False) else "❌ INVALID"
        print(f"{cue.display_name}: {file_path} - {validation}")
    
    all_valid = all(validation_results.values())
    if all_valid:
        print("\n🎉 All datasets generated and validated successfully!")
        print("Ready for response generation.")
    else:
        print("\n⚠️  Some datasets have validation issues.")
        print("Please check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main() 