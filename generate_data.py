#!/usr/bin/env python3

import sys
sys.path.append('formatters')
import json
import os
from datetime import datetime
from formatter import StanfordProfessorFormatter, FewShotSquaresFormatter

def generate_datasets():
    """Generate both datasets using the formatters"""
    print("=== GENERATING DATASETS ===")
    
    print("Generating Stanford Professor dataset...")
    stanford_formatter = StanfordProfessorFormatter()
    stanford_formatter.generate_dataset()
    print(f"Generated data/stanford_professor.jsonl")
    
    print("Generating FewShot Squares dataset...")
    fewshot_formatter = FewShotSquaresFormatter()
    fewshot_formatter.generate_dataset()
    print(f"Generated data/fewshot_squares.jsonl")
    
    print("Dataset generation complete!\n")


#GENERATED DATA CHECK

def check_dataset(filename, dataset_name, max_entries=50):
    """Check a dataset file for matching biased/unbiased answers"""
    print(f"=== CHECKING {dataset_name.upper()} DATASET ===")
    
    if not os.path.exists(filename):
        print(f"❌ {filename} does not exist!")
        return False
    
    # Get file info
    mod_time = os.path.getmtime(filename)
    mod_date = datetime.fromtimestamp(mod_time)
    file_size = os.path.getsize(filename)
    
    print(f"File: {filename}")
    print(f"Last modified: {mod_date}")
    print(f"File size: {file_size} bytes")
    
    # Count total lines
    with open(filename, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"Total entries: {total_lines}")
    print(f"Checking first {min(max_entries, total_lines)} entries...\n")
    
    same_answer_count = 0
    total_checked = 0
    problematic_entries = []
    
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_entries:
                break
            
            try:
                entry = json.loads(line)
                total_checked += 1
                
                unbiased = entry['unbiased_answer']
                biased = entry['biased_answer']
                
                if unbiased == biased:
                    same_answer_count += 1
                    problematic_entries.append(i + 1)
                    print(f"❌ Entry {i+1}: SAME ANSWERS! unbiased='{unbiased}' == biased='{biased}'")
                    print(f"   Question: {entry['unbiased_question'][:80]}...")
                    print(f"   Choices: {entry['unbiased_choices']}")
                    print()
                else:
                    if i < 5:  # Show first few good entries
                        print(f"✅ Entry {i+1}: Different - unbiased='{unbiased}' != biased='{biased}'")
            
            except Exception as e:
                print(f"❌ Error parsing entry {i+1}: {e}")
    
    # Summary
    print(f"\n--- {dataset_name} RESULTS ---")
    print(f"Entries checked: {total_checked}")
    print(f"Same answers: {same_answer_count}")
    print(f"Different answers: {total_checked - same_answer_count}")
    
    if same_answer_count > 0:
        percentage = (same_answer_count / total_checked) * 100
        print(f"❌ PROBLEM: {percentage:.1f}% of entries have matching answers!")
        print(f"Problematic entry numbers: {problematic_entries}")
        return False
    else:
        print(f"✅ SUCCESS: All entries have different answers!")
        return True

def main():    
    # Step 1: Generate datasets
    generate_datasets()
    
    # Step 2: Generated check datasets
    stanford_ok = check_dataset("data/stanford_professor.jsonl", "Stanford Professor", 100)
    print()
    fewshot_ok = check_dataset("data/fewshot_squares.jsonl", "FewShot Squares", 100)
    
    # Final summary
    print("\n" + "=" * 50)
    print("FINAL RESULTS:")
    print(f"Stanford Professor Dataset: {'✅ PASS' if stanford_ok else '❌ FAIL'}")
    print(f"FewShot Squares Dataset: {'✅ PASS' if fewshot_ok else '❌ FAIL'}")
    
    if stanford_ok and fewshot_ok:
        print("\n🎉 ALL DATASETS GENERATED SUCCESSFULLY!")
        print("Both datasets have 100% different biased/unbiased answers.")
    else:
        print("\n🚨 DATASET GENERATION HAS ISSUES!")
        print("Some entries still have matching biased/unbiased answers.")
        print("This indicates there's still a bug in the formatter logic.")
    
    print(f"\nCompleted at: {datetime.now()}")

if __name__ == "__main__":
    main() 