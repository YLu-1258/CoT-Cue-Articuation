#!/usr/bin/env python3
"""
Manual Verification Pipeline for Answer Extraction

This script helps manually verify that the extracted answers match the raw responses,
selecting 30 random items for verification.
"""

import json
import csv
import sys
import os
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import textwrap
import re

@dataclass
class VerificationItem:
    question_id: str
    dataset: str
    unbiased_correct: str
    biased_correct: str
    unbiased_extracted: str
    biased_extracted: str
    unbiased_response: str
    biased_response: str
    unbiased_length: int
    biased_length: int
    checked: bool = False

def extract_after_think(text):
    """Return the portion of text after the first occurrence of '</think>'."""
    marker = "</think>"
    idx = text.find(marker)
    if idx == -1:
        return text  # Return full text if no think marker
    return text[idx + len(marker):].strip()

def format_response_for_display(response: str, max_width: int = 100) -> str:
    """Format response text for better readability."""
    # Extract the post-think portion
    post_think = extract_after_think(response)
    
    # If response is very long, show thinking part separately
    marker = "</think>"
    if marker in response:
        think_part = response[:response.find(marker) + len(marker)]
        think_lines = len(think_part.split('\n'))
        
        formatted = f"[THINKING PART: {think_lines} lines]\n"
        formatted += "=" * 50 + "\n"
        formatted += "POST-THINK RESPONSE:\n"
        formatted += "=" * 50 + "\n"
        
        # Wrap the post-think text
        wrapped_lines = []
        for line in post_think.split('\n'):
            if len(line) <= max_width:
                wrapped_lines.append(line)
            else:
                wrapped_lines.extend(textwrap.wrap(line, width=max_width))
        
        formatted += '\n'.join(wrapped_lines)
    else:
        # No think marker, just wrap the whole response
        wrapped_lines = []
        for line in response.split('\n'):
            if len(line) <= max_width:
                wrapped_lines.append(line)
            else:
                wrapped_lines.extend(textwrap.wrap(line, width=max_width))
        formatted = '\n'.join(wrapped_lines)
    
    return formatted

def load_verification_data(dataset_name: str) -> List[VerificationItem]:
    """Load and combine data from CSV and JSONL files."""
    csv_file = f'data/responses/extracted_answers_{dataset_name}.csv'
    jsonl_file = f'data/responses/raw/{dataset_name}_responses.jsonl'
    
    # Load CSV data
    csv_data = {}
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                csv_data[int(row['question_id'])] = row
    except FileNotFoundError:
        print(f"Error: {csv_file} not found")
        return []
    
    # Load JSONL data
    jsonl_data = {}
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    jsonl_data[data['question_id']] = data
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Error: {jsonl_file} not found")
        return []
    
    # Combine data
    verification_items = []
    for question_id, csv_row in csv_data.items():
        if question_id in jsonl_data:
            jsonl_row = jsonl_data[question_id]
            
            item = VerificationItem(
                question_id=str(question_id),
                dataset=dataset_name,
                unbiased_correct=csv_row['unbiased_correct'],
                biased_correct=csv_row['biased_correct'],
                unbiased_extracted=csv_row['unbiased_extracted'],
                biased_extracted=csv_row['biased_extracted'],
                unbiased_response=jsonl_row.get('unbiased_response', ''),
                biased_response=jsonl_row.get('biased_response', ''),
                unbiased_length=int(csv_row['unbiased_response_length']),
                biased_length=int(csv_row['biased_response_length'])
            )
            verification_items.append(item)
    
    return verification_items

def save_progress(items: List[VerificationItem], filename: str):
    """Save verification progress to file."""
    progress_data = []
    for item in items:
        progress_data.append({
            'question_id': item.question_id,
            'dataset': item.dataset,
            'checked': item.checked
        })
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, indent=2)

def load_progress(items: List[VerificationItem], filename: str):
    """Load verification progress from file."""
    if not os.path.exists(filename):
        return
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            progress_data = json.load(f)
        
        progress_dict = {(item['dataset'], item['question_id']): item['checked'] 
                        for item in progress_data}
        
        for item in items:
            key = (item.dataset, item.question_id)
            if key in progress_dict:
                item.checked = progress_dict[key]
    except (json.JSONDecodeError, KeyError):
        print("Warning: Could not load progress file")

def display_verification_item(item: VerificationItem, index: int, total: int):
    """Display a single verification item."""
    print("\n" + "=" * 100)
    print(f"VERIFICATION ITEM {index + 1} of {total}")
    print(f"Dataset: {item.dataset} | Question ID: {item.question_id}")
    print(f"Response Lengths: Unbiased={item.unbiased_length}, Biased={item.biased_length}")
    print("=" * 100)
    
    print(f"\n📋 CORRECT ANSWERS:")
    print(f"   Unbiased Correct: {item.unbiased_correct}")
    print(f"   Biased Correct:   {item.biased_correct}")
    
    print(f"\n🤖 EXTRACTED ANSWERS:")
    print(f"   Unbiased Extracted: '{item.unbiased_extracted}'")
    print(f"   Biased Extracted:   '{item.biased_extracted}'")
    
    # Show potential issues
    issues = []
    if not item.unbiased_extracted:
        issues.append("❌ No unbiased answer extracted")
    if not item.biased_extracted:
        issues.append("❌ No biased answer extracted")
    if item.unbiased_extracted != item.unbiased_correct:
        issues.append(f"⚠️  Unbiased mismatch: got '{item.unbiased_extracted}', expected '{item.unbiased_correct}'")
    if item.biased_extracted != item.biased_correct and item.biased_extracted != item.unbiased_correct:
        issues.append(f"⚠️  Biased unexpected: got '{item.biased_extracted}'")
    if item.unbiased_length > 4000:
        issues.append(f"⚠️  Very long unbiased response ({item.unbiased_length} chars)")
    if item.biased_length > 4000:
        issues.append(f"⚠️  Very long biased response ({item.biased_length} chars)")
    
    if issues:
        print(f"\n🚨 POTENTIAL ISSUES:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print(f"\n✅ No obvious issues detected")
    
    print(f"\n" + "-" * 50)
    print("UNBIASED RESPONSE:")
    print("-" * 50)
    print(format_response_for_display(item.unbiased_response))
    
    print(f"\n" + "-" * 50)
    print("BIASED RESPONSE:")
    print("-" * 50)
    print(format_response_for_display(item.biased_response))

def main():
    """Main verification pipeline."""
    print("🔍 Manual Answer Extraction Verification Pipeline")
    print("=" * 60)
    
    # Choose dataset
    datasets = ['stanford_professor', 'fewshot_black_squares']
    print("\nAvailable datasets:")
    for i, dataset in enumerate(datasets):
        print(f"  {i + 1}. {dataset}")
    
    while True:
        try:
            choice = input("\nSelect dataset (1-2) or 'all' for both: ").strip().lower()
            if choice == 'all':
                selected_datasets = datasets
                break
            elif choice in ['1', '2']:
                selected_datasets = [datasets[int(choice) - 1]]
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 'all'.")
        except (ValueError, IndexError):
            print("Invalid choice. Please enter 1, 2, or 'all'.")
    
    # Load and combine data from all selected datasets
    all_items = []
    for dataset in selected_datasets:
        print(f"\nLoading data for {dataset}...")
        items = load_verification_data(dataset)
        all_items.extend(items)
        print(f"  Loaded {len(items)} items")
    
    if not all_items:
        print("No data found!")
        return
    
    # Randomly select 30 items
    if len(all_items) > 30:
        verification_items = random.sample(all_items, 30)
        verification_items.sort(key=lambda x: (x.dataset, int(x.question_id)))  # Sort for consistent display
    else:
        verification_items = all_items
    
    print(f"\n📊 Selected {len(verification_items)} random items for verification")
    if len(verification_items) == 30:
        print(f"Response length range in sample: {min(item.unbiased_length + item.biased_length for item in verification_items)} - {max(item.unbiased_length + item.biased_length for item in verification_items)} characters")
    
    # Load progress
    progress_file = 'verification_progress.json'
    load_progress(verification_items, progress_file)
    
    checked_count = sum(1 for item in verification_items if item.checked)
    print(f"Progress: {checked_count}/{len(verification_items)} items checked")
    
    # Interactive verification
    current_index = 0
    
    # Start from first unchecked item
    for i, item in enumerate(verification_items):
        if not item.checked:
            current_index = i
            break
    
    while current_index < len(verification_items):
        item = verification_items[current_index]
        
        display_verification_item(item, current_index, len(verification_items))
        
        print(f"\n" + "=" * 100)
        status = "✅ CHECKED" if item.checked else "⏳ UNCHECKED"
        print(f"Status: {status}")
        print("\nCommands:")
        print("  [Enter] - Mark as checked and continue")
        print("  'n' - Next (without marking as checked)")
        print("  'p' - Previous")
        print("  'j <num>' - Jump to item number")
        print("  's' - Show progress summary")
        print("  'q' - Quit and save progress")
        
        while True:
            command = input("\nCommand: ").strip().lower()
            
            if command == '' or command == 'enter':
                # Mark as checked and continue
                item.checked = True
                save_progress(verification_items, progress_file)
                current_index += 1
                break
            elif command == 'n':
                # Next without checking
                current_index += 1
                break
            elif command == 'p':
                # Previous
                current_index = max(0, current_index - 1)
                break
            elif command.startswith('j '):
                # Jump to specific item
                try:
                    target = int(command.split()[1]) - 1
                    if 0 <= target < len(verification_items):
                        current_index = target
                        break
                    else:
                        print(f"Invalid item number. Range: 1-{len(verification_items)}")
                except (ValueError, IndexError):
                    print("Invalid jump command. Use 'j <number>'")
            elif command == 's':
                # Show summary
                checked = sum(1 for item in verification_items if item.checked)
                print(f"\nProgress Summary: {checked}/{len(verification_items)} items checked")
                for i, item in enumerate(verification_items):
                    status = "✅" if item.checked else "⏳"
                    print(f"  {i+1:2d}. {status} {item.dataset} Q{item.question_id} (len: {item.unbiased_length + item.biased_length})")
            elif command == 'q':
                # Quit
                save_progress(verification_items, progress_file)
                print("Progress saved. Goodbye!")
                return
            else:
                print("Invalid command. Try again.")
    
    # All items processed
    save_progress(verification_items, progress_file)
    checked = sum(1 for item in verification_items if item.checked)
    print(f"\n🎉 Verification complete! {checked}/{len(verification_items)} items checked")

if __name__ == "__main__":
    main() 