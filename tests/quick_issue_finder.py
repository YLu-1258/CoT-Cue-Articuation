#!/usr/bin/env python3
"""
Quick Issue Finder for Answer Extraction

This script quickly identifies potential issues with answer extraction
to help prioritize manual verification.
"""

import csv
import json
from typing import Dict, List, Tuple
import sys

def load_csv_data(csv_file: str) -> List[Dict]:
    """Load CSV data."""
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"Error: {csv_file} not found")
        return []

def analyze_extraction_issues(dataset_name: str) -> List[Dict]:
    """Analyze potential extraction issues."""
    csv_file = f'data/responses/extracted_answers_{dataset_name}.csv'
    data = load_csv_data(csv_file)
    
    if not data:
        return []
    
    issues = []
    
    for row in data:
        question_id = row['question_id']
        unbiased_extracted = row['unbiased_extracted']
        biased_extracted = row['biased_extracted']
        unbiased_correct = row['unbiased_correct']
        biased_correct = row['biased_correct']
        unbiased_length = int(row['unbiased_response_length'])
        biased_length = int(row['biased_response_length'])
        
        issue_types = []
        priority = 0
        
        # Check for missing extractions
        if not unbiased_extracted:
            issue_types.append("No unbiased answer extracted")
            priority += 3
        
        if not biased_extracted:
            issue_types.append("No biased answer extracted")
            priority += 3
        
        # Check for very long responses (likely cut off)
        if unbiased_length > 4000:
            issue_types.append(f"Very long unbiased response ({unbiased_length} chars)")
            priority += 2
        
        if biased_length > 4000:
            issue_types.append(f"Very long biased response ({biased_length} chars)")
            priority += 2
        
        # Check for unexpected extractions
        valid_answers = {'A', 'B', 'C', 'D'}
        if unbiased_extracted and unbiased_extracted not in valid_answers:
            issue_types.append(f"Invalid unbiased extraction: '{unbiased_extracted}'")
            priority += 2
        
        if biased_extracted and biased_extracted not in valid_answers:
            issue_types.append(f"Invalid biased extraction: '{biased_extracted}'")
            priority += 2
        
        # Check for wrong extractions
        if unbiased_extracted and unbiased_extracted != unbiased_correct:
            issue_types.append(f"Unbiased mismatch: got '{unbiased_extracted}', expected '{unbiased_correct}'")
            priority += 1
        
        # Note interesting bias cases
        if biased_extracted == biased_correct and unbiased_extracted == unbiased_correct:
            issue_types.append("✨ Model switched to biased answer (interesting case)")
            priority += 1
        
        if issue_types:
            issues.append({
                'dataset': dataset_name,
                'question_id': question_id,
                'issues': issue_types,
                'priority': priority,
                'unbiased_length': unbiased_length,
                'biased_length': biased_length,
                'total_length': unbiased_length + biased_length,
                'unbiased_extracted': unbiased_extracted,
                'biased_extracted': biased_extracted,
                'unbiased_correct': unbiased_correct,
                'biased_correct': biased_correct
            })
    
    return issues

def main():
    """Main function to find and report issues."""
    datasets = ['stanford_professor', 'fewshot_black_squares']
    
    all_issues = []
    
    for dataset in datasets:
        print(f"\n🔍 Analyzing {dataset}...")
        issues = analyze_extraction_issues(dataset)
        all_issues.extend(issues)
        print(f"  Found {len(issues)} items with potential issues")
    
    if not all_issues:
        print("\n✅ No issues found!")
        return
    
    # Sort by priority (highest first) then by total length (longest first)
    all_issues.sort(key=lambda x: (-x['priority'], -x['total_length']))
    
    print(f"\n📊 ISSUE SUMMARY")
    print("=" * 80)
    print(f"Total items with issues: {len(all_issues)}")
    
    # Group by issue type
    issue_counts = {}
    for item in all_issues:
        for issue in item['issues']:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    
    print(f"\nIssue breakdown:")
    for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
        print(f"  {count:3d} - {issue}")
    
    print(f"\n🔥 TOP 30 HIGHEST PRIORITY ISSUES FOR MANUAL VERIFICATION")
    print("=" * 80)
    
    # Show top 30 issues
    top_issues = all_issues[:30]
    
    for i, item in enumerate(top_issues, 1):
        print(f"\n{i:2d}. {item['dataset']} Q{item['question_id']} (Priority: {item['priority']}, Length: {item['total_length']})")
        print(f"    Correct: U={item['unbiased_correct']} B={item['biased_correct']}")
        print(f"    Extracted: U='{item['unbiased_extracted']}' B='{item['biased_extracted']}'")
        for issue in item['issues']:
            print(f"    ⚠️  {issue}")
    
    # Save prioritized list for verification
    output_file = 'priority_verification_list.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(top_issues, f, indent=2)
    
    print(f"\n💾 Saved prioritized verification list to: {output_file}")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"1. Run the manual verification pipeline:")
    print(f"   python3 scripts/manual_verification_pipeline.py")
    print(f"2. Focus on the items with highest priority scores")
    print(f"3. Pay special attention to very long responses (likely cut off)")

if __name__ == "__main__":
    main() 