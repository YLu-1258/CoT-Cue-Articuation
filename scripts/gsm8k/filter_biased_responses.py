#!/usr/bin/env python3
"""
Filter GSM8K responses to find cases where the model's predicted answer matches the injected answer.

This indicates that the model was influenced by the bias in the prompt.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def load_responses_jsonl(file_path: Path) -> List[Dict]:
    """Load responses from JSONL file."""
    responses = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    response = json.loads(line.strip())
                    responses.append(response)
                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON decode error on line {line_num}: {e}")
                    continue
    return responses


def extract_injected_answer(response_entry: Dict) -> str:
    """Extract injected answer from response entry."""
    # Try to get from original entry
    if "original_entry" in response_entry:
        original = response_entry["original_entry"]
        if "extra_info" in original and "injected_answer" in original["extra_info"]:
            return str(original["extra_info"]["injected_answer"])
    
    return None


def analyze_bias_influence(responses: List[Dict]) -> Dict:
    """Analyze responses for bias influence."""
    total_responses = 0
    successful_responses = 0
    biased_responses = 0
    correct_but_biased = 0
    incorrect_but_biased = 0
    
    biased_examples = []
    unbiased_examples = []
    
    for response in responses:
        if response.get("status") != "success":
            continue
            
        total_responses += 1
        successful_responses += 1
        
        predicted_answer = response.get("predicted_answer")
        ground_truth = response.get("ground_truth")
        injected_answer = extract_injected_answer(response)
        is_correct = response.get("is_correct", False)
        
        if predicted_answer is not None and injected_answer is not None:
            # Check if predicted answer matches injected answer
            try:
                if float(predicted_answer) == float(injected_answer):
                    biased_responses += 1
                    
                    # Add bias information to the response
                    enhanced_response = response.copy()
                    enhanced_response["injected_answer"] = injected_answer
                    enhanced_response["matches_injection"] = True
                    enhanced_response["bias_type"] = "correct_bias" if is_correct else "incorrect_bias"
                    
                    biased_examples.append(enhanced_response)
                    
                    if is_correct:
                        correct_but_biased += 1
                    else:
                        incorrect_but_biased += 1
                else:
                    # Not biased
                    enhanced_response = response.copy()
                    enhanced_response["injected_answer"] = injected_answer
                    enhanced_response["matches_injection"] = False
                    enhanced_response["bias_type"] = "unbiased"
                    unbiased_examples.append(enhanced_response)
                    
            except (ValueError, TypeError):
                # String comparison fallback
                if str(predicted_answer) == str(injected_answer):
                    biased_responses += 1
                    
                    enhanced_response = response.copy()
                    enhanced_response["injected_answer"] = injected_answer
                    enhanced_response["matches_injection"] = True
                    enhanced_response["bias_type"] = "correct_bias" if is_correct else "incorrect_bias"
                    
                    biased_examples.append(enhanced_response)
                    
                    if is_correct:
                        correct_but_biased += 1
                    else:
                        incorrect_but_biased += 1
                else:
                    enhanced_response = response.copy()
                    enhanced_response["injected_answer"] = injected_answer
                    enhanced_response["matches_injection"] = False
                    enhanced_response["bias_type"] = "unbiased"
                    unbiased_examples.append(enhanced_response)
    
    # Calculate rates
    bias_rate = biased_responses / successful_responses if successful_responses > 0 else 0
    correct_bias_rate = correct_but_biased / biased_responses if biased_responses > 0 else 0
    incorrect_bias_rate = incorrect_but_biased / biased_responses if biased_responses > 0 else 0
    
    return {
        "metrics": {
            "total_responses": total_responses,
            "successful_responses": successful_responses,
            "biased_responses": biased_responses,
            "correct_but_biased": correct_but_biased,
            "incorrect_but_biased": incorrect_but_biased,
            "bias_rate": bias_rate,
            "correct_bias_rate": correct_bias_rate,
            "incorrect_bias_rate": incorrect_bias_rate
        },
        "biased_examples": biased_examples,
        "unbiased_examples": unbiased_examples
    }


def save_filtered_responses(results: Dict, output_dir: Path, base_name: str):
    """Save filtered responses to separate files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save biased responses
    biased_file = output_dir / f"{base_name}_biased.jsonl"
    with open(biased_file, 'w') as f:
        for response in results["biased_examples"]:
            f.write(json.dumps(response) + "\n")
    
    # Save unbiased responses
    unbiased_file = output_dir / f"{base_name}_unbiased.jsonl"
    with open(unbiased_file, 'w') as f:
        for response in results["unbiased_examples"]:
            f.write(json.dumps(response) + "\n")
    
    # Save analysis summary
    summary_file = output_dir / f"{base_name}_bias_analysis.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "metrics": results["metrics"],
            "sample_biased": results["biased_examples"][:5],
            "sample_unbiased": results["unbiased_examples"][:5]
        }, f, indent=2)
    
    return biased_file, unbiased_file, summary_file


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Filter GSM8K responses for bias influence")
    parser.add_argument("--responses-file", type=str, required=True, help="Path to responses JSONL file")
    parser.add_argument("--output-dir", type=str, default="data/gsm8k/filtered_biased_responses", help="Output directory")
    parser.add_argument("--base-name", type=str, help="Base name for output files (default: derived from input)")
    
    args = parser.parse_args()
    
    print("🚀 GSM8K Bias Analysis and Filtering")
    print("=" * 50)
    
    # Check input file exists
    responses_file = Path(args.responses_file)
    if not responses_file.exists():
        print(f"❌ Responses file not found: {responses_file}")
        sys.exit(1)
    
    # Determine base name
    base_name = args.base_name if args.base_name else responses_file.stem
    output_dir = Path(args.output_dir)
    
    print(f"📁 Input file: {responses_file}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📝 Base name: {base_name}")
    print()
    
    # Load responses
    print("📖 Loading responses...")
    try:
        responses = load_responses_jsonl(responses_file)
        print(f"Loaded {len(responses)} responses")
    except Exception as e:
        print(f"❌ Failed to load responses: {e}")
        sys.exit(1)
    
    # Analyze bias influence
    print("\n🔍 Analyzing bias influence...")
    results = analyze_bias_influence(responses)
    
    # Print analysis results
    metrics = results["metrics"]
    print("\n" + "="*50)
    print("📊 BIAS ANALYSIS RESULTS")
    print("="*50)
    print(f"Total responses:        {metrics['total_responses']}")
    print(f"Successful responses:   {metrics['successful_responses']}")
    print(f"Biased responses:       {metrics['biased_responses']}")
    print(f"  - Correct & biased:   {metrics['correct_but_biased']}")
    print(f"  - Incorrect & biased: {metrics['incorrect_but_biased']}")
    print()
    print(f"Bias rate:              {metrics['bias_rate']:.3f} ({metrics['biased_responses']}/{metrics['successful_responses']})")
    print(f"Correct bias rate:      {metrics['correct_bias_rate']:.3f} ({metrics['correct_but_biased']}/{metrics['biased_responses']})")
    print(f"Incorrect bias rate:    {metrics['incorrect_bias_rate']:.3f} ({metrics['incorrect_but_biased']}/{metrics['biased_responses']})")
    
    # Save filtered results
    print("\n💾 Saving filtered results...")
    biased_file, unbiased_file, summary_file = save_filtered_responses(results, output_dir, base_name)
    
    print(f"✅ Biased responses saved to: {biased_file}")
    print(f"✅ Unbiased responses saved to: {unbiased_file}")
    print(f"✅ Analysis summary saved to: {summary_file}")
    
    # Show some examples
    if results["biased_examples"]:
        print("\n🎯 Sample biased responses:")
        for i, example in enumerate(results["biased_examples"][:3], 1):
            print(f"\n{i}. Question {example['question_id']}:")
            print(f"   Ground truth: {example['ground_truth']}")
            print(f"   Injected: {example['injected_answer']}")
            print(f"   Predicted: {example['predicted_answer']}")
            print(f"   Correct: {example['is_correct']}")
    
    print(f"\n🎉 Bias analysis completed!")
    print(f"📊 Found {metrics['biased_responses']} biased responses out of {metrics['successful_responses']} total")
    print(f"🎯 Bias rate: {metrics['bias_rate']:.1%}")


if __name__ == "__main__":
    main() 