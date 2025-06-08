#!/usr/bin/env python3
"""
Generate model responses for CoT cue articulation experiments.

This script takes the generated datasets and produces model responses
for both biased and unbiased prompts.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.client import LLMClient
from llm.response_generator import ResponseGenerator
from enums.cue import Cue


def main():
    """Generate responses for all datasets."""
    parser = argparse.ArgumentParser(description="Generate model responses")
    parser.add_argument("--port", type=int, required=True, help="LLM server port")
    parser.add_argument("--model-id", type=str, help="Specific model ID to use")
    parser.add_argument("--max-workers", type=int, help="Number of parallel workers")
    parser.add_argument("--cue", type=str, choices=[cue.value for cue in Cue], 
                       help="Generate responses for specific cue only")
    
    args = parser.parse_args()
    
    print("🚀 CoT Cue Articulation - Response Generation")
    print("=" * 50)
    
    # Create LLM client
    try:
        client = LLMClient.local(port=args.port, model_id=args.model_id)
        print(f"📡 Connected to: {client}")
        print(client.test_connection())
    except Exception as e:
        print(f"❌ Failed to connect to LLM server: {e}")
        print(f"Make sure the server is running on port {args.port}")
        sys.exit(1)
    
    # Create response generator
    max_workers = args.max_workers if args.max_workers else 8  # Default to 8 if not specified
    generator = ResponseGenerator(client, "data/responses", max_workers)
    
    # Generate responses
    if args.cue:
        # Generate for specific cue
        cue = Cue(args.cue)
        input_file = Path("data") / f"{cue.value}.jsonl"
        
        if not input_file.exists():
            print(f"❌ Dataset not found: {input_file}")
            print("Run 'python scripts/generate_data.py' first")
            sys.exit(1)
        print(input_file)
        output_file = generator.generate_responses_for_dataset(input_file, cue)
        print(f"\n✅ Responses generated: {output_file}")
    else:
        # Generate for all cues
        results = generator.generate_all_responses()
        
        print("=" * 50)
        print("📊 RESPONSE GENERATION SUMMARY:")
        
        for cue, output_file in results.items():
            print(f"{cue.display_name}: {output_file}")
        
        if results:
            print("\n🎉 All responses generated successfully!")
            print("Ready for evaluation.")
        else:
            print("\n⚠️  No datasets found for response generation.")
            print("Run 'python scripts/generate_data.py' first")


if __name__ == "__main__":
    main() 