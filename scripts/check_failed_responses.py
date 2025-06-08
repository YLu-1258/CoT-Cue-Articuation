#!/usr/bin/env python3
"""
Check for failed responses in response files.
Quick utility to see which responses need regeneration.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enums.cue import Cue


def check_failed_responses(response_dir: str = "data/responses") -> dict:
    """Check all response files for failed responses."""
    response_path = Path(response_dir)
    results = {}
    total_stats = {"total": 0, "failed": 0, "success": 0}
    
    for cue in Cue:
        response_file = response_path / f"{cue.value}_responses.jsonl"
        
        if not response_file.exists():
            results[cue.value] = {"exists": False}
            continue
        
        stats = {"total": 0, "failed": 0, "success": 0, "failed_questions": []}
        
        with open(response_file, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line.strip())
                    stats["total"] += 1
                    total_stats["total"] += 1
                    
                    if entry.get('status') == 'error':
                        stats["failed"] += 1
                        total_stats["failed"] += 1
                        stats["failed_questions"].append({
                            "question_id": entry.get("question_id", "unknown"),
                            "error": entry.get("error", "unknown error")
                        })
                    else:
                        stats["success"] += 1
                        total_stats["success"] += 1
        
        stats["exists"] = True
        results[cue.value] = stats
    
    results["_total"] = total_stats
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Check for failed responses")
    parser.add_argument("--response-dir", type=str, default="data/responses",
                       help="Directory containing response files")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed error information")
    parser.add_argument("--cue", type=str, choices=[cue.value for cue in Cue],
                       help="Check specific cue only")
    
    args = parser.parse_args()
    
    print("🔍 Checking for failed responses...")
    print("=" * 60)
    
    results = check_failed_responses(args.response_dir)
    total_stats = results.pop("_total")
    
    if args.cue:
        # Show specific cue only
        cue_results = results.get(args.cue, {})
        if not cue_results.get("exists", False):
            print(f"❌ Response file not found for {args.cue}")
            return 1
        
        print(f"📊 Results for {args.cue}:")
        print(f"   Total responses: {cue_results['total']}")
        print(f"   Successful: {cue_results['success']}")
        print(f"   Failed: {cue_results['failed']}")
        
        if cue_results['failed'] > 0 and args.detailed:
            print(f"\n❌ Failed questions for {args.cue}:")
            for failed_q in cue_results['failed_questions']:
                print(f"   Question {failed_q['question_id']}: {failed_q['error']}")
    
    else:
        # Show summary for all cues
        print("📊 Response Status Summary:")
        print("-" * 50)
        
        for cue_value, stats in results.items():
            if not stats.get("exists", False):
                print(f"{cue_value:20} | ❌ File not found")
                continue
            
            success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
            status_icon = "✅" if stats['failed'] == 0 else "⚠️" if stats['failed'] < 5 else "❌"
            
            print(f"{cue_value:20} | {status_icon} {stats['success']:4d}/{stats['total']:4d} ({success_rate:5.1f}%) | Failed: {stats['failed']:3d}")
            
            if args.detailed and stats['failed'] > 0:
                print(f"                     | Failed questions: {[str(q['question_id']) for q in stats['failed_questions'][:10]]}")
                if len(stats['failed_questions']) > 10:
                    print(f"                     | ... and {len(stats['failed_questions']) - 10} more")
        
        print("-" * 50)
        total_success_rate = (total_stats['success'] / total_stats['total'] * 100) if total_stats['total'] > 0 else 0
        print(f"{'TOTAL':20} | 🎯 {total_stats['success']:4d}/{total_stats['total']:4d} ({total_success_rate:5.1f}%) | Failed: {total_stats['failed']:3d}")
        
        if total_stats['failed'] > 0:
            print(f"\n💡 To regenerate failed responses, run:")
            print(f"   python scripts/regenerate_failed_responses.py --ports [YOUR_PORTS]")
            
            # Show which cues have failures
            failed_cues = [cue for cue, stats in results.items() if stats.get('failed', 0) > 0]
            if failed_cues:
                print(f"\n🎯 Cues with failures: {', '.join(failed_cues)}")
        else:
            print(f"\n🎉 All responses are successful! No regeneration needed.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 