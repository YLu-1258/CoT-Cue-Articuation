"""Data generation utilities."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_generation.formatters import BaseFormatter, StanfordProfessorFormatter, FewShotSquaresFormatter
from enums.cue import Cue


class DataGenerator:
    """Handles generation and validation of datasets."""
    
    def __init__(self, output_dir: str = "data"):
        """Initialize data generator with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.formatters = {
            Cue.STANFORD_PROFESSOR: StanfordProfessorFormatter(),
            Cue.FEW_SHOT_BLACK_SQUARES: FewShotSquaresFormatter()
        }
    
    def generate_dataset(self, cue: Cue, filename: Optional[str] = None) -> Path:
        """Generate dataset for a specific cue type."""
        if filename is None:
            filename = f"{cue.value}.jsonl"
        
        output_path = self.output_dir / filename
        formatter = self.formatters[cue]
        
        print(f"Generating {cue.display_name} dataset...")
        
        with open(output_path, "w") as f:
            for entry in formatter.dataset:
                data_entry = formatter.create_entry(entry)
                f.write(json.dumps(data_entry) + "\n")
        
        print(f"✅ Generated {output_path}")
        return output_path
    
    def generate_all_datasets(self) -> Dict[Cue, Path]:
        """Generate datasets for all cue types."""
        print("=== GENERATING ALL DATASETS ===")
        
        results = {}
        for cue in Cue:
            results[cue] = self.generate_dataset(cue)
        
        print("✅ All datasets generated successfully!\n")
        return results
    
    def validate_dataset(
        self, 
        filepath: Path, 
        cue: Cue, 
        max_entries: int = 50
    ) -> bool:
        """Validate a dataset file for quality and correctness."""
        print(f"=== VALIDATING {cue.display_name.upper()} DATASET ===")
        
        if not filepath.exists():
            print(f"❌ {filepath} does not exist!")
            return False
        
        # File info
        stat = filepath.stat()
        mod_time = datetime.fromtimestamp(stat.st_mtime)
        
        print(f"File: {filepath}")
        print(f"Last modified: {mod_time}")
        print(f"File size: {stat.st_size} bytes")
        
        # Count and validate entries
        entries = self._load_jsonl(filepath)
        total_entries = len(entries)
        
        print(f"Total entries: {total_entries}")
        print(f"Checking first {min(max_entries, total_entries)} entries...\n")
        
        same_answer_count = 0
        problematic_entries = []
        
        for i, entry in enumerate(entries[:max_entries]):
            try:
                unbiased = entry['unbiased_answer']
                biased = entry['biased_answer']
                
                if unbiased == biased:
                    same_answer_count += 1
                    problematic_entries.append(i + 1)
                    print(f"❌ Entry {i+1}: SAME ANSWERS! "
                          f"unbiased='{unbiased}' == biased='{biased}'")
                    print(f"   Question: {entry['unbiased_question'][:80]}...")
                    print()
                else:
                    if i < 3:  # Show first few good entries
                        print(f"✅ Entry {i+1}: Different - "
                              f"unbiased='{unbiased}' != biased='{biased}'")
                        
            except KeyError as e:
                print(f"❌ Entry {i+1}: Missing key {e}")
            except Exception as e:
                print(f"❌ Entry {i+1}: Error {e}")
        
        # Summary
        checked = min(max_entries, total_entries)
        different_count = checked - same_answer_count
        
        print(f"\n--- {cue.display_name} RESULTS ---")
        print(f"Entries checked: {checked}")
        print(f"Same answers: {same_answer_count}")  
        print(f"Different answers: {different_count}")
        
        if same_answer_count > 0:
            percentage = (same_answer_count / checked) * 100
            print(f"❌ PROBLEM: {percentage:.1f}% of entries have matching answers!")
            return False
        else:
            print(f"✅ SUCCESS: All entries have different answers!")
            return True
    
    def validate_all_datasets(self) -> Dict[Cue, bool]:
        """Validate all generated datasets."""
        results = {}
        
        for cue in Cue:
            filepath = self.output_dir / f"{cue.value}.jsonl"
            if filepath.exists():
                results[cue] = self.validate_dataset(filepath, cue, max_entries=100)
                print()
            else:
                print(f"❌ Dataset for {cue.display_name} not found at {filepath}")
                results[cue] = False
        
        # Final summary
        print("=" * 50)
        print("VALIDATION SUMMARY:")
        
        all_passed = True
        for cue, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{cue.display_name}: {status}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n🎉 ALL DATASETS VALIDATED SUCCESSFULLY!")
        else:
            print("\n🚨 SOME DATASETS HAVE ISSUES!")
        
        return results
    
    def _load_jsonl(self, filepath: Path) -> List[Dict]:
        """Load JSONL file into list of dictionaries."""
        entries = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line.strip()))
        return entries 