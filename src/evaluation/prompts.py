"""Evaluation prompts and templates for model response evaluation."""

import yaml
from pathlib import Path

def _load_prompts():
    """Load prompts from YAML file."""
    prompts_file = Path(__file__).parent / "prompts.yaml"
    with open(prompts_file, 'r') as f:
        return yaml.safe_load(f)

# Load prompts from YAML
_prompts = _load_prompts()

# Export constants for backward compatibility
EVALUATION_PROMPT_TEMPLATE = _prompts["evaluation_prompt_template"]
CUE_DESCRIPTIONS = _prompts["cue_descriptions"]
CUE_SPECIFIC_CASES = _prompts["cue_specific_cases"]
EVALUATION_SYSTEM_PROMPT = _prompts["system_prompt"] 