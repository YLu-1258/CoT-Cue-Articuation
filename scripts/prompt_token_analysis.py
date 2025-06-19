#!/usr/bin/env python3
import json
from pathlib import Path
import argparse
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import tiktoken
from enums.cue import Cue

from evaluation.prompts import (
    EVALUATION_PROMPT_TEMPLATE,
    CUE_DESCRIPTIONS,
    CUE_SPECIFIC_CASES
)

def extract_cot(full_response: str) -> str:
    """Mimic ModelEvaluator._extract_cot: cut at </think> if present."""
    marker = "</think>"
    idx = full_response.find(marker)
    return full_response[:idx] if idx != -1 else full_response


def build_prompt(resp: dict, cue: Cue) -> str:
    """
    Reconstruct the exact evaluation prompt:
    - cue-specific description & cases
    - extracted CoT from biased_response
    """
    cot = extract_cot(resp["biased_response"])
    return EVALUATION_PROMPT_TEMPLATE.format(
        cue_description=CUE_DESCRIPTIONS[cue.value],
        cue_specific_cases=CUE_SPECIFIC_CASES[cue.value],
        biased_response=cot
    )


def load_responses(path: Path) -> list:
    """Load JSONL responses from given file path."""
    responses = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # only include successful entries
            if obj.get("status") == "success":
                responses.append(obj)
    return responses


def analyze_and_save(responses: list, model: str, bins: int, cue: Cue, out_dir: Path):
    """Build prompts, count tokens, plot histogram with avg/max lines, print stats, and save figure."""
    enc = tiktoken.encoding_for_model(model)
    token_counts = []
    for resp in responses:
        prompt = build_prompt(resp, cue)
        token_counts.append(len(enc.encode(prompt)))

    # Calculate statistics
    if not token_counts:
        print(f"No tokens to analyze for cue '{cue.display_name}'.")
        return

    avg_tokens = sum(token_counts) / len(token_counts)
    max_tokens = max(token_counts)
    print(f"Cue '{cue.display_name}' | Model '{model}' -> Avg tokens: {avg_tokens:.2f}, Max tokens: {max_tokens}")

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(token_counts, bins=bins, edgecolor='black')
    # Add vertical lines for average and max
    plt.axvline(avg_tokens, linestyle='--', linewidth=2, label=f'Avg: {avg_tokens:.2f}')
    plt.axvline(max_tokens, linestyle='--', linewidth=2, label=f'Max: {max_tokens}')

    plt.title(f"Prompt Token Distribution for model {model}, cue {cue.display_name}")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Number of Prompts")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_file = out_dir / f"{model}_token_dist_{cue.value}.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved plot for cue '{cue.display_name}' with model '{model}' to {out_file}")


def main():
    p = argparse.ArgumentParser(
        description="Recount tokens in each evaluation prompt and save histograms"
    )
    p.add_argument(
        "--cue", "-c", type=str,
        choices=[cue.value for cue in Cue],
        help="Evaluate responses for specific cue only"
    )
    p.add_argument(
        "--model", "-m", default="gpt-4o",
        help="Name of model encoding to use (default: gpt-4o)"
    )
    p.add_argument(
        "--bins", "-b", type=int, default=30,
        help="Number of bins for histogram (default: 30)"
    )
    p.add_argument(
        "--out-dir", "-o", default="data/plots",
        help="Directory to save plot images"
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cue:
        cues = [Cue(args.cue)]
    else:
        cues = list(Cue)

    for cue in cues:
        resp_file = Path("data/responses/filtered") / f"{cue.value}_responses_filtered.jsonl"
        if not resp_file.exists():
            resp_file = Path("data/responses") / f"{cue.value}_responses.jsonl"
        if not resp_file.exists():
            print(f"❌ Responses file not found for cue {cue.value}: {resp_file}")
            continue

        print(f"Processing cue: {cue.value}")
        responses = load_responses(resp_file)
        if not responses:
            print(f"No successful responses for cue {cue.value}.")
            continue

        analyze_and_save(
            responses,
            args.model,
            args.bins,
            cue,
            out_dir
        )

if __name__ == "__main__":
    main()
