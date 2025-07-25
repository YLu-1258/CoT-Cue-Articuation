import re

def compute_score(
    data_source,
    solution_str: str,
    ground_truth=None,
    extra_info=None
) -> float:
    """
    Reward the model for expressing doubt/uncertainty via
    epistemic hedge phrases. Returns a score in [0,1].
    """

    # 2. Define common doubt-markers (add more as you discover them)
    doubt_markers = [
        r"\but I\b",
        r"\bnot sure\b",
        r"\buncertain\b",
        r"\bI am not sure\b",
        r"\bI am uncertain\b",
        r"\bBut I could be wrong\b",
        r"\bI could be mistaken\b",
        r"\bI might be wrong\b",
        r"\bI could be wrong\b",
        r"\bI may be wrong\b",
        r"\bWait but\b",
        r"\bWait, but\b"
    ]

    # 3. Count total occurrences of all markers (case-insensitive)
    text = solution_str
    total_hits = 0
    for pattern in doubt_markers:
        total_hits += len(re.findall(pattern, text, flags=re.IGNORECASE))

     # 3. Set an IDEAL range
    min_ideal = 3   # you want at least two hedges
    max_ideal = 5   # but no more than five

    # 4. Compute a piecewise linear “doubt_score” in [0,1]
    if total_hits < min_ideal:
        # linearly ramp up from 0 (at 0 hits) to 1 (at min_ideal hits)
        doubt_score = total_hits / min_ideal
    elif total_hits > max_ideal:
        # linearly ramp down from 1 (at max_ideal hits) to 0 (at 2*max_ideal hits)
        # you can choose a cap point; here we assume 2×max_ideal is “way too many”
        doubt_score = max(0.0, 1 - (total_hits - max_ideal) / max_ideal)
    else:
        doubt_score = 1.0

    return doubt_score
