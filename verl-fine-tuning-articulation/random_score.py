import random

def compute_score(data_source, solution_str, ground_truth=None, extra_info=None) -> float:
    """
    A simple reward function that gives a random score
    """
    score = random.uniform(0, 1)  # Generate a random score between 0 and 1
    
    return score