import tiktoken

def compute_score(data_source, solution_str, ground_truth=None, extra_info=None) -> float:
    """
    A simple reward function that gives a score based on the length of the response.
    The longer the response, the higher the score, up to a maximum length.
    """
    max_length = 1024  # Define a maximum length for scoring
    response_length = len(tiktoken.encoding_for_model("gpt-4").encode(solution_str))
    
    # Normalize the score to be between 0 and 1
    score = min(response_length / max_length, 1.0)
    
    return score