import json
import sys
import os

# Insert the folder that contains enums.py onto sys.path
this_dir = os.path.dirname(os.path.dirname(__file__))
enums = os.path.join(this_dir, "enums")
sys.path.insert(0, enums)

from Cue import Cue

def load_data(file_path: str = "data/responses/stanford_professor_responses.jsonl"):
    responses = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            responses.append(data)
    return responses

def calculate_average_token_length():
    lengths = []
    for cue in Cue:
        selected_cue = cue.value
        responses = load_data(f"data/responses/{selected_cue}_responses.jsonl")
        for response in responses:
            if response["status"] == "success":
                # append both lengths into the same list
                lengths.append(len(response["unbiased_response"]))
                lengths.append(len(response["biased_response"]))

    if not lengths:
        return 0

    return 2 * (sum(lengths) / 4) / len(lengths)

if __name__ == "__main__":
    max_tokens = calculate_average_token_length()
    print("Desired max tokens: ", max_tokens)
