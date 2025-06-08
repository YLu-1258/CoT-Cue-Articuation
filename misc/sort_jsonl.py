import json

files = ['data/responses/stanford_professor_responses.jsonl', 'data/responses/fewshot_black_squares_responses.jsonl']

# Read JSONL lines
for file in files:
    with open(file, 'r') as f:
        data = [json.loads(line) for line in f]

    # Sort by question_id
    sorted_data = sorted(data, key=lambda x: x['question_id'])

    # Write back to JSONL
    with open(file, 'w') as f:
        for item in sorted_data:
            f.write(json.dumps(item) + '\n')

    print(f"Sorted JSONL saved to {file}")