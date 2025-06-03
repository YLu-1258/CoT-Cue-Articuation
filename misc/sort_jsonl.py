import json

# Read the JSONL file
data = []
with open('stanford_professor_responses_gpu.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line.strip()))

# Sort by question_id
data_sorted = sorted(data, key=lambda x: x['question_id'])

# Write sorted data back to JSONL
with open('stanford_professor_responses_sorted.jsonl', 'w') as f:
    for item in data_sorted:
        f.write(json.dumps(item) + '\n')

print(f'Sorted {len(data_sorted)} responses by question_id')
print(f'Question IDs range: {data_sorted[0]["question_id"]} to {data_sorted[-1]["question_id"]}') 