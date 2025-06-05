import json

# Ask user for the input filename
input_filename = input("Enter the filename to sort (e.g., stanford_professor_responses_gpu.jsonl): ")

# Read the JSONL file
data = []
with open(input_filename, 'r') as f:
    for line in f:
        data.append(json.loads(line.strip()))

# Sort by question_id
data_sorted = sorted(data, key=lambda x: x['question_id'])

# Generate output filename
if input_filename.endswith('.jsonl'):
    output_filename = input_filename.replace('.jsonl', '_sorted.jsonl')
else:
    output_filename = input_filename + '_sorted.jsonl'

# Write sorted data back to JSONL
with open(output_filename, 'w') as f:
    for item in data_sorted:
        f.write(json.dumps(item) + '\n')

print(f'Sorted {len(data_sorted)} responses by question_id')
print(f'Question IDs range: {data_sorted[0]["question_id"]} to {data_sorted[-1]["question_id"]}')
print(f'Output saved to: {output_filename}') 