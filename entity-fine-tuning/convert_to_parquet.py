from pyarrow import json
import pyarrow.parquet as pq

file_path = "/data/alexl/CoT-Cue-Articuation/data/prompts/gsm8k-correctness/stanford_professor.jsonl"
table = json.read_json(file_path) 
pq.write_table(table, file_path.replace('.jsonl', '.parquet'))