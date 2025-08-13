# split the data into train and test sets
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from sklearn.model_selection import train_test_split
import os
import json
from tqdm import tqdm

def load_jsonl(file_path):
    df = pd.read_json(file_path, lines=True)
    return df

def save_parquet(df, file_path):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, file_path)

if __name__ == "__main__":
    file_path = "/data/alexl/CoT-Cue-Articuation/data/prompts/rl/gsm8k-correctness/test_stanford_professor.jsonl"
    df = load_jsonl(file_path)
    
    
    output_dir = os.path.dirname(file_path)
    parquet_file = os.path.join(output_dir, file_path.split("/")[-1].replace(".jsonl", ".parquet"))
    
    save_parquet(df, parquet_file)
    
    print(f"Data saved to {parquet_file} with {len(df)} records.")