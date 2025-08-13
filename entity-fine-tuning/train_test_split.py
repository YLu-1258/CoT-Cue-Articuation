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

def split_train_test(df, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df

def save_parquet(df, file_path):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, file_path)

if __name__ == "__main__":
    file_path = "/data/alexl/CoT-Cue-Articuation/data/prompts/rl/gsm8k-correctness/train_stanford_professor.jsonl"
    df = load_jsonl(file_path)
    
    train_df, test_df = split_train_test(df)
    
    output_dir = os.path.dirname(file_path)
    train_file = os.path.join(output_dir, file_path.split("/")[-1].replace("train_stanford_professor.jsonl", "train_stanford_professor.parquet"))
    test_file = os.path.join(output_dir, file_path.split("/")[-1].replace("train_stanford_professor.jsonl", "validation_stanford_professor.parquet"))
    
    save_parquet(train_df, train_file)
    save_parquet(test_df, test_file)
    
    print(f"Train set saved to {train_file} with {len(train_df)} records.")
    print(f"Test set saved to {test_file} with {len(test_df)} records.")