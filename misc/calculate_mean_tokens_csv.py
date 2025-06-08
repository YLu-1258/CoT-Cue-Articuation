import pandas as pd
import argparse
import sys
import os

# Insert the folder that contains enums.py onto sys.path
this_dir = os.path.dirname(os.path.dirname(__file__))
enums = os.path.join(this_dir, "enums")
sys.path.insert(0, enums)

from Cue import Cue

def calculate_average_lengths():
    """
    Reads the CSV at csv_path and returns the average values
    for the 'unbiased_response_length' and 'biased_response_length' columns.
    """
    try:
        dfs = []
        for cue in Cue:
            df_cue = pd.read_csv(f"data/responses/extracted_answers_{cue.value}.csv")
            dfs.append(df_cue)
    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    if not dfs:
        raise RuntimeError("No response files found to concatenate.")
    df_all = pd.concat(dfs, ignore_index=True)

    # Ensure the necessary columns exist
    required_cols = ['unbiased_response_length', 'biased_response_length']
    for col in required_cols:
        if col not in df_all.columns:
            print(f"Column '{col}' not found in the CSV.", file=sys.stderr)
            sys.exit(1)

    n_unbiased = df_all['unbiased_response_length'].count()
    n_biased   = df_all['biased_response_length'].count()

    # sum of all lengths (ignoring NaNs)
    sum_unbiased = df_all['unbiased_response_length'].sum()
    sum_biased   = df_all['biased_response_length'].sum()

    combined_mean = (sum_unbiased + sum_biased) / (n_unbiased + n_biased)
    return combined_mean

def main():

    avg = calculate_average_lengths()
    print(f"Max Tokens to use: {2 * avg}")

if __name__ == "__main__":
    main()