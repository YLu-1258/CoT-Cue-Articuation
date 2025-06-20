python - <<'PY'
import pandas as pd
df = pd.read_json("data/prompts/gsm8k/test_stanford_professor.jsonl", lines=True)
df.to_parquet("data/prompts/gsm8k/test_stanford_professor.parquet", index=False)
df = pd.read_json("data/prompts/gsm8k/train_stanford_professor.jsonl", lines=True)
df.to_parquet("data/prompts/gsm8k/train_stanford_professor.parquet", index=False)
PY
