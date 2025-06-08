# CoT Cue Articulation Research

## Usage

### 1. Generate Datasets
```bash
python scripts/generate_data.py
```

### 2. Generate Model Responses
```bash
python scripts/generate_responses.py --port <PORT> --max-workers <WORKERS>
```

# 3. Generate responses (much faster with 4 GPUs!)
```bash
python scripts/generate_responses_multi_gpu.py --ports 6000 6001 6002 6003

### 3. Evaluate Responses
```bash
python scripts/evaluate_responses.py --port <PORT> --max-workers <WORKERS>
```

## Parameters

- `--port`: **Required**. Port number where your LLM server is running
- `--max-workers`: Number of parallel workers (optional, defaults to 8)
- `--model-id`: Specific model ID to use (optional, auto-detects if not specified)
- `--cue`: Process specific cue only: `stanford_professor` or `fewshot_black_squares`
