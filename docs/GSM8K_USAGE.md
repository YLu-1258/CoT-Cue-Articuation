# GSM8K Response Generation and Evaluation

This directory contains scripts for generating and evaluating GSM8K responses using a vLLM-hosted HuggingFace model.

## Quick Start

1. **Start your vLLM server** with a HuggingFace model:
   ```bash
   python -m vllm.entrypoints.openai.api_server \
       --model YOUR_MODEL_PATH \
       --port 8000 \
       --max-model-len 4096 \
       --dtype auto
   ```

2. **Run the example script** (processes 200 questions by default):
   ```bash
   ./scripts/run_gsm8k_example.sh
   ```

   Or run manually:
   ```bash
   python scripts/generate_gsm8k_responses.py --port 8000 --max-examples 200
   ```

## Script Features

### `generate_gsm8k_responses.py`

- **Automatic Dataset Loading**: Downloads GSM8K dataset from HuggingFace
- **Parallel Processing**: Configurable number of worker threads
- **Resume Support**: Automatically resumes from partial results
- **Real-time Evaluation**: Shows running accuracy during generation
- **Comprehensive Metrics**: Detailed evaluation with sample outputs
- **Flexible Answer Extraction**: Uses both strict and flexible answer parsing

### Key Arguments

```bash
python scripts/generate_gsm8k_responses.py \
    --port 8000 \                    # vLLM server port
    --max-examples 200 \             # Number of questions to process
    --max-workers 4 \                # Parallel workers
    --temperature 0.0 \              # Sampling temperature
    --max-tokens 1024 \              # Max tokens per response
    --output-dir data/gsm8k_responses \  # Output directory
    --split test                     # Dataset split (test/train)
```

### Evaluation Only

To evaluate existing results without generating new responses:

```bash
python scripts/generate_gsm8k_responses.py \
    --evaluate-only data/gsm8k_responses/gsm8k_test_responses.jsonl
```

## Output Format

The script generates:

1. **Response File** (`gsm8k_test_responses.jsonl`): JSONL with responses and evaluations
2. **Evaluation File** (`gsm8k_test_responses_evaluation.json`): Detailed metrics and samples

### Sample Response Entry

```json
{
    "question_id": 0,
    "question": "Natalia sold clips to 48 of her friends...",
    "formatted_question": "Natalia sold clips to 48 of her friends... Let's think step by step and output the final answer after \"####\".",
    "response": "I need to find how much money Natalia made... #### 192",
    "ground_truth": "192",
    "predicted_answer": "192",
    "is_correct": true,
    "score": 1.0,
    "official_score": 1.0,
    "status": "success"
}
```

### Evaluation Metrics

```json
{
    "metrics": {
        "total_questions": 200,
        "successful_responses": 198,
        "failed_responses": 2,
        "correct_answers": 156,
        "success_rate": 0.990,
        "accuracy": 0.788,
        "overall_accuracy": 0.780
    }
}
```

## Answer Extraction

The script uses the GSM8K evaluation utilities from the VERL library:

- **Strict Mode**: Looks for answers in `#### NUMBER` format
- **Flexible Mode**: Extracts any numerical value from the response
- **Numerical Comparison**: Handles cases like "42.0" vs "42"

## Error Handling

- **Connection Issues**: Validates vLLM server connection before starting
- **Resume Capability**: Automatically resumes from interruptions
- **Error Logging**: Failed responses are logged with error details
- **Graceful Degradation**: Continues processing even if individual questions fail

## Performance Tips

1. **Adjust Workers**: Use `--max-workers` based on your GPU memory
2. **Temperature**: Use 0.0 for deterministic results, higher for diversity
3. **Batch Size**: vLLM automatically batches requests for efficiency
4. **Resume**: Always use resume capability for long runs

## Example vLLM Server Commands

### For Different Model Sizes

**Small Models (1-7B parameters):**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model microsoft/DialoGPT-medium \
    --port 8000 \
    --max-model-len 2048
```

**Large Models (13B+ parameters):**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-13b-chat-hf \
    --port 8000 \
    --max-model-len 4096 \
    --tensor-parallel-size 2
```

## Troubleshooting

### Common Issues

1. **"Connection failed"**: Ensure vLLM server is running and accessible
2. **"No models available"**: Check vLLM server model loading
3. **Memory errors**: Reduce `--max-workers` or vLLM batch size
4. **Slow generation**: Check GPU utilization and network latency

### Debug Commands

```bash
# Test vLLM server connection
curl http://localhost:8000/v1/models

# Check server health
curl http://localhost:8000/health

# Test single completion
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "your-model", "prompt": "What is 2+2?", "max_tokens": 50}'
``` 