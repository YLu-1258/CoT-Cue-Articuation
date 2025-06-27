#!/bin/bash

# Example script for running GSM8K response generation and evaluation
# Make sure your vLLM server is running before executing this script

# Configuration
PORT=6002                    # vLLM server port
MODEL_ID=""                  # Leave empty for auto-detection or specify model name
MAX_WORKERS=4               # Number of parallel workers
TEMPERATURE=0.0             # Sampling temperature (0.0 for deterministic)
MAX_TOKENS=1024             # Maximum tokens to generate
MAX_EXAMPLES=1000            # Number of examples to process
OUTPUT_DIR="data/gsm8k/unbiased_responses"

echo "🚀 Running GSM8K Response Generation Example"
echo "============================================="
echo "Port: $PORT"
echo "Max Examples: $MAX_EXAMPLES"
echo "Max Workers: $MAX_WORKERS"
echo "Temperature: $TEMPERATURE"
echo "Max Tokens: $MAX_TOKENS"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Check if vLLM server is running
echo "🔍 Checking if vLLM server is running on port $PORT..."
if ! curl -s http://localhost:$PORT/v1/models > /dev/null; then
    echo "❌ vLLM server not responding on port $PORT"
    echo ""
    echo "Please start your vLLM server first. Example:"
    echo "python -m vllm.entrypoints.openai.api_server \\"
    echo "    --model YOUR_MODEL_PATH \\"
    echo "    --port $PORT \\"
    echo "    --max-model-len 4096 \\"
    echo "    --dtype auto"
    echo ""
    exit 1
fi

echo "✅ vLLM server is running"
echo ""

# Run the GSM8K response generation
echo "🎯 Starting GSM8K response generation..."
python scripts/generate_gsm8k_responses.py \
    --port $PORT \
    --max-workers $MAX_WORKERS \
    --temperature $TEMPERATURE \
    --max-tokens $MAX_TOKENS \
    --max-examples $MAX_EXAMPLES \
    --output-dir $OUTPUT_DIR \
    --split test

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 GSM8K response generation completed successfully!"
    echo "📁 Results saved in: $OUTPUT_DIR"
    echo ""
    echo "You can also evaluate existing results with:"
    echo "python scripts/generate_gsm8k_responses.py --evaluate-only $OUTPUT_DIR/gsm8k_test_responses.jsonl"
else
    echo ""
    echo "❌ GSM8K response generation failed!"
    exit 1
fi 