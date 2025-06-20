#!/bin/bash
# VERL Environment Setup Script (Native Installation - No Docker)
# For CUDA 11.5 compatibility

set -e  # Exit on any error

echo "🚀 Setting up VERL environment for Qwen3-4B fine-tuning (CUDA 11.5 compatible)"
echo "=================================================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Please run this script from the verl-fine-tuning directory"
    echo "   cd verl-fine-tuning && bash setup_environment.sh"
    exit 1
fi

# Check CUDA version
echo "🔧 Checking CUDA version..."
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    echo "   CUDA version: $cuda_version"
    
    if [[ "$cuda_version" == "11.5" ]]; then
        echo "✅ CUDA 11.5 detected - using compatible configuration"
        export CUDA_11_5=true
    elif [[ "$cuda_version" > "11.7" ]]; then
        echo "✅ CUDA $cuda_version detected - can use FlashAttention"
        export CUDA_11_5=false
    else
        echo "⚠️  CUDA $cuda_version may have compatibility issues"
    fi
else
    echo "⚠️  nvcc not found - please ensure CUDA is properly installed"
fi

# Check GPU availability
echo "🖥️  Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "   Found $gpu_count GPU(s):"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | nl
else
    echo "❌ Error: nvidia-smi not found - please ensure NVIDIA drivers are installed"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Install VERL from source
echo "🔧 Installing VERL from source..."
if [ ! -d "verl" ]; then
    echo "   Cloning VERL repository..."
    git clone https://github.com/volcengine/verl.git
fi

cd verl
echo "   Installing VERL in development mode..."
pip install -e .
cd ..

# Verify installation
echo "🧪 Verifying installation..."
python -c "import verl; print('✅ VERL installed successfully')" || {
    echo "❌ VERL installation failed"
    exit 1
}

python -c "import torch; print(f'✅ PyTorch CUDA: {torch.version.cuda}')" || {
    echo "❌ PyTorch not properly installed"
    exit 1
}

python -c "import transformers; print(f'✅ Transformers: {transformers.__version__}')" || {
    echo "❌ Transformers not properly installed"
    exit 1
}

# Create necessary directories
echo "📁 Creating output directories..."
mkdir -p outputs
mkdir -p logs

# Set environment variables for CUDA 11.5 if needed
if [ "$CUDA_11_5" = true ]; then
    echo "🔧 Setting CUDA 11.5 environment variables..."
    echo "export FLASH_ATTENTION_SKIP_CUDA_CHECK=TRUE" >> ~/.bashrc
    echo "export DISABLE_FLASH_ATTN=1" >> ~/.bashrc
    export FLASH_ATTENTION_SKIP_CUDA_CHECK=TRUE
    export DISABLE_FLASH_ATTN=1
fi

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Prepare GSM8K dataset:"
echo "      python scripts/prepare_gsm8k_data.py --local_dir /data/kevinchu/gsm8k"
echo ""
if [ "$CUDA_11_5" = true ]; then
    echo "   2. Run GRPO training (CUDA 11.5 compatible):"
    echo "      python scripts/run_grpo_training_cuda115.py"
    echo ""
    echo "   3. Run SFT training (CUDA 11.5 compatible):"
    echo "      python scripts/run_sft_training_cuda115.py"
else
    echo "   2. Run GRPO training:"
    echo "      python scripts/run_grpo_training.py"
    echo ""
    echo "   3. Run SFT training:"
    echo "      python scripts/run_sft_training.py"
fi
echo ""
echo "🎉 Happy fine-tuning!" 