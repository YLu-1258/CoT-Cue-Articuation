#!/bin/bash

echo "🤖 vLLM MODEL STATUS CHECKER"
echo "=============================="

echo -e "\n📊 CURRENT MODELS RUNNING:"
echo "GPU 0 | Port 6005 | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
echo "GPU 1 | Port 6003 | deepseek-ai/DeepSeek-R1-Distill-Llama-8B" 
echo "GPU 4 | Port 6002 | checkpoints/meta-llama/Meta-Llama-3.1-8B-Instruct"

echo -e "\n🔍 LIVE PORT TEST:"
for port in 6002 6003 6004 6005 6006 6007; do
    echo -n "Port $port: "
    if curl -s --connect-timeout 2 "http://localhost:$port/v1/models" >/dev/null 2>&1; then
        model=$(curl -s "http://localhost:$port/v1/models" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['data'][0]['id'])" 2>/dev/null)
        echo "✅ $model"
    else
        echo "❌ Not responding"
    fi
done

echo -e "\n🖥️  GPU MEMORY USAGE:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | while IFS=, read gpu name mem_used mem_total; do
    usage_percent=$((mem_used * 100 / mem_total))
    if [ $usage_percent -gt 80 ]; then
        status="🔴 HIGH"
    elif [ $usage_percent -gt 20 ]; then
        status="🟡 MED"
    else
        status="🟢 LOW"
    fi
    echo "GPU $gpu: $usage_percent% ($mem_used/$mem_total MB) $status"
done

echo -e "\n📈 PROCESS INFO:"
ps aux | grep "vllm serve" | grep -v grep | wc -l | xargs echo "Active vLLM processes:" 