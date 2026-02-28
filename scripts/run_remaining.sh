#!/bin/bash
# Run remaining benchmark models sequentially
# phi4 adversarial is already running (PID tracked separately)
# This script runs qwen2.5 and deepseek-r1 after phi4 finishes

cd /home/aietlab/Mourya/Benchmark/cogbenchv2
PYTHON=/home/aietlab/Mourya/Benchmark/cogbench/venv/bin/python

echo "============================================"
echo "Waiting for phi4:14b adversarial to finish..."
echo "============================================"

# Wait for phi4 adversarial to complete (PID 1561428)
while kill -0 1561428 2>/dev/null; do
    sleep 30
done
echo "phi4:14b adversarial done!"

# Unload phi4 from Ollama
echo "Unloading phi4:14b..."
curl -s http://localhost:11434/api/generate -d '{"model":"phi4:14b","keep_alive":0}' > /dev/null 2>&1
sleep 5

echo ""
echo "============================================"
echo "Running qwen2.5:14b (standard + adversarial)"
echo "============================================"
$PYTHON -u scripts/run_benchmark.py --model qwen2.5:14b --mode both 2>&1

# Unload qwen after done
echo "Unloading qwen2.5:14b..."
curl -s http://localhost:11434/api/generate -d '{"model":"qwen2.5:14b","keep_alive":0}' > /dev/null 2>&1
sleep 5

echo ""
echo "============================================"
echo "Running deepseek-r1:14b (standard + adversarial)"
echo "============================================"
$PYTHON -u scripts/run_benchmark.py --model deepseek-r1:14b --mode both 2>&1

echo ""
echo "============================================"
echo "ALL MODELS COMPLETE"
echo "============================================"

# Run final leaderboard population
echo "Populating leaderboard..."
$PYTHON -u scripts/populate_leaderboard.py 2>&1

echo "Done!"
