#!/bin/bash
# Run remaining/new model generations and evaluations
# 1. Complete DeepSeek-R1 adversarial (60 missing)
# 2. Run Mistral Nemo 12B (standard + adversarial)
# 3. Run Gemma 2 9B (standard + adversarial)
# 4. Re-evaluate all and update leaderboard

set -e
cd /home/aietlab/Mourya/Benchmark/cogbenchv2
PYTHON=/home/aietlab/Mourya/Benchmark/cogbench/venv/bin/python
LOG=data/results/run_new_models.log

echo "=== Starting new model runs: $(date) ===" | tee -a "$LOG"

# 1. Complete DeepSeek-R1 adversarial
echo "--- DeepSeek-R1 adversarial (completing 60 missing) ---" | tee -a "$LOG"
$PYTHON scripts/run_benchmark.py --model deepseek-r1:14b --mode adversarial 2>&1 | tee -a "$LOG"

# 2. Mistral Nemo 12B
echo "--- Mistral Nemo 12B standard ---" | tee -a "$LOG"
$PYTHON scripts/run_benchmark.py --model mistral-nemo:12b --mode standard 2>&1 | tee -a "$LOG"
echo "--- Mistral Nemo 12B adversarial ---" | tee -a "$LOG"
$PYTHON scripts/run_benchmark.py --model mistral-nemo:12b --mode adversarial 2>&1 | tee -a "$LOG"

# 3. Gemma 2 9B
echo "--- Gemma 2 9B standard ---" | tee -a "$LOG"
$PYTHON scripts/run_benchmark.py --model gemma2:9b --mode standard 2>&1 | tee -a "$LOG"
echo "--- Gemma 2 9B adversarial ---" | tee -a "$LOG"
$PYTHON scripts/run_benchmark.py --model gemma2:9b --mode adversarial 2>&1 | tee -a "$LOG"

# 4. Re-evaluate all models
echo "--- Re-evaluating all models ---" | tee -a "$LOG"
$PYTHON scripts/reeval_all.py 2>&1 | tee -a "$LOG"

# 5. Update leaderboard
echo "--- Populating leaderboard ---" | tee -a "$LOG"
$PYTHON scripts/populate_leaderboard.py 2>&1 | tee -a "$LOG"

echo "=== All done: $(date) ===" | tee -a "$LOG"
