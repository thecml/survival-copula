#!/bin/bash
set -euo pipefail

base_path="$(cd "$(dirname "$0")" && pwd)"
echo "$base_path"

results_path="$base_path/../results"
if [ -d "$results_path" ]; then
  rm -f "$results_path"/*.csv
fi

# Synthetic datasets
python3 $base_path/../src/experiments/train_synthetic_censoring.py
python3 $base_path/../src/experiments/train_synthetic_correct_copula.py
python3 $base_path/../src/experiments/train_synthetic_correct_copula_cgq.py
python3 $base_path/../src/experiments/train_synthetic_wrong_copula.py
