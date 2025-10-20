#!/bin/bash

base_path=$(dirname "$0")            # relative
base_path=$(cd "$MY_PATH" && pwd)    # absolutized and normalized
if [[ -z "$base_path" ]] ; then  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  exit 1  # fail
fi
echo "$base_path"

results_path="$base_path/../results"
if [ -d "$results_path" ]; then
  rm -f "$results_path"/*.csv
fi

# Synthetic datasets
python3 $base_path/../src/experiments/train_synthetic_censoring.py
python3 $base_path/../src/experiments/train_synthetic_correct_copula.py
python3 $base_path/../src/experiments/train_synthetic_wrong_copula.py