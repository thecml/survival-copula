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

# Semi-synthetic datasets
seeds=({0..9})
dataset_names=("metabric" "gbsg" "nacd" "support" "whas" "aids" "mimic_all" "seer_brain" "seer_liver" "seer_stomach")
strategies=('original' 'top_1' 'top_5' 'top_10' 'random_25')
for seed in "${seeds[@]}"; do
    for dataset_name in "${dataset_names[@]}"; do
        for strategy in "${strategies[@]}"; do
            echo "Running with seed=$seed, dataset_name=$dataset_name, strategy=$strategy"
            python3 $base_path/../src/experiments/train_semisynthetic_datasets.py --seed "$seed" --dataset_name "$dataset_name" --strategy "$strategy"
        done
    done
done
