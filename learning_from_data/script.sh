#!/bin/bash

tasks=("nonnegative_pca_recovery")  # Define tasks as an array
models=("tanh" "relu")  # Models array

for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
        echo "Running learning_from_data.main with task: $task and model: $model"
        CUDA_VISIBLE_DEVICES=3 python -m learning_from_data.main --task "$task" --model "$model" --lr 0.001
    done
done