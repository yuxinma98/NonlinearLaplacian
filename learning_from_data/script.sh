#!/bin/bash

tasks=("planted_submatrix" "nonnegative_pca")
models=("tanh" "relu")

for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
        echo "Running learning_from_data.main with task: $task and model: $model"
        CUDA_VISIBLE_DEVICES=3 python -m learning_from_data.main --task "$task" --model "$model"
    done
done