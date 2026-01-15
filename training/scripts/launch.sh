#!/bin/bash

models=("unsloth/gemma-3-4b-it" "unsloth/gemma-3-12b-it" "unsloth/gemma-3-27b-it")

for model in "${models[@]}"; do
    sbatch train_sft.sh "$model"
done

models=("unsloth/Qwen2.5-VL-7B-Instruct" "unsloth/Qwen2.5-VL-32B-Instruct")

for model in "${models[@]}"; do
    sbatch train_sft.sh "$model"
done