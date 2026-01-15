#!/bin/bash

export UNSLOTH_DISABLE_PATCHING=1

source ~/.bashrc
conda activate xray_llm


model=$1
mode=$2
csv_path=$3

python3 ../src/Inference_navigation.py $model $mode $csv_path
