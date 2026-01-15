#!/bin/bash

export UNSLOTH_DISABLE_PATCHING=1

source ~/.bashrc
conda activate xray_llm


model=$1
mode=$2
file_path=$3

python3 ../src/test_localization.py $model $mode $file_path
