#!/bin/bash

source ~/.bashrc
conda activate xray_llm

model=$1

python3 ../src/train.py train=SFT model_id=$model
