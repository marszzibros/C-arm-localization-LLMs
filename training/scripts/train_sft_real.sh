#!/bin/bash

source ~/.bashrc
conda activate xray_llm

model=$1

python3 ../src/train.py train=SFT_real model_id=$model
