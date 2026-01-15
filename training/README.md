# SFT Training

This directory contains the code for Supervised Fine-Tuning (SFT) of VLMs.

## Directory Structure

- **conf/**: Hydra configuration files.
    - `config.yaml`: Main configuration.
    - `train/`: Training specific configs (e.g., `SFT_real.yaml`).
    - `lora/`: LoRA configuration.
- **scripts/**: Shell scripts to launch training.
- **src/**: Source code for the trainer and model.
- **prompts/**: System prompts used during training.

## Usage

### Train SFT (Real Data)

To launch the SFT training on real data:

```bash
cd scripts
bash train_sft_real.sh <model_id>
```

Example:
```bash
bash train_sft_real.sh unsloth/gemma-3-12b-it
```

### Train SFT (Simulated/Other)

To launch SFT training with other configurations:

```bash
cd scripts
bash train_sft.sh <model_id>
```

This script defaults to `train=SFT` configuration.
