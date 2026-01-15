# Inference

This directory contains scripts for running inference using the trained models.

## Usage

### Localization Task

To run the localization test:

```bash
cd src
python3 tester.py <model_id> <mode>
```

Arguments:
- `<model_id>`: The Hugging Face model ID or path to the checkpoint.
- `<mode>`: The test mode (e.g., `test`, `real`).

Example:
```bash
python3 tester.py unsloth/gemma-3-12b-it test
```

### Navigation Task

To run the navigation test:

```bash
cd src
python3 tester2.py <model_id> <mode> <next_path>
```

Arguments:
- `<model_id>`: The Hugging Face model ID or path to the checkpoint.
- `<mode>`: The test mode (e.g., `test`, `flip`).
- `<next_path>`: Path to the next step data (optional/mode dependent).

## Source Code

- **Inference_localization.py**: Core logic for localization inference.
- **Inference_navigation.py**: Core logic for navigation inference.
