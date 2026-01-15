# Evaluation

This directory contains scripts for evaluating the performance of the models.

## Localization Evaluation

Located in `localization/`:

- **landmark_match_real.py**: Evaluates landmark matching performance on real data.
- **landmark_match_synthetic.py**: Evaluates landmark matching performance on synthetic data.
- **dl_real_analysis.py**: Deep Learning based analysis for real data.
- **dl_synthetic_analysis.py**: Deep Learning based analysis for synthetic data.

Usage:
You can run these scripts directly to print evaluation metrics (Hit@2, Recall@2, Precision@2, F1@2).

```bash
cd localization
python3 landmark_match_real.py
```

## Retrieval Evaluation

Located in `retrieval/`:

- **synthetic_retrieval_p_r_k.py**: Calculates Precision, Recall, and Average Precision at K for synthetic retrieval tasks.

Usage:
```bash
cd retrieval
python3 synthetic_retrieval_p_r_k.py
```

This will generate a CSV file with the results.
