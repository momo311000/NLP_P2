# NLP Project 2.4

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the needed libraries:

```powershell
python -m pip install -r requirements.txt
```

## Task 1 and Task 2

Run the data exploration and preprocessing script:

```powershell
python tasks1_2.py
```

The script creates:

- `outputs/tasks1_2/` for statistics, report notes, and graphs
- `data/processed/` for cleaned data, the 10% sample, and train/validation/test splits

The input files, output folder, sample size, and split ratios are set in:

```text
tasks1_2_config.json
```

## Task 3

Run Task 1 and Task 2 first, because Task 3 uses the train/validation/test files from `data/processed/`.

```powershell
python tasks3.py
```

The Task 3 settings are in:

```text
task3_config.json
```

The script saves only the report files in:

```text
outputs/task3/
```

Important Task 3 report files:

- `task3_results.csv`
- `hyperparameter_tuning_summary.csv`
- `embedding_comparison.csv`
- `direction_comparison.csv`
- `character_vs_word_comparison.csv`
- `bleu_comparison.png`
- `token_accuracy_comparison.png`
- each experiment folder: `loss_curve.png`, `sample_predictions.csv`, `length_analysis.csv`, `sentence_feature_analysis.csv`
