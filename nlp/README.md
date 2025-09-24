# Installation and Usage Guide

## Quick Start
To reproduce the raw outputs from the paper (assuming uv installed), run:

```bash
uv run scripts/run_ensemble.py
```

This will install the dependencies and run the same analysis as reported in the paper. Code for reproducing the plots will be run before camera ready.

Note, you can specify the output directory by copying (and editing) `.sample-env` to `.env` and editing `OUTPUT_DIR`:

```bash
cp .sample-env .env
```

## What's Included (apart from code)
- **data/split/Polish/**: Pre-split Polish dataset (used in paper)
  - train.tsv 
  - test.tsv  
  - valid.tsv 
- **pyproject.toml**: Python dependencies

## Expected Output
The script will create output files in the current directory with model performance and fairness metrics.

## System Requirements

- Python 3.10+ (tested on Python 3.12)
- CUDA GPU recommended for training
- Minimum 8GB RAM
- ~2GB disk space for outputs# ensemble_fair_dev
