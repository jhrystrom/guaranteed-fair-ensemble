# Guaranteed Fair Ensemble
## Installation and setup
We recommend using [`uv`](https://docs.astral.sh/uv/) to reproduce the results. See [here](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions. Note, that you can also use `pip` + `venv` if you prefer - the project is `pip`-installable. 

Once uv is installed, to install all dependencies run: 
```bash
uv sync
```

## Datasets
TODO

## Running the analysis (medical imaging)
### Training models + classifiers
To reproduce all the baselines in the medical part of the paper, run:

```bash
uv run scripts/train_sweeps.py
```

Note, this will train a lot of different models and take quite some time. 

### Fitting fair frontier
Once you've trained the models, you can fit the fair frontiers and run predictions on the test set. This can be done using the following command: 

```bash
uv run scripts/fit_fair_frotniers --dataset <dataset>
```

### Reproducing plots
To reproduce the plots, you can simply run

```bash
uv run scripts/all_plots.py
```

This will populate the `plots/` directory with plots from the paper. Currently, only some parts are supported

## NLP analysis (hate speech)
TODO
