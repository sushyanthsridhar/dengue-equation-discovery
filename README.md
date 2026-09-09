# Dengue Equation Discovery

Code accompanying *FunSearch and LLM-Guided Discovery of Interpretable Dengue Models for Public Health Decision Support*.

The pipeline compresses a 36-dimensional weekly climate and surveillance panel for the 28 modelled provinces of the Dominican Republic into a 10-dimensional latent state, clusters provinces by that latent state, discovers a sparse transmission-rate equation with an island-based FunSearch search augmented by a locally hosted language model, fits it hierarchically across provinces, and turns the fitted equation into forecasts, outbreak probabilities and a province risk ranking.

## Data split

Training 2015 to 2018, validation 2019, a combined training plus validation fit on 2015 to 2019 for the final model, and test years 2022 and 2023, evaluated once. 2020 and 2021 are excluded at every stage.

## Requirements

Python 3.10 or later, plus the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

The language model step calls a local [Ollama](https://ollama.com) server. Start it and pull the model before running `discover_equation.py`:

```
ollama pull mistral:latest
ollama serve
```

The endpoint and model are set at the top of `discover_equation.py` and `forecast.py` as `OLLAMA_URL` and `OLLAMA_MODEL`.

## Input data

The pipeline starts from `extended_input.csv`, a province-week panel covering 32 provinces, 52 weeks per year, 2015 to 2023. It is not included in this repository. It carries the identifier columns `province`, `year`, `month`, `week`, plus `total_cases`, `population`, and the raw and derived meteorological, large-scale climate and spatial neighbour features described in Table 1 of the paper. `normalize_input.py` turns it into `extended_input_normalized.csv`, which every later stage reads.

## Running the pipeline

Run from the repository root, in order. Each step writes the inputs the next one needs.

| Step | Script | Reads | Writes |
| --- | --- | --- | --- |
| 1 | `normalize_input.py` | `extended_input.csv` | `extended_input_normalized.csv` |
| 2 | `tune_autoencoder.py` | `extended_input_normalized.csv` | `best_config_autoencoder.txt` |
| 3 | `train_autoencoder.py` | normalized panel, config | `models/`, `latents/latent_dim10.csv` |
| 4 | `eval_autoencoder.py` | normalized panel, config, model | `latents/latent_test_dim10.csv`, `plots/` |
| 5 | `cluster.py` | `latents/latent_dim10.csv` | `province_clusters.csv` |
| 6 | `discover_equation.py` | latents, model, clusters | `fs_programs/`, `discover_equation_grid_results.csv`, `quality_scores.csv` |
| 7 | `forecast.py` | latents, model, clusters, `fs_programs/` | `outputs/` |

`autoencoder.py` is a standalone alternative to steps 2 to 4. It runs the hyperparameter search, trains the final encoder and evaluates reconstruction in a single script. `utils.py` is a shared module, imported rather than run: it loads the normalized panel and appends the circular week encoding, the two incidence lags and the week-on-week momentum term, taking the 31 stored columns to the 36 features the encoder consumes.

Step 6 is the expensive one. It sweeps a grid of FunSearch configurations, each a full search plus one round of language-model term proposal and a hierarchical fit, and selects the winning configuration on validation alone. Step 7 then takes that single winner and touches the test years exactly once.

## Files

| File | What it does |
| --- | --- |
| `normalize_input.py` | Computes per-province incidence and its rate of change, then scales every column to zero to one. |
| `utils.py` | Loads the panel and appends the time-based features every later script needs. |
| `autoencoder.py` | Defines the encoder, searches its hyperparameters, trains it and evaluates reconstruction. |
| `tune_autoencoder.py` | Selects the encoder hyperparameters by a 100-trial Bayesian search. |
| `train_autoencoder.py` | Trains the encoder and produces the latent panel used downstream. |
| `eval_autoencoder.py` | Checks the trained encoder's reconstruction quality on the held-out test years. |
| `cluster.py` | Groups provinces by latent-state similarity for the hierarchical fit. |
| `discover_equation.py` | Province quality scoring, FunSearch term discovery, one round of LLM term proposal, the hierarchical global, cluster and province fit, and test evaluation, across a grid of configurations. |
| `forecast.py` | Turns one fitted equation into forecasts, outbreak probabilities, driver attribution, province risk ranking, and ablation checks on the LLM and hierarchy components. |

Each script has a matching `*_description.txt` giving a short plain-language summary, and `best_config_autoencoder.txt` holds the selected encoder configuration.

## Notes on reproducibility

The language-model proposal step is non-deterministic, so the paper reports five independent runs and quotes the mean and standard deviation rather than a single figure. The search itself is seeded, but the proposed terms vary between runs by design.

`best_config_autoencoder.txt` as committed holds the configuration reported in the paper. Re-running `tune_autoencoder.py` overwrites it, and because the search is over a stochastic training process the recovered configuration may differ.
