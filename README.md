# Dengue Equation Discovery

Code accompanying *FunSearch and LLM-Guided Discovery of Interpretable Dengue Models for Public Health Decision Support*.

The pipeline compresses a 36-dimensional weekly climate and surveillance panel for the 28 modelled provinces of the Dominican Republic into a 10-dimensional latent state, clusters provinces by that latent state, discovers a sparse transmission-rate equation with an island-based FunSearch search augmented by a locally hosted language model, fits it hierarchically across provinces, and turns the fitted equation into forecasts, outbreak probabilities, reliability flags, and a province risk ranking.

## Repository layout

```
src/       All 17 pipeline and analysis scripts (see Files, below).
results/   Raw CSV outputs the scripts produce, which every table in the
           paper and in csv/ is built from.
csv/       The paper's own tables, extracted from its LaTeX, one file per
           table, named by the section that reports it.
figures/   The paper's figures, named to match its \includegraphics calls.
readmes/   One plain-language description per script (src/) and per table
           (csv/), matched by filename.
```

`extended_input.csv`, `extended_input_normalized.csv`, `best_config_autoencoder.txt`, `province_clusters.csv`, `models/`, `latents/`, `plots/`, `fs_programs/`, and `outputs/` are intermediate pipeline artifacts, generated at repo root by the scripts in `src/` when you run them; they are not committed (see `.gitignore`) except where noted below.

## Data split

Training 2015 to 2018, validation 2019, a combined training plus validation fit on 2015 to 2019 for the final model, and test years 2022 and 2023, evaluated once. 2020 and 2021 are excluded at every stage.

## Requirements

Python 3.10 or later, plus the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

The language model step calls a local [Ollama](https://ollama.com) server. Start it and pull the model before running `src/discover_equation.py`:

```
ollama pull mistral:latest
ollama serve
```

The endpoint and model are set at the top of `src/discover_equation.py` and `src/forecast.py` as `OLLAMA_URL` and `OLLAMA_MODEL`.

## Input data

The pipeline starts from `extended_input.csv`, a province-week panel covering 32 provinces, 52 weeks per year, 2015 to 2023. It is not included in this repository. It carries the identifier columns `province`, `year`, `month`, `week`, plus `total_cases`, `population`, and the raw and derived meteorological, large-scale climate and spatial neighbour features described in `csv/sec3_1_feature_groups.csv` (Table 1 of the paper). `src/normalize_input.py` turns it into `extended_input_normalized.csv`, which every later stage reads.

Place `extended_input.csv` at the repository root before running anything in `src/` — every script resolves its inputs and outputs relative to the repo root, not to `src/`, regardless of which directory you run it from.

## Running the pipeline

Run from the repository root, in order. Each step writes the inputs the next one needs, all at repo root (not inside `src/`).

| Step | Script | Reads | Writes |
| --- | --- | --- | --- |
| 1 | `src/normalize_input.py` | `extended_input.csv` | `extended_input_normalized.csv` |
| 2 | `src/tune_autoencoder.py` | `extended_input_normalized.csv` | `best_config_autoencoder.txt` |
| 3 | `src/train_autoencoder.py` | normalized panel, config | `models/`, `latents/latent_dim10.csv` |
| 4 | `src/eval_autoencoder.py` | normalized panel, config, model | `latents/latent_test_dim10.csv`, `plots/` |
| 5 | `src/cluster.py` | `latents/latent_dim10.csv` | `province_clusters.csv` |
| 6 | `src/discover_equation.py` | latents, model, clusters | `fs_programs/`, `discover_equation_grid_results.csv`, `quality_scores.csv` |
| 7 | `src/forecast.py` | latents, model, clusters, `fs_programs/` | `outputs/` |

Example, from the repository root:

```
python3 src/normalize_input.py
python3 src/tune_autoencoder.py
...
python3 src/forecast.py
```

`src/autoencoder.py` is a standalone alternative to steps 2 to 4. It runs the hyperparameter search, trains the final encoder and evaluates reconstruction in a single script. `src/utils.py` is a shared module, imported rather than run: it loads the normalized panel and appends the circular week encoding, the two incidence lags and the week-on-week momentum term, taking the 31 stored columns to the 36 features the encoder consumes.

Step 6 is the expensive one. It sweeps a grid of FunSearch configurations, each a full search plus one round of language-model term proposal and a hierarchical fit, and selects the winning configuration on validation alone. Step 7 then takes that single winner and touches the test years exactly once.

## Analysis and benchmark scripts

These run after step 7, against the model `src/forecast.py` has already fit, and produce every table in `csv/` and every number in the paper's Computational Study section. All of them write into `results/` at the repo root.

| Script | What it does |
| --- | --- |
| `src/compute_pooled_r2.py` | Refits the winning model 5 independent times (the stability sweep) and computes pooled and mean-per-province test $R^2$ for each run. |
| `src/compute_latent_covariate_associations.py` | Correlates each LLM-selected latent factor against named raw covariates, training years only. |
| `src/compute_decision_support.py` | Builds the public-health decision-support layer: outbreak probabilities, reliability flags, risk ranking, and alert precision by cutoff. |
| `src/llm_vs_random_matched_experiment.py` | Runs the LLM proposer and a matched random proposer side by side under an identical budget, across 5 seeds. |
| `src/llm_diagnostics.py` | Logs LLM call runtime, memory, and suggestion-retention diagnostics across repeated rounds. |
| `src/plot_global_equation_performance.py` | Scores the fitted equation's forecasts over the full 2015-2023 span, pooled and national-mean. |
| `src/benchmark_ar_fourier.py` | Fits the seasonal-naive, persistence, AR-only, and AR+Fourier baselines. |
| `src/benchmark_lstm_gb_sindy.py` | Fits the LSTM, Gradient Boosting, and fixed-library SINDy baselines. |

## Files

| File | What it does |
| --- | --- |
| `src/normalize_input.py` | Computes per-province incidence and its rate of change, then scales every column to zero to one. |
| `src/utils.py` | Loads the panel and appends the time-based features every later script needs. |
| `src/autoencoder.py` | Defines the encoder, searches its hyperparameters, trains it and evaluates reconstruction. |
| `src/tune_autoencoder.py` | Selects the encoder hyperparameters by a 100-trial Bayesian search. |
| `src/train_autoencoder.py` | Trains the encoder and produces the latent panel used downstream. |
| `src/eval_autoencoder.py` | Checks the trained encoder's reconstruction quality on the held-out test years. |
| `src/cluster.py` | Groups provinces by latent-state similarity for the hierarchical fit. |
| `src/discover_equation.py` | Province quality scoring and FunSearch term discovery across a grid of configurations. |
| `src/forecast.py` | LLM-guided term proposal, the hierarchical global/cluster/province fit, and forecast evaluation under the temporal split. |

Every script above, plus the eight analysis and benchmark scripts, has a matching plain-language description in `readmes/`. `best_config_autoencoder.txt` holds the selected encoder configuration.

## Notes on reproducibility

The language-model proposal step is non-deterministic, so the paper reports five independent runs and quotes the mean and standard deviation rather than a single figure. The search itself is seeded, but the proposed terms vary between runs by design.

`best_config_autoencoder.txt` as committed holds the configuration reported in the paper. Re-running `src/tune_autoencoder.py` overwrites it, and because the search is over a stochastic training process the recovered configuration may differ.

Every script in `src/` resolves its data and output paths relative to the repository root (two directories up from its own location), not to its current working directory, so they can be run as `python3 src/<script>.py` from the repo root regardless of shell `cwd`.
