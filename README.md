# Dengue Equation Discovery

Code accompanying *FunSearch and LLM-Guided Discovery of Interpretable Dengue Models for Public Health Decision Support*.

The pipeline compresses a 36-dimensional weekly climate and surveillance panel for the 28 modelled provinces of the Dominican Republic into a 10-dimensional latent state, clusters provinces by that latent state, discovers a sparse transmission-rate equation with an island-based FunSearch search augmented by a locally hosted language model, fits it hierarchically across global, cluster, and province levels, and turns the fitted equation into forecasts, outbreak probabilities, reliability flags, and a province risk ranking.

## Repository layout

```
src/       All 20 pipeline and analysis scripts (see Files, below).
figures/   The paper's figures, named to match its \includegraphics calls.
tables/    The paper's own tables, extracted from its LaTeX, one file per
           table, named by the section that reports it (see tables/README.md
           for the section-by-table mapping).
```

Input data (`extended_input.csv`, `extended_input_normalized.csv`), trained artifacts (`models/`, `latents/`, `plots/`), and search/run outputs (`fs_programs/`, `outputs/`, `province_clusters.csv`, `best_config_autoencoder.txt`) are intermediate pipeline artifacts, produced locally by running the scripts in `src/`; they are not committed to this repository (see `.gitignore`).

## Data split

Training 2015 to 2018, validation 2019, a combined training plus validation fit on 2015 to 2019 for the final model, and test years 2022 and 2023, evaluated once. 2020 and 2021 are excluded at every stage (see the paper's Appendix on the COVID period).

## Requirements

Python 3.10 or later, plus the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

The language-model term-proposal step calls a local [Ollama](https://ollama.com) server. Start it and pull the model before running `src/discover_equation.py` or `src/forecast.py`:

```
ollama pull mistral:latest
ollama serve
```

`forecast.py` tries a configured primary model first and falls back to `mistral:latest` automatically if that model has not been pulled locally.

## Running the pipeline

Run from the repository root, in order. Each step reads the previous step's output.

| Step | Script | Produces |
| --- | --- | --- |
| 1 | `normalize_input.py` | Normalized, `[0,1]`-scaled feature panel |
| 2 | `tune_autoencoder.py` | Selected encoder hyperparameters |
| 3 | `train_autoencoder.py` | Trained encoder, latent panel |
| 4 | `eval_autoencoder.py` | Reconstruction quality on the held-out test years |
| 5 | `cluster.py` | Province cluster assignments |
| 6 | `discover_equation.py` | FunSearch-discovered term library, per province quality scores |
| 7 | `forecast.py` | LLM-guided term refinement, hierarchical global/cluster/province fit, one-step-ahead forecast evaluation |
| 8+ | `compute_*.py`, `benchmark_*.py`, `llm_*.py`, `plot_*.py` | The analyses and tables described below; each can be run once step 7 has produced a fitted model |

The language-model proposal step is non-deterministic, so the paper reports five independent runs of steps 6-7 and quotes the mean and standard deviation rather than a single figure.

## Files

### Autoencoder
| File | What it does |
| --- | --- |
| `autoencoder.py` | Defines the encoder that compresses the weekly covariates into a ten-dimensional latent state. |
| `train_autoencoder.py` | Trains the encoder and produces the latent panel used downstream. |
| `tune_autoencoder.py` | Selects the encoder hyperparameters used for training. |
| `eval_autoencoder.py` | Checks the trained encoder's reconstruction quality. |

### Clustering
| File | What it does |
| --- | --- |
| `cluster.py` | Groups provinces by latent-state similarity for the hierarchical fit. |

### Equation discovery pipeline
| File | What it does |
| --- | --- |
| `normalize_input.py` | Applies the multiplicative `[0,1]` normalization to the raw feature panel. |
| `utils.py` | Loads and prepares the panel data for the pipeline. |
| `discover_equation.py` | Runs province quality scoring and FunSearch library discovery across a grid of configurations. |
| `forecast.py` | Runs the LLM-guided term-proposal rounds, the hierarchical global/cluster/province fit, and forecast evaluation under the temporal split. |

### Decision support and analysis layer
| File | What it does |
| --- | --- |
| `compute_decision_support.py` | Turns one fitted equation into forecasts, outbreak probabilities, driver attribution, province risk ranking, and alert precision by threshold. |
| `compute_roc_auc.py` | Computes outbreak-alert ROC-AUC, pooled and quality-gated. |
| `compute_error_metrics.py` | Computes pooled and mean-per-province MAE, RMSE, and nRMSE. |
| `compute_pooled_r2.py` | Aggregates stability runs into pooled and mean-per-province R2. |
| `compute_latent_covariate_associations.py` | Correlates selected latent factors against named raw covariates. |
| `compute_horizon_skill.py` | Forecast-skill decay by horizon (supplementary analysis). |
| `llm_vs_random_matched_experiment.py` | Runs the matched-seed ablation isolating the LLM's contribution. |
| `llm_diagnostics.py` | Records LLM call runtime, memory, and term-retention diagnostics. |
| `plot_global_equation_performance.py` | Produces global equation performance over time, pooled and national-mean. |
| `benchmark_ar_fourier.py` | Runs the autoregression and seasonal-naive benchmark comparisons. |
| `benchmark_lstm_gb_sindy.py` | Runs the LSTM, Gradient Boosting, and fixed-library SINDy benchmark comparisons. |

## Notes on reproducibility

The language-model proposal step is non-deterministic: the search itself is seeded, but proposed terms vary between runs by design, which is why the paper reports means and standard deviations across five independent runs rather than a single figure. `best_config_autoencoder.txt`, once generated by `tune_autoencoder.py`, holds the configuration reported in the paper; re-running that search is over a stochastic training process and may recover a different configuration.
