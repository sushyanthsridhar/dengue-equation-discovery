# Dengue Equation Discovery — Final Repo (post-referee-report version)

Code accompanying *FunSearch and LLM-Guided Discovery of Interpretable Dengue Models for Public Health Decision Support*.

This folder is the release version of the pipeline as revised in response to the referee report. It supersedes the original `discover_equation.py` / `forecast.py` at the repository root: the validation split no longer leaks into the FunSearch inner search (see "What changed" below), and the LLM term-proposal prompt has been substantially strengthened. Use the two scripts in this folder, not the ones in the Git_Repo root, for anything going forward.

The pipeline compresses a 36-dimensional weekly climate and surveillance panel for the 28 modelled provinces of the Dominican Republic into a 10-dimensional latent state, clusters provinces by that latent state, discovers a sparse transmission-rate equation with an island-based FunSearch search augmented by a locally hosted language model, fits it hierarchically across provinces, and turns the fitted equation into forecasts, outbreak probabilities and a province risk ranking.

## What changed relative to the original scripts

1. **Temporal validation leak fixed.** `discover_equation.py`'s inner FunSearch validation year was corrected to `FS_INNER_VAL = [2018]` (previously drawn from a year that overlapped the outer validation window), so the equation search, the outer model selection, and the final test evaluation now use a strictly ordered, non-overlapping split: train on 2015-2018, validate on 2019, refit on 2015-2019, test once on 2022-2023.
2. **Richer, grounded LLM prompt** in `forecast.py` (and, more conservatively, in `discover_equation.py`):
   - The round-over-round feedback loop actually runs (`MAX_LLM_ROUNDS = 4` in `forecast.py`; kept at 1 inside the 72-cell grid search in `discover_equation.py` for runtime reasons).
   - Latent dimensions z1-z10 are now described to the model in terms of their real-world meaning, computed from their Pearson correlation against the actual covariate panel (temperature, rainfall, humidity, ENSO indices, school calendar, etc.) on the training years only.
   - Each LLM-proposed term is scored against the pooled training residual before being added to the candidate pool, and only terms clearing a minimum absolute correlation are kept (`score_llm_candidates`).
   - The prompt includes a concrete worked example of a power/nonlinear term (`y_lag1 ** 2`) to steer the model away from purely linear suggestions.
   - `query_llm` now tries a configured primary model first and automatically falls back to `mistral:latest` if the primary is not available locally, logging which model actually answered each call.
3. A small `random.choices` zero-weight edge case in FunSearch's parent-selection step (all-scoreless island) is now guarded with a `random.choice` fallback.

See `results_report.txt` in `results/` for the full quantitative comparison (before/after prompt fix, ablations, per-run global equations, per-province results) that these changes produced.

## Headline benchmark result (see `results_report.txt` section 11 for the full writeup)

One-week-ahead test R2, pooled / mean per province:

| Model | Pooled R2 | Mean per-province R2 |
| --- | --- | --- |
| Gradient Boosting (black box) | 0.506 | 0.437 |
| This pipeline (tau9_lasso_spread, mean of 5 runs) | n/a | 0.440 |
| LSTM (black box) | 0.433 | 0.344 |
| Fixed-library, non-LLM, non-hierarchical pooled SINDy | 0.409 | 0.424 |
| Persistence baseline | 0.484 | 0.399 |

Gradient Boosting attains higher pooled accuracy and ties the sparse model on mean per-province accuracy, but with no inspectable equation. The sparse model trades a small amount of raw accuracy for an explicit, per-province-decomposable equation; the fixed-library ablation shows most of that accuracy comes from the base library and hierarchical fit itself, with FunSearch discovery and LLM-proposed terms adding a smaller additional gain on top.

## Matched LLM-versus-random-proposer result (see `results_report.txt` section 12)

The referee report's decisive experiment, run over 5 seeds with the proposal budget matched round by round between an LLM proposer and a random proposer:

| Metric | LLM proposer | Random proposer |
| --- | --- | --- |
| Validation R2 | 0.520 (+/- 0.014) | 0.477 (+/- 0.023) |
| Final test R2 | 0.443 (+/- 0.017) | 0.353 (+/- 0.033) |
| Valid proposals | 28.2 | 26.2 |
| Parse/validation failures | 3.4 | 5.4 |
| Retained proposals | 13.6 | 14.8 |
| Evaluations to best validation | 7.8 | 7.8 |

The LLM proposer beats the random proposer on both R2 metrics, consistently across all 5 seeds, while retaining fewer terms and with comparable or fewer parse failures -- the win is not explained by proposing more or having more survive by chance. This is the paper's justification for using the LLM as the term-proposal mechanism rather than a random baseline.

## Conventional autoregression + Fourier baseline (see `results_report.txt` section 13)

| Variant | Val R2 | Test R2 (pooled) | Test R2 (mean per province) |
| --- | --- | --- | --- |
| ar_only (own lags only) | 0.669 | 0.480 | 0.426 |
| ar_fourier (lags + seasonal terms) | 0.669 | 0.487 | 0.438 |

A plain linear autoregression with fixed seasonal terms lands within about 0.002 of the full pipeline's mean-per-province test R2 (0.4404). This is expected and reported deliberately, not avoided: it shows most one-week-ahead predictability in this system comes from dengue's own persistence and seasonality, and it sets the realistic accuracy ceiling the paper's other, controlled comparisons (the matched LLM-versus-random experiment, and the hierarchy ablation) should be read against -- see `results_report.txt` section 13 for the full framing and suggested paper language.

## LLM runtime and memory diagnostics (see `results_report.txt` section 14)

| Phase | Mean time | Range |
| --- | --- | --- |
| EM fit + evaluation | 3.78 s | 3.02 - 5.66 s |
| Ollama call | 38.51 s | 30.64 - 45.71 s |
| Prompt construction + screening | 0.035 s | negligible |
| **Total per round** | **42.33 s** | **34.50 - 50.91 s** |

The Ollama call accounts for about 91% of per-round wall time; EM refitting and screening are comparatively free. Process memory stayed flat at ~390MB RSS across every phase and every round in all 3 seeds tested -- no memory growth or leak. Mean 7.9 suggestions proposed per round, 7.25 valid, 3.4 retained after screening, consistent with the counts reported in the matched LLM-versus-random experiment above.

## Global equation performance figure (see `results_report.txt` section 15)

`figures/global_equation_performance.png` -- predicted vs. actual weekly incidence, averaged across all 28 provinces:

| Split | Pooled R2 (per province-week) | National-mean R2 |
| --- | --- | --- |
| Fit, 2015-2019 (in-sample) | 0.704 | 0.969 |
| Test, 2022-2023 (held out) | 0.499 | 0.670 |

The model tracks in-sample seasonal peaks closely and follows the timing and shape of out-of-sample outbreaks well, with some underestimation of peak magnitude during the sharp late-2023 outbreak (see section 15 for the full write-up and a caveat on the final data point). The higher national-mean R2 relative to the pooled per-province-week R2 reported elsewhere reflects the smoothing effect of averaging across 28 provinces, not a different result.

## Data split

Training 2015 to 2018, validation 2019, a combined training plus validation fit on 2015 to 2019 for the final model, and test years 2022 and 2023, evaluated once. 2020 and 2021 are excluded at every stage.

## Requirements

Python 3.10 or later, plus the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

The language model step calls a local [Ollama](https://ollama.com) server. Start it and pull the model(s) before running `discover_equation.py` or `forecast.py`:

```
ollama pull mistral:latest
ollama serve
```

`forecast.py` will also try a stronger primary model first (see `OLLAMA_MODEL_PRIMARY` near the top of the file) and fall back to `mistral:latest` automatically if that model has not been pulled locally. Pull whatever model you set as primary with `ollama pull <tag>` first, and confirm the exact tag with `ollama list`.

## Input data

This folder does not carry the data files themselves — `extended_input_normalized.csv`, `latents/`, `models/`, and `province_clusters.csv` are expected one directory up, in the `Git_Repo` root, exactly where the existing pipeline already produces and reads them (see the root `.gitignore`, which deliberately excludes all of these from version control). `HERE`/`PARENT` path resolution in both scripts already points there automatically, since this folder sits at the same depth as `referee_report/` inside `Git_Repo`. To run this pipeline standalone elsewhere, first run steps 1 to 5 below from the repository root (or copy their outputs alongside this folder's parent directory).

## Running the pipeline

Run from the repository root, in order. Each step writes the inputs the next one needs.

| Step | Script | Reads | Writes |
| --- | --- | --- | --- |
| 1 | `normalize_input.py` | `extended_input.csv` | `extended_input_normalized.csv` |
| 2 | `tune_autoencoder.py` | `extended_input_normalized.csv` | `best_config_autoencoder.txt` |
| 3 | `train_autoencoder.py` | normalized panel, config | `models/`, `latents/latent_dim10.csv` |
| 4 | `eval_autoencoder.py` | normalized panel, config, model | `latents/latent_test_dim10.csv`, `plots/` |
| 5 | `cluster.py` | `latents/latent_dim10.csv` | `province_clusters.csv` |
| 6 | `discover_equation.py` (this folder) | latents, model, clusters | `fs_programs/`, `discover_equation_grid_results.csv`, `quality_scores.csv` |
| 7 | `forecast.py` (this folder) | latents, model, clusters, `fs_programs/` | `outputs/` |
| 8 | `benchmark_lstm_gb_sindy.py` (this folder) | latents, model, clusters, `fs_programs/` (run after step 7) | `outputs/lstm_gb_benchmark.csv`, `outputs/fixed_library_sindy_pooled.csv`, and related per-province/equation files |
| 9 | `llm_vs_random_matched_experiment.py` (this folder) | latents, model, clusters, `fs_programs/` (run after step 7) | `outputs/llm_vs_random_matched_per_seed.csv`, `outputs/llm_vs_random_matched_summary.csv` |
| 10 | `benchmark_ar_fourier.py` (this folder) | latents, model, clusters, `fs_programs/` (run after step 7) | `outputs/ar_fourier_benchmark.csv`, `outputs/ar_only_per_province.csv`, `outputs/ar_fourier_per_province.csv` |
| 11 | `llm_diagnostics.py` (this folder) | latents, model, clusters, `fs_programs/` (run after step 7) | `outputs/llm_diagnostics_per_round.csv`, `outputs/llm_diagnostics_summary.csv` |
| 12 | `plot_global_equation_performance.py` (this folder) | latents, model, clusters, `fs_programs/` (run after step 7) | `figures/global_equation_performance.png`, `results/global_equation_performance_per_province.csv`, `results/global_equation_performance_national_mean.csv` |

Step 7's `forecast.py` is set to load the already-discovered winning FunSearch program (`fs_programs/tau9_lasso_spread_best.json`, included in this folder) via `USE_SAVED_FS_PROGRAM = True`, so it can be run directly without re-running the full step-6 grid search. Set `USE_SAVED_FS_PROGRAM = False` to force a fresh search instead.

Step 8's `benchmark_lstm_gb_sindy.py` adds the black-box (LSTM, Gradient Boosting) and fixed-library SINDy baselines requested in the referee report (Section 3.3 / Table 10, and the "strongly recommended" fixed-library item under Section 5). It imports `forecast.py` as a module, so it reuses the exact same data, scaler, and temporal split; run it any time after step 7 has produced `fs_programs/tau9_lasso_spread_best.json`.

Step 9's `llm_vs_random_matched_experiment.py` is the referee report's decisive, matched LLM-versus-random-proposer experiment (Section 2.3 / 3.1 / 4.4, Table 9) -- the third of the referee's three "true submission blockers." It runs 5 seeds, advancing an LLM-proposer arm and a random-proposer arm in lockstep round by round from the same starting structure, with the random arm's proposal count matched to the LLM's each round, and reports validation R2, test R2, valid proposals, parse failures, retained proposals, and evaluations-to-best-validation, exactly the table the referee's report requests. It also imports `forecast.py` as a module and can be run any time after step 7.

Step 11's `llm_diagnostics.py` covers the remaining "strongly recommended if inexpensive" Section 5 item -- LLM runtime and memory diagnostics -- by instrumenting the same round loop with wall-clock timers around EM fit+evaluation, prompt construction, the Ollama call, and candidate screening, plus process RSS memory after each phase, across 3 seeds. Parse-failure and acceptance counts are already covered by step 9's matched experiment; this script adds only the timing and memory pieces. It also imports `forecast.py` as a module and can be run any time after step 7.

Step 12's `plot_global_equation_performance.py` is a paper figure, not a referee-report item: it refits the actual winning model and plots national-mean (across all 28 provinces) predicted vs. actual weekly incidence, for both the 2015-2019 fit period and the 2022-2023 held-out test period, with R2 annotated for each. Like step 11, it reruns a live LLM round loop, so it takes a few minutes and calls Ollama.

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
| `discover_equation.py` | Province quality scoring, FunSearch term discovery (corrected inner validation split), LLM term proposal with latent grounding and residual scoring, the hierarchical global, cluster and province fit, and test evaluation, across a grid of configurations. |
| `forecast.py` | Turns one fitted equation into forecasts, outbreak probabilities, driver attribution, province risk ranking, and ablation checks on the LLM and hierarchy components. Runs the multi-round, grounded LLM prompt loop. |
| `benchmark_lstm_gb_sindy.py` | Adds an LSTM benchmark, a Gradient Boosting benchmark, and a fixed-library non-LLM non-hierarchical pooled SINDy baseline, all evaluated one-week-ahead on the same test split as the sparse model, for the Table 10 comparison the referee report asks for. |
| `llm_vs_random_matched_experiment.py` | The matched LLM-versus-random-proposer experiment (referee Section 2.3 / 3.1 / 4.4, Table 9): 5 seeds, budget-matched round by round, reporting validation/test R2, valid proposals, parse failures, retained proposals, and evaluations-to-best-validation for both arms. |
| `benchmark_ar_fourier.py` | The conventional autoregression + Fourier/dynamic-harmonic baseline the referee recommended under Section 5: a pooled linear model of next-week incidence from its own lags alone, and from lags plus fixed seasonal terms, with no latent representation, no LLM, no hierarchy. |
| `llm_diagnostics.py` | Per-round wall-clock timing (EM fit+eval, prompt build, Ollama call, screening) and process RSS memory for the LLM-guided term-proposal loop, across 3 seeds -- the runtime/memory piece of the referee's Section 5 LLM diagnostics recommendation. |
| `plot_global_equation_performance.py` | Paper figure: refits the winning model and plots national-mean predicted vs. actual weekly incidence across all 28 provinces, for the 2015-2019 fit period and the 2022-2023 held-out test period, with R2 annotated. |
| `fs_programs/tau9_lasso_spread_best.json` | The winning FunSearch program found under the corrected split, used by `forecast.py` when `USE_SAVED_FS_PROGRAM = True`. |
| `fs_programs/tau9_lasso_spread_history.csv` | Full FunSearch search history for the winning grid cell. |
| `results/` | Key output tables from the runs reported in `results_report.txt`: grid search results, stability runs, ablations, the five per-run global equations, per-province stability, model test evaluation, the LSTM/Gradient Boosting/fixed-library SINDy benchmark tables, the matched LLM-versus-random-proposer experiment tables, and the autoregression + Fourier baseline tables. |
| `results_report.txt` | Detailed writeup of the results above, prepared for rewriting the paper's results section, including an explicit note on which Ollama model actually answered each LLM call in the reported runs. |

Each pipeline script also has a matching `*_description.txt` giving a short plain-language summary, and `best_config_autoencoder.txt` holds the selected encoder configuration.

## Notes on reproducibility

The language-model proposal step is non-deterministic, so the paper reports five independent runs and quotes the mean and standard deviation rather than a single figure. The search itself is seeded, but the proposed terms vary between runs by design.

`best_config_autoencoder.txt` as committed holds the configuration reported in the paper. Re-running `tune_autoencoder.py` overwrites it, and because the search is over a stochastic training process the recovered configuration may differ.
