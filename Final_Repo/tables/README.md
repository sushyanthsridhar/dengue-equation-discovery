# Tables

Every CSV here reproduces exactly one table from the paper. Filenames are
`sec<N>_<label>.csv` keyed to the paper's actual current section numbering
(Section 5.2 has two subsections, 5.2.1 and 5.2.2, and 5.3 has two tables,
5.3a and 5.3b).

| File | Paper table | Section |
|---|---|---|
| sec3_1_feature_groups.csv | Table 1 (feature groups) | 3.1 |
| sec3_2_autoencoder_config.csv | Table 2 (autoencoder config) | 3.2 |
| sec3_3_province_clusters.csv | Table 3 (province clusters) | 3.3 |
| sec4_4_year_splits.csv | Table 4 (year ranges per stage) | 4.4 |
| sec5_1_heterogeneity_summary.csv | Table 5 (heterogeneity) | 5.1 |
| sec5_2_1_roc_auc.csv | Table 6 (ROC-AUC) | 5.2.1 |
| sec5_2_2_error_metrics.csv | Table 7 (error magnitude) | 5.2.2 |
| sec5_3a_global_equation_representative_run.csv | Table 8 (global equation) | 5.3 |
| sec5_3b_global_equation_performance_over_time.csv | Table 9 (open-loop rollout) | 5.3 |
| sec5_4_llm_vs_random.csv | Table 10 (LLM vs random) | 5.4 |
| sec5_5_hierarchy_ablation_representative_run.csv | Table 11 (hierarchy ablation) | 5.5 |
| sec5_6_benchmark_comparison.csv | Table 12 (benchmarks) | 5.6 |
| sec5_7_latent_covariate_associations.csv | Table 13 (latent associations) | 5.7 |
| sec5_8_llm_diagnostics.csv | Table 14 (LLM diagnostics) | 5.8 |
| sec5_9_alert_precision_by_cutoff.csv | Table 15 (alert precision) | 5.9 |
| appendixA1_heterogeneity_burden_and_timing.csv | Appendix Table A.1 | Appendix A |
| appendixA1_heterogeneity_climate_response.csv | Appendix Table A.2 | Appendix A |
| appendixC_covid_years_role.csv | Appendix Table C.1 | Appendix C |

Each CSV has a matching `_readme.txt` explaining its source script and, where
relevant, the exact rows in `../results/` it was derived from.
