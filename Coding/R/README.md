# R scripts

Run from the repository root.

The scripts cover M0 (Cox, linear covariates), M1 (Cox with natural cubic splines, three degrees of freedom), M2a (grouped complementary log-log, linear), M2b (the same link with those splines), M3 (full-sample time-varying terms, not scored out of fold), and M4 (gradient-boosted trees, stored as XGB).

1. `source("Coding/R/survival_testing.R")` fits M0–M3 and writes `oof_predictions.csv` and the test tables.
2. Leave R, run notebook `07_xgboost.ipynb` (writes `oof_predictions_xgboost.csv`; SHAP plots go to `Output/07_xgboost` and the copies in `Images/16` and `Images/17`), then return to the same R session.
3. In that session, `source("Coding/R/metrics.r")`, then `source("Coding/R/figure.r")`.

`metrics.r` writes `metrics_per_fold.csv`, `metrics_summary.csv`, and `model_comparison.csv`.

`figure.r` writes:

- `Images/09_ph_test_transforms.png`
- `Images/10_coefficients_m0.png`
- `Images/11_tie_agreement.png`
- `Images/12_calibration.png`
- `Images/13_auc_by_fold.png`

Figures 09–11 need the objects still in that R session; 12 and 13 read the CSVs.

`export_inference_tables.R` rewrites full-sample tables.

`sensitivity_m2b_28day.R` refits M2b on the 28-day feature file; the `*m2b_28day*` CSVs already in `final/` are an older 14-day dump, not that refit.

Packages: `survival` 3.8.6, `tidyverse`. `splines` is base R.
