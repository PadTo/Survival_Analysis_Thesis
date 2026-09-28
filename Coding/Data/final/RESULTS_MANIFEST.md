# Results manifest — 56-day left-truncation, entry-time + Cox boundary corrected (2026-09-18/19, v3)

All paths below are relative to `Coding/Data/final/` unless stated otherwise.

Plain-language write-up of the two bugs (for thesis LLMs):
`Coding/docs/entry_time_and_cox_baseline.md`.

## v2 correction: immortal-time / entry-point bug (same day)

The first pass of this run (v1) filtered users by activity span
(`MIN_ACTIVITY_SPAN_DAYS = 56`) but still exported each retained user's
interval rows starting at their own day 0. Since span >= 56 is required for
inclusion, every retained user's rows before duration 42 (14-day grid) /
duration 28 (28-day grid) are **structurally guaranteed to carry no event**
(the earliest possible label, via the features-at-t/event-at-t+1 shift, lands
on the row starting at `MIN_ACTIVITY_SPAN_DAYS - interval_in_days`) — i.e.
immortal time, plus those same rows had zero-filled 28–56-day lookback/drift
features for want of pre-day-0 history. Fixed in `src/data_processing.py`
(`apply_feature_engineering`): the exported frame now starts at
`interval_start = MIN_ACTIVITY_SPAN_DAYS - interval_in_days` per user (day 42
for 14-day intervals, day 28 for 28-day intervals); the dropped early rows are
still used internally to compute that first row's lookback windows. No events
were lost (2,774 events unchanged) — 18,000 zero-information rows were.

Effect on headline metrics (v1 → v2, mean AUC): M0 0.721→0.654, M1
0.723→0.655, M2a 0.721→0.680, M2b 0.723→0.683, XGB 0.703→0.673. This is
expected: the removed rows were "free", perfectly-predictable negatives that
inflated AUC.

## v3 correction: Cox baseline-hazard boundary bug (same day)

v2 exposed a second, unrelated bug that only bites Cox (M0/M1), not cloglog
(M2a/M2b): `cox_hazard()` in `survival_testing.R` built `Lambda0 <-
approxfun(bh$time, bh$hazard, rule = 2)` directly from `basehaz()`'s table,
which starts at the *first observed event time* (56, since entry is now 42),
not at 0. `rule = 2` flat-extrapolates any query below that to the table's
first (already-jumped) value, so `Lambda0(56) - Lambda0(42)` collapsed to
exactly 0 for every row in the entry interval -- which, post-v2, is the
single highest-hazard interval in the dataset (interval_start = 42: 184
events / 6,000 rows = 3.07%, versus a 2.22% baseline). M0/M1 predicted
`h = 0` for all 6,000 of those rows, killing their rank ordering pre-fix and
depressing calibration when patched with a linear ramp from `(0,0)` instead
(tried and rejected: slope dropped to ~0.78). Fixed by making
`Lambda0(t) = 0` exactly for any `t` before the first observed event
(true by the Breslow estimator's own definition), leaving every other row's
interpolation/extrapolation untouched. Verified: `mean_h` at the entry
interval is now 0.0301 against an observed rate of 0.0300 (was exactly 0.0);
the only remaining `h == 0` rows (1,983 of 124,783) sit at duration >= 700,
the sparse right tail, none at the entry point.

Effect on headline metrics (v2 → v3, mean AUC): M0 0.654→0.680, M1
0.655→0.682; M2a/M2b/XGB unaffected (cloglog and XGBoost never went through
`cox_hazard()`). **v3 numbers below are the ones to use** — all five models
now cluster at AUC 0.673-0.683 with calibration slopes 0.89-1.00.

Cohort: **6,000 personal users**, 5 folds of 1,200 users each
(`user_folds_personal_14.csv`). 14-day interval frame: 124,783 rows, 2,774
events (2.22%). Verified: no NaNs in any `h`/`beta` column, SHAP array shape
matches OOF row/feature counts, no error cells in the notebook, only benign
spline-boundary-knot warnings from R.

## 1. Main model comparison (M0, M1, M2a, M2b, XGB) — 14-day intervals

- `metrics_summary.csv` — one row per model: mean/SD AUC, Brier, calibration
  slope across the 5 outer folds. **Primary table for the results chapter.**
- `metrics_per_fold.csv` — same metrics, one row per (model, fold).
- `model_comparison.csv` — wide view, one row per metric, one column per model.
- `oof_predictions.csv` — row-level out-of-fold hazards for M0/M1/M2a/M2b
  (`model, fold, user_id, d, h`).
- `oof_predictions_xgboost.csv` — same schema for XGB (nested CV, nested
  nested inner GroupKFold(5) hyperparameter search by logloss).
- `xgboost_best_params_per_fold.csv` — winning XGBoost hyperparameters +
  mean inner-fold logloss, one row per outer fold.

## 2. Cox / cloglog inference on the full (56-day-truncated) sample

- `coefficients_m0.csv` — M0 hazard ratios per SD, 95% CI, p-values (forest
  plot data, `Images/10_coefficients_m0.png`).
- `m3_time_varying_drift.csv` — linear-in-time `tt()` drift check for the
  3 covariates flagged by the rank-transform PH test.
- `m3_time_varying_drift_log.csv` — same check, log-in-time functional form
  (robustness of M3).

## 3. XGBoost SHAP (56-day truncation)

- `xgboost_shap_values_oof.npy` — out-of-fold SHAP values, shape
  `(124783, 27)`, row-aligned with `oof_predictions_xgboost.csv`.
- `xgboost_shap_importance.csv` — mean |SHAP| per feature, sorted descending.
  Top driver: `days_since_last_new_vehicle`, then `vehicle_age_sd_overall`,
  `sd_gap_0_56_days`.
- Figures: `Output/07_xgboost/01_shap_importance.png` (bar chart) and
  `Output/07_xgboost/02_shap_summary.png` (beeswarm); mirrored into
  `Images/16_shap_importance.png` and `Images/17_shap_summary.png`.

## 4. Sensitivity check — 28-day intervals, cloglog + restricted cubic splines (M2b)

- `metrics_summary_m2b_28day.csv` — mean/SD AUC, Brier, calibration slope for
  M2b refit on 28-day intervals (63,927 rows, same 6,000 users/folds).
- `metrics_per_fold_m2b_28day.csv` — per-fold breakdown of the above.
- `metrics_sensitivity_m2b_14_vs_28.csv` — side-by-side M2b_14d vs M2b_28d
  summary row (**use this one for a direct interval-width comparison**).
- `oof_predictions_m2b_28day.csv` — row-level OOF hazards for the 28-day M2b.

## 4b. Full-sample inference tables (from `export_inference_tables.R`)

- `ph_zph_global.csv` — `cox.zph` GLOBAL test, all 4 transforms (χ², df, p).
- `ph_zph_covariates.csv` — per-covariate `cox.zph` table, all 4 transforms.
- `ph_zph_smallest_p.csv` — 4 smallest rank-transform covariate p-values.
- `lrt_linearity.csv` — M0-vs-M1 (Cox) and M2a-vs-M2b (cloglog) LRTs.
- `m0_m2a_coefficients.csv` / `m0_m2a_agreement.csv` — per-term comparison and
  summary (correlation, max/mean |rel. diff|) behind `11_tie_agreement.png`.
  Note: PH-test and M0/M2a-agreement numbers are essentially unchanged by the
  v2 entry-time fix — expected, since Cox's partial likelihood (and its
  Schoenfeld residuals) is invariant to removing risk-set rows from a window
  with zero events in it. `lrt_linearity.csv`'s df shifted 38→36 because one
  covariate's continuous/spline-eligibility flipped under the smaller row
  count (`spline_safe` check in `survival_testing.R`).

## 5. Figures (all regenerated under the 56-day threshold)

`Images/09_ph_test_transforms.png`, `10_coefficients_m0.png`,
`11_tie_agreement.png`, `12_calibration.png`, `13_auc_by_fold.png`,
`16_shap_importance.png`, `17_shap_summary.png`.

## 6. Prior run for reference (28-day threshold, superseded)

Archived in `archive_leftTrunc28_2026-09-15/`: `metrics_summary_leftTrunc28.csv`,
`metrics_per_fold_leftTrunc28.csv`, `model_comparison_leftTrunc28.csv`
(6,458 users). Feature/OOF-level CSVs for that run were not kept — only these
three summary tables survived the overwrite.
