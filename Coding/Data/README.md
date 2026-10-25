# Data

The CSV files are not in git. They are OBDeleven app logs and cannot be shared. `.gitignore` ignores `*.csv`.

## Raw

`raw/` holds three inputs:

- `activity.csv`
- `user.csv`
- `vehicle.csv`

Notebook `02_data_cleaning_and_saving.ipynb` reads those three files. It does not read `activity_1000.csv`, `user_1000.csv`, or `vehicle_1000.csv`.

## Interim

Notebook 02 splits customers into personal and professional segments and writes the cleaned tables to `interim/`. Notebook `04_preparing_data_for_modelling.ipynb` reads `interim/personal_users_filtered.csv`.

## Final

Notebook 04 writes three files to `final/`:

- `features_personal_14_day_intervals_new_features.csv`, the 14-day blocks
- `features_personal_28_day_intervals_new_features.csv`, the same customers on 28-day blocks
- `user_folds_personal_14.csv`, five folds of 1,200 customers

The 14-day file has 124,783 rows, 6,000 customers, and 2,774 events. Customers with a span under 56 days are dropped. The first row starts at day 42.

The models read the 14-day file and the fold file. The R scripts and notebook `07_xgboost.ipynb` write `oof_predictions.csv`, `oof_predictions_xgboost.csv`, `metrics_summary.csv`, `metrics_per_fold.csv`, `model_comparison.csv`, `lrt_linearity.csv`, `ph_zph_global.csv`, and `xgboost_shap_importance.csv` into `final/`. `metrics_summary.csv` stores M4 as `XGB`.

`final/archive_leftTrunc28_2026-09-15/` is an older run.
