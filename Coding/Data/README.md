# Data directory

CSV files are intentionally excluded from version control. Place source data in
`raw/`; the notebooks write derived datasets to `interim/` and `final/`.

## Raw inputs

Notebook `02_data_cleaning_and_saving.ipynb` currently reads:

- `raw/activity_1000.csv`
- `raw/vehicle_1000.csv`
- `raw/user_1000.csv`

The path configuration also supports the complete source files:

- `raw/activity.csv`
- `raw/user.csv`
- `raw/vehicle.csv`

## Generated datasets

Notebook `02` writes cleaned segment datasets to `interim/`, including
`personal_users_filtered.csv` and `professional_users_filtered.csv`.

Notebook `04` reads the filtered interim data and writes feature-engineered
person-interval datasets to `final/`:

- `features_personal_14_day_intervals_new_features.csv`
- `features_personal_28_day_intervals_new_features.csv`

The R modelling script consumes the 14-day final dataset. Do not commit raw or
generated CSV files.

## Left-truncation threshold (MIN_ACTIVITY_SPAN_DAYS)

All files currently in `final/` (features, `oof_predictions*.csv`, `metrics_*.csv`,
`coefficients_m0.csv`, `xgboost_*`, `model_comparison.csv`) were regenerated with
`MIN_ACTIVITY_SPAN_DAYS = 56` in `src/constants/cleaning.py` (early-churner /
left-truncation filter in notebook `02`) as of 2026-09-18. This is an in-place
overwrite of the pipeline outputs — filenames are unchanged from the previous
28-day-threshold run, so there is no threshold marker in the live filenames.

The previous run's (`MIN_ACTIVITY_SPAN_DAYS = 28`) aggregate metrics are archived,
clearly labelled, in `final/archive_leftTrunc28_2026-09-15/`:
`metrics_summary_leftTrunc28.csv`, `metrics_per_fold_leftTrunc28.csv`,
`model_comparison_leftTrunc28.csv`. The underlying feature/OOF CSVs for that run
were not kept (overwritten before archiving), only these summary tables.
