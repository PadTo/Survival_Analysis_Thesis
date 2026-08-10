# Data directory

CSV files are intentionally excluded from version control. Place source data in
`raw/`; the notebooks write derived datasets to `interim/` and `final/`.

## Raw inputs

Notebook `02_data_cleaning_and_saving.ipynb` currently reads:

- `raw/activity_1000.csv`
- `raw/vehicle_1000.csv`

The path configuration also supports the complete source files:

- `raw/activity.csv`
- `raw/user.csv`
- `raw/vehicle.csv`
- `raw/user_1000.csv`

## Generated datasets

Notebook `02` writes cleaned segment datasets to `interim/`, including
`personal_users_filtered.csv` and `professional_users_filtered.csv`.

Notebook `04` reads the filtered interim data and writes feature-engineered
person-interval datasets to `final/`:

- `features_personal_14_day_intervals_new_features.csv`
- `features_personal_28_day_intervals_new_features.csv`

The validation notebook and R modelling script consume the 14-day final
dataset. Do not commit raw or generated CSV files.
