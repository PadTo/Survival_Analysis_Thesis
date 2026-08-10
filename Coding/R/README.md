# R modelling

`survival_modelling.R` consumes the 14-day personal-user feature dataset
created by notebook `04`. It locates the `Coding/` directory when run from the
repository root, `Coding/`, or `Coding/R/`.

Install these R packages before running the script:

- `survival`
- `tidyverse`
- `dplyr`

The Python pipeline performs cleaning, whole-dataset vehicle-year imputation,
and person-interval feature engineering. Splitting, resampling, survival
modelling, and evaluation belong in this R stage.
