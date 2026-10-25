# Testing What Is Usually Assumed: Proportional Hazards and Linearity in a Counting-Process Model of Customer Churn

Master's thesis by Tomas Padegimas, Data Science, Statistics, and Decision Analysis, Department of Computer and Systems Sciences, autumn 2026.

Churn studies usually fit the Cox model without testing its two assumptions: proportional hazards and linear covariate effects. This thesis tests both on app activity logs from OBDeleven, a vehicle diagnostics company. Churn is defined as 160 days without activity. Only private customers, the personal segment, are modelled. This repository holds the code behind the thesis.

## Research questions

1. Do the data reject proportional hazards or linearity, and how large are any departures?
2. Does relaxing linearity, or both assumptions together, improve discrimination and calibration for customers not used in fitting?

Discrimination is how well a model ranks customers who churn above those who do not. Calibration is whether the predicted churn probabilities match the observed rate.

## Data and design

1. **Three source tables.** Activity, vehicle and user tables are merged into one analysis table.
2. **Personal and professional customers.** The data hold no segment label. One is built from how each customer spreads activity across vehicles: a private owner works on one or a few cars, while a workshop scans many. Only the personal segment is modelled.
3. **Churn.** A customer churns when a gap between active days reaches 160 days, or when their last activity is at least 160 days before the end of the data. Churn is dated at the last active day before that gap. Follow-up ends 160 days before the end of the data, and customers still active then are censored.
4. **Person-interval frame.** Each customer's follow-up is cut into 14-day rows, one row per customer and interval. This is a counting-process frame, so covariates can change from one row to the next. Covariates are rolling 28- and 56-day aggregates that end at the end of the row. The row's outcome is churn in the next 14 days.
5. **Span filter.** Customers whose activity spans less than 56 days are dropped, and each customer is at risk from day 56. The first modelled row covers days 42 to 56, because its windows close on day 56.

The estimation sample has 6,000 customers, 124,783 rows and 2,774 churn events (2.22% of rows). Of 33 covariates built, 27 enter the model. The coefficients are associations, and they apply only to customers who stayed at least 56 days.

The Cox model links the covariates to the hazard, the instantaneous churn risk at time $t$:

$$\lambda(t \mid x) = \lambda_0(t)\exp(\beta' x)$$

Here $\lambda_0(t)$ is a baseline hazard shared by all customers. For two customers who differ by one unit in $x_k$, the ratio of their hazards is $\exp(\beta_k)$, the hazard ratio. It is a relative change in the hazard, not a change in probability. Proportional hazards means this ratio stays constant over follow-up. Linearity means each covariate enters as a straight line in $\beta' x$.

## Models

Proportional hazards is tested on M0 with the Schoenfeld residual test (`cox.zph`), globally and per covariate, under four time transforms (identity, rank, log and Kaplan–Meier). Linearity is tested with likelihood ratio tests of a linear model against the same model with splines.

| Model | Specification |
|---|---|
| M0 | Cox, all covariates linear |
| M1 | Cox, natural cubic splines on continuous covariates (`ns()`, three degrees of freedom) |
| M2a | Grouped complementary log-log, one intercept per interval, all covariates linear |
| M2b | M2a with the same splines as M1 |
| M3 | M0 with time-varying terms on the covariates the rank-transform test flags. Full sample only, not scored out of fold |
| M4 | Gradient-boosted trees (XGBoost) on the same rows and folds, plus the interval index in place of a baseline hazard. Stored as `XGB` in the output files |

M0 against M1, and M2a against M2b, test linearity. M0 against M2a checks the handling of tied churn times, since events fall on interval boundaries. M3 measures the size of any departure from proportional hazards. M4 makes neither assumption and serves as the performance benchmark. Coefficients and tests come from one fit of each model on the full sample.

## How the models are scored

M0, M1, M2a, M2b and M4 are refitted in five-fold cross-validation. Folds are assigned by customer, 1,200 customers each, so no model is scored on a customer it was fitted on. The folds are fixed in `user_folds_personal_14.csv`. M4 tunes its hyperparameters on log loss in an inner five-fold split, also by customer.

Each row is scored with the predicted probability of churn in its interval. Three quantities are computed on each held-out fold and averaged:

- **AUC**, the area under the ROC curve: the probability that a row with churn gets a higher predicted risk than a row without.
- **Brier score**, the mean squared error of the predicted probabilities.
- **Calibration slope**, the slope of held-out outcomes on the linear predictor. A slope below 1 means the predictions are too spread out.

## Results

Out-of-fold performance, mean across five folds (standard deviation in brackets):

| Model | AUC | Brier | Calibration slope |
|---|---|---|---|
| M0 | 0.6803 (0.0102) | 0.021536 | 0.919 (0.080) |
| M1 | 0.6822 (0.0099) | 0.021534 | 0.893 (0.080) |
| M2a | 0.6804 (0.0108) | 0.021536 | 0.934 (0.081) |
| M2b | 0.6826 (0.0106) | 0.021533 | 0.907 (0.079) |
| M4 | 0.6856 (0.0121) | 0.021517 | 0.981 (0.092) |

The global test does not reject proportional hazards (χ² from 22.6 to 28.7 on 27 df across the four transforms). Linearity is rejected under both the Cox and the complementary log-log link (χ² 109.85 and 111.61 on 36 df, p < 10⁻⁸). Splines raise out-of-fold AUC by 0.0019, while M4 raises it by 0.0053 over M0 and has the calibration slope closest to 1. Brier scores sit together near 0.0215, against 0.02174 for predicting the base rate for every row.

Plots are in [`Images/`](Images/): proportional hazards tests by transform, M0 hazard ratios, M0 against M2a coefficients, calibration, AUC by fold, and SHAP values for M4, where the interval index has the largest mean |SHAP|.

## Reproduce

Run from the repository root.

1. Create the Python environment:

   ```
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

2. In `Coding/notebooks/`, run notebook 02 up to the raw save. Notebook 01 can run once those interim files exist. Finish the filtered save in notebook 02. Then run notebook 04 to build the person-interval rows and the folds.
3. In R, `source("Coding/R/survival_testing.R")`. This fits M0 to M3, runs the tests, and writes out-of-fold predictions for M0 to M2b.
4. Run notebook 07 (M4).
5. In the same R session, `source("Coding/R/metrics.r")`, then `source("Coding/R/figure.r")`. Three figures need objects from step 3 and are skipped if the session no longer holds them.

Skip notebooks 03, 05, and 06; the reported PH test is `cox.zph` in R. `Coding/R/export_inference_tables.R` only rewrites the full-sample tables.

The thesis used R `survival` 3.8.6 with `tidyverse`, and Python `xgboost` 3.4.1. Details per script are in [`Coding/R/README.md`](Coding/R/README.md).

## Data

The activity logs belong to OBDeleven and cannot be shared. All `*.csv` files are gitignored. To run the code, place `activity.csv`, `user.csv` and `vehicle.csv` in `Coding/Data/raw/`. The models read `Coding/Data/final/features_personal_14_day_intervals_new_features.csv`. See [`Coding/Data/README.md`](Coding/Data/README.md).
