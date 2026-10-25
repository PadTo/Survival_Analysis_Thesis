# Testing What Is Usually Assumed: Proportional Hazards and Linearity in a Counting-Process Model of Customer Churn

Master's thesis by Tomas Padegimas, Data Science, Statistics, and Decision Analysis, Department of Computer and Systems Sciences, autumn 2026.

The full thesis PDF is available at [thesis.pdf](thesis.pdf).

**What.** I tested the two usual Cox assumptions—proportional hazards and linear covariate effects—on OBDeleven app activity logs for private customers, with churn defined as 160 days without activity.

**How.** I compared M0–M4 on the same counting-process rows and scored discrimination and calibration out of fold.

**Process.** Merge and clean the logs, segment personal vs professional, define churn, build 14-day person-interval rows, drop short spans, fit M0–M3 in R and M4 in Python, then score out of fold.

---

## Process

1. **Merge and clean.** Activity, vehicle and user tables are merged into one analysis table.

2. **Segment personal vs professional.** The data hold no segment label; one is built from how each customer spreads activity across vehicles (a private owner works on one or a few cars, a workshop scans many), and only the personal segment is modelled.

3. **Define churn (160 days).** A customer churns when a gap between active days reaches 160 days, or when their last activity is at least 160 days before the end of the data; churn is dated at the last active day before that gap, follow-up ends 160 days before the end of the data, and customers still active then are censored.

4. **Build 14-day person-interval rows.** Each customer's follow-up is cut into 14-day rows so covariates can change from one row to the next; covariates are rolling 28- and 56-day aggregates that end at the end of the row, and the row's outcome is churn in the next 14 days.

5. **Drop spans under 56 days (first row at day 42).** Customers whose activity spans less than 56 days are dropped; each customer is at risk from day 56, and the first modelled row covers days 42 to 56 because its windows close on day 56.

6. **Fit M0–M3 in R.** Coefficients and tests come from one fit of each model on the full sample (6,000 customers, 124,783 rows, 2,774 events; 2.22% of rows); of 33 covariates built, 27 enter the model.

7. **Fit M4 in the notebook.** Gradient-boosted trees (XGBoost) on the same rows and folds, with the interval index in place of a baseline hazard.

8. **Score out of fold.** M0, M1, M2a, M2b and M4 are refitted in five-fold cross-validation by customer (1,200 per fold, folds fixed in `user_folds_personal_14.csv`), so no model is scored on a customer it was fitted on.

The Cox model links the covariates to the hazard, the instantaneous churn risk at time $t$:

$$\lambda(t \mid x) = \lambda_0(t)\exp(\beta' x)$$

Here $\lambda_0(t)$ is a baseline hazard shared by all customers. For two customers who differ by one unit in $x_k$, the ratio of their hazards is $\exp(\beta_k)$, the hazard ratio—a relative change in the hazard, not a change in probability. Proportional hazards means this ratio stays constant over follow-up. Linearity means each covariate enters as a straight line in $\beta' x$.

## Research questions

1. Do the data reject proportional hazards or linearity, and how large are any departures?
2. Does relaxing linearity, or both assumptions together, improve discrimination and calibration for customers not used in fitting?

Discrimination is how well a model ranks customers who churn above those who do not. Calibration is whether the predicted churn probabilities match the observed rate.

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

M0 against M1, and M2a against M2b, test linearity. M0 against M2a checks the handling of tied churn times, since events fall on interval boundaries. M3 measures the size of any departure from proportional hazards. M4 makes neither assumption and serves as the performance benchmark.

Each held-out row gets a predicted probability of churn in its interval. Scoring uses AUC (the probability that a row with churn gets a higher predicted risk than a row without), Brier score (mean squared error of the predicted probabilities), and calibration slope (held-out outcomes on the linear predictor; a slope below 1 means the predictions are too spread out). M4 tunes its hyperparameters on log loss in an inner five-fold split by customer. The coefficients are associations, and they apply only to customers who stayed at least 56 days.

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

2. In `Coding/notebooks/`, run notebook 02 up to the raw save.

3. Run notebook 01 once those interim files exist.

4. Finish the filtered save in notebook 02, then run notebook 04 to build the person-interval rows and the folds.

5. In R, `source("Coding/R/survival_testing.R")` to fit M0 to M3, run the tests, and write out-of-fold predictions for M0 to M2b.

6. Run notebook 07 (M4).

7. In the same R session, `source("Coding/R/metrics.r")`, then `source("Coding/R/figure.r")`. Three figures need objects from step 5 and are skipped if the session no longer holds them.

Skip notebooks 03, 05, and 06; the reported PH test is `cox.zph` in R. `Coding/R/export_inference_tables.R` only rewrites the full-sample tables.

The thesis used R `survival` 3.8.6 with `tidyverse`, and Python `xgboost` 3.4.1. Details per script are in [`Coding/R/README.md`](Coding/R/README.md).

## Data

The activity logs belong to OBDeleven and cannot be shared. All `*.csv` files are gitignored. To run the code, place `activity.csv`, `user.csv` and `vehicle.csv` in `Coding/Data/raw/`. The models read `Coding/Data/final/features_personal_14_day_intervals_new_features.csv`. See [`Coding/Data/README.md`](Coding/Data/README.md) and [`Coding/R/README.md`](Coding/R/README.md).
