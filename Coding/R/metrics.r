suppressPackageStartupMessages({
    library(tidyverse)
})

final_dir <- file.path(coding_root, "Data", "final")

# Cox / grouped-cloglog OOF predictions (M0, M1, M2a, M2b) come from
# survival_testing.R; XGBoost OOF predictions (XGB) come from notebook 07.
# Both files share the same (model, fold, user_id, d, h) schema, so every
# metric below is computed identically across all five models.
MODEL_LEVELS <- c("M0", "M1", "M2a", "M2b", "XGB")

oof <- bind_rows(
    read_csv(file.path(final_dir, "oof_predictions.csv"), show_col_types = FALSE),
    read_csv(file.path(final_dir, "oof_predictions_xgboost.csv"), show_col_types = FALSE)
) |>
    mutate(
        d = as.integer(d),
        model = factor(model, levels = MODEL_LEVELS)
    )

# ---- AUC without extra packages (Mann-Whitney) ---------------------

auc_fast <- function(d, p) {
    n1 <- sum(d == 1)
    n0 <- sum(d == 0)
    if (n1 == 0 || n0 == 0) {
        return(NA_real_)
    }
    r <- rank(p)
    (sum(r[d == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# ---- calibration slope on the cloglog scale ------------------------
# slope of 1 means correctly scaled, below 1 too extreme, above 1 too flat.
# rows with h exactly 0 or 1 have an infinite predictor and are dropped.

cal_slope <- function(d, p) {
    ok <- p > 0 & p < 1
    if (sum(ok) < 50) {
        return(NA_real_)
    }
    x <- log(-log(1 - p[ok]))
    fit <- glm(d[ok] ~ x, family = binomial(link = "cloglog"))
    unname(coef(fit)[2])
}

# ---- per fold, then summarised, identically for every model --------

per_fold <- oof |>
    group_by(model, fold) |>
    summarise(
        n = n(),
        events = sum(d),
        auc = auc_fast(d, h),
        brier = mean((h - d)^2),
        slope = cal_slope(d, h),
        .groups = "drop"
    )

per_fold |> print(n = Inf)

summary_table <- per_fold |>
    group_by(model) |>
    summarise(
        auc_mean = mean(auc), auc_sd = sd(auc),
        brier_mean = mean(brier), brier_sd = sd(brier),
        slope_mean = mean(slope), slope_sd = sd(slope),
        .groups = "drop"
    ) |>
    arrange(desc(auc_mean)) |>
    mutate(rank_by_auc = row_number(), .after = model)

summary_table |> print(width = Inf)

write_csv(per_fold, file.path(final_dir, "metrics_per_fold.csv"))
write_csv(summary_table, file.path(final_dir, "metrics_summary.csv"))

# ---- wide side-by-side comparison -----------------------------------
# One row per metric, one column per model, for a single glance across
# every model fitted in survival_testing.R plus XGBoost from notebook 07.
model_comparison <- summary_table |>
    select(model, auc_mean, brier_mean, slope_mean) |>
    pivot_longer(-model, names_to = "metric", values_to = "value") |>
    pivot_wider(names_from = model, values_from = value) |>
    mutate(metric = recode(metric,
        auc_mean   = "AUC",
        brier_mean = "Brier score",
        slope_mean = "Calibration slope"
    )) |>
    relocate(any_of(MODEL_LEVELS), .after = metric)

model_comparison |> print(width = Inf)
write_csv(model_comparison, file.path(final_dir, "model_comparison.csv"))

# Figures (styled consistently with src/constants/thesis_plotting_style.py)
# are generated in figure.r, which reads metrics_per_fold.csv / the oof
# CSVs back in.
