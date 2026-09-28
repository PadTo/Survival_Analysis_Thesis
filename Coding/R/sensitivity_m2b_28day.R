# Sensitivity: M2b (grouped cloglog + covariate splines) on 28-day intervals.
# Reuses the same user folds as the 14-day analysis; records AUC / Brier /
# cloglog calibration slope per fold and overall.

suppressPackageStartupMessages({
  library(tidyverse)
  library(splines)
})

coding_root <- local({
  candidates <- c(
    file.path(getwd(), "Coding"),
    getwd(),
    dirname(getwd())
  )
  for (candidate in candidates) {
    if (dir.exists(file.path(candidate, "Data", "final"))) {
      return(normalizePath(candidate))
    }
  }
  stop("Could not locate Coding/ with Data/final/")
})

final_dir <- file.path(coding_root, "Data", "final")
interval_days <- 28L

features <- read_csv(file.path(
  final_dir,
  "features_personal_28_day_intervals_new_features.csv"
), show_col_types = FALSE)

folds <- read_csv(file.path(
  final_dir,
  "user_folds_personal_14.csv"
), show_col_types = FALSE)

n_before <- nrow(features)
features <- left_join(features, folds, by = "user_id")
stopifnot(nrow(features) == n_before, !any(is.na(features$fold)))

features <- features |> mutate(interval_id = interval_start / interval_days)

MIN_EVENTS <- 10L
int_counts <- features |>
  count(interval_id, wt = NULL, name = "n") |>
  left_join(
    features |>
      group_by(interval_id) |>
      summarise(ev = sum(churn_triggered_adjusted), .groups = "drop"),
    by = "interval_id"
  ) |>
  arrange(interval_id)

print(int_counts, n = Inf)

first_dense <- min(int_counts$interval_id[int_counts$ev >= MIN_EVENTS])
thin <- int_counts$interval_id[
  int_counts$ev < MIN_EVENTS & int_counts$interval_id > first_dense
]
max_int <- if (length(thin) == 0) {
  max(int_counts$interval_id)
} else {
  min(thin) - 1
}

features <- features |>
  mutate(interval_id = factor(pmin(interval_id, max_int)))

cat(
  "interval levels:", nlevels(features$interval_id),
  " (capped at", max_int, ")\n"
)

stopifnot(all(sapply(sort(unique(features$fold)), function(k) {
  nlevels(droplevels(features$interval_id[features$fold != k])) ==
    nlevels(features$interval_id)
})))

BANNED <- c("recency", "CV_gap_0_56_days", "account_tenure_days")
ID_CLOCK <- c(
  "user_id", "fold", "interval_start", "interval_end",
  "interval_id", "churn_triggered_adjusted"
)
covars <- setdiff(names(features), c(BANNED, ID_CLOCK))

NS_DF <- 3L
spline_safe <- function(x, fold, df = NS_DF) {
  knot_probs <- seq.int(0, 1, length.out = df + 1L)[-c(1L, df + 1L)]
  top_knot_prob <- max(knot_probs)
  all(sapply(sort(unique(fold)), function(k) {
    xk <- x[fold != k]
    quantile(xk, probs = top_knot_prob, na.rm = TRUE, type = 7) > min(xk, na.rm = TRUE)
  }))
}
is_cont <- sapply(covars, function(v) {
  x <- features[[v]]
  length(unique(x)) > 10 && spline_safe(x, features$fold)
})
names(is_cont) <- covars
CONT <- covars[is_cont]
DISC <- covars[!is_cont]

cat(
  "covariates:", length(covars),
  " continuous:", length(CONT),
  " discrete:", length(DISC), "\n"
)

spl <- function(v) sprintf("ns(%s, df = %d)", v, NS_DF)

fit_m2b <- function(train, test) {
  form <- reformulate(
    c("interval_id", sapply(CONT, spl), DISC),
    response = "churn_triggered_adjusted"
  )
  fit <- glm(form, data = train, family = binomial(link = "cloglog"))
  as.numeric(predict(fit, newdata = test, type = "response"))
}

oof <- map_dfr(sort(unique(features$fold)), function(k) {
  train <- filter(features, fold != k)
  test <- filter(features, fold == k)
  cat("M2b_28d fold", k, "...")
  h <- fit_m2b(train, test)
  cat(" done\n")
  tibble(
    model   = "M2b_28d",
    fold    = k,
    user_id = test$user_id,
    d       = test$churn_triggered_adjusted,
    h       = h
  )
})

write_csv(oof, file.path(final_dir, "oof_predictions_m2b_28day.csv"))

auc_fast <- function(d, p) {
  n1 <- sum(d == 1)
  n0 <- sum(d == 0)
  if (n1 == 0 || n0 == 0) {
    return(NA_real_)
  }
  r <- rank(p)
  (sum(r[d == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

cal_slope <- function(d, p) {
  ok <- p > 0 & p < 1
  if (sum(ok) < 50) {
    return(NA_real_)
  }
  x <- log(-log(1 - p[ok]))
  fit <- glm(d[ok] ~ x, family = binomial(link = "cloglog"))
  unname(coef(fit)[2])
}

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

summary_table <- per_fold |>
  group_by(model) |>
  summarise(
    auc_mean = mean(auc), auc_sd = sd(auc),
    brier_mean = mean(brier), brier_sd = sd(brier),
    slope_mean = mean(slope), slope_sd = sd(slope),
    .groups = "drop"
  )

# Side-by-side vs 14-day M2b from the main metrics file
m2b_14 <- read_csv(file.path(final_dir, "metrics_summary.csv"), show_col_types = FALSE) |>
  filter(model == "M2b") |>
  mutate(model = "M2b_14d") |>
  select(model, auc_mean, auc_sd, brier_mean, brier_sd, slope_mean, slope_sd)

comparison <- bind_rows(m2b_14, summary_table)

cat("\n--- per fold (28-day M2b) ---\n")
print(per_fold, n = Inf, width = Inf)
cat("\n--- summary vs 14-day M2b ---\n")
print(comparison, width = Inf)

write_csv(per_fold, file.path(final_dir, "metrics_per_fold_m2b_28day.csv"))
write_csv(summary_table, file.path(final_dir, "metrics_summary_m2b_28day.csv"))
write_csv(comparison, file.path(final_dir, "metrics_sensitivity_m2b_14_vs_28.csv"))
