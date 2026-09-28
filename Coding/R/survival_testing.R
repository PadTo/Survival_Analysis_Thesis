suppressPackageStartupMessages({
  library(tidyverse)
  library(survival)
  library(splines)
})

# ---- 1. Data -------------------------------------------------------
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

features <- read_csv(file.path(
  final_dir,
  "features_personal_14_day_intervals_new_features.csv"
), show_col_types = FALSE)

folds <- read_csv(file.path(
  final_dir,
  "user_folds_personal_14.csv"
), show_col_types = FALSE)

n_before <- nrow(features)
features <- left_join(features, folds, by = "user_id")
stopifnot(nrow(features) == n_before, !any(is.na(features$fold)))

# interval index 0, 1, 2, ... used as the time axis by the grouped models
features <- features |> mutate(interval_id = interval_start / 14)



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

# cap at the last interval before events first drop below the floor
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


# ---- 2. Which columns are covariates -------------------------------

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

# ---- 3. Shared pieces ----------------------------------------------

surv_lhs <- "Surv(interval_start, interval_end, churn_triggered_adjusted)"

# natural cubic spline == restricted cubic spline (linear past the
# boundary knots). df = 3 places 2 interior knots plus 2 boundary knots.
spl <- function(v) sprintf("ns(%s, df = %d)", v, NS_DF)

# turn a fitted Cox model into a predicted hazard for each test row,
# using the Breslow baseline accumulated over that row's own interval:
#   h = 1 - exp(-dLambda0 * exp(lp))
#
# basehaz(centered = FALSE) returns Lambda_0 at x = 0, so the linear
# predictor must also be taken at x = 0. predict()'s default
# reference = "sample" subtracts the training-set mean and would leave
# the two on different scales, inflating h several-fold.
#
# basehaz()'s table starts at the first observed event time (56 here, since
# left-truncation moved entry to day 42), not at 0. With entry no longer at
# day 0, test rows can now start (interval_start = 42) *before* that first
# tabulated point, and approxfun(..., rule = 2) flat-extrapolates them to
# the table's first value -- identical to Lambda0(56) -- so dLambda0
# collapses to exactly 0 for the entire first (and highest-hazard) interval.
# Lambda_0(t) = 0 exactly for any t before the first observed event, by the
# Breslow estimator's own definition (no hazard has been observed to
# accumulate yet) -- not merely close to 0, so the left tail is handled as
# an exact 0 rather than a linear ramp toward the first jump (rule = 2 on
# the right keeps every other row's existing flat-extrapolation/interpolation
# behaviour beyond the last observed event unchanged).
cox_hazard <- function(fit, test) {
  bh <- basehaz(fit, centered = FALSE)
  Lambda0_tab <- approxfun(bh$time, bh$hazard, rule = c(1, 2))
  Lambda0 <- function(t) ifelse(t < min(bh$time), 0, Lambda0_tab(t))
  dLambda0 <- Lambda0(test$interval_end) - Lambda0(test$interval_start)
  lp <- as.numeric(predict(fit, newdata = test, type = "lp", reference = "zero"))
  1 - exp(-dLambda0 * exp(lp))
}

# ---- 4. One model = one function (train, test) -> predicted hazard --

# M0: Cox, all covariates linear
fit_m0 <- function(train, test) {
  form <- reformulate(c(covars, "cluster(user_id)"), response = surv_lhs)
  fit <- coxph(form, data = train, ties = "efron")
  cox_hazard(fit, test)
}

# M1: Cox, restricted cubic splines on continuous covariates
fit_m1 <- function(train, test) {
  form <- reformulate(
    c(sapply(CONT, spl), DISC, "cluster(user_id)"),
    response = surv_lhs
  )
  fit <- coxph(form, data = train, ties = "efron")
  cox_hazard(fit, test)
}

# M2a: grouped cloglog, free intercept per interval, covariates linear
fit_m2a <- function(train, test) {
  form <- reformulate(
    c("interval_id", covars),
    response = "churn_triggered_adjusted"
  )
  fit <- glm(form, data = train, family = binomial(link = "cloglog"))
  as.numeric(predict(fit, newdata = test, type = "response"))
}

# M2b: grouped cloglog, free intercept per interval, splines on covariates
fit_m2b <- function(train, test) {
  form <- reformulate(
    c("interval_id", sapply(CONT, spl), DISC),
    response = "churn_triggered_adjusted"
  )
  fit <- glm(form, data = train, family = binomial(link = "cloglog"))
  as.numeric(predict(fit, newdata = test, type = "response"))
}

# ---- 5. One runner, used by every model ----------------------------

run_cv <- function(model_fn, model_name) {
  map_dfr(sort(unique(features$fold)), function(k) {
    train <- filter(features, fold != k)
    test <- filter(features, fold == k)

    cat(model_name, "fold", k, "...")
    h <- model_fn(train, test)
    cat(" done\n")

    tibble(
      model   = model_name,
      fold    = k,
      user_id = test$user_id,
      d       = test$churn_triggered_adjusted,
      h       = h
    )
  })
}

# ---- 6. Run ---------------------------------------------------------

oof <- bind_rows(
  run_cv(fit_m0, "M0"),
  run_cv(fit_m1, "M1"),
  run_cv(fit_m2a, "M2a"),
  run_cv(fit_m2b, "M2b")
)

write_csv(oof, file.path(final_dir, "oof_predictions.csv"))

# ---- 7. Sanity checks ----------------------------------------------
# calib should sit near 1. Much above it means the predicted hazards are
# systematically too high, which is what an uncentred linear predictor
# against a centred baseline produces.

oof |>
  group_by(model) |>
  summarise(
    n        = n(),
    events   = sum(d),
    obs_rate = mean(d),
    mean_h   = mean(h),
    calib    = mean(h) / mean(d),
    h_min    = min(h),
    h_median = median(h),
    h_max    = max(h),
    h_zero   = mean(h == 0),
    h_na     = mean(is.na(h)),
    .groups  = "drop"
  ) |>
  print(width = Inf)


# ---- 3b. JOB 1: inference on full data ------------------------------
# cluster() gives sandwich SEs for reporting. anova.coxph and cox.zph
# need the ordinary information matrix, so the LRT and PH tests use
# unclustered copies of the same mean models.

form_m0_mean <- reformulate(covars, response = surv_lhs)
form_m1_mean <- reformulate(c(sapply(CONT, spl), DISC), response = surv_lhs)
form_m0 <- reformulate(c(covars, "cluster(user_id)"), response = surv_lhs)
form_m1 <- reformulate(
  c(sapply(CONT, spl), DISC, "cluster(user_id)"),
  response = surv_lhs
)
form_m2a <- reformulate(c("interval_id", covars),
  response = "churn_triggered_adjusted"
)
form_m2b <- reformulate(c("interval_id", sapply(CONT, spl), DISC),
  response = "churn_triggered_adjusted"
)

m0_lr <- coxph(form_m0_mean, data = features, ties = "efron")
m1_lr <- coxph(form_m1_mean, data = features, ties = "efron")
m0 <- coxph(form_m0, data = features, ties = "efron")
m1 <- coxph(form_m1, data = features, ties = "efron")
m2a <- glm(form_m2a, data = features, family = binomial(link = "cloglog"))
m2b <- glm(form_m2b, data = features, family = binomial(link = "cloglog"))

cat("fitted coefficients in M0:", sum(!is.na(coef(m0))), "\n")

# RQ2 -- linearity, under each link
a_cox <- anova(m0_lr, m1_lr)
a_cl <- anova(m2a, m2b, test = "LRT")
print(a_cox)
print(a_cl)

lrt_linearity <- tibble(
  comparison = c("M0 vs M1 (Cox, restricted cubic splines)",
                 "M2a vs M2b (cloglog, restricted cubic splines)"),
  statistic = c(a_cox$Chisq[2], a_cl$Deviance[2]),
  df = c(a_cox$Df[2], a_cl$Df[2]),
  p = c(a_cox$`Pr(>|Chi|)`[2], a_cl$`Pr(>Chi)`[2])
)
print(lrt_linearity, width = Inf)
write_csv(lrt_linearity, file.path(final_dir, "lrt_linearity.csv"))

# RQ1 -- proportional hazards, all four transforms (Park & Hendry 2015)
zph <- lapply(
  c("identity", "rank", "log", "km"),
  function(tr) cox.zph(m0_lr, transform = tr)
)
names(zph) <- c("identity", "rank", "log", "km")
lapply(zph, print)

ph_zph_global <- imap_dfr(zph, function(z, nm) {
  g <- as.data.frame(z$table)["GLOBAL", ]
  tibble(transform = nm, chisq = unname(g$chisq), df = unname(g$df), p = unname(g$p))
})
ph_zph_covariates <- imap_dfr(zph, function(z, nm) {
  tab <- as.data.frame(z$table)
  tab <- tab[rownames(tab) != "GLOBAL", , drop = FALSE]
  tibble(
    transform = nm,
    covariate = rownames(tab),
    chisq = tab$chisq,
    df = tab$df,
    p = tab$p
  )
})
ph_zph_smallest_p <- ph_zph_covariates |>
  filter(transform == "rank") |>
  arrange(p) |>
  slice_head(n = 4)

print(ph_zph_global, width = Inf)
print(ph_zph_smallest_p, width = Inf)
write_csv(ph_zph_global, file.path(final_dir, "ph_zph_global.csv"))
write_csv(ph_zph_covariates, file.path(final_dir, "ph_zph_covariates.csv"))
write_csv(ph_zph_smallest_p, file.path(final_dir, "ph_zph_smallest_p.csv"))

# model.matrix(m0) has one column per coefficient (e.g. dummy-coded
# active_flag_0_56_daysTRUE), aligned 1:1 with names(coef(m0)); sd() on the
# raw `features` columns of the same name would silently return NA for any
# non-numeric term, since e.g. "active_flag_0_56_daysTRUE" is not itself a
# column of `features` (features$active_flag_0_56_days is logical).
mm <- model.matrix(m0)
stopifnot(identical(colnames(mm), names(coef(m0))))

coef_tab <- tibble(
  term = names(coef(m0)),
  beta = as.numeric(coef(m0)),
  se   = sqrt(diag(vcov(m0))),
  sd_x = apply(mm, 2, sd)
) |>
  mutate(
    hr_sd = exp(beta * sd_x),
    lo    = exp((beta - 1.96 * se) * sd_x),
    hi    = exp((beta + 1.96 * se) * sd_x),
    z     = beta / se,
    p     = 2 * pnorm(-abs(z))
  ) |>
  arrange(p)

print(coef_tab, n = Inf)
write_csv(coef_tab, file.path(final_dir, "coefficients_m0.csv"))

cmp <- inner_join(
  tibble(term = names(coef(m0)), beta_m0 = as.numeric(coef(m0))),
  tibble(term = names(coef(m2a)), beta_m2a = as.numeric(coef(m2a))),
  by = "term"
) |>
  mutate(
    diff     = beta_m2a - beta_m0,
    rel_diff = diff / abs(beta_m0),
    hr_m0    = exp(beta_m0),
    hr_m2a   = exp(beta_m2a)
  ) |>
  arrange(desc(abs(rel_diff)))

print(cmp, n = Inf)

m0_m2a_agreement <- tibble(
  n_shared_terms = nrow(cmp),
  correlation = cor(cmp$beta_m0, cmp$beta_m2a),
  max_abs_rel_diff = max(abs(cmp$rel_diff)),
  mean_abs_rel_diff = mean(abs(cmp$rel_diff))
)
print(m0_m2a_agreement, width = Inf)
write_csv(cmp, file.path(final_dir, "m0_m2a_coefficients.csv"))
write_csv(m0_m2a_agreement, file.path(final_dir, "m0_m2a_agreement.csv"))

cat("max |relative difference|:", max(abs(cmp$rel_diff)), "\n")
cat("correlation:", cor(cmp$beta_m0, cmp$beta_m2a), "\n")

# ---- 3c. M3 -- does the PH violation from the rank-transform test matter? --
# M0 assumes each covariate's log hazard ratio is constant over time. For the
# covariates the rank-transform cox.zph flagged at p < 0.1 (rank chosen as
# primary transform: ~50% censoring in this segment points away from km/log,
# see PH-test notes above), M3 adds a tt() term interacting that covariate
# with time (linear-in-days, matching the "identity" transform's assumption).
# tt()'s own coefficient is the slope of the log hazard ratio per day; if
# that slope times the observed follow-up span is small next to the main
# effect, M0's single time-constant beta is a fine summary despite the
# significant test -- the PH violation is statistically real but practically
# negligible. cluster(user_id) is safe here (summary()/coef() tolerate the
# robust variance; only anova()/cox.zph() do not).

TT_ALPHA <- 0.1
rank_tab <- zph$rank$table
tt_covars <- rownames(rank_tab)[rank_tab[, "p"] < TT_ALPHA & rownames(rank_tab) != "GLOBAL"]
cat("M3 tt() terms (rank p <", TT_ALPHA, "):", paste(tt_covars, collapse = ", "), "\n")

form_m3 <- reformulate(
  c(covars, sprintf("tt(%s)", tt_covars), "cluster(user_id)"),
  response = surv_lhs
)
m3 <- coxph(
  form_m3,
  data = features,
  ties = "efron",
  tt = function(x, t, ...) x * t
)

m3_coef <- summary(m3)$coefficients
print(m3_coef[c(tt_covars, sprintf("tt(%s)", tt_covars)), ])

# Gap check: drift in log-HR from the earliest to the latest observed day,
# relative to the size of the (time-constant) main-effect coefficient from
# M0 -- not from M3's own main-effect term, which is itself re-estimated
# jointly with its tt() partner and answers a slightly different question
# (the fitted value at t = 0, not the M0-style time-averaged effect).
t_span <- diff(range(features$interval_start))
m3_drift <- tibble(
  term        = tt_covars,
  beta_m0     = coef(m0)[tt_covars],
  beta_tt     = coef(m3)[sprintf("tt(%s)", tt_covars)],
  p_tt        = m3_coef[sprintf("tt(%s)", tt_covars), "Pr(>|z|)"],
) |>
  mutate(
    drift_log_hr = beta_tt * t_span,
    rel_drift    = abs(drift_log_hr) / abs(beta_m0),
    hr_m0        = exp(beta_m0),
    hr_drifted   = exp(beta_m0 + drift_log_hr)
  ) |>
  arrange(desc(rel_drift))

print(m3_drift, n = Inf)
write_csv(m3_drift, file.path(final_dir, "m3_time_varying_drift.csv"))

# ---- 3d. M3 robustness check -- does the tt() functional form matter? -----
# tt = x * t (linear-in-days, above) matches cox.zph's "identity" transform,
# but "identity" is the one transform that did NOT flag prop_scan_0_56
# (p = 0.175) -- only rank/log/km did (p = 0.040/0.021/0.021), so a
# log-in-days interaction is arguably the more consistent functional form to
# test the drift these covariates were actually flagged for. t here is
# always the event/censoring ("stop") time, minimum 14 days (never the
# interval_start entry time, which can be 0), so log(t) is well-defined.
form_m3_log <- reformulate(
  c(covars, sprintf("tt(%s)", tt_covars), "cluster(user_id)"),
  response = surv_lhs
)
m3_log <- coxph(
  form_m3_log,
  data = features,
  ties = "efron",
  tt = function(x, t, ...) x * log(t)
)

m3_log_coef <- summary(m3_log)$coefficients
print(m3_log_coef[c(tt_covars, sprintf("tt(%s)", tt_covars)), ])

log_t_span <- diff(log(range(features$interval_end)))
m3_log_drift <- tibble(
  term        = tt_covars,
  beta_m0     = coef(m0)[tt_covars],
  beta_tt     = coef(m3_log)[sprintf("tt(%s)", tt_covars)],
  p_tt        = m3_log_coef[sprintf("tt(%s)", tt_covars), "Pr(>|z|)"],
) |>
  mutate(
    drift_log_hr = beta_tt * log_t_span,
    rel_drift    = abs(drift_log_hr) / abs(beta_m0),
    hr_m0        = exp(beta_m0),
    hr_drifted   = exp(beta_m0 + drift_log_hr)
  ) |>
  arrange(desc(rel_drift))

print(m3_log_drift, n = Inf)
write_csv(m3_log_drift, file.path(final_dir, "m3_time_varying_drift_log.csv"))
