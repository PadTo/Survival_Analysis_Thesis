# Full-sample inference tables only (no CV): cox.zph, LRTs, M0 vs M2a.
# Same data, bans, splines, and models as survival_testing.R.

suppressPackageStartupMessages({
  library(tidyverse)
  library(survival)
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

surv_lhs <- "Surv(interval_start, interval_end, churn_triggered_adjusted)"
spl <- function(v) sprintf("ns(%s, df = %d)", v, NS_DF)

form_m0_mean <- reformulate(covars, response = surv_lhs)
form_m1_mean <- reformulate(c(sapply(CONT, spl), DISC), response = surv_lhs)
form_m0 <- reformulate(c(covars, "cluster(user_id)"), response = surv_lhs)
form_m2a <- reformulate(c("interval_id", covars),
  response = "churn_triggered_adjusted"
)
form_m2b <- reformulate(c("interval_id", sapply(CONT, spl), DISC),
  response = "churn_triggered_adjusted"
)

m0_lr <- coxph(form_m0_mean, data = features, ties = "efron")
m1_lr <- coxph(form_m1_mean, data = features, ties = "efron")
m0 <- coxph(form_m0, data = features, ties = "efron")
m2a <- glm(form_m2a, data = features, family = binomial(link = "cloglog"))
m2b <- glm(form_m2b, data = features, family = binomial(link = "cloglog"))

a_cox <- anova(m0_lr, m1_lr)
a_cl <- anova(m2a, m2b, test = "LRT")
lrt_linearity <- tibble(
  comparison = c(
    "M0 vs M1 (Cox, restricted cubic splines)",
    "M2a vs M2b (cloglog, restricted cubic splines)"
  ),
  statistic = c(a_cox$Chisq[2], a_cl$Deviance[2]),
  df = c(a_cox$Df[2], a_cl$Df[2]),
  p = c(a_cox$`Pr(>|Chi|)`[2], a_cl$`Pr(>Chi)`[2])
)
write_csv(lrt_linearity, file.path(final_dir, "lrt_linearity.csv"))

zph <- lapply(
  c("identity", "rank", "log", "km"),
  function(tr) cox.zph(m0_lr, transform = tr)
)
names(zph) <- c("identity", "rank", "log", "km")

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

write_csv(ph_zph_global, file.path(final_dir, "ph_zph_global.csv"))
write_csv(ph_zph_covariates, file.path(final_dir, "ph_zph_covariates.csv"))
write_csv(ph_zph_smallest_p, file.path(final_dir, "ph_zph_smallest_p.csv"))

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

m0_m2a_agreement <- tibble(
  n_shared_terms = nrow(cmp),
  correlation = cor(cmp$beta_m0, cmp$beta_m2a),
  max_abs_rel_diff = max(abs(cmp$rel_diff)),
  mean_abs_rel_diff = mean(abs(cmp$rel_diff))
)
write_csv(cmp, file.path(final_dir, "m0_m2a_coefficients.csv"))
write_csv(m0_m2a_agreement, file.path(final_dir, "m0_m2a_agreement.csv"))

cat("wrote inference tables to", final_dir, "\n")
print(ph_zph_global, width = Inf)
print(ph_zph_smallest_p, width = Inf)
print(lrt_linearity, width = Inf)
print(m0_m2a_agreement, width = Inf)
