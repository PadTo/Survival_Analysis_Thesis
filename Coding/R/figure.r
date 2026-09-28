suppressPackageStartupMessages({
    library(tidyverse)
})

fig_dir <- file.path(dirname(coding_root), "Images")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

# ---- Styling, mirrored from src/constants/thesis_plotting_style.py --
# Kept as literal hex copies (not read from the .py file) so figure.r has
# no Python dependency; if the palette changes there, update it here too.

THESIS_PALETTE <- c(
    Professional = "#2C6E8F",
    Personal     = "#4C956C",
    Neutral      = "#8D99AE"
)
THESIS_COLOURS <- c(
    background     = "#FFFFFF",
    text           = "#22333B",
    secondary_text = "#5A6B73",
    grid           = "#E7EBED",
    axis           = "#C7D0D5",
    accent         = "#E9A23B",
    highlight      = "#A23B47"
)
# Dashed reference / threshold lines (HR = 1, p = 0.05, y = x) use this
# colour throughout, matching the "highlight = reference line" role in the
# Python styling constants would otherwise clash with, since highlight is
# also used below as a model colour; secondary_text reads as neutral ink.
REF_LINE_COLOUR <- unname(THESIS_COLOURS[["secondary_text"]])

theme_thesis <- function(base_size = 10) {
    theme_minimal(base_size = base_size) +
        theme(
            plot.background  = element_rect(fill = THESIS_COLOURS[["background"]], colour = NA),
            panel.background = element_rect(fill = THESIS_COLOURS[["background"]], colour = NA),
            panel.grid.major = element_line(colour = THESIS_COLOURS[["grid"]], linewidth = 0.4),
            panel.grid.minor = element_blank(),
            axis.line        = element_line(colour = THESIS_COLOURS[["axis"]], linewidth = 0.4),
            axis.ticks       = element_blank(),
            axis.text        = element_text(colour = THESIS_COLOURS[["secondary_text"]]),
            axis.title       = element_text(colour = THESIS_COLOURS[["text"]]),
            plot.title       = element_text(colour = THESIS_COLOURS[["text"]], face = "bold", hjust = 0),
            strip.text       = element_text(colour = THESIS_COLOURS[["text"]], face = "bold"),
            strip.background = element_blank(),
            legend.position  = "bottom",
            legend.key       = element_rect(fill = THESIS_COLOURS[["background"]], colour = NA),
            legend.title     = element_text(colour = THESIS_COLOURS[["text"]]),
            legend.text      = element_text(colour = THESIS_COLOURS[["secondary_text"]])
        )
}
base_theme <- theme_thesis()

# readable feature labels
LABELS <- c(
    n_sessions_0_28_days = "Sessions (28d)",
    prop_scan_0_56 = "Scan proportion",
    sessions_intensity_drift_0_28_vs_28_56_days = "Session drift",
    prop_main_0_56 = "Main-app proportion",
    vehicle_age_sd_overall = "Vehicle age SD",
    vehicle_mean_mileage_ordinal = "Mean mileage ordinal",
    prop_oca_0_56 = "One-click app proportion",
    prop_history_screen_0_56 = "History-screen proportion",
    prop_clear_0_56 = "Clear proportion",
    prop_in_prod = "In-production proportion",
    prop_coding_0_56 = "Coding proportion",
    vehicle_mean_age_overall = "Mean vehicle age",
    prop_live_data_0_56 = "Live-data proportion",
    usage_concentration = "Usage concentration",
    overall_prop_vehicle_make_skoda = "<U+0160>koda proportion",
    sd_gap_0_56_days = "Gap SD (56d)",
    overall_prop_vehicle_make_volkswagen = "Volkswagen proportion",
    prop_main_drift_0_28_vs_28_56 = "Main-app drift",
    active_flag_0_56_daysTRUE = "Active flag (56d)",
    active_flag_0_56_days = "Active flag (56d)",
    days_since_last_new_vehicle = "Days since last new vehicle",
    overall_prop_vehicle_make_audi = "Audi proportion",
    mean_gap_0_56_days = "Gap mean (56d)",
    weekend_share_0_28_days = "Weekend share (28d)",
    actions_per_session_0_28_days = "Actions per session (28d)",
    n_new_vehicles_0_28_days = "New vehicles (28d)",
    onboarding_delay_days = "Onboarding delay",
    actions_per_session_intensity_drift_0_28_vs_28_56_days = "Actions-per-session drift"
)
lab <- function(x) ifelse(is.na(LABELS[x]), x, LABELS[x])

# readable model labels + one consistent colour per model, reused across
# every multi-model figure (calibration, AUC by fold, ...).
MODEL_LEVELS <- c("M0", "M1", "M2a", "M2b", "XGB")
MODEL_LABELS <- c(
    M0  = "Cox (linear)",
    M1  = "Cox (splines)",
    M2a = "Cloglog (linear)",
    M2b = "Cloglog (splines)",
    XGB = "XGBoost"
)
MODEL_COLOURS <- c(
    M0  = unname(THESIS_PALETTE[["Professional"]]),
    M1  = unname(THESIS_PALETTE[["Personal"]]),
    M2a = unname(THESIS_COLOURS[["accent"]]),
    M2b = unname(THESIS_PALETTE[["Neutral"]]),
    XGB = unname(THESIS_COLOURS[["highlight"]])
)
model_scale <- function(...) {
    scale_colour_manual(values = setNames(MODEL_COLOURS, MODEL_LABELS[names(MODEL_COLOURS)]), ...)
}

# ---- Fig 09: PH test, all four transforms ---------------------------
# Skip when figure.r is sourced without a live zph object (e.g. metrics-only rerun).

if (exists("zph") && exists("coef_tab") && exists("cmp")) {

TRANSFORM_COLOURS <- c(
    Identity       = unname(THESIS_PALETTE[["Professional"]]),
    Rank           = unname(THESIS_PALETTE[["Personal"]]),
    Log            = unname(THESIS_COLOURS[["accent"]]),
    `Kaplan-Meier` = unname(THESIS_COLOURS[["highlight"]])
)

zph_df <- imap_dfr(zph, function(z, nm) {
    tab <- as.data.frame(z$table)
    tibble(transform = nm, covariate = rownames(tab), p = tab$p)
}) |>
    filter(covariate != "GLOBAL") |>
    mutate(
        covariate = lab(covariate),
        covariate = fct_reorder(covariate, p, .fun = min),
        transform = factor(transform,
            levels = c("identity", "rank", "log", "km"),
            labels = c("Identity", "Rank", "Log", "Kaplan-Meier")
        )
    )

p09 <- ggplot(zph_df, aes(p, covariate, shape = transform, colour = transform)) +
    geom_vline(xintercept = 0.05, linetype = 2, colour = REF_LINE_COLOUR) +
    annotate("text",
        x = 0.05, y = -Inf, label = "0.05",
        hjust = -0.2, vjust = -0.5, size = 2.6, colour = THESIS_COLOURS[["secondary_text"]]
    ) +
    geom_point(size = 2, alpha = 0.85) +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25)) +
    scale_colour_manual(values = TRANSFORM_COLOURS) +
    labs(x = "p-value", y = NULL, shape = NULL, colour = NULL) +
    base_theme

ggsave(file.path(fig_dir, "09_ph_test_transforms.png"), p09,
    width = 9, height = 7, dpi = 300
)

# ---- Fig 10: coefficients, forest plot ------------------------------
# Colour by direction (crossing 1 = not significant), matching the
# Personal/highlight "stays longer / leaves sooner" convention used for
# the counting-process forest plot in notebook 06.

p10 <- coef_tab |>
    filter(!is.na(hr_sd)) |>
    mutate(
        term = lab(term),
        term = fct_reorder(term, hr_sd),
        crosses = lo <= 1 & 1 <= hi,
        direction = case_when(
            crosses ~ "Not significant",
            hr_sd < 1 ~ "Lower hazard",
            TRUE ~ "Higher hazard"
        ),
        direction = factor(direction,
            levels = c("Not significant", "Lower hazard", "Higher hazard")
        )
    ) |>
    ggplot(aes(hr_sd, term)) +
    geom_vline(xintercept = 1, linetype = 2, colour = REF_LINE_COLOUR) +
    geom_errorbarh(aes(xmin = lo, xmax = hi, colour = direction),
        height = 0, linewidth = 0.5
    ) +
    geom_point(aes(colour = direction), size = 2) +
    scale_x_log10() +
    scale_colour_manual(
        values = c(
            "Not significant" = unname(THESIS_PALETTE[["Neutral"]]),
            "Lower hazard"    = unname(THESIS_PALETTE[["Personal"]]),
            "Higher hazard"   = unname(THESIS_COLOURS[["highlight"]])
        ),
        name = NULL
    ) +
    labs(x = "Hazard ratio per one standard deviation (log scale)", y = NULL) +
    base_theme

ggsave(file.path(fig_dir, "10_coefficients_m0.png"), p10,
    width = 8, height = 7, dpi = 300
)

# ---- Fig 11: M0 against M2a coefficients ----------------------------

rng <- range(c(cmp$beta_m0, cmp$beta_m2a))

p11 <- ggplot(cmp, aes(beta_m0, beta_m2a)) +
    geom_abline(linetype = 2, colour = REF_LINE_COLOUR) +
    geom_point(size = 2, alpha = 0.85, colour = unname(THESIS_PALETTE[["Professional"]])) +
    coord_equal(xlim = rng, ylim = rng) +
    labs(
        x = expression(hat(beta) * " from M0 (Cox, Efron ties)"),
        y = expression(hat(beta) * " from M2a (grouped cloglog)")
    ) +
    base_theme

ggsave(file.path(fig_dir, "11_tie_agreement.png"), p11,
    width = 5.5, height = 5.5, dpi = 300
)

}  # end exists("zph") — fig 09–11

# ---- Fig 12: calibration, all five models ---------------------------
# oof_predictions.csv (M0/M1/M2a/M2b, from survival_testing.R) and
# oof_predictions_xgboost.csv (XGB, from notebook 07) share the same
# (model, fold, user_id, d, h) schema, so they combine directly.

oof <- bind_rows(
    read_csv(file.path(final_dir, "oof_predictions.csv"), show_col_types = FALSE),
    read_csv(file.path(final_dir, "oof_predictions_xgboost.csv"), show_col_types = FALSE)
) |>
    mutate(
        d = as.integer(d),
        model = factor(model, levels = MODEL_LEVELS, labels = MODEL_LABELS)
    )

cal_dat <- oof |>
    group_by(model) |>
    mutate(bin = ntile(h, 20)) |>
    group_by(model, bin) |>
    summarise(pred = mean(h), obs = mean(d), n = n(), .groups = "drop")

p12 <- ggplot(cal_dat, aes(pred, obs, colour = model)) +
    geom_abline(linetype = 2, colour = REF_LINE_COLOUR) +
    geom_point(size = 1.8, alpha = 0.85) +
    facet_wrap(~model, nrow = 1) +
    model_scale() +
    labs(x = "Predicted interval hazard", y = "Observed event rate") +
    base_theme +
    theme(legend.position = "none")

ggsave(file.path(fig_dir, "12_calibration.png"), p12,
    width = 11.5, height = 3, dpi = 300
)

# ---- Fig 13: AUC by fold, all five models ---------------------------

per_fold <- read_csv(file.path(final_dir, "metrics_per_fold.csv"),
    show_col_types = FALSE
) |>
    mutate(model = factor(model, levels = MODEL_LEVELS, labels = MODEL_LABELS))

p13 <- per_fold |>
    ggplot(aes(factor(fold), auc, colour = model, shape = model, group = model)) +
    geom_line(linewidth = 0.4, alpha = 0.6) +
    geom_point(size = 2.5) +
    model_scale(name = NULL) +
    scale_shape_discrete(name = NULL) +
    labs(x = "Fold", y = "Out-of-fold AUC") +
    base_theme

ggsave(file.path(fig_dir, "13_auc_by_fold.png"), p13,
    width = 8, height = 4.5, dpi = 300
)

cat("figures written to", normalizePath(fig_dir), "\n")
