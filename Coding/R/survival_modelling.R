library(survival)
library(tidyverse)
library(dplyr)

candidate_roots <- c(getwd(), file.path(getwd(), "Coding"), dirname(getwd()))
project_root <- candidate_roots[
  dir.exists(file.path(candidate_roots, "src")) &
    dir.exists(file.path(candidate_roots, "notebooks"))
][1]

if (is.na(project_root)) {
  stop("Could not locate the Coding project directory.")
}

personal_features_path <- file.path(
  project_root,
  "Data",
  "final",
  "features_personal_14_day_intervals_new_features.csv"
)

data_personal <- read.csv(personal_features_path)
data_personal <- data_personal[order(data_personal$user_id, data_personal$interval_start), ]
data_personal$churn_triggered_adjusted = as.numeric(as.logical(data_personal$churn_triggered_adjusted))
head(data_personal)


df2 <- data_personal %>%
  mutate(across(-c(user_id, interval_start, interval_end, churn_triggered_adjusted),
                ~ as.vector(scale(.))))
head(df2)
head(data_personal)


fit <- coxph(
    Surv(time=interval_start, time2=interval_end, event=churn_triggered_adjusted) ~ 
    prop_clear_0_56 +
    prop_coding_0_56 +
    prop_history_screen_0_56 +
    prop_live_data_0_56 +
    prop_oca_0_56 +
    prop_scan_0_56 +
    prop_main_0_56 +
    prop_main_drift_0_28_vs_28_56 +
    n_sessions_0_28_days +
    sessions_intensity_drift_0_28_vs_28_56_days +
    n_actions_0_28_days +
    actions_intensity_drift_0_28_vs_28_56_days +
    recency +
    vehicle_mean_age_overall +
    prop_in_prod +
    overall_prop_vehicle_make_audi +
    overall_prop_vehicle_make_bmw_group +
    overall_prop_vehicle_make_premium +
    overall_prop_vehicle_make_seat +
    overall_prop_vehicle_make_skoda +
    overall_prop_vehicle_make_volkswagen,
  data = df2,
  robust = TRUE,
  id = user_id,
  ties = "efron"
)

table(data_personal$churn_triggered_adjusted)
class(data_personal$churn_triggered_adjusted)
unique(data_personal$churn_triggered_adjusted)
sum(is.na(data_personal$churn_triggered_adjusted))

fit

