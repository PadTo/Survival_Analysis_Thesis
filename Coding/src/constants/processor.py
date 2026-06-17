# ============================================================
#  Pipeline configuration (tunable numeric parameters)
# ============================================================
LOOKBACK_PERIODS: tuple[int, ...] = (1, 2, 3)
INTERVAL_IN_DAYS: int = 14
SUM_TO_LIMIT_IN_DAYS: int = 56
FIRST_PERIOD_IN_DAYS: int = 28
SECOND_PERIOD_IN_DAYS: int = 56
N_NEIGHBOURS: int = 5

# ============================================================
#  Sentinel values filling empty-interval / missing rows
# ============================================================
UNKNOWN_VALUE: str = "unknown"
NONE_ACTIVITY_VALUE: str = "none"


# ============================================================
#  Generated-column name templates (single source of truth).
#  {a} / {name} = token, {start}/{end}/{r0,r1,r2} = day boundaries.
# ============================================================
COL_TEMPLATE_FORMAT: str = "n_{a}_{start}_{end}_days"
PROP_COL_TEMPLATE: str = "prop_{a}_{start}_{end}"
TOTAL_ACTIONS_COL_TEMPLATE: str = "total_actions_{start}_{end}"
OVERALL_PROP_COL_TEMPLATE: str = "overall_prop_{name}"
CUMULATIVE_COUNT_COL_TEMPLATE: str = "cumulative_count_{name}"
DRIFT_COL_TEMPLATE: str = "prop_{a}_drift_{r0}_{r1}_vs_{r1}_{r2}"
INTENSITY_DRIFT_COL_TEMPLATE: str = "{a}_intensity_drift_{r0}_{r1}_vs_{r1}_{r2}_days"


# ============================================================
#  Aggregation tokens (the {a}/{name} filled into the templates)
# ============================================================
SESSIONS_TOKEN: str = "sessions"
ACTIONS_TOKEN: str = "actions"
STILL_IN_PRODUCTION_TOKEN: str = "still_in_production"
TOTAL_CARS_TOKEN: str = "total_cars"
TOTAL_APP_TOKEN: str = "total_app"


# ============================================================
#  Fixed (non-templated) generated column names
# ============================================================
FIRST_YEAR_COL: str = "first_year"
START_DATE_COL: str = "start_date"
END_DATE_COL: str = "end_date"
LAST_ACTIVITY_DATE_COL: str = "last_activity_date"
RECENCY_COL: str = "recency"
INTERVAL_END_COL: str = "interval_end"
PROP_IN_PROD_COL: str = "prop_in_prod"
CUMULATIVE_COUNT_TOTAL_COL: str = "cumulative_count_total"
MEAN_AGE_INTERMEDIATE_COL: str = "vehicle_mean_age_intermediate"
VEHICLE_AGE_COLUMN_NAME: str = "vehicle_age"
VEHICLE_MEAN_AGE_COLUMN_NAME: str = "vehicle_mean_age_in_an_interval"
VEHICLE_MEAN_OVERALL_AGE_COLUMN_NAME: str = "vehicle_mean_age_overall"
FIRST_DISTINCT_COLUMN_NAME: str = "first_distinct"
CHURN_TRIGGERED_SHIFTED_COLUMN_NAME: str = "churn_triggered_adjusted"