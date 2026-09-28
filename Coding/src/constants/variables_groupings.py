brand_group_map = {
    "volkswagen": "volkswagen",
    "audi": "audi",
    "skoda": "skoda",
    "seat": "other_rare",
    "bmw": "other_rare",
    "mini": "other_rare",
    "bmw moto": "other_rare",
    "porsche": "other_rare",
    "lamborghini": "other_rare",
    "bentley": "other_rare",
    "mercedes-benz": "other_rare",
    "ford": "other_rare",
    "toyota": "other_rare",
    "man": "other_rare",
    "lexus": "other_rare",
    "toyota gr": "other_rare",
    "rolls-royce": "other_rare",
}

activity_type_groups = {
    "coding": "coding",
    "oca": "oca",
    "scan": "scan",
    "clear": "clear",
    "history_screen": "history_screen",
    "live_data": "live_data",
    "native_live_data": "other",
    "vehicle_lookup": "other",
    "mileage_rollback": "other",
    "cba_checklist": "other",
    "trip_tracker": "other",
    "vehicle_report": "other",
    "engine_readiness": "other",
    "car_alert": "other",
    "battery_check": "other",
    "maintenance_reminder": "other",
}

app_type_groups = {"main": "main", "vag": "vag"}


pretty_old_features = {
    "prop_clear_0_56": "Clear (prop)",
    "prop_coding_0_56": "Coding (prop)",
    "prop_history_screen_0_56": "History screen (prop)",
    "prop_live_data_0_56": "Live data (prop)",
    "prop_oca_0_56": "OCA (prop)",
    "prop_scan_0_56": "Scan (prop)",
    "prop_main_0_56": "Main app (prop)",
    "prop_main_drift_0_28_vs_28_56": "Main app drift",
    "n_sessions_0_28_days": "Sessions (28d)",
    "sessions_intensity_drift_0_28_vs_28_56_days": "Session drift",
    "n_actions_0_28_days": "Actions (28d)",
    "actions_intensity_drift_0_28_vs_28_56_days": "Action drift",
    "recency": "Recency",
    "vehicle_mean_age_overall": "Mean vehicle age",
    "prop_in_prod": "In production (prop)",
    "overall_prop_vehicle_make_audi": "Audi (prop)",
    "overall_prop_vehicle_make_bmw_group": "BMW group (prop)",
    "overall_prop_vehicle_make_premium": "Premium (prop)",
    "overall_prop_vehicle_make_seat": "SEAT (prop)",
    "overall_prop_vehicle_make_skoda": "Škoda (prop)",
    "overall_prop_vehicle_make_volkswagen": "VW (prop)",
}


pretty_new_features = {
    "prop_clear_0_56": "Clear (prop)",
    "prop_coding_0_56": "Coding (prop)",
    "prop_history_screen_0_56": "History screen (prop)",
    "prop_live_data_0_56": "Live data (prop)",
    "prop_oca_0_56": "OCA (prop)",
    "prop_scan_0_56": "Scan (prop)",
    "prop_main_0_56": "Main app (prop)",
    "prop_main_drift_0_28_vs_28_56": "Main app drift",
    "n_sessions_0_28_days": "Sessions (28d)",
    "sessions_intensity_drift_0_28_vs_28_56_days": "Session drift",
    "actions_per_session_0_28_days": "Actions/session (28d)",
    "actions_per_session_intensity_drift_0_28_vs_28_56_days": "Actions/session drift",
    "recency": "Recency",
    "weekend_share_0_28_days": "Weekend share (28d)",
    "usage_concentration": "Usage concentration",
    "vehicle_mean_age_overall": "Mean vehicle age",
    "vehicle_age_sd_overall": "Vehicle age SD",
    "prop_in_prod": "In production (prop)",
    "vehicle_mean_mileage_ordinal": "Mean mileage ordinal",
    "n_new_vehicles_0_28_days": "New vehicles (28d)",
    "days_since_last_new_vehicle": "Days since last new vehicle",
    "overall_prop_vehicle_make_volkswagen": "VW (prop)",
    "overall_prop_vehicle_make_audi": "Audi (prop)",
    "overall_prop_vehicle_make_skoda": "Škoda (prop)",
    "CV_gap_0_56_days": "Gap CV (56d)",
    "sd_gap_0_56_days": "Gap SD (56d)",
    "mean_gap_0_56_days": "Mean gap (56d)",
    "active_flag_0_56_days": "Active flag (56d)",
    "account_tenure_days": "Account tenure",
    "onboarding_delay_days": "Onboarding delay",
    "interval_id": "Interval index (duration bucket)",
}

COLUMS_TO_EXCLUDE: list = [
    "user_id",
    "interval_start",
    "interval_end",
    "churn_triggered_adjusted",
]

# Dropped at Cox fit time (not mere ID/clock columns).
# Activity props and make props are compositional when they sum to 1; the
# reference category from each simplex ("other" / "other_rare") is already
# excluded at feature-engineering time in data_processing.py, so no extra
# category needs to be banned here.
COX_BAN_LIST: list = [
    "recency",
    "CV_gap_0_56_days",
    "account_tenure_days",
]
