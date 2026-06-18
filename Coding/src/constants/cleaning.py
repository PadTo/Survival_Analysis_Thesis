

# ============================================================
#  Output file handling
# ============================================================
CSV_EXTENSION: str               = ".csv"
DEFAULT_OUTPUT_FILENAME: str     = "no_name.csv"
PERSONAL_USERS_FILENAME: str     = "personal_users_dataset.csv"
PROFESSIONAL_USERS_FILENAME: str = "professional_users_dataset.csv"


# ============================================================
#  Tuning defaults
# ============================================================
DEFAULT_THRESHOLD_DAYS: int         = 160
DEFAULT_HHI_THRESHOLD: int          = 6
DEFAULT_CAR_SHARE_ABS: int          = 4
DEFAULT_CAR_SHARE_FRACTION: float   = 0.8
QUANTILE_FILTER: float              = 0.9
BURST_TIME_HR: float                = 1/3
MIN_ACTIVITY_SPAN_DAYS: int         = 1 # This should remain 1 as it filters out one-day users
