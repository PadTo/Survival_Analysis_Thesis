# ============================================================
#  Pipeline configuration (tunable numeric parameters)
# ============================================================
LOOKBACK_PERIODS: tuple[int, ...] = (1, 2, 3)
INTERVAL_IN_DAYS: int = 14
SUM_TO_LIMIT_IN_DAYS: int = 56
FIRST_PERIOD_IN_DAYS: int = 28
SECOND_PERIOD_IN_DAYS: int = 56
N_NEIGHBOURS: int = 5
