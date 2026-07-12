import polars as pl
import pandas as pd
from datetime import timedelta
from pathlib import Path
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from .constants.cleaning import BURST_TIME_HR
from .constants.columns import (
    USER_ID_COL, ACTIVITY_DATE_COL, CHURN_ADJUSTED_DATE_COL,
    CHURN_TRIGGERED_COL, VEHICLE_ID_COL, VEHICLE_MAKE_COL,
    VEHICLE_MODEL_COL, VEHICLE_START_YEAR_COL, VEHICLE_END_YEAR_COL,
    VEHICLE_MILEAGE_COL, APP_COL, ACTIVITY_TYPE_COL,
    STILL_IN_PRODUCTION_COL, INTERVAL_START_COL,
)

# ============================================================
#  Internal column names
# ============================================================
START_DATE_COL             = "start_date"
END_DATE_COL               = "end_date"
N_INTERVALS_COL            = "n_intervals"
TOTAL_ACTIONS_COL          = "total_actions"
EMPTY_RATE_COL             = "empty_interval_rate"
MEAN_INTERVALS_COL         = "mean_intervals"
MEDIAN_INTERVALS_COL       = "median_intervals"
P25_INTERVALS_COL          = "p25_intervals"
P75_INTERVALS_COL          = "p75_intervals"
INTERVAL_DAYS_COL          = "interval_days"


class DataSummary:

    # Currently unused
    def _filter_rows_from_burst(self,
                            df: pl.DataFrame,
                            burst_time_hr: float = BURST_TIME_HR) -> pl.DataFrame:
        """Remove burst rows, then collapse to one row per distinct active day per user."""
        df = df.with_columns(
            pl.col(ACTIVITY_DATE_COL).str.to_datetime(time_unit="us", time_zone="UTC")
        )

        df = df.sort([USER_ID_COL, VEHICLE_ID_COL, ACTIVITY_DATE_COL])

        df = df.with_columns(
            pl.col(ACTIVITY_DATE_COL)
            .shift(1)
            .over([USER_ID_COL, VEHICLE_ID_COL])
            .alias("_shifted_date")
        )

        df = df.with_columns(
            ((pl.col(ACTIVITY_DATE_COL) - pl.col("_shifted_date"))
            >= pl.duration(hours=burst_time_hr))
            .fill_null(True)
            .alias("_is_not_burst")
        )

        df = df.filter(pl.col("_is_not_burst")).drop(["_shifted_date", "_is_not_burst"])

        # Collapse to one row per distinct active day per user
        df = df.group_by(USER_ID_COL).agg(
            pl.col(ACTIVITY_DATE_COL).dt.date().unique()
        ).explode(ACTIVITY_DATE_COL)

        return df

    def _cast_activity_date_col_to_datetime(self,
                                            df: pl.DataFrame,
                                            as_date: bool = True):
        """Parse the activity-date string; as_date drops the time when only the calendar day matters."""
        expr = pl.col(ACTIVITY_DATE_COL).str.to_datetime(time_unit="us", time_zone="UTC")

        if as_date:
            expr = expr.dt.date()

        return df.with_columns(expr)
    
    def _transform_df_to_single_day_activity(self, df: pl.DataFrame):
        """Collapse to one row per distinct active day per user, so repeated same-day use isn't counted as multiple gaps."""
        df = df.group_by(USER_ID_COL).agg(pl.col(ACTIVITY_DATE_COL).unique()).explode(ACTIVITY_DATE_COL)
        return df

    def _build_interval_grid(self,
                             df: pl.DataFrame,
                             interval: int,
                             churn_adjusted_date_col_name: str) -> pl.DataFrame:

        df = df.with_columns(pl.col(churn_adjusted_date_col_name).str
                            .to_datetime(time_unit="us", time_zone="UTC")
                            .dt.date())
        user_min_max_dates = df.group_by(USER_ID_COL).agg(
            pl.col(churn_adjusted_date_col_name).min().alias(START_DATE_COL),
            pl.col(churn_adjusted_date_col_name).max().alias(END_DATE_COL)
        )

        intervals = user_min_max_dates.with_columns(
            pl.date_ranges(
                start=pl.col(START_DATE_COL),
                end=pl.col(END_DATE_COL),
                interval=timedelta(days=interval)
            ).alias(INTERVAL_START_COL)
        ).explode(INTERVAL_START_COL).select([USER_ID_COL, INTERVAL_START_COL])

        # Backward asof: each activity attaches to the most recent interval start at or before its date
        df_with_intervals = df.join_asof(
            intervals,
            left_on=churn_adjusted_date_col_name,
            right_on=INTERVAL_START_COL,
            by=USER_ID_COL,
            strategy="backward"
        )

        # Re-joining from the full interval grid reintroduces intervals that had zero activity
        df_with_intervals = intervals.join(
            df_with_intervals,
            on=[USER_ID_COL, INTERVAL_START_COL],
            how="left"
        )

        return df_with_intervals

    def return_interval_summary(self,
                                df: pl.DataFrame,
                                interval_sizes: list[int],
                                churn_adjusted_date_col_name: str | None = None) -> pd.DataFrame:
        """
        Summarises how interval width affects the number of intervals per user
        and the empty-interval rate.

        The two statistics together justify the chosen interval width: more intervals
        per user gives more time-points for the model, but a high empty-interval rate
        means most of those time-points carry no signal.
        """
        churn_adjusted_date_col_name = churn_adjusted_date_col_name or CHURN_ADJUSTED_DATE_COL

        results = []

        for interval in interval_sizes:
            df_with_intervals = self._build_interval_grid(df, interval, churn_adjusted_date_col_name)

            interval_counts = (
                df_with_intervals
                .group_by(USER_ID_COL)
                .agg(pl.col(INTERVAL_START_COL).n_unique().alias(N_INTERVALS_COL))
            )

            # Empty interval = no activity logged in that window
            empty_interval_rate = (
                df_with_intervals
                .group_by([USER_ID_COL, INTERVAL_START_COL])
                .agg(pl.col(ACTIVITY_TYPE_COL).is_not_null().sum().alias(TOTAL_ACTIONS_COL))
                .with_columns((pl.col(TOTAL_ACTIONS_COL) == 0).alias("is_empty"))
                .select(pl.col("is_empty").mean().alias(EMPTY_RATE_COL))
                .item()
            )

            stats = interval_counts.select([
                pl.col(N_INTERVALS_COL).mean().alias(MEAN_INTERVALS_COL),
                pl.col(N_INTERVALS_COL).median().alias(MEDIAN_INTERVALS_COL),
                pl.col(N_INTERVALS_COL).quantile(0.25).alias(P25_INTERVALS_COL),
                pl.col(N_INTERVALS_COL).quantile(0.75).alias(P75_INTERVALS_COL),
            ]).to_pandas()

            stats[INTERVAL_DAYS_COL] = interval
            stats[EMPTY_RATE_COL]    = round(empty_interval_rate, 4)

            results.append(stats)

        return pd.concat(results, ignore_index=True)[[
            INTERVAL_DAYS_COL, MEAN_INTERVALS_COL, MEDIAN_INTERVALS_COL,
            P25_INTERVALS_COL, P75_INTERVALS_COL, EMPTY_RATE_COL
        ]]
    
    def return_activity_gap_summary(self,
                                df: pl.DataFrame,
                                percentiles: list[float] | None = None) -> pd.DataFrame:
        """
        Summarises the inter-activity gap distribution.

        Collapsed to one row per distinct active day per user before computing gaps,
        so repeated same-day use is not counted as multiple gap events.
        """
        percentiles = percentiles or [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        
        df = self._cast_activity_date_col_to_datetime(df)
        df = self._transform_df_to_single_day_activity(df)
        
        df = df.sort([USER_ID_COL, ACTIVITY_DATE_COL])

        df = df.with_columns(
            pl.col(ACTIVITY_DATE_COL)
            .shift(1)
            .over(USER_ID_COL)
            .alias("_prev_date")
        )

        gaps = (
            df.with_columns(
                (pl.col(ACTIVITY_DATE_COL) - pl.col("_prev_date"))
                .dt.total_days()
                .alias("gap_days")
            )
            .filter(pl.col("gap_days").is_not_null())
            .select("gap_days")
            .to_pandas()["gap_days"]
        )

        return gaps.describe(percentiles=percentiles).to_frame("gap_days")

    def compute_vif(self, df: pl.DataFrame, pretty: dict, id_label_cols: list):
        features = df.drop(id_label_cols).to_pandas()
        # Booleans break np.isfinite inside variance_inflation_factor; cast to int
        bool_cols = features.select_dtypes(include=["bool"]).columns
        features[bool_cols] = features[bool_cols].astype(int)
        n_before = len(features)
        features = features.dropna()  # VIF can't handle NaNs
        n_dropped = n_before - len(features)
        if n_dropped:
            print(f"  Dropped {n_dropped} rows with NaNs ({len(features):,} remaining)")
        X = add_constant(features)  # intercept needed for correct VIF
        vif = pd.DataFrame({
            "feature": X.columns,
            "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        })
        vif = vif[vif["feature"] != "const"].copy()
        vif["feature"] = vif["feature"].map(lambda c: pretty.get(c, c))
        return vif.sort_values("VIF", ascending=False).reset_index(drop=True)