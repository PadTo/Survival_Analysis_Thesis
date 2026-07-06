import pandas as pd
from pathlib import Path
import polars as pl
from datetime import timedelta
from .constants.variables_groupings import activity_type_groups, app_type_groups, brand_group_map
from .imputing import Imputer
from .constants.columns import (
    USER_ID_COL, ACTIVITY_DATE_COL, CHURN_ADJUSTED_DATE_COL,
    CHURN_TRIGGERED_COL, VEHICLE_ID_COL, VEHICLE_MAKE_COL,
    VEHICLE_MODEL_COL, VEHICLE_START_YEAR_COL, VEHICLE_END_YEAR_COL,
    VEHICLE_MILEAGE_COL, APP_COL, ACTIVITY_TYPE_COL, ACTIVITY_DATE_COL,
    STILL_IN_PRODUCTION_COL, INTERVAL_START_COL
)
from .constants.processor import (
    LOOKBACK_PERIODS, INTERVAL_IN_DAYS, SUM_TO_LIMIT_IN_DAYS,
    FIRST_PERIOD_IN_DAYS, SECOND_PERIOD_IN_DAYS, INTERVAL_END_COL,
    CHURN_TRIGGERED_SHIFTED_COLUMN_NAME)
 

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
ACTION_PER_SESSION_COL_TEMPLATE: str = "actions_per_session_{start}_{end}_days"

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
START_DATE_COL: str = "start_date"
END_DATE_COL: str = "end_date"
LAST_ACTIVITY_DATE_COL: str = "last_activity_date"
RECENCY_COL: str = "recency"
PROP_IN_PROD_COL: str = "prop_in_prod"
CUMULATIVE_COUNT_TOTAL_COL: str = "cumulative_count_total"
MEAN_AGE_INTERMEDIATE_COL: str = "vehicle_mean_age_intermediate"
VEHICLE_AGE_COLUMN_NAME: str = "vehicle_age"
VEHICLE_MEAN_AGE_COLUMN_NAME: str = "vehicle_mean_age_in_an_interval"
VEHICLE_MEAN_OVERALL_AGE_COLUMN_NAME: str = "vehicle_mean_age_overall"
FIRST_DISTINCT_COLUMN_NAME: str = "first_distinct"


class DataProcessor():

    def __init__(self, df: pd.DataFrame | pl.DataFrame) -> None:
        self.df = df.copy(deep=True) if type(df) == pd.DataFrame else df
        

    def _prepare_df(self,
                    df: pl.DataFrame,
                    churn_adjusted_date_col_name: str | None = None,
                    activity_date_col_name: str | None = None) -> tuple[pl.DataFrame, list]:
        """
        Parses dates, normalises categorical text, groups categories, and one-hot
        encodes vehicle make.

        Returns the prepared frame plus the generated make dummy column names, since
        downstream aggregation needs to know which columns the one-hot step created.
        """

        churn_adjusted_date_col_name = churn_adjusted_date_col_name or CHURN_ADJUSTED_DATE_COL
        activity_date_col_name = activity_date_col_name or ACTIVITY_DATE_COL

        # Sorting by user and date: the interval grid and downstream join_asof both assume chronological order per user
        df = df.with_columns(
            pl.col(churn_adjusted_date_col_name)
            .str.to_datetime(time_unit="us", time_zone="UTC")
            .dt.date()
            .alias(churn_adjusted_date_col_name)
        ).sort(by=[USER_ID_COL, ACTIVITY_DATE_COL])

        df = df.with_columns(
            pl.col(activity_date_col_name)
            .str.to_datetime(time_unit="us", time_zone="UTC")
            .dt.date()
            .alias(activity_date_col_name)
        ).sort(by=[USER_ID_COL, ACTIVITY_DATE_COL])

        # Collapsing raw categories into their groupings so counts aggregate at the group level, not the raw-value level
        df = df.with_columns(
            pl.col(ACTIVITY_TYPE_COL)
                .str.to_lowercase()
                .str.strip_chars()
                .replace(activity_type_groups),
            pl.col(APP_COL)
                .str.to_lowercase()
                .str.strip_chars()
                .replace(app_type_groups),
            pl.col(VEHICLE_MAKE_COL)
                .str.to_lowercase()
                .str.strip_chars()
                .replace(brand_group_map)
        )

        # Creating an age column of a vehicle later needed when computing mean age of fleet per user per history
        df = df.with_columns((pl.col(VEHICLE_END_YEAR_COL) - pl.col(VEHICLE_START_YEAR_COL)).alias(VEHICLE_AGE_COLUMN_NAME))

        # Setting aside vehicle make column to concatenate it later (needed for mean age feature aggregation computations)
        df_make_col = df.select(pl.col(VEHICLE_MAKE_COL))

        # One-hot encoding make so each group becomes a 0/1 column that can be summed per interval
        df = df.to_dummies(VEHICLE_MAKE_COL)
        df = pl.concat([df, df_make_col], how="horizontal")

        vehicle_make_column_names = [c for c in df.columns if (c.lower().strip()).startswith(f"{VEHICLE_MAKE_COL}_")]


        return (df, vehicle_make_column_names)

    def _generate_intervals(self,
                            df: pl.DataFrame,
                            activity_date_col_name: str | None = None,
                            interval: int = INTERVAL_IN_DAYS,
                            vehicle_make_column_names: list | None = None) -> pl.DataFrame:
        """
        Builds a per-user grid of fixed-width intervals and assigns each activity
        row to its interval.

        The grid is the counting-process backbone: every user gets a row per
        interval across their observed span, including intervals with no activity,
        because an empty interval (no usage) is itself the signal of interest.
        """

        activity_date_col_name = activity_date_col_name or ACTIVITY_DATE_COL

        user_min_max_dates = df.group_by(USER_ID_COL).agg(
            pl.col(activity_date_col_name).min().alias(START_DATE_COL),
            pl.col(activity_date_col_name).max().alias(END_DATE_COL)
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
            left_on=activity_date_col_name,
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

        df_with_intervals = self._fill_null_values(df_with_intervals,
                                                   vehicle_make_column_names)
        
        return df_with_intervals

    def _fill_null_values(self,
                          df: pl.DataFrame,
                          vehicle_make_column_names: list | None):
        """
        Fills nulls left by the empty intervals introduced in the interval grid.

        Empty intervals carry no activity, so their fills encode "no activity"
        explicitly: counts and flags go to a zero/False/sentinel value rather than
        being left null, so later sums and comparisons treat them as inactivity.
        """

        col_fill_list = [
            pl.col(CHURN_TRIGGERED_COL).fill_null(False),
            pl.col(VEHICLE_ID_COL).fill_null(UNKNOWN_VALUE),
            pl.col(VEHICLE_MODEL_COL).fill_null(UNKNOWN_VALUE),
            pl.col(VEHICLE_MAKE_COL).fill_null(UNKNOWN_VALUE),
            pl.col(VEHICLE_START_YEAR_COL).fill_null(0),
            pl.col(VEHICLE_END_YEAR_COL).fill_null(0),
            pl.col(VEHICLE_MILEAGE_COL).fill_null(UNKNOWN_VALUE),
            pl.col(ACTIVITY_DATE_COL).fill_null(pl.lit(None).cast(pl.Date)),
            pl.col(APP_COL).fill_null(UNKNOWN_VALUE),
            pl.col(ACTIVITY_TYPE_COL).fill_null(NONE_ACTIVITY_VALUE),
            pl.col(STILL_IN_PRODUCTION_COL).fill_null(False),
            pl.col(CHURN_ADJUSTED_DATE_COL).fill_null(pl.lit(None).cast(pl.Date))]

        # Make dummies are absent on empty intervals; zero means the make was not used that interval
        if vehicle_make_column_names:
            col_fill_list += [pl.col(vehicle_make).fill_null(0) for vehicle_make in vehicle_make_column_names]

        df = df.with_columns(col_fill_list)

        return df

    def _build_lagged_columns(self,
                              base_columns: dict[str, str],
                              lookback_periods: tuple[int, ...] = LOOKBACK_PERIODS,
                              interval_in_days: int = INTERVAL_IN_DAYS,
                              fill_value: int | None = None) -> tuple[list, dict[tuple[int, int], list[str]]]:
        """
        Builds shift expressions and a window registry for a set of base columns.

        For each base column at the current (0, interval) window, produces a lagged
        copy per lookback period and records it in a registry keyed by (start, end)
        day window. The registry lets later steps select columns by time range
        without re-deriving column names by hand.

        Args:
            base_columns:     token -> source column name at the (0, interval) window.
            lookback_periods: Shift periods, expressed in intervals.
            interval_in_days: Width of each interval in days.
            fill_value:       Fill for the shift. None leaves shifted-out rows null.

        Returns:
            Tuple of (lag_expressions, col_registry).
        """
        lag_expressions: list = []

        # Registering the current window first so range selections can include the unshifted columns
        col_registry: dict[tuple[int, int], list[str]] = {
            (0, interval_in_days): [
                COL_TEMPLATE_FORMAT.format(a=token, start=0, end=interval_in_days)
                for token in base_columns.keys()
            ]
        }

        for period in lookback_periods:

            start_interval = interval_in_days * period
            end_interval = interval_in_days * (period + 1)

            for token, source_col in base_columns.items():
                # Filling shifted-out rows with 0 where requested: an absent prior interval is no activity, not a missing measurement
                if fill_value is not None:
                    shifted = pl.col(source_col).shift(period, fill_value=fill_value)
                else:
                    shifted = pl.col(source_col).shift(period)

                lag_expressions.append(
                    shifted.alias(COL_TEMPLATE_FORMAT.format(a=token, start=start_interval, end=end_interval))
                )

            col_registry[(start_interval, end_interval)] = [
                COL_TEMPLATE_FORMAT.format(a=token, start=start_interval, end=end_interval)
                for token in base_columns
            ]

        return lag_expressions, col_registry

    def _columns_within_window(self,
                           col_registry: dict[tuple[int, int], list[str]],
                           start_from_days: int = 0,
                           end_at_days: int | None = None,
                           token_filter: str | None = None) -> list[str]:
        """
        Selects registry columns whose window falls within [start_from_days,
        end_at_days], optionally restricted to names containing token_filter.

        Used to assemble the set of lagged columns that sum into a given time
        window, independent of how many intervals fit inside that window.
        """

        return [
            col
            for (start, end), cols in col_registry.items()
            if start >= start_from_days and (end_at_days is None or end <= end_at_days)
            for col in cols
            if token_filter is None or token_filter in col
        ]

    def _generate_activity_feature_aggregation(self,

                                           lookback_periods: tuple[int, ...] = LOOKBACK_PERIODS,
                                           interval_in_days: int = INTERVAL_IN_DAYS,
                                           sum_to_limit_in_days: int = SUM_TO_LIMIT_IN_DAYS) -> tuple[list, ...]:
        """
        Builds the staged expressions for activity-type composition features.

        Returns four expression lists meant to be applied in order, because each
        stage references columns the previous stage created:
            agg         per-activity counts in the current interval
            post_agg_1  those counts lagged across the lookback periods
            post_agg_2  per-activity and total counts summed over the lookback window
            post_agg_3  per-activity proportions of the windowed total

        Only the proportions are kept as features; the counts are intermediates.

        Args:
            lookback_periods:     Shift periods in intervals, e.g. (1, 2, 3).
            interval_in_days:     Width of each interval in days. Default INTERVAL_IN_DAYS.
            sum_to_limit_in_days: Upper day bound of the window summed into the
                                  final proportions. Default 56.

        Returns:
            Tuple of (agg, post_agg_1, post_agg_2, post_agg_3, column_names_to_keep).
        """

        post_agg_1: list = []
        post_agg_2: list = []
        post_agg_3: list = []
        column_names_to_keep: list = []
        # Sorting the activity groups so the produced column order is stable across runs (set iteration order is not)
        activity_types: list[str] = list(sorted(set(activity_type_groups.values())))

        agg: list = [
            (pl.col(ACTIVITY_TYPE_COL) == a).sum().alias(
                COL_TEMPLATE_FORMAT.format(a=a, start=0, end=interval_in_days))
            for a in activity_types
        ]

        base_columns = {
            a: COL_TEMPLATE_FORMAT.format(a=a, start=0, end=interval_in_days)
            for a in activity_types
        }
        post_agg_1, col_registry = self._build_lagged_columns(
            base_columns, lookback_periods, interval_in_days, fill_value=0)

        columns_to_sum_over_for_total_activity_count = self._columns_within_window(
            col_registry, end_at_days=sum_to_limit_in_days)

        # Total over the window is the proportion denominator shared by every activity type
        total_actions_col_name = TOTAL_ACTIONS_COL_TEMPLATE.format(start=0, end=sum_to_limit_in_days)
        post_agg_2 += [
            pl.sum_horizontal([pl.col(c) for c in columns_to_sum_over_for_total_activity_count])
            .alias(total_actions_col_name)
        ]

        for activity_type in activity_types:

            columns = self._columns_within_window(
                col_registry, end_at_days=sum_to_limit_in_days, token_filter=activity_type)

            column_name = COL_TEMPLATE_FORMAT.format(a=activity_type, start=0, end=sum_to_limit_in_days)
            post_agg_2 += [
                pl.sum_horizontal([pl.col(c) for c in columns])
                .alias(column_name)
            ]

            prop_column_name = PROP_COL_TEMPLATE.format(a=activity_type, start=0, end=sum_to_limit_in_days)
            # fill_nan(0): an inactive window gives 0/0, which reads as zero share of that activity
            post_agg_3 += [
                (pl.col(column_name) / pl.col(total_actions_col_name)).alias(prop_column_name).fill_nan(0),
            ]

            if prop_column_name != "prop_other_0_56":
                column_names_to_keep += [prop_column_name]

        return agg, post_agg_1, post_agg_2, post_agg_3, column_names_to_keep

    def _generate_vehicle_characteristics_features_aggregation(self,
                                                          vehicle_make_col_names: list,
                                                          interval_in_days: int = INTERVAL_IN_DAYS) -> tuple[list, ...]:
        """
            Builds make-portfolio features from the one-hot make columns.

            Counts accumulate cumulatively per user rather than resetting per interval:
            the set of makes a user has connected is a slow-moving characteristic of who
            they are, not a per-window behaviour. From the running counts it derives each
            make's overall share. Cumulative counts themselves are intermediates and are
            not kept.

            Returns:
                Tuple of (agg, post_agg_1, post_agg_2, post_agg_3, column_names_to_keep).
        """

        agg: list = []
        post_agg_1 = []
        post_agg_2 = []
        post_agg_3 = []
        column_names_to_keep: list = []
        cum_count_col_names: list = []

        ##___MEAN AGE___##
        agg += [(pl.col(VEHICLE_AGE_COLUMN_NAME))
                .filter((pl.col(FIRST_DISTINCT_COLUMN_NAME) == True )
                        &
                        (pl.col(VEHICLE_MAKE_COL) != UNKNOWN_VALUE))
                .mean()
                .alias(VEHICLE_MEAN_AGE_COLUMN_NAME)]

        post_agg_1 += [(pl.col(VEHICLE_MEAN_AGE_COLUMN_NAME)
                        .cum_sum()
                        .truediv(pl.col(VEHICLE_MEAN_AGE_COLUMN_NAME).cum_count()))
                        .alias(MEAN_AGE_INTERMEDIATE_COL)]  # Can't forward fill inplace -> assigning an intermediate column

        # Null values for intervals with no activity -> forward fill to carry information from
        # previous periods
        post_agg_2 += [pl.col(MEAN_AGE_INTERMEDIATE_COL)
                       .forward_fill()
                       .alias(VEHICLE_MEAN_OVERALL_AGE_COLUMN_NAME)
                       .over(USER_ID_COL)]

        column_names_to_keep += [VEHICLE_MEAN_OVERALL_AGE_COLUMN_NAME]

        ##__STILL IN PRODUCTION PROPORTIONS___##

        still_in_prod_count = COL_TEMPLATE_FORMAT.format(a=STILL_IN_PRODUCTION_TOKEN,
                                                  start=0,
                                                  end=interval_in_days)
        total_car_count = COL_TEMPLATE_FORMAT.format(a=TOTAL_CARS_TOKEN,
                                                  start=0,
                                                  end=interval_in_days)

        agg += [pl.col(STILL_IN_PRODUCTION_COL)
                .filter((pl.col(FIRST_DISTINCT_COLUMN_NAME) == True)
                        &
                        (pl.col(VEHICLE_MAKE_COL) != UNKNOWN_VALUE))
                .sum()
                .alias(still_in_prod_count),

                pl.col(STILL_IN_PRODUCTION_COL)
                .filter((pl.col(FIRST_DISTINCT_COLUMN_NAME) == True)
                        &
                        (pl.col(VEHICLE_MAKE_COL) != UNKNOWN_VALUE))
                .count()
                .alias(total_car_count)]

        
        post_agg_1 += [(pl.col(still_in_prod_count).cum_sum().over(USER_ID_COL) /
                        pl.col(total_car_count).cum_sum().over(USER_ID_COL))
                        .fill_nan(0)
                        .alias(PROP_IN_PROD_COL)]


        column_names_to_keep += [PROP_IN_PROD_COL]


        ##___MAKE PROPORTIONS___##
        for vehicle_make_col_name in vehicle_make_col_names:
            
            n_count_col_name = COL_TEMPLATE_FORMAT.format(
                                                a=vehicle_make_col_name,
                                                start=0,
                                                end=interval_in_days)

            cum_count_col_name = CUMULATIVE_COUNT_COL_TEMPLATE.format(
                name=vehicle_make_col_name)

            cum_count_col_names += [cum_count_col_name]

            agg += [(pl.col(vehicle_make_col_name) == 1).sum().alias(n_count_col_name)]

            # Accumulating per user so the count reflects the whole garage seen up to each interval
            post_agg_1 += [pl.col(n_count_col_name)
                        .cum_sum()
                        .over(pl.col(USER_ID_COL))
                        .alias(cum_count_col_name)]


        # Total across makes is the denominator for the per-make shares
        post_agg_2 += [
            pl.sum_horizontal([pl.col(c) for c in cum_count_col_names])
            .alias(CUMULATIVE_COUNT_TOTAL_COL)
        ]

        post_agg_3 += [
            (pl.col(col) / pl.col(CUMULATIVE_COUNT_TOTAL_COL))
            .fill_nan(0)
            .alias(OVERALL_PROP_COL_TEMPLATE.format(name=col.replace("cumulative_count_", "")))
            for col in cum_count_col_names
        ]

        column_names_to_keep += [
            OVERALL_PROP_COL_TEMPLATE.format(name=col.replace("cumulative_count_", ""))
            for col in cum_count_col_names if col != "cumulative_count_vehicle_make_other_rare"
        ]
  

        return agg, post_agg_1, post_agg_2, post_agg_3, column_names_to_keep

    def _generate_app_column_features_aggregation(self,
                                             lookback_periods: tuple[int, ...] = LOOKBACK_PERIODS,
                                             interval_in_days: int = INTERVAL_IN_DAYS,
                                             sum_to_limit_in_days: int = SUM_TO_LIMIT_IN_DAYS,
                                             first_period: int = FIRST_PERIOD_IN_DAYS,
                                             second_period: int = SECOND_PERIOD_IN_DAYS) -> tuple[list, ...]:
        """
        Builds app-usage composition and migration features.

        Produces each app's share over the full lookback window, plus a drift on the
        main app: its share in the recent window minus its share in the prior window,
        which captures users migrating between the two apps. Drift lands in a fourth
        stage because it subtracts two proportions created in the third stage.

        Returns:
            Tuple of (agg, post_agg_1, post_agg_2, post_agg_3, post_agg_4, column_names_to_keep).
        """
        post_agg_1: list = []
        post_agg_2: list = []
        post_agg_3: list = []
        column_names_to_keep: list = []
        apps: list[str] = list(app_type_groups.keys())

        agg: list = [
            (pl.col(APP_COL) == app).sum().alias(
                COL_TEMPLATE_FORMAT.format(a=app, start=0, end=interval_in_days))
            for app in apps
        ]

        base_columns = {
            app: COL_TEMPLATE_FORMAT.format(a=app, start=0, end=interval_in_days)
            for app in apps
        }
        post_agg_1, col_registry = self._build_lagged_columns(
            base_columns, lookback_periods, interval_in_days, fill_value=0)

        columns_to_sum_over_for_total_app_count = self._columns_within_window(
            col_registry, end_at_days=sum_to_limit_in_days)

        # Per-window totals act as the denominators for the windowed main-app proportions feeding the drift
        columns_to_sum_first_window_app = self._columns_within_window(
            col_registry, start_from_days=0, end_at_days=first_period)

        columns_to_sum_second_window_app = self._columns_within_window(
            col_registry, start_from_days=first_period, end_at_days=second_period)

        total_app_col_name_0_56  = COL_TEMPLATE_FORMAT.format(a=TOTAL_APP_TOKEN, start=0,            end=sum_to_limit_in_days)
        total_app_0_28_col_name  = COL_TEMPLATE_FORMAT.format(a=TOTAL_APP_TOKEN, start=0,            end=first_period)
        total_app_28_56_col_name = COL_TEMPLATE_FORMAT.format(a=TOTAL_APP_TOKEN, start=first_period, end=second_period)

        post_agg_2 += [
            pl.sum_horizontal([pl.col(c) for c in columns_to_sum_over_for_total_app_count]).alias(total_app_col_name_0_56),
            pl.sum_horizontal([pl.col(c) for c in columns_to_sum_first_window_app]).alias(total_app_0_28_col_name),
            pl.sum_horizontal([pl.col(c) for c in columns_to_sum_second_window_app]).alias(total_app_28_56_col_name),
        ]

        # Only the main app is summed per window; with two apps the other is its complement, so both shares would be perfectly collinear
        main_token = list(base_columns.keys())[0]

        cols_main_0_28  = self._columns_within_window(col_registry, start_from_days=0,            end_at_days=first_period,  token_filter=main_token)
        cols_main_28_56 = self._columns_within_window(col_registry, start_from_days=first_period, end_at_days=second_period, token_filter=main_token)

        main_0_28_col  = COL_TEMPLATE_FORMAT.format(a=main_token, start=0,            end=first_period)
        main_28_56_col = COL_TEMPLATE_FORMAT.format(a=main_token, start=first_period, end=second_period)

        post_agg_2 += [
            pl.sum_horizontal([pl.col(c) for c in cols_main_0_28]).alias(main_0_28_col),
            pl.sum_horizontal([pl.col(c) for c in cols_main_28_56]).alias(main_28_56_col),
        ]

        for app in apps:
            columns = self._columns_within_window(
                col_registry, end_at_days=sum_to_limit_in_days, token_filter=app)

            column_name_0_56 = COL_TEMPLATE_FORMAT.format(a=app, start=0, end=sum_to_limit_in_days)
            prop_column_name_0_56 = PROP_COL_TEMPLATE.format(a=app, start=0, end=sum_to_limit_in_days)

            post_agg_2 += [
                pl.sum_horizontal([pl.col(c) for c in columns]).alias(column_name_0_56)
            ]

            # fill_nan(0): an inactive window gives 0/0, read as zero share of that app
            post_agg_3 += [
                (pl.col(column_name_0_56) / pl.col(total_app_col_name_0_56))
                .alias(prop_column_name_0_56).fill_nan(0)
            ]

        prop_main_0_28_col  = PROP_COL_TEMPLATE.format(a=main_token, start=0,            end=first_period)
        prop_main_28_56_col = PROP_COL_TEMPLATE.format(a=main_token, start=first_period, end=second_period)
        prop_main_0_56_col  = PROP_COL_TEMPLATE.format(a=main_token, start=0,            end=second_period)
        main_drift_col      = DRIFT_COL_TEMPLATE.format(a=main_token, r0=0, r1=first_period, r2=second_period)

        post_agg_3 += [
            (pl.col(main_0_28_col)  / pl.col(total_app_0_28_col_name)).alias(prop_main_0_28_col).fill_nan(0),
            (pl.col(main_28_56_col) / pl.col(total_app_28_56_col_name)).alias(prop_main_28_56_col).fill_nan(0),
        ]

        # Drift sits in its own pass because it consumes the two windowed proportions built just above
        post_agg_4: list = [
            (pl.col(prop_main_0_28_col) - pl.col(prop_main_28_56_col)).alias(main_drift_col)
        ]

        column_names_to_keep += [prop_main_0_56_col, main_drift_col]

        return agg, post_agg_1, post_agg_2, post_agg_3, post_agg_4, column_names_to_keep

    def _generate_behaviour_features_aggregation(self,
                                    lookback_periods: tuple[int, ...] = LOOKBACK_PERIODS,
                                    interval_in_days: int = INTERVAL_IN_DAYS,
                                    first_period: int = FIRST_PERIOD_IN_DAYS,
                                    second_period: int = SECOND_PERIOD_IN_DAYS) -> tuple[list, ...]:
        """
        Builds volume, recency, and engagement-trend features.

        Sessions (distinct active days) and actions are counted per interval, lagged,
        and summed into recent and prior windows. Each yields a recent-window level
        plus a drift (recent minus prior) that signals whether engagement is rising
        or falling. Recency is days from the last activity to the interval end.

        Returns:
            Tuple of (agg, post_agg_1, post_agg_2, post_agg_3, column_names_to_keep).
        """
        agg: list = []
        post_agg_1: list = []
        post_agg_2: list = []
        post_agg_3: list = []
        column_names_to_keep: list = []


        session_base_col = COL_TEMPLATE_FORMAT.format(a=SESSIONS_TOKEN, start=0, end=interval_in_days)
        action_base_col = COL_TEMPLATE_FORMAT.format(a=ACTIONS_TOKEN, start=0, end=interval_in_days)

        agg = [
            pl.col(ACTIVITY_DATE_COL).filter(pl.col(ACTIVITY_DATE_COL)
            .is_not_null())
            .max()
            .alias(LAST_ACTIVITY_DATE_COL),

            # Counting unique days, not rows, so multiple actions on one day count as a single session
            pl.col(ACTIVITY_DATE_COL).filter(pl.col(ACTIVITY_DATE_COL)
            .is_not_null())
            .n_unique()
            .alias(session_base_col),

            pl.col(ACTIVITY_TYPE_COL).filter(pl.col(ACTIVITY_TYPE_COL) != NONE_ACTIVITY_VALUE)
            .count()
            .alias(action_base_col)

        ]

        post_agg_1 = [
            # Carrying the last activity date forward so empty intervals keep the most recent real date for recency
            pl.col(LAST_ACTIVITY_DATE_COL).forward_fill().over(USER_ID_COL),
            # Interval end is the recency reference point: how stale the user is by the close of the interval
            (pl.col(INTERVAL_START_COL) + timedelta(days=interval_in_days)).alias(INTERVAL_END_COL)]

        session_lags, session_registry = self._build_lagged_columns(
            {SESSIONS_TOKEN: session_base_col}, lookback_periods, interval_in_days, fill_value=0)
        post_agg_1 += session_lags

        action_lags, action_registry = self._build_lagged_columns(
            {ACTIONS_TOKEN: action_base_col}, lookback_periods, interval_in_days, fill_value=0)
        post_agg_1 += action_lags


        # --- Sessions: recent vs prior window + drift ---
        columns_to_sum_recent_window_sessions = self._columns_within_window(
            session_registry,
            start_from_days=0,
            end_at_days=first_period)

        columns_to_sum_prior_window_sessions = self._columns_within_window(
            session_registry,
            start_from_days=first_period,
            end_at_days=second_period)

        recent_window_session_col = COL_TEMPLATE_FORMAT.format(a=SESSIONS_TOKEN,
                                                               start=0,
                                                               end=first_period)

        prior_window_session_col = COL_TEMPLATE_FORMAT.format(a=SESSIONS_TOKEN,
                                                              start=first_period,
                                                              end=second_period)

        post_agg_2 += [pl.sum_horizontal([col for col in columns_to_sum_recent_window_sessions])
                    .alias(recent_window_session_col),
                    pl.sum_horizontal([col for col in columns_to_sum_prior_window_sessions])
                    .alias(prior_window_session_col)]

        # Casting to signed Int32: session counts are u32, so a falling drift would underflow to a large positive number
        session_intensity_drift_col = INTENSITY_DRIFT_COL_TEMPLATE.format(a=SESSIONS_TOKEN, r0=0, r1=first_period, r2=second_period)
        post_agg_3 += [(pl.col(recent_window_session_col).cast(pl.Int32) - pl.col(prior_window_session_col).cast(pl.Int32)).alias(session_intensity_drift_col)]


        # --- Actions: recent vs prior window + drift ---
        columns_to_sum_recent_window_actions = self._columns_within_window(
            action_registry,
            start_from_days=0,
            end_at_days=first_period)

        columns_to_sum_prior_window_actions = self._columns_within_window(
            action_registry,
            start_from_days=first_period,
            end_at_days=second_period)

        recent_window_action_col = COL_TEMPLATE_FORMAT.format(a=ACTIONS_TOKEN,
                                                             start=0,
                                                             end=first_period)

        prior_window_action_col = COL_TEMPLATE_FORMAT.format(a=ACTIONS_TOKEN,
                                                            start=first_period,
                                                            end=second_period)

        post_agg_2 += [pl.sum_horizontal([col for col in columns_to_sum_recent_window_actions])
                    .alias(recent_window_action_col),
                    pl.sum_horizontal([col for col in columns_to_sum_prior_window_actions])
                    .alias(prior_window_action_col)]

        # Casting to signed Int32 for the same underflow reason as the session drift
        action_intensity_drift_col = INTENSITY_DRIFT_COL_TEMPLATE.format(a=ACTIONS_TOKEN, r0=0, r1=first_period, r2=second_period)
        post_agg_3 += [(pl.col(recent_window_action_col).cast(pl.Int32) - pl.col(prior_window_action_col).cast(pl.Int32)).alias(action_intensity_drift_col)]


        post_agg_2 += [(pl.col(INTERVAL_END_COL) - pl.col(LAST_ACTIVITY_DATE_COL)).dt.total_days().alias(RECENCY_COL)]

        # Actions per session  (depth) to decouple colinearity between session and action counts
        actions_per_session_recent = ACTION_PER_SESSION_COL_TEMPLATE.format(start  = 0,            end=first_period)
        actions_per_session_prior = ACTION_PER_SESSION_COL_TEMPLATE.format( start  = first_period, end=second_period)
        action_per_session_drift = INTENSITY_DRIFT_COL_TEMPLATE.format(a="actions_per_session", r0=0, r1=first_period, r2=second_period)

        post_agg_2 += [(pl.col(recent_window_action_col) / pl.col(recent_window_session_col))
                       .fill_nan(0)
                       .alias(actions_per_session_recent),
                       (pl.col(prior_window_action_col) / pl.col(prior_window_session_col)).
                       fill_nan(0)
                       .alias(actions_per_session_prior)]
        
        post_agg_3 +=[(pl.col(actions_per_session_recent) - pl.col(actions_per_session_prior)).alias(action_per_session_drift)]

        column_names_to_keep += [
            recent_window_session_col, # Volume
            session_intensity_drift_col, # Volume drift
            actions_per_session_recent, # Depth
            action_per_session_drift, # Depth drift
            RECENCY_COL,
            INTERVAL_END_COL]


        return agg, post_agg_1, post_agg_2, post_agg_3, column_names_to_keep

    def _generate_churn_triggered_features_agg(self)-> tuple[list,...]:
        
        agg: list = []
        post_agg_1: list = []
        column_names_to_keep: list = []


        agg += [pl.col(CHURN_TRIGGERED_COL).max()]


        # First-interval churners can survive the upstream span filter.
        # This keeps their label on the first interval instead of shifting it into a null.
        post_agg_1 += [
            pl.col(CHURN_TRIGGERED_COL)
            # Even though churn is triggered in the last period, the feature values
            # must be taken from the previous period (shifting for correct assignment)
            .shift(-1) # Depends on the prior sort, won't work if the dataset isn't correctly sorted
            .over(USER_ID_COL)
            .fill_null(False) # This does not matter as much because the last row will be removed either way (incomplete intervals)
            .alias(CHURN_TRIGGERED_SHIFTED_COLUMN_NAME)]


        
        column_names_to_keep += [CHURN_TRIGGERED_SHIFTED_COLUMN_NAME]
        return agg, post_agg_1, column_names_to_keep


    def _transform_intervals_to_start_stop(self):
        
        post_agg_1: list = []
        post_agg_2: list = []
        
        post_agg_1 = [(pl.col(INTERVAL_START_COL) - pl.col(INTERVAL_START_COL).min().over(USER_ID_COL)).dt.total_days().alias(INTERVAL_START_COL)]
        post_agg_2 = [pl.col(INTERVAL_START_COL).shift(-1).over(USER_ID_COL).alias(INTERVAL_END_COL)]
      
        return post_agg_1, post_agg_2

    def apply_feature_engineering(self,
                                   df: pl.DataFrame,
                                   churn_adjusted_date_col_name: str | None = None,
                                   interval_in_days: int = INTERVAL_IN_DAYS,
                                   lookback_periods: tuple[int, ...] = LOOKBACK_PERIODS,
                                   sum_to_limit_in_days: int = SUM_TO_LIMIT_IN_DAYS,
                                   first_period: int = FIRST_PERIOD_IN_DAYS,
                                   second_period: int = SECOND_PERIOD_IN_DAYS,
                                   save_file_to: Path | None = None) -> pl.DataFrame:
        """
        Runs the full feature-engineering pipeline and returns the per-interval frame.

        Each feature generator returns staged expression lists. They are concatenated
        stage-by-stage and applied as successive .with_columns() passes, because a
        later stage references columns an earlier stage created and Polars evaluates
        all expressions in a single .with_columns() against the pre-pass frame.
        """

        imputer_object = Imputer(df)
        churn_adjusted_date_col_name = churn_adjusted_date_col_name or CHURN_ADJUSTED_DATE_COL

        try:
            # Accepting the mild leakge for imputing on the whole dataset (only 412 missing vehicle start year values)
            df, _, _, _ = imputer_object.KNN_impute_vehicle_start_year(df)


            group_and_sort_by_columns = [USER_ID_COL, INTERVAL_START_COL]
            df, vehicle_make_column_names = self._prepare_df(df, churn_adjusted_date_col_name)
            df_with_intervals = self._generate_intervals(df,
                                                         ACTIVITY_DATE_COL, #NOTE: Temporary fix, should be as a variable 
                                                         interval_in_days,
                                                         vehicle_make_column_names)

         

            # Flagging the first occurrence of each distinct vehicle per user-day so the
            # mean-age / still-in-production aggregations count each vehicle once.
            df_with_intervals = df_with_intervals.with_columns(
                pl.col(VEHICLE_AGE_COLUMN_NAME)
                .is_first_distinct()

                # Not over CHURND_ADJUSTED_DATE, because usage is on ACTIVITY_DATE_COL
                # Churn date is only used to generate intervals properly for the counting
                # process in survival analysis
                .over([USER_ID_COL, ACTIVITY_DATE_COL])
                .alias(FIRST_DISTINCT_COLUMN_NAME)
            )

          
            agg: list = []
            post_agg_1: list = []
            post_agg_2: list = []
            post_agg_3: list = []
            post_agg_4: list = []
            column_names_to_keep: list = []

            agg_af, post_agg_1_af, post_agg_2_af, post_agg_3_af, column_names_to_keep_af = \
                self._generate_activity_feature_aggregation(
                    lookback_periods,
                    interval_in_days)

            agg_app, post_agg_1_app, post_agg_2_app, post_agg_3_app, post_agg_4_app, column_names_to_keep_app = \
                self._generate_app_column_features_aggregation(
                    lookback_periods,
                    interval_in_days,
                    sum_to_limit_in_days,
                    first_period,
                    second_period
                    )
            agg_b, post_agg_1_b, post_agg_2_b, post_agg_3_b, column_names_to_keep_b  = \
                self._generate_behaviour_features_aggregation(
                    lookback_periods,
                    interval_in_days,
                    first_period,
                    second_period)


            agg_vehicle, post_agg_1_vehicle, post_agg_2_vehicle,post_agg_3_vehicle, column_names_to_keep_vehicle = \
                self._generate_vehicle_characteristics_features_aggregation(
                    vehicle_make_column_names,
                    interval_in_days)
            
            agg_churn, post_agg_1_churn, column_names_to_keep_churn = \
                self._generate_churn_triggered_features_agg()

            post_agg_1_start_stop, post_agg_2_start_stop = self._transform_intervals_to_start_stop()

         
            agg += agg_af + agg_app + agg_b + agg_vehicle + agg_churn

    
            column_names_to_keep += [USER_ID_COL, INTERVAL_START_COL]
            column_names_to_keep += (
                column_names_to_keep_af +
                column_names_to_keep_app +
                column_names_to_keep_b +
                column_names_to_keep_vehicle +
                column_names_to_keep_churn
            )

            
            post_agg_1 += post_agg_1_af + post_agg_1_app + post_agg_1_b + post_agg_1_vehicle + post_agg_1_churn + post_agg_1_start_stop
            post_agg_2 += post_agg_2_af + post_agg_2_app + post_agg_2_b + post_agg_2_vehicle + post_agg_2_start_stop
            post_agg_3 += post_agg_3_af + post_agg_3_app + post_agg_3_b + post_agg_3_vehicle
            post_agg_4 += post_agg_4_app

            # Sorting after agg: group_by does not preserve order, but post aggregation require chronological rows per user
            df = (df_with_intervals
                .group_by(group_and_sort_by_columns)
                .agg(agg)
                .sort(group_and_sort_by_columns)
                .with_columns(post_agg_1)
                .with_columns(post_agg_2)
                .with_columns(post_agg_3)
                .with_columns(post_agg_4)
            )

        
            # Drop each user's final interval: it is incomplete (no full interval of activity
            # observed) and its real churn label has already been shifted back onto the prior row

            df = (df
                .with_columns(
                    (pl.col(INTERVAL_START_COL) == pl.col(INTERVAL_START_COL).max().over(USER_ID_COL))
                    .alias("_is_last_interval")
                )
                # Drop the incomplete final interval
                .filter(~pl.col("_is_last_interval"))
                .drop(["_is_last_interval"])
                .select(column_names_to_keep)
            )

            # Just for better readability when examining the data
            initial_column_ordering = [USER_ID_COL, INTERVAL_START_COL, INTERVAL_END_COL]
            df = df.select(initial_column_ordering + [c for c in df.columns if c not in initial_column_ordering])
           
            if save_file_to:
                df.write_csv(save_file_to)     

            return df
            
        except Exception as e:
            print(f"Unexpected error has occurred: {e}")
            return pl.DataFrame()



# from src.constants import paths_to_files_and_folders as const

# path_to_personal_filtered = const.PATH_TO_INTERIM_DATA / "personal_users_filtered.csv"
# data_ = pl.read_csv(path_to_personal_filtered)

# dp = DataProcessor(data_)


# # print(data_["churn_triggered"].sum())
# data_prepared, _ = dp._prepare_df(data_)
# dp._generate_intervals(data_prepared)