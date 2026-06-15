import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import polars as pl
from sklearn.model_selection import train_test_split
from datetime import timedelta
from .constants.variables_and_grouping import activity_type_groups, app_type_groups, brand_group_map
from .constants.columns import (
    USER_ID_COL, ACTIVITY_DATE_COL, CHURN_ADJUSTED_DATE_COL,
    CHURN_TRIGGERED_COL, VEHICLE_ID_COL, VEHICLE_MAKE_COL,
    VEHICLE_MODEL_COL, VEHICLE_START_YEAR_COL, VEHICLE_END_YEAR_COL,
    VEHICLE_MILEAGE_COL, APP_COL, ACTIVITY_TYPE_COL,
    STILL_IN_PRODUCTION_COL, INTERVAL_START_COL,
    YEAR_BIN_LOWER, YEAR_BIN_UPPER, YEAR_BIN_UPPER_REPLACEMENT
)




LOOKBACK_PERIODS = (1,2,3)
INTERVAL_IN_DAYS: int = 14
SUM_TO_LIMIT_IN_DAYS = 56
N_NEIGHBOURS = 5
COL_TEMPLATE_FORMAT = "n_{a}_{start}_{end}_days"


class DataProcessor():

    def __init__(self, df: pd.DataFrame | pl.DataFrame) -> None:
        self.df = df.copy(deep=True) if type(df) == pd.DataFrame else df

    def KNN_impute_vehicle_start_year(self, df: pl.DataFrame, n_neighbours: int = N_NEIGHBOURS) -> pl.DataFrame:
        """
        Imputes missing vehicle_start_year values using KNN based on
        vehicle make and mileage.

        To avoid running KNN on the full dataset, imputation is performed
        on unique vehicles only, then mapped back to the original DataFrame.

        Steps:
            1. Extract unique vehicles (vehicle_id, make, mileage, start_year).
            2. One-hot encode vehicle_make (nominal — no natural order).
            3. Label encode vehicle_mileage (ordinal — ascending mileage bands).
            4. Fit KNN imputer and impute missing vehicle_start_year values.
            5. Map imputed start years back to the original DataFrame via vehicle_id.

        Args:
            df:            Input Polars DataFrame containing vehicle metadata.
            n_neighbours:  Number of nearest neighbours for KNN imputation.
                           Default is 5.

        Returns:
            Polars DataFrame with missing vehicle_start_year values filled.
            Returns original DataFrame unchanged if an error occurs.
        """
        KNN_imputer = KNNImputer(n_neighbors=n_neighbours)
        one_hot_enc = OneHotEncoder(sparse_output=False)
        label_enc = LabelEncoder()

        try:
            vehicle_df = (df
                .select([VEHICLE_ID_COL, VEHICLE_MAKE_COL,
                         VEHICLE_MILEAGE_COL, VEHICLE_START_YEAR_COL])
                .unique(subset=[VEHICLE_ID_COL])
                .to_pandas())

            vehicle_make_one_hot = one_hot_enc.fit_transform(vehicle_df[[VEHICLE_MAKE_COL]])
            vehicle_mileage_label = label_enc.fit_transform(vehicle_df[VEHICLE_MILEAGE_COL])

            imputed_df = pd.DataFrame(
                vehicle_make_one_hot,
                columns=one_hot_enc.get_feature_names_out(),
                index=vehicle_df.index
            )
            imputed_df["vehicle_mileage_cat"] = vehicle_mileage_label
            imputed_df[VEHICLE_START_YEAR_COL] = vehicle_df[VEHICLE_START_YEAR_COL].values

            X_imputed = KNN_imputer.fit_transform(imputed_df)
            vehicle_df[VEHICLE_START_YEAR_COL] = X_imputed[:, -1]

            imputed_col = f"{VEHICLE_START_YEAR_COL}_imputed"
            lookup = pl.DataFrame({
                VEHICLE_ID_COL: vehicle_df[VEHICLE_ID_COL],
                imputed_col: vehicle_df[VEHICLE_START_YEAR_COL]
            })

            df = (df
                .join(lookup, on=VEHICLE_ID_COL, how="left")
                .with_columns(pl.col(imputed_col).alias(VEHICLE_START_YEAR_COL))
                .drop(imputed_col))

            return df

        except Exception as e:
            print(f"Unexpected error has occurred: {e}")
            return df

    def split_train_val_test(self,
                             df: pl.DataFrame,
                             random_state: int = 42,
                             test_size: float = 0.1,
                             val_size: float | None = None) -> tuple[pl.DataFrame, ...]:
        """
        Splits the dataset into train, test (and optionally validation) sets,
        stratified by each user's first activity year. Splitting is done at
        the user level — all rows for a given user end up in the same set.

        Rare years are binned before stratification:
            - 2020 and earlier → grouped as 2020
            - 2026             → grouped as 2025

        Args:
            df:            Input Polars DataFrame with user activity logs.
            random_state:  Random seed for reproducibility. Default 42.
            test_size:     Proportion of users to allocate to test set. Default 0.1.
            val_size:      Proportion of users to allocate to validation set.
                           If None, no validation set is created. Default None.

        Returns:
            Tuple of (train_data, val_data, test_data) if val_size is set,
            otherwise (train_data, test_data).
        """
        first_year_col: str = "first_year"

        user_years = (df
            .with_columns(pl.col(ACTIVITY_DATE_COL).cast(pl.Date))
            .group_by(USER_ID_COL)
            .agg(pl.col(ACTIVITY_DATE_COL).min().dt.year().alias(first_year_col))
            .with_columns(
                pl.col(first_year_col).map_elements(
                    lambda x: YEAR_BIN_LOWER if x <= YEAR_BIN_LOWER
                              else (YEAR_BIN_UPPER_REPLACEMENT if x == YEAR_BIN_UPPER else x)
                )
            )
        )

        user_years_pd = user_years.to_pandas()

        user_train_labels, user_test_labels, _, _ = train_test_split(
            user_years_pd[USER_ID_COL],
            user_years_pd[first_year_col],
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=user_years_pd[first_year_col]
        )

        train_data = df.filter(pl.col(USER_ID_COL).is_in(user_train_labels.tolist()))
        test_data = df.filter(pl.col(USER_ID_COL).is_in(user_test_labels.tolist()))

        if val_size is not None:
            train_years = user_years_pd[user_years_pd[USER_ID_COL].isin(user_train_labels)]

            user_train_labels, user_val_labels, _, _ = train_test_split(
                train_years[USER_ID_COL],
                train_years[first_year_col],
                test_size=val_size / (1 - test_size),
                random_state=random_state,
                shuffle=True,
                stratify=train_years[first_year_col]
            )

            train_data = df.filter(pl.col(USER_ID_COL).is_in(user_train_labels.tolist()))
            val_data = df.filter(pl.col(USER_ID_COL).is_in(user_val_labels.tolist()))

            print(len(train_data), len(val_data), len(test_data))
            return train_data, val_data, test_data

        print(len(train_data), len(test_data))
        return train_data, test_data

    def _prepare_df(self,
                    df: pl.DataFrame,
                    churn_adjusted_date_col_name: str | None = None,
                    activity_date_col_name: str | None = None) -> pl.DataFrame:

        churn_adjusted_date_col_name = churn_adjusted_date_col_name or CHURN_ADJUSTED_DATE_COL
        activity_date_col_name = activity_date_col_name or ACTIVITY_DATE_COL

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
        
        df = df.to_dummies(VEHICLE_MAKE_COL)
        return df

    def _generate_intervals(self,
                            df: pl.DataFrame,
                            churn_adjusted_date_col_name: str | None = None,
                            interval: int = INTERVAL_IN_DAYS) -> pl.DataFrame:

        churn_adjusted_date_col_name = churn_adjusted_date_col_name or CHURN_ADJUSTED_DATE_COL

        user_min_max_dates = df.group_by(USER_ID_COL).agg(
            pl.col(churn_adjusted_date_col_name).min().alias("start_date"),
            pl.col(churn_adjusted_date_col_name).max().alias("end_date")
        )

        intervals = user_min_max_dates.with_columns(
            pl.date_ranges(
                start=pl.col("start_date"),
                end=pl.col("end_date"),
                interval=timedelta(days=interval)
            ).alias(INTERVAL_START_COL)
        ).explode(INTERVAL_START_COL).select([USER_ID_COL, INTERVAL_START_COL])

        df_with_intervals = df.join_asof(
            intervals,
            left_on=churn_adjusted_date_col_name,
            right_on=INTERVAL_START_COL,
            by=USER_ID_COL,
            strategy="backward"
        )

        df_with_intervals = intervals.join(
            df_with_intervals,
            on=[USER_ID_COL, INTERVAL_START_COL],
            how="left"
        )

        df_with_intervals = df_with_intervals.with_columns([
            pl.col(CHURN_TRIGGERED_COL).fill_null(False),
            pl.col(VEHICLE_ID_COL).fill_null("unknown"),
            pl.col(VEHICLE_MODEL_COL).fill_null("unknown"),
            pl.col(VEHICLE_START_YEAR_COL).fill_null(0),
            pl.col(VEHICLE_END_YEAR_COL).fill_null(0),
            pl.col(VEHICLE_MILEAGE_COL).fill_null("unknown"),
            pl.col(ACTIVITY_DATE_COL).fill_null(pl.lit(None).cast(pl.Date)),
            pl.col(APP_COL).fill_null("unknown"),
            pl.col(ACTIVITY_TYPE_COL).fill_null("none"),
            pl.col(STILL_IN_PRODUCTION_COL).fill_null(False),
            pl.col(churn_adjusted_date_col_name).fill_null(pl.lit(None).cast(pl.Date)),
        ])

        return df_with_intervals

    def _build_lagged_columns(self,
                              base_columns: dict[str, str],
                              lookback_periods: tuple[int, ...] = LOOKBACK_PERIODS,
                              interval_in_days: int = INTERVAL_IN_DAYS,
                              fill_value: int | None = None) -> tuple[list, dict[tuple[int, int], list[str]]]:
        """
        Builds shift expressions and a window registry for a set of base columns.

        base_columns maps a token (used in the COL_TEMPLATE_FORMAT name) to the
        source column name at the (0, interval) window. For each lookback period it
        produces a shifted column aliased to the (start, end) window, and records
        every produced column in the registry keyed by (start, end).

        Args:
            base_columns:     token -> source column name at the (0, interval) window.
            lookback_periods: Shift periods, expressed in intervals.
            interval_in_days: Width of each interval in days.
            fill_value:       Fill value for the shift. If None, no fill is applied.

        Returns:
            Tuple of (lag_expressions, col_registry).
        """
        lag_expressions: list = []

        # Initial column registration (the current, un-shifted window)
        col_registry: dict[tuple[int, int], list[str]] = {
            (0, interval_in_days): [
                COL_TEMPLATE_FORMAT.format(a=token, start=0, end=interval_in_days)
                for token in base_columns.keys()
            ]
        }

        # Shifting the columns by lookback periods
        for period in lookback_periods:

            start_interval = interval_in_days * period
            end_interval = interval_in_days * (period + 1)

            for token, source_col in base_columns.items():
                # Taking the initially computed column which is with start = 0 and end = interval_in_days
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
        Returns registry columns whose window ends within sum_to_limit_in_days,
        optionally restricted to those whose name contains token_filter.
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
            Builds three-stage aggregation expressions for activity type features.

            Stage 1 (agg): For each interval, counts total actions and per-activity-type
            action counts, producing columns named n_{activity}_{0}_{interval}_days.

            Stage 2 (post_agg_1): Shifts the current-interval counts backwards by each
            lookback period, producing lagged columns n_{activity}_{start}_{end}_days
            for each period in lookback_periods. Operates via .with_columns() after agg.

            Stage 3 (post_agg_2): Sums the lagged columns within the sum_to_limit_in_days
            window to produce total per-activity counts and an overall total action count.
            Requires post_agg_1 columns to exist first.

            Stage 4 (post_agg_3): Computes per-activity proportions by dividing summed
            activity counts by total action count. Requires post_agg_2 columns to exist.

            Args:
                lookback_periods:     Tuple of shift periods (in intervals) to look back.
                                    E.g. (1, 2, 3) with interval_in_days=14 looks back
                                    14, 28, and 42 days.
                interval_in_days:     Width of each interval in days. Default INTERVAL_IN_DAYS.
                sum_to_limit_in_days: Upper bound (in days) for summing lagged columns into
                                    the final activity count features. Default 56.

            Returns:
                Tuple of (agg, post_agg_1, post_agg_2, post_agg_3, column_names_to_keep)
                where each list is applied sequentially.
        """

        post_agg_1: list = []
        post_agg_2: list = []
        post_agg_3: list = []
        column_names_to_keep: list = []
        activity_types: list[str] = list(sorted(set(activity_type_groups.values())))

        # Building the first aggregation
        agg: list = [
            (pl.col(ACTIVITY_TYPE_COL) == a).sum().alias(
                COL_TEMPLATE_FORMAT.format(a=a, start=0, end=interval_in_days))
            for a in activity_types
        ]

        # Shifting the columns by lookback periods (lagged columns + window registry)
        base_columns = {
            a: COL_TEMPLATE_FORMAT.format(a=a, start=0, end=interval_in_days)
            for a in activity_types
        }
        post_agg_1, col_registry = self._build_lagged_columns(
            base_columns, lookback_periods, interval_in_days, fill_value=0)

        columns_to_sum_over_for_total_activity_count = self._columns_within_window(
            col_registry, end_at_days=sum_to_limit_in_days)

        # Building intermediate and final post aggregation processing script for total counts in the entire lookback window
        total_actions_col_name = f"total_actions_{0}_{sum_to_limit_in_days}"
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

            prop_column_name = f"prop_{activity_type}_{0}_{sum_to_limit_in_days}"
            post_agg_3 += [
                (pl.col(column_name) / pl.col(total_actions_col_name)).alias(prop_column_name).fill_nan(0),
            ]

            column_names_to_keep += [prop_column_name]

        return agg, post_agg_1, post_agg_2, post_agg_3, column_names_to_keep

    def _generate_vehicle_characteristics_feature_aggregation(self) -> tuple[list, ...]:
        make_groups: list[str] = list(set(brand_group_map.values()))

        agg: list = [
            pl.col(VEHICLE_ID_COL).filter(pl.col(VEHICLE_ID_COL) != "unknown").n_unique().alias("n_unique_vehicles"),
            pl.col(VEHICLE_ID_COL).filter(pl.col(VEHICLE_ID_COL) != "unknown").count().alias("n_vehicles"),
        ] + [
            (pl.col(VEHICLE_MAKE_COL) == make).sum().alias(f"n_{make}")
            for make in make_groups
        ]

        post_agg = []

        return agg, post_agg

    def _generate_app_column_feature_aggregation(self,
                                             lookback_periods: tuple[int, ...] = LOOKBACK_PERIODS,
                                             interval_in_days: int = INTERVAL_IN_DAYS,
                                             sum_to_limit_in_days: int = SUM_TO_LIMIT_IN_DAYS,
                                             first_period: int = 28,
                                             second_period: int = 56) -> tuple[list, ...]:
        post_agg_1: list = []
        post_agg_2: list = []
        post_agg_3: list = []
        column_names_to_keep: list = []
        apps: list[str] = list(app_type_groups.keys())

        # Building the first aggregation
        agg: list = [
            (pl.col(APP_COL) == app).sum().alias(
                COL_TEMPLATE_FORMAT.format(a=app, start=0, end=interval_in_days))
            for app in apps
        ]

        # Shifting the columns by lookback periods (lagged columns + window registry)
        base_columns = {
            app: COL_TEMPLATE_FORMAT.format(a=app, start=0, end=interval_in_days)
            for app in apps
        }
        post_agg_1, col_registry = self._build_lagged_columns(
            base_columns, lookback_periods, interval_in_days, fill_value=0)

        # Total app counts across the full lookback window
        columns_to_sum_over_for_total_app_count = self._columns_within_window(
            col_registry, end_at_days=sum_to_limit_in_days)

        # Total app counts split into first and second period windows (denominators for proportions)
        columns_to_sum_first_window_app = self._columns_within_window(
            col_registry, start_from_days=0, end_at_days=first_period)

        columns_to_sum_second_window_app = self._columns_within_window(
            col_registry, start_from_days=first_period, end_at_days=second_period)

        total_app_name = "total_app"
        total_app_col_name_0_56  = COL_TEMPLATE_FORMAT.format(a=total_app_name, start=0,            end=sum_to_limit_in_days)
        total_app_0_28_col_name  = COL_TEMPLATE_FORMAT.format(a=total_app_name, start=0,            end=first_period)
        total_app_28_56_col_name = COL_TEMPLATE_FORMAT.format(a=total_app_name, start=first_period, end=second_period)

        post_agg_2 += [
            pl.sum_horizontal([pl.col(c) for c in columns_to_sum_over_for_total_app_count]).alias(total_app_col_name_0_56),
            pl.sum_horizontal([pl.col(c) for c in columns_to_sum_first_window_app]).alias(total_app_0_28_col_name),
            pl.sum_horizontal([pl.col(c) for c in columns_to_sum_second_window_app]).alias(total_app_28_56_col_name),
        ]

        # main counts in both windows (only main needed - vag is implicit as 1 - prop_main)
        main_token = list(base_columns.keys())[0]

        cols_main_0_28  = self._columns_within_window(col_registry, start_from_days=0,            end_at_days=first_period,  token_filter=main_token)
        cols_main_28_56 = self._columns_within_window(col_registry, start_from_days=first_period, end_at_days=second_period, token_filter=main_token)

        main_0_28_col  = COL_TEMPLATE_FORMAT.format(a=main_token, start=0,            end=first_period)
        main_28_56_col = COL_TEMPLATE_FORMAT.format(a=main_token, start=first_period, end=second_period)

        post_agg_2 += [
            pl.sum_horizontal([pl.col(c) for c in cols_main_0_28]).alias(main_0_28_col),
            pl.sum_horizontal([pl.col(c) for c in cols_main_28_56]).alias(main_28_56_col),
        ]

        # Per-app proportion over the full 0-56 window
        for app in apps:
            columns = self._columns_within_window(
                col_registry, end_at_days=sum_to_limit_in_days, token_filter=app)

            column_name_0_56 = COL_TEMPLATE_FORMAT.format(a=app, start=0, end=sum_to_limit_in_days)
            prop_column_name_0_56 = f"prop_{app}_{0}_{sum_to_limit_in_days}"

            post_agg_2 += [
                pl.sum_horizontal([pl.col(c) for c in columns]).alias(column_name_0_56)
            ]

            post_agg_3 += [
                (pl.col(column_name_0_56) / pl.col(total_app_col_name_0_56))
                .alias(prop_column_name_0_56).fill_nan(0)
            ]

        # main proportion in each window + drift (prop_main_0_28 - prop_main_28_56)
        prop_main_0_28_col  = f"prop_{main_token}_{0}_{first_period}"
        prop_main_28_56_col = f"prop_{main_token}_{first_period}_{second_period}"
        prop_main_0_56_col  = f"prop_{main_token}_{0}_{second_period}"
        main_drift_col      = f"prop_{main_token}_drift_{0}_{first_period}_vs_{first_period}_{second_period}"

        post_agg_3 += [
            (pl.col(main_0_28_col)  / pl.col(total_app_0_28_col_name)).alias(prop_main_0_28_col).fill_nan(0),
            (pl.col(main_28_56_col) / pl.col(total_app_28_56_col_name)).alias(prop_main_28_56_col).fill_nan(0),
        ]

        # Drift requires prop columns to exist
        post_agg_4: list = [
            (pl.col(prop_main_0_28_col) - pl.col(prop_main_28_56_col)).alias(main_drift_col)
        ]

        column_names_to_keep += [prop_main_0_56_col, main_drift_col]

        return agg, post_agg_1, post_agg_2, post_agg_3, post_agg_4, column_names_to_keep

    def _generate_behaviour_features(self,
                                    lookback_periods: tuple[int, ...] = LOOKBACK_PERIODS,
                                    interval_in_days: int = INTERVAL_IN_DAYS,
                                    first_period: int = 28,
                                    second_period: int = 56) -> tuple[list, ...]:
        agg: list = []
        post_agg_1: list = []
        post_agg_2: list = []
        post_agg_3: list = []
        column_names_to_keep: list = []


        last_activity_date_col = "last_activity_date"
        days_since_last_activity_col = "recency"
        interval_end_col = "interval_end"
        sessions_name = "sessions"
        actions_name = "actions"

        session_base_col = COL_TEMPLATE_FORMAT.format(a=sessions_name, start=0, end=interval_in_days)
        action_base_col = COL_TEMPLATE_FORMAT.format(a=actions_name, start=0, end=interval_in_days)

        agg = [
            pl.col(ACTIVITY_DATE_COL).filter(pl.col(ACTIVITY_DATE_COL)
            .is_not_null())
            .max()
            .alias(last_activity_date_col), # Computing last activity in an interval

            pl.col(ACTIVITY_DATE_COL).filter(pl.col(ACTIVITY_DATE_COL)
            .is_not_null())
            .n_unique()
            .alias(session_base_col), # Computing number of unique sessions per interval, counted only for unique days

            pl.col(ACTIVITY_TYPE_COL).filter(pl.col(ACTIVITY_TYPE_COL) != "none")
            .count()
            .alias(action_base_col) # Computing number of actions per interval

        ]

        post_agg_1 = [
            pl.col(last_activity_date_col).forward_fill().over(USER_ID_COL),
            # This is the ending interval calculated by taking start interval + interval gap for recency
            (pl.col(INTERVAL_START_COL) + timedelta(days=interval_in_days)).alias(interval_end_col)]

        # Laging session counts
        session_lags, session_registry = self._build_lagged_columns(
            {sessions_name: session_base_col}, lookback_periods, interval_in_days, fill_value=0)
        post_agg_1 += session_lags

        # Lagging action counts
        action_lags, action_registry = self._build_lagged_columns(
            {actions_name: action_base_col}, lookback_periods, interval_in_days, fill_value=0)
        post_agg_1 += action_lags


        # --- Sessions: window sums + drift ---
        columns_to_sum_first_window_sessions = self._columns_within_window(
            session_registry,
            start_from_days=0,
            end_at_days=first_period)

        columns_to_sum_second_window_sessions = self._columns_within_window(
            session_registry,
            start_from_days=first_period,
            end_at_days=second_period)

        shorter_lookback_name_session = COL_TEMPLATE_FORMAT.format(a=sessions_name,
                                                                start=0,
                                                                end=first_period)

        longer_lookback_name_session = COL_TEMPLATE_FORMAT.format(a=sessions_name,
                                                                start=first_period,
                                                                end=second_period)

        # Session count with lookbacks
        post_agg_2 += [pl.sum_horizontal([col for col in columns_to_sum_first_window_sessions])
                    .alias(shorter_lookback_name_session),
                    pl.sum_horizontal([col for col in columns_to_sum_second_window_sessions])
                    .alias(longer_lookback_name_session)]

        session_intensity_drift_col = "sessions_intensity_drift_0_28_vs_28_56_days"
        post_agg_3 += [(pl.col(shorter_lookback_name_session).cast(pl.Int32) - pl.col(longer_lookback_name_session).cast(pl.Int32)).alias(session_intensity_drift_col)]


        # --- Actions: window sums + drift ---
        columns_to_sum_first_window_actions = self._columns_within_window(
            action_registry,
            start_from_days=0,
            end_at_days=first_period)

        columns_to_sum_second_window_actions = self._columns_within_window(
            action_registry,
            start_from_days=first_period,
            end_at_days=second_period)

        shorter_lookback_name_action = COL_TEMPLATE_FORMAT.format(a=actions_name,
                                                                start=0,
                                                                end=first_period)

        longer_lookback_name_action = COL_TEMPLATE_FORMAT.format(a=actions_name,
                                                                start=first_period,
                                                                end=second_period)

        # Action count with lookbacks
        post_agg_2 += [pl.sum_horizontal([col for col in columns_to_sum_first_window_actions])
                    .alias(shorter_lookback_name_action),
                    pl.sum_horizontal([col for col in columns_to_sum_second_window_actions])
                    .alias(longer_lookback_name_action)]

        action_intensity_drift_col = "actions_intensity_drift_0_28_vs_28_56_days"
        post_agg_3 += [(pl.col(shorter_lookback_name_action).cast(pl.Int32) - pl.col(longer_lookback_name_action).cast(pl.Int32)).alias(action_intensity_drift_col)]


        # Recency
        post_agg_2 += [(pl.col(interval_end_col) - pl.col(last_activity_date_col)).alias(days_since_last_activity_col)]


        column_names_to_keep += [
            shorter_lookback_name_session,
            session_intensity_drift_col,
            shorter_lookback_name_action,
            action_intensity_drift_col,
            days_since_last_activity_col]


        return agg, post_agg_1, post_agg_2, post_agg_3, column_names_to_keep
    
    def apply_feature_engineering(self,
                                   df: pl.DataFrame,
                                   churn_adjusted_date_col_name: str | None = None,
                                   interval_in_days: int = INTERVAL_IN_DAYS,
                                   lookback_periods: tuple[int, ...] = LOOKBACK_PERIODS,
                                   sum_to_limit_in_days: int = SUM_TO_LIMIT_IN_DAYS,
                                   first_period: int = 28,
                                   second_period: int = 56) -> pl.DataFrame | None:

        churn_adjusted_date_col_name = churn_adjusted_date_col_name or CHURN_ADJUSTED_DATE_COL

        try:
            group_and_sort_by_columns = [USER_ID_COL, INTERVAL_START_COL]
            df = self._prepare_df(df, churn_adjusted_date_col_name)
            df_with_intervals = self._generate_intervals(df, churn_adjusted_date_col_name, interval_in_days)


            print(df)

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
                self._generate_app_column_feature_aggregation(
                lookback_periods,
                interval_in_days,
                sum_to_limit_in_days
            )
            agg_b, post_agg_1_b, post_agg_2_b, post_agg_3_b, column_names_to_keep_b  = \
                self._generate_behaviour_features(
                lookback_periods,
                interval_in_days)
            # agg_app, post_agg_app = self._generate_app_column_feature_aggregation()
            # agg_v, post_agg_v = self._generate_vehicle_make_feature_aggregation()

            # agg += agg_b + agg_app + agg_v
            # post_agg += post_agg_app + post_agg_v + [pl.col("user_id"), pl.col("interval_start")]


            # agg +=  agg_app
            # column_names_to_keep += column_names_to_keep_app
            # post_agg_1 += post_agg_1_app
            # post_agg_2 += post_agg_2_app
            # post_agg_3 += post_agg_3_app
            # post_agg_4 += post_agg_4_app

            # d = df_with_intervals\
            #     .group_by(group_and_sort_by_columns)\
            #     .agg(agg)\
            #     .sort(group_and_sort_by_columns)\
            #     .with_columns(post_agg_1)\
            #     .with_columns(post_agg_2)\
            #     .with_columns(post_agg_3)\
            #     .with_columns(post_agg_4)\
            #     .select(column_names_to_keep)
            # print(d)
            # print(d.columns)
            # return d
        except Exception as e:
            print(f"Unexpected error has occurred: {e}")

    def split_data_and_process(self) -> None:
        pass



df_vh = pl.read_csv(Path(r"C:\Users\Tomas\Desktop\Thesis Stuff\Survival_Analysis_Thesis\Coding\Data\interim\personal_users_dataset.csv"))
dp = DataProcessor(df_vh)
# df = dp.KNN_impute_vehicle_start_year(df_vh)
# dp.split_train_val_test(df_vh)

# dp.feature_engineering(df_vh)

# dp._generate_activity_feature_aggregation()
d: pl.DataFrame = dp.apply_feature_engineering(df_vh)