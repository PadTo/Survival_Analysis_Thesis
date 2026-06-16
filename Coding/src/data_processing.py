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
VEHICLE_AGE_COLUMN_NAME = "vehicle_age"
FIRST_DISTINCT_COLUMN_NAME = "first_distinct"
COL_TEMPLATE_FORMAT = "n_{a}_{start}_{end}_days"



class DataProcessor():

    def __init__(self, df: pd.DataFrame | pl.DataFrame) -> None:
        self.df = df.copy(deep=True) if type(df) == pd.DataFrame else df

    def KNN_impute_vehicle_start_year(self, df: pl.DataFrame, n_neighbours: int = N_NEIGHBOURS) -> pl.DataFrame:
        """
        Imputes missing vehicle_start_year from vehicle make and mileage using KNN.

        Imputation runs on unique vehicles only, then maps back to the full frame:
        running KNN over every activity row would repeat identical vehicles many
        times and waste compute without changing the result.

        Args:
            df:            Input Polars DataFrame containing vehicle metadata.
            n_neighbours:  Number of nearest neighbours for KNN imputation. Default 5.

        Returns:
            Polars DataFrame with missing vehicle_start_year values filled.
            Returns the original DataFrame unchanged if an error occurs.
        """
        KNN_imputer = KNNImputer(n_neighbors=n_neighbours)
        one_hot_enc = OneHotEncoder(sparse_output=False)
        label_enc = LabelEncoder()

        try:
            # Deduplicating to one row per vehicle so KNN fits on distinct vehicles only
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
            # Start year placed last so the imputed target is recoverable as the final column
            imputed_df[VEHICLE_START_YEAR_COL] = vehicle_df[VEHICLE_START_YEAR_COL].values

            X_imputed = KNN_imputer.fit_transform(imputed_df)
            vehicle_df[VEHICLE_START_YEAR_COL] = X_imputed[:, -1]

            imputed_col = f"{VEHICLE_START_YEAR_COL}_imputed"
            lookup = pl.DataFrame({
                VEHICLE_ID_COL: vehicle_df[VEHICLE_ID_COL],
                imputed_col: vehicle_df[VEHICLE_START_YEAR_COL]
            })

            # Mapping imputed values back to every activity row via vehicle_id
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
        Splits the dataset into train, test (and optionally validation) sets.

        Splitting is done at the user level so that all rows for a given user land
        in the same set, preventing leakage of a user across train and test.
        Stratification is on each user's first activity year, with rare years
        binned (<= YEAR_BIN_LOWER grouped down, YEAR_BIN_UPPER grouped to the
        replacement) so that stratification does not fail on sparse year classes.

        Args:
            df:            Input Polars DataFrame with user activity logs.
            random_state:  Random seed for reproducibility. Default 42.
            test_size:     Proportion of users allocated to the test set. Default 0.1.
            val_size:      Proportion of users allocated to validation. If None, no
                           validation set is produced. Default None.

        Returns:
            (train_data, val_data, test_data) if val_size is set, else (train_data, test_data).
        """
        first_year_col: str = "first_year"

        # Binning rare first-activity years so stratification has enough members per class
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

            # Rescaling val_size relative to the remaining train pool so the final split matches the requested fraction of the whole
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
    
        # One-hot encoding make so each group becomes a 0/1 column that can be summed per interval
        df = df.to_dummies(VEHICLE_MAKE_COL)

        vehicle_make_column_names = [c for c in df.columns if (c.lower().strip()).startswith(VEHICLE_MAKE_COL)]


        return (df, vehicle_make_column_names)

    def _generate_intervals(self,
                            df: pl.DataFrame,
                            churn_adjusted_date_col_name: str | None = None,
                            interval: int = INTERVAL_IN_DAYS,
                            vehicle_make_column_names: list | None = None) -> pl.DataFrame:
        """
        Builds a per-user grid of fixed-width intervals and assigns each activity
        row to its interval.

        The grid is the counting-process backbone: every user gets a row per
        interval across their observed span, including intervals with no activity,
        because an empty interval (no usage) is itself the signal of interest.
        """

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

        df_with_intervals = self._fill_null_values(df_with_intervals,
                                                   vehicle_make_column_names)

        return df_with_intervals

    def _fill_null_values(self,
                          df: pl.DataFrame,
                          vehicle_make_column_names: list | None):
        """
        Fills nulls left by the empty intervals introduced in the interval grid.

        Empty intervals carry no activity, so their fills encode "no activity"
        explicitly: counts and flags go to a zero/False/"none" sentinel rather than
        being left null, so later sums and comparisons treat them as inactivity.
        """

        col_fill_list = [
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
            # fill_nan(0): an inactive window gives 0/0, which reads as zero share of that activity
            post_agg_3 += [
                (pl.col(column_name) / pl.col(total_actions_col_name)).alias(prop_column_name).fill_nan(0),
            ]

            column_names_to_keep += [prop_column_name]

        return agg, post_agg_1, post_agg_2, post_agg_3, column_names_to_keep

    def _generate_vehicle_characteristics_feature_aggregation(self,
                                                          vehicle_make_col_names: list,
                                                          interval_in_days: int = INTERVAL_IN_DAYS) -> tuple[list, ...]:
        """
        Builds make-portfolio features from the one-hot make columns.

        Counts accumulate cumulatively per user rather than resetting per interval:
        the set of makes a user has connected is a slow-moving characteristic of who
        they are, not a per-window behaviour. From the running counts it derives a
        dominant-make flag (the user's primary make so far) and each make's overall
        share. Cumulative counts themselves are intermediates and are not kept.

        Returns:
            Tuple of (agg, post_agg_1, post_agg_2, post_agg_3, column_names_to_keep).
        """

        agg: list = []
        post_agg_1 = []
        post_agg_2 = []
        post_agg_3 = []
        column_names_to_keep: list = []
        cum_count_col_name_template = "cumulative_count_{vehicle_make_col_name}"
        cum_count_col_names: list = []
        total_cum_count_col = "cumulative_count_total"

        for vehicle_make_col_name in vehicle_make_col_names:

            n_count_col_name = COL_TEMPLATE_FORMAT.format(
                                                a=vehicle_make_col_name,
                                                start=0,
                                                end=interval_in_days)

            cum_count_col_name = cum_count_col_name_template.format(
                vehicle_make_col_name=vehicle_make_col_name)

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
            .alias(total_cum_count_col)
        ]

        # Dominant make is the row-wise argmax over cumulative counts, flagged 1 for the leader
        dominant_col_name = "dominant_{name}"
        post_agg_2 += [
            (pl.col(col) == pl.max_horizontal([pl.col(c) for c in cum_count_col_names]))
            .cast(pl.Int8)
            .alias(dominant_col_name.format(name=col))
            for col in cum_count_col_names
        ]

        prop_col_name_template = "overall_prop_{vehicle_make_col_name}"
        post_agg_3 += [
            (pl.col(col) / pl.col(total_cum_count_col))
            .fill_nan(0)
            .alias(prop_col_name_template.format(vehicle_make_col_name=col.replace("cumulative_count_", "")))
            for col in cum_count_col_names
        ]

        column_names_to_keep += [dominant_col_name.format(name=col) for col in cum_count_col_names]
        column_names_to_keep += [
            prop_col_name_template.format(vehicle_make_col_name=col.replace("cumulative_count_", ""))
            for col in cum_count_col_names
        ]

        return agg, post_agg_1, post_agg_2, post_agg_3, column_names_to_keep

    def _generate_app_column_feature_aggregation(self,
                                             lookback_periods: tuple[int, ...] = LOOKBACK_PERIODS,
                                             interval_in_days: int = INTERVAL_IN_DAYS,
                                             sum_to_limit_in_days: int = SUM_TO_LIMIT_IN_DAYS,
                                             first_period: int = 28,
                                             second_period: int = 56) -> tuple[list, ...]:
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

        total_app_name = "total_app"
        total_app_col_name_0_56  = COL_TEMPLATE_FORMAT.format(a=total_app_name, start=0,            end=sum_to_limit_in_days)
        total_app_0_28_col_name  = COL_TEMPLATE_FORMAT.format(a=total_app_name, start=0,            end=first_period)
        total_app_28_56_col_name = COL_TEMPLATE_FORMAT.format(a=total_app_name, start=first_period, end=second_period)

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
            prop_column_name_0_56 = f"prop_{app}_{0}_{sum_to_limit_in_days}"

            post_agg_2 += [
                pl.sum_horizontal([pl.col(c) for c in columns]).alias(column_name_0_56)
            ]

            # fill_nan(0): an inactive window gives 0/0, read as zero share of that app
            post_agg_3 += [
                (pl.col(column_name_0_56) / pl.col(total_app_col_name_0_56))
                .alias(prop_column_name_0_56).fill_nan(0)
            ]

        prop_main_0_28_col  = f"prop_{main_token}_{0}_{first_period}"
        prop_main_28_56_col = f"prop_{main_token}_{first_period}_{second_period}"
        prop_main_0_56_col  = f"prop_{main_token}_{0}_{second_period}"
        main_drift_col      = f"prop_{main_token}_drift_{0}_{first_period}_vs_{first_period}_{second_period}"

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
                                    first_period: int = 28,
                                    second_period: int = 56) -> tuple[list, ...]:
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
            .alias(last_activity_date_col),

            # Counting unique days, not rows, so multiple actions on one day count as a single session
            pl.col(ACTIVITY_DATE_COL).filter(pl.col(ACTIVITY_DATE_COL)
            .is_not_null())
            .n_unique()
            .alias(session_base_col),

            pl.col(ACTIVITY_TYPE_COL).filter(pl.col(ACTIVITY_TYPE_COL) != "none")
            .count()
            .alias(action_base_col)

        ]

        post_agg_1 = [
            # Carrying the last activity date forward so empty intervals keep the most recent real date for recency
            pl.col(last_activity_date_col).forward_fill().over(USER_ID_COL),
            # Interval end is the recency reference point: how stale the user is by the close of the interval
            (pl.col(INTERVAL_START_COL) + timedelta(days=interval_in_days)).alias(interval_end_col)]

        session_lags, session_registry = self._build_lagged_columns(
            {sessions_name: session_base_col}, lookback_periods, interval_in_days, fill_value=0)
        post_agg_1 += session_lags

        action_lags, action_registry = self._build_lagged_columns(
            {actions_name: action_base_col}, lookback_periods, interval_in_days, fill_value=0)
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

        recent_window_session_col = COL_TEMPLATE_FORMAT.format(a=sessions_name,
                                                               start=0,
                                                               end=first_period)

        prior_window_session_col = COL_TEMPLATE_FORMAT.format(a=sessions_name,
                                                              start=first_period,
                                                              end=second_period)

        post_agg_2 += [pl.sum_horizontal([col for col in columns_to_sum_recent_window_sessions])
                    .alias(recent_window_session_col),
                    pl.sum_horizontal([col for col in columns_to_sum_prior_window_sessions])
                    .alias(prior_window_session_col)]

        # Casting to signed Int32: session counts are u32, so a falling drift would underflow to a large positive number
        session_intensity_drift_col = "sessions_intensity_drift_0_28_vs_28_56_days"
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

        recent_window_action_col = COL_TEMPLATE_FORMAT.format(a=actions_name,
                                                             start=0,
                                                             end=first_period)

        prior_window_action_col = COL_TEMPLATE_FORMAT.format(a=actions_name,
                                                            start=first_period,
                                                            end=second_period)

        post_agg_2 += [pl.sum_horizontal([col for col in columns_to_sum_recent_window_actions])
                    .alias(recent_window_action_col),
                    pl.sum_horizontal([col for col in columns_to_sum_prior_window_actions])
                    .alias(prior_window_action_col)]

        # Casting to signed Int32 for the same underflow reason as the session drift
        action_intensity_drift_col = "actions_intensity_drift_0_28_vs_28_56_days"
        post_agg_3 += [(pl.col(recent_window_action_col).cast(pl.Int32) - pl.col(prior_window_action_col).cast(pl.Int32)).alias(action_intensity_drift_col)]


        post_agg_2 += [(pl.col(interval_end_col) - pl.col(last_activity_date_col)).alias(days_since_last_activity_col)]


        column_names_to_keep += [
            recent_window_session_col,
            session_intensity_drift_col,
            recent_window_action_col,
            action_intensity_drift_col,
            days_since_last_activity_col,
            interval_end_col]


        return agg, post_agg_1, post_agg_2, post_agg_3, column_names_to_keep

    def apply_feature_engineering(self,
                                   df: pl.DataFrame,
                                   churn_adjusted_date_col_name: str | None = None,
                                   interval_in_days: int = INTERVAL_IN_DAYS,
                                   lookback_periods: tuple[int, ...] = LOOKBACK_PERIODS,
                                   sum_to_limit_in_days: int = SUM_TO_LIMIT_IN_DAYS,
                                   first_period: int = 28,
                                   second_period: int = 56) -> pl.DataFrame | None:
        """
        Runs the full feature-engineering pipeline and returns the per-interval frame.

        Each feature generator returns staged expression lists. They are concatenated
        stage-by-stage and applied as successive .with_columns() passes, because a
        later stage references columns an earlier stage created and Polars evaluates
        all expressions in a single .with_columns() against the pre-pass frame.
        """

        churn_adjusted_date_col_name = churn_adjusted_date_col_name or CHURN_ADJUSTED_DATE_COL

        try:
            group_and_sort_by_columns = [USER_ID_COL, INTERVAL_START_COL]
            df, vehicle_make_column_names = self._prepare_df(df, churn_adjusted_date_col_name)
            df_with_intervals = self._generate_intervals(df,
                                                         churn_adjusted_date_col_name,
                                                         interval_in_days,
                                                         vehicle_make_column_names)
            
            # Calculating mean age in an 
            df_with_intervals = df_with_intervals.with_columns(
                pl.col(VEHICLE_AGE_COLUMN_NAME)
                .is_first_distinct()

                # Not over CHURND_ADJUSTED_DATE, because usage is on ACTIVITY_DATE_COL
                # Churn date is only used to generate intervals properly for the counting 
                # process in survival analysis
                .over([USER_ID_COL,ACTIVITY_DATE_COL])
                .alias(FIRST_DISTINCT_COLUMN_NAME)
            )

            print(df_with_intervals)
            print(df_with_intervals.columns)

            agg: list = []
            post_agg_1: list = []
            post_agg_2: list = []
            post_agg_3: list = []
            post_agg_4: list = []
            column_names_to_keep: list = []
            column_names_to_keep += [USER_ID_COL]

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
                self._generate_behaviour_features_aggregation(
                    lookback_periods,
                    interval_in_days)


            agg_vehicle, post_agg_1_vehicle, post_agg_2_vehicle,post_agg_3_vehicle, column_names_to_keep_vehicle = \
                self._generate_vehicle_characteristics_feature_aggregation(
                    vehicle_make_column_names)


            agg +=  agg_vehicle
            column_names_to_keep += column_names_to_keep_vehicle
            post_agg_1 += post_agg_1_vehicle
            post_agg_2 += post_agg_2_vehicle
            post_agg_3 += post_agg_3_vehicle
            post_agg_4 += post_agg_4_app

            # Sorting after agg: group_by does not preserve order, but post aggregation require chronological rows per user
            d = df_with_intervals\
                .group_by(group_and_sort_by_columns)\
                .agg(agg)\
                .sort(group_and_sort_by_columns)\
                .with_columns(post_agg_1)\
                .with_columns(post_agg_2)\
                # .with_columns(post_agg_3)\
                # .select(column_names_to_keep)
                # .with_columns(post_agg_4)\
                # .select(column_names_to_keep)

            return d
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

# print(d.select([USER_ID_COL, INTERVAL_START_COL, "n_vehicle_make_bmw_group_0_14_days"]).sort([USER_ID_COL, INTERVAL_START_COL]))
# print(d.select([USER_ID_COL, INTERVAL_START_COL, "cummulative_count_vehicle_make_bmw_group"]).sort([USER_ID_COL, INTERVAL_START_COL]))