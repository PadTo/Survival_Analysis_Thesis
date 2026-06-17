import datetime
from pathlib import Path
import pandas as pd
from .constants import paths_to_files_and_folders
from .constants.columns import (
    USER_ID_COL, ACTIVITY_DATE_COL, CHURN_ADJUSTED_DATE_COL,
    CHURN_TRIGGERED_COL, VEHICLE_ID_COL, VEHICLE_MAKE_COL,
    VEHICLE_MODEL_COL, VEHICLE_END_YEAR_COL, VEHICLE_MILEAGE_COL,
    STILL_IN_PRODUCTION_COL,
)
from .constants.cleaning import (
    CSV_EXTENSION, DEFAULT_OUTPUT_FILENAME,
    PERSONAL_USERS_FILENAME, PROFESSIONAL_USERS_FILENAME,
    DEFAULT_THRESHOLD_DAYS, DEFAULT_HHI_THRESHOLD, DEFAULT_CAR_SHARE_ABS,
    DEFAULT_CAR_SHARE_FRACTION, QUANTILE_FILTER, BURST_TIME_HR,
    MIN_ACTIVITY_SPAN_DAYS
)


# ============================================================
#  Internal working columns (created and dropped within methods)
# ============================================================
NEXT_DATE_COL: str    = "next_date"
ACTIVITY_GAP_COL: str = "activity_gap"
SHIFTED_DATE_COL: str = "activity_date_shifted"

# ============================================================
#  Sentinel values
# ============================================================
FILL_VALUE_CHURN_TRIGGERED: bool = False

class DataCleaner:
    def __init__(self,
                 df_activity: pd.DataFrame,
                 df_vehicle: pd.DataFrame) -> None:
        self.df_activity = df_activity.copy(deep=True)
        self.df_vehicle = df_vehicle.copy(deep=True)

        self.max_date = pd.to_datetime(df_activity[ACTIVITY_DATE_COL].max())

    def __step_counter(self, step_counter: int = 1) -> int:
        print(f"_Step {step_counter}_")
        return step_counter + 1

    def basic_filter_and_merge_df(self,
                                  filter_nan_cols: list[str] | None = None) -> pd.DataFrame:
        """
        Merges vehicle metadata with activity logs and applies basic cleaning.
        """

        # Users with the same vehicle_id appearing under different metadata rows are
        # ambiguous - keeping them would silently duplicate activity rows on merge
        dupe_users_mask = (
            self.df_vehicle
            .groupby([USER_ID_COL, VEHICLE_ID_COL])[VEHICLE_ID_COL]
            .count()
            .gt(1)
            .groupby(level=0)
            .any()
        )
        dupe_user_ids = dupe_users_mask[dupe_users_mask].index

        self.df_vehicle = self.df_vehicle[
            ~self.df_vehicle[USER_ID_COL].isin(dupe_user_ids)
        ]

        print(f"Rows before merging: {len(self.df_activity)}")
        df = self.df_vehicle.merge(
            self.df_activity,
            on=[VEHICLE_ID_COL, USER_ID_COL],
            how="right"
        )
        print(f"Rows after merging: {len(df)}")
        print()

        if filter_nan_cols:
            print(f"Filtering rows with missing values in columns: {filter_nan_cols}")

            row_count_before = len(df)
            df = df.dropna(subset=filter_nan_cols).copy()
            row_count_after = len(df)

            print(f"Rows before NaN filtering: {row_count_before}")
            print(f"Rows after NaN filtering: {row_count_after}")
            print(f"Rows removed: {row_count_before - row_count_after}")
            print()

        # Activity rows with no matching vehicle cannot carry make/model/age metadata
        # downstream, so the users that don't exist in the vehicle dataset are removed
        print("Removing user_ids that aren't present in vehicle dataset")
        print(f"Rows before user filtering: {len(df)}")

        unique_users = self.df_vehicle[USER_ID_COL].unique()
        unique_users_mask = df[USER_ID_COL].isin(unique_users)

        df = df[unique_users_mask]

        print(f"Rows after user filtering: {len(df)}")
        print()

        print("Removing duplicate rows")
        print(f"Rows before deduplication: {len(df)}")

        df.drop_duplicates(inplace=True)

        print(f"Rows after deduplication: {len(df)}")
        print()

        return df.copy()

    def filter_after_inactivity(self,
                                df: pd.DataFrame,
                                threshold_value: int = DEFAULT_THRESHOLD_DAYS
                                ) -> pd.DataFrame:
        """
        Labels churn and truncates each user's history at their first churn event.

        Churn is defined as either a gap between consecutive activities exceeding
        the threshold, or a last activity that is more than threshold days before
        the dataset max date. Rows after the first trigger are dropped because they
        would represent activity in a post-churn state that the model should not see.
        """

        df = df.copy(deep=True)

        df[ACTIVITY_DATE_COL] = pd.to_datetime(df[ACTIVITY_DATE_COL])
        df = df.sort_values([USER_ID_COL, ACTIVITY_DATE_COL])

        df[NEXT_DATE_COL] = df.groupby(USER_ID_COL)[ACTIVITY_DATE_COL].shift(-1)
        df[ACTIVITY_GAP_COL] = df[NEXT_DATE_COL] - df[ACTIVITY_DATE_COL]

        # Condition 1: gap between consecutive activities exceeds threshold
        gap_churn = (
            df[ACTIVITY_GAP_COL] >= pd.Timedelta(days=threshold_value)
        ) & df[ACTIVITY_GAP_COL].notna()

        # Condition 2: last activity is more than threshold days before max_date
        is_last_row = df[NEXT_DATE_COL].isna()

        end_churn = is_last_row & (
            (self.max_date - df[ACTIVITY_DATE_COL]) >= pd.Timedelta(days=threshold_value)
        )

        df[CHURN_TRIGGERED_COL] = gap_churn | end_churn

        # cummax propagates the first True forward so all rows after the trigger
        # are also marked, letting the shift-mask below drop them cleanly
        df[CHURN_TRIGGERED_COL] = df.groupby(USER_ID_COL)[CHURN_TRIGGERED_COL].cummax()

        # shift(1) offsets the cummax flag by one row so the trigger row itself
        # is kept (it is the churn event) but every row after it is dropped
        row_mask = (
            df.groupby(USER_ID_COL)[CHURN_TRIGGERED_COL]
            .shift(1, fill_value=FILL_VALUE_CHURN_TRIGGERED)
        )

        df = df[~row_mask]

        df[CHURN_ADJUSTED_DATE_COL] = df[ACTIVITY_DATE_COL]

        # Churn date is pushed forward by the threshold so the interval grid
        # covers the full at-risk window, not just the last observed activity
        df.loc[df[CHURN_TRIGGERED_COL], CHURN_ADJUSTED_DATE_COL] += pd.Timedelta(days=threshold_value)

        df.drop(columns=[ACTIVITY_GAP_COL, NEXT_DATE_COL], inplace=True)

        return df.copy()

    def filter_nan_vehicle_metadata(self,
                                    df: pd.DataFrame,
                                    columns_by_which_to_drop: list[str] | None = None
                                    ) -> pd.DataFrame:
        """
        Removes rows with missing values in vehicle metadata columns.
        """
        return df.dropna(subset=columns_by_which_to_drop).copy()

    def filter_one_day_users(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes users whose activity history spans only one day.

        Single-day users have no observable inactivity pattern and cannot
        contribute meaningful interval-level features to the survival model.
        """
        df = df.copy(deep=True)
        df[ACTIVITY_DATE_COL] = pd.to_datetime(df[ACTIVITY_DATE_COL])

        earliest_activity_per_user = df.groupby(USER_ID_COL)[ACTIVITY_DATE_COL].min()
        latest_activity_per_user   = df.groupby(USER_ID_COL)[ACTIVITY_DATE_COL].max()

        diff_between_earliest_and_last_usage = latest_activity_per_user - earliest_activity_per_user

        representative_users_mask = (
            diff_between_earliest_and_last_usage > pd.Timedelta(MIN_ACTIVITY_SPAN_DAYS, unit="days")
        )

        representative_users = earliest_activity_per_user.index[representative_users_mask]

        print(f"Number of one-day-users: {sum(~representative_users_mask)}")

        return df[df[USER_ID_COL].isin(representative_users)].copy()

    def filter_users_by_set_cutoff_date(self,
                                        df: pd.DataFrame,
                                        threshold_value: int = DEFAULT_THRESHOLD_DAYS
                                        ) -> pd.DataFrame:
        """
        Removes users whose observation window is shorter than the churn threshold.

        A user first seen less than threshold_value days before max_date cannot
        have churned by definition, so including them would inflate the censored
        fraction without contributing any churn signal.
        """
        df[ACTIVITY_DATE_COL] = pd.to_datetime(df[ACTIVITY_DATE_COL])

        users_first_app_use_date = pd.Series(
            df.groupby(USER_ID_COL)[ACTIVITY_DATE_COL].min()
        )

        filtered_users = users_first_app_use_date[
            (self.max_date - users_first_app_use_date) > pd.Timedelta(days=threshold_value)
        ]

        return df[df[USER_ID_COL].isin(filtered_users.index)]

    def _filter_rows_from_burst(self,
                                df: pd.DataFrame,
                                burst_time_hr: float = BURST_TIME_HR) -> pd.DataFrame:
        df = df.copy()
        df[ACTIVITY_DATE_COL] = pd.to_datetime(df[ACTIVITY_DATE_COL])
        df = df.sort_values([USER_ID_COL, VEHICLE_ID_COL, ACTIVITY_DATE_COL])

        shifted = df.groupby([USER_ID_COL, VEHICLE_ID_COL])[ACTIVITY_DATE_COL].shift(1)
        gap = df[ACTIVITY_DATE_COL] - shifted

        # First row per user-vehicle pair has no prior - keep it via NaT check
        is_not_burst = gap.isna() | (gap >= pd.Timedelta(hours=burst_time_hr))

        return df[is_not_burst].copy()

    def filter_users_by_type(self,
                             df: pd.DataFrame,
                             inverse_hhi_threshold: int = DEFAULT_HHI_THRESHOLD,
                             car_share_threshold_abs: int = DEFAULT_CAR_SHARE_ABS,
                             car_share_threshold_fraction: float = DEFAULT_CAR_SHARE_FRACTION,
                             return_personal_use_users: bool = True,
                             quantile_filter_threshold: float = QUANTILE_FILTER
                             ) -> pd.DataFrame:
        """
        Splits users into personal or professional groups via HHI and car-share heuristics.

        The inverse HHI captures vehicle diversity: a low HHI (high diversity)
        signals a fleet operator. The car-share check catches concentrated usage
        even when total vehicle count is small, since a mechanic may service many
        makes but each appears only once. The quantile filter catches high unique
        vehicle counts that both HHI and car share miss at scale.

        The three masks are OR'd so that a user classified as personal by any
        criterion is kept in the personal group, erring on the side of inclusion.
        """

        def car_share_check(x: pd.Series,
                            threshold: int = DEFAULT_CAR_SHARE_ABS,
                            threshold_fraction: float = DEFAULT_CAR_SHARE_FRACTION) -> bool:
            # Users with fewer vehicles than the threshold cannot be fleet operators
            # by the absolute criterion, so they pass unconditionally
            if len(x) <= threshold:
                return True
            return x.sort_values(ascending=False).cumsum().iloc[threshold - 1] >= threshold_fraction

        df = self._filter_rows_from_burst(df)

        quantile_filter_value = (
            df.groupby(USER_ID_COL)[VEHICLE_ID_COL]
            .nunique()
            .quantile(quantile_filter_threshold)
        )

        per_user_prop_vehicle = (
            df.groupby(USER_ID_COL)[VEHICLE_ID_COL]
            .value_counts(normalize=True)
        )

        per_user_effective_hhi = (
            per_user_prop_vehicle.pow(2)
            .groupby(by=USER_ID_COL)
            .sum()
            .pow(-1)
        )

        quantile_mask_vehicle_mask = (
            df.groupby(USER_ID_COL)[VEHICLE_ID_COL].nunique() <= quantile_filter_value
        )

        effective_hhi_mask = per_user_effective_hhi <= inverse_hhi_threshold

        car_share_mask = (
            per_user_prop_vehicle
            .groupby(USER_ID_COL)
            .apply(lambda x: car_share_check(x, car_share_threshold_abs, car_share_threshold_fraction))
        )

        # Sorting indices so the OR operation aligns correctly across all three masks
        quantile_mask_vehicle_mask.sort_index(inplace=True)
        effective_hhi_mask.sort_index(inplace=True)
        car_share_mask.sort_index(inplace=True)

        combined_or_mask = quantile_mask_vehicle_mask | effective_hhi_mask | car_share_mask

        if return_personal_use_users:
            print("Filtering by personal users!")
            users = combined_or_mask[combined_or_mask].index
        else:
            print("Filtering by professional users!")
            users = combined_or_mask[~combined_or_mask].index

        return df[df[USER_ID_COL].isin(users)].copy()

    def get_clean_data(self,
                       merge_data_frames: bool = True,
                       filter_inactivity: bool = False,
                       filter_nan_cols: list[str] | None = None,
                       filter_by_user_type: bool = False,
                       return_personal_use_users: bool = False,
                       filter_one_day_users: bool = False,
                       filter_by_set_cutoff_date: bool = False,
                       transform_vehicle_end_year_to_present: bool = False,
                       filter_nan_vehicle_metadata: bool = False,
                       threshold_value: int = DEFAULT_THRESHOLD_DAYS,
                       inverse_hhi_threshold: int = DEFAULT_HHI_THRESHOLD,
                       car_share_threshold_abs: int = DEFAULT_CAR_SHARE_ABS,
                       car_share_threshold_fraction: float = DEFAULT_CAR_SHARE_FRACTION,
                       save_file_to: Path | None = None
                       ) -> pd.DataFrame:
        """
        Orchestrates the full cleaning pipeline.

        Steps are opt-in via boolean flags so the caller controls exactly which
        transformations run without subclassing or monkey-patching. Order matters:
        end-year imputation should precede inactivity filtering so churn_adjusted_date
        is computed on a complete vehicle age column.
        """
        step = 1
        df = self.df_activity.copy()

        if merge_data_frames:
            step = self.__step_counter(step)
            df = self.basic_filter_and_merge_df(filter_nan_cols)

        if transform_vehicle_end_year_to_present:
            step = self.__step_counter(step)

            print("Transforming missing vehicle_end_year values to current year")

            current_year = datetime.date.today().year
            missing_before = df[VEHICLE_END_YEAR_COL].isna().sum()

            df = df.copy()
            # Null end year means the model is still in production; flagging before
            # filling so the boolean is derived from the original missingness pattern
            df[STILL_IN_PRODUCTION_COL] = df[VEHICLE_END_YEAR_COL].isna()
            df[VEHICLE_END_YEAR_COL] = df[VEHICLE_END_YEAR_COL].fillna(current_year)

            print(f"Current year used: {current_year}")
            print(f"Missing vehicle_end_year values filled: {missing_before}")
            print()

        if filter_one_day_users:
            step = self.__step_counter(step)

            row_count_before = len(df)
            df = self.filter_one_day_users(df)
            row_count_after = len(df)

            print(f"Rows before one-day-user filtering: {row_count_before}")
            print(f"Rows after one-day-user filtering: {row_count_after}")
            print(f"Rows removed: {row_count_before - row_count_after}")
            print()

        if filter_by_set_cutoff_date:
            step = self.__step_counter(step)

            row_count_before = len(df)
            df = self.filter_users_by_set_cutoff_date(df, threshold_value)
            row_count_after = len(df)

            print(f"Rows before set cutoff date filtering: {row_count_before}")
            print(f"Rows after set cutoff date filtering: {row_count_after}")
            print(f"Rows removed: {row_count_before - row_count_after}")
            print()

        if filter_by_user_type:
            step = self.__step_counter(step)

            print(f"Rows before user type filtering: {len(df)}")

            df = self.filter_users_by_type(
                df,
                inverse_hhi_threshold,
                car_share_threshold_abs,
                car_share_threshold_fraction,
                return_personal_use_users
            )

            print(f"Rows after user type filtering: {len(df)}")
            print()

        if filter_nan_vehicle_metadata:
            columns = [VEHICLE_MAKE_COL, VEHICLE_MODEL_COL, VEHICLE_MILEAGE_COL]

            step = self.__step_counter(step)

            row_count_before = len(df)
            df = self.filter_nan_vehicle_metadata(df, columns)
            row_count_after = len(df)

            print(f"Rows before vehicle metadata filtering: {row_count_before}")
            print(f"Rows after vehicle metadata filtering: {row_count_after}")
            print(f"Rows removed: {row_count_before - row_count_after}")
            print()

        if filter_inactivity:
            step = self.__step_counter(step)

            print(f"Filtering activity after inactivity threshold: {threshold_value} days")

            row_count_before = len(df)
            user_count_before = df[USER_ID_COL].nunique()

            df = self.filter_after_inactivity(df, threshold_value=threshold_value)

            row_count_after = len(df)
            user_count_after = df[USER_ID_COL].nunique()
            churned_users = df[CHURN_TRIGGERED_COL].sum() if CHURN_TRIGGERED_COL in df.columns else 0

            print(f"Rows before inactivity filtering: {row_count_before}")
            print(f"Rows after inactivity filtering: {row_count_after}")
            print(f"Rows removed: {row_count_before - row_count_after}")
            print(f"Unique users before: {user_count_before}")
            print(f"Unique users after: {user_count_after}")
            print(f"Churn-triggering rows: {churned_users}")
            print()

        print()
        print("Cleaning complete")
        print(f"Final rows: {len(df)}")
        print(f"Final unique users: {df[USER_ID_COL].nunique()}")
        print()

        if save_file_to:
            step = self.__step_counter(step)

            print(f"Saving File to {str(save_file_to)}")

            string_length = len(str(save_file_to).split("\\")[-1])
            last_4_letters = str(save_file_to).split("\\")[-1][string_length - 4:]

            file_name = DEFAULT_OUTPUT_FILENAME

            if filter_by_user_type and return_personal_use_users:
                file_name = PERSONAL_USERS_FILENAME
            elif filter_by_user_type and not return_personal_use_users:
                file_name = PROFESSIONAL_USERS_FILENAME

            # If the path already ends in .csv the caller named the file explicitly;
            # otherwise a default filename is appended to the directory path
            if last_4_letters == CSV_EXTENSION:
                output_path = save_file_to
                df.to_csv(output_path, index=False)
                return df

            output_path = save_file_to / file_name
            df.to_csv(output_path, index=False)

        return df


# Example usage:

df_ac = pd.read_csv(paths_to_files_and_folders.PATH_TO_RAW_ACTIVITY_DATA_1000)
df_vh = pd.read_csv(paths_to_files_and_folders.PATH_TO_RAW_VEHICLE_DATA_1000)

data_cleaner = DataCleaner(df_ac, df_vh)

personal = data_cleaner.get_clean_data(
                            merge_data_frames=True,
                            # filter_inactivity=True,
                            filter_nan_cols=None,
                            filter_by_user_type=True,
                            return_personal_use_users=True,
                            filter_one_day_users=True,
                            # filter_by_set_cutoff_date=True,
                            transform_vehicle_end_year_to_present=True,
                            # filter_nan_vehicle_metadata=True,
                            # threshold_value=180,
                            inverse_hhi_threshold=6,
                            car_share_threshold_abs=4,
                            car_share_threshold_fraction=0.8)