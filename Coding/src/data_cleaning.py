import datetime
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from .constants import paths_to_files_and_folders
from .constants.columns import (
    USER_ID_COL, ACTIVITY_DATE_COL, CHURN_ADJUSTED_DATE_COL,
    CHURN_TRIGGERED_COL, VEHICLE_ID_COL, VEHICLE_MAKE_COL,
    VEHICLE_MODEL_COL, VEHICLE_END_YEAR_COL, VEHICLE_MILEAGE_COL,
    STILL_IN_PRODUCTION_COL,
)


# ============================================================
#  Internal working columns (created and dropped within methods)
# ============================================================
NEXT_DATE_COL: str = "next_date"
ACTIVITY_GAP_COL: str = "activity_gap"


# ============================================================
#  Output file handling
# ============================================================
CSV_EXTENSION: str = ".csv"
DEFAULT_OUTPUT_FILENAME: str = "no_name.csv"
PERSONAL_USERS_FILENAME: str = "personal_users_dataset.csv"
PROFESSIONAL_USERS_FILENAME: str = "professional_users_dataset.csv"


# ============================================================
#  Tuning defaults
# ============================================================
DEFAULT_THRESHOLD_DAYS: int = 160
DEFAULT_HHI_THRESHOLD: int = 6
DEFAULT_CAR_SHARE_ABS: int = 4
DEFAULT_CAR_SHARE_FRACTION: float = 0.8
MIN_ACTIVITY_SPAN_DAYS: int = 1


# ============================================================
#  Sentinel values
# ============================================================
FILL_VALUE_CHURN_TRIGGERED: bool = False


@dataclass
class ColumnConfig:
    """
    Central schema configuration: every column name the pipeline reads or writes.
    """
    user_id: str = USER_ID_COL
    activity_date: str = ACTIVITY_DATE_COL
    vehicle_id: str = VEHICLE_ID_COL
    vehicle_make: str = VEHICLE_MAKE_COL
    vehicle_model: str = VEHICLE_MODEL_COL
    vehicle_mileage: str = VEHICLE_MILEAGE_COL
    vehicle_end_year: str = VEHICLE_END_YEAR_COL
    still_in_production: str = STILL_IN_PRODUCTION_COL
    churn_triggered: str = CHURN_TRIGGERED_COL
    churn_adjusted_date: str = CHURN_ADJUSTED_DATE_COL


class DataCleaner:
    def __init__(self,
                 df_activity: pd.DataFrame,
                 df_vehicle: pd.DataFrame,
                 columns: ColumnConfig | None = None) -> None:
        self.df_activity = df_activity.copy(deep=True)
        self.df_vehicle = df_vehicle.copy(deep=True)
        self.cols = columns or ColumnConfig()

        self.max_date = pd.to_datetime(df_activity[self.cols.activity_date].max())

    def __step_counter(self, step_counter: int = 1) -> int:
        print(f"_Step {step_counter}_")
        return step_counter + 1

    def basic_filter_and_merge_df(self,
                                  filter_nan_cols: list[str] | None = None) -> pd.DataFrame:
        """
        Merges vehicle metadata with activity logs and applies basic cleaning.
        """

        # Users with the same vehicle_id appearing under different metadata rows are
        # ambiguous — keeping them would silently duplicate activity rows on merge
        dupe_users_mask = (
            self.df_vehicle
            .groupby([self.cols.user_id, self.cols.vehicle_id])[self.cols.vehicle_id]
            .count()
            .gt(1)
            .groupby(level=0)
            .any()
        )
        dupe_user_ids = dupe_users_mask[dupe_users_mask].index

        self.df_vehicle = self.df_vehicle[
            ~self.df_vehicle[self.cols.user_id].isin(dupe_user_ids)
        ]

        print(f"Rows before merging: {len(self.df_activity)}")
        df = self.df_vehicle.merge(
            self.df_activity,
            on=[self.cols.vehicle_id, self.cols.user_id],
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
        # downstream, so they are removed rather than carried as nulls
        print("Removing user_ids that aren't present in vehicle dataset")
        print(f"Rows before user filtering: {len(df)}")

        unique_users = self.df_vehicle[self.cols.user_id].unique()
        unique_users_mask = df[self.cols.user_id].isin(unique_users)

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

        df[self.cols.activity_date] = pd.to_datetime(df[self.cols.activity_date])
        df = df.sort_values([self.cols.user_id, self.cols.activity_date])

        df[NEXT_DATE_COL] = (
            df.groupby(self.cols.user_id)[self.cols.activity_date].shift(-1)
        )

        df[ACTIVITY_GAP_COL] = (
            df[NEXT_DATE_COL] - df[self.cols.activity_date]
        )

        # Condition 1: gap between consecutive activities exceeds threshold
        gap_churn = (
            df[ACTIVITY_GAP_COL] >= pd.Timedelta(days=threshold_value)
        ) & df[ACTIVITY_GAP_COL].notna()

        # Condition 2: last activity is more than threshold days before max_date
        is_last_row = df[NEXT_DATE_COL].isna()

        end_churn = is_last_row & (
            (self.max_date - df[self.cols.activity_date]) >= pd.Timedelta(days=threshold_value)
        )

        df[self.cols.churn_triggered] = gap_churn | end_churn

        # cummax propagates the first True forward so all rows after the trigger
        # are also marked, letting the shift-mask below drop them cleanly
        df[self.cols.churn_triggered] = (
            df.groupby(self.cols.user_id)[self.cols.churn_triggered].cummax()
        )

        # shift(1) offsets the cummax flag by one row so the trigger row itself
        # is kept (it is the churn event) but every row after it is dropped
        row_mask = (
            df.groupby(self.cols.user_id)[self.cols.churn_triggered]
            .shift(1, fill_value=FILL_VALUE_CHURN_TRIGGERED)
        )

        df = df[~row_mask]

        df[self.cols.churn_adjusted_date] = df[self.cols.activity_date]

        # Churn date is pushed forward by the threshold so the interval grid
        # covers the full at-risk window, not just the last observed activity
        df.loc[
            df[self.cols.churn_triggered],
            self.cols.churn_adjusted_date
        ] += pd.Timedelta(days=threshold_value)

        df.drop(columns=[ACTIVITY_GAP_COL, NEXT_DATE_COL], inplace=True)

        return df.copy()

    def filter_nan_vehicle_metadata(self,
                                    df: pd.DataFrame,
                                    columns_by_which_to_drop: list[str] | None = None
                                    ) -> pd.DataFrame:
        """
        Removes rows with missing values in vehicle metadata columns.
        """
        df = df.dropna(subset=columns_by_which_to_drop).copy()
        return df.copy()

    def filter_one_day_users(self,
                             df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes users whose activity history spans only one day.

        Single-day users have no observable inactivity pattern and cannot
        contribute meaningful interval-level features to the survival model.
        """
        df = df.copy(deep=True)
        df[self.cols.activity_date] = pd.to_datetime(df[self.cols.activity_date])

        earliest_activity_per_user = df.groupby(self.cols.user_id)[self.cols.activity_date].min()
        latest_activity_per_user = df.groupby(self.cols.user_id)[self.cols.activity_date].max()

        diff_between_earliest_and_last_usage = (
            latest_activity_per_user - earliest_activity_per_user
        )

        representative_users_mask = (
            diff_between_earliest_and_last_usage > pd.Timedelta(
                MIN_ACTIVITY_SPAN_DAYS, unit="days"
            )
        )

        representative_users = earliest_activity_per_user.index[representative_users_mask]

        print(f"Number of one-day-users: {sum(~representative_users_mask)}")

        representative_user_df_mask = df[self.cols.user_id].isin(representative_users)

        return df[representative_user_df_mask].copy()

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
        df[self.cols.activity_date] = pd.to_datetime(df[self.cols.activity_date])

        users_first_app_use_date = pd.Series(
            df.groupby(self.cols.user_id)[self.cols.activity_date].min()
        )

        filtered_users = users_first_app_use_date[
            (self.max_date - users_first_app_use_date) >
            pd.Timedelta(days=threshold_value)
        ]

        valid_users_mask = df[self.cols.user_id].isin(filtered_users.index)

        return df[valid_users_mask]

    def filter_users_by_type(self,
                             df: pd.DataFrame,
                             inverse_hhi_threshold: int = DEFAULT_HHI_THRESHOLD,
                             car_share_threshold_abs: int = DEFAULT_CAR_SHARE_ABS,
                             car_share_threshold_fraction: float = DEFAULT_CAR_SHARE_FRACTION,
                             return_personal_use_users: bool = True
                             ) -> pd.DataFrame:
        """
        Splits users into personal or professional groups via HHI and car-share heuristics.

        The inverse HHI captures vehicle diversity: a low HHI (high diversity)
        signals a fleet operator. The car-share check catches concentrated usage
        even when total vehicle count is small, since a mechanic may service many
        makes but each appears only once.

        The two masks are OR'd so that a user classified as personal by either
        criterion is kept in the personal group, erring on the side of inclusion.
        """

        def car_share_check(x: pd.Series,
                            threshold: int = DEFAULT_CAR_SHARE_ABS,
                            threshold_fraction: float = DEFAULT_CAR_SHARE_FRACTION) -> bool:
            # Users with fewer vehicles than the threshold cannot be fleet operators
            # by the absolute criterion, so they pass unconditionally
            if len(x) <= threshold:
                return True
            return x.cumsum().iloc[threshold - 1] >= threshold_fraction

        per_user_prop_vehicle = (
            df.groupby(self.cols.user_id)[self.cols.vehicle_id]
            .value_counts(normalize=True)
        )

        per_user_effective_hhi = (
            per_user_prop_vehicle.pow(2)
            .groupby(by=self.cols.user_id)
            .sum()
            .pow(-1)
        )

        effective_hhi_mask = per_user_effective_hhi <= inverse_hhi_threshold

        car_share_mask = (
            per_user_prop_vehicle
            .groupby(self.cols.user_id)
            .apply(
                lambda x: car_share_check(
                    x,
                    car_share_threshold_abs,
                    car_share_threshold_fraction
                )
            )
        )

        combined_or_mask = effective_hhi_mask | car_share_mask

        if return_personal_use_users:
            print("Filtering by personal users!")
            users = combined_or_mask[combined_or_mask].index
        else:
            print("Filtering by professional users!")
            users = combined_or_mask[~combined_or_mask].index

        users_mask = df[self.cols.user_id].isin(users)

        return df[users_mask].copy()

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
            missing_before = df[self.cols.vehicle_end_year].isna().sum()

            df = df.copy()
            # Null end year means the model is still in production; flagging before
            # filling so the boolean is derived from the original missingness pattern
            df[self.cols.still_in_production] = df[self.cols.vehicle_end_year].isna()
            df[self.cols.vehicle_end_year] = df[self.cols.vehicle_end_year].fillna(current_year)

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
            columns = [
                self.cols.vehicle_make,
                self.cols.vehicle_model,
                self.cols.vehicle_mileage
            ]

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
            user_count_before = df[self.cols.user_id].nunique()

            df = self.filter_after_inactivity(
                df,
                threshold_value=threshold_value
            )

            row_count_after = len(df)
            user_count_after = df[self.cols.user_id].nunique()
            churned_users = (
                df[self.cols.churn_triggered].sum()
                if self.cols.churn_triggered in df.columns else 0
            )

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
        print(f"Final unique users: {df[self.cols.user_id].nunique()}")
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
                file_name = str(save_file_to).split("\\")[-1]
                output_path = save_file_to
                df.to_csv(output_path, index=False)
                return df

            output_path = save_file_to / file_name
            df.to_csv(output_path, index=False)

        return df


# Example usage:

# df_ac = pd.read_csv(paths_to_files_and_folders.PATH_TO_RAW_ACTIVITY_DATA_1000)
# df_vh = pd.read_csv(paths_to_files_and_folders.PATH_TO_RAW_VEHICLE_DATA_1000)

# data_cleaner = DataCleaner(df_ac, df_vh)

# data_cleaner.get_clean_data(
#     merge_data_frames=True,
#     filter_nan_cols=["vehicle_id"],
#     transform_vehicle_end_year_to_present=True,
#     filter_by_user_type=True,
#     filter_by_set_cutoff_date=True,
#     return_personal_use_users=True,
#     threshold_value=120,
#     save_file_to=Path(r"C:\Users\Tomas\Desktop\Thesis Stuff\Coding\Data\interim"),
#     filter_nan_vehicle_metadata=True,
#     filter_one_day_users=True
# )