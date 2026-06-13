import pandas as pd
import datetime
from .constants import paths_to_files_and_folders
from pathlib import Path


import datetime
from pathlib import Path

import pandas as pd

from .constants.columns import (
    USER_ID_COL, ACTIVITY_DATE_COL, CHURN_ADJUSTED_DATE_COL,
    CHURN_TRIGGERED_COL, VEHICLE_ID_COL, VEHICLE_MAKE_COL,
    VEHICLE_MODEL_COL, VEHICLE_END_YEAR_COL, VEHICLE_MILEAGE_COL,
    STILL_IN_PRODUCTION_COL,
)


NEXT_DATE_COL: str = "next_date"
ACTIVITY_GAP_COL: str = "activity_gap"

MERGE_KEYS: list[str] = [VEHICLE_ID_COL, USER_ID_COL]

VEHICLE_METADATA_COLS: list[str] = [VEHICLE_MAKE_COL, VEHICLE_MODEL_COL, VEHICLE_MILEAGE_COL]


CSV_EXTENSION: str = ".csv"
DEFAULT_OUTPUT_FILENAME: str = "no_name.csv"
PERSONAL_USERS_FILENAME: str = "personal_users_dataset.csv"
PROFESSIONAL_USERS_FILENAME: str = "professional_users_dataset.csv"


DEFAULT_THRESHOLD_DAYS: int = 30
DEFAULT_HHI_THRESHOLD: int = 6
DEFAULT_CAR_SHARE_ABS: int = 4
DEFAULT_CAR_SHARE_FRACTION: float = 0.8
MIN_ACTIVITY_SPAN_DAYS: int = 1


class DataCleaner:
    def __init__(self,
                 df_activity: pd.DataFrame,
                 df_vehicle: pd.DataFrame) -> None:
        """
        Initialises the cleaner with activity and vehicle dataframes.

        Deep copies of both inputs are stored so the originals are never
        mutated. The maximum activity date is cached as the study end date
        (max_date), used later for censoring/churn decisions.

        Args:
            df_activity:  Raw activity logs.
            df_vehicle:   Raw vehicle metadata.
        """
        self.df_activity = df_activity.copy(deep=True)
        self.df_vehicle = df_vehicle.copy(deep=True)

        self.max_date = pd.to_datetime(df_activity[ACTIVITY_DATE_COL].max())

    def __step_counter(self, step_counter: int = 1) -> int:
        """
        Prints a step banner and returns the next step number.

        Args:
            step_counter:  Current step number to print. Default is 1.

        Returns:
            The incremented step number (step_counter + 1).
        """
        print(f"__Step {step_counter}__ ")
        return step_counter + 1

    def basic_filter_and_merge_df(self,
                                  filter_nan_cols: list[str] | None = None,
                                  user_id_col_name: str = USER_ID_COL,
                                  vehicle_id_col_name: str = VEHICLE_ID_COL) -> pd.DataFrame:
        """
        Merges vehicle metadata with activity logs and applies basic cleaning.

        Steps performed:
        1. Merge  — Right join of vehicle metadata (df_vehicle) onto activity logs (df_activity) on vehicle_id + user_id. All activity rows are kept; vehicle metadata is added where available.

        2. NaN filter (optional) — Drops rows with missing values in the specified columns.

        3. User filter — Removes activity rows for users not present in the vehicle dataset (no vehicle metadata at all).

        4. Deduplication — Removes exact duplicate rows.

        Args:
            filter_nan_cols:    List of column names to check for missing values. If None, no NaN filtering is applied.
            user_id_col_name:   Column identifying each user.

        Returns:
            Merged and cleaned DataFrame.
        """

        # Filtering users with duplicate vehicle id's, but different vehicle characteristics
        dupe_users = (
            self.df_vehicle
            .groupby([user_id_col_name, vehicle_id_col_name])[vehicle_id_col_name]
            .count()
            .gt(1)
            .groupby(level=0)
            .any()
        )
        dupe_user_ids = dupe_users[dupe_users].index

        self.df_vehicle = self.df_vehicle[~self.df_vehicle[user_id_col_name].isin(dupe_user_ids)]


        print(f"Rows before merging: {len(self.df_activity)}")
        df = self.df_vehicle.merge(self.df_activity, on=MERGE_KEYS, how="right")
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

        # Removing user_id's that aren't present in vehicle dataset
        print("Removing user_id's that aren't present in vehicle dataset")
        print(f"Rows before user filtering: {len(df)}")
        unique_users = self.df_vehicle[user_id_col_name].unique()
        unique_users_mask = df[user_id_col_name].isin(unique_users)

        df = df[unique_users_mask]
        print(f"Rows after user filtering: {len(df)}")
        print()

        # Removing user_id's that aren't present in vehicle dataset
        print("Removing duplicate rows")
        print(f"Rows before deduplication: {len(df)}")
        df.drop_duplicates(inplace=True)
        print(f"Rows after deduplication: {len(df)}")
        print()

        return df.copy()

    def filter_after_inactivity(self,
                                df: pd.DataFrame,
                                user_id_col_name: str = USER_ID_COL,
                                time_col_name: str = ACTIVITY_DATE_COL,
                                threshold_value: int = DEFAULT_THRESHOLD_DAYS,
                                ) -> pd.DataFrame:
        """
        Identifies and handles user churn based on inactivity gaps.

        A user is considered churned if:
        1. The gap between two consecutive activities exceeds the threshold, OR
        2. Their last recorded activity is more than threshold days before
            the end of the study (max_date)

        For each user, only the first churn trigger is kept - all activity
        rows after that point are removed.

        A churn_adjusted_date column is added:
        - Churned users:  last activity date + threshold (the churn moment)
        - Censored users: last activity date (still active at end of study)

        Args:
            df:                Input DataFrame with user activity logs.
            user_id_col_name:  Column identifying each user.
            time_col_name:     Column containing activity timestamps.
            threshold_value:   Inactivity threshold in days. Default is 30.

        Returns:
            DataFrame truncated at the first churn event per user,
            with churn_triggered and churn_adjusted_date columns added.
        """
        future_date_name = NEXT_DATE_COL
        activity_gap_name = ACTIVITY_GAP_COL
        churn_triggered_col_name = CHURN_TRIGGERED_COL
        churn_date_col_name = CHURN_ADJUSTED_DATE_COL

        df = df.copy(deep=True)

        df[time_col_name] = pd.to_datetime(df[time_col_name])
        df = df.sort_values([user_id_col_name, time_col_name])

        # Next activity date per user
        df[future_date_name] = df.groupby([user_id_col_name])[time_col_name].shift(-1)
        df[activity_gap_name] = (df[future_date_name] - df[time_col_name])

        # Condition 1: gap between consecutive activities exceeds threshold
        gap_churn = (
            df[activity_gap_name] >= pd.Timedelta(days=threshold_value)
        ) & df[activity_gap_name].notna()

        # Condition 2: last activity is more than threshold days before max_date
        is_last_row = df[future_date_name].isna()
        end_churn = is_last_row & (
            (self.max_date - df[time_col_name]) >= pd.Timedelta(days=threshold_value)
        )

        df[churn_triggered_col_name] = gap_churn | end_churn

        # Keep only first churn trigger per user (cummax trick)
        df[churn_triggered_col_name] = (
            df.groupby(user_id_col_name)[churn_triggered_col_name]
            .cummax())

        row_mask = df.groupby([user_id_col_name])[churn_triggered_col_name].shift(1, fill_value=False)

        df = df[~row_mask]

        df[churn_date_col_name] = df[time_col_name]

        # Note to self: Using df[[df[churn_triggered_col_name],churn_date_col_name]] = ... doesn't work because the operation makes a copy
        # of the dataframe modifying it instaed of the original

        # Churn date = activity_date + threshold
        df.loc[df[churn_triggered_col_name], churn_date_col_name] += pd.Timedelta(days=threshold_value)

        df.drop(columns=[activity_gap_name, future_date_name], inplace=True)

        print(df[churn_triggered_col_name].sum())
        print(df[user_id_col_name][df[churn_triggered_col_name]].nunique())

        return df.copy()

    def filter_nan_vehicle_metadata(self,
                                    df: pd.DataFrame,
                                    columns_by_which_to_drop: list[str] | None = None
                                    ) -> pd.DataFrame:
        """
        Removes rows with missing values in the specified vehicle metadata columns.

        Args:
            df:                       Input DataFrame.
            columns_by_which_to_drop: List of column names to check for NaN values.
                                        Rows with NaN in any of these columns are dropped.

        Returns:
            DataFrame with incomplete vehicle metadata rows removed.
        """
        df = df.dropna(subset=columns_by_which_to_drop).copy()

        return df.copy()

    def filter_one_day_users(self,
                             df: pd.DataFrame,
                             time_col_name: str = ACTIVITY_DATE_COL,
                             user_id_col_name: str = USER_ID_COL) -> pd.DataFrame:
        """
        Removes users whose entire activity history spans only one day.

        Users with fewer than 1 day between their first and last activity
        are considered too sparse for meaningful analysis and are excluded.

        Args:
            df:               Input DataFrame with user activity logs.
            time_col_name:    Column containing activity timestamps.
            user_id_col_name: Column identifying each user.

        Returns:
            DataFrame with single-day users removed.
        """
        df = df.copy(deep=True)
        df[time_col_name] = pd.to_datetime(df[time_col_name])

        earliest_activity_per_user = df.groupby(user_id_col_name)[time_col_name].min()
        latest_activity_per_user = df.groupby(user_id_col_name)[time_col_name].max()

        diff_between_earliest_and_last_usage = (latest_activity_per_user - earliest_activity_per_user)
        representative_users_mask = diff_between_earliest_and_last_usage > pd.Timedelta(MIN_ACTIVITY_SPAN_DAYS, unit="days")

        # Those that have used the device > 1 days
        representative_users = earliest_activity_per_user.index[representative_users_mask]
        print(f"Number of one-day-users: {sum(~representative_users_mask)}")

        representative_user_df_mask = df[user_id_col_name].isin(representative_users)
        return df[representative_user_df_mask].copy()

    def filter_users_by_set_cutoff_date(self,
                                        df: pd.DataFrame,
                                        time_col_name: str = ACTIVITY_DATE_COL,
                                        threshold_value: int = DEFAULT_THRESHOLD_DAYS) -> pd.DataFrame:
        """
        Filters users based on their observation window length.

        End of study is defined as the maximum date in the dataset - the point
        where data collection ended. By subtracting each user's first activity
        date from this end date, we get the total number of days they were
        observable. Users with fewer days than the threshold are removed, as they
        have insufficient history for analysis.

        Args:
            df:               Input DataFrame containing user activity logs.
            time_col_name:    Column name for activity timestamps.
            threshold_value:  Minimum number of days a user must be observable
                          to be included. Default is 30 days.

        Returns:
            Filtered DataFrame containing only users with sufficient history.
        """
        df[time_col_name] = pd.to_datetime(df[time_col_name])

        users_first_app_use_date = pd.Series(df.groupby([USER_ID_COL])[time_col_name].min())

        filtered_users = users_first_app_use_date[(self.max_date - users_first_app_use_date) > pd.Timedelta(days=threshold_value)]

        valid_users_mask = df[USER_ID_COL].isin(filtered_users.index)
        return df[valid_users_mask]

    def filter_users_by_type(self,
                             df: pd.DataFrame,
                             user_id_col_name: str = USER_ID_COL,
                             vehicle_id_col_name: str = VEHICLE_ID_COL,
                             effective_hhi_threshold: int = DEFAULT_HHI_THRESHOLD,
                             car_share_threshold_abs: int = DEFAULT_CAR_SHARE_ABS,
                             car_share_threshold_fraction: float = DEFAULT_CAR_SHARE_FRACTION,
                             return_personal_use_users: bool = True) -> pd.DataFrame:
        """
        Splits users into personal or professional groups based on vehicle
        usage concentration.

        Two conditions identify professional/fleet users (OR logic):
        1. Effective HHI <= threshold — user spreads activity across
            many vehicles (low concentration = fleet-like behaviour).
        2. Car share check — top N vehicles account for >= 80% of
            activity (high concentration on a small number of vehicles).

        Args:
            df:                          Input DataFrame.
            user_id_col_name:            Column identifying each user.
            vehicle_id_col_name:         Column identifying each vehicle.
            effective_hhi_threshold:     HHI threshold for fleet classification.
            car_share_threshold_abs:     Number of top vehicles to check.
            car_share_threshold_fraction: Minimum fraction of activity on top vehicles.
            return_personal_use_users:   If True, returns personal users; else professional.

        Returns:
            DataFrame filtered to the selected user type.
        """

        def car_share_check(x: pd.Series,
                            threshold: int = DEFAULT_CAR_SHARE_ABS,
                            threshold_fraction: float = DEFAULT_CAR_SHARE_FRACTION) -> bool:
            if len(x) <= threshold:
                return True

            else:
                return x.cumsum().iloc[threshold - 1] >= threshold_fraction
        per_user_prop_vehicle = (
            df
            .groupby(user_id_col_name)[vehicle_id_col_name]
            .value_counts(normalize=True))

        per_user_effective_hhi = (
            per_user_prop_vehicle.pow(2).groupby(by=user_id_col_name)
                                 .sum()
                                 .pow(-1)
        )

        effective_hhi_mask = per_user_effective_hhi <= effective_hhi_threshold
        car_share_mask = (per_user_prop_vehicle
                          .groupby(user_id_col_name)
                          .apply(lambda x: car_share_check(x, car_share_threshold_abs, car_share_threshold_fraction)))

        combined_OR_mask = effective_hhi_mask | car_share_mask

        if return_personal_use_users:
            print("Filtering by personal users!")
            users = combined_OR_mask[combined_OR_mask].index
        else:
            print("Filtering by professional users!")
            users = combined_OR_mask[~combined_OR_mask].index

        users_mask = df[user_id_col_name].isin(users)

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
                       user_id_col_name: str = USER_ID_COL,
                       time_col_name: str = ACTIVITY_DATE_COL,
                       vehicle_id_col_name: str = VEHICLE_ID_COL,
                       threshold_value: int = DEFAULT_THRESHOLD_DAYS,
                       inverse_hhi_threshold: int = DEFAULT_HHI_THRESHOLD,
                       car_share_threshold_abs: int = DEFAULT_CAR_SHARE_ABS,
                       car_share_threshold_fraction: float = DEFAULT_CAR_SHARE_FRACTION,
                       save_file_to: Path | None = None,
                       ) -> pd.DataFrame:
        """
        Main data cleaning pipeline. Runs a configurable sequence of filtering
        and transformation steps on the raw activity data.

        Steps (enabled via boolean flags, executed in order):

        1. merge_data_frames          — Merges raw dataframes and applies basic filters.
        2. transform_vehicle_end_year — Fills missing vehicle_end_year with the current year and adds a still_in_production boolean flag.
        3. filter_one_day_users       — Removes users whose entire history spans only one day.
        4. filter_by_set_cutoff_date  — Removes users with insufficient observation history relative to the cutoff date.
        5. filter_by_user_type        — Splits users into personal or professional groups based on HHI and vehicle share thresholds.
        6. filter_nan_vehicle_metadata — Removes rows with missing vehicle make, model, or mileage.
        7. filter_inactivity          — Identifies churn events based on inactivity gaps,truncates each user's history at their first churn trigger, and adds churn_triggered and churn_adjusted_date columns.

        Args:
            merge_data_frames:                    Whether to merge and basic-filter raw data.
            filter_inactivity:                    Whether to apply inactivity-based churn labeling.
            filter_nan_cols:                      Columns to check for NaN values during merge step.
            filter_by_user_type:                  Whether to split by personal/professional.
            return_personal_use_users:            If True, returns personal users; else professional.
            filter_one_day_users:                 Whether to remove single-day users.
            filter_by_set_cutoff_date:            Whether to remove users with insufficient history.
            transform_vehicle_end_year_to_present: Whether to fill missing vehicle_end_year.
            filter_nan_vehicle_metadata:          Whether to remove rows with missing vehicle info.
            user_id_col_name:                     Column identifying each user.
            time_col_name:                        Column containing activity timestamps.
            vehicle_id_col_name:                  Column identifying each vehicle.
            threshold_value:                      Inactivity threshold in days. Default 30.
            inverse_hhi_threshold:                HHI threshold for user type classification.
            car_share_threshold_abs:              Minimum number of dominant vehicles.
            car_share_threshold_fraction:         Minimum fraction of activity on top vehicles.
            save_file_to:                         Path to save the output CSV. If the path ends
                                                      in .csv, saves directly to that file.

        Returns:
            Cleaned DataFrame with all selected transformations applied.
        """
        step = 1

        if merge_data_frames:
            step = self.__step_counter(step)

            df = self.basic_filter_and_merge_df(filter_nan_cols)

        # TODO: Vehicle end year as a variable
        if transform_vehicle_end_year_to_present:
            step = self.__step_counter(step)
            print("Transforming missing vehicle_end_year values to current year")

            current_year = datetime.date.today().year
            missing_before = df[VEHICLE_END_YEAR_COL].isna().sum()

            df = df.copy()
            df[STILL_IN_PRODUCTION_COL] = df[VEHICLE_END_YEAR_COL].isna()
            df[VEHICLE_END_YEAR_COL] = df[VEHICLE_END_YEAR_COL].fillna(current_year)

            print(f"Current year used: {current_year}")
            print(f"Missing vehicle_end_year values filled: {missing_before}")
            print()

        if filter_one_day_users:
            step = self.__step_counter(step)
            row_count_before = len(df)
            df = self.filter_one_day_users(df, time_col_name, user_id_col_name)
            row_count_after = len(df)

            print(f"Rows before one-day-user filtering: {row_count_before}")
            print(f"Rows after one-day-user filtering: {row_count_after}")
            print(f"Rows removed: {row_count_before - row_count_after}")
            print()

        if filter_by_set_cutoff_date:
            step = self.__step_counter(step)
            row_count_before = len(df)

            df = self.filter_users_by_set_cutoff_date(df,
                                                      time_col_name,
                                                      threshold_value)
            row_count_after = len(df)

            print(f"Rows before set cutoff date filtering: {row_count_before}")
            print(f"Rows after set cutoff date filtering: {row_count_after}")
            print(f"Rows removed: {row_count_before - row_count_after}")
            print()

        if filter_by_user_type:
            step = self.__step_counter(step)

            print(f"Rows before user type filtering: {len(df)}")

            df = self.filter_users_by_type(df, user_id_col_name,
                                           vehicle_id_col_name,
                                           inverse_hhi_threshold,
                                           car_share_threshold_abs,
                                           car_share_threshold_fraction,
                                           return_personal_use_users)

            print(f"Rows after user type filtering: {len(df)}")
            print()

        # TODO: IMPUTE VEHICLE START YEAR BASED ON KNN (MILEAGE,MAKE,MODEL)

        # This goes after filtering by user_type (important, because the code segments based on vehicle_id's)
        if filter_nan_vehicle_metadata:

            columns = VEHICLE_METADATA_COLS

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
            user_count_before = df[user_id_col_name].nunique()

            df = self.filter_after_inactivity(
                df,
                user_id_col_name=user_id_col_name,
                time_col_name=time_col_name,
                threshold_value=threshold_value
            )

            row_count_after = len(df)
            user_count_after = df[user_id_col_name].nunique()
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
        print(f"Final unique users: {df[user_id_col_name].nunique()}")
        print()

        if save_file_to:
            step = self.__step_counter(step)
            string_length = len(str(save_file_to).split("\\")[-1])
            last_4_letters = str(save_file_to).split("\\")[-1][string_length - 4:]

            file_name = DEFAULT_OUTPUT_FILENAME

            if filter_by_user_type and return_personal_use_users:
                file_name = PERSONAL_USERS_FILENAME

            elif filter_by_user_type and not return_personal_use_users:
                file_name = PROFESSIONAL_USERS_FILENAME

            if last_4_letters == CSV_EXTENSION:
                file_name = str(save_file_to).split("\\")[-1]
                output_path = save_file_to
                df.to_csv(output_path)
                return df

            output_path = save_file_to / file_name
            df.to_csv(output_path)

        return df

# df_ac = pd.read_csv(paths_to_files_and_folders.PATH_TO_RAW_ACTIVITY_DATA_1000)
# df_vh = pd.read_csv(paths_to_files_and_folders.PATH_TO_RAW_VEHICLE_DATA_1000)
# df_vh = pd.read_csv(Path(r"C:\Users\Tomas\Desktop\Thesis Stuff\Coding\Data\interim\personal_users_dataset.csv"))



# data_cleaner = DataCleaner(df_ac,df_vh)

# data_cleaner.get_clean_data(merge_data_frames=True,
#                             filter_nan_cols=["vehicle_id"],
#                             transform_vehicle_end_year_to_present=True,
#                             filter_by_user_type=True,
#                             flter_by_set_cutoff_date=True,
#                             return_personal_use_users=True,
#                             threshold_value=120,
#                             save_file_to=Path(r"C:\Users\Tomas\Desktop\Thesis Stuff\Coding\Data\interim"),
#                             filter_nan_vehicle_metadata=True,
#                             filter_one_day_users=True)

# data_cleaner.get_clean_data(merge_data_frames=True,
#                             filter_nan_cols=["vehicle_id"])

# data_cleaner.filter_after_inactivity(df_ac, threshold_value=180)
# data_cleaner.filter_users_by_set_cutoff_date(df_ac)