
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
N_NEIGHBOURS = 5
COL_TEMPLATE_FORMAT_AT = "n_{a}_{start}_{end}_days" 



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
                .str.strip_chars(),
            pl.col(VEHICLE_MAKE_COL)
                .str.to_lowercase()
                .str.strip_chars()
                .replace(brand_group_map)
        )

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
            pl.col(VEHICLE_MAKE_COL).fill_null("unknown"),
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

    def _generate_activity_feature_aggregation(self,
                                               lookback_periods: tuple[int, ...] = LOOKBACK_PERIODS,
                                               interval_in_days: int = INTERVAL_IN_DAYS) -> tuple[list, list]:
        
        
        
        post_agg: list = []
        column_names_to_keep: list = []
        col_registry:
        activity_types: list[str] = list(sorted(set(activity_type_groups.values())))


        agg: list = [(pl.col(ACTIVITY_TYPE_COL) != "none").sum().alias("n_actions_current")] + \
                    [(pl.col(ACTIVITY_TYPE_COL) == a).sum().alias(COL_TEMPLATE_FORMAT_AT.format(a=a,
                                                                                                start = 0,
                                                                                                end   = interval_in_days)) for a in activity_types]

        # post_agg: list = [(pl.col(f"n_{a}") / pl.col("n_actions")).fill_nan(0.0).alias(f"prop_{a}") for a in activity_types]


        for period in lookback_periods:
            post_agg += [
                pl.col(COL_TEMPLATE_FORMAT_AT.format(a=a, start=0, end=interval_in_days))
                .shift(period, fill_value=0)
                .alias(COL_TEMPLATE_FORMAT_AT.format(a=a, start=interval_in_days * period, end=interval_in_days * (period + 1)))
                for a in activity_types
                ]
            
            post_agg += [
                pl.col()
            ]
        # post_agg += [pl.col()]
        # column_names_to_keep += 

        return agg, post_agg
    
    def _generate_vehicle_make_feature_aggregation(self) -> tuple[list, list]:
        make_groups: list[str] = list(set(brand_group_map.values()))

        agg: list = [
            pl.col(VEHICLE_MAKE_COL).filter(pl.col(VEHICLE_MAKE_COL) != "unknown").n_unique().alias("n_unique_makes"),
            pl.col(VEHICLE_ID_COL).filter(pl.col(VEHICLE_ID_COL) != "unknown").n_unique().alias("n_unique_vehicles"),
            pl.col(VEHICLE_ID_COL).filter(pl.col(VEHICLE_ID_COL) != "unknown").count().alias("n_vehicles"),
        ] + [
            (pl.col(VEHICLE_MAKE_COL) == make).sum().alias(f"n_{make}")
            for make in make_groups
        ]

        post_agg: list = [
            (pl.col(f"n_{make}") / pl.col("n_vehicles")).fill_nan(0.0).alias(f"prop_{make}")
            for make in make_groups
        ]

        return agg, post_agg

    def _generate_app_column_feature_aggregation(self) -> tuple[list, list]:
        apps: list[str] = list(app_type_groups.keys())

        agg: list = [
            (pl.col(APP_COL) != "unknown").sum().alias("n_app_total")
        ] + [
            (pl.col(APP_COL) == app).sum().alias(f"n_{app}")
            for app in apps
        ]

        post_agg: list = [
            (pl.col(f"n_{app}") / pl.col("n_app_total")).fill_nan(0.0).alias(f"prop_{app}")
            for app in apps
        ]

        return agg, post_agg

    def _generate_behaviour_features(self) -> tuple[list,list]:
        agg: list = [
            pl.col(CHURN_TRIGGERED_COL).max(),
            pl.col(ACTIVITY_DATE_COL).filter(pl.col(ACTIVITY_DATE_COL).is_not_null()).n_unique().alias("n_sessions"),
            (pl.col(ACTIVITY_TYPE_COL) != "none").sum().alias("n_activities"),
            pl.col(ACTIVITY_TYPE_COL).filter(pl.col(ACTIVITY_TYPE_COL) != "none").n_unique().alias("n_unique_actions"),
            pl.col(ACTIVITY_DATE_COL).filter(pl.col(ACTIVITY_DATE_COL).is_not_null()).max().alias("last_activity_date")
        ]

        post_agg: list = [pl.col("last_activity_date").forward_fill().over(USER_ID_COL)]
        return agg, post_agg

    def apply_feature_engineering(self,
                                   df: pl.DataFrame,
                                   churn_adjusted_date_col_name: str | None = None,
                                   interval_in_days: int = INTERVAL_IN_DAYS,
                                   lookback_periods: tuple[int, ...] = LOOKBACK_PERIODS) -> pl.DataFrame | None:

        churn_adjusted_date_col_name = churn_adjusted_date_col_name or CHURN_ADJUSTED_DATE_COL

        try:
            group_by_columns = [USER_ID_COL, INTERVAL_START_COL]
            df = self._prepare_df(df, churn_adjusted_date_col_name)
            df_with_intervals = self._generate_intervals(df, churn_adjusted_date_col_name, interval_in_days)
            # print(df_with_intervals)
            # print(df_with_intervals.columns)
        

            agg, post_agg = self._generate_activity_feature_aggregation(lookback_periods,
                                                                        interval_in_days)
            # agg_b, post_agg_b = self._generate_behaviour_features()
            # agg_app, post_agg_app = self._generate_app_column_feature_aggregation()
            # agg_v, post_agg_v = self._generate_vehicle_make_feature_aggregation()

            # agg += agg_b + agg_app + agg_v
            # post_agg += post_agg_app + post_agg_v + [pl.col("user_id"), pl.col("interval_start")]

            d = df_with_intervals.group_by(group_by_columns).agg(agg).sort(group_by_columns).with_columns(post_agg)

            print(d.columns)

            # print(d.sort(["user_id","interval_start"]))
            # print(d)
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
dp.apply_feature_engineering(df_vh)

