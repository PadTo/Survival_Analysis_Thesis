import math
import pandas as pd
import polars as pl
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from .constants.columns import (
    USER_ID_COL, ACTIVITY_DATE_COL,
    VEHICLE_ID_COL, VEHICLE_MAKE_COL,
    VEHICLE_MILEAGE_COL, VEHICLE_START_YEAR_COL,
    YEAR_BIN_LOWER, YEAR_BIN_UPPER, YEAR_BIN_UPPER_REPLACEMENT
)
from .constants.processor import N_NEIGHBOURS


# ============================================================
#  Fixed (non-templated) generated column names
# ============================================================
FIRST_YEAR_COL: str = "first_year"


class DataSplitter():

    def __init__(self, df: pd.DataFrame | pl.DataFrame) -> None:
        self.df = df.copy(deep=True) if type(df) == pd.DataFrame else df


    def _create_order_mileage_buckets(self,
                                     df:pl.DataFrame):
        ordered_buckets = (
            df
            .select(pl.col(VEHICLE_MILEAGE_COL).unique())
            .with_columns(
                pl.col(VEHICLE_MILEAGE_COL)
                .str.replace_all(r"[ >]", "")        # strip spaces and '>'
                .str.split("-")
                .list.first()                         # lower edge
                .cast(pl.Int64)
                .alias("_lower")
            )
            .sort("_lower")
            .get_column(VEHICLE_MILEAGE_COL)
            .to_list()
        )

        return ordered_buckets

    def KNN_impute_vehicle_start_year(self,
                                      df: pl.DataFrame,
                                      KNN_imputer_fitted: KNNImputer | None = None,
                                      one_hot_enc_fitted: OneHotEncoder | None = None,
                                      ordinal_enc_fitted: OrdinalEncoder | None = None,
                                      n_neighbours: int = N_NEIGHBOURS) -> tuple[pl.DataFrame, KNNImputer,OneHotEncoder,OrdinalEncoder]:

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
        
        if KNN_imputer_fitted is None:
            KNN_imputer = KNNImputer(n_neighbors=n_neighbours)
        else:
            KNN_imputer = KNN_imputer_fitted

        if one_hot_enc_fitted is None:
            one_hot_enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        else:
            one_hot_enc = one_hot_enc_fitted

        if ordinal_enc_fitted is None:
            ordered_buckets = self._create_order_mileage_buckets(df)
            ordinal_enc = OrdinalEncoder(categories=[ordered_buckets],
                                         handle_unknown="use_encoded_value",
                                         unknown_value=-1,)
        else:
            ordinal_enc = ordinal_enc_fitted

        # Deduplicating to one row per vehicle so KNN fits on distinct vehicles only
        vehicle_df = (df
            .select([VEHICLE_ID_COL, VEHICLE_MAKE_COL,
                     VEHICLE_MILEAGE_COL, VEHICLE_START_YEAR_COL])
            .unique(subset=[VEHICLE_ID_COL])
            .to_pandas())

        if one_hot_enc_fitted is None:
            vehicle_make_one_hot = one_hot_enc.fit_transform(vehicle_df[[VEHICLE_MAKE_COL]])
        else:
            vehicle_make_one_hot = one_hot_enc.transform(vehicle_df[[VEHICLE_MAKE_COL]])

        if ordinal_enc_fitted is None:
            vehicle_mileage_label = ordinal_enc.fit_transform(vehicle_df[[VEHICLE_MILEAGE_COL]])
        else:
            vehicle_mileage_label = ordinal_enc.transform(vehicle_df[[VEHICLE_MILEAGE_COL]])

        imputed_df = pd.DataFrame(
            vehicle_make_one_hot,
            columns=one_hot_enc.get_feature_names_out(),
            index=vehicle_df.index
        )

        imputed_df["vehicle_mileage_cat"] = vehicle_mileage_label[:, 0]
        # Start year placed last so the imputed target is recoverable as the final column
        imputed_df[VEHICLE_START_YEAR_COL] = vehicle_df[VEHICLE_START_YEAR_COL].values

        if KNN_imputer_fitted is not None:
            X_imputed = KNN_imputer.transform(imputed_df)
        else:
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

        return df, KNN_imputer, one_hot_enc, ordinal_enc

    def split_train_val_test(self,
                             df: pl.DataFrame,
                             random_state: int = 42,
                             train_size: float = 0.9,
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
        val_size_value = val_size if val_size is not None else 0

        if not math.isclose(train_size + test_size + val_size_value, 1.0):
            raise ValueError(f"Splits must sum to 1, got {train_size + test_size + val_size_value}")

        first_year_col: str = FIRST_YEAR_COL

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

    def prepare_dataset(    self,
                            df: pl.DataFrame,
                            random_state: int = 42,
                            train_size : float = 0.8,
                            test_size: float = 0.1,
                            val_size: float | None = None,
                            personal: bool = True,
                            save_path: Path | None = None) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame] | tuple[pl.DataFrame, pl.DataFrame]:
        """
        Steps:
            1. Splits the dataset into train, val, and test if val_size is set, otherwise train and test
            2. Imputes missing vehicle start_years using KNN imputer via ordinal and one hot encoders on mileage and make columns, respectively
            3. (OPTIONAL) Save the dataset into the specificed location

        Args:
            df:            Input Polars DataFrame with user activity logs.
            random_state:  Random seed for reproducibility. Default 42.
            test_size:     Proportion of users allocated to the test set. Default 0.1.
            val_size:      Proportion of users allocated to validation. If None, no
                           validation set is produced. Default None.

        Returns:
            (train_data_imputed, val_data_imputed, test_data_imputed) if val_size is set, else (train_data_imputed, test_data_imputed).
        """

        
        if save_path:
            save_path.mkdir(parents=True,exist_ok=True)
            user_group = "personal" if personal else "professional"
            train_data_path = save_path / ("training_data_" + user_group + ".csv")
            validation_data_path = save_path / ("validation_data_" + user_group + ".csv")
            testing_data_path = save_path / ("testing_data_" + user_group + ".csv")
 

        if val_size:

            train_df, val_df, test_df = self.split_train_val_test(df,
                                                                random_state = random_state,
                                                                train_size=train_size,
                                                                test_size=test_size,
                                                                val_size=val_size)

            train_df_imputed, KNN_imputer, one_hot_enc, ordinal_enc = self.KNN_impute_vehicle_start_year(train_df)
            val_df_imputed  , _, _, _   = self.KNN_impute_vehicle_start_year( val_df, KNN_imputer, one_hot_enc, ordinal_enc)
            test_df_imputed , _, _, _   = self.KNN_impute_vehicle_start_year(test_df, KNN_imputer, one_hot_enc, ordinal_enc)



            if save_path:
                train_df_imputed.write_csv(train_data_path)
                val_df_imputed.write_csv(validation_data_path)
                test_df_imputed.write_csv(testing_data_path)

            return train_df_imputed, val_df_imputed, test_df_imputed
        
        else:
            train_df, test_df = self.split_train_val_test(df,
                                                        random_state = random_state,
                                                        train_size=train_size,
                                                        test_size=test_size)

            train_df_imputed, KNN_imputer, one_hot_enc, ordinal_enc = self.KNN_impute_vehicle_start_year(train_df)
            test_df_imputed , _, _, _   = self.KNN_impute_vehicle_start_year(test_df, KNN_imputer, one_hot_enc, ordinal_enc)

            if save_path:
                train_df_imputed.write_csv(train_data_path)
                test_df_imputed.write_csv(testing_data_path)

            return train_df_imputed, test_df_imputed