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


class ModelTraining():

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
        ) # type: ignore

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

    def train_models(self):
        
        pass
    


# import src.constants.paths_to_files_and_folders as paths

# path_to_personal = paths.PATH_TO_INTERIM_DATA / "personal_users_filtered.csv"
# data = pl.read_csv(path_to_personal)


# data_splitter = DataSplitter(data)

# data_splitter.prepare_dataset(data,
#                               train_size=0.8,
#                               test_size=0.1,
#                               val_size=0.1,
#                               personal=True,
#                               save_path=Path(paths.PATH_TO_INTERIM_DATA))