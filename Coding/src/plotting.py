import matplotlib.pyplot as plot
import polars as pl
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.axes import Axes
from .constants.thesis_style import *
from .constants.columns import (
    USER_ID_COL, ACTIVITY_DATE_COL, CHURN_ADJUSTED_DATE_COL,
    CHURN_TRIGGERED_COL, VEHICLE_ID_COL, VEHICLE_MAKE_COL,
    VEHICLE_MODEL_COL, VEHICLE_START_YEAR_COL, VEHICLE_END_YEAR_COL,
    VEHICLE_MILEAGE_COL, APP_COL, ACTIVITY_TYPE_COL,
    STILL_IN_PRODUCTION_COL, INTERVAL_START_COL,
    YEAR_BIN_LOWER, YEAR_BIN_UPPER, YEAR_BIN_UPPER_REPLACEMENT
)

# ============================================================
#  Tuning defaults
# ============================================================
DEFAULT_CAR_SHARE_ABS = 4
BURST_TIME_HR = 1/30 # 2 Minutes

# ============================================================
#  Fixed (non-templated) generated column names
# ============================================================
ACTIVITY_DATE_COL_SHIFTED = ACTIVITY_DATE_COL + "_shifted"
ACTIVITY_GAP_COL_NAME = "activity_gap"
TOTAL_CARS_COL_NAME = "total_cars"
VEHICLE_COUNT_COL_NAME = "vehicle_count"
UNIQUE_VEHICLE_COUNT_COL_NAME = "unique_vehicle_count"
TOTAL_VEHICLE_COUNT_COL_NAME = "total_vehicle_count"
PROP_VEHICLE_COUNT_COL_NAME = "prop_vehicle"
EFFECTIVE_HHI_COL_NAME = "effective_hhi"
CAR_SHARE_CUMSUM_COL_NAME = "car_share"
RANK_COL_NAME = "rank"

# TODO: Implement strata plot of users in different years
# TODO: Rename plotting to plotting and summaries (as this class is suppose to be for generating summaries for checking conditions)
# TODO: Implement function that calculates intervals per user for different interval sizes with which to 
#       to summarize the data (14days,28days,56days)
# TODO: Implement a function that calculates seasonality
# TODO: Implement a function for activity type distribution
# TODO: Implement a function for effective_HHI and top_4_car share distributions 
class PlottingData:
    def __init__(self):
     
        self.personal_colour = PALETTE["Personal"]
        self.professional_colour = PALETTE["Professional"]
        self.seaborn_theme = SEABORN_THEME
        sns.set_theme(**SEABORN_THEME)

    def _cast_activity_date_col_to_datetime(self,
                                        df: pl.DataFrame,
                                        as_date: bool = True):
        expr = pl.col(ACTIVITY_DATE_COL).str.to_datetime(time_unit="us", time_zone="UTC")
        
        if as_date:
            expr = expr.dt.date()

        return df.with_columns(expr)
        
    def _transform_df_to_single_day_activity(self,
                                             df:pl.DataFrame):

        df = df.group_by(USER_ID_COL).agg(pl.col(ACTIVITY_DATE_COL).unique()).explode(ACTIVITY_DATE_COL)
        return df

    def _calculate_activity_gaps(self,
                                 df:pl.DataFrame):
        

        df = df.sort([USER_ID_COL,ACTIVITY_DATE_COL])

        df = df.with_columns(
                        pl.col(ACTIVITY_DATE_COL)
                        .shift(1)
                        .over(USER_ID_COL)
                        .alias(ACTIVITY_DATE_COL_SHIFTED))
        

        df = df.with_columns(
                        (pl.col(ACTIVITY_DATE_COL) - pl.col(ACTIVITY_DATE_COL_SHIFTED))
                        .alias(ACTIVITY_GAP_COL_NAME))  
        
        df = df.filter(pl.col(ACTIVITY_GAP_COL_NAME).is_not_null())
        
        # Converting the columns to actual numbers instead of duration type
        # Otherwise when converting this dataframe to pandas an error will occur
        df = df.select(pl.col(ACTIVITY_GAP_COL_NAME).dt.total_days()) 
        
        return df
    
    def _process_df_for_gap_distribution(self,df: pl.DataFrame):
        df = self._cast_activity_date_col_to_datetime(df)
        df = self._transform_df_to_single_day_activity(df)
        df = self._calculate_activity_gaps(df)
        print(df)
        return df
                
    def return_activity_gap_distribution(self,
                                         df: pl.DataFrame,
                                         ax: Axes | None = None,
                                         personal: bool = True,
                                         set_y_scale_to_log: bool = False, 
                                         quantiles: list[float] | None = None,
                                         quantile_boundary: float | None = None):
        

        try:

            activity_gaps = self._process_df_for_gap_distribution(df)
            pandas_data = activity_gaps.to_pandas()
            print(pandas_data)

            if ax is None:
                _, ax = plot.subplots()

            if quantile_boundary:
                boundary_value = activity_gaps[ACTIVITY_GAP_COL_NAME].quantile(quantile_boundary)
                ax.set_xlim(left=0,right=boundary_value)

            if set_y_scale_to_log:
                ax.set_yscale("log")
                ax.set_ylabel("Count (log scale)")
                scale_label = "log scale"

            else:
                ax.set_ylabel("Count")
                scale_label = "linear scale"

            if quantiles:
                for quantile in quantiles:
                    quantile_value = activity_gaps[ACTIVITY_GAP_COL_NAME].quantile(quantile)
                
                    ax.axvline(quantile_value,
                            color=COLORS["highlight"],
                            linestyle="--",
                            linewidth=1.0)
                    
                    ax.annotate(
                        f"{quantile * 100:.1f}%",
                        xy=(quantile_value, 0.95),
                        xycoords=("data", "axes fraction"),
                        xytext=(4, 0),
                        textcoords="offset points",
                        rotation=90,
                        va="top",
                        ha="left",
                        color=COLORS["highlight"]
                    )


            if personal:
                hist_color = self.personal_colour
            else:
                hist_color = self.professional_colour
        
            sns.histplot(data=pandas_data,x=ACTIVITY_GAP_COL_NAME, ax=ax,color=hist_color)
            sns.despine(ax=ax)
            ax.set_xlabel("Days since previous activity")
            ax.set_ylabel(scale_label)
            
            return ax

        except Exception as e:
            print(f"Unexpected error {e}")
            raise e

    def _filter_rows_from_burst(self,
                            df: pl.DataFrame,
                            burst_time_hr: float = BURST_TIME_HR):
        df = self._cast_activity_date_col_to_datetime(df, as_date=False)

        df = df.sort([USER_ID_COL, VEHICLE_ID_COL, ACTIVITY_DATE_COL])

        df = df.with_columns(
            pl.col(ACTIVITY_DATE_COL)
            .shift(1)
            .over([USER_ID_COL, VEHICLE_ID_COL])
            .alias(ACTIVITY_DATE_COL_SHIFTED)
        )

        df = df.with_columns(
            ((pl.col(ACTIVITY_DATE_COL) - pl.col(ACTIVITY_DATE_COL_SHIFTED))
            >= pl.duration(hours=burst_time_hr))
            .fill_null(True)  # first row per user-vehicle pair has no prior - keep it
            .alias("is_not_burst")
        )

        return df.filter(pl.col("is_not_burst")).drop([ACTIVITY_DATE_COL_SHIFTED, "is_not_burst"])
    
    def _compute_effective_hhi(self,
                               df: pl.DataFrame):
        
   
        vehicle_counts = df.group_by([USER_ID_COL,VEHICLE_ID_COL]).agg(pl.len().alias(VEHICLE_COUNT_COL_NAME))
        total_counts  = \
            vehicle_counts.group_by(USER_ID_COL)\
            .agg(pl.col(VEHICLE_COUNT_COL_NAME)\
            .sum()\
            .alias(TOTAL_VEHICLE_COUNT_COL_NAME))  

        vehicle_prop =  \
            vehicle_counts\
            .join(total_counts, on=USER_ID_COL)\
            .with_columns((pl.col(VEHICLE_COUNT_COL_NAME) /
                          pl.col(TOTAL_VEHICLE_COUNT_COL_NAME)).alias(PROP_VEHICLE_COUNT_COL_NAME))  

        effective_hhi_per_user = \
            vehicle_prop\
            .group_by(USER_ID_COL)\
            .agg(pl.col(PROP_VEHICLE_COUNT_COL_NAME).pow(2).sum().pow(-1).alias(EFFECTIVE_HHI_COL_NAME))

  
        return effective_hhi_per_user
    
    def _compute_percentage_car_share(self,
                                  df: pl.DataFrame,
                                  top_n: int = DEFAULT_CAR_SHARE_ABS):


        vehicle_counts = (
            df.group_by([USER_ID_COL, VEHICLE_ID_COL])
            .agg(pl.len().alias(VEHICLE_COUNT_COL_NAME))
        )

        total_counts = (
            vehicle_counts.group_by(USER_ID_COL)
            .agg(pl.col(VEHICLE_COUNT_COL_NAME).sum().alias(TOTAL_VEHICLE_COUNT_COL_NAME))
        )

        vehicle_props = (
            vehicle_counts
            .join(total_counts, on=USER_ID_COL)
            .with_columns((pl.col(VEHICLE_COUNT_COL_NAME) / pl.col(TOTAL_VEHICLE_COUNT_COL_NAME))
                        .alias(PROP_VEHICLE_COUNT_COL_NAME))
            .sort([USER_ID_COL, PROP_VEHICLE_COUNT_COL_NAME], descending=[False, True])
        )

        vehicle_props = vehicle_props.with_columns(
            pl.col(PROP_VEHICLE_COUNT_COL_NAME).cum_sum().over(USER_ID_COL).alias(CAR_SHARE_CUMSUM_COL_NAME),
            pl.col(VEHICLE_ID_COL).cum_count().over(USER_ID_COL).alias(RANK_COL_NAME)
        )

        # Users with fewer than top_n vehicles get their max cumsum (always 1.0)
        # rather than being dropped from the output
        top_n_share = (
            vehicle_props
            .group_by(USER_ID_COL)
            .agg(
                pl.when(pl.col(RANK_COL_NAME).max() < top_n)
                .then(pl.lit(1.0))
                .otherwise(pl.col(CAR_SHARE_CUMSUM_COL_NAME).filter(pl.col(RANK_COL_NAME) == top_n).first())
                .alias(CAR_SHARE_CUMSUM_COL_NAME)
            )
        )

        return top_n_share

    def _process_df_for_user_split(self,
                                   df:pl.DataFrame) -> pd.DataFrame:
        
        df = self._filter_rows_from_burst(df)
        
        sort_by = USER_ID_COL
        unique_vehicles_per_user = df.group_by(USER_ID_COL).agg(pl.col(VEHICLE_ID_COL).n_unique().alias(UNIQUE_VEHICLE_COUNT_COL_NAME)).sort(sort_by)
        effective_hhi_per_user = self._compute_effective_hhi(df).sort(sort_by)
        percentage_car_share = self._compute_percentage_car_share(df).sort(sort_by)

        joined_df =\
            unique_vehicles_per_user\
            .join(effective_hhi_per_user,on=USER_ID_COL)\
            .join(percentage_car_share,on=USER_ID_COL)

        pandas_df = joined_df.to_pandas().drop(columns=USER_ID_COL)
        
        return pandas_df

    def return_user_split_plot(self,
                               df:pl.DataFrame,
                               ax: Axes | None = None,
                               personal: bool = True):
        try:
            pandas_df = self._process_df_for_user_split(df)
            
            if not ax:
                _, ax = plot.subplots()

            if personal:
                plot_color = self.personal_colour
            else:
                plot_color = self.professional_colour
        
            sns.scatterplot(
                data=pandas_df,
                x=UNIQUE_VEHICLE_COUNT_COL_NAME,
                y=CAR_SHARE_CUMSUM_COL_NAME,
                size=EFFECTIVE_HHI_COL_NAME,
                color=plot_color,
                edgecolor=COLORS["text"],
                ax=ax, sizes=(5, 200), legend=True
                )

            return ax


        except Exception as e:
            print(f"Unexpected error {e}")
            raise e


data = pl.read_csv(r"C:\Users\Tomas\Desktop\Thesis Stuff\Survival_Analysis_Thesis\Coding\Data\interim\personal_users_raw.csv")



data_plotter = PlottingData()
data_plotter.return_user_split_plot(data)

plot.show()
