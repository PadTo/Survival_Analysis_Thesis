import matplotlib.pyplot as plot
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.axes import Axes
from .constants.thesis_style import *

# TODO: Implement strata plot of users in different years
# TODO: Implement KM curve based on vehicle make
class PlottingData:
    def __init__(self):
     
        self.personal_colour = PALETTE["Personal"]
        self.professional_colour = PALETTE["Professional"]
        self.seaborn_theme = SEABORN_THEME
        sns.set_theme(**SEABORN_THEME)



    def _convert_activity_date(self, df:pd.DataFrame):
        try:

            if "activity_date" not in df.columns:
                raise KeyError("activity_date")
            
            df["activity_date"] = pd.to_datetime(
                df["activity_date"],
                format="mixed")

            
        except KeyError:
            print("activity_date column doesn't exist in the dataframe.")
        except Exception as e:
            print(f"unexpected error occured {e}")


    def _calculate_activity_gaps(self,
                                 df:pd.DataFrame,
                                 burst_time_hr: int | float = 1):
        

        self._convert_activity_date(df)
        try:

            if "user_id" not in df.columns:
                raise KeyError("user_id")
            
            print(df.sort_values(by=["user_id","event_data"]))



            # df.sort_values(by=["user_id","activity_date"],inplace=True)
            # time_diff = (
            #     df["activity_date"] - df.groupby("user_id")["activity_date"].shift(1)
            #     ).fillna(pd.Timedelta(0))
            
            
            # time_diff_hr   = time_diff.dt.total_seconds().div(3600).round()
            # time_diff_mask = time_diff_hr >= burst_time_hr
            
            # time_diff_hr_filtered = time_diff_hr[time_diff_mask]
            # return time_diff_hr_filtered / 24
            

            
        except KeyError:
            print("user_id column doesn't exist in the dataframe.")

        except Exception as e:
            print(f"unexpected error occured {e}")

        

   

    def return_activity_gap_distribution(self,
                                         df: pd.DataFrame,
                                         ax: Axes | None = None,
                                         personal: bool = True,
                                         set_y_scale_to_log: bool = False, 
                                         quantiles: list[float] | None = None,
                                         quantile_boundary: float | None = None, 
                                         burst_time_hr: int | float = 1):
        
        activity_gaps = self._calculate_activity_gaps(df,burst_time_hr)
        activity_gaps = pd.Series(activity_gaps)

        if ax is None:
            _, ax = plot.subplots()

        if quantile_boundary:
            boundary_value = activity_gaps.quantile(quantile_boundary)
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
                quantile_value = activity_gaps.quantile(quantile)
            
                ax.axvline(quantile_value,
                          color=COLORS["highlight"],
                          linestyle="--",
                          linewidth="1")
                
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

        sns.histplot(x=activity_gaps, ax=ax,color=hist_color)
        sns.despine(ax=ax)
        ax.set_xlabel("Days since previous activity")
        ax.set_ylabel(scale_label)
        
        return ax


data = pd.read_csv(r"C:\Users\Tomas\Desktop\Thesis Stuff\Coding\Data\interim\personal_users_dataset.csv")



data_plotter = PlottingData()

# print(data_plotter.return_activity_gap_distribution(data,
#                                                     quantile_boundary=0.9999,
#                                                     set_y_scale_to_log=False,
#                                                     quantiles=[0.8,0.95,0.99,0.999]))

data_plotter._calculate_activity_gaps(data)

# plot.show()