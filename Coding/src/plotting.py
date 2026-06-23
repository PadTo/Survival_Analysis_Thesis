import matplotlib.pyplot as plot
import polars as pl
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.axes import Axes
import matplotlib as mpl
from .constants.thesis_plotting_style import PALETTE, COLORS, SEABORN_THEME, SCATTER_PLOT_SIZE_RANGE
from .constants.columns import (
    USER_ID_COL, ACTIVITY_DATE_COL, CHURN_ADJUSTED_DATE_COL,
    CHURN_TRIGGERED_COL, VEHICLE_ID_COL, VEHICLE_MAKE_COL,
    VEHICLE_MODEL_COL, VEHICLE_START_YEAR_COL, VEHICLE_END_YEAR_COL,
    VEHICLE_MILEAGE_COL, APP_COL, ACTIVITY_TYPE_COL,
    STILL_IN_PRODUCTION_COL, INTERVAL_START_COL,
    YEAR_BIN_LOWER, YEAR_BIN_UPPER, YEAR_BIN_UPPER_REPLACEMENT
)
from .constants.thesis_plotting_style import IN_FIGURE_TEXT_SIZE_MULTIPLIER
from .constants.cleaning import DEFAULT_CAR_SHARE_ABS, BURST_TIME_HR
from src.constants.processor import CHURN_TRIGGERED_SHIFTED_COLUMN_NAME, INTERVAL_END_COL

# ============================================================
#  Fixed (non-templated) generated column names
# ============================================================
ACTIVITY_DATE_COL_SHIFTED      = ACTIVITY_DATE_COL + "_shifted"
ACTIVITY_GAP_COL_NAME          = "activity_gap"
TOTAL_CARS_COL_NAME            = "total_cars"
VEHICLE_COUNT_COL_NAME         = "vehicle_count"
UNIQUE_VEHICLE_COUNT_COL_NAME  = "unique_vehicle_count"
TOTAL_VEHICLE_COUNT_COL_NAME   = "total_vehicle_count"
PROP_VEHICLE_COUNT_COL_NAME    = "prop_vehicle"
EFFECTIVE_HHI_COL_NAME         = "effective_hhi"
CAR_SHARE_CUMSUM_COL_NAME      = "car_share"
RANK_COL_NAME                  = "rank"
IS_NOT_BURST_COL_NAME          = "is_not_burst"

# ============================================================
#  Visual tuning (aesthetics only)
# ============================================================
BAR_GRADIENT_FADE     = 0.45   # how much the lowest-ranked bar fades vs the top one
BAR_EDGE_WIDTH        = 0.8
BAR_HEADROOM          = 1.24   # y-limit multiplier so value labels never clip
GRID_LINEWIDTH        = 0.6
GRID_ALPHA            = 0.35

# TODO: Implement strata plot of users in different years

class PlottingData:
    """
    Exploratory / diagnostic plots for the survival dataset.

    The methods fall into two groups by the pipeline stage they belong to:

    - Pre-segmentation, pre-churn-filtering (raw merged data): the gap
      distribution and seasonality, which describe the natural usage process and
      drive parameter choices (churn threshold, interval width, lookback). Any
      truncation would distort what they are meant to measure.
    - Post-segmentation (a single personal/professional group): the user-split
      scatter, used to confirm the split landed correctly.

    The composition plots (activity-type, vehicle-make) read sensibly at either
    stage: pooled for an overall picture, or per group to contrast the two.
    """

    def __init__(self):
        self.personal_colour  = PALETTE["Personal"]
        self.professional_colour = PALETTE["Professional"]
        self.seaborn_theme = SEABORN_THEME
        sns.set_theme(**SEABORN_THEME)

    # ------------------------------------------------------------
    #  Shared styling helpers (aesthetics only, no data logic)
    # ------------------------------------------------------------
    def _segment_colour(self, personal: bool | None) -> str:
        """Resolve the fill colour: personal, professional, or neutral when the group is unspecified."""
        if personal is None:
            return PALETTE["Neutral"]
        return self.personal_colour if personal else self.professional_colour

    def _style_categorical_axis(self,
                                ax: Axes,
                                x_label: str,
                                y_label: str,
                                max_value: float,
                                rotation: int = 35) -> None:
        """Apply the shared categorical-bar axis styling so every bar plot reads identically."""
        # Dropping the left spine too: with a horizontal grid the y-axis line is redundant
        sns.despine(ax=ax, left=True)
        ax.set_xlabel(x_label, labelpad=10)
        ax.set_ylabel(y_label, labelpad=10)
        ax.tick_params(axis="x", rotation=rotation, length=0)
        ax.tick_params(axis="y", length=0)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        # Horizontal-only grid keeps the focus on bar heights without vertical clutter
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", linestyle="--", linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max_value * BAR_HEADROOM)

    def _apply_bar_gradient(self, ax: Axes, base_colour: str) -> None:
        """Fade bar alpha down the rank so ordered bars read as a visual hierarchy."""
        patches = list(ax.patches)
        n = len(patches)
        for i, bar in enumerate(patches):
            bar.set_facecolor(base_colour)
            # Fading saturation down the rank gives ordered bars a sense of hierarchy
            bar.set_alpha(1.0 - BAR_GRADIENT_FADE * (i / max(n - 1, 1)))
            bar.set_edgecolor(COLORS["text"])
            bar.set_linewidth(BAR_EDGE_WIDTH)

    def _annotate_bar_values(self, ax: Axes, counts, total: float) -> None:
        """Label each bar with its share (emphasised) and raw count (secondary)."""
        base = mpl.rcParams["font.size"]
        pct_size = base * IN_FIGURE_TEXT_SIZE_MULTIPLIER * 1.1
        count_size = base * IN_FIGURE_TEXT_SIZE_MULTIPLIER * 0.9

        for bar, count in zip(ax.patches, counts):
            x = bar.get_x() + bar.get_width() / 2
            height = bar.get_height()

            # Percentage sits closest to the bar, emphasised
            ax.annotate(
                f"{count / total * 100:.1f}%",
                xy=(x, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom",
                fontweight="bold",
                color=COLORS["text"],
                fontsize=pct_size,
            )
            # Raw count above it, de-emphasised in the secondary colour
            ax.annotate(
                f"n={count:,.0f}",
                xy=(x, height),
                xytext=(0, 4 + pct_size + 3),
                textcoords="offset points",
                ha="center", va="bottom",
                color=COLORS["secondary_text"],
                fontsize=count_size,
            )

    # ------------------------------------------------------------
    #  Activity-gap distribution
    # ------------------------------------------------------------
    def _cast_activity_date_col_to_datetime(self,
                                            df: pl.DataFrame,
                                            as_date: bool = True):
        """Parse the activity-date string; as_date drops the time when only the calendar day matters."""
        expr = pl.col(ACTIVITY_DATE_COL).str.to_datetime(time_unit="us", time_zone="UTC")

        if as_date:
            expr = expr.dt.date()

        return df.with_columns(expr)

    def _transform_df_to_single_day_activity(self, df: pl.DataFrame):
        """Collapse to one row per distinct active day per user, so repeated same-day use isn't counted as multiple gaps."""
        df = df.group_by(USER_ID_COL).agg(pl.col(ACTIVITY_DATE_COL).unique()).explode(ACTIVITY_DATE_COL)
        return df

    def _calculate_activity_gaps(self, df: pl.DataFrame):
        """Measure consecutive day-to-day gaps per user, returned as whole days."""
        df = df.sort([USER_ID_COL, ACTIVITY_DATE_COL])

        df = df.with_columns(
            pl.col(ACTIVITY_DATE_COL)
            .shift(1)
            .over(USER_ID_COL)
            .alias(ACTIVITY_DATE_COL_SHIFTED)
        )

        df = df.with_columns(
            (pl.col(ACTIVITY_DATE_COL) - pl.col(ACTIVITY_DATE_COL_SHIFTED))
            .alias(ACTIVITY_GAP_COL_NAME)
        )

        df = df.filter(pl.col(ACTIVITY_GAP_COL_NAME).is_not_null())

        # Converting the columns to actual numbers instead of duration type
        # Otherwise when converting this dataframe to pandas an error will occur
        df = df.select(pl.col(ACTIVITY_GAP_COL_NAME).dt.total_days())

        return df

    def _process_df_for_gap_distribution(self, df: pl.DataFrame):
        """Full gap-distribution pipeline: parse dates, dedupe to active days, measure gaps."""
        df = self._cast_activity_date_col_to_datetime(df)
        df = self._transform_df_to_single_day_activity(df)
        df = self._calculate_activity_gaps(df)
        return df

    def return_activity_gap_distribution(self,
                                         df: pl.DataFrame,
                                         ax: Axes | None = None,
                                         personal: bool = True,
                                         set_y_scale_to_log: bool = False,
                                         quantiles: list[float] | None = None,
                                         quantile_boundary: float | None = None):
        """
        Plot the distribution of days between consecutive active days per user.

        When to use: run on the raw merged dataset *before* any inactivity
        truncation or cutoff-date filtering. Those steps cut each user's history
        at their first churn, removing the long right tail of gaps and
        understating the true spread. The unfiltered distribution is exactly what
        justifies the churn threshold and the interval / lookback sizes, so it has
        to be read off data that still contains every observed gap.

        quantiles draws reference lines (e.g. median, 95th percentile);
        quantile_boundary trims the x-axis so the long tail doesn't flatten the
        visible mass; set_y_scale_to_log helps when the count drops off sharply.
        """
        try:
            activity_gaps = self._process_df_for_gap_distribution(df)
            pandas_data = activity_gaps.to_pandas()

            if ax is None:
                _, ax = plot.subplots()

            if quantile_boundary:
                boundary_value = activity_gaps[ACTIVITY_GAP_COL_NAME].quantile(quantile_boundary)
                ax.set_xlim(left=0, right=boundary_value)

            if set_y_scale_to_log:
                ax.set_yscale("log")
                scale_label = "log scale"
            else:
                scale_label = "linear scale"

            hist_color = self._segment_colour(personal)

            if quantile_boundary:
                boundary_value = activity_gaps[ACTIVITY_GAP_COL_NAME].quantile(quantile_boundary)
                # Clip the data itself rather than just the axis so histplot bins within the visible range
                pandas_data = pandas_data[pandas_data[ACTIVITY_GAP_COL_NAME] <= boundary_value]

            sns.histplot(
                data=pandas_data,
                x=ACTIVITY_GAP_COL_NAME,
                ax=ax,
                color=hist_color,
                edgecolor=COLORS["background"],
                linewidth=0.4,
                alpha=0.9,
                kde=True,
                line_kws={"linewidth": 1.6},
                bins=60,   # fixed count, computed over the clipped range
            )

            # Recolour the KDE curve so it reads as an accent against the bars
            if ax.lines:
                ax.lines[-1].set_color(COLORS["text"])

            base = mpl.rcParams["font.size"]
            if quantiles:
                for quantile in quantiles:
                    quantile_value = activity_gaps[ACTIVITY_GAP_COL_NAME].quantile(quantile)

                    ax.axvline(quantile_value,
                               color=COLORS["highlight"],
                               linestyle="--",
                               linewidth=1.1)

                    ax.annotate(
                        f"{quantile * 100:.0f}%  ({quantile_value:.0f}d)",
                        xy=(quantile_value, 0.97),
                        xycoords=("data", "axes fraction"),
                        xytext=(5, 0),
                        textcoords="offset points",
                        rotation=90,
                        va="top",
                        ha="left",
                        color=COLORS["highlight"],
                        fontsize=base * IN_FIGURE_TEXT_SIZE_MULTIPLIER,
                        fontweight="bold",
                    )

            sns.despine(ax=ax, left=True)
            ax.grid(axis="x", visible=False)
            ax.grid(axis="y", linestyle="--", linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", length=0)
            ax.set_xlabel("Days since previous activity", labelpad=10)
            ax.set_ylabel(f"Count ({scale_label})", labelpad=10)

            return ax

        except Exception as e:
            print(f"Unexpected error {e}")
            raise e

    # ------------------------------------------------------------
    #  User-split scatter
    # ------------------------------------------------------------
    def _filter_rows_from_burst(self,
                                df: pl.DataFrame,
                                burst_time_hr: float = BURST_TIME_HR):
        """Drop rapid re-scans of the same vehicle (within burst_time_hr): one session, not distinct activity."""
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
            .alias(IS_NOT_BURST_COL_NAME)
        )

        return df.filter(pl.col(IS_NOT_BURST_COL_NAME)).drop([ACTIVITY_DATE_COL_SHIFTED, IS_NOT_BURST_COL_NAME])

    def _compute_effective_hhi(self, df: pl.DataFrame):
        """Per-user inverse Herfindahl over vehicle usage shares: how many vehicles a user effectively spreads activity across."""
        vehicle_counts = df.group_by([USER_ID_COL, VEHICLE_ID_COL]).agg(pl.len().alias(VEHICLE_COUNT_COL_NAME))

        total_counts = (
            vehicle_counts.group_by(USER_ID_COL)
            .agg(pl.col(VEHICLE_COUNT_COL_NAME).sum().alias(TOTAL_VEHICLE_COUNT_COL_NAME))
        )

        vehicle_prop = (
            vehicle_counts
            .join(total_counts, on=USER_ID_COL)
            .with_columns((pl.col(VEHICLE_COUNT_COL_NAME) / pl.col(TOTAL_VEHICLE_COUNT_COL_NAME)).alias(PROP_VEHICLE_COUNT_COL_NAME))
        )

        return (
            vehicle_prop
            .group_by(USER_ID_COL)
            .agg(pl.col(PROP_VEHICLE_COUNT_COL_NAME).pow(2).sum().pow(-1).alias(EFFECTIVE_HHI_COL_NAME))
        )

    def _compute_percentage_car_share(self,
                                      df: pl.DataFrame,
                                      top_n: int = DEFAULT_CAR_SHARE_ABS):
        """Per-user cumulative share of the top_n most-used vehicles; users with fewer than top_n vehicles score 1.0 instead of dropping out."""
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
        return (
            vehicle_props
            .group_by(USER_ID_COL)
            .agg(
                pl.when(pl.col(RANK_COL_NAME).max() < top_n)
                .then(pl.lit(1.0))
                .otherwise(pl.col(CAR_SHARE_CUMSUM_COL_NAME).filter(pl.col(RANK_COL_NAME) == top_n).first())
                .alias(CAR_SHARE_CUMSUM_COL_NAME)
            )
        )

    def _process_df_for_user_split(self, df: pl.DataFrame) -> pd.DataFrame:
        """Assemble the three split diagnostics (unique count, HHI, top-n share) per user, on burst-filtered data to match how the split was computed."""
        df = self._filter_rows_from_burst(df)

        sort_by = USER_ID_COL
        unique_vehicles_per_user = df.group_by(USER_ID_COL).agg(pl.col(VEHICLE_ID_COL).n_unique().alias(UNIQUE_VEHICLE_COUNT_COL_NAME)).sort(sort_by)
        effective_hhi_per_user   = self._compute_effective_hhi(df).sort(sort_by)
        percentage_car_share     = self._compute_percentage_car_share(df).sort(sort_by)

        joined_df = (
            unique_vehicles_per_user
            .join(effective_hhi_per_user, on=USER_ID_COL)
            .join(percentage_car_share,   on=USER_ID_COL)
        )

        return joined_df.to_pandas().drop(columns=USER_ID_COL)

    def return_user_split_plot(self,
                               df: pl.DataFrame,
                               ax: Axes | None = None,
                               personal: bool = True):
        """
        Scatter each user by unique-vehicle count vs top-4 vehicle share, sized by
        effective HHI.

        When to use: run on a dataset that has *already* been segmented into a
        single group (e.g. the personal-users file) to confirm the split landed
        where intended. Personal users should cluster at low vehicle counts with
        high concentration (top-4 share near 1, low HHI); professionals spread out
        toward many vehicles and lower concentration. The diagnostics are computed
        on burst-filtered data here, so the plot reflects the same vehicle counts
        the classifier itself saw rather than counts inflated by rapid re-scans.
        """
        try:
            pandas_df = self._process_df_for_user_split(df)

            if not ax:
                _, ax = plot.subplots()

            plot_color = self._segment_colour(personal)

            sns.scatterplot(
                data=pandas_df,
                x=UNIQUE_VEHICLE_COUNT_COL_NAME,
                y=CAR_SHARE_CUMSUM_COL_NAME,
                size=EFFECTIVE_HHI_COL_NAME,
                color=plot_color,
                edgecolor=COLORS["text"],
                linewidth=0.4,
                alpha=0.75,
                ax=ax,
                sizes=SCATTER_PLOT_SIZE_RANGE,
                legend="brief",
            )

            sns.despine(ax=ax)
            ax.grid(linestyle="--", linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)
            ax.set_axisbelow(True)
            ax.set_xlabel("Unique vehicles", labelpad=10)
            ax.set_ylabel("Top-4 vehicle share", labelpad=10)

            # Give the auto-generated size legend a readable title
            legend = ax.get_legend()
            if legend is not None:
                legend.set_title("Effective HHI")

            return ax

        except Exception as e:
            print(f"Unexpected error {e}")
            raise e

    # ------------------------------------------------------------
    #  Activity-type distribution
    # ------------------------------------------------------------
    def _process_df_for_activity_distribution(self, df: pl.DataFrame) -> pd.DataFrame:
        """Count activities per type, ranked descending."""
        return (
            df.group_by(ACTIVITY_TYPE_COL)
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .to_pandas()
        )

    def return_activity_distribution_plot(self,
                                          df: pl.DataFrame,
                                          ax: Axes | None = None,
                                          personal: bool = True):
        """
        Ranked bar chart of activity counts per activity type.

        When to use: works at either stage. On the pooled dataset it shows the
        overall activity mix; run separately on each segment to contrast how
        personal and professional users spend their actions. Counts are at the
        raw-activity level (no burst filtering) because the composition of what
        users do is the quantity of interest, not the count of distinct sessions.
        """
        try:
            pandas_df = self._process_df_for_activity_distribution(df)
            total = pandas_df["count"].sum()

            if ax is None:
                _, ax = plot.subplots()

            plot_color = self._segment_colour(personal)

            sns.barplot(data=pandas_df, x=ACTIVITY_TYPE_COL, y="count", ax=ax, color=plot_color)

            self._apply_bar_gradient(ax, plot_color)
            self._annotate_bar_values(ax, pandas_df["count"], total)
            self._style_categorical_axis(ax, "Activity type", "Number of activities", pandas_df["count"].max())

            return ax

        except Exception as e:
            print(f"Unexpected error {e}")
            raise e

    # ------------------------------------------------------------
    #  Seasonality
    # ------------------------------------------------------------
    def _process_df_for_seasonality(self, df: pl.DataFrame) -> pd.DataFrame:
        """Burst-filter, then extract the calendar month of each activity."""
        df = self._filter_rows_from_burst(df)

        df = df.with_columns(
            pl.col(ACTIVITY_DATE_COL).dt.month().alias("month_num"),
            pl.col(ACTIVITY_DATE_COL).dt.strftime("%B").alias("month")
        )

        return df.select(["month", "month_num"]).to_pandas()

    def return_seasonality_plot(self,
                                df: pl.DataFrame,
                                ax: Axes | None = None,
                                personal: bool = True):
        """
        Activity counts by calendar month, with the peak month highlighted.

        When to use: best on data *before* inactivity truncation, since cutting
        each user at their churn point removes their later months and biases the
        seasonal shape toward whenever cohorts happened to join. Burst filtering is
        applied first so a single session of repeated scans counts once toward its
        month rather than inflating it. Compare segments by running it on each
        group separately.
        """
        try:
            pandas_df = self._process_df_for_seasonality(df)

            if ax is None:
                _, ax = plot.subplots()

            plot_color = self._segment_colour(personal)

            month_order = [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ]

            sns.countplot(data=pandas_df, x="month", ax=ax, order=month_order, color=plot_color)

            # Flat colour for all months, then highlight the peak in the accent colour
            heights = [bar.get_height() for bar in ax.patches]
            peak_index = int(np.argmax(heights)) if heights else None
            for i, bar in enumerate(ax.patches):
                bar.set_facecolor(COLORS["highlight"] if i == peak_index else plot_color)
                bar.set_edgecolor(COLORS["text"])
                bar.set_linewidth(BAR_EDGE_WIDTH)

            self._style_categorical_axis(ax, "Month", "Activity count",
                                         max(heights) if heights else 1, rotation=45)

            return ax

        except Exception as e:
            print(f"Unexpected error {e}")
            raise e

    # ------------------------------------------------------------
    #  Vehicle-make distribution
    # ------------------------------------------------------------
    def _return_vehicle_make_summary(self, df: pl.DataFrame) -> pd.DataFrame:
        """
        Summarises vehicle make distribution across the dataset.

        Counts are at the vehicle level (deduplicated by vehicle_id) rather than
        activity level, so a user scanning the same car 100 times doesn't inflate
        that make's share.
        """
        return (
            df.unique(subset=[VEHICLE_ID_COL])
            .group_by(VEHICLE_MAKE_COL)
            .agg(pl.len().alias("count"))
            .with_columns(
                (pl.col("count") / pl.col("count").sum() * 100)
                .round(2)
                .alias("pct")
            )
            .sort("count", descending=True)
            .to_pandas()
        )

    def return_vehicle_make_distribution_plot(self,
                                              df: pl.DataFrame,
                                              ax: Axes | None = None,
                                              personal: bool | None = None):
        """
        Ranked bar chart of vehicle-make composition, counted at the vehicle level.

        When to use: either stage. On the pooled set it shows the overall make
        mix; per segment it contrasts the portfolios (professionals often skew
        toward a wider or different make spread). Because counts deduplicate by
        vehicle_id, a user scanning one car hundreds of times contributes a single
        vehicle, so burst filtering is unnecessary here. This is the view that
        justifies the make grouping (brand_group_map) and the per-make proportion
        features. Leaving personal unset uses the neutral colour, which suits a
        pooled (un-split) chart.
        """
        try:
            pandas_df = self._return_vehicle_make_summary(df)
            total = pandas_df["count"].sum()

            if ax is None:
                _, ax = plot.subplots()


            plot_color = self._segment_colour(personal)


            sns.barplot(data=pandas_df, x=VEHICLE_MAKE_COL, y="count", ax=ax, color=plot_color)

            self._apply_bar_gradient(ax, plot_color)
            self._annotate_bar_values(ax, pandas_df["count"], total)
            self._style_categorical_axis(ax, "Vehicle make", "Number of vehicles", pandas_df["count"].max())

            return ax

        except Exception as e:
            print(f"Unexpected error {e}")
            raise e

    # ------------------------------------------------------------
    #  Tie count distribution
    # ------------------------------------------------------------
    def return_interval_ties_summary(self,
                                    df: pl.DataFrame,
                                    ax: Axes | None = None,
                                    personal: bool | None = None):
        try:
            df = df.filter(pl.col(CHURN_TRIGGERED_SHIFTED_COLUMN_NAME))
            df = df.group_by([INTERVAL_START_COL, INTERVAL_END_COL]).agg(
                pl.col(USER_ID_COL).count().alias("ties")
            ).sort([INTERVAL_START_COL, INTERVAL_END_COL])
            df = df.filter(pl.col("ties") > 1)

            # Count how many intervals have each tie size
            tie_counts = (df
                .group_by("ties")
                .agg(pl.len().alias("count"))
                .sort("ties")
                
            )

            pandas_df = tie_counts.to_pandas()
            if ax is None:
                _, ax = plot.subplots()

            plot_color = self._segment_colour(personal)

            sns.barplot(data=tie_counts, x="ties", y="count", ax=ax, color=plot_color, edgecolor=COLORS["text"], linewidth=0.8)
            
           
            self._annotate_bar_values(ax,pandas_df["count"],pandas_df["count"].sum())
            self._style_categorical_axis(ax, "Number of people in a tie", "Frequency of such ties ", pandas_df["count"].max())
   

            return ax

        except Exception as e:
            print(f"Unexpected error {e}")
            raise e

# from src.constants import paths_to_files_and_folders as const
# test_features_personal = const.PATH_TO_INTERIM_DATA / "personal_users_filtered_full_feature.csv"
# personal_test = pl.read_csv(test_features_personal)

# dp = PlottingData()

# dp.return_interval_ties_summary(personal_test,
#                                 personal=True)
# plot.tight_layout()
# plot.show()