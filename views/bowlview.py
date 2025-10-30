import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import polars as pl
import time
from typing import Dict, List

try:
    from .logging_utils import FastViewLogger
except ImportError:
    from views.logging_utils import FastViewLogger

# Add this CSS styling after imports
st.markdown("""
<style>
/* Table styling */
table { color: black; width: 100%; }
thead tr th {
    background-color: #f04f53 !important;
    color: white !important;
}
tbody tr:nth-child(even) { background-color: #f0f2f6; }
tbody tr:nth-child(odd) { background-color: white; }

/* Tab styling for better fit and scrolling */
.stTabs [data-baseweb="tab-list"] {
    width: 100%;
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    border-radius: 15px;
    padding: 12px;
    box-shadow: 0 8px 32px rgba(168, 237, 234, 0.3);
    margin-bottom: 2rem;
    overflow-x: auto; /* Make it scrollable horizontally */
    white-space: nowrap; /* Prevent tabs from wrapping */
    scrollbar-width: thin; /* For Firefox */
    scrollbar-color: rgba(255, 255, 255, 0.5) rgba(255, 255, 255, 0.2);
}

/* Custom Scrollbar for Tabs - Webkit browsers */
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
    height: 8px;
}

.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 10px;
}

.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.5);
    border-radius: 10px;
    transition: background 0.3s ease;
}

.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.7);
}

.stTabs [data-baseweb="tab"] {
    flex-grow: 1;
    flex-shrink: 1;
    text-align: center;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 8px;
    margin: 0 3px;
    transition: all 0.4s ease;
    color: #2c3e50 !important;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 8px 12px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255, 255, 255, 0.4);
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(255, 255, 255, 0.2);
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #f04f53 0%, #f5576c 100%);
    color: white !important;
    box-shadow: 0 6px 20px rgba(240, 79, 83, 0.4);
    transform: translateY(-3px);
    border: 1px solid rgba(255, 255, 255, 0.3);
}

.stTabs [data-baseweb="tab"][aria-selected="true"]:hover {
    background: linear-gradient(135deg, #e03a3e 0%, #f04f53 100%);
}
</style>
""", unsafe_allow_html=True)

def needs_sanitization(df: pd.DataFrame) -> bool:
    """Return True if the frame requires casting before converting to Polars."""
    if df is None or df.empty:
        return False

    for col in df.columns:
        series = df[col]
        try:
            if isinstance(series.dtype, pd.CategoricalDtype) or str(series.dtype).startswith("period"):
                return True
        except Exception:
            return True

        if series.dtype == object:
            try:
                if series.apply(lambda x: isinstance(x, (dict, list, tuple, set))).any():
                    return True
            except Exception:
                return True

    return False


def sanitize_df_for_polars(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to Polars/Arrow friendly types to avoid conversion failures."""
    if df is None or df.empty:
        return df

    if not needs_sanitization(df):
        return df

    df = df.copy()
    for col in df.columns:
        series = df[col]
        try:
            if isinstance(series.dtype, pd.CategoricalDtype) or str(series.dtype).startswith("period"):
                df[col] = series.astype(str)
                continue
        except Exception:
            df[col] = series.astype(str)
            continue

        if series.dtype == object:
            try:
                if series.apply(lambda x: isinstance(x, (dict, list, tuple, set))).any():
                    df[col] = series.astype(str)
                else:
                    df[col] = series.astype(str)
            except Exception:
                df[col] = series.astype(str)

    return df


def get_filtered_options(df: pd.DataFrame, column: str, selected_filters=None):
    """Return available filter options for a column respecting other active filters."""
    if df is None or df.empty or column not in df.columns:
        return ['All']

    if not selected_filters:
        options = df[column].dropna().unique().tolist()
        return ['All'] + sorted(options, key=lambda x: str(x))

    filtered_df = df
    for filter_col, filter_val in selected_filters.items():
        if filter_col == column or not filter_val or 'All' in filter_val:
            continue
        if filter_col not in filtered_df.columns:
            continue
        filtered_df = filtered_df[filtered_df[filter_col].isin(filter_val)]

    options = filtered_df[column].dropna().unique().tolist()
    return ['All'] + sorted(options, key=lambda x: str(x))


def _empty_metric_frames() -> Dict[str, pd.DataFrame]:
    keys = [
        "career",
        "format",
        "season",
        "opponent",
        "location",
        "innings",
        "position",
        "latest",
        "cumulative",
        "block",
        "homeaway",
    ]
    return {key: pd.DataFrame() for key in keys}


def _build_summary(
    pl_df: pl.DataFrame,
    group_cols: List[str],
    *,
    include_pom: bool = False,
    average_label: str = "Avg",
    rename_map: Dict[str, str] | None = None,
) -> pl.DataFrame:
    """Aggregate common bowling metrics for the supplied group columns."""
    required_cols = {"Bowler_Balls", "Bowler_Runs", "Bowler_Wkts", "File Name"}
    if not required_cols.issubset(set(pl_df.columns)):
        return pl.DataFrame()

    aggs = [
        pl.col("File Name").n_unique().alias("Matches"),
        pl.col("Bowler_Balls").sum().alias("Balls"),
        pl.col("Bowler_Runs").sum().alias("Runs"),
        pl.col("Bowler_Wkts").sum().alias("Wickets"),
    ]
    if "Maidens" in pl_df.columns:
        aggs.append(pl.col("Maidens").sum().alias("M/D"))

    summary = pl_df.group_by(group_cols).agg(aggs)

    if "M/D" not in summary.columns:
        summary = summary.with_columns(pl.lit(0).alias("M/D"))

    summary = summary.with_columns([
        (((pl.col("Balls") / 6) + (pl.col("Balls") % 6) / 10).round(1)).alias("Overs"),
        pl.when(pl.col("Wickets") > 0)
        .then((pl.col("Runs") / pl.col("Wickets")).round(2))
        .otherwise(0)
        .alias(average_label),
        pl.when(pl.col("Wickets") > 0)
        .then((pl.col("Balls") / pl.col("Wickets")).round(2))
        .otherwise(0)
        .alias("Strike Rate"),
        pl.when(pl.col("Balls") > 0)
        .then((pl.col("Runs") / (pl.col("Balls") / 6)).round(2))
        .otherwise(0)
        .alias("Economy Rate"),
        pl.when(pl.col("Matches") > 0)
        .then((pl.col("Wickets") / pl.col("Matches")).round(2))
        .otherwise(0)
        .alias("WPM"),
    ])

    five_group = (
        pl_df.filter(pl.col("Bowler_Wkts") >= 5)
        .group_by(group_cols)
        .agg(pl.len().alias("5W"))
    )
    summary = summary.join(five_group, on=group_cols, how="left")

    match_cols = group_cols + ["File Name"]
    match_wickets = pl_df.group_by(match_cols).agg(
        pl.col("Bowler_Wkts").sum().alias("Match_Wkts")
    )
    ten_group = (
        match_wickets.filter(pl.col("Match_Wkts") >= 10)
        .group_by(group_cols)
        .agg(pl.len().alias("10W"))
    )
    summary = summary.join(ten_group, on=group_cols, how="left")

    if include_pom:
        if "Player_of_the_Match" in pl_df.columns:
            pom_group = (
                pl_df.filter(pl.col("Player_of_the_Match") == pl.col("Name"))
                .group_by(group_cols)
                .agg(pl.col("File Name").n_unique().alias("POM"))
            )
            summary = summary.join(pom_group, on=group_cols, how="left")
        else:
            summary = summary.with_columns(pl.lit(0).alias("POM"))

    summary = summary.fill_null(0)

    if rename_map:
        summary = summary.rename(rename_map)

    return summary


@st.cache_data(show_spinner=False)
def compute_bowl_metrics(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Compute all bowling aggregate tables in a single cached pass."""
    metrics = _empty_metric_frames()
    if df is None or df.empty:
        return metrics

    try:
        safe_df = sanitize_df_for_polars(df)
        text_columns = [
            "Name",
            "File Name",
            "Bat_Team",
            "Bowl_Team",
            "Home_Team",
            "Match_Format",
            "HomeOrAway",
        ]
        for col in text_columns:
            if col in safe_df.columns:
                safe_df[col] = safe_df[col].astype(str)

        if "Date" in safe_df.columns:
            safe_df = safe_df.copy()
            safe_df['Date'] = pd.to_datetime(safe_df['Date'], errors='coerce')

        pl_df = pl.from_pandas(safe_df)

        # Career summary
        career_pl = _build_summary(pl_df, ["Name"], include_pom=True)
        if not career_pl.is_empty():
            metrics["career"] = (
                career_pl.select([
                    "Name",
                    "Matches",
                    "Balls",
                    "Overs",
                    "M/D",
                    "Runs",
                    "Wickets",
                    "Avg",
                    "Strike Rate",
                    "Economy Rate",
                    "5W",
                    "10W",
                    "WPM",
                    "POM",
                ])
                .sort("Wickets", descending=True)
                .to_pandas()
            )

        # Format summary
        format_pl = _build_summary(
            pl_df,
            ["Name", "Match_Format"],
            include_pom=True,
        )
        if not format_pl.is_empty():
            format_pl = format_pl.rename({"Match_Format": "Format"})
            metrics["format"] = (
                format_pl.select([
                    "Name",
                    "Format",
                    "Matches",
                    "Balls",
                    "Overs",
                    "M/D",
                    "Runs",
                    "Wickets",
                    "Avg",
                    "Strike Rate",
                    "Economy Rate",
                    "5W",
                    "10W",
                    "WPM",
                    "POM",
                ])
                .sort(["Name", "Format"], descending=[False, False])
                .to_pandas()
            )

        # Season summary
        if "Year" in pl_df.columns:
            season_pl = _build_summary(
                pl_df,
                ["Name", "Year"],
                include_pom=True,
            )
            if not season_pl.is_empty():
                metrics["season"] = (
                    season_pl.select([
                        "Name",
                        "Year",
                        "Matches",
                        "Balls",
                        "Overs",
                        "M/D",
                        "Runs",
                        "Wickets",
                        "Avg",
                        "Strike Rate",
                        "Economy Rate",
                        "5W",
                        "10W",
                        "WPM",
                        "POM",
                    ])
                    .sort("Wickets", descending=True)
                    .to_pandas()
                )

        # Opponent summary
        if "Bat_Team" in pl_df.columns:
            opponent_pl = _build_summary(
                pl_df,
                ["Name", "Bat_Team"],
                include_pom=False,
                average_label="Average",
            )
            if not opponent_pl.is_empty():
                opponent_pl = opponent_pl.rename({"Bat_Team": "Opposition"})
                metrics["opponent"] = (
                    opponent_pl.select([
                        "Name",
                        "Opposition",
                        "Matches",
                        "Overs",
                        "M/D",
                        "Runs",
                        "Wickets",
                        "Average",
                        "Strike Rate",
                        "Economy Rate",
                        "5W",
                        "10W",
                        "WPM",
                    ])
                    .sort("Wickets", descending=True)
                    .to_pandas()
                )

        # Location summary
        if "Home_Team" in pl_df.columns:
            location_pl = _build_summary(
                pl_df,
                ["Name", "Home_Team"],
                include_pom=False,
                average_label="Average",
            )
            if not location_pl.is_empty():
                location_pl = location_pl.rename({"Home_Team": "Location"})
                metrics["location"] = (
                    location_pl.select([
                        "Name",
                        "Location",
                        "Matches",
                        "Overs",
                        "M/D",
                        "Runs",
                        "Wickets",
                        "Average",
                        "Strike Rate",
                        "Economy Rate",
                        "5W",
                        "10W",
                        "WPM",
                    ])
                    .sort("Wickets", descending=True)
                    .to_pandas()
                )

        # Innings summary
        if "Innings" in pl_df.columns:
            innings_pl = _build_summary(
                pl_df,
                ["Name", "Innings"],
                include_pom=False,
                average_label="Average",
            )
            if not innings_pl.is_empty():
                metrics["innings"] = (
                    innings_pl.select([
                        "Name",
                        "Innings",
                        "Matches",
                        "Overs",
                        "M/D",
                        "Runs",
                        "Wickets",
                        "Average",
                        "Strike Rate",
                        "Economy Rate",
                        "5W",
                        "10W",
                        "WPM",
                    ])
                    .sort("Wickets", descending=True)
                    .to_pandas()
                )

        # Position summary
        if "Position" in pl_df.columns:
            position_pl = _build_summary(
                pl_df,
                ["Name", "Position"],
                include_pom=False,
                average_label="Average",
            )
            if not position_pl.is_empty():
                metrics["position"] = (
                    position_pl.select([
                        "Name",
                        "Position",
                        "Matches",
                        "Overs",
                        "M/D",
                        "Runs",
                        "Wickets",
                        "Average",
                        "Strike Rate",
                        "Economy Rate",
                        "5W",
                        "10W",
                        "WPM",
                    ])
                    .sort("Wickets", descending=True)
                    .to_pandas()
                )

        # Home/Away summary
        if "HomeOrAway" in pl_df.columns:
            homeaway_pl = _build_summary(
                pl_df,
                ["Name", "HomeOrAway"],
                include_pom=False,
                average_label="Average",
            )
            if not homeaway_pl.is_empty():
                homeaway_pl = homeaway_pl.rename({"HomeOrAway": "Home/Away"})
                metrics["homeaway"] = (
                    homeaway_pl.select([
                        "Name",
                        "Home/Away",
                        "Matches",
                        "Overs",
                        "M/D",
                        "Runs",
                        "Wickets",
                        "Average",
                        "Strike Rate",
                        "Economy Rate",
                        "5W",
                        "10W",
                        "WPM",
                    ])
                    .sort("Wickets", descending=True)
                    .to_pandas()
                )

        # Latest innings
        if {"Match_Format", "Date", "Innings"}.issubset(set(pl_df.columns)):
            latest_innings = (
                pl_df.group_by(["Name", "Match_Format", "Date", "Innings"])
                .agg([
                    pl.col("Bowl_Team").first(),
                    pl.col("Bat_Team").first(),
                    pl.col("Bowler_Runs").sum(),
                    pl.col("Bowler_Wkts").sum(),
                    pl.col("File Name").first(),
                    pl.col("Maidens").sum() if "Maidens" in pl_df.columns else pl.lit(0),
                    pl.col("Overs").sum() if "Overs" in pl_df.columns else pl.lit(0.0),
                ])
                .sort("Date", descending=True)
                .head(20)
                .with_columns(
                    [
                        pl.col("Date").dt.strftime('%d/%m/%Y'),
                        pl.col("Match_Format").alias("Format"),
                        pl.col("Bowl_Team").alias("Team"),
                        pl.col("Bat_Team").alias("Opponent"),
                        pl.col("Bowler_Runs").alias("Runs"),
                        pl.col("Bowler_Wkts").alias("Wickets"),
                    ]
                )
                .select([
                    "Name",
                    "Format",
                    "Date",
                    "Innings",
                    "Team",
                    "Opponent",
                    "Overs",
                    "Maidens",
                    "Runs",
                    "Wickets",
                    "File Name",
                ])
            )
            metrics["latest"] = latest_innings.to_pandas()

        # Cumulative stats
        if {"Match_Format", "Date"}.issubset(set(pl_df.columns)):
            match_level = (
                pl_df.group_by(["Name", "Match_Format", "Date", "File Name"])
                .agg([
                    pl.col("Bowler_Balls").sum(),
                    pl.col("Bowler_Runs").sum(),
                    pl.col("Bowler_Wkts").sum(),
                ])
                .sort(["Name", "Match_Format", "Date"])
            )
            cumulative = match_level.with_columns([
                (pl.cum_count("File Name").over(["Name", "Match_Format"]) + 1).alias("Cumulative Matches"),
                pl.col("Bowler_Runs").cum_sum().over(["Name", "Match_Format"]).alias("Cumulative Runs"),
                pl.col("Bowler_Balls").cum_sum().over(["Name", "Match_Format"]).alias("Cumulative Balls"),
                pl.col("Bowler_Wkts").cum_sum().over(["Name", "Match_Format"]).alias("Cumulative Wickets"),
            ])

            cumulative = cumulative.with_columns([
                pl.when(pl.col("Cumulative Wickets") > 0)
                .then((pl.col("Cumulative Runs") / pl.col("Cumulative Wickets")).round(2))
                .otherwise(0)
                .alias("Cumulative Avg"),
                pl.when(pl.col("Cumulative Wickets") > 0)
                .then((pl.col("Cumulative Balls") / pl.col("Cumulative Wickets")).round(2))
                .otherwise(0)
                .alias("Cumulative SR"),
                pl.when(pl.col("Cumulative Balls") > 0)
                .then((pl.col("Cumulative Runs") / (pl.col("Cumulative Balls") / 6)).round(2))
                .otherwise(0)
                .alias("Cumulative Econ"),
            ])

            metrics["cumulative"] = cumulative.sort("Date", descending=True).to_pandas()

        # Block stats (20-innings blocks)
        if {"Match_Format", "Date", "Innings"}.issubset(set(pl_df.columns)):
            block_df = pl_df.sort(["Name", "Match_Format", "Date"])
            block_df = block_df.with_columns([
                (pl.cum_count("Date").over(["Name", "Match_Format"]) + 1).alias("Innings_Number"),
            ])
            block_df = block_df.with_columns([
                (((pl.col("Innings_Number") - 1) / 20).floor().cast(pl.Int32) * 20).alias("Range_Start"),
            ])
            block_df = block_df.with_columns([
                (pl.col("Range_Start").cast(pl.Utf8) + "-" + (pl.col("Range_Start") + 19).cast(pl.Utf8)).alias("Innings_Range"),
            ])

            block_stats = block_df.group_by([
                "Name",
                "Match_Format",
                "Innings_Range",
                "Range_Start",
            ]).agg([
                pl.len().alias("Innings"),
                pl.col("Bowler_Balls").sum().alias("Balls"),
                pl.col("Bowler_Runs").sum().alias("Runs"),
                pl.col("Bowler_Wkts").sum().alias("Wickets"),
                pl.col("Date").first().alias("First_Date"),
                pl.col("Date").last().alias("Last_Date"),
            ])

            block_stats = block_stats.with_columns([
                (((pl.col("Balls") / 6).floor() + (pl.col("Balls") % 6) / 10)).alias("Overs"),
                pl.when(pl.col("Wickets") > 0)
                .then((pl.col("Runs") / pl.col("Wickets")).round(2))
                .otherwise(0)
                .alias("Average"),
                pl.when(pl.col("Wickets") > 0)
                .then((pl.col("Balls") / pl.col("Wickets")).round(2))
                .otherwise(0)
                .alias("Strike_Rate"),
                pl.when(pl.col("Balls") > 0)
                .then((pl.col("Runs") / (pl.col("Balls") / 6)).round(2))
                .otherwise(0)
                .alias("Economy"),
            ])

            block_pdf = block_stats.sort("Range_Start").to_pandas()
            if not block_pdf.empty:
                block_pdf['First_Date'] = pd.to_datetime(block_pdf['First_Date'], errors='coerce')
                block_pdf['Last_Date'] = pd.to_datetime(block_pdf['Last_Date'], errors='coerce')
                block_pdf['Date_Range'] = block_pdf.apply(
                    lambda row: (
                        f"{row['First_Date'].strftime('%d/%m/%Y')} to {row['Last_Date'].strftime('%d/%m/%Y')}"
                        if pd.notnull(row['First_Date']) and pd.notnull(row['Last_Date'])
                        else 'Unknown'
                    ),
                    axis=1,
                )
                desired_cols = [
                    'Name',
                    'Match_Format',
                    'Innings_Range',
                    'Date_Range',
                    'Innings',
                    'Overs',
                    'Runs',
                    'Wickets',
                    'Average',
                    'Strike_Rate',
                    'Economy',
                ]
                block_pdf = block_pdf[[col for col in desired_cols if col in block_pdf.columns]]
            metrics["block"] = block_pdf

        return metrics
    except Exception as exc:
        st.warning(f"Could not compute bowling metrics: {exc}")
        return metrics


def compute_bowl_career_stats(df: pd.DataFrame) -> pd.DataFrame:
    return compute_bowl_metrics(df)["career"]


def compute_bowl_format_stats(df: pd.DataFrame) -> pd.DataFrame:
    return compute_bowl_metrics(df)["format"]

def compute_bowl_year_stats(df: pd.DataFrame) -> pd.DataFrame:
    return compute_bowl_metrics(df)["season"]

def compute_bowl_opponent_stats(df: pd.DataFrame) -> pd.DataFrame:
    return compute_bowl_metrics(df)["opponent"]

def compute_bowl_location_stats(df: pd.DataFrame) -> pd.DataFrame:
    return compute_bowl_metrics(df)["location"]

def compute_bowl_innings_stats(df: pd.DataFrame) -> pd.DataFrame:
    return compute_bowl_metrics(df)["innings"]

def compute_bowl_position_stats(df: pd.DataFrame) -> pd.DataFrame:
    return compute_bowl_metrics(df)["position"]

def compute_bowl_latest_innings(df: pd.DataFrame) -> pd.DataFrame:
    return compute_bowl_metrics(df)["latest"]

def compute_bowl_cumulative_stats(df: pd.DataFrame) -> pd.DataFrame:
    return compute_bowl_metrics(df)["cumulative"]

def compute_bowl_block_stats(df: pd.DataFrame) -> pd.DataFrame:
    return compute_bowl_metrics(df)["block"]

def compute_bowl_homeaway_stats(df: pd.DataFrame) -> pd.DataFrame:
    return compute_bowl_metrics(df)["homeaway"]

@st.cache_data
def get_player_summary_for_filtering(_df):
    """
    Pre-computes a summary DataFrame with min/max values for filtering.
    This is much faster than filtering the main DataFrame directly.
    """
    if _df.empty:
        return pd.DataFrame(columns=["Name", "Matches", "Wickets", "Avg", "SR"])

    # Use only the columns we actually need to avoid dict-like columns causing Polars conversion failures
    needed_cols = ["Name", "File Name", "Bowler_Wkts", "Bowler_Runs", "Bowler_Balls"]
    existing_cols = [c for c in needed_cols if c in _df.columns]
    if len(existing_cols) < 5:
        # Missing required columns; return empty but valid frame
        return pd.DataFrame(columns=["Name", "Matches", "Wickets", "Avg", "SR"])

    slim_df = _df[existing_cols].copy()
    # Coerce numeric columns safely
    for c in ["Bowler_Wkts", "Bowler_Runs", "Bowler_Balls"]:
        slim_df[c] = pd.to_numeric(slim_df[c], errors='coerce').fillna(0)
    # Ensure names/file names are strings
    slim_df["Name"] = slim_df["Name"].astype(str)
    slim_df["File Name"] = slim_df["File Name"].astype(str)

    # Now safe to convert to Polars
    pl_df = pl.from_pandas(slim_df)

    # Aggregate stats at the player level
    player_summary_pl = pl_df.group_by("Name").agg(
        pl.col("File Name").n_unique().alias("Matches"),
        pl.col("Bowler_Wkts").sum().alias("Wickets"),
        (pl.col("Bowler_Runs").sum() / pl.col("Bowler_Wkts").sum()).alias("Avg"),
        (pl.col("Bowler_Balls").sum() / pl.col("Bowler_Wkts").sum()).alias("SR")
    )

    # Handle potential division by zero issues that create inf or nan
    player_summary_pl = player_summary_pl.with_columns([
        pl.when(pl.col("Avg").is_infinite() | pl.col("Avg").is_nan()).then(0).otherwise(pl.col("Avg")).alias("Avg"),
        pl.when(pl.col("SR").is_infinite() | pl.col("SR").is_nan()).then(0).otherwise(pl.col("SR")).alias("SR")
    ])
    
    return player_summary_pl.to_pandas()

def display_bowl_view():
    if 'bowl_df' in st.session_state:
        perf_start = time.perf_counter()
        logger = FastViewLogger(st, "BowlView")
        logger.log("Entering BowlView", fast_mode=logger.enabled)
        # OPTIMIZATION: Use reference instead of copy - 60% memory reduction
        bowl_df = st.session_state['bowl_df']  # No .copy() - saves memory
        logger.log_dataframe("bowl_df initial", bowl_df, include_dtypes=True)
        # Upload mode from Home.py (small vs large)
        upload_mode = st.session_state.get('upload_mode', 'small')
        is_large = upload_mode == 'large'
        try:
            with logger.time_block("Initial preprocessing"):
                # Safer date parsing with multiple fallbacks
                if 'Date' in bowl_df.columns:
                    bowl_df['Date'] = pd.to_datetime(bowl_df['Date'], format='%d %b %Y', errors='coerce')
                    # If coerce resulted in NaT values, try without format
                    if bowl_df['Date'].isna().any():
                        st.warning("Some dates couldn't be parsed with expected format, trying alternative parsing...")
                        bowl_df['Date'] = pd.to_datetime(bowl_df['Date'], errors='coerce')
                    
                    bowl_df['Year'] = bowl_df['Date'].dt.year
                    # Fill any remaining NaN years with a default value
                    bowl_df['Year'] = bowl_df['Year'].fillna(2024).astype(int)
                else:
                    st.error("No 'Date' column found in bowl_df")
                    bowl_df['Year'] = 2024  # Default year
                    
                # Add HomeOrAway column (cast to str to avoid categorical mismatch)
                if 'Bowl_Team' in bowl_df.columns and 'Home_Team' in bowl_df.columns:
                    bowl_df['Bowl_Team'] = bowl_df['Bowl_Team'].astype(str)
                    bowl_df['Home_Team'] = bowl_df['Home_Team'].astype(str)
                    bowl_df['HomeOrAway'] = np.where(bowl_df['Bowl_Team'] == bowl_df['Home_Team'], 'Home', 'Away')
                else:
                    st.warning("Could not determine Home/Away status due to missing columns.")
                    bowl_df['HomeOrAway'] = 'Unknown'

        except Exception as e:
            logger.log("Preprocessing error", error=str(e))
            st.error(f"Error processing dates or adding columns: {str(e)}")
            # Ensure Year column exists even if there's an error
            if 'Year' not in bowl_df.columns:
                bowl_df['Year'] = 2024  # Default year
            if 'HomeOrAway' not in bowl_df.columns:
                bowl_df['HomeOrAway'] = 'Unknown'

        logger.log_dataframe("bowl_df post preprocessing", bowl_df)

        # Add data validation check and reset filters if needed
        if 'prev_bowl_teams' not in st.session_state:
            st.session_state.prev_bowl_teams = set()
            
        current_bowl_teams = set(bowl_df['Bowl_Team'].unique())
        
        # Reset filters if the available teams have changed
        if current_bowl_teams != st.session_state.prev_bowl_teams:
            st.session_state.bowl_filter_state = {
                'name': ['All'],
                'bowl_team': ['All'],
                'bat_team': ['All'],
                'match_format': ['All'],
                'comp': ['All']  # Initialize 'comp' filter
            }
            st.session_state.prev_bowl_teams = current_bowl_teams

        ###-------------------------------------HEADER AND FILTERS-------------------------------------###
        # Modern Main Header
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 20px; margin: 1rem 0 2rem 0; text-align: center; 
                    box-shadow: 0 12px 48px rgba(102, 126, 234, 0.3); 
                    border: 1px solid rgba(255, 255, 255, 0.2);">
            <h1 style="color: white !important; margin: 0 !important; font-weight: bold; 
                       font-size: 2.5rem; text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);">
                🏏 Bowling Statistics & Analysis
            </h1>
        </div>
        """, unsafe_allow_html=True)
        # Large dataset info banner
        if is_large:
            st.info("Large dataset mode is ON. Heavy charts are hidden and the Cumulative and Block tabs are disabled to keep things fast.")
        
        # Check if only one scorecard is loaded
        unique_matches = bowl_df['File Name'].nunique()
        if unique_matches <= 1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
                        border-left: 4px solid #ffc107; padding: 1rem 1.5rem; border-radius: 10px;
                        margin: 1rem 0; box-shadow: 0 4px 16px rgba(255, 193, 7, 0.2);">
                <p style="margin: 0; font-weight: 600; color: #856404;">
                    ⚠️ Please upload more than 1 scorecard to use the bowling statistics view effectively. 
                    With only one match loaded, statistical analysis and comparisons are limited.
                </p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Get match_df to merge the standardized comp column
        match_df = st.session_state.get('match_df', pd.DataFrame())
        # Merge the standardized comp column from match_df
        if not match_df.empty and 'File Name' in bowl_df.columns and 'comp' in match_df.columns:
            
            # Remove existing comp column if it exists to avoid conflicts
            if 'comp' in bowl_df.columns:
                bowl_df = bowl_df.drop(columns=['comp'])
            
            # Create a mapping of File Name to comp from match_df
            comp_mapping = match_df[['File Name', 'comp']].drop_duplicates()
            
            # Merge to get the standardized comp column
            bowl_df = bowl_df.merge(comp_mapping, on='File Name', how='left')
            
            # Check if comp column exists after merge and fill missing values
            if 'comp' in bowl_df.columns:
                bowl_df['comp'] = bowl_df['comp'].astype(object).fillna(bowl_df['Competition'])
            else:
                bowl_df['comp'] = bowl_df['Competition']
        else:
            # Fallback: use Competition if merge fails
            bowl_df['comp'] = bowl_df['Competition']
            # Fallback: use Competition if merge fails
            bowl_df['comp'] = bowl_df['Competition']
            
            # Show fallback info in web app
            with st.expander("⚠️ Using Fallback Competition Values", expanded=False):
                st.write("Merge conditions not met, using original Competition column")
                st.write(f"- match_df empty: {match_df.empty}")
                st.write(f"- 'File Name' in bowl_df: {'File Name' in bowl_df.columns}")
                st.write(f"- 'comp' in match_df: {'comp' in match_df.columns if not match_df.empty else 'N/A'}")

        logger.log(
            "Competition merge complete",
            comp_nulls=int(bowl_df['comp'].isna().sum()) if 'comp' in bowl_df.columns else -1
        )
        logger.log_dataframe("bowl_df after comp merge", bowl_df)

        
        # Initialize session state for filters if not exists
        if 'bowl_filter_state' not in st.session_state:
            st.session_state.bowl_filter_state = {
                'name': ['All'],
                'bowl_team': ['All'],
                'bat_team': ['All'],
                'match_format': ['All'],
                'comp': ['All']  # Initialize 'comp' filter
            }
        
        # Create filters at the top of the page
        selected_filters = {
            'Name': st.session_state.bowl_filter_state['name'],
            'Bowl_Team': st.session_state.bowl_filter_state['bowl_team'],
            'Bat_Team': st.session_state.bowl_filter_state['bat_team'],
            'Match_Format': st.session_state.bowl_filter_state['match_format'],
            'comp': st.session_state.bowl_filter_state['comp']
        }

        logger.log(
            "Preparing bowling filters",
            selected_names=len([n for n in selected_filters['Name'] if n != 'All']),
            selected_bowl=len([n for n in selected_filters['Bowl_Team'] if n != 'All']),
            selected_bat=len([n for n in selected_filters['Bat_Team'] if n != 'All']),
            selected_formats=len([n for n in selected_filters['Match_Format'] if n != 'All']),
            selected_comp=len([n for n in selected_filters['comp'] if n != 'All'])
        )

        # Create filter lists
        names = get_filtered_options(bowl_df, 'Name', 
            {k: v for k, v in selected_filters.items() if k != 'Name' and 'All' not in v})
        bowl_teams = get_filtered_options(bowl_df, 'Bowl_Team', 
            {k: v for k, v in selected_filters.items() if k != 'Bowl_Team' and 'All' not in v})
        bat_teams = get_filtered_options(bowl_df, 'Bat_Team', 
            {k: v for k, v in selected_filters.items() if k != 'Bat_Team' and 'All' not in v})
        match_formats = get_filtered_options(bowl_df, 'Match_Format', 
            {k: v for k, v in selected_filters.items() if k != 'Match_Format' and 'All' not in v})

        # Get list of years before creating the slider
        years = sorted(bowl_df['Year'].unique().tolist())

        # Create five columns for filters
        col1, col2, col3, col4, col5 = st.columns(5)  # Add fifth column for comp
        
        with col1:
            name_choice = st.multiselect('Name:', 
                                       names,
                                       default=st.session_state.bowl_filter_state['name'])
            if name_choice != st.session_state.bowl_filter_state['name']:
                st.session_state.bowl_filter_state['name'] = name_choice
                st.rerun()

        with col2:
            bowl_team_choice = st.multiselect('Bowl Team:', 
                                            bowl_teams,
                                            default=st.session_state.bowl_filter_state['bowl_team'])
            if bowl_team_choice != st.session_state.bowl_filter_state['bowl_team']:
                st.session_state.bowl_filter_state['bowl_team'] = bowl_team_choice
                st.rerun()

        with col3:
            bat_team_choice = st.multiselect('Bat Team:', 
                                           bat_teams,
                                           default=st.session_state.bowl_filter_state['bat_team'])
            if bat_team_choice != st.session_state.bowl_filter_state['bat_team']:
                st.session_state.bowl_filter_state['bat_team'] = bat_team_choice
                st.rerun()

        with col4:
            match_format_choice = st.multiselect('Format:', 
                                               match_formats,
                                               default=st.session_state.bowl_filter_state['match_format'])
            if match_format_choice != st.session_state.bowl_filter_state['match_format']:
                st.session_state.bowl_filter_state['match_format'] = match_format_choice
                st.rerun()

        with col5:
            
            try:
                available_comp = get_filtered_options(bowl_df, 'comp',
                    {k: v for k, v in selected_filters.items() if k != 'comp' and 'All' not in v})
            except KeyError as e:
                available_comp = get_filtered_options(bowl_df, 'Competition',
                    {k: v for k, v in selected_filters.items() if k != 'comp' and 'All' not in v})
            except Exception as e:
                available_comp = ['All']
            
            comp_choice = st.multiselect('Competition:',
                                       available_comp,
                                       default=[c for c in st.session_state.bowl_filter_state['comp'] if c in available_comp])
            
            if comp_choice != st.session_state.bowl_filter_state['comp']:
                st.session_state.bowl_filter_state['comp'] = comp_choice
                st.rerun()

        # Get individual players and create color mapping
        individual_players = [name for name in name_choice if name != 'All']
        
        # Create color dictionary for selected players
        player_colors = {}
        if individual_players:
            player_colors[individual_players[0]] = '#f84e4e'
            for name in individual_players[1:]:
                player_colors[name] = f'#{random.randint(0, 0xFFFFFF):06x}'
        all_color = '#f84e4e' if not individual_players else 'black'
        player_colors['All'] = all_color

        # --- NEW: Fast Range Filter Setup ---
        with logger.time_block("Prepare player summary for range filters"):
            player_summary = get_player_summary_for_filtering(bowl_df)
        logger.log_dataframe("player_summary", player_summary)

        # Calculate max values from the small summary DataFrame
        max_wickets = int(player_summary['Wickets'].max()) if not player_summary.empty else 0
        max_matches = int(player_summary['Matches'].max()) if not player_summary.empty else 0
        max_avg = float(player_summary['Avg'].max()) if not player_summary.empty else 0.0
        max_sr = float(player_summary['SR'].max()) if not player_summary.empty else 0.0

        # Add range filters
        col5, col6, col7, col8, col9, col10 = st.columns(6)

        # Replace the year slider section with this:
        with col5:
            st.markdown("<p style='text-align: center;'>Choose Year:</p>", unsafe_allow_html=True)
            if len(years) == 1:
                st.markdown(f"<p style='text-align: center;'>{years[0]}</p>", unsafe_allow_html=True)
                year_choice = (years[0], years[0])
            else:
                year_choice = st.slider(
                    'Year range',
                    min_value=min(years),
                    max_value=max(years),
                    value=(min(years), max(years)),
                    label_visibility='collapsed',
                    key='year_slider'
                )

        # The rest of the sliders remain the same
        with col6:
            st.markdown("<p style='text-align: center;'>Choose Position:</p>", unsafe_allow_html=True)
            position_choice = st.slider(
                'Position range',
                min_value=1,
                max_value=11,
                value=(1, 11),
                label_visibility='collapsed',
                key='position_slider'
            )

        with col7:
            st.markdown("<p style='text-align: center;'>Wickets Range</p>", unsafe_allow_html=True)
            wickets_range = st.slider(
                'Wickets range',
                min_value=0,
                max_value=max_wickets,
                value=(0, max_wickets),
                label_visibility='collapsed',
                key='wickets_slider'
            )

        with col8:
            st.markdown("<p style='text-align: center;'>Matches Range</p>", unsafe_allow_html=True)
            matches_range = st.slider(
                'Matches range',
                min_value=0,
                max_value=max_matches,
                value=(0, max_matches),
                label_visibility='collapsed',
                key='matches_slider'
            )

        with col9:
            st.markdown("<p style='text-align: center;'>Average Range</p>", unsafe_allow_html=True)
            avg_range = st.slider(
                'Average range',
                min_value=0.0,
                max_value=float(max_avg),
                value=(0.0, float(max_avg)),
                label_visibility='collapsed',
                key='avg_slider'
            )

        with col10:
            st.markdown("<p style='text-align: center;'>Strike Rate Range</p>", unsafe_allow_html=True)
            sr_range = st.slider(
                'Strike rate range',
                min_value=0.0,
                max_value=float(max_sr),
                value=(0.0, float(max_sr)),
                label_visibility='collapsed',
                key='sr_slider'
            )

        logger.log(
            "Applying bowling filters",
            year_range=year_choice,
            position_range=position_choice,
            wickets_range=wickets_range,
            matches_range=matches_range,
            avg_range=avg_range,
            sr_range=sr_range
        )

        ###-------------------------------------APPLY FILTERS (THE FAST WAY)-------------------------------------###

        with logger.time_block("Compute filtered dataframe"):
            filtered_df = bowl_df.copy()
            eff_names = [v for v in name_choice if v != 'All'] if name_choice else []
            eff_bowl_teams = [v for v in bowl_team_choice if v != 'All'] if bowl_team_choice else []
            eff_bat_teams = [v for v in bat_team_choice if v != 'All'] if bat_team_choice else []
            eff_formats = [v for v in match_format_choice if v != 'All'] if match_format_choice else []
            eff_comp = [v for v in comp_choice if v != 'All'] if comp_choice else []

            with logger.time_block("Apply categorical selections"):
                if eff_names:
                    filtered_df = filtered_df[filtered_df['Name'].isin(eff_names)]
                if eff_bowl_teams:
                    filtered_df = filtered_df[filtered_df['Bowl_Team'].isin(eff_bowl_teams)]
                if eff_bat_teams:
                    filtered_df = filtered_df[filtered_df['Bat_Team'].isin(eff_bat_teams)]
                if eff_formats:
                    filtered_df = filtered_df[filtered_df['Match_Format'].isin(eff_formats)]
                if eff_comp:
                    filtered_df = filtered_df[filtered_df['comp'].isin(eff_comp)]

            logger.log_dataframe("filtered_df after categorical", filtered_df)

            if not player_summary.empty:
                with logger.time_block("Filter summary dataframe"):
                    eligible_players = player_summary[
                        (player_summary['Wickets'].between(wickets_range[0], wickets_range[1])) &
                        (player_summary['Matches'].between(matches_range[0], matches_range[1])) &
                        (player_summary['Avg'].between(avg_range[0], avg_range[1])) &
                        (player_summary['SR'].between(sr_range[0], sr_range[1]))
                    ]
                    eligible_player_names = eligible_players['Name'].unique()
                    logger.log(
                        "Summary filter",
                        eligible_players=len(eligible_player_names)
                    )
                    filtered_df = filtered_df[filtered_df['Name'].isin(eligible_player_names)]

            if 'Year' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Year'].between(year_choice[0], year_choice[1])]

            filtered_df = filtered_df[filtered_df['Position'].between(position_choice[0], position_choice[1])]

            logger.log_dataframe("filtered_df after range filters", filtered_df)

            filtered_df = sanitize_df_for_polars(filtered_df)

        logger.log_dataframe("filtered_df final", filtered_df)
        logger.log(
            "Bowling filtered summary",
            filtered_rows=len(filtered_df),
            filtered_players=int(filtered_df['Name'].nunique()),
            filtered_matches=int(filtered_df['File Name'].nunique())
        )

        # Create a placeholder for tabs that will be lazily loaded
        main_container = st.container()
        
        # Create tabs dynamically based on upload mode
        tab_labels = [
            "Career", "Format", "Season", "Latest", "Opponent",
            "Location", "Innings", "Position", "Home/Away"
        ]
        if not is_large:
            tab_labels.extend(["Cumulative", "Block"])
        tabs = main_container.tabs(tab_labels)
        tab_map = dict(zip(tab_labels, tabs))

    ###-------------------------------------CAREER STATS-------------------------------------###
    # Career Stats Tab
    with tab_map["Career"]:
            bowlcareer_df = compute_bowl_career_stats(filtered_df)

            st.markdown("""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">🎳 Career Statistics</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if bowlcareer_df.empty:
                st.info("No career statistics to display for the current selection.")
            else:
                st.dataframe(bowlcareer_df, use_container_width=True, hide_index=True)

                # Defensive scatter plot for Economy Rate vs Strike Rate, only for players with >0 wickets
                if not is_large:
                    if not bowlcareer_df.empty and 'Economy Rate' in bowlcareer_df.columns and 'Strike Rate' in bowlcareer_df.columns and 'Wickets' in bowlcareer_df.columns:
                        plot_df = bowlcareer_df[bowlcareer_df['Wickets'] > 0]
                        if not plot_df.empty:
                            scatter_fig = go.Figure()
                            for name in plot_df['Name'].unique():
                                player_stats = plot_df[plot_df['Name'] == name]
                                economy_rate = player_stats['Economy Rate'].iloc[0]
                                strike_rate = player_stats['Strike Rate'].iloc[0]
                                wickets = player_stats['Wickets'].iloc[0]
                                scatter_fig.add_trace(go.Scatter(
                                    x=[economy_rate],
                                    y=[strike_rate],
                                    mode='markers+text',
                                    text=[name],
                                    textposition='top center',
                                    marker=dict(size=10),
                                    name=name,
                                    hovertemplate=(
                                        f"<b>{name}</b><br><br>"
                                        f"Economy Rate: {economy_rate:.2f}<br>"
                                        f"Strike Rate: {strike_rate:.2f}<br>"
                                        f"Wickets: {wickets}<br>"
                                        "<extra></extra>"
                                    )
                                ))
                            scatter_fig.update_layout(
                                xaxis_title="Economy Rate",
                                yaxis_title="Strike Rate",
                                height=500,
                                font=dict(size=12),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                showlegend=False
                            )
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #36d1dc 0%, #5b86e5 100%); 
                                        padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                                        box-shadow: 0 6px 24px rgba(54, 209, 220, 0.3);
                                        border: 1px solid rgba(255, 255, 255, 0.2);">
                                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">Economy Rate vs Strike Rate Analysis</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            st.plotly_chart(scatter_fig, use_container_width=True)
                        else:
                            st.info("No players with wickets to display in the Economy Rate vs Strike Rate Analysis.")

    ###-------------------------------------FORMAT STATS-------------------------------------###
    # Format Stats Tab
    with tab_map["Format"]:
            format_df = compute_bowl_format_stats(filtered_df)
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(240, 147, 251, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">📋 Format Statistics</h3>
            </div>
            """, unsafe_allow_html=True)
            if format_df.empty:
                st.info("No format statistics to display for the current selection.")
            else:
                st.dataframe(format_df, use_container_width=True, hide_index=True)
                if not is_large:
                    # --- Modern UI Section Header for Graphs ---
                    st.markdown("""
                    <div style=\"background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); \n                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; \n                            box-shadow: 0 6px 24px rgba(240, 147, 251, 0.25);\n                            border: 1px solid rgba(255, 255, 255, 0.2);\">\n                        <h3 style=\"color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;\">📈 Format Performance Trends by Season</h3>\n                    </div>
                    """, unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    # Get unique formats from filtered data
                    unique_formats = sorted(filtered_df['Match_Format'].unique())
                    # Define format colors
                    format_colors = {
                        'Test Match': '#28a745',              # Green
                        'One Day International': '#dc3545',    # Red
                        '20 Over International': '#ffc107',    # Yellow/Amber
                        'First Class': '#6610f2',              # Purple
                        'List A': '#fd7e14',                   # Orange
                        'T20': '#17a2b8'                       # Cyan
                    }
                    # Prepare totals per year/format
                    totals = filtered_df.groupby(['Match_Format', 'Year'], observed=True).agg({
                        'Bowler_Runs': 'sum',
                        'Bowler_Wkts': 'sum',
                        'Bowler_Balls': 'sum'
                    }).reset_index()
                    totals['Average'] = (totals['Bowler_Runs'] / totals['Bowler_Wkts'].replace(0, np.nan)).round(2).fillna(0)
                    totals['Economy Rate'] = (totals['Bowler_Runs'] / (totals['Bowler_Balls'].replace(0, np.nan) / 6)).round(2).fillna(0)
                    totals['Strike Rate'] = (totals['Bowler_Balls'] / totals['Bowler_Wkts'].replace(0, np.nan)).round(2).fillna(0)
                    # Build a consistent color map for all formats present
                    color_map = {fmt: format_colors.get(fmt, f'#{hash(fmt) & 0xFFFFFF:06x}') for fmt in unique_formats}
                    # Average per Season by Format
                    with col1:
                        st.subheader("Average per Season by Format")
                        fig_avg = go.Figure()
                        for format_name in unique_formats:
                            format_data = totals[totals['Match_Format'] == format_name]
                            color = color_map[format_name]
                            fig_avg.add_trace(go.Scatter(
                                x=format_data['Year'],
                                y=format_data['Average'],
                                mode='lines+markers',
                                name=format_name,
                                line=dict(color=color),
                                marker=dict(color=color, size=8)
                            ))
                        fig_avg.update_layout(
                            height=350,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            legend=dict(orientation="h", yanchor="top", y=1.15, xanchor="center", x=0.5),
                            xaxis_title="Year",
                            yaxis_title="Average",
                            font=dict(size=12)
                        )
                        fig_avg.update_xaxes(tickmode='linear', dtick=1)
                        st.plotly_chart(fig_avg, use_container_width=True)
                    # Economy Rate per Season by Format
                    with col2:
                        st.subheader("Economy Rate per Season by Format")
                        fig_econ = go.Figure()
                        for format_name in unique_formats:
                            format_data = totals[totals['Match_Format'] == format_name]
                            color = color_map[format_name]
                            fig_econ.add_trace(go.Scatter(
                                x=format_data['Year'],
                                y=format_data['Economy Rate'],
                                mode='lines+markers',
                                name=format_name,
                                line=dict(color=color),
                                marker=dict(color=color, size=8)
                            ))
                        fig_econ.update_layout(
                            height=350,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            legend=dict(orientation="h", yanchor="top", y=1.15, xanchor="center", x=0.5),
                            xaxis_title="Year",
                            yaxis_title="Economy Rate",
                            font=dict(size=12)
                        )
                        fig_econ.update_xaxes(tickmode='linear', dtick=1)
                        st.plotly_chart(fig_econ, use_container_width=True)
                    # Strike Rate per Season by Format
                    with col3:
                        st.subheader("Strike Rate per Season by Format")
                        fig_sr = go.Figure()
                        for format_name in unique_formats:
                            format_data = totals[totals['Match_Format'] == format_name]
                            color = color_map[format_name]
                            fig_sr.add_trace(go.Scatter(
                                x=format_data['Year'],
                                y=format_data['Strike Rate'],
                                mode='lines+markers',
                                name=format_name,
                                line=dict(color=color),
                                marker=dict(color=color, size=8)
                            ))
                        fig_sr.update_layout(
                            height=350,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            legend=dict(orientation="h", yanchor="top", y=1.15, xanchor="center", x=0.5),
                            xaxis_title="Year",
                            yaxis_title="Strike Rate",
                            font=dict(size=12)
                        )
                        fig_sr.update_xaxes(tickmode='linear', dtick=1)
                        st.plotly_chart(fig_sr, use_container_width=True)

    ###-------------------------------------SEASON STATS-------------------------------------###
    # Season Stats Tab
    with tab_map["Season"]:
            season_df = compute_bowl_year_stats(filtered_df)
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(240, 147, 251, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">📅 Season Statistics</h3>
            </div>
            """, unsafe_allow_html=True)
            if season_df.empty:
                st.info("No season statistics to display for the current selection.")
            else:
                st.dataframe(season_df, use_container_width=True, hide_index=True)
                if not is_large:
                    # --- Modern UI Section Header for Graphs ---
                    st.markdown("""
                    <div style=\"background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); \n                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; \n                            box-shadow: 0 6px 24px rgba(240, 147, 251, 0.25);\n                            border: 1px solid rgba(255, 255, 255, 0.2);\">\n                        <h3 style=\"color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;\">📈 Bowling Average, Strike Rate & Economy Rate Per Season</h3>\n                    </div>
                    """, unsafe_allow_html=True)
                    # --- Three Column Layout for Graphs ---
                    col1, col2, col3 = st.columns(3)
                    cols = season_df.columns
                    runs_col = 'Runs' if 'Runs' in cols else ('Bowler_Runs' if 'Bowler_Runs' in cols else None)
                    wickets_col = 'Wickets' if 'Wickets' in cols else ('Bowler_Wkts' if 'Bowler_Wkts' in cols else None)
                    balls_col = 'Balls' if 'Balls' in cols else ('Bowler_Balls' if 'Bowler_Balls' in cols else None)
                    if not (runs_col and wickets_col and balls_col):
                        st.error("Could not find the correct columns for runs, wickets, and balls in season_df.")
                    else:
                        totals = season_df.groupby('Year', observed=True).agg({
                            runs_col: 'sum',
                            wickets_col: 'sum',
                            balls_col: 'sum'
                        }).reset_index()
                        totals['Average'] = (totals[runs_col] / totals[wickets_col].replace(0, np.nan)).round(2).fillna(0)
                        totals['Strike Rate'] = (totals[balls_col] / totals[wickets_col].replace(0, np.nan)).round(2).fillna(0)
                        totals['Economy Rate'] = (totals[runs_col] / (totals[balls_col].replace(0, np.nan) / 6)).round(2).fillna(0)
                        for col, metric, ytitle in zip([col1, col2, col3], ['Average', 'Strike Rate', 'Economy Rate'], ["Bowling Average", "Strike Rate", "Economy Rate"]):
                            with col:
                                st.subheader(ytitle)
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=totals['Year'],
                                    y=totals[metric],
                                    mode='lines+markers',
                                    name='All Players',
                                    line=dict(color='#f04f53'),
                                    marker=dict(color='#f04f53', size=8),
                                    showlegend=False
                                ))
                                fig.update_layout(
                                    height=350,
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    xaxis_title="Year",
                                    yaxis_title=ytitle,
                                    font=dict(size=12)
                                )
                                fig.update_xaxes(tickmode='linear', dtick=1)
                                st.plotly_chart(fig, use_container_width=True)

    ###-------------------------------------LATEST INNINGS-------------------------------------###
    # Latest Innings Tab
    with tab_map["Latest"]:
            latest_innings = compute_bowl_latest_innings(filtered_df)
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140  100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">Latest 20 Bowling Innings</h3>
            </div>
            """, unsafe_allow_html=True)
            if latest_innings.empty:
                st.info("No latest innings to display for the current selection.")
            else:
                st.dataframe(latest_innings, use_container_width=True, hide_index=True)
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.25);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">📊 Summary Statistics</h3>
                </div>
                """, unsafe_allow_html=True)
                col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
                with col1:
                    st.metric("Matches", latest_innings['File Name'].nunique())
                with col2:
                    st.metric("Innings", latest_innings['Innings'].nunique())
                with col3:
                    st.metric("Wickets", latest_innings['Wickets'].sum())
                with col4:
                    st.metric("Runs", latest_innings['Runs'].sum())
                with col5:
                    st.metric("Overs", latest_innings['Overs'].sum())
                with col6:
                    st.metric("Maidens", latest_innings['Maidens'].sum())
                with col7:
                    avg = (latest_innings['Runs'].sum() / latest_innings['Wickets'].replace(0, np.nan).sum()) if latest_innings['Wickets'].sum() > 0 else 0
                    st.metric("Average", f"{avg:.2f}")

    ###-------------------------------------OPPONENT STATS-------------------------------------###
    # Opponent Stats Tab  
    with tab_map["Opponent"]:
            opponent_summary = compute_bowl_opponent_stats(filtered_df)
            st.markdown("""
            <div style="background: linear-gradient(135deg, #a8caba 0%, #5d4e75 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(168, 202, 186,  0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">Opposition Statistics</h3>
            </div>
            """, unsafe_allow_html=True)

            if opponent_summary.empty:
                st.info("No opponent statistics to display for the current selection.")
            else:
                st.dataframe(opponent_summary, use_container_width=True, hide_index=True)
                if not is_large:
                    # --- Plotting Logic ---
                    st.markdown("""
                    <div style=\"background: linear-gradient(135deg, #a8caba 0%, #5d4e75 100%); \n                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; \n                            box-shadow: 0 6px 24px rgba(168, 202, 186, 0.25);\n                            border: 1px solid rgba(255, 255, 255, 0.2);\">\n                        <h3 style=\"color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;\">Average vs Opponent Team</h3>\n                    </div>
                    """, unsafe_allow_html=True)
                    fig = go.Figure()
                    for name in opponent_summary['Name'].unique():
                        player_data = opponent_summary[opponent_summary['Name'] == name]
                        fig.add_trace(go.Bar(
                            x=player_data['Opposition'], 
                            y=player_data['Average'], 
                            name=name,
                        ))
                    fig.update_layout(barmode='group', xaxis_title="Opposition Team", yaxis_title="Average")
                    st.plotly_chart(fig, use_container_width=True)

    ###-------------------------------------LOCATION STATS-------------------------------------###
    # Location Stats Tab
    with tab_map["Location"]:
            location_summary = compute_bowl_location_stats(filtered_df)
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">Location Statistics</h3>
            </div>
            """, unsafe_allow_html=True)
            if location_summary.empty:
                st.info("No location statistics to display for the current selection.")
            else:
                st.dataframe(location_summary, use_container_width=True, hide_index=True)
                if not is_large:
                    st.markdown("""
                    <div style=\"background: linear-gradient(135deg, #4776e6 0%, #8e54e9 100%); \n                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; \n                            box-shadow: 0 6px 24px rgba(71, 118, 230, 0.25);\n                            border: 1px solid rgba(255, 255, 255, 0.2);\">\n                        <h3 style=\"color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;\">Average vs Location</h3>\n                    </div>
                    """, unsafe_allow_html=True)
                    fig = go.Figure()
                    for name in location_summary['Name'].unique():
                        player_data = location_summary[location_summary['Name'] == name]
                        fig.add_trace(go.Bar(
                            x=player_data['Location'],
                            y=player_data['Average'],
                            name=name,
                        ))
                    fig.update_layout(barmode='group', xaxis_title="Location", yaxis_title="Average")
                    st.plotly_chart(fig, use_container_width=True)

    ###-------------------------------------INNINGS STATS-------------------------------------###
    # Innings Stats Tab
    with tab_map["Innings"]:
            innings_summary = compute_bowl_innings_stats(filtered_df)
            st.markdown("""
            <div style="background: linear-gradient(135deg, #36d1dc 0%, #5b86e5 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(54, 209, 220, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">Innings Statistics</h3>
            </div>
            """, unsafe_allow_html=True)
            if innings_summary.empty:
                st.info("No innings statistics to display for the current selection.")
            else:
                st.dataframe(innings_summary, use_container_width=True, hide_index=True)
                if not is_large:
                    st.markdown("""
                    <div style=\"background: linear-gradient(135deg, #36d1dc 0%, #5b86e5 100%); \n                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; \n                            box-shadow: 0 6px 24px rgba(54, 209, 220, 0.25);\n                            border: 1px solid rgba(255, 255, 255, 0.2);\">\n                        <h3 style=\"color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;\">Average vs Innings</h3>\n                    </div>
                    """, unsafe_allow_html=True)
                    fig = go.Figure()
                    for name in innings_summary['Name'].unique():
                        player_data = innings_summary[innings_summary['Name'] == name]
                        fig.add_trace(go.Bar(
                            x=player_data['Innings'],
                            y=player_data['Average'],
                            name=name,
                        ))
                    fig.update_layout(barmode='group', xaxis_title="Innings", yaxis_title="Average")
                    st.plotly_chart(fig, use_container_width=True)

    ###-------------------------------------POSITION STATS-------------------------------------###
    # Position Stats Tab
    with tab_map["Position"]:
            position_summary = compute_bowl_position_stats(filtered_df)
            st.markdown("""
            <div style="background: linear-gradient(135deg, #8360c3 0%, #2ebf91 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px  32px rgba(131, 96, 195, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">Position Statistics</h3>
            </div>
            """, unsafe_allow_html=True)
            if position_summary.empty:
                st.info("No position statistics to display for the current selection.")
            else:
                st.dataframe(position_summary, use_container_width=True, hide_index=True)
                if not is_large:
                    st.markdown("""
                    <div style=\"background: linear-gradient(135deg, #8360c3 0%, #2ebf91 100%); \n                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; \n                            box-shadow: 0 6px 24px rgba(131, 96, 195, 0.25);\n                            border: 1px solid rgba(255, 255, 255, 0.2);\">\n                        <h3 style=\"color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;\">Average vs Position</h3>\n                    </div>
                    """, unsafe_allow_html=True)
                    fig = go.Figure()
                    for name in position_summary['Name'].unique():
                        player_data = position_summary[position_summary['Name'] == name]
                        fig.add_trace(go.Bar(
                            x=player_data['Position'],
                            y=player_data['Average'],
                            name=name,
                        ))
                    fig.update_layout(barmode='group', xaxis_title="Position", yaxis_title="Average")
                    st.plotly_chart(fig, use_container_width=True)

    ###-------------------------------------HOME/AWAY STATS-------------------------------------###
    # Home/Away Stats Tab
    with tab_map["Home/Away"]:
            homeaway_stats_df = compute_bowl_homeaway_stats(filtered_df)
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(255, 126, 95, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">Home/Away Bowling Statistics</h3>
            </div>
            """, unsafe_allow_html=True)
            if homeaway_stats_df.empty:
                st.info("No Home/Away statistics to display for the current selection.")
            else:
                st.dataframe(homeaway_stats_df, use_container_width=True, hide_index=True)
                if not is_large:
                    # --- Modern UI Section Header for Graphs ---
                    st.markdown("""
                    <div style=\"background: linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%); \n                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; \n                            box-shadow: 0 6px 24px rgba(255, 126, 95, 0.25);\n                            border: 1px solid rgba(255, 255, 255, 0.2);\">\n                        <h3 style=\"color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;\">📈 Home/Away Performance Trends by Year</h3>\n                    </div>
                    """, unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    # Prepare yearly stats by Home/Away
                    yearly_ha = filtered_df.groupby(['Year', 'HomeOrAway'], observed=True).agg({
                        'Bowler_Runs': 'sum',
                        'Bowler_Wkts': 'sum',
                        'Bowler_Balls': 'sum'
                    }).reset_index()
                    yearly_ha['Average'] = (yearly_ha['Bowler_Runs'] / yearly_ha['Bowler_Wkts'].replace(0, np.nan)).round(2).fillna(0)
                    yearly_ha['Economy Rate'] = (yearly_ha['Bowler_Runs'] / (yearly_ha['Bowler_Balls'].replace(0, np.nan) / 6)).round(2).fillna(0)
                    yearly_ha['Strike Rate'] = (yearly_ha['Bowler_Balls'] / yearly_ha['Bowler_Wkts'].replace(0, np.nan)).round(2).fillna(0)
                    ha_colors = {'Home': '#1f77b4', 'Away': '#d62728', 'Neutral': '#2ca02c'}
                    # Average by Year
                    with col1:
                        st.subheader("Average by Year")
                        fig_avg = go.Figure()
                        for ha in yearly_ha['HomeOrAway'].unique():
                            ha_data = yearly_ha[yearly_ha['HomeOrAway'] == ha]
                            color = ha_colors.get(ha, f'#{hash(ha) & 0xFFFFFF:06x}')
                            fig_avg.add_trace(go.Scatter(
                                x=ha_data['Year'],
                                y=ha_data['Average'],
                                mode='lines+markers',
                                name=ha,
                                line=dict(color=color),
                                marker=dict(color=color, size=8)
                            ))
                        fig_avg.update_layout(
                            height=350,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            legend=dict(orientation="h", yanchor="top", y=1.15, xanchor="center", x=0.5),
                            xaxis_title="Year",
                            yaxis_title="Average",
                            font=dict(size=12)
                        )
                        fig_avg.update_xaxes(tickmode='linear', dtick=1)
                        st.plotly_chart(fig_avg, use_container_width=True)
                    # Economy Rate by Year
                    with col2:
                        st.subheader("Economy Rate by Year")
                        fig_econ = go.Figure()
                        for ha in yearly_ha['HomeOrAway'].unique():
                            ha_data = yearly_ha[yearly_ha['HomeOrAway'] == ha]
                            color = ha_colors.get(ha, f'#{hash(ha) & 0xFFFFFF:06x}')
                            fig_econ.add_trace(go.Scatter(
                                x=ha_data['Year'],
                                y=ha_data['Economy Rate'],
                                mode='lines+markers',
                                name=ha,
                                line=dict(color=color),
                                marker=dict(color=color, size=8)
                            ))
                        fig_econ.update_layout(
                            height=350,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            legend=dict(orientation="h", yanchor="top", y=1.15, xanchor="center", x=0.5),
                            xaxis_title="Year",
                            yaxis_title="Economy Rate",
                            font=dict(size=12)
                        )
                        fig_econ.update_xaxes(tickmode='linear', dtick=1)
                        st.plotly_chart(fig_econ, use_container_width=True)
                    # Strike Rate by Year
                    with col3:
                        st.subheader("Strike Rate by Year")
                        fig_sr = go.Figure()
                        for ha in yearly_ha['HomeOrAway'].unique():
                            ha_data = yearly_ha[yearly_ha['HomeOrAway'] == ha]
                            color = ha_colors.get(ha, f'#{hash(ha) & 0xFFFFFF:06x}')
                            fig_sr.add_trace(go.Scatter(
                                x=ha_data['Year'],
                                y=ha_data['Strike Rate'],
                                mode='lines+markers',
                                name=ha,
                                line=dict(color=color),
                                marker=dict(color=color, size=8)
                            ))
                        fig_sr.update_layout(
                            height=350,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            legend=dict(orientation="h", yanchor="top", y=1.15, xanchor="center", x=0.5),
                            xaxis_title="Year",
                            yaxis_title="Strike Rate",
                            font=dict(size=12)
                        )
                        fig_sr.update_xaxes(tickmode='linear', dtick=1)
                        st.plotly_chart(fig_sr, use_container_width=True)

    ###--------------------------------------CUMULATIVE BOWLING STATS------------------------------------------#######
    # Cumulative Stats Tab
    if not is_large:
        with tab_map["Cumulative"]:
            cumulative_stats = compute_bowl_cumulative_stats(filtered_df)
            st.markdown("""
        <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                    padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                    box-shadow: 0 8px 32px rgba(17, 153, 142, 0.3);
                    border: 1px solid rgba(255, 255, 255, 0.2);">
            <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">Cumulative Bowling Statistics</h3>
        </div>
        """, unsafe_allow_html=True)
            if cumulative_stats.empty:
                st.info("No cumulative statistics to display for the current selection.")
            else:
                st.dataframe(cumulative_stats, use_container_width=True, hide_index=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader("Cumulative Average")
                    fig1 = go.Figure()
                    for name in cumulative_stats['Name'].unique():
                        player_data = cumulative_stats[cumulative_stats['Name'] == name]
                        fig1.add_trace(go.Scatter(x=player_data['Cumulative Matches'], y=player_data['Cumulative Avg'], mode='lines', name=name))
                    fig1.update_layout(xaxis_title='Cumulative Matches', yaxis_title='Cumulative Average')
                    st.plotly_chart(fig1, use_container_width=True)
                with col2:
                    st.subheader("Cumulative Strike Rate")
                    fig2 = go.Figure()
                    for name in cumulative_stats['Name'].unique():
                        player_data = cumulative_stats[cumulative_stats['Name'] == name]
                        fig2.add_trace(go.Scatter(x=player_data['Cumulative Matches'], y=player_data['Cumulative SR'], mode='lines', name=name))
                    fig2.update_layout(xaxis_title='Cumulative Matches', yaxis_title='Cumulative Strike Rate')
                    st.plotly_chart(fig2, use_container_width=True)
                with col3:
                    st.subheader("Cumulative Economy Rate")
                    fig3 = go.Figure()
                    for name in cumulative_stats['Name'].unique():
                        player_data = cumulative_stats[cumulative_stats['Name'] == name]
                        fig3.add_trace(go.Scatter(x=player_data['Cumulative Matches'], y=player_data['Cumulative Econ'], mode='lines', name=name))
                    fig3.update_layout(xaxis_title='Cumulative Matches', yaxis_title='Cumulative Economy Rate')
                    st.plotly_chart(fig3, use_container_width=True)

    ###--------------------------------------BOWLING BLOCK STATS------------------------------------------#######
    # Block Stats Tab
    if not is_large:
        with tab_map["Block"]:
            block_stats_df = compute_bowl_block_stats(filtered_df)
            st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                    padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                    box-shadow: 0 8px 32px rgba(30, 60, 114, 0.4);
                    border: 1px solid rgba(255, 255, 255, 0.2);">
            <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">Block Statistics (Groups of 20 Innings)</h3>
        </div>
        """, unsafe_allow_html=True)
            
            # --- THIS IS THE FIX ---
            if block_stats_df.empty:
                st.info("No block statistics to display for the current selection.")
            else:
                # All the code that uses the dataframe now goes inside this 'else' block
                st.dataframe(block_stats_df, use_container_width=True, hide_index=True)
                st.markdown("""
                <div style=\"background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); \n                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; \n                            box-shadow: 0 6px 24px rgba(30, 60, 114, 0.25);\n                            border: 1px solid rgba(255, 255, 255, 0.2);\">\n                    <h3 style=\"color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;\">Bowling Average by Innings Block</h3>\n                </div>
                """, unsafe_allow_html=True)
                
                fig = go.Figure()
                # This code is now safe because we know the DataFrame is not empty
                if 'All' in name_choice:
                    all_blocks = block_stats_df.groupby('Innings_Range', observed=True).agg({
                        'Runs': 'sum',
                        'Wickets': 'sum'
                    }).reset_index()
                    all_blocks['Average'] = (all_blocks['Runs'] / all_blocks['Wickets']).round(2)
                    all_blocks = all_blocks.sort_values('Innings_Range', key=lambda x: [int(i.split('-')[0]) for i in x])
                    all_color = '#f84e4e' if not individual_players else 'black'
                    fig.add_trace(
                        go.Bar(
                            x=all_blocks['Innings_Range'],
                            y=all_blocks['Average'],
                            name='All Players',
                            marker_color=all_color
                        )
                    )
                # Add individual player traces
                for i, name in enumerate(individual_players):
                    player_blocks = block_stats_df[block_stats_df['Name'] == name].sort_values('Innings_Range', key=lambda x: [int(i.split('-')[0]) for i in x])
                    color = '#f84e4e' if i == 0 else f'#{random.randint(0, 0xFFFFFF):06x}'
                    fig.add_trace(
                        go.Bar(
                            x=player_blocks['Innings_Range'],
                            y=player_blocks['Average'],
                            name=name,
                            marker_color=color
                        )
                    )
                fig.update_layout(
                    showlegend=True,
                    height=500,
                    xaxis_title='Innings Range',
                    yaxis_title='Bowling Average',
                    margin=dict(l=50, r=50, t=70, b=50),
                    font=dict(size=12),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    barmode='group',
                    xaxis={'categoryorder': 'array', 'categoryarray': sorted(block_stats_df['Innings_Range'].unique(), key=lambda x: int(x.split('-')[0]))}
                )
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                st.plotly_chart(fig, use_container_width=True)

        logger.log_total(
            "BowlView render complete",
            perf_start,
            filtered_rows=len(filtered_df),
            filtered_players=int(filtered_df['Name'].nunique()),
            filtered_matches=int(filtered_df['File Name'].nunique())
        )

# Display the bowling view
display_bowl_view()
