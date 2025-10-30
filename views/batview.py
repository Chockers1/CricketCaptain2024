import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import json
from datetime import timedelta
from functools import wraps
from typing import Dict, List
import time
import subprocess

import polars as pl

try:
    from .logging_utils import FastViewLogger
except ImportError:  # Fallback when views package not imported relatively
    from views.logging_utils import FastViewLogger

# Modern UI Styling with Cricket Theme
st.markdown("""
<style>
/* Main Header Styling */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 20px;
    margin: 1rem 0 2rem 0;
    text-align: center;
    box-shadow: 0 12px 48px rgba(102, 126, 234, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.main-header h1 {
    color: white !important;
    margin: 0 !important;
    font-weight: bold;
    font-size: 2.5rem;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}

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

/* Table Styling */
table { 
    color: black; 
    width: 100%; 
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
}

thead tr th {
    background: linear-gradient(135deg, #f04f53 0%, #f5576c 100%) !important;
    color: white !important;
    font-weight: bold !important;
    text-align: center !important;
    padding: 1rem !important;
}

tbody tr:nth-child(even) { 
    background-color: #f0f2f6; 
}

tbody tr:nth-child(odd) { 
    background-color: white; 
}

tbody tr:hover {
    background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%) !important;
    transform: scale(1.01);
    transition: all 0.2s ease;
}

/* Section Headers */
.section-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem 1.5rem;
    border-radius: 15px;
    margin: 2rem 0 1.5rem 0;
    text-align: center;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.section-header h3 {
    color: white !important;
    margin: 0 !important;
    font-weight: bold;
    font-size: 1.3rem;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
}

/* Modern Cards */
.stat-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.8);
    transition: all 0.3s ease;
}

.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
}

.stat-value {
    font-size: 1.5rem;
    font-weight: bold;
    color: #f04f53;
    margin: 0;
}

.stat-label {
    font-size: 0.9rem;
    color: #6c757d;
    margin: 0;
}

/* Modern Warning */
.modern-warning {
    background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    border-left: 4px solid #ffc107;
    padding: 1rem 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
    box-shadow: 0 4px 16px rgba(255, 193, 7, 0.2);
}

/* Responsive Design */
@media (max-width: 768px) {
    .main-header h1 {
        font-size: 2rem;
    }
}
</style>
""", unsafe_allow_html=True)

# Check if seaborn is installed, if not, install it
try:
    import seaborn as sns
except ImportError:
    import subprocess
    subprocess.check_call(["python", "-m", "pip", "install", "seaborn"])
    import seaborn as sns

import matplotlib.pyplot as plt

# Increase cache expiry to reduce recalculations for unchanged data
CACHE_EXPIRY = timedelta(hours=72)  # Increased from 24 to 72 hours

def handle_redis_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            return None
    return wrapper

def generate_cache_key(filters):
    """Generate a unique cache key based on filter parameters"""
    # Only include non-default filter values to reduce cache key variations
    filtered_params = {}
    for key, value in filters.items():
        if isinstance(value, list) and len(value) == 1 and value[0] == 'All':
            continue  # Skip default "All" selections
        if isinstance(value, tuple) and value[0] == value[1]:
            continue  # Skip default range selections
        filtered_params[key] = value
        
    sorted_filters = dict(sorted(filtered_params.items()))
    return f"bat_stats_{hash(json.dumps(sorted_filters))}"

def get_filtered_options(df, column, selected_filters=None):
    """
    Get available options for a column based on current filter selections.
    
    Args:
        df: The DataFrame to filter
        column: The column to get unique values from
        selected_filters: Dictionary of current filter selections
    """
    if selected_filters is None:
        return ['All'] + sorted(df[column].unique().tolist())
    
    filtered_df = df.copy()
    
    # Apply each active filter
    for filter_col, filter_val in selected_filters.items():
        if filter_val and 'All' not in filter_val and filter_col != column:
            filtered_df = filtered_df[filtered_df[filter_col].isin(filter_val)]
    
    return ['All'] + sorted(filtered_df[column].unique().tolist())

def _needs_sanitization(df: pd.DataFrame) -> bool:
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


def _sanitize_df_for_polars(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to Polars/Arrow friendly types to avoid conversion failures."""
    if df is None or df.empty:
        return df

    if not _needs_sanitization(df):
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


def _empty_bat_metric_frames() -> Dict[str, pd.DataFrame]:
    keys = [
        "career",
        "format",
        "season",
        "opponent",
        "location",
        "innings",
        "position",
        "homeaway",
        "cumulative",
        "block",
    ]
    return {key: pd.DataFrame() for key in keys}


def _sanitize_batting_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    safe_df = _sanitize_df_for_polars(df).copy()

    text_columns = [
        "Name",
        "File Name",
        "Match_Format",
        "HomeOrAway",
        "Player_of_the_Match",
        "How Out",
        "comp",
    ]

    for col in text_columns:
        if col in safe_df.columns:
            safe_df[col] = safe_df[col].astype(str)
        else:
            safe_df[col] = ""

    # Ensure consistent team columns
    if "Bat_Team_y" not in safe_df.columns:
        if "Bat_Team" in safe_df.columns:
            safe_df["Bat_Team_y"] = safe_df["Bat_Team"].astype(str)
        elif "Bat_Team_x" in safe_df.columns:
            safe_df["Bat_Team_y"] = safe_df["Bat_Team_x"].astype(str)
        else:
            safe_df["Bat_Team_y"] = ""
    else:
        safe_df["Bat_Team_y"] = safe_df["Bat_Team_y"].astype(str)

    if "Bowl_Team_y" not in safe_df.columns:
        if "Bowl_Team" in safe_df.columns:
            safe_df["Bowl_Team_y"] = safe_df["Bowl_Team"].astype(str)
        elif "Bowl_Team_x" in safe_df.columns:
            safe_df["Bowl_Team_y"] = safe_df["Bowl_Team_x"].astype(str)
        else:
            safe_df["Bowl_Team_y"] = ""
    else:
        safe_df["Bowl_Team_y"] = safe_df["Bowl_Team_y"].astype(str)

    home_col = "Home Team" if "Home Team" in safe_df.columns else "Home_Team"
    away_col = "Away Team" if "Away Team" in safe_df.columns else "Away_Team"

    if home_col not in safe_df.columns:
        safe_df[home_col] = ""
    if away_col not in safe_df.columns:
        safe_df[away_col] = ""

    safe_df[home_col] = safe_df[home_col].astype(str)
    safe_df[away_col] = safe_df[away_col].astype(str)

    numeric_columns = [
        "Batted",
        "Out",
        "Not Out",
        "Balls",
        "Runs",
        "4s",
        "6s",
        "50s",
        "100s",
        "150s",
        "200s",
        "<25&Out",
        "Caught",
        "Bowled",
        "LBW",
        "Run Out",
        "Stumped",
        "Total_Runs",
        "Overs",
        "Wickets",
        "Team Balls",
        "Position",
        "Innings",
        "Year",
    ]

    for col in numeric_columns:
        if col not in safe_df.columns:
            safe_df[col] = 0
        safe_df[col] = pd.to_numeric(safe_df[col], errors="coerce").fillna(0)

    # Enforce integer types where appropriate
    for col in ["Position", "Innings", "Year"]:
        safe_df[col] = safe_df[col].astype(int)

    if "Date" in safe_df.columns:
        safe_df["Date"] = pd.to_datetime(safe_df["Date"], errors="coerce")
    else:
        safe_df["Date"] = pd.NaT

    return safe_df


def _bat_summary(
    pl_df: pl.DataFrame,
    group_cols: List[str],
    *,
    avg_match_avg: float,
    avg_match_sr: float,
    include_pom: bool = False,
) -> pl.DataFrame:
    base_aggs = [
        pl.col("File Name").n_unique().alias("Matches"),
        pl.col("Batted").sum().alias("Inns"),
        pl.col("Out").sum().alias("Out"),
        pl.col("Not Out").sum().alias("Not Out"),
        pl.col("Balls").sum().alias("Balls"),
        pl.col("Runs").sum().alias("Runs"),
        pl.col("Runs").max().alias("HS"),
        pl.col("4s").sum().alias("4s"),
        pl.col("6s").sum().alias("6s"),
        pl.col("50s").sum().alias("50s"),
        pl.col("100s").sum().alias("100s"),
        pl.col("150s").sum().alias("150s"),
        pl.col("200s").sum().alias("200s"),
        pl.col("<25&Out").sum().alias("<25&Out"),
        pl.col("Caught").sum().alias("Caught"),
        pl.col("Bowled").sum().alias("Bowled"),
        pl.col("LBW").sum().alias("LBW"),
        pl.col("Run Out").sum().alias("Run Out"),
        pl.col("Stumped").sum().alias("Stumped"),
        pl.col("Total_Runs").sum().alias("Team Runs"),
        pl.col("Overs").sum().alias("Overs"),
        pl.col("Wickets").sum().alias("Wickets"),
        pl.col("Team Balls").sum().alias("Team Balls"),
    ]

    summary = pl_df.group_by(group_cols).agg(base_aggs)

    if include_pom and "Player_of_the_Match" in pl_df.columns:
        pom_group = (
            pl_df.filter(pl.col("Player_of_the_Match") == pl.col("Name"))
            .group_by(group_cols)
            .agg(pl.col("File Name").n_unique().alias("POM"))
        )
        summary = summary.join(pom_group, on=group_cols, how="left")
    else:
        summary = summary.with_columns(pl.lit(0).alias("POM"))

    summary = summary.fill_null(0)

    summary = summary.with_columns([
        pl.when(pl.col("Out") > 0)
        .then((pl.col("Runs") / pl.col("Out")).round(2))
        .otherwise(0)
        .alias("Avg"),
        pl.when(pl.col("Out") > 0)
        .then((pl.col("Balls") / pl.col("Out")).round(2))
        .otherwise(0)
        .alias("BPO"),
        pl.when(pl.col("Balls") > 0)
        .then((pl.col("Runs") / pl.col("Balls") * 100).round(2))
        .otherwise(0)
        .alias("SR"),
        pl.when(pl.col("Wickets") > 0)
        .then((pl.col("Team Runs") / pl.col("Wickets")).round(2))
        .otherwise(0)
        .alias("TeamAvg"),
        pl.when(pl.col("Team Balls") > 0)
        .then((pl.col("Team Runs") / pl.col("Team Balls") * 100).round(2))
        .otherwise(0)
        .alias("TeamSR"),
        pl.when((pl.col("4s") + pl.col("6s")) > 0)
        .then((pl.col("Balls") / (pl.col("4s") + pl.col("6s"))).round(2))
        .otherwise(0)
        .alias("BPB"),
        pl.when(pl.col("Runs") > 0)
        .then((((pl.col("4s") * 4) + (pl.col("6s") * 6)) / pl.col("Runs") * 100).round(2))
        .otherwise(0)
        .alias("BoundaryPct"),
        pl.when(pl.col("Matches") > 0)
        .then((pl.col("Runs") / pl.col("Matches")).round(2))
        .otherwise(0)
        .alias("RPM"),
        pl.when(pl.col("Inns") > 0)
        .then(((pl.col("50s") + pl.col("100s") + pl.col("150s") + pl.col("200s")) / pl.col("Inns") * 100).round(2))
        .otherwise(0)
        .alias("FiftyPlusPI"),
        pl.when(pl.col("Inns") > 0)
        .then(((pl.col("100s") + pl.col("150s") + pl.col("200s")) / pl.col("Inns") * 100).round(2))
        .otherwise(0)
        .alias("HundredPI"),
        pl.when(pl.col("Inns") > 0)
        .then(((pl.col("150s") + pl.col("200s")) / pl.col("Inns") * 100).round(2))
        .otherwise(0)
        .alias("HundredFiftyPI"),
        pl.when(pl.col("Inns") > 0)
        .then((pl.col("200s") / pl.col("Inns") * 100).round(2))
        .otherwise(0)
        .alias("DoubleHundredPI"),
        pl.when(pl.col("Inns") > 0)
        .then((pl.col("<25&Out") / pl.col("Inns") * 100).round(2))
        .otherwise(0)
        .alias("Below25OutPI"),
        pl.when((pl.col("50s") + pl.col("100s")) > 0)
        .then((pl.col("100s") / (pl.col("50s") + pl.col("100s")) * 100).round(2))
        .otherwise(0)
        .alias("ConversionRate"),
        pl.when(pl.col("Inns") > 0)
        .then((pl.col("Caught") / pl.col("Inns") * 100).round(2))
        .otherwise(0)
        .alias("CaughtPct"),
        pl.when(pl.col("Inns") > 0)
        .then((pl.col("Bowled") / pl.col("Inns") * 100).round(2))
        .otherwise(0)
        .alias("BowledPct"),
        pl.when(pl.col("Inns") > 0)
        .then((pl.col("LBW") / pl.col("Inns") * 100).round(2))
        .otherwise(0)
        .alias("LBWPct"),
        pl.when(pl.col("Inns") > 0)
        .then((pl.col("Run Out") / pl.col("Inns") * 100).round(2))
        .otherwise(0)
        .alias("RunOutPct"),
        pl.when(pl.col("Inns") > 0)
        .then((pl.col("Stumped") / pl.col("Inns") * 100).round(2))
        .otherwise(0)
        .alias("StumpedPct"),
        pl.when(pl.col("Inns") > 0)
        .then((pl.col("Not Out") / pl.col("Inns") * 100).round(2))
        .otherwise(0)
        .alias("NotOutPct"),
        pl.when(pl.col("Matches") > 0)
        .then((pl.col("POM") / pl.col("Matches") * 100).round(2))
        .otherwise(0)
        .alias("POMPerMatch"),
    ])

    summary = summary.with_columns([
        pl.when(pl.col("TeamAvg") > 0)
        .then((pl.col("Avg") / pl.col("TeamAvg") * 100).round(2))
        .otherwise(0)
        .alias("TeamPlusAvg"),
        pl.when(pl.col("TeamSR") > 0)
        .then((pl.col("SR") / pl.col("TeamSR") * 100).round(2))
        .otherwise(0)
        .alias("TeamPlusSR"),
    ])

    if avg_match_avg > 0:
        summary = summary.with_columns(
            (pl.col("Avg") / pl.lit(avg_match_avg) * 100).round(2).alias("MatchPlusAvg")
        )
    else:
        summary = summary.with_columns(pl.lit(0).alias("MatchPlusAvg"))

    if avg_match_sr > 0:
        summary = summary.with_columns(
            (pl.col("SR") / pl.lit(avg_match_sr) * 100).round(2).alias("MatchPlusSR")
        )
    else:
        summary = summary.with_columns(pl.lit(0).alias("MatchPlusSR"))

    return summary.fill_null(0)


def _finalize_metric_frame(
    summary: pl.DataFrame,
    rename_map: Dict[str, str],
    column_order: List[str],
    *,
    sort_by: str | None = None,
    ascending: bool = False,
) -> pd.DataFrame:
    if summary.is_empty():
        return pd.DataFrame(columns=column_order)

    pdf = summary.to_pandas()
    pdf = pdf.rename(columns=rename_map)

    for col in column_order:
        if col not in pdf.columns:
            pdf[col] = 0

    pdf = pdf[column_order]

    if sort_by and sort_by in pdf.columns:
        pdf = pdf.sort_values(by=sort_by, ascending=ascending)

    return pdf.reset_index(drop=True)


def _build_cumulative_stats(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Name",
        "Match_Format",
        "Date",
        "File Name",
        "Batted",
        "Out",
        "Balls",
        "Runs",
        "100s",
        "Cumulative Innings",
        "Cumulative Runs",
        "Cumulative Balls",
        "Cumulative Outs",
        "Cumulative 100s",
        "Cumulative Avg",
        "Cumulative SR",
    ]

    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    work_df = df[[
        "Name",
        "Match_Format",
        "Date",
        "File Name",
        "Batted",
        "Out",
        "Balls",
        "Runs",
        "100s",
    ]].copy()

    work_df = work_df.sort_values(by=["Name", "Match_Format", "Date", "File Name"])

    match_level_df = (
        work_df.groupby(["Name", "Match_Format", "Date", "File Name"], observed=True)
        .agg({
            "Batted": "sum",
            "Out": "sum",
            "Balls": "sum",
            "Runs": "sum",
            "100s": "sum",
        })
        .reset_index()
    )

    if "Date" in match_level_df.columns:
        match_level_df["Date"] = pd.to_datetime(match_level_df["Date"], errors="coerce")

    match_level_df["Cumulative Innings"] = match_level_df.groupby(["Name", "Match_Format"], observed=True)["Batted"].cumsum()
    match_level_df["Cumulative Runs"] = match_level_df.groupby(["Name", "Match_Format"], observed=True)["Runs"].cumsum()
    match_level_df["Cumulative Balls"] = match_level_df.groupby(["Name", "Match_Format"], observed=True)["Balls"].cumsum()
    match_level_df["Cumulative Outs"] = match_level_df.groupby(["Name", "Match_Format"], observed=True)["Out"].cumsum()
    match_level_df["Cumulative 100s"] = match_level_df.groupby(["Name", "Match_Format"], observed=True)["100s"].cumsum()

    match_level_df["Cumulative Avg"] = (
        match_level_df["Cumulative Runs"] /
        match_level_df["Cumulative Outs"].replace(0, np.nan)
    ).fillna(0).round(2)
    match_level_df["Cumulative SR"] = (
        match_level_df["Cumulative Runs"] /
        match_level_df["Cumulative Balls"].replace(0, np.nan) * 100
    ).fillna(0).round(2)

    match_level_df = match_level_df.sort_values(by="Date", ascending=False)
    return match_level_df[columns]


def _build_block_stats(cumulative_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Name",
        "Match_Format",
        "Match_Block",
        "Matches",
        "Runs",
        "Balls",
        "Outs",
        "First_Date",
        "Last_Date",
        "Avg",
        "SR",
        "Match_Range",
        "Date_Range",
    ]

    if cumulative_df is None or cumulative_df.empty:
        return pd.DataFrame(columns=columns)

    block_df = cumulative_df.copy()
    if "Date" in block_df.columns:
        block_df["Date"] = pd.to_datetime(block_df["Date"], errors="coerce")

    block_df["Match_Block"] = ((block_df["Cumulative Innings"] - 1) // 20)

    agg_df = (
        block_df.groupby(["Name", "Match_Format", "Match_Block"], observed=True)
        .agg({
            "Batted": "count",
            "Runs": "sum",
            "Balls": "sum",
            "Out": "sum",
            "Date": ["min", "max"],
        })
        .reset_index()
    )

    agg_df.columns = [
        "Name",
        "Match_Format",
        "Match_Block",
        "Matches",
        "Runs",
        "Balls",
        "Outs",
        "First_Date",
        "Last_Date",
    ]

    agg_df["Avg"] = (
        agg_df["Runs"] / agg_df["Outs"].replace(0, np.nan)
    ).fillna(0).round(2)
    agg_df["SR"] = (
        agg_df["Runs"] / agg_df["Balls"].replace(0, np.nan) * 100
    ).fillna(0).round(2)

    agg_df["Match_Range"] = agg_df["Match_Block"].apply(lambda x: f"{x * 20 + 1}-{x * 20 + 20}")
    agg_df["Date_Range"] = (
        agg_df["First_Date"].dt.strftime("%d/%m/%Y")
        + " to "
        + agg_df["Last_Date"].dt.strftime("%d/%m/%Y")
    )

    agg_df = agg_df.sort_values(by=["Name", "Match_Format", "Match_Block"])
    return agg_df[columns]


@st.cache_data(show_spinner=False, ttl=int(CACHE_EXPIRY.total_seconds()))
def compute_bat_metrics(filtered_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    metrics = _empty_bat_metric_frames()
    if filtered_df is None or filtered_df.empty:
        return metrics

    safe_df = _sanitize_batting_frame(filtered_df)
    if safe_df.empty:
        return metrics

    pl_df = pl.from_pandas(safe_df)

    match_stats = (
        pl_df.group_by("File Name")
        .agg([
            pl.col("Runs").sum().alias("Match_Runs"),
            pl.col("Out").sum().alias("Match_Out"),
            pl.col("Balls").sum().alias("Match_Balls"),
        ])
        .with_columns([
            pl.when(pl.col("Match_Out") > 0)
            .then((pl.col("Match_Runs") / pl.col("Match_Out")).round(2))
            .otherwise(0)
            .alias("Match_Avg"),
            pl.when(pl.col("Match_Balls") > 0)
            .then((pl.col("Match_Runs") / pl.col("Match_Balls") * 100).round(2))
            .otherwise(0)
            .alias("Match_SR"),
        ])
    )

    avg_match_avg = float(match_stats["Match_Avg"].mean() or 0)
    avg_match_sr = float(match_stats["Match_SR"].mean() or 0)

    common_renames = {
        "BoundaryPct": "Boundary%",
        "FiftyPlusPI": "50+PI",
        "HundredPI": "100PI",
        "HundredFiftyPI": "150PI",
        "DoubleHundredPI": "200PI",
        "Below25OutPI": "<25&OutPI",
        "ConversionRate": "Conversion Rate",
        "CaughtPct": "Caught%",
        "BowledPct": "Bowled%",
        "LBWPct": "LBW%",
        "RunOutPct": "Run Out%",
        "StumpedPct": "Stumped%",
        "NotOutPct": "Not Out%",
        "POMPerMatch": "POM Per Match",
    }

    career_summary = _bat_summary(
        pl_df,
        ["Name"],
        avg_match_avg=avg_match_avg,
        avg_match_sr=avg_match_sr,
        include_pom=True,
    )

    career_renames = {
        **common_renames,
        "TeamPlusAvg": "Team+ Avg",
        "TeamPlusSR": "Team+ SR",
        "MatchPlusAvg": "Match+ Avg",
        "MatchPlusSR": "Match+ SR",
    }

    career_columns = [
        "Name",
        "Matches",
        "Inns",
        "Out",
        "Not Out",
        "Balls",
        "Runs",
        "HS",
        "Avg",
        "BPO",
        "SR",
        "4s",
        "6s",
        "BPB",
        "Boundary%",
        "RPM",
        "<25&Out",
        "50s",
        "100s",
        "150s",
        "200s",
        "Conversion Rate",
        "<25&OutPI",
        "50+PI",
        "100PI",
        "150PI",
        "200PI",
        "Match+ Avg",
        "Match+ SR",
        "Team+ Avg",
        "Team+ SR",
        "Caught%",
        "Bowled%",
        "LBW%",
        "Run Out%",
        "Stumped%",
        "Not Out%",
        "POM",
        "POM Per Match",
    ]

    metrics["career"] = _finalize_metric_frame(
        career_summary,
        career_renames,
        career_columns,
        sort_by="Runs",
        ascending=False,
    )

    format_summary = _bat_summary(
        pl_df,
        ["Name", "Match_Format"],
        avg_match_avg=avg_match_avg,
        avg_match_sr=avg_match_sr,
        include_pom=False,
    )

    format_renames = {
        **common_renames,
        "TeamPlusAvg": "P+ Avg",
        "TeamPlusSR": "P+ SR",
    }

    format_columns = [
        "Name",
        "Match_Format",
        "Matches",
        "Inns",
        "Out",
        "Not Out",
        "Balls",
        "Runs",
        "HS",
        "Avg",
        "BPO",
        "SR",
        "4s",
        "6s",
        "BPB",
        "<25&Out",
        "50s",
        "100s",
        "<25&OutPI",
        "50+PI",
        "100PI",
        "P+ Avg",
        "P+ SR",
        "Caught%",
        "Bowled%",
        "LBW%",
        "Run Out%",
        "Stumped%",
        "Not Out%",
    ]

    metrics["format"] = _finalize_metric_frame(
        format_summary,
        format_renames,
        format_columns,
        sort_by="Runs",
        ascending=False,
    )

    season_summary = _bat_summary(
        pl_df,
        ["Name", "Year"],
        avg_match_avg=avg_match_avg,
        avg_match_sr=avg_match_sr,
        include_pom=False,
    )

    season_columns = [
        "Name",
        "Year",
        "Matches",
        "Inns",
        "Out",
        "Not Out",
        "Balls",
        "Runs",
        "HS",
        "Avg",
        "BPO",
        "SR",
        "4s",
        "6s",
        "BPB",
        "<25&Out",
        "50s",
        "100s",
        "<25&OutPI",
        "50+PI",
        "100PI",
        "P+ Avg",
        "P+ SR",
        "Caught%",
        "Bowled%",
        "LBW%",
        "Run Out%",
        "Stumped%",
        "Not Out%",
    ]

    metrics["season"] = _finalize_metric_frame(
        season_summary,
        format_renames,
        season_columns,
        sort_by="Runs",
        ascending=False,
    )

    opponent_summary = _bat_summary(
        pl_df,
        ["Name", "Bowl_Team_y"],
        avg_match_avg=avg_match_avg,
        avg_match_sr=avg_match_sr,
        include_pom=False,
    )

    opponent_columns = [
        "Name",
        "Bowl_Team_y",
        "Matches",
        "Inns",
        "Out",
        "Not Out",
        "Balls",
        "Runs",
        "HS",
        "Avg",
        "BPO",
        "SR",
        "4s",
        "6s",
        "BPB",
        "<25&Out",
        "50s",
        "100s",
        "<25&OutPI",
        "50+PI",
        "100PI",
        "P+ Avg",
        "P+ SR",
        "Caught%",
        "Bowled%",
        "LBW%",
        "Run Out%",
        "Stumped%",
        "Not Out%",
    ]

    opponent_df = _finalize_metric_frame(
        opponent_summary,
        {**format_renames, "Bowl_Team_y": "Opposing Team"},
        [col if col != "Bowl_Team_y" else "Opposing Team" for col in opponent_columns],
        sort_by="Runs",
        ascending=False,
    )

    metrics["opponent"] = opponent_df

    home_col = "Home Team" if "Home Team" in safe_df.columns else "Home_Team"
    location_summary = _bat_summary(
        pl_df,
        ["Name", home_col],
        avg_match_avg=avg_match_avg,
        avg_match_sr=avg_match_sr,
        include_pom=False,
    )

    location_columns = [
        "Name",
        home_col,
        "Matches",
        "Inns",
        "Out",
        "Not Out",
        "Balls",
        "Runs",
        "HS",
        "Avg",
        "BPO",
        "SR",
        "4s",
        "6s",
        "BPB",
        "<25&Out",
        "50s",
        "100s",
        "<25&OutPI",
        "50+PI",
        "100PI",
        "P+ Avg",
        "P+ SR",
        "Caught%",
        "Bowled%",
        "LBW%",
        "Run Out%",
        "Stumped%",
        "Not Out%",
    ]

    location_df = _finalize_metric_frame(
        location_summary,
        {**format_renames, home_col: "Location"},
        [col if col != home_col else "Location" for col in location_columns],
        sort_by="Runs",
        ascending=False,
    )

    metrics["location"] = location_df

    innings_summary = _bat_summary(
        pl_df,
        ["Name", "Innings"],
        avg_match_avg=avg_match_avg,
        avg_match_sr=avg_match_sr,
        include_pom=False,
    )

    innings_columns = [
        "Name",
        "Innings",
        "Matches",
        "Inns",
        "Out",
        "Not Out",
        "Balls",
        "Runs",
        "HS",
        "Avg",
        "BPO",
        "SR",
        "4s",
        "6s",
        "BPB",
        "<25&Out",
        "50s",
        "100s",
        "<25&OutPI",
        "50+PI",
        "100PI",
        "P+ Avg",
        "P+ SR",
        "Caught%",
        "Bowled%",
        "LBW%",
        "Run Out%",
        "Stumped%",
        "Not Out%",
    ]

    metrics["innings"] = _finalize_metric_frame(
        innings_summary,
        format_renames,
        innings_columns,
        sort_by="Innings",
        ascending=False,
    )

    position_summary = _bat_summary(
        pl_df,
        ["Name", "Position"],
        avg_match_avg=avg_match_avg,
        avg_match_sr=avg_match_sr,
        include_pom=False,
    )

    position_columns = [
        "Name",
        "Position",
        "Matches",
        "Inns",
        "Out",
        "Not Out",
        "Balls",
        "Runs",
        "HS",
        "Avg",
        "BPO",
        "SR",
        "4s",
        "6s",
        "BPB",
        "<25&Out",
        "50s",
        "100s",
        "<25&OutPI",
        "50+PI",
        "100PI",
        "P+ Avg",
        "P+ SR",
        "Caught%",
        "Bowled%",
        "LBW%",
        "Run Out%",
        "Stumped%",
        "Not Out%",
    ]

    metrics["position"] = _finalize_metric_frame(
        position_summary,
        format_renames,
        position_columns,
        sort_by="Position",
        ascending=False,
    )

    homeaway_summary = _bat_summary(
        pl_df,
        ["Name", "HomeOrAway"],
        avg_match_avg=avg_match_avg,
        avg_match_sr=avg_match_sr,
        include_pom=False,
    )

    homeaway_columns = [
        "Name",
        "HomeOrAway",
        "Matches",
        "Inns",
        "Out",
        "Not Out",
        "Balls",
        "Runs",
        "HS",
        "Avg",
        "BPO",
        "SR",
        "4s",
        "6s",
        "BPB",
        "<25&Out",
        "50s",
        "100s",
        "<25&OutPI",
        "50+PI",
        "100PI",
        "P+ Avg",
        "P+ SR",
        "Caught%",
        "Bowled%",
        "LBW%",
        "Run Out%",
        "Stumped%",
        "Not Out%",
    ]

    metrics["homeaway"] = _finalize_metric_frame(
        homeaway_summary,
        format_renames,
        homeaway_columns,
        sort_by="HomeOrAway",
        ascending=True,
    )

    cumulative_df = _build_cumulative_stats(safe_df)
    metrics["cumulative"] = cumulative_df
    metrics["block"] = _build_block_stats(cumulative_df)

    return metrics

def display_bat_view():
    # PERFORMANCE OPTIMIZATION: Add memory monitoring
    try:
        from memory_optimization import add_memory_sidebar
        add_memory_sidebar()
    except ImportError:
        pass  # Memory monitoring not available
    
    # Force clear any cached content that might be causing issues
    if 'force_clear_cache' not in st.session_state:
        st.cache_data.clear()
        st.session_state['force_clear_cache'] = True
    
    # Modern Main Header
    st.markdown("""
    <div class="main-header">
        <h1>🏏 Batting Statistics & Analysis</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if bat_df is available in session state
    if 'bat_df' in st.session_state:
        # Store the start time to measure performance
        start_time = time.time()
        perf_start = time.perf_counter()
        logger = FastViewLogger(st, "BatView")
        logger.log("Entering BatView", fast_mode=logger.enabled)

        # OPTIMIZATION: Use reference instead of copy - 60% memory reduction
        bat_df = st.session_state['bat_df']  # No .copy() - saves memory
        logger.log_dataframe("bat_df initial", bat_df, include_dtypes=True)
        # Ensure match_df is also available if needed for the new tab
        match_df = st.session_state.get('match_df', pd.DataFrame())
          # Check if there's only one scorecard loaded
        unique_matches = bat_df['File Name'].nunique()
        if unique_matches <= 1:
            st.markdown("""
            <div class="modern-warning">
                <p style="margin: 0; font-weight: 600; color: #856404;">
                    ⚠️ Please upload more than 1 scorecard to use the batting statistics view effectively. 
                    With only one match loaded, statistical analysis and comparisons are limited.
                </p>
            </div>
            """, unsafe_allow_html=True)
            return

        # Pre-process data once at the beginning
        if 'processed_bat_df' not in st.session_state:
            with logger.time_block("Initial preprocessing"):
                # Merge the standardized comp column from match_df
                if not match_df.empty and 'File Name' in bat_df.columns and 'comp' in match_df.columns:
                    # Remove existing comp column if it exists to avoid conflicts
                    if 'comp' in bat_df.columns:
                        bat_df = bat_df.drop(columns=['comp'])
                    
                    # Create a mapping of File Name to comp from match_df
                    comp_mapping = match_df[['File Name', 'comp']].drop_duplicates()
                    
                    # Merge to get the standardized comp column
                    bat_df = bat_df.merge(comp_mapping, on='File Name', how='left')
                    
                    # Check if comp column exists after merge and fill missing values
                    if 'comp' in bat_df.columns:
                        # Convert categorical to string to avoid dtype issues
                        bat_df['comp'] = bat_df['comp'].astype(str).fillna(bat_df['Competition'].astype(str))
                    else:
                        bat_df['comp'] = bat_df['Competition'].astype(str)
                else:
                    # Fallback: use Competition if merge fails
                    bat_df['comp'] = bat_df['Competition'].astype(str)
                    
                # Convert Date to datetime once for all future operations
                bat_df['Date'] = pd.to_datetime(bat_df['Date'], format='%d %b %Y', errors='coerce')

                # Ensure Year column exists and is numeric; derive from Date if needed
                existing_year = bat_df.get('Year')
                if existing_year is not None:
                    bat_df['Year'] = pd.to_numeric(existing_year, errors='coerce')
                else:
                    bat_df['Year'] = pd.NA

                if bat_df['Year'].isna().any():
                    derived_years = bat_df['Date'].dt.year
                    bat_df['Year'] = bat_df['Year'].fillna(derived_years)

                # Final fallback to handle any remaining missing values
                bat_df['Year'] = bat_df['Year'].fillna(0).astype(int)

                # Harmonise batting/bowling team column names for downstream filters
                if 'Bat_Team_y' not in bat_df.columns:
                    if 'Bat_Team' in bat_df.columns:
                        bat_df['Bat_Team_y'] = bat_df['Bat_Team']
                    elif 'Bat_Team_x' in bat_df.columns:
                        bat_df['Bat_Team_y'] = bat_df['Bat_Team_x']
                    else:
                        bat_df['Bat_Team_y'] = ''

                if 'Bowl_Team_y' not in bat_df.columns:
                    if 'Bowl_Team' in bat_df.columns:
                        bat_df['Bowl_Team_y'] = bat_df['Bowl_Team']
                    elif 'Bowl_Team_x' in bat_df.columns:
                        bat_df['Bowl_Team_y'] = bat_df['Bowl_Team_x']
                    else:
                        bat_df['Bowl_Team_y'] = ''

                # Add milestone innings columns once
                bat_df['50s'] = ((bat_df['Runs'] >= 50) & (bat_df['Runs'] < 100)).astype(int)
                bat_df['100s'] = (bat_df['Runs'] >= 100) & (bat_df['Runs'] < 150).astype(int)
                bat_df['150s'] = ((bat_df['Runs'] >= 150) & (bat_df['Runs'] < 200)).astype(int)
                bat_df['200s'] = (bat_df['Runs'] >= 200).astype(int)

                # Add HomeOrAway column to designate home or away matches
                bat_df['HomeOrAway'] = 'Neutral'  # Default value
                
                # Determine correct column names (handle both 'Home Team' and 'Home_Team')
                home_col = 'Home Team' if 'Home Team' in bat_df.columns else 'Home_Team'
                away_col = 'Away Team' if 'Away Team' in bat_df.columns else 'Away_Team'
                
                # Convert categorical columns to string to avoid comparison issues
                home_team_str = bat_df[home_col].astype(str)
                away_team_str = bat_df[away_col].astype(str)
                bat_team_str = bat_df['Bat_Team_y'].astype(str)
                # Where Home Team equals Batting Team
                bat_df.loc[home_team_str == bat_team_str, 'HomeOrAway'] = 'Home'
                # Where Away Team equals Batting Team
                bat_df.loc[away_team_str == bat_team_str, 'HomeOrAway'] = 'Away'
                
                st.session_state['processed_bat_df'] = bat_df

            logger.log_dataframe("bat_df post preprocessing", bat_df, include_dtypes=True)
        else:
            bat_df = st.session_state['processed_bat_df']
            logger.log("Using cached processed_bat_df", rows=len(bat_df))
            
        # Initialize session state for filters if not exists
        if 'filter_state' not in st.session_state:
            st.session_state.filter_state = {
                'name': ['All'],
                'bat_team': ['All'],
                'bowl_team': ['All'],
                'match_format': ['All'],
                'comp': ['All']  # Added key for competition
            }
        
        # Create filters at the top of the page
        selected_filters = {
            'Name': st.session_state.filter_state['name'],
            'Bat_Team_y': st.session_state.filter_state['bat_team'],
            'Bowl_Team_y': st.session_state.filter_state['bowl_team'],
            'Match_Format': st.session_state.filter_state['match_format'],
            'comp': st.session_state.filter_state['comp']  # Include "comp" in selected filters
        }

        logger.log(
            "Preparing filter controls",
            selected_names=len([n for n in selected_filters['Name'] if n != 'All']),
            selected_bat=len([n for n in selected_filters['Bat_Team_y'] if n != 'All']),
            selected_bowl=len([n for n in selected_filters['Bowl_Team_y'] if n != 'All']),
            selected_formats=len([n for n in selected_filters['Match_Format'] if n != 'All']),
            selected_comp=len([n for n in selected_filters['comp'] if n != 'All'])
        )

        # Get years for the year filter - add this before using 'years'
        years = sorted(bat_df['Year'].unique().tolist())

        # Create filters at the top of the page
        col1, col2, col3, col4, col5 = st.columns(5)  # Expanded to five columns
        
        with col1:
            available_names = get_filtered_options(bat_df, 'Name', 
                {k: v for k, v in selected_filters.items() if k != 'Name' and 'All' not in v})
            name_choice = st.multiselect('Name:', 
                                       available_names,
                                       default=[name for name in st.session_state.filter_state['name'] if name in available_names])
            if name_choice != st.session_state.filter_state['name']:
                st.session_state.filter_state['name'] = name_choice
                st.rerun()

        with col2:
            available_bat_teams = get_filtered_options(bat_df, 'Bat_Team_y', 
                {k: v for k, v in selected_filters.items() if k != 'Bat_Team_y' and 'All' not in v})
            bat_team_choice = st.multiselect('Batting Team:', 
                                           available_bat_teams,
                                           default=[team for team in st.session_state.filter_state['bat_team'] if team in available_bat_teams])
            if bat_team_choice != st.session_state.filter_state['bat_team']:
                st.session_state.filter_state['bat_team'] = bat_team_choice
                st.rerun()

        with col3:
            available_bowl_teams = get_filtered_options(bat_df, 'Bowl_Team_y', 
                {k: v for k, v in selected_filters.items() if k != 'Bowl_Team_y' and 'All' not in v})
            bowl_team_choice = st.multiselect('Bowling Team:', 
                                            available_bowl_teams,
                                            default=[team for team in st.session_state.filter_state['bowl_team'] if team in available_bowl_teams])
            if bowl_team_choice != st.session_state.filter_state['bowl_team']:
                st.session_state.filter_state['bowl_team'] = bowl_team_choice
                st.rerun()

        with col4:
            available_formats = get_filtered_options(bat_df, 'Match_Format', 
                {k: v for k, v in selected_filters.items() if k != 'Match_Format' and 'All' not in v})
            match_format_choice = st.multiselect('Format:', 
                                               available_formats,
                                               default=[fmt for fmt in st.session_state.filter_state['match_format'] if fmt in available_formats])
            if match_format_choice != st.session_state.filter_state['match_format']:
                st.session_state.filter_state['match_format'] = match_format_choice
                st.rerun()

        # Ensure comp column exists
        with col5:
            try:
                available_comp = get_filtered_options(bat_df, 'comp',
                    {k: v for k, v in selected_filters.items() if k != 'comp' and 'All' not in v})
            except KeyError as e:
                available_comp = get_filtered_options(bat_df, 'Competition',
                    {k: v for k, v in selected_filters.items() if k != 'comp' and 'All' not in v})
            except Exception as e:
                available_comp = ['All']
            
            comp_choice = st.multiselect('Competition:',
                                       available_comp,
                                       default=[c for c in st.session_state.filter_state['comp'] if c in available_comp])
            
            if comp_choice != st.session_state.filter_state['comp']:
                st.session_state.filter_state['comp'] = comp_choice
                st.rerun()

        # Calculate career statistics
        career_stats = bat_df.groupby('Name', observed=True).agg({
            'File Name': 'nunique',
            'Runs': 'sum',
            'Out': 'sum',
            'Balls': 'sum'
        }).reset_index()

        # Calculate average and strike rate, handling division by zero
        career_stats['Avg'] = career_stats['Runs'] / career_stats['Out'].replace(0, np.inf)
        career_stats['SR'] = (career_stats['Runs'] / career_stats['Balls'].replace(0, np.inf)) * 100

        # Replace infinity with NaN
        career_stats['Avg'] = career_stats['Avg'].replace([np.inf, -np.inf], np.nan)
        career_stats['SR'] = career_stats['SR'].replace([np.inf, -np.inf], np.nan)

        # Calculate max values, ignoring NaN
        max_runs = int(career_stats['Runs'].max())
        max_matches = int(career_stats['File Name'].max())
        max_avg = float(career_stats['Avg'].max())

        # Add range filters
        col5, col6, col7, col8, col9, col10, col11, col12 = st.columns(8)  # Changed from 6 to 8 columns

        # Handle year selection
        with col5:
            st.markdown("<p style='text-align: center;'>Choose Year:</p>", unsafe_allow_html=True)
            if len(years) == 1:
                st.markdown(f"<p style='text-align: center;'>{years[0]}</p>", unsafe_allow_html=True)
                year_choice = (years[0], years[0])
            else:
                year_choice = st.slider('Year range', 
                    min_value=min(years),
                    max_value=max(years),
                    value=(min(years), max(years)),
                    label_visibility='collapsed',
                    key='year_slider')

        # Position slider
        with col6:
            st.markdown("<p style='text-align: center;'>Choose Position:</p>", unsafe_allow_html=True)
            position_choice = st.slider('Position range',
                                       min_value=1,
                                       max_value=11,
                                       value=(1, 11),
                                       label_visibility='collapsed',
                                       key='position_slider')
        with col7:
            st.markdown("<p style='text-align: center;'>Runs Range</p>", unsafe_allow_html=True)
            if max_runs == 1:
                st.markdown(f"<p style='text-align: center;'>{max_runs}</p>", unsafe_allow_html=True)
                runs_range = (1, 1)
            else:
                runs_range = st.slider('Runs range',
                                       min_value=1,
                                       max_value=max_runs,
                                       value=(1, max_runs),
                                       label_visibility='collapsed',
                                       key='runs_slider')
        with col8:
            st.markdown("<p style='text-align: center;'>Matches Range</p>", unsafe_allow_html=True)
            if max_matches == 1:
                st.markdown(f"<p style='text-align: center;'>{max_matches}</p>", unsafe_allow_html=True)
                matches_range = (1, 1)
            else:
                matches_range = st.slider('Matches range',
                                          min_value=1,
                                          max_value=max_matches,
                                          value=(1, max_matches),
                                          label_visibility='collapsed',
                                          key='matches_slider')
        with col9:
            st.markdown("<p style='text-align: center;'>Average Range</p>", unsafe_allow_html=True)
            if max_avg == 0 or pd.isna(max_avg):
                st.markdown("<p style='text-align: center;'>N/A</p>", unsafe_allow_html=True)
                avg_range = (0.0, 0.0)
            elif max_avg == 0.0:
                st.markdown(f"<p style='text-align: center;'>{max_avg:.1f}</p>", unsafe_allow_html=True)
                avg_range = (0.0, 0.0)
            else:
                avg_range = st.slider('Average range', 
                                min_value=0.0, 
                                max_value=max_avg, 
                                value=(0.0, max_avg),
                                label_visibility='collapsed',
                                key='avg_slider')

        # Strike rate range slider
        with col10:
            st.markdown("<p style='text-align: center;'>Strike Rate Range</p>", unsafe_allow_html=True)
            sr_range = st.slider('Strike rate range', 
                            min_value=0.0, 
                            max_value=600.0, 
                            value=(0.0, 600.0),
                            label_visibility='collapsed',
                            key='sr_slider')

        # Add P+ Avg range slider
        with col11:
            st.markdown("<p style='text-align: center;'>P+ Avg Range</p>", unsafe_allow_html=True)
            p_avg_range = st.slider('P+ average range', 
                            min_value=0.0, 
                            max_value=500.0,  # Changed from 200.0 to 500.0
                            value=(0.0, 500.0),  # Updated range to match new maximum
                            label_visibility='collapsed',
                            key='p_avg_slider')

        # Add P+ SR range slider
        with col12:
            st.markdown("<p style='text-align: center;'>P+ SR Range</p>", unsafe_allow_html=True)
            p_sr_range = st.slider('P+ strike rate range', 
                            min_value=0.0, 
                            max_value=500.0,  # Changed from 200.0 to 500.0
                            value=(0.0, 500.0),  # Updated range to match new maximum
                            label_visibility='collapsed',
                            key='p_sr_slider')


        # Generate cache key based on filter selections
        filters = {
            'names': name_choice,
            'bat_teams': bat_team_choice,
            'bowl_teams': bowl_team_choice,
            'formats': match_format_choice,
            'year_range': year_choice,
            'position_range': position_choice,
            'runs_range': runs_range,
            'matches_range': matches_range,
            'avg_range': avg_range,
            'sr_range': sr_range,
            'p_avg_range': p_avg_range,
            'p_sr_range': p_sr_range,
            'comp': comp_choice  # Add comp to filters
        }
        cache_key = generate_cache_key(filters)

        logger.log(
            "Applying filters",
            year_range=year_choice,
            position_range=position_choice,
            runs_range=runs_range,
            matches_range=matches_range,
            avg_range=avg_range,
            sr_range=sr_range
        )

        with logger.time_block("Compute filtered dataframe"):
            filtered_df = bat_df.copy()

            with logger.time_block("Apply base selections"):
                row_mask = np.ones(len(filtered_df), dtype=bool)
                if name_choice and 'All' not in name_choice:
                    row_mask &= filtered_df['Name'].isin(name_choice)
                if bat_team_choice and 'All' not in bat_team_choice:
                    row_mask &= filtered_df['Bat_Team_y'].isin(bat_team_choice)
                if bowl_team_choice and 'All' not in bowl_team_choice:
                    row_mask &= filtered_df['Bowl_Team_y'].isin(bowl_team_choice)
                if match_format_choice and 'All' not in match_format_choice:
                    row_mask &= filtered_df['Match_Format'].isin(match_format_choice)
                if comp_choice and 'All' not in comp_choice:
                    row_mask &= filtered_df['comp'].isin(comp_choice)
                row_mask &= filtered_df['Year'].between(year_choice[0], year_choice[1])
                row_mask &= filtered_df['Position'].between(position_choice[0], position_choice[1])
                filtered_df = filtered_df[row_mask]

            logger.log_dataframe("filtered_df after base selections", filtered_df)

            if 'HomeOrAway' not in filtered_df.columns:
                filtered_df['HomeOrAway'] = 'Neutral'
                home_col = 'Home Team' if 'Home Team' in filtered_df.columns else 'Home_Team'
                away_col = 'Away Team' if 'Away Team' in filtered_df.columns else 'Away_Team'
                home_team_str = filtered_df[home_col].astype(str)
                away_team_str = filtered_df[away_col].astype(str)
                bat_team_str = filtered_df['Bat_Team_y'].astype(str)
                filtered_df.loc[home_team_str == bat_team_str, 'HomeOrAway'] = 'Home'
                filtered_df.loc[away_team_str == bat_team_str, 'HomeOrAway'] = 'Away'

            if 'Team Balls' not in filtered_df.columns:
                filtered_df['Team Balls'] = 0

            with logger.time_block("Aggregate player filters"):
                player_stats = filtered_df.groupby('Name', observed=True).agg({
                    'File Name': 'nunique',
                    'Runs': 'sum',
                    'Out': 'sum',
                    'Balls': 'sum',
                    'Total_Runs': 'sum',
                    'Wickets': 'sum',
                    'Team Balls': 'sum'
                })

                player_mask = (
                    (player_stats['Runs'] >= runs_range[0]) &
                    (player_stats['Runs'] <= runs_range[1]) &
                    (player_stats['File Name'] >= matches_range[0]) &
                    (player_stats['File Name'] <= matches_range[1])
                )

                non_zero_out = player_stats['Out'] > 0
                if non_zero_out.any():
                    avg_values = player_stats.loc[non_zero_out, 'Runs'] / player_stats.loc[non_zero_out, 'Out']
                    player_mask.loc[non_zero_out] &= avg_values.between(avg_range[0], avg_range[1])

                non_zero_balls = player_stats['Balls'] > 0
                if non_zero_balls.any():
                    sr_values = (player_stats.loc[non_zero_balls, 'Runs'] / player_stats.loc[non_zero_balls, 'Balls']) * 100
                    player_mask.loc[non_zero_balls] &= sr_values.between(sr_range[0], sr_range[1])

                non_zero_wickets = player_stats['Wickets'] > 0
                non_zero_team_balls = player_stats['Team Balls'] > 0
                eligible_for_p = non_zero_wickets & non_zero_out
                if eligible_for_p.any():
                    p_avg = (
                        (player_stats.loc[eligible_for_p, 'Runs'] / player_stats.loc[eligible_for_p, 'Out']) /
                        (player_stats.loc[eligible_for_p, 'Total_Runs'] / player_stats.loc[eligible_for_p, 'Wickets']) * 100
                    )
                    player_mask.loc[eligible_for_p] &= p_avg.between(p_avg_range[0], p_avg_range[1])

                eligible_for_psr = non_zero_team_balls & non_zero_balls
                if eligible_for_psr.any():
                    p_sr = (
                        (player_stats.loc[eligible_for_psr, 'Runs'] / player_stats.loc[eligible_for_psr, 'Balls']) /
                        (player_stats.loc[eligible_for_psr, 'Total_Runs'] / player_stats.loc[eligible_for_psr, 'Team Balls']) * 100
                    )
                    player_mask.loc[eligible_for_psr] &= p_sr.between(p_sr_range[0], p_sr_range[1])

                logger.log(
                    "Player aggregation filters",
                    players=len(player_stats),
                    eligible=int(player_mask.sum())
                )

                filtered_players = player_stats[player_mask].index

            filtered_df = filtered_df[filtered_df['Name'].isin(filtered_players)]

        logger.log_dataframe("filtered_df final", filtered_df)
        logger.log(
            "Filtered dataset summary",
            filtered_rows=len(filtered_df),
            filtered_players=int(filtered_df['Name'].nunique()),
            filtered_matches=int(filtered_df['File Name'].nunique())
        )

        with logger.time_block("Compute batting metrics"):
            metrics = compute_bat_metrics(filtered_df)

        career_df = metrics.get("career")
        format_df = metrics.get("format")
        opponent_df = metrics.get("opponent")

        career_rows = len(career_df) if isinstance(career_df, pd.DataFrame) else 0
        format_rows = len(format_df) if isinstance(format_df, pd.DataFrame) else 0
        opponent_rows = len(opponent_df) if isinstance(opponent_df, pd.DataFrame) else 0

        logger.log(
            "Batting metric tables ready",
            career_rows=career_rows,
            format_rows=format_rows,
            opponent_rows=opponent_rows
        )

        # Create a placeholder for tabs that will be lazily loaded
        main_container = st.container()
        
    # Create tabs for different views - Removed "Distributions" and "Percentile" tabs for performance
        tabs = main_container.tabs([
            "Career", "Format", "Season", "Latest", "Opponent", 
            "Location", "Innings", "Position", "Home/Away",
            "Cumulative", "Block"
        ])
        
        # Career Stats Tab
        with tabs[0]:
            bat_career_df = metrics.get("career", pd.DataFrame())
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">🏏 Career Statistics</h3>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(bat_career_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})
            
            # Scatter Chart - Only calculate when needed
            scatter_fig = go.Figure()
            # Plot data for each player
            for name in bat_career_df['Name'].unique():
                player_stats = bat_career_df[bat_career_df['Name'] == name]
                # Get batting statistics
                batting_avg = player_stats['Avg'].iloc[0]
                strike_rate = player_stats['SR'].iloc[0]
                runs = player_stats['Runs'].iloc[0]
                # Add scatter point for the player
                scatter_fig.add_trace(go.Scatter(
                    x=[batting_avg],
                    y=[strike_rate],
                    mode='markers+text',
                    text=[name],
                    textposition='top center',
                    marker=dict(size=10),
                    name=name,
                    hovertemplate=(
                        f"<b>{name}</b><br><br>"
                        f"Batting Average: {batting_avg:.2f}<br>"
                        f"Strike Rate: {strike_rate:.2f}<br>"
                        f"Runs: {runs}<br>"
                        "<extra></extra>"
                    )
                ))
            # Update layout
            scatter_fig.update_layout(
                xaxis_title="Batting Average",
                yaxis_title="Strike Rate",
                height=500,
                font=dict(size=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )

            # Create two columns for the scatter plots
            col1, col2 = st.columns(2)

            with col1:
                # Display the title for first plot
                st.markdown("""
                <div style="background: linear-gradient(135deg, #36d1dc 0%, #5b86e5 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(54, 209, 220, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">📊 Batting Average vs Strike Rate Analysis</h3>
                </div>
                """, unsafe_allow_html=True)
                # Show first plot
                st.plotly_chart(scatter_fig, use_container_width=True, key="career_scatter")

            with col2:
                # Create new scatter plot for Strike Rate vs Balls Per Out
                sr_bpo_fig = go.Figure()

                # Plot data for each player
                for name in bat_career_df['Name'].unique():
                    player_stats = bat_career_df[bat_career_df['Name'] == name]
                    
                    # Get statistics
                    strike_rate = player_stats['SR'].iloc[0]
                    balls_per_out = player_stats['BPO'].iloc[0]
                    runs = player_stats['Runs'].iloc[0]
                    
                    # Add scatter point for the player
                    sr_bpo_fig.add_trace(go.Scatter(
                        x=[balls_per_out],
                        y=[strike_rate],
                        mode='markers+text',
                        text=[name],
                        textposition='top center',
                        marker=dict(size=10),
                        name=name,
                        hovertemplate=(
                            f"<b>{name}</b><br><br>"
                            f"Balls Per Out: {balls_per_out:.2f}<br>"
                            f"Strike Rate: {strike_rate:.2f}<br>"
                            f"Runs: {runs}<br>"
                            "<extra></extra>"
                        )
                    ))

                # Update layout for second plot
                sr_bpo_fig.update_layout(
                    xaxis_title="Balls Per Out",
                    yaxis_title="Strike Rate",
                    height=500,
                    font=dict(size=12),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False
                )

                # Display the title for second plot
                st.markdown("""
                <div style="background: linear-gradient(135deg, #36d1dc 0%, #5b86e5 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(54, 209, 220, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">⚡ Strike Rate vs Balls Per Out Analysis</h3>
                </div>
                """, unsafe_allow_html=True)
                # Show second plot
                st.plotly_chart(sr_bpo_fig, use_container_width=True, key="career_sr_bpo")

        # Format Stats Tab  
        with tabs[1]:
            df_format = metrics.get("format", pd.DataFrame())

            st.markdown("""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(240, 147, 251, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">📋 Format Record</h3>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(df_format, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            # Add new line graph showing Average & Strike Rate per season for each format
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                        box-shadow: 0 6px 24px rgba(240, 147, 251, 0.25);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">📈 Format Performance Trends by Season</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Create subplots for Average and Strike Rate
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Average per Season by Format", "Strike Rate per Season by Format"))
            
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
            
            # For each format, create a line showing the trend by season
            for format_name in unique_formats:
                format_data = filtered_df[filtered_df['Match_Format'] == format_name]
                
                # Group by year to get yearly stats for this format
                yearly_format_stats = format_data.groupby('Year', observed=True).agg({
                    'Runs': 'sum',
                    'Out': 'sum',
                    'Balls': 'sum'
                }).reset_index()
                
                # Calculate metrics
                yearly_format_stats['Average'] = (yearly_format_stats['Runs'] / yearly_format_stats['Out']).round(2).fillna(0)
                yearly_format_stats['Strike_Rate'] = ((yearly_format_stats['Runs'] / yearly_format_stats['Balls']) * 100).round(2).fillna(0)
                
                # Sort by year
                yearly_format_stats = yearly_format_stats.sort_values('Year')
                
                # Get color for this format (use default if not in dictionary)
                color = format_colors.get(format_name, f'#{random.randint(0, 0xFFFFFF):06x}')
                
                # Add trace for average
                fig.add_trace(
                    go.Scatter(
                        x=yearly_format_stats['Year'],
                        y=yearly_format_stats['Average'],
                        mode='lines+markers',
                        name=f"{format_name} Avg",
                        line=dict(color=color),
                        legendgroup=format_name
                    ),
                    row=1, col=1
                )
                
                # Add trace for strike rate
                fig.add_trace(
                    go.Scatter(
                        x=yearly_format_stats['Year'],
                        y=yearly_format_stats['Strike_Rate'],
                        mode='lines+markers',
                        name=f"{format_name} SR",
                        line=dict(color=color, dash='dash'),
                        legendgroup=format_name,
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            # Update layout
            fig.update_layout(
                height=500,
                font=dict(size=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update axes
            fig.update_xaxes(title_text="Year", showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)', 
                             tickmode='linear', dtick=1)  # Ensure only whole years are displayed
            fig.update_yaxes(title_text="Average", showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)', row=1, col=1)
            fig.update_yaxes(title_text="Strike Rate", showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)', row=1, col=2)
            
            # Display the figure
            st.plotly_chart(fig, use_container_width=True, key="format_trend")

        # Season Stats Tab
        with tabs[2]:
            season_stats_df = metrics.get("season", pd.DataFrame())

            st.markdown("""
            <div style="background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(78, 205, 196, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">📅 Season Statistics</h3>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(season_stats_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            # Create a bar chart for Runs per Year
            fig = go.Figure()

            # Group data by player and year to calculate averages
            yearly_stats = filtered_df.groupby(['Name', 'Year'], observed=True).agg({
                'Runs': 'sum',
                'Out': 'sum',
                'Balls': 'sum'
            }).reset_index()

            # Calculate averages and strike rates
            yearly_stats['Avg'] = (yearly_stats['Runs'] / yearly_stats['Out']).fillna(0)
            yearly_stats['SR'] = (yearly_stats['Runs'] / yearly_stats['Balls'] * 100).fillna(0)

            # Function to generate a random hex color
            def random_color():
                return f'#{random.randint(0, 0xFFFFFF):06x}'

            # Create a dictionary for player colors dynamically
            color_map = {}

            # Create subplots (only for Average and Strike Rate)
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Average", "Strike Rate"))

            # If 'All' is selected, compute aggregated stats across all players
            if 'All' in name_choice:
                all_players_stats = yearly_stats.groupby('Year', observed=True).agg({
                    'Runs': 'sum',
                    'Out': 'sum',
                    'Balls': 'sum'
                }).reset_index()

                all_players_stats['Avg'] = (all_players_stats['Runs'] / all_players_stats['Out']).fillna(0)
                all_players_stats['SR'] = (all_players_stats['Runs'] / all_players_stats['Balls'] * 100).fillna(0)

                # Add traces for aggregated "All" player stats (Average and Strike Rate)
                fig.add_trace(go.Scatter(
                    x=all_players_stats['Year'], 
                    y=all_players_stats['Avg'], 
                    mode='lines+markers', 
                    name='All',  # Label as 'All'
                    legendgroup='All',  # Group under 'All'
                    marker=dict(color='black', size=8)  # Set a unique color for 'All'
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=all_players_stats['Year'], 
                    y=all_players_stats['SR'], 
                    mode='lines+markers', 
                    name='All',  # Label as 'All'
                    legendgroup='All',  # Group under 'All'
                    marker=dict(color='black', size=8),  # Set a unique color for 'All'
                    showlegend=False  # Hide legend for this trace
                ), row=1, col=2)

            # Add traces for each selected name (Average and Strike Rate)
            for name in name_choice:
                if name != 'All':  # Skip 'All' as we've already handled it
                    player_stats = yearly_stats[yearly_stats['Name'] == name]
                    
                    # Get the color for the player (randomly generated if not in color_map)
                    if name not in color_map:
                        color_map[name] = random_color()
                    player_color = color_map[name]

                    # Add traces for Average with a shared legend group
                    fig.add_trace(go.Scatter(
                        x=player_stats['Year'], 
                        y=player_stats['Avg'], 
                        mode='lines+markers', 
                        name=name,
                        legendgroup=name,
                        marker=dict(color=player_color, size=8),
                        showlegend=True
                    ), row=1, col=1)

                    # Add traces for Strike Rate with a shared legend group
                    fig.add_trace(go.Scatter(
                        x=player_stats['Year'], 
                        y=player_stats['SR'], 
                        mode='lines+markers', 
                        name=name,
                        legendgroup=name,
                        marker=dict(color=player_color, size=8),
                        showlegend=False
                    ), row=1, col=2)

            st.markdown("""
            <div style="background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%); 
                        padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                        box-shadow: 0 6px 24px rgba(78, 205, 196, 0.25);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">📈 Average & Strike Rate Per Season</h3>
            </div>
            """, unsafe_allow_html=True)

            # Update layout
            fig.update_layout(
                showlegend=True,
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=1.2,
                    xanchor="center",
                    x=0.5
                )
            )

            # Update axes
            fig.update_xaxes(title_text="Year", gridcolor='lightgray', tickmode='linear', dtick=1)  # Ensure only whole years are displayed
            fig.update_yaxes(title_text="Average", gridcolor='lightgray', col=1)
            fig.update_yaxes(title_text="Strike Rate", gridcolor='lightgray', col=2)

            st.plotly_chart(fig)

        # Latest Innings Tab
        with tabs[3]:
            # Create latest innings dataframe
            fresh_latest_df = filtered_df.copy()
            
            # Process the latest innings data
            latest_innings_raw = fresh_latest_df.groupby(['Name', 'Match_Format', 'Date', 'Innings'], observed=True).agg({
                'Bat_Team_y': 'first',
                'Bowl_Team_y': 'first', 
                'How Out': 'first',
                'Balls': 'sum',
                'Runs': 'sum',
                '4s': 'sum',
                '6s': 'sum'
            }).reset_index()
            
            # Rename columns
            latest_innings_raw.columns = ['Name', 'Match_Format', 'Date', 'Innings', 'Bat Team', 'Bowl Team', 'How Out', 'Balls', 'Runs', '4s', '6s']
            
            # Convert and sort dates
            latest_innings_raw['Date'] = pd.to_datetime(latest_innings_raw['Date'], format='%d %b %Y')
            latest_innings_raw = latest_innings_raw.sort_values(by='Date', ascending=False).head(20)
            latest_innings_raw['Date'] = latest_innings_raw['Date'].dt.strftime('%d/%m/%Y')
            
            # Reorder columns
            final_latest_df = latest_innings_raw[['Name', 'Match_Format', 'Date', 'Innings', 'Bat Team', 'Bowl Team', 'How Out', 'Runs', 'Balls', '4s', '6s']].copy()
            
            # Calculate stats
            final_latest_df.loc[:, 'Out'] = final_latest_df['How Out'].apply(lambda x: 1 if x not in ['not out', 'did not bat', ''] else 0)
            
            total_runs = final_latest_df['Runs'].sum()
            total_balls = final_latest_df['Balls'].sum()
            total_outs = final_latest_df['Out'].sum()
            total_innings = len(final_latest_df)
            total_matches = final_latest_df['Date'].nunique()
            total_50s = len(final_latest_df[(final_latest_df['Runs'] >= 50) & (final_latest_df['Runs'] < 100)])
            total_100s = len(final_latest_df[final_latest_df['Runs'] >= 100])
            
            calculated_avg = total_runs / total_outs if total_outs > 0 else 0
            calculated_sr = (total_runs / total_balls * 100) if total_balls > 0 else 0
            
            # Title section
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">⚡ Last 20 Innings</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics section
            st.markdown("### 📊 Summary Statistics")
            col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
            
            with col1:
                st.metric("Matches", total_matches, border=True)
            with col2:
                st.metric("Innings", total_innings, border=True)
            with col3:
                st.metric("Outs", total_outs, border=True)
            with col4:
                st.metric("Runs", total_runs, border=True)
            with col5:
                st.metric("Balls", total_balls, border=True)
            with col6:
                st.metric("50s", total_50s, border=True)
            with col7:
                st.metric("100s", total_100s, border=True)
            with col8:
                st.metric("Average", f"{calculated_avg:.2f}", border=True)
            with col9:
                st.metric("Strike Rate", f"{calculated_sr:.2f}", border=True)
            
            # Dataframe section
            st.markdown("### 📋 Recent Innings Details")
            
            # Simple styling function for runs
            def style_runs_column(val):
                if val <= 20:
                    return 'background-color: #ffebee; color: #c62828;'
                elif 21 <= val <= 49:
                    return 'background-color: #fff3e0; color: #ef6c00;'
                elif 50 <= val < 100:
                    return 'background-color: #e8f5e8; color: #2e7d32;'
                elif val >= 100:
                    return 'background-color: #e3f2fd; color: #1565c0;'
                return ''
            
            # Apply styling and display
            styled_latest_df = final_latest_df.style.map(style_runs_column, subset=['Runs'])
            st.dataframe(styled_latest_df, height=735, use_container_width=True, hide_index=True)

        # Opponent Stats Tab  
        with tabs[4]:
            opponents_stats_df = metrics.get("opponent", pd.DataFrame())

            st.markdown("""
            <div style="background: linear-gradient(135deg, #a8caba 0%, #5d4e75 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(168, 202, 186, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">🏆 Opponent Statistics</h3>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(opponents_stats_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})
            
            # Bar chart for opponent averages
            opponent_avg_df = filtered_df.groupby('Bowl_Team_y', observed=True).agg(
                Runs=('Runs', 'sum'),
                Out=('Out', 'sum')
            ).reset_index()
            opponent_avg_df['Avg'] = (opponent_avg_df['Runs'] / opponent_avg_df['Out'].replace(0, np.nan)).fillna(0).round(2)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=opponent_avg_df['Bowl_Team_y'], y=opponent_avg_df['Avg'], name='Average'))
            st.markdown("""
            <div style="background: linear-gradient(135deg, #a8caba 0%, #5d4e75 100%); 
                        padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                        box-shadow: 0 6px 24px rgba(168, 202, 186, 0.25);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h2 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.4rem; text-align: center;">📈 Average Runs Against Opponents</h2>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True, key="opponent_bar")

        # Location Stats Tab
        with tabs[5]:
            location_stats_df = metrics.get("location", pd.DataFrame())

            st.markdown("""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">📍 Location Statistics</h3>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(location_stats_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            # Bar chart for location averages
            # Determine correct column name
            home_col = 'Home Team' if 'Home Team' in filtered_df.columns else 'Home_Team'
            
            location_avg_df = filtered_df.groupby(home_col, observed=True).agg(
                Runs=('Runs', 'sum'),
                Out=('Out', 'sum')
            ).reset_index()
            location_avg_df['Avg'] = (location_avg_df['Runs'] / location_avg_df['Out'].replace(0, np.nan)).fillna(0).round(2)
            fig = go.Figure(go.Bar(x=location_avg_df[home_col], y=location_avg_df['Avg'], name='Average'))
            st.markdown("""
            <div style="background: linear-gradient(135deg, #4776e6 0%, #8e54e9 100%); 
                        padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                        box-shadow: 0 6px 24px rgba(71, 118, 230, 0.25);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">📍 Average Runs by Location</h3>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True, key="location_bar")

        # Innings Stats Tab
        with tabs[6]:
            innings_stats_df = metrics.get("innings", pd.DataFrame())

            st.markdown("""
            <div style="background: linear-gradient(135deg, #36d1dc 0%, #5b86e5 100%); ...">
                <h3 ...>🎯 Innings Statistics</h3>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(innings_stats_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            # --- New Plotting Logic: Average & Strike Rate by Innings ---
            st.markdown("""
            <div style="background: linear-gradient(135deg, #36d1dc 0%, #5b86e5 100%); ...">
                <h3 ...>🎯 Average & Strike Rate by Innings Number</h3>
            </div>
            """, unsafe_allow_html=True)
            innings_grouped = filtered_df.groupby('Innings', observed=True).agg(
                Runs=('Runs', 'sum'),
                Out=('Out', 'sum'),
                Balls=('Balls', 'sum')
            ).reset_index()
            innings_grouped['Avg'] = (innings_grouped['Runs'] / innings_grouped['Out'].replace(0, np.nan)).fillna(0)
            innings_grouped['SR'] = (innings_grouped['Runs'] / innings_grouped['Balls'].replace(0, np.nan) * 100).fillna(0)
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Average by Innings", "Strike Rate by Innings"))
            fig.add_trace(go.Scatter(x=innings_grouped['Innings'], y=innings_grouped['Avg'], mode='lines+markers', name="Average"), row=1, col=1)
            fig.add_trace(go.Scatter(x=innings_grouped['Innings'], y=innings_grouped['SR'], mode='lines+markers', name="Strike Rate"), row=1, col=2)
            fig.update_xaxes(tickmode='linear', dtick=1, row=1, col=1)
            fig.update_xaxes(tickmode='linear', dtick=1, row=1, col=2)
            st.plotly_chart(fig, use_container_width=True, key="innings_trend")

        # Position Stats Tab
        with tabs[7]:
            position_stats_df = metrics.get("position", pd.DataFrame())
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #8360c3 0%, #2ebf91 100%); ...">
                <h3 ...>📍 Position Statistics</h3>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(position_stats_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            # --- New Plotting Logic: Average & Strike Rate by Position ---
            st.markdown("""
            <div style="background: linear-gradient(135deg, #8360c3 0%, #2ebf91 100%); ...">
                <h3 ...>📍 Average & Strike Rate by Batting Position</h3>
            </div>
            """, unsafe_allow_html=True)
            position_grouped = filtered_df.groupby('Position', observed=True).agg(
                Runs=('Runs', 'sum'),
                Out=('Out', 'sum'),
                Balls=('Balls', 'sum')
            ).reset_index()
            position_grouped['Avg'] = (position_grouped['Runs'] / position_grouped['Out'].replace(0, np.nan)).fillna(0)
            position_grouped['SR'] = (position_grouped['Runs'] / position_grouped['Balls'].replace(0, np.nan) * 100).fillna(0)
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Average by Position", "Strike Rate by Position"))
            fig.add_trace(go.Scatter(x=position_grouped['Position'], y=position_grouped['Avg'], mode='lines+markers', name="Average"), row=1, col=1)
            fig.add_trace(go.Scatter(x=position_grouped['Position'], y=position_grouped['SR'], mode='lines+markers', name="Strike Rate"), row=1, col=2)
            fig.update_xaxes(tickmode='linear', dtick=1, row=1, col=1)
            fig.update_xaxes(tickmode='linear', dtick=1, row=1, col=2)
            st.plotly_chart(fig, use_container_width=True, key="position_trend")

        # Home/Away Stats Tab
        with tabs[8]:
            homeaway_stats_df = metrics.get("homeaway", pd.DataFrame())

            st.markdown("""
            <div style="background: linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(255, 126, 95, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">🏠 Home/Away Statistics</h3>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(homeaway_stats_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})
            
            # --- Plotting Logic ---
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%); ...">
                <h3 ...>🏠 Home vs Away Performance Trends by Year</h3>
            </div>
            """, unsafe_allow_html=True)
            
            yearly_homeaway_stats = filtered_df.groupby(['Year', 'HomeOrAway'], observed=True).agg(
                Runs=('Runs', 'sum'), Out=('Out', 'sum'), Balls=('Balls', 'sum')
            ).reset_index()
            yearly_homeaway_stats['Average'] = (yearly_homeaway_stats['Runs'] / yearly_homeaway_stats['Out'].replace(0, np.nan)).fillna(0)
            yearly_homeaway_stats['Strike_Rate'] = (yearly_homeaway_stats['Runs'] / yearly_homeaway_stats['Balls'].replace(0, np.nan) * 100).fillna(0)

            fig = make_subplots(rows=1, cols=2, subplot_titles=("Average by Year", "Strike Rate by Year"))
            colors = {'Home': '#1f77b4', 'Away': '#d62728', 'Neutral': '#2ca02c'}
            
            for location in yearly_homeaway_stats['HomeOrAway'].unique():
                location_data = yearly_homeaway_stats[yearly_homeaway_stats['HomeOrAway'] == location]
                fig.add_trace(go.Scatter(x=location_data['Year'], y=location_data['Average'], mode='lines+markers', name=f"{location} Avg", line=dict(color=colors.get(location))), row=1, col=1)
                fig.add_trace(go.Scatter(x=location_data['Year'], y=location_data['Strike_Rate'], mode='lines+markers', name=f"{location} SR", line=dict(color=colors.get(location), dash='dot'), showlegend=False), row=1, col=2)
            
            # Ensure x-axis shows only full integer years
            fig.update_xaxes(tickmode='linear', dtick=1, row=1, col=1)
            fig.update_xaxes(tickmode='linear', dtick=1, row=1, col=2)
            st.plotly_chart(fig, use_container_width=True, key="homeaway_trend")

        # Cumulative Stats Tab
        with tabs[9]:
            cumulative_stats_df = metrics.get("cumulative", pd.DataFrame())
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(17, 153, 142, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">📈 Cumulative Statistics</h3>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(cumulative_stats_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})
            
            # --- Plotting Logic ---
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Cumulative Average")
                fig1 = go.Figure()
                for name in cumulative_stats_df['Name'].unique():
                    player_data = cumulative_stats_df[cumulative_stats_df['Name'] == name]
                    fig1.add_trace(go.Scatter(x=player_data['Cumulative Innings'], y=player_data['Cumulative Avg'], mode='lines', name=name))
                st.plotly_chart(fig1, use_container_width=True, key="cumulative_avg")
            with col2:
                st.subheader("Cumulative Strike Rate")
                fig2 = go.Figure()
                for name in cumulative_stats_df['Name'].unique():
                    player_data = cumulative_stats_df[cumulative_stats_df['Name'] == name]
                    fig2.add_trace(go.Scatter(x=player_data['Cumulative Innings'], y=player_data['Cumulative SR'], mode='lines', name=name))
                st.plotly_chart(fig2, use_container_width=True, key="cumulative_sr")
            with col3:
                st.subheader("Cumulative Runs")
                fig3 = go.Figure()
                for name in cumulative_stats_df['Name'].unique():
                    player_data = cumulative_stats_df[cumulative_stats_df['Name'] == name]
                    fig3.add_trace(go.Scatter(x=player_data['Cumulative Innings'], y=player_data['Cumulative Runs'], mode='lines', name=name))
                st.plotly_chart(fig3, use_container_width=True, key="cumulative_runs")

        # Block Stats Tab
        with tabs[10]:
            block_stats_df = metrics.get("block", pd.DataFrame())

            st.markdown("""
            <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(30, 60, 114, 0.4);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">📊 Block Statistics (Groups of 20 Innings)</h3>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(block_stats_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            # --- Plotting Logic ---
            st.subheader("Batting Average per Block")
            fig = go.Figure()
            for name in block_stats_df['Name'].unique():
                player_data = block_stats_df[block_stats_df['Name'] == name]
                fig.add_trace(go.Bar(x=player_data['Match_Range'], y=player_data['Avg'], name=name))
            
            fig.update_layout(xaxis_title='Innings Range', yaxis_title='Batting Average', barmode='group')
            st.plotly_chart(fig, use_container_width=True, key="block_avg")

 

        logger.log_total(
            "BatView render complete",
            perf_start,
            total_rows=len(bat_df),
            filtered_rows=len(filtered_df) if 'filtered_df' in locals() else 0
        )
    else:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f59e0b, #d97706);
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #fbbf24;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <p style="color: white; margin: 0; font-weight: 500;">
                ⚠️ Please upload a file first.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Call the function to display the batting view
display_bat_view()
 
