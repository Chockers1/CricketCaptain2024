import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import polars as pl

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

@st.cache_data
def compute_bowl_career_stats(df):
    if df.empty:
        return pd.DataFrame()
    
    try:
        pl_df = pl.from_pandas(df)
        
        # --- DEFENSIVE AGGREGATIONS ---
        aggs = [
            pl.col("File Name").n_unique().alias("Matches"),
            pl.col("Bowler_Balls").sum().alias("Balls"),
            pl.col("Bowler_Runs").sum().alias("Runs"),
            pl.col("Bowler_Wkts").sum().alias("Wickets")
        ]
        if "Maidens" in pl_df.columns:
            aggs.append(pl.col("Maidens").sum().alias("M/D"))

        bowlcareer_pl = pl_df.group_by("Name").agg(aggs)

        if "M/D" not in bowlcareer_pl.columns:
            bowlcareer_pl = bowlcareer_pl.with_columns(pl.lit(0).alias("M/D"))

        # --- Feature Engineering ---
        bowlcareer_pl = bowlcareer_pl.with_columns([
            (((pl.col("Balls") / 6) + (pl.col("Balls") % 6) / 10).round(1)).alias("Overs"),
            (pl.col("Runs") / pl.col("Wickets").cast(pl.Float64)).round(2).alias("Avg"),
            (pl.col("Balls") / pl.col("Wickets").cast(pl.Float64)).round(2).alias("Strike Rate"),
            (pl.col("Runs") / (pl.col("Balls").cast(pl.Float64) / 6)).round(2).alias("Economy Rate"),
            (pl.col("Wickets") / pl.col("Matches").cast(pl.Float64)).round(2).alias("WPM")
        ])

        # --- DEFENSIVE Complex Aggregations ---
        five_wickets = pl_df.filter(pl.col("Bowler_Wkts") >= 5).group_by("Name").agg(pl.count().alias("5W"))
        bowlcareer_pl = bowlcareer_pl.join(five_wickets, on="Name", how="left")

        match_wickets = pl_df.group_by(["Name", "File Name"]).agg(pl.col("Bowler_Wkts").sum())
        ten_wickets = match_wickets.filter(pl.col("Bowler_Wkts") >= 10).group_by("Name").agg(pl.count().alias("10W"))
        bowlcareer_pl = bowlcareer_pl.join(ten_wickets, on="Name", how="left")

        if "Player_of_the_Match" in pl_df.columns:
            pom_counts = pl_df.filter(pl.col("Player_of_the_Match") == pl.col("Name")).group_by("Name").agg(pl.col("File Name").n_unique().alias("POM"))
            bowlcareer_pl = bowlcareer_pl.join(pom_counts, on="Name", how="left")
        else:
            bowlcareer_pl = bowlcareer_pl.with_columns(pl.lit(0).alias("POM"))

        # --- Final Touches ---
        final_pl = bowlcareer_pl.select([
            "Name", "Matches", "Balls", "Overs", "M/D", "Runs", "Wickets", "Avg",
            "Strike Rate", "Economy Rate", "5W", "10W", "WPM", "POM"
        ]).fill_null(0).sort("Wickets", descending=True)

        return final_pl.to_pandas()
    except Exception as e:
        st.warning(f"Could not compute career stats: {e}")
        return pd.DataFrame()

@st.cache_data
def compute_bowl_format_stats(df):
    if df.empty:
        return pd.DataFrame()

    pl_df = pl.from_pandas(df)
    
    try:
        # --- DEFENSIVE AGGREGATIONS ---
        aggs = [
            pl.col("File Name").n_unique().alias("Matches"),
            pl.col("Bowler_Balls").sum().alias("Balls"),
            pl.col("Bowler_Runs").sum().alias("Runs"),
            pl.col("Bowler_Wkts").sum().alias("Wickets")
        ]
        if "Maidens" in pl_df.columns:
            aggs.append(pl.col("Maidens").sum().alias("M/D"))
            
        bowlformat_pl = pl_df.group_by(["Name", "Match_Format"]).agg(aggs)

        if "M/D" not in bowlformat_pl.columns:
            bowlformat_pl = bowlformat_pl.with_columns(pl.lit(0).alias("M/D"))

        # --- Feature Engineering ---
        bowlformat_pl = bowlformat_pl.with_columns([
            (((pl.col("Balls") / 6) + (pl.col("Balls") % 6) / 10).round(1)).alias("Overs"),
            (pl.col("Runs") / pl.col("Wickets").cast(pl.Float64)).round(2).alias("Avg"),
            (pl.col("Balls") / pl.col("Wickets").cast(pl.Float64)).round(2).alias("Strike Rate"),
            (pl.col("Runs") / (pl.col("Balls").cast(pl.Float64) / 6)).round(2).alias("Economy Rate"),
            (pl.col("Wickets") / pl.col("Matches").cast(pl.Float64)).round(2).alias("WPM")
        ])

        # --- DEFENSIVE Complex Aggregations ---
        five_wickets = pl_df.filter(pl.col("Bowler_Wkts") >= 5).group_by(["Name", "Match_Format"]).agg(pl.count().alias("5W"))
        bowlformat_pl = bowlformat_pl.join(five_wickets, on=["Name", "Match_Format"], how="left")

        match_wickets = pl_df.group_by(["Name", "File Name", "Match_Format"]).agg(pl.col("Bowler_Wkts").sum())
        ten_wickets = match_wickets.filter(pl.col("Bowler_Wkts") >= 10).group_by(["Name", "Match_Format"]).agg(pl.count().alias("10W"))
        bowlformat_pl = bowlformat_pl.join(ten_wickets, on=["Name", "Match_Format"], how="left")

        if "Player_of_the_Match" in pl_df.columns:
            pom_counts = pl_df.filter(pl.col("Player_of_the_Match") == pl.col("Name")).group_by(["Name", "Match_Format"]).agg(pl.col("File Name").n_unique().alias("POM"))
            bowlformat_pl = bowlformat_pl.join(pom_counts, on=["Name", "Match_Format"], how="left")
        else:
            bowlformat_pl = bowlformat_pl.with_columns(pl.lit(0).alias("POM"))
        
        bowlformat_pl = bowlformat_pl.rename({"Match_Format": "Format"})

        # --- Final Touches ---
        final_pl = bowlformat_pl.select([
            "Name", "Format", "Matches", "Balls", "Overs", "M/D", "Runs", "Wickets", "Avg",
            "Strike Rate", "Economy Rate", "5W", "10W", "WPM", "POM"
        ]).fill_null(0).sort("Wickets", descending=True)

        # SUCCESS: Return the final DataFrame if everything worked
        return final_pl.to_pandas()

    except Exception as e:
        # FAILURE: Warn the user and return an empty DataFrame so the app doesn't crash
        st.warning(f"Could not compute format stats due to an error: {e}. This may be because required columns are missing for the current filter.")
        return pd.DataFrame()

@st.cache_data
def compute_bowl_year_stats(df):
    """
    Computes bowling statistics for each player, grouped by year using Polars.

    Args:
        df (pd.DataFrame): The input DataFrame with bowling data.

    Returns:
        pd.DataFrame: A DataFrame with bowling statistics per player and year.
    """
    if df.empty:
        return pd.DataFrame()

    try:
        pl_df = pl.from_pandas(df)

        # --- DEFENSIVE AGGREGATIONS ---
        aggs = [
            pl.col("File Name").n_unique().alias("Matches"),
            pl.col("Bowler_Balls").sum().alias("Balls"),
            pl.col("Bowler_Runs").sum().alias("Runs"),
            pl.col("Bowler_Wkts").sum().alias("Wickets")
        ]
        if "Maidens" in pl_df.columns:
            aggs.append(pl.col("Maidens").sum().alias("M/D"))

        season_pl = pl_df.group_by(["Name", "Year"]).agg(aggs)

        if "M/D" not in season_pl.columns:
            season_pl = season_pl.with_columns(pl.lit(0).alias("M/D"))

        # --- Feature Engineering ---
        season_pl = season_pl.with_columns([
            (((pl.col("Balls") / 6) + (pl.col("Balls") % 6) / 10).round(1)).alias("Overs"),
            (pl.col("Runs") / pl.col("Wickets").cast(pl.Float64)).round(2).alias("Avg"),
            (pl.col("Balls") / pl.col("Wickets").cast(pl.Float64)).round(2).alias("Strike Rate"),
            (pl.col("Runs") / (pl.col("Balls").cast(pl.Float64) / 6)).round(2).alias("Economy Rate"),
            (pl.col("Wickets") / pl.col("Matches").cast(pl.Float64)).round(2).alias("WPM")
        ])

        # --- DEFENSIVE Complex Aggregations ---
        five_wickets = pl_df.filter(pl.col("Bowler_Wkts") >= 5).group_by(["Name", "Year"]).agg(pl.count().alias("5W"))
        season_pl = season_pl.join(five_wickets, on=["Name", "Year"], how="left")

        match_wickets = pl_df.group_by(["Name", "File Name", "Year"]).agg(pl.col("Bowler_Wkts").sum())
        ten_wickets = match_wickets.filter(pl.col("Bowler_Wkts") >= 10).group_by(["Name", "Year"]).agg(pl.count().alias("10W"))
        season_pl = season_pl.join(ten_wickets, on=["Name", "Year"], how="left")

        if "Player_of_the_Match" in pl_df.columns:
            pom_counts = pl_df.filter(pl.col("Player_of_the_Match") == pl.col("Name")).group_by(["Name", "Year"]).agg(pl.col("File Name").n_unique().alias("POM"))
            season_pl = season_pl.join(pom_counts, on=["Name", "Year"], how="left")
        else:
            season_pl = season_pl.with_columns(pl.lit(0).alias("POM"))

        # --- Final Touches ---
        final_pl = season_pl.select([
            "Name", "Year", "Matches", "Balls", "Overs", "M/D", "Runs", "Wickets", "Avg",
            "Strike Rate", "Economy Rate", "5W", "10W", "WPM", "POM"
        ]).fill_null(0).sort("Wickets", descending=True)

        return final_pl.to_pandas()
    except Exception as e:
        st.warning(f"Could not compute year stats: {e}")
        return pd.DataFrame()

@st.cache_data
def compute_bowl_opponent_stats(df):
    if df.empty:
        return pd.DataFrame()

    try:
        pl_df = pl.from_pandas(df)

        # --- DEFENSIVE AGGREGATIONS ---
        aggs = [
            pl.col("File Name").n_unique().alias("Matches"),
            pl.col("Bowler_Balls").sum().alias("Balls"),
            pl.col("Bowler_Runs").sum().alias("Runs"),
            pl.col("Bowler_Wkts").sum().alias("Wickets")
        ]
        if "Maidens" in pl_df.columns:
            aggs.append(pl.col("Maidens").sum().alias("M/D"))

        opponent_pl = pl_df.group_by(["Name", "Bat_Team"]).agg(aggs).rename({"Bat_Team": "Opposition"})

        if "M/D" not in opponent_pl.columns:
            opponent_pl = opponent_pl.with_columns(pl.lit(0).alias("M/D"))


        # --- Feature Engineering ---
        opponent_pl = opponent_pl.with_columns([
            (((pl.col("Balls") / 6) + (pl.col("Balls") % 6) / 10).round(1)).alias("Overs"),
            (pl.col("Runs") / pl.col("Wickets").cast(pl.Float64)).round(2).alias("Average"),
            (pl.col("Balls") / pl.col("Wickets").cast(pl.Float64)).round(2).alias("Strike Rate"),
            (pl.col("Runs") / (pl.col("Balls").cast(pl.Float64) / 6)).round(2).alias("Economy Rate"),
            (pl.col("Wickets") / pl.col("Matches").cast(pl.Float64)).round(2).alias("WPM")
        ])

        # --- Complex Aggregations ---
        five_wickets = pl_df.filter(pl.col("Bowler_Wkts") >= 5).group_by(["Name", "Bat_Team"]).agg(pl.count().alias("5W"))
        match_wickets = pl_df.group_by(["Name", "Bat_Team", "File Name"]).agg(pl.col("Bowler_Wkts").sum())
        ten_wickets = match_wickets.filter(pl.col("Bowler_Wkts") >= 10).group_by(["Name", "Bat_Team"]).agg(pl.count().alias("10W"))

        # --- Joins ---
        opponent_pl = opponent_pl.join(five_wickets, left_on=["Name", "Opposition"], right_on=["Name", "Bat_Team"], how="left")
        opponent_pl = opponent_pl.join(ten_wickets, left_on=["Name", "Opposition"], right_on=["Name", "Bat_Team"], how="left")

        # --- Final Touches ---
        final_pl = opponent_pl.select([
            "Name", "Opposition", "Matches", "Overs", "M/D", "Runs", "Wickets", "Average",
            "Strike Rate", "Economy Rate", "5W", "10W", "WPM"
        ]).fill_null(0).sort("Wickets", descending=True)

        return final_pl.to_pandas()
    except Exception as e:
        st.warning(f"Could not compute opponent stats: {e}")
        return pd.DataFrame()

@st.cache_data
def compute_bowl_location_stats(df):
    if df.empty:
        return pd.DataFrame()

    try:
        pl_df = pl.from_pandas(df)

        # --- DEFENSIVE AGGREGATIONS ---
        aggs = [
            pl.col("File Name").n_unique().alias("Matches"),
            pl.col("Bowler_Balls").sum().alias("Balls"),
            pl.col("Bowler_Runs").sum().alias("Runs"),
            pl.col("Bowler_Wkts").sum().alias("Wickets")
        ]
        if "Maidens" in pl_df.columns:
            aggs.append(pl.col("Maidens").sum().alias("M/D"))

        location_pl = pl_df.group_by(["Name", "Home_Team"]).agg(aggs).rename({"Home_Team": "Location"})

        if "M/D" not in location_pl.columns:
            location_pl = location_pl.with_columns(pl.lit(0).alias("M/D"))

        # --- Feature Engineering ---
        location_pl = location_pl.with_columns([
            (((pl.col("Balls") / 6) + (pl.col("Balls") % 6) / 10).round(1)).alias("Overs"),
            (pl.col("Runs") / pl.col("Wickets").cast(pl.Float64)).round(2).alias("Average"),
            (pl.col("Balls") / pl.col("Wickets").cast(pl.Float64)).round(2).alias("Strike Rate"),
            (pl.col("Runs") / (pl.col("Balls").cast(pl.Float64) / 6)).round(2).alias("Economy Rate"),
            (pl.col("Wickets") / pl.col("Matches").cast(pl.Float64)).round(2).alias("WPM")
        ])

        # --- Complex Aggregations ---
        five_wickets = pl_df.filter(pl.col("Bowler_Wkts") >= 5).group_by(["Name", "Home_Team"]).agg(pl.count().alias("5W"))
        match_wickets = pl_df.group_by(["Name", "Home_Team", "File Name"]).agg(pl.col("Bowler_Wkts").sum())
        ten_wickets = match_wickets.filter(pl.col("Bowler_Wkts") >= 10).group_by(["Name", "Home_Team"]).agg(pl.count().alias("10W"))

        # --- Joins ---
        location_pl = location_pl.join(five_wickets, left_on=["Name", "Location"], right_on=["Name", "Home_Team"], how="left")
        location_pl = location_pl.join(ten_wickets, left_on=["Name", "Location"], right_on=["Name", "Home_Team"], how="left")

        # --- Final Touches ---
        final_pl = location_pl.select([
            "Name", "Location", "Matches", "Overs", "M/D", "Runs", "Wickets", "Average",
            "Strike Rate", "Economy Rate", "5W", "10W", "WPM"
        ]).fill_null(0).sort("Wickets", descending=True)

        return final_pl.to_pandas()
    except Exception as e:
        st.warning(f"Could not compute location stats: {e}")
        return pd.DataFrame()

@st.cache_data
def compute_bowl_innings_stats(df):
    if df.empty:
        return pd.DataFrame()

    try:
        pl_df = pl.from_pandas(df)

        # --- DEFENSIVE AGGREGATIONS ---
        aggs = [
            pl.col("File Name").n_unique().alias("Matches"),
            pl.col("Bowler_Balls").sum().alias("Balls"),
            pl.col("Bowler_Runs").sum().alias("Runs"),
            pl.col("Bowler_Wkts").sum().alias("Wickets")
        ]
        if "Maidens" in pl_df.columns:
            aggs.append(pl.col("Maidens").sum().alias("M/D"))

        innings_pl = pl_df.group_by(["Name", "Innings"]).agg(aggs)

        if "M/D" not in innings_pl.columns:
            innings_pl = innings_pl.with_columns(pl.lit(0).alias("M/D"))

        # --- Feature Engineering ---
        innings_pl = innings_pl.with_columns([
            (((pl.col("Balls") / 6) + (pl.col("Balls") % 6) / 10).round(1)).alias("Overs"),
            (pl.col("Runs") / pl.col("Wickets").cast(pl.Float64)).round(2).alias("Average"),
            (pl.col("Balls") / pl.col("Wickets").cast(pl.Float64)).round(2).alias("Strike Rate"),
            (pl.col("Runs") / (pl.col("Balls").cast(pl.Float64) / 6)).round(2).alias("Economy Rate"),
            (pl.col("Wickets") / pl.col("Matches").cast(pl.Float64)).round(2).alias("WPM")
        ])

        # --- Complex Aggregations ---
        five_wickets = pl_df.filter(pl.col("Bowler_Wkts") >= 5).group_by(["Name", "Innings"]).agg(pl.count().alias("5W"))
        match_wickets = pl_df.group_by(["Name", "Innings", "File Name"]).agg(pl.col("Bowler_Wkts").sum())
        ten_wickets = match_wickets.filter(pl.col("Bowler_Wkts") >= 10).group_by(["Name", "Innings"]).agg(pl.count().alias("10W"))

        # --- Joins ---
        innings_pl = innings_pl.join(five_wickets, on=["Name", "Innings"], how="left")
        innings_pl = innings_pl.join(ten_wickets, on=["Name", "Innings"], how="left")

        # --- Final Touches ---
        final_pl = innings_pl.select([
            "Name", "Innings", "Matches", "Overs", "M/D", "Runs", "Wickets", "Average",
            "Strike Rate", "Economy Rate", "5W", "10W", "WPM"
        ]).fill_null(0).sort("Wickets", descending=True)

        return final_pl.to_pandas()
    except Exception as e:
        st.warning(f"Could not compute innings stats: {e}")
        return pd.DataFrame()

@st.cache_data
def compute_bowl_position_stats(df):
    if df.empty:
        return pd.DataFrame()

    try:
        pl_df = pl.from_pandas(df)

        # --- DEFENSIVE AGGREGATIONS ---
        aggs = [
            pl.col("File Name").n_unique().alias("Matches"),
            pl.col("Bowler_Balls").sum().alias("Balls"),
            pl.col("Bowler_Runs").sum().alias("Runs"),
            pl.col("Bowler_Wkts").sum().alias("Wickets")
        ]
        if "Maidens" in pl_df.columns:
            aggs.append(pl.col("Maidens").sum().alias("M/D"))

        position_pl = pl_df.group_by(["Name", "Position"]).agg(aggs)

        if "M/D" not in position_pl.columns:
            position_pl = position_pl.with_columns(pl.lit(0).alias("M/D"))

        # --- Feature Engineering ---
        position_pl = position_pl.with_columns([
            (((pl.col("Balls") / 6) + (pl.col("Balls") % 6) / 10).round(1)).alias("Overs"),
            (pl.col("Runs") / pl.col("Wickets").cast(pl.Float64)).round(2).alias("Average"),
            (pl.col("Balls") / pl.col("Wickets").cast(pl.Float64)).round(2).alias("Strike Rate"),
            (pl.col("Runs") / (pl.col("Balls").cast(pl.Float64) / 6)).round(2).alias("Economy Rate"),
            (pl.col("Wickets") / pl.col("Matches").cast(pl.Float64)).round(2).alias("WPM")
        ])

        # --- Complex Aggregations ---
        five_wickets = pl_df.filter(pl.col("Bowler_Wkts") >= 5).group_by(["Name", "Position"]).agg(pl.count().alias("5W"))
        match_wickets = pl_df.group_by(["Name", "Position", "File Name"]).agg(pl.col("Bowler_Wkts").sum())
        ten_wickets = match_wickets.filter(pl.col("Bowler_Wkts") >= 10).group_by(["Name", "Position"]).agg(pl.count().alias("10W"))

        # --- Joins ---
        position_pl = position_pl.join(five_wickets, on=["Name", "Position"], how="left")
        position_pl = position_pl.join(ten_wickets, on=["Name", "Position"], how="left")

        # --- Final Touches ---
        final_pl = position_pl.select([
            "Name", "Position", "Matches", "Overs", "M/D", "Runs", "Wickets", "Average",
            "Strike Rate", "Economy Rate", "5W", "10W", "WPM"
        ]).fill_null(0).sort("Wickets", descending=True)

        return final_pl.to_pandas()
    except Exception as e:
        st.warning(f"Could not compute position stats: {e}")
        return pd.DataFrame()

@st.cache_data
def compute_bowl_latest_innings(df):
    if df.empty:
        return pd.DataFrame()

    try:
        pl_df = pl.from_pandas(df)

        # Date is already in datetime format from display_bowl_view

        # --- DEFENSIVE AGGREGATIONS ---
        aggs = [
            pl.col('Bowl_Team').first(),
            pl.col('Bat_Team').first(),
            pl.col('Bowler_Runs').sum(),
            pl.col('Bowler_Wkts').sum(),
            pl.col('File Name').first()
        ]
        if "Maidens" in pl_df.columns:
            aggs.append(pl.col("Maidens").sum())
        if "Overs" in pl_df.columns:
            aggs.append(pl.col("Overs").sum())

        latest_innings_pl = pl_df.group_by(['Name', 'Match_Format', 'Date', 'Innings']).agg(aggs)

        # Add columns with 0 if they were not in the original df
        if "Maidens" not in latest_innings_pl.columns:
            latest_innings_pl = latest_innings_pl.with_columns(pl.lit(0).alias("Maidens"))
        if "Overs" not in latest_innings_pl.columns:
            latest_innings_pl = latest_innings_pl.with_columns(pl.lit(0.0).alias("Overs"))

        # --- Sort, Format, and Select ---
        final_pl = (
            latest_innings_pl.sort(by='Date', descending=True)
            .head(20)
            .with_columns(pl.col("Date").dt.strftime('%d/%m/%Y'))
            .rename({
                'Match_Format': 'Format',
                'Bowl_Team': 'Team',
                'Bat_Team': 'Opponent',
                'Bowler_Runs': 'Runs',
                'Bowler_Wkts': 'Wickets'
            })
            .select([
                "Name", "Format", "Date", "Innings", "Team", "Opponent", 
                "Overs", "Maidens", "Runs", "Wickets", "File Name"
            ])
        )

        return final_pl.to_pandas()
    except Exception as e:
        st.warning(f"Could not compute latest innings: {e}")
        return pd.DataFrame()

@st.cache_data
def compute_bowl_cumulative_stats(df):
    if df.empty:
        return pd.DataFrame()

    try:
        pl_df = pl.from_pandas(df)

        # Date is already in datetime format from display_bowl_view

        # Aggregate stats at the match level first
        match_level_pl = pl_df.group_by(['Name', 'Match_Format', 'Date', 'File Name']).agg([
            pl.col('Bowler_Balls').sum(),
            pl.col('Bowler_Runs').sum(),
            pl.col('Bowler_Wkts').sum()
        ]).sort(['Name', 'Match_Format', 'Date'])

        # Calculate cumulative stats using window functions
        cumulative_pl = match_level_pl.with_columns([
            (pl.cum_count('File Name').over(['Name', 'Match_Format']) + 1).alias('Cumulative Matches'),
            pl.col('Bowler_Runs').cum_sum().over(['Name', 'Match_Format']).alias('Cumulative Runs'),
            pl.col('Bowler_Balls').cum_sum().over(['Name', 'Match_Format']).alias('Cumulative Balls'),
            pl.col('Bowler_Wkts').cum_sum().over(['Name', 'Match_Format']).alias('Cumulative Wickets')
        ])

        # Calculate derived cumulative stats (Avg, SR, Econ)
        final_pl = cumulative_pl.with_columns([
            pl.when(pl.col('Cumulative Wickets') > 0)
              .then(pl.col('Cumulative Runs') / pl.col('Cumulative Wickets'))
              .otherwise(0)
              .round(2).alias('Cumulative Avg'),
            pl.when(pl.col('Cumulative Wickets') > 0)
              .then(pl.col('Cumulative Balls') / pl.col('Cumulative Wickets'))
              .otherwise(0)
              .round(2).alias('Cumulative SR'),
            pl.when(pl.col('Cumulative Balls') > 0)
              .then(pl.col('Cumulative Runs') / (pl.col('Cumulative Balls') / 6))
              .otherwise(0)
              .round(2).alias('Cumulative Econ')
        ])

        return final_pl.sort(by='Date', descending=True).to_pandas()
    except Exception as e:
        st.warning(f"Could not compute cumulative stats: {e}")
        return pd.DataFrame()

@st.cache_data
def compute_bowl_block_stats(df):
    if df.empty:
        return pd.DataFrame()

    try:
        pl_df = pl.from_pandas(df)

        # --- Initial Preparation ---
        # Date is already in datetime format from display_bowl_view
        pl_df = pl_df.sort(['Name', 'Match_Format', 'Date'])

        # --- Create Innings and Block Identifiers ---
        pl_df = pl_df.with_columns([
            (pl.cum_count("Date").over(["Name", "Match_Format"]) + 1).alias("Innings_Number")
        ])
        pl_df = pl_df.with_columns([
            (((pl.col("Innings_Number") - 1) / 20).floor().cast(pl.Int32) * 20).alias("Range_Start")
        ])
        pl_df = pl_df.with_columns([
            (pl.col("Range_Start").cast(pl.Utf8) + "-" + (pl.col("Range_Start") + 19).cast(pl.Utf8)).alias("Innings_Range")
        ])

        # --- Group by Blocks and Aggregate ---
        block_stats_pl = pl_df.group_by(['Name', 'Match_Format', 'Innings_Range', 'Range_Start']).agg([
            pl.count().alias('Innings'),
            pl.col('Bowler_Balls').sum().alias('Balls'),
            pl.col('Bowler_Runs').sum().alias('Runs'),
            pl.col('Bowler_Wkts').sum().alias('Wickets'),
            pl.col('Date').first().alias('First_Date'),
            pl.col('Date').last().alias('Last_Date')
        ])

        # --- Calculate Derived Statistics ---
        block_stats_pl = block_stats_pl.with_columns([
            (((pl.col("Balls") / 6).floor() + (pl.col("Balls") % 6) / 10)).alias("Overs"),
        ])
        block_stats_pl = block_stats_pl.with_columns([
            pl.when(pl.col("Wickets") > 0).then(pl.col('Runs') / pl.col('Wickets')).otherwise(0).round(2).alias('Average'),
            pl.when(pl.col("Wickets") > 0).then(pl.col('Balls') / pl.col('Wickets')).otherwise(0).round(2).alias('Strike_Rate'),
        ])
        # Economy Rate needs Overs, so calculate separately
        block_stats_pl = block_stats_pl.with_columns(
            pl.when(pl.col("Overs") > 0).then(pl.col('Runs') / pl.col('Overs')).otherwise(0).round(2).alias('Economy')
        )

        # --- Format Dates and Create Date Range ---
        block_stats_pl = block_stats_pl.with_columns([
            (pl.col('First_Date').dt.strftime('%d/%m/%Y') + " to " + pl.col('Last_Date').dt.strftime('%d/%m/%Y')).alias('Date_Range')
        ])

        # --- Final Selection and Sorting ---
        final_pl = (
            block_stats_pl.sort(by='Range_Start', descending=False)
            .select([
                'Name', 'Match_Format', 'Innings_Range', 'Date_Range', 'Innings', 
                'Overs', 'Runs', 'Wickets', 'Average', 'Strike_Rate', 'Economy'
            ])
        )

        return final_pl.to_pandas()
    except Exception as e:
        st.warning(f"Could not compute block stats: {e}")
        return pd.DataFrame()

@st.cache_data
def compute_bowl_homeaway_stats(df):
    if df.empty:
        return pd.DataFrame()

    try:
        pl_df = pl.from_pandas(df)

        # --- DEFENSIVE AGGREGATIONS ---
        aggs = [
            pl.col("File Name").n_unique().alias("Matches"),
            pl.col("Bowler_Balls").sum().alias("Balls"),
            pl.col("Bowler_Runs").sum().alias("Runs"),
            pl.col("Bowler_Wkts").sum().alias("Wickets")
        ]
        if "Maidens" in pl_df.columns:
            aggs.append(pl.col("Maidens").sum().alias("M/D"))

        homeaway_pl = pl_df.group_by(["Name", "HomeOrAway"]).agg(aggs)

        if "M/D" not in homeaway_pl.columns:
            homeaway_pl = homeaway_pl.with_columns(pl.lit(0).alias("M/D"))

        # --- Feature Engineering ---
        homeaway_pl = homeaway_pl.with_columns([
            (((pl.col("Balls") / 6) + (pl.col("Balls") % 6) / 10).round(1)).alias("Overs"),
            (pl.col("Runs") / pl.col("Wickets").cast(pl.Float64)).round(2).alias("Average"),
            (pl.col("Balls") / pl.col("Wickets").cast(pl.Float64)).round(2).alias("Strike Rate"),
            (pl.col("Runs") / (pl.col("Balls").cast(pl.Float64) / 6)).round(2).alias("Economy Rate"),
            (pl.col("Wickets") / pl.col("Matches").cast(pl.Float64)).round(2).alias("WPM")
        ])

        # --- Complex Aggregations ---
        five_wickets = pl_df.filter(pl.col("Bowler_Wkts") >= 5).group_by(["Name", "HomeOrAway"]).agg(pl.count().alias("5W"))
        match_wickets = pl_df.group_by(["Name", "HomeOrAway", "File Name"]).agg(pl.col("Bowler_Wkts").sum())
        ten_wickets = match_wickets.filter(pl.col("Bowler_Wkts") >= 10).group_by(["Name", "HomeOrAway"]).agg(pl.count().alias("10W"))

        # --- Joins ---
        homeaway_pl = homeaway_pl.join(five_wickets, on=["Name", "HomeOrAway"], how="left")
        homeaway_pl = homeaway_pl.join(ten_wickets, on=["Name", "HomeOrAway"], how="left")

        # --- Final Touches ---
        final_pl = homeaway_pl.rename({"HomeOrAway": "Home/Away"}).select([
            "Name", "Home/Away", "Matches", "Overs", "M/D", "Runs", "Wickets", "Average",
            "Strike Rate", "Economy Rate", "5W", "10W", "WPM"
        ]).fill_null(0).sort("Wickets", descending=True)

        return final_pl.to_pandas()
    except Exception as e:
        st.warning(f"Could not compute home/away stats: {e}")
        return pd.DataFrame()

@st.cache_data
def get_player_summary_for_filtering(_df):
    """
    Pre-computes a summary DataFrame with min/max values for filtering.
    This is much faster than filtering the main DataFrame directly.
    """
    if _df.empty:
        return pd.DataFrame(columns=["Name", "Matches", "Wickets", "Avg", "SR"])

    pl_df = pl.from_pandas(_df)

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
        # Get the bowling dataframe with safer date parsing
        bowl_df = st.session_state['bowl_df'].copy()
        

            
        try:
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
                
            # Add HomeOrAway column
            if 'Bowl_Team' in bowl_df.columns and 'Home_Team' in bowl_df.columns:
                bowl_df['HomeOrAway'] = np.where(bowl_df['Bowl_Team'] == bowl_df['Home_Team'], 'Home', 'Away')
            else:
                st.warning("Could not determine Home/Away status due to missing columns.")
                bowl_df['HomeOrAway'] = 'Unknown'

        except Exception as e:
            st.error(f"Error processing dates or adding columns: {str(e)}")
            # Ensure Year column exists even if there's an error
            if 'Year' not in bowl_df.columns:
                bowl_df['Year'] = 2024  # Default year
            if 'HomeOrAway' not in bowl_df.columns:
                bowl_df['HomeOrAway'] = 'Unknown'

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
            st.session_state.prev_bowl_teams = current_bowl_teams        ###-------------------------------------HEADER AND FILTERS-------------------------------------###
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
                bowl_df['comp'] = bowl_df['comp'].fillna(bowl_df['Competition'])
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
        # Get the pre-calculated summary stats for ALL players ONCE.
        player_summary = get_player_summary_for_filtering(bowl_df)

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
                year_choice = (years[0], years[0])  # Set the year range to the single year
            else:
                year_choice = st.slider('', 
                        min_value=min(years),
                        max_value=max(years),
                        value=(min(years), max(years)),
                        label_visibility='collapsed',
                        key='year_slider')

        # The rest of the sliders remain the same
        with col6:
            st.markdown("<p style='text-align: center;'>Choose Position:</p>", unsafe_allow_html=True)
            position_choice = st.slider('', 
                        min_value=1, 
                        max_value=11, 
                        value=(1, 11),
                        label_visibility='collapsed',
                        key='position_slider')

        with col7:
            st.markdown("<p style='text-align: center;'>Wickets Range</p>", unsafe_allow_html=True)
            wickets_range = st.slider('', 
                                    min_value=0, 
                                    max_value=max_wickets, 
                                    value=(0, max_wickets),
                                    label_visibility='collapsed',
                                    key='wickets_slider')

        with col8:
            st.markdown("<p style='text-align: center;'>Matches Range</p>", unsafe_allow_html=True)
            matches_range = st.slider('', 
                                    min_value=1, 
                                    max_value=max_matches, 
                                    value=(1, max_matches),
                                    label_visibility='collapsed',
                                    key='matches_slider')

        with col9:
            st.markdown("<p style='text-align: center;'>Average Range</p>", unsafe_allow_html=True)
            avg_range = st.slider('', 
                                min_value=0.0, 
                                max_value=max_avg, 
                                value=(0.0, max_avg),
                                label_visibility='collapsed',
                                key='avg_slider')

        with col10:
            st.markdown("<p style='text-align: center;'>Strike Rate Range</p>", unsafe_allow_html=True)
            sr_range = st.slider('', 
                                min_value=0.0, 
                                max_value=max_sr, 
                                value=(0.0, max_sr),
                                label_visibility='collapsed',
                                key='sr_slider')

###-------------------------------------APPLY FILTERS (THE FAST WAY)-------------------------------------###
        
        # 1. Start with the cheap multiselect filters on the main dataframe
        filtered_df = bowl_df.copy()
        if name_choice and 'All' not in name_choice:
            filtered_df = filtered_df[filtered_df['Name'].isin(name_choice)]
        if bowl_team_choice and 'All' not in bowl_team_choice:
            filtered_df = filtered_df[filtered_df['Bowl_Team'].isin(bowl_team_choice)]
        if bat_team_choice and 'All' not in bat_team_choice:
            filtered_df = filtered_df[filtered_df['Bat_Team'].isin(bat_team_choice)]
        if match_format_choice and 'All' not in match_format_choice:
            filtered_df = filtered_df[filtered_df['Match_Format'].isin(match_format_choice)]
        if comp_choice and 'All' not in comp_choice:
            filtered_df = filtered_df[filtered_df['comp'].isin(comp_choice)]

        # 2. Filter the SMALL player_summary DataFrame based on the sliders. This is VERY fast.
        if not player_summary.empty:
            eligible_players = player_summary[
                (player_summary['Wickets'].between(wickets_range[0], wickets_range[1])) &
                (player_summary['Matches'].between(matches_range[0], matches_range[1])) &
                (player_summary['Avg'].between(avg_range[0], avg_range[1])) &
                (player_summary['SR'].between(sr_range[0], sr_range[1]))
            ]
            
            # 3. Get the list of names that passed the filter.
            eligible_player_names = eligible_players['Name'].unique()
            
            # 4. Apply a final, lightning-fast .isin() filter to the main DataFrame
            filtered_df = filtered_df[filtered_df['Name'].isin(eligible_player_names)]
        
        # 5. Apply the last remaining filters
        if 'Year' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Year'].between(year_choice[0], year_choice[1])]
        
        filtered_df = filtered_df[filtered_df['Position'].between(position_choice[0], position_choice[1])]


        # Create a placeholder for tabs that will be lazily loaded
        main_container = st.container()
        
        # Create tabs for different views with clean, short names
        tabs = main_container.tabs([
            "Career", "Format", "Season", "Latest", "Opponent", 
            "Location", "Innings", "Position", "Home/Away",
            "Cumulative", "Block"
        ])

        ###-------------------------------------CAREER STATS-------------------------------------###
        # Career Stats Tab
        with tabs[0]:
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
                else:
                    st.info("Not enough data to display the Economy Rate vs Strike Rate Analysis.")

        ###-------------------------------------FORMAT STATS-------------------------------------###
        # Format Stats Tab
        with tabs[1]:
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
                # --- Modern UI Section Header for Graphs ---
                st.markdown("""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(240, 147, 251, 0.25);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">📈 Format Performance Trends by Season</h3>
                </div>
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
                totals = filtered_df.groupby(['Match_Format', 'Year']).agg({
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
        with tabs[2]:
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
                # --- Modern UI Section Header for Graphs ---
                st.markdown("""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(240, 147, 251, 0.25);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">📈 Bowling Average, Strike Rate & Economy Rate Per Season</h3>
                </div>
                """, unsafe_allow_html=True)
                # --- Three Column Layout for Graphs ---
                col1, col2, col3 = st.columns(3)
                # Use actual column names from season_df for totals
                # Try to find the correct columns for runs, wickets, balls
                # Print or inspect season_df.columns if unsure
                # For now, try 'Runs', 'Wickets', 'Balls'
                # If not present, fallback to 'Bowler_Runs', 'Bowler_Wkts', 'Bowler_Balls'
                cols = season_df.columns
                runs_col = 'Runs' if 'Runs' in cols else ('Bowler_Runs' if 'Bowler_Runs' in cols else None)
                wickets_col = 'Wickets' if 'Wickets' in cols else ('Bowler_Wkts' if 'Bowler_Wkts' in cols else None)
                balls_col = 'Balls' if 'Balls' in cols else ('Bowler_Balls' if 'Bowler_Balls' in cols else None)
                if not (runs_col and wickets_col and balls_col):
                    st.error("Could not find the correct columns for runs, wickets, and balls in season_df.")
                else:
                    totals = season_df.groupby('Year').agg({
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
        with tabs[3]:
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
        with tabs[4]:
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

                # --- Plotting Logic ---
                st.markdown("""
                <div style="background: linear-gradient(135deg, #a8caba 0%, #5d4e75 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(168, 202, 186, 0.25);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">Average vs Opponent Team</h3>
                </div>
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
        with tabs[5]:
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
                st.markdown("""
                <div style="background: linear-gradient(135deg, #4776e6 0%, #8e54e9 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(71, 118, 230, 0.25);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">Average vs Location</h3>
                </div>
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
        with tabs[6]:
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
                st.markdown("""
                <div style="background: linear-gradient(135deg, #36d1dc 0%, #5b86e5 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(54, 209, 220, 0.25);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">Average vs Innings</h3>
                </div>
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
        with tabs[7]:
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
                st.markdown("""
                <div style="background: linear-gradient(135deg, #8360c3 0%, #2ebf91 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(131, 96, 195, 0.25);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">Average vs Position</h3>
                </div>
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
        with tabs[8]:
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
                # --- Modern UI Section Header for Graphs ---
                st.markdown("""
                <div style="background: linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(255, 126, 95, 0.25);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">📈 Home/Away Performance Trends by Year</h3>
                </div>
                """, unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                # Prepare yearly stats by Home/Away
                yearly_ha = filtered_df.groupby(['Year', 'HomeOrAway']).agg({
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
        with tabs[9]:
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
        with tabs[10]:
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
                <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(30, 60, 114, 0.25);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">Bowling Average by Innings Block</h3>
                </div>
                """, unsafe_allow_html=True)
                
                fig = go.Figure()
                # This code is now safe because we know the DataFrame is not empty
                if 'All' in name_choice:
                    all_blocks = block_stats_df.groupby('Innings_Range').agg({
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

    else:
        st.error("No bowling data available. Please upload scorecards first.")

# Display the bowling view
display_bowl_view()
