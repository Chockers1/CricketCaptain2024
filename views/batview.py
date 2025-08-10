import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import plotly.graph_objects as go
import json
from datetime import timedelta
from functools import wraps
import time
import subprocess

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

# Add the new cached function for career stats at the top
@st.cache_data
def compute_career_stats(filtered_df):
    # First, calculate file-level statistics
    file_stats = filtered_df.groupby('File Name').agg({
        'Runs': 'sum',
        'Team Balls': 'sum',
        'Wickets': 'sum',
        'Total_Runs': 'sum'
    }).reset_index()

    # Calculate match-level metrics
    file_stats['Match_Avg'] = file_stats['Runs'] / file_stats['Wickets'].replace(0, np.inf)
    file_stats['Match_SR'] = (file_stats['Runs'] / file_stats['Team Balls'].replace(0, np.inf)) * 100

    # Calculate average match metrics
    avg_match_stats = {
        'Match_Avg': file_stats['Match_Avg'].mean(),
        'Match_SR': file_stats['Match_SR'].mean()
    }

    # Calculate milestone innings based on run ranges
    filtered_df['50s'] = ((filtered_df['Runs'] >= 50) & (filtered_df['Runs'] < 100)).astype(int)
    filtered_df['100s'] = ((filtered_df['Runs'] >= 100))
    filtered_df['150s'] = ((filtered_df['Runs'] >= 150) & (filtered_df['Runs'] < 200)).astype(int)
    filtered_df['200s'] = (filtered_df['Runs'] >= 200).astype(int)

    # Create the bat_career_df by grouping by 'Name' and summing the required statistics
    bat_career_df = filtered_df.groupby('Name').agg({
        'File Name': 'nunique',
        'Batted': 'sum',
        'Out': 'sum',
        'Not Out': 'sum',    
        'Balls': 'sum',
        'Runs': ['sum', 'max'],
        '4s': 'sum',
        '6s': 'sum',
        '50s': 'sum',  # Now sums innings where 50 <= runs < 100
        '100s': 'sum', # Now sums innings where 100 <= runs < 150
        '150s': 'sum', # Now sums innings where 150 <= runs < 200
        '200s': 'sum', # Now sums innings where runs >= 200
        '<25&Out': 'sum',
        'Caught': 'sum',
        'Bowled': 'sum',
        'LBW': 'sum',
        'Run Out': 'sum',
        'Stumped': 'sum',
        'Total_Runs': 'sum',
        'Overs': 'sum',
        'Wickets': 'sum',
        'Team Balls': 'sum'
    }).reset_index()

    # Add filename-level aggregations
    filename_stats = filtered_df.groupby('File Name').agg({
        'Runs': 'sum',
        'Out': 'sum', 
        'Balls': 'sum'
    }).reset_index()

    # Calculate match-level averages
    filename_stats['Match_Avg'] = (filename_stats['Runs'] / filename_stats['Out']).fillna(0)
    filename_stats['Match_SR'] = (filename_stats['Runs'] / filename_stats['Balls'] * 100).fillna(0)

    # Calculate average match metrics across all files
    avg_match_avg = filename_stats['Match_Avg'].mean()
    avg_match_sr = filename_stats['Match_SR'].mean()

    # Add filename sums to bat_career_df
    bat_career_df['Filename_Runs'] = filename_stats['Runs'].sum()
    bat_career_df['Filename_Out'] = filename_stats['Out'].sum()
    bat_career_df['Filename_Balls'] = filename_stats['Balls'].sum()

    # Flatten multi-level columns
    bat_career_df.columns = [
        'Name', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 
        'Runs', 'HS', '4s', '6s', '50s', '100s', '150s', '200s', 
        '<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 
        'Team Runs', 'Overs', 'Wickets', 'Team Balls', 'Match_Runs', 
        'Match_Out', 'Match_Balls'
    ]

    # Calculate average runs per out, strike rate, and balls per out
    bat_career_df['Avg'] = (bat_career_df['Runs'] / bat_career_df['Out'].replace(0, np.nan)).round(2).fillna(0)
    bat_career_df['SR'] = ((bat_career_df['Runs'] / bat_career_df['Balls'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_career_df['BPO'] = (bat_career_df['Balls'] / bat_career_df['Out'].replace(0, np.nan)).round(2).fillna(0)

    # Calculate new columns for team statistics
    bat_career_df['Team Avg'] = (bat_career_df['Team Runs'] / bat_career_df['Wickets'].replace(0, np.nan)).round(2).fillna(0)
    bat_career_df['Team SR'] = (bat_career_df['Team Runs'] / bat_career_df['Team Balls'].replace(0, np.nan) * 100).round(2).fillna(0)

    # Calculate P+ Avg and P+ SR
    bat_career_df['Team+ Avg'] = (bat_career_df['Avg'] / bat_career_df['Team Avg'].replace(0, np.nan) * 100).round(2).fillna(0)
    bat_career_df['Team+ SR'] = (bat_career_df['SR'] / bat_career_df['Team SR'].replace(0, np.nan) * 100).round(2).fillna(0)

    # Calculate match comparison metrics
    bat_career_df['Match+ Avg'] = (bat_career_df['Avg'] / (avg_match_avg if avg_match_avg != 0 else np.nan) * 100).round(2).fillna(0)
    bat_career_df['Match+ SR'] = (bat_career_df['SR'] / (avg_match_sr if avg_match_sr != 0 else np.nan) * 100).round(2).fillna(0)

    # Calculate BPB (Balls Per Boundary)
    boundaries = bat_career_df['4s'] + bat_career_df['6s']
    bat_career_df['BPB'] = (bat_career_df['Balls'] / boundaries.replace(0, np.nan)).round(2).fillna(0)
    bat_career_df['Boundary%'] = (((bat_career_df['4s'] * 4) + (bat_career_df['6s'] * 6))   / (bat_career_df['Runs'].replace(0, 1))* 100 ).round(2)
    bat_career_df['RPM'] = (bat_career_df['Runs'] / bat_career_df['Matches'].replace(0, 1)).round(2)

    # Calculate Contribution Percentage (average percentage of team's total runs)
    bat_career_df['Avg Contribution %'] = (
        (bat_career_df['Runs'] / bat_career_df['Team Runs'].replace(0, np.nan)) * 100
    ).round(2).fillna(0)

    # Calculate new statistics
    inns = bat_career_df['Inns'].replace(0, np.nan)
    bat_career_df['50+PI'] = (((bat_career_df['50s'] + bat_career_df['100s'] + bat_career_df['150s'] + bat_career_df['200s']) / inns) * 100).round(2).fillna(0)
    bat_career_df['100PI'] = (((bat_career_df['100s'] + bat_career_df['150s'] + bat_career_df['200s']) / inns) * 100).round(2).fillna(0)
    bat_career_df['150PI'] = (((bat_career_df['150s'] + bat_career_df['200s']) / inns) * 100).round(2).fillna(0)
    bat_career_df['200PI'] = ((bat_career_df['200s'] / inns) * 100).round(2).fillna(0)
    bat_career_df['<25&OutPI'] = ((bat_career_df['<25&Out'] / inns) * 100).round(2).fillna(0)
    bat_career_df['Conversion Rate'] = ((bat_career_df['100s'] / (bat_career_df['50s'] + bat_career_df['100s']).replace(0, 1)) * 100).round(2)

    # Calculate dismissal percentages
    bat_career_df['Caught%'] = ((bat_career_df['Caught'] / inns) * 100).round(2).fillna(0)
    bat_career_df['Bowled%'] = ((bat_career_df['Bowled'] / inns) * 100).round(2).fillna(0)
    bat_career_df['LBW%'] = ((bat_career_df['LBW'] / inns) * 100).round(2).fillna(0)
    bat_career_df['Run Out%'] = ((bat_career_df['Run Out'] / inns) * 100).round(2).fillna(0)
    bat_career_df['Stumped%'] = ((bat_career_df['Stumped'] / inns) * 100).round(2).fillna(0)
    bat_career_df['Not Out%'] = ((bat_career_df['Not Out'] / inns) * 100).round(2).fillna(0)

    # Count Player of the Match awards
    pom_counts = filtered_df[filtered_df['Player_of_the_Match'].astype(str) == filtered_df['Name'].astype(str)].groupby('Name')['File Name'].nunique().reset_index(name='POM')
    bat_career_df = bat_career_df.merge(pom_counts, on='Name', how='left')
    bat_career_df['POM'] = bat_career_df['POM'].fillna(0).astype(int)
    bat_career_df['POM Per Match'] = (bat_career_df['POM'] / bat_career_df['Matches'].replace(0, 1)*100).round(2)

    # Reorder columns and drop Team Avg and Team SR
    bat_career_df = bat_career_df[['Name', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 
                                   'Runs', 'HS', 'Avg', 'BPO', 'SR', '4s', '6s', 'BPB',
                                   'Boundary%', 'RPM', 'Avg Contribution %', '<25&Out', '50s', '100s', '150s', '200s',
                                   'Conversion Rate', '<25&OutPI', '50+PI', '100PI', '150PI', '200PI',
                                   'Match+ Avg', 'Match+ SR', 'Team+ Avg', 'Team+ SR', 'Caught%', 'Bowled%', 'LBW%', 
                                   'Run Out%', 'Stumped%', 'Not Out%', 'POM', 'POM Per Match']]
    # Sort the DataFrame by 'Runs' in descending order
    bat_career_df = bat_career_df.sort_values(by='Runs', ascending=False)
    return bat_career_df

@st.cache_data
def compute_match_impact_stats(filtered_df, match_df):
    """
    Optimized function to compute batting statistics by match result.
    Returns a dictionary with the main table and pre-pivoted chart data.
    """
    if match_df.empty or filtered_df.empty:
        return {'table': pd.DataFrame(), 'chart_data': pd.DataFrame()}

    # 1. VECTORIZED RESULT DETERMINATION
    merged_df = filtered_df.merge(
        match_df[['File Name', 'Home_Win', 'Away_Won', 'Home_Lost', 'Away_Lost', 'Home_Drawn', 'Away_Drawn', 'Tie']],
        on='File Name',
        how='left'
    )

    # Decide which columns are present for team names
    def pick_col(df, options):
        for c in options:
            if c in df.columns:
                return c
        return None

    bat_col = pick_col(merged_df, ['Bat_Team_y', 'Bat_Team', 'Bat Team', 'Bat_Team_x'])
    home_col = pick_col(merged_df, ['Home Team', 'Home_Team'])
    away_col = pick_col(merged_df, ['Away Team', 'Away_Team'])

    # Normalize into temp columns
    merged_df['_bat'] = merged_df[bat_col].astype(str).str.strip().str.lower() if bat_col else ''
    merged_df['_home'] = merged_df[home_col].astype(str).str.strip().str.lower() if home_col else ''
    merged_df['_away'] = merged_df[away_col].astype(str).str.strip().str.lower() if away_col else ''

    # Ensure result flags exist and are numeric
    for flag in ['Home_Win', 'Away_Won', 'Home_Lost', 'Away_Lost', 'Home_Drawn', 'Away_Drawn', 'Tie']:
        if flag not in merged_df.columns:
            merged_df[flag] = 0
        merged_df[flag] = pd.to_numeric(merged_df[flag], errors='coerce').fillna(0).astype(int)

    conditions = [
        (merged_df['_bat'] == merged_df['_home']) & (merged_df['Home_Win'] == 1),
        (merged_df['_bat'] == merged_df['_away']) & (merged_df['Away_Won'] == 1),
        (merged_df['_bat'] == merged_df['_home']) & (merged_df['Home_Lost'] == 1),
        (merged_df['_bat'] == merged_df['_away']) & (merged_df['Away_Lost'] == 1),
        (merged_df['_bat'] == merged_df['_home']) & (merged_df['Home_Drawn'] == 1),
        (merged_df['_bat'] == merged_df['_away']) & (merged_df['Away_Drawn'] == 1),
        (merged_df['Tie'] == 1)
    ]
    choices = ['Won', 'Won', 'Lost', 'Lost', 'Draw', 'Draw', 'Tie']
    merged_df['Match_Result'] = np.select(conditions, choices, default='Unknown')
    merged_df = merged_df[merged_df['Match_Result'].isin(['Won', 'Lost', 'Draw', 'Tie'])]

    if merged_df.empty:
        return {'table': pd.DataFrame(), 'chart_data': pd.DataFrame()}

    # 2. SINGLE, EFFICIENT GROUPBY
    agg_cols = {
        'File Name': 'nunique', 'Batted': 'sum', 'Out': 'sum', 'Not Out': 'sum',
        'Balls': 'sum', 'Runs': 'sum', '4s': 'sum', '6s': 'sum',
        '50s': 'sum', '100s': 'sum', '150s': 'sum', '200s': 'sum'
    }
    result_stats = merged_df.groupby(['Name', 'Match_Result']).agg(agg_cols).reset_index()
    result_stats.rename(columns={'File Name': 'Matches', 'Batted': 'Inns', 'Not Out': 'Not_Out'}, inplace=True)

    # 3. CALCULATE CAREER STATS FROM THE AGGREGATED RESULTS
    career_stats = result_stats.groupby('Name').agg({
        'Matches': 'sum', 'Inns': 'sum', 'Out': 'sum', 'Not_Out': 'sum', 'Balls': 'sum',
        'Runs': 'sum', '4s': 'sum', '6s': 'sum', '50s': 'sum', '100s': 'sum', '150s': 'sum', '200s': 'sum'
    }).reset_index()
    career_stats['Match_Result'] = 'Career'
    # Add High Score for Career
    hs_df = filtered_df.groupby('Name')['Runs'].max().reset_index().rename(columns={'Runs': 'HS'})
    career_stats = career_stats.merge(hs_df, on='Name', how='left')

    # Add High Score for result-specific stats
    hs_result_df = merged_df.groupby(['Name', 'Match_Result'])['Runs'].max().reset_index().rename(columns={'Runs': 'HS'})
    result_stats = result_stats.merge(hs_result_df, on=['Name', 'Match_Result'], how='left')

    # Combine long-format table
    all_stats = pd.concat([career_stats, result_stats], ignore_index=True)

    # 4. VECTORIZED METRIC CALCULATIONS
    all_stats['Avg'] = (all_stats['Runs'] / all_stats['Out'].replace(0, np.nan)).fillna(0).round(2)
    all_stats['SR'] = (all_stats['Runs'] / all_stats['Balls'].replace(0, np.nan) * 100).fillna(0).round(2)
    all_stats['Boundary%'] = (((all_stats['4s'] * 4 + all_stats['6s'] * 6) / all_stats['Runs'].replace(0, np.nan)) * 100).fillna(0).round(2)
    all_stats['50+PI'] = (((all_stats['50s'] + all_stats['100s'] + all_stats['150s'] + all_stats['200s']) / all_stats['Inns'].replace(0, np.nan)) * 100).fillna(0).round(2)

    # Sort and clean up
    result_order = ['Career', 'Won', 'Lost', 'Draw', 'Tie']
    all_stats['Match_Result'] = pd.Categorical(all_stats['Match_Result'], categories=result_order, ordered=True)
    career_runs_map = all_stats[all_stats['Match_Result'] == 'Career'].set_index('Name')['Runs']
    all_stats['Career_Runs_Sort'] = all_stats['Name'].map(career_runs_map)
    all_stats = all_stats.sort_values(['Career_Runs_Sort', 'Name', 'Match_Result'], ascending=[False, True, True]).drop(columns='Career_Runs_Sort')

    # Display columns
    display_columns = ['Name', 'Match_Result', 'Matches', 'Inns', 'Runs', 'HS', 'Avg', 'SR', '50+PI', 'Boundary%', '50s', '100s', '150s', '200s', '4s', '6s']
    final_table = all_stats[display_columns].rename(columns={'Match_Result': 'Result'})

    # 5. PRE-PIVOT DATA FOR CHARTS
    chart_data = final_table.pivot(index='Name', columns='Result', values=['Avg', 'Matches', 'SR']).reset_index()
    chart_data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in chart_data.columns.values]
    chart_data = chart_data.merge(career_runs_map.reset_index(name='Career_Runs'), on='Name', how='left')
    # Avoid filling categorical/text columns with 0 to prevent Categorical dtype errors
    chart_data = chart_data.sort_values('Career_Runs', ascending=False)
    numeric_cols = chart_data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        chart_data[numeric_cols] = chart_data[numeric_cols].fillna(0)

    return {'table': final_table, 'chart_data': chart_data}

@st.cache_data
def compute_format_stats(filtered_df):
    # Group by both 'Name' and 'Match_Format' and sum the required statistics
    df_format = filtered_df.groupby(['Name', 'Match_Format']).agg({
        'File Name': 'nunique',
        'Batted': 'sum',
        'Out': 'sum',
        'Not Out': 'sum',    
        'Balls': 'sum',
        'Runs': ['sum', 'max'],
        '4s': 'sum',
        '6s': 'sum',
        '50s': 'sum',
        '100s': 'sum',
        '200s': 'sum',
        '<25&Out': 'sum',
        'Caught': 'sum',
        'Bowled': 'sum',
        'LBW': 'sum',
        'Run Out': 'sum',
        'Stumped': 'sum',
        'Total_Runs': 'sum',
        'Overs': 'sum',
        'Wickets': 'sum',
        'Team Balls': 'sum'
    }).reset_index()

    # Flatten multi-level columns
    df_format.columns = ['Name', 'Match_Format', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 
                       'Runs', 'HS', '4s', '6s', '50s', '100s', '200s', 
                       '<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 
                       'Team Runs', 'Overs', 'Wickets', 'Team Balls']

    # Calculate average runs per out, strike rate, and balls per out
    df_format['Avg'] = (df_format['Runs'] / df_format['Out'].replace(0, np.nan)).round(2).fillna(0)
    df_format['SR'] = ((df_format['Runs'] / df_format['Balls'].replace(0, np.nan)) * 100).round(2).fillna(0)
    df_format['BPO'] = (df_format['Balls'] / df_format['Out'].replace(0, np.nan)).round(2).fillna(0)

    # Calculate new columns for team statistics
    df_format['Team Avg'] = (df_format['Team Runs'] / df_format['Wickets'].replace(0, np.nan)).round(2).fillna(0)
    df_format['Team SR'] = (df_format['Team Runs'] / df_format['Team Balls'].replace(0, np.nan) * 100).round(2).fillna(0)

    # Calculate P+ Avg and P+ SR
    df_format['P+ Avg'] = (df_format['Avg'] / df_format['Team Avg'].replace(0, np.nan) * 100).round(2).fillna(0)
    df_format['P+ SR'] = (df_format['SR'] / df_format['Team SR'].replace(0, np.nan) * 100).round(2).fillna(0)

    # Calculate BPB (Balls Per Boundary)
    df_format['BPB'] = (df_format['Balls'] / (df_format['4s'] + df_format['6s']).replace(0, 1)).round(2)

    # Calculate Contribution Percentage
    df_format['Avg Contribution %'] = (
        (df_format['Runs'] / df_format['Team Runs'].replace(0, np.nan)) * 100
    ).round(2).fillna(0)

    # Calculate new statistics
    df_format['50+PI'] = (((df_format['50s'] + df_format['100s']) / df_format['Inns']) * 100).round(2).fillna(0)
    df_format['100PI'] = ((df_format['100s'] / df_format['Inns']) * 100).round(2).fillna(0)
    df_format['<25&OutPI'] = ((df_format['<25&Out'] / df_format['Inns']) * 100).round(2).fillna(0)

    # Calculate dismissal percentages
    df_format['Caught%'] = ((df_format['Caught'] / df_format['Inns']) * 100).round(2).fillna(0)
    df_format['Bowled%'] = ((df_format['Bowled'] / df_format['Inns']) * 100).round(2).fillna(0)
    df_format['LBW%'] = ((df_format['LBW'] / df_format['Inns']) * 100).round(2).fillna(0)
    df_format['Run Out%'] = ((df_format['Run Out'] / df_format['Inns']) * 100).round(2).fillna(0)
    df_format['Stumped%'] = ((df_format['Stumped'] / df_format['Inns']) * 100).round(2).fillna(0)
    df_format['Not Out%'] = ((df_format['Not Out'] / df_format['Inns']) * 100).round(2).fillna(0)

    # Reorder columns and drop Team Avg and Team SR
    df_format = df_format[['Name', 'Match_Format', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 'Runs', 'HS', 
                        'Avg', 'BPO', 'SR', '4s', '6s', 'BPB', 'Avg Contribution %', '<25&Out', '50s', '100s', 
                        '<25&OutPI', '50+PI', '100PI', 'P+ Avg', 'P+ SR', 
                        'Caught%', 'Bowled%', 'LBW%', 'Run Out%', 'Stumped%', 'Not Out%']]
    
    # Sort the DataFrame by Runs in descending order
    df_format = df_format.sort_values(by='Runs', ascending=False)
    return df_format

@st.cache_data
def compute_season_stats(filtered_df):
    # Calculate season statistics
    season_stats_df = filtered_df.groupby(['Name', 'Year']).agg({
        'File Name': 'nunique',
        'Batted': 'sum',
        'Out': 'sum',
        'Not Out': 'sum',    
        'Balls': 'sum',
        'Runs': ['sum', 'max'],
        '4s': 'sum',
        '6s': 'sum',
        '50s': 'sum',
        '100s': 'sum',
        '200s': 'sum',
        '<25&Out': 'sum',
        'Caught': 'sum',
        'Bowled': 'sum',
        'LBW': 'sum',
        'Run Out': 'sum',
        'Stumped': 'sum',
        'Total_Runs': 'sum',
        'Overs': 'sum',
        'Wickets': 'sum',
        'Team Balls': 'sum'
    }).reset_index()

    # Flatten multi-level columns
    season_stats_df.columns = ['Name', 'Year', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 
                           'Runs', 'HS', '4s', '6s', '50s', '100s', '200s', 
                           '<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 
                           'Team Runs', 'Overs', 'Wickets', 'Team Balls']

    # Calculate average runs per out, strike rate, and balls per out
    season_stats_df['Avg'] = (season_stats_df['Runs'] / season_stats_df['Out'].replace(0, np.nan)).round(2).fillna(0)
    season_stats_df['SR'] = ((season_stats_df['Runs'] / season_stats_df['Balls'].replace(0, np.nan)) * 100).round(2).fillna(0)
    season_stats_df['BPO'] = (season_stats_df['Balls'] / season_stats_df['Out'].replace(0, np.nan)).round(2).fillna(0)

    # Calculate new columns for team statistics
    season_stats_df['Team Avg'] = (season_stats_df['Team Runs'] / season_stats_df['Wickets'].replace(0, np.nan)).round(2).fillna(0)
    season_stats_df['Team SR'] = (season_stats_df['Team Runs'] / season_stats_df['Team Balls'].replace(0, np.nan) * 100).round(2).fillna(0)

    # Calculate P+ Avg and P+ SR
    season_stats_df['P+ Avg'] = (season_stats_df['Avg'] / season_stats_df['Team Avg'].replace(0, np.nan) * 100).round(2).fillna(0)
    season_stats_df['P+ SR'] = (season_stats_df['SR'] / season_stats_df['Team SR'].replace(0, np.nan) * 100).round(2).fillna(0)

    # Calculate BPB (Balls Per Boundary)
    season_stats_df['BPB'] = (season_stats_df['Balls'] / (season_stats_df['4s'] + season_stats_df['6s']).replace(0, 1)).round(2)

    # Calculate Contribution Percentage
    season_stats_df['Avg Contribution %'] = (
        (season_stats_df['Runs'] / season_stats_df['Team Runs'].replace(0, np.nan)) * 100
    ).round(2).fillna(0)

    # Calculate new statistics
    season_stats_df['50+PI'] = (((season_stats_df['50s'] + season_stats_df['100s']) / season_stats_df['Inns']) * 100).round(2).fillna(0)
    season_stats_df['100PI'] = ((season_stats_df['100s'] / season_stats_df['Inns']) * 100).round(2).fillna(0)
    season_stats_df['<25&OutPI'] = ((season_stats_df['<25&Out'] / season_stats_df['Inns']) * 100).round(2).fillna(0)

    # Calculate dismissal percentages
    season_stats_df['Caught%'] = ((season_stats_df['Caught'] / season_stats_df['Inns']) * 100).round(2).fillna(0)
    season_stats_df['Bowled%'] = ((season_stats_df['Bowled'] / season_stats_df['Inns']) * 100).round(2).fillna(0)
    season_stats_df['LBW%'] = ((season_stats_df['LBW'] / season_stats_df['Inns']) * 100).round(2).fillna(0)
    season_stats_df['Run Out%'] = ((season_stats_df['Run Out'] / season_stats_df['Inns']) * 100).round(2).fillna(0)
    season_stats_df['Stumped%'] = ((season_stats_df['Stumped'] / season_stats_df['Inns']) * 100).round(2).fillna(0)
    season_stats_df['Not Out%'] = ((season_stats_df['Not Out'] / season_stats_df['Inns']) * 100).round(2).fillna(0)

    # Reorder columns
    season_stats_df = season_stats_df[['Name', 'Year', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 'Runs', 'HS', 
                                   'Avg', 'BPO', 'SR', '4s', '6s', 'BPB', 'Avg Contribution %', '<25&Out', '50s', '100s', 
                                   '<25&OutPI', '50+PI', '100PI', 'P+ Avg', 'P+ SR', 
                                   'Caught%', 'Bowled%', 'LBW%', 'Run Out%', 'Stumped%', 'Not Out%']]
    season_stats_df = season_stats_df.sort_values(by='Runs', ascending=False)
    return season_stats_df

@st.cache_data
def compute_opponent_stats(filtered_df):
    # Calculate opponents statistics by grouping
    opponents_stats_df = filtered_df.groupby(['Name', 'Bowl_Team_y']).agg({
        'File Name': 'nunique',
        'Batted': 'sum',
        'Out': 'sum',
        'Not Out': 'sum',    
        'Balls': 'sum',
        'Runs': ['sum', 'max'],
        '4s': 'sum',
        '6s': 'sum',
        '50s': 'sum',
        '100s': 'sum',
        '200s': 'sum',
        '<25&Out': 'sum',
        'Caught': 'sum',
        'Bowled': 'sum',
        'LBW': 'sum',
        'Run Out': 'sum',
        'Stumped': 'sum',
        'Total_Runs': 'sum',
        'Overs': 'sum',
        'Wickets': 'sum',
        'Team Balls': 'sum'
    }).reset_index()

    # Flatten multi-level columns
    opponents_stats_df.columns = ['Name', 'Opposing Team', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 
                                'Runs', 'HS', '4s', '6s', '50s', '100s', '200s', 
                                '<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 
                                'Team Runs', 'Overs', 'Wickets', 'Team Balls']
    return opponents_stats_df

@st.cache_data
def compute_location_stats(filtered_df):
    # Calculate opponents statistics by Home Team
    opponents_stats_df = filtered_df.groupby(['Name', 'Home Team']).agg({
        'File Name': 'nunique',
        'Batted': 'sum',
        'Out': 'sum',
        'Not Out': 'sum',    
        'Balls': 'sum',
        'Runs': ['sum', 'max'],
        '4s': 'sum',
        '6s': 'sum',
        '50s': 'sum',
        '100s': 'sum',
        '200s': 'sum',
        '<25&Out': 'sum',
        'Caught': 'sum',
        'Bowled': 'sum',
        'LBW': 'sum',
        'Run Out': 'sum',
        'Stumped': 'sum',
        'Total_Runs': 'sum',
        'Overs': 'sum',
        'Wickets': 'sum',
        'Team Balls': 'sum'
    }).reset_index()

    # Flatten multi-level columns
    opponents_stats_df.columns = ['Name', 'Location', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 
                                'Runs', 'HS', '4s', '6s', '50s', '100s', '200s', 
                                '<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 
                                'Team Runs', 'Overs', 'Wickets', 'Team Balls']
    # Calculate average runs per out, strike rate, and balls per out
    opponents_stats_df['Avg'] = (opponents_stats_df['Runs'] / opponents_stats_df['Out'].replace(0, np.nan)).round(2).fillna(0)
    opponents_stats_df['SR'] = ((opponents_stats_df['Runs'] / opponents_stats_df['Balls'].replace(0, np.nan)) * 100).round(2).fillna(0)
    opponents_stats_df['BPO'] = (opponents_stats_df['Balls'] / opponents_stats_df['Out'].replace(0, np.nan)).round(2).fillna(0)
    # Calculate new columns for team statistics
    opponents_stats_df['Team Avg'] = (opponents_stats_df['Team Runs'] / opponents_stats_df['Wickets'].replace(0, np.nan)).round(2).fillna(0)
    opponents_stats_df['Team SR'] = (opponents_stats_df['Team Runs'] / opponents_stats_df['Team Balls'].replace(0, np.nan) * 100).round(2).fillna(0)
    # Calculate P+ Avg and P+ SR
    opponents_stats_df['P+ Avg'] = (opponents_stats_df['Avg'] / opponents_stats_df['Team Avg'].replace(0, np.nan) * 100).round(2).fillna(0)
    opponents_stats_df['P+ SR'] = (opponents_stats_df['SR'] / opponents_stats_df['Team SR'].replace(0, np.nan) * 100).round(2).fillna(0)
    # Calculate BPB (Balls Per Boundary)
    opponents_stats_df['BPB'] = (opponents_stats_df['Balls'] / (opponents_stats_df['4s'] + opponents_stats_df['6s']).replace(0, 1)).round(2)
    # Calculate new statistics
    opponents_stats_df['50+PI'] = (((opponents_stats_df['50s'] + opponents_stats_df['100s']) / opponents_stats_df['Inns']) * 100).round(2).fillna(0)
    opponents_stats_df['100PI'] = ((opponents_stats_df['100s'] / opponents_stats_df['Inns']) * 100).round(2).fillna(0)
    opponents_stats_df['<25&OutPI'] = ((opponents_stats_df['<25&Out'] / opponents_stats_df['Inns']) * 100).round(2).fillna(0)
    # Calculate dismissal percentages
    opponents_stats_df['Caught%'] = ((opponents_stats_df['Caught'] / opponents_stats_df['Inns']) * 100).round(2).fillna(0)
    opponents_stats_df['Bowled%'] = ((opponents_stats_df['Bowled'] / opponents_stats_df['Inns']) * 100).round(2).fillna(0)
    opponents_stats_df['LBW%'] = ((opponents_stats_df['LBW'] / opponents_stats_df['Inns']) * 100).round(2).fillna(0)
    opponents_stats_df['Run Out%'] = ((opponents_stats_df['Run Out'] / opponents_stats_df['Inns']) * 100).round(2).fillna(0)
    opponents_stats_df['Stumped%'] = ((opponents_stats_df['Stumped'] / opponents_stats_df['Inns']) * 100).round(2).fillna(0)
    opponents_stats_df['Not Out%'] = ((opponents_stats_df['Not Out'] / opponents_stats_df['Inns']) * 100).round(2).fillna(0)
    # Reorder columns
    opponents_stats_df = opponents_stats_df[['Name', 'Location', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 'Runs', 'HS', 
                                        'Avg', 'BPO', 'SR', '4s', '6s', 'BPB', '<25&Out', '50s', '100s', 
                                        '<25&OutPI', '50+PI', '100PI', 'P+ Avg', 'P+ SR', 
                                        'Caught%', 'Bowled%', 'LBW%', 'Run Out%', 'Stumped%', 'Not Out%']]
    opponents_stats_df = opponents_stats_df.sort_values(by='Runs', ascending=False)
    return opponents_stats_df

@st.cache_data
def compute_innings_stats(filtered_df):
    # Calculate innings statistics by Name
    innings_stats_df = filtered_df.groupby(['Name', 'Innings']).agg({
        'File Name': 'nunique',
        'Batted': 'sum',
        'Out': 'sum',
        'Not Out': 'sum',    
        'Balls': 'sum',
        'Runs': ['sum', 'max'],
        '4s': 'sum',
        '6s': 'sum',
        '50s': 'sum',
        '100s': 'sum',
        '200s': 'sum',
        '<25&Out': 'sum',
        'Caught': 'sum',
        'Bowled': 'sum',
        'LBW': 'sum',
        'Run Out': 'sum',
        'Stumped': 'sum',
        'Total_Runs': 'sum',
        'Overs': 'sum',
        'Wickets': 'sum',
        'Team Balls': 'sum'
    }).reset_index()
    # Flatten multi-level columns
    innings_stats_df.columns = ['Name', 'Innings', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 
                              'Runs', 'HS', '4s', '6s', '50s', '100s', '200s', 
                              '<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 
                              'Team Runs', 'Overs', 'Wickets', 'Team Balls']
    # Calculate average runs per out, strike rate, and balls per out
    innings_stats_df['Avg'] = (innings_stats_df['Runs'] / innings_stats_df['Out'].replace(0, np.nan)).round(2).fillna(0)
    innings_stats_df['SR'] = ((innings_stats_df['Runs'] / innings_stats_df['Balls'].replace(0, np.nan)) * 100).round(2).fillna(0)
    innings_stats_df['BPO'] = (innings_stats_df['Balls'] / innings_stats_df['Out'].replace(0, np.nan)).round(2).fillna(0)
    # Calculate new columns for team statistics
    innings_stats_df['Team Avg'] = (innings_stats_df['Team Runs'] / innings_stats_df['Wickets'].replace(0, np.nan)).round(2).fillna(0)
    innings_stats_df['Team SR'] = (innings_stats_df['Team Runs'] / innings_stats_df['Team Balls'].replace(0, np.nan) * 100).round(2).fillna(0)
    # Calculate P+ Avg and P+ SR
    innings_stats_df['P+ Avg'] = (innings_stats_df['Avg'] / innings_stats_df['Team Avg'].replace(0, np.nan) * 100).round(2).fillna(0)
    innings_stats_df['P+ SR'] = (innings_stats_df['SR'] / innings_stats_df['Team SR'].replace(0, np.nan) * 100).round(2).fillna(0)
    # Calculate BPB (Balls Per Boundary)
    innings_stats_df['BPB'] = (innings_stats_df['Balls'] / (innings_stats_df['4s'] + innings_stats_df['6s']).replace(0, 1)).round(2)
    # Calculate new statistics
    innings_stats_df['50+PI'] = (((innings_stats_df['50s'] + innings_stats_df['100s']) / innings_stats_df['Inns']) * 100).round(2).fillna(0)
    innings_stats_df['100PI'] = ((innings_stats_df['100s'] / innings_stats_df['Inns']) * 100).round(2).fillna(0)
    innings_stats_df['<25&OutPI'] = ((innings_stats_df['<25&Out'] / innings_stats_df['Inns']) * 100).round(2).fillna(0)
    # Calculate dismissal percentages
    innings_stats_df['Caught%'] = ((innings_stats_df['Caught'] / innings_stats_df['Inns']) * 100).round(2).fillna(0)
    innings_stats_df['Bowled%'] = ((innings_stats_df['Bowled'] / innings_stats_df['Inns']) * 100).round(2).fillna(0)
    innings_stats_df['LBW%'] = ((innings_stats_df['LBW'] / innings_stats_df['Inns']) * 100).round(2).fillna(0)
    innings_stats_df['Run Out%'] = ((innings_stats_df['Run Out'] / innings_stats_df['Inns']) * 100).round(2).fillna(0)
    innings_stats_df['Stumped%'] = ((innings_stats_df['Stumped'] / innings_stats_df['Inns']) * 100).round(2).fillna(0)
    innings_stats_df['Not Out%'] = ((innings_stats_df['Not Out'] / innings_stats_df['Inns']) * 100).round(2).fillna(0)
    # Reorder columns
    innings_stats_df = innings_stats_df[['Name', 'Innings', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 
                                       'Runs', 'HS', 'Avg', 'BPO', 'SR', '4s', '6s', 'BPB', 
                                       '<25&Out', '50s', '100s', '<25&OutPI', '50+PI', '100PI', 
                                       'P+ Avg', 'P+ SR', 'Caught%', 'Bowled%', 'LBW%', 
                                       'Run Out%', 'Stumped%', 'Not Out%']]
    innings_stats_df = innings_stats_df.sort_values(by='Innings', ascending=False)
    return innings_stats_df

@st.cache_data
def compute_position_stats(filtered_df):
    # Calculate opponents statistics by Position
    position_stats_df = filtered_df.groupby(['Name', 'Position']).agg({
        'File Name': 'nunique',
        'Batted': 'sum',
        'Out': 'sum',
        'Not Out': 'sum',    
        'Balls': 'sum',
        'Runs': ['sum', 'max'],
        '4s': 'sum',
        '6s': 'sum',
        '50s': 'sum',
        '100s': 'sum',
        '200s': 'sum',
        '<25&Out': 'sum',
        'Caught': 'sum',
        'Bowled': 'sum',
        'LBW': 'sum',
        'Run Out': 'sum',
        'Stumped': 'sum',
        'Total_Runs': 'sum',
        'Overs': 'sum',
        'Wickets': 'sum',
        'Team Balls': 'sum'
    }).reset_index()
    # Flatten multi-level columns
    position_stats_df.columns = ['Name', 'Position', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 
                               'Runs', 'HS', '4s', '6s', '50s', '100s', '200s', 
                               '<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 
                               'Team Runs', 'Overs', 'Wickets', 'Team Balls']
    # Calculate average runs per out, strike rate, and balls per out
    position_stats_df['Avg'] = (position_stats_df['Runs'] / position_stats_df['Out'].replace(0, np.nan)).round(2).fillna(0)
    position_stats_df['SR'] = ((position_stats_df['Runs'] / position_stats_df['Balls'].replace(0, np.nan)) * 100).round(2).fillna(0)
    position_stats_df['BPO'] = (position_stats_df['Balls'] / position_stats_df['Out'].replace(0, np.nan)).round(2).fillna(0)
    # Calculate new columns for team statistics
    position_stats_df['Team Avg'] = (position_stats_df['Team Runs'] / position_stats_df['Wickets'].replace(0, np.nan)).round(2).fillna(0)
    position_stats_df['Team SR'] = (position_stats_df['Team Runs'] / position_stats_df['Team Balls'].replace(0, np.nan) * 100).round(2).fillna(0)
    # Calculate P+ Avg and P+ SR
    position_stats_df['P+ Avg'] = (position_stats_df['Avg'] / position_stats_df['Team Avg'].replace(0, np.nan) * 100).round(2).fillna(0)
    position_stats_df['P+ SR'] = (position_stats_df['SR'] / position_stats_df['Team SR'].replace(0, np.nan) * 100).round(2).fillna(0)
    # Calculate BPB (Balls Per Boundary)
    position_stats_df['BPB'] = (position_stats_df['Balls'] / (position_stats_df['4s'] + position_stats_df['6s']).replace(0, 1)).round(2)
    # Calculate new statistics
    position_stats_df['50+PI'] = (((position_stats_df['50s'] + position_stats_df['100s']) / position_stats_df['Inns']) * 100).round(2).fillna(0)
    position_stats_df['100PI'] = ((position_stats_df['100s'] / position_stats_df['Inns']) * 100).round(2).fillna(0)
    position_stats_df['<25&OutPI'] = ((position_stats_df['<25&Out'] / position_stats_df['Inns']) * 100).round(2).fillna(0)
    # Calculate dismissal percentages
    position_stats_df['Caught%'] = ((position_stats_df['Caught'] / position_stats_df['Inns']) * 100).round(2).fillna(0)
    position_stats_df['Bowled%'] = ((position_stats_df['Bowled'] / position_stats_df['Inns']) * 100).round(2).fillna(0)
    position_stats_df['LBW%'] = ((position_stats_df['LBW'] / position_stats_df['Inns']) * 100).round(2).fillna(0)
    position_stats_df['Run Out%'] = ((position_stats_df['Run Out'] / position_stats_df['Inns']) * 100).round(2).fillna(0)
    position_stats_df['Stumped%'] = ((position_stats_df['Stumped'] / position_stats_df['Inns']) * 100).round(2).fillna(0)
    position_stats_df['Not Out%'] = ((position_stats_df['Not Out'] / position_stats_df['Inns']) * 100).round(2).fillna(0)
    # Reorder columns
    position_stats_df = position_stats_df[['Name', 'Position', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 'Runs', 'HS', 
                                       'Avg', 'BPO', 'SR', '4s', '6s', 'BPB', '<25&Out', '50s', '100s', 
                                       '<25&OutPI', '50+PI', '100PI', 'P+ Avg', 'P+ SR', 
                                       'Caught%', 'Bowled%', 'LBW%', 'Run Out%', 'Stumped%', 'Not Out%']]
    position_stats_df = position_stats_df.sort_values(by='Position', ascending=False)
    return position_stats_df

@st.cache_data
def compute_homeaway_stats(filtered_df):
    # Calculate statistics by HomeOrAway designation
    homeaway_stats_df = filtered_df.groupby(['Name', 'HomeOrAway']).agg({
        'File Name': 'nunique', 'Batted': 'sum', 'Out': 'sum', 'Not Out': 'sum', 'Balls': 'sum',
        'Runs': ['sum', 'max'], '4s': 'sum', '6s': 'sum', '50s': 'sum', '100s': 'sum', '200s': 'sum',
        '<25&Out': 'sum', 'Caught': 'sum', 'Bowled': 'sum', 'LBW': 'sum', 'Run Out': 'sum', 'Stumped': 'sum',
        'Total_Runs': 'sum', 'Overs': 'sum', 'Wickets': 'sum', 'Team Balls': 'sum'
    }).reset_index()
    homeaway_stats_df.columns = ['Name', 'HomeOrAway', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 'Runs', 'HS', 
                                '4s', '6s', '50s', '100s', '200s', '<25&Out', 'Caught', 'Bowled', 'LBW', 
                                'Run Out', 'Stumped', 'Team Runs', 'Overs', 'Wickets', 'Team Balls']

    # --- Vectorized Calculations ---
    out_gt_zero = homeaway_stats_df['Out'] > 0
    inns_gt_zero = homeaway_stats_df['Inns'] > 0
    
    homeaway_stats_df['Avg'] = (homeaway_stats_df['Runs'] / homeaway_stats_df['Out'].replace(0, np.nan)).fillna(0).round(2)
    homeaway_stats_df['SR'] = (homeaway_stats_df['Runs'] / homeaway_stats_df['Balls'].replace(0, np.nan) * 100).fillna(0).round(2)
    # ... (add any other detailed calculations you need for this view) ...
    
    # Sort by Name and then put Career first, followed by Home, Away, and Neutral
    homeaway_stats_df['HomeOrAway_order'] = homeaway_stats_df['HomeOrAway'].map({'Career': 0, 'Home': 1, 'Away': 2, 'Neutral': 3})
    homeaway_stats_df = homeaway_stats_df.sort_values(by=['Name', 'HomeOrAway_order']).drop('HomeOrAway_order', axis=1)
    
    return homeaway_stats_df

@st.cache_data
def compute_cumulative_stats(filtered_df):
    # Performance mode: skip heavy computation on large uploads
    if st.session_state.get('upload_mode', 'small') == 'large':
        return pd.DataFrame()

    # Make a copy to avoid SettingWithCopyWarning
    df = filtered_df.copy()

    # Ensure Date is in datetime format for correct sorting later
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # 1. Group by the essential keys to get stats per player per match.
    #    'File Name' and 'Name' uniquely identify a player's innings in a match.
    #    We use 'first' to carry along the date and format for that match.
    essential_group_cols = ['File Name', 'Name']

    match_level_df = df.groupby(essential_group_cols).agg(
        # Carry these along for sorting and later grouping
        Date=('Date', 'first'),
        Match_Format=('Match_Format', 'first'),
        # Aggregate the actual stats
        Batted=('Batted', 'sum'),
        Out=('Out', 'sum'),
        Balls=('Balls', 'sum'),
        Runs=('Runs', 'sum'),
        Fifties=('50s', 'sum'),
        Hundreds=('100s', 'sum')
    ).reset_index()

    # 2. Sort by Player, Format, and then Date to prepare for cumulative calculation
    match_level_df = match_level_df.sort_values(by=['Name', 'Match_Format', 'Date'])

    # 3. Now calculate the cumulative stats. The data is much smaller and properly structured.
    match_level_df['Cumulative Innings'] = match_level_df.groupby(['Name', 'Match_Format'])['Batted'].cumsum()
    match_level_df['Cumulative Runs'] = match_level_df.groupby(['Name', 'Match_Format'])['Runs'].cumsum()
    match_level_df['Cumulative Balls'] = match_level_df.groupby(['Name', 'Match_Format'])['Balls'].cumsum()
    match_level_df['Cumulative Outs'] = match_level_df.groupby(['Name', 'Match_Format'])['Out'].cumsum()
    match_level_df['Cumulative 100s'] = match_level_df.groupby(['Name', 'Match_Format'])['Hundreds'].cumsum()

    # 4. Calculate running averages and rates
    cumulative_outs = match_level_df['Cumulative Outs'].replace(0, np.nan)
    cumulative_balls = match_level_df['Cumulative Balls'].replace(0, np.nan)
    match_level_df['Cumulative Avg'] = (match_level_df['Cumulative Runs'] / cumulative_outs).fillna(0).round(2)
    match_level_df['Cumulative SR'] = (match_level_df['Cumulative Runs'] / cumulative_balls * 100).fillna(0).round(2)

    return match_level_df.sort_values(by='Date', ascending=False)

@st.cache_data
def compute_block_stats(filtered_df):
    # Performance mode: skip heavy computation on large uploads
    if st.session_state.get('upload_mode', 'small') == 'large':
        return pd.DataFrame()

    # Use the cumulative stats as a base
    cumulative_df = compute_cumulative_stats(filtered_df)

    # Define blocks of 20 matches
    cumulative_df['Match_Block'] = ((cumulative_df['Cumulative Innings'] - 1) // 20)

    # Ensure 'Date' is datetime (prevents min/max errors)
    if 'Date' in cumulative_df.columns:
        cumulative_df['Date'] = pd.to_datetime(cumulative_df['Date'], errors='coerce')

    # Convert groupby columns to string to avoid issues with unordered categoricals
    group_cols = ['Name', 'Match_Format', 'Match_Block']
    for col in group_cols:
        if col in cumulative_df.columns:
            if pd.api.types.is_categorical_dtype(cumulative_df[col]):
                cumulative_df[col] = cumulative_df[col].astype(str)
            elif pd.api.types.is_object_dtype(cumulative_df[col]):
                cumulative_df[col] = cumulative_df[col].astype(str)

    # Ensure aggregation columns are not categorical (esp. 'Batted', 'Runs', 'Balls', 'Out', 'Date')
    agg_cols = ['Batted', 'Runs', 'Balls', 'Out', 'Date']
    for col in agg_cols:
        if col in cumulative_df.columns and pd.api.types.is_categorical_dtype(cumulative_df[col]):
            if col == 'Date':
                cumulative_df[col] = pd.to_datetime(cumulative_df[col].astype(str), errors='coerce')
            else:
                cumulative_df[col] = pd.to_numeric(cumulative_df[col].astype(str), errors='coerce')

    # Group by blocks and calculate stats for that block
    block_stats_df = cumulative_df.groupby(group_cols).agg(
        Matches=('Batted', 'count'),
        Runs=('Runs', 'sum'),
        Balls=('Balls', 'sum'),
        Outs=('Out', 'sum'),
        First_Date=('Date', 'min'),
        Last_Date=('Date', 'max')
    ).reset_index()

    block_stats_df['Avg'] = (block_stats_df['Runs'] / block_stats_df['Outs'].replace(0, np.nan)).fillna(0).round(2)
    block_stats_df['SR'] = (block_stats_df['Runs'] / block_stats_df['Balls'].replace(0, np.nan) * 100).fillna(0).round(2)

    # Create nice labels for the blocks
    block_stats_df['Match_Range'] = block_stats_df['Match_Block'].apply(lambda x: f"{int(x)*20+1}-{int(x)*20+20}")
    block_stats_df['Date_Range'] = block_stats_df['First_Date'].dt.strftime('%d/%m/%Y') + ' to ' + block_stats_df['Last_Date'].dt.strftime('%d/%m/%Y')

    return block_stats_df.sort_values(by=['Name', 'Match_Format', 'Match_Block'])

@st.cache_data
def compute_recent_form(df, by='innings', num_recent=20):
    """
    Computes recent form statistics for each player, per format.

    Args:
        df (pd.DataFrame): The filtered dataframe of batting stats.
        by (str): 'innings' or 'matches'. Determines how to slice recent data.
        num_recent (int): The number of recent innings or matches to consider.

    Returns:
        pd.DataFrame: A dataframe with recent form stats.
    """
    if df.empty:
        return pd.DataFrame()

    # Ensure date is in datetime format for sorting
    df_copy = df.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'], format='%d %b %Y', errors='coerce')
    
    # Sort by date to get the most recent entries first
    df_sorted = df_copy.sort_values(by=['Name', 'Match_Format', 'Date', 'Innings'], ascending=[True, True, False, False])
    
    # Use groupby to get the top N for each player-format combination
    if by == 'innings':
        # Slicing by innings is direct
        recent_df = df_sorted.groupby(['Name', 'Match_Format']).head(num_recent)
    elif by == 'matches':
        # Slicing by matches is more complex
        def get_top_matches(group):
            unique_matches = group.drop_duplicates(subset='File Name')['File Name'].head(num_recent)
            return group[group['File Name'].isin(unique_matches)]
        recent_df = df_sorted.groupby(['Name', 'Match_Format']).apply(get_top_matches).reset_index(drop=True)
    else:
        raise ValueError("Argument 'by' must be 'innings' or 'matches'")

    # Now, aggregate the stats from these recent innings/matches
    recent_stats = recent_df.groupby(['Name', 'Match_Format']).agg(
        Matches=('File Name', 'nunique'),
        Innings=('Innings', 'count'),
        Runs=('Runs', 'sum'),
        Out=('Out', 'sum'),
        Balls=('Balls', 'sum'),
        Fours=('4s', 'sum'),
        Sixes=('6s', 'sum'),
        Fifties=pd.NamedAgg(column='Runs', aggfunc=lambda x: ((x >= 50) & (x < 100)).sum()),
        Hundreds=pd.NamedAgg(column='Runs', aggfunc=lambda x: (x >= 100).sum())
    ).reset_index()

    # Calculate derived stats
    recent_stats['Avg'] = (recent_stats['Runs'] / recent_stats['Out'].replace(0, np.nan)).fillna(0).round(2)
    recent_stats['SR'] = (recent_stats['Runs'] / recent_stats['Balls'].replace(0, np.nan) * 100).fillna(0).round(2)
    recent_stats['BPO'] = (recent_stats['Balls'] / recent_stats['Out'].replace(0, np.nan)).fillna(0).round(2)
    
    # Reorder and rename columns
    final_cols = {
        'Name': 'Name', 'Match_Format': 'Format', 'Matches': 'Matches', 'Innings': 'Innings',
        'Runs': 'Runs', 'Avg': 'Avg', 'SR': 'Strike Rate', 'BPO': 'Balls Per Out',
        'Fifties': '50s', 'Hundreds': '100s', 'Fours': '4s', 'Sixes': '6s'
    }
    recent_stats = recent_stats.rename(columns=final_cols)[final_cols.values()]
    
    return recent_stats.sort_values(['Name', 'Format'])

def display_bat_view():
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

        bat_df = st.session_state['bat_df']
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
                bat_df['comp'] = bat_df['comp'].astype(object).fillna(bat_df['Competition'])
            else:
                bat_df['comp'] = bat_df['Competition']
        else:
            # Fallback: use Competition if merge fails
            bat_df['comp'] = bat_df['Competition']            # Convert Date to datetime once for all future operations
            bat_df['Date'] = pd.to_datetime(bat_df['Date'], format='%d %b %Y', errors='coerce')
            
            # Add milestone innings columns once
            bat_df['50s'] = ((bat_df['Runs'] >= 50) & (bat_df['Runs'] < 100)).astype(int)
            bat_df['100s'] = (bat_df['Runs'] >= 100) & (bat_df['Runs'] < 150).astype(int)
            bat_df['150s'] = ((bat_df['Runs'] >= 150) & (bat_df['Runs'] < 200)).astype(int)
            bat_df['200s'] = (bat_df['Runs'] >= 200).astype(int)
            
            # Convert Year to int once
            bat_df['Year'] = bat_df['Year'].astype(int)
            
            # Add HomeOrAway column to designate home or away matches
            bat_df['HomeOrAway'] = 'Neutral'  # Default value
            # Where Home Team equals Batting Team (convert to string to avoid Categorical comparison error)
            bat_df.loc[bat_df['Home Team'].astype(str) == bat_df['Bat_Team_y'].astype(str), 'HomeOrAway'] = 'Home'
            # Where Away Team equals Batting Team
            bat_df.loc[bat_df['Away Team'].astype(str) == bat_df['Bat_Team_y'].astype(str), 'HomeOrAway'] = 'Away'
            
            st.session_state['processed_bat_df'] = bat_df

        bat_df = st.session_state['processed_bat_df']
            
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
        career_stats = bat_df.groupby('Name').agg({
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
                year_choice = st.slider('', 
                    min_value=min(years),
                    max_value=max(years),
                    value=(min(years), max(years)),
                    label_visibility='collapsed',
                    key='year_slider')

        # Position slider
        with col6:
            st.markdown("<p style='text-align: center;'>Choose Position:</p>", unsafe_allow_html=True)
            position_choice = st.slider('', 
                   min_value=1, 
                   max_value=11, 
                   value=(1, 11),
                   label_visibility='collapsed',
                   key='position_slider')        # Runs range slider
        with col7:
            st.markdown("<p style='text-align: center;'>Runs Range</p>", unsafe_allow_html=True)
            if max_runs == 1:
                st.markdown(f"<p style='text-align: center;'>{max_runs}</p>", unsafe_allow_html=True)
                runs_range = (1, 1)
            else:
                runs_range = st.slider('', 
                                min_value=1, 
                                max_value=max_runs, 
                                value=(1, max_runs),
                                label_visibility='collapsed',
                                key='runs_slider')# Matches range slider
        with col8:
            st.markdown("<p style='text-align: center;'>Matches Range</p>", unsafe_allow_html=True)
            if max_matches == 1:
                st.markdown(f"<p style='text-align: center;'>{max_matches}</p>", unsafe_allow_html=True)
                matches_range = (1, 1)
            else:
                matches_range = st.slider('', 
                                    min_value=1, 
                                    max_value=max_matches, 
                                    value=(1, max_matches),
                                    label_visibility='collapsed',
                                    key='matches_slider')        # Average range slider
        with col9:
            st.markdown("<p style='text-align: center;'>Average Range</p>", unsafe_allow_html=True)
            if max_avg == 0 or pd.isna(max_avg):
                st.markdown("<p style='text-align: center;'>N/A</p>", unsafe_allow_html=True)
                avg_range = (0.0, 0.0)
            elif max_avg == 0.0:
                st.markdown(f"<p style='text-align: center;'>{max_avg:.1f}</p>", unsafe_allow_html=True)
                avg_range = (0.0, 0.0)
            else:
                avg_range = st.slider('', 
                                min_value=0.0, 
                                max_value=max_avg, 
                                value=(0.0, max_avg),
                                label_visibility='collapsed',
                                key='avg_slider')

        # Strike rate range slider
        with col10:
            st.markdown("<p style='text-align: center;'>Strike Rate Range</p>", unsafe_allow_html=True)
            sr_range = st.slider('', 
                            min_value=0.0, 
                            max_value=600.0, 
                            value=(0.0, 600.0),
                            label_visibility='collapsed',
                            key='sr_slider')

        # Add P+ Avg range slider
        with col11:
            st.markdown("<p style='text-align: center;'>P+ Avg Range</p>", unsafe_allow_html=True)
            p_avg_range = st.slider('', 
                            min_value=0.0, 
                            max_value=500.0,  # Changed from 200.0 to 500.0
                            value=(0.0, 500.0),  # Updated range to match new maximum
                            label_visibility='collapsed',
                            key='p_avg_slider')

        # Add P+ SR range slider
        with col12:
            st.markdown("<p style='text-align: center;'>P+ SR Range</p>", unsafe_allow_html=True)
            p_sr_range = st.slider('', 
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

        # Directly calculate filtered_df (no Redis/cache)
        filtered_df = bat_df.copy()
        # Use vectorized operations for filtering instead of chained conditionals
        mask = np.ones(len(filtered_df), dtype=bool)
        # Treat 'All' as no filter; if specific values are selected alongside 'All', ignore 'All'
        eff_names = [x for x in name_choice if x != 'All'] if name_choice else []
        if eff_names:
            mask &= filtered_df['Name'].isin(eff_names)

        eff_bat_teams = [x for x in bat_team_choice if x != 'All'] if bat_team_choice else []
        if eff_bat_teams:
            mask &= filtered_df['Bat_Team_y'].isin(eff_bat_teams)

        eff_bowl_teams = [x for x in bowl_team_choice if x != 'All'] if bowl_team_choice else []
        if eff_bowl_teams:
            mask &= filtered_df['Bowl_Team_y'].isin(eff_bowl_teams)

        eff_formats = [x for x in match_format_choice if x != 'All'] if match_format_choice else []
        if eff_formats:
            mask &= filtered_df['Match_Format'].isin(eff_formats)

        eff_comp = [x for x in comp_choice if x != 'All'] if comp_choice else []
        if eff_comp:
            mask &= filtered_df['comp'].isin(eff_comp)
        # Apply range filters
        mask &= filtered_df['Year'].between(year_choice[0], year_choice[1])
        mask &= filtered_df['Position'].between(position_choice[0], position_choice[1])
        filtered_df = filtered_df[mask]
        # Ensure 'Name' is plain string to prevent categorical groupby from adding unobserved categories
        try:
            filtered_df['Name'] = filtered_df['Name'].astype(str)
        except Exception:
            pass
        # Ensure HomeOrAway column exists in the filtered DataFrame
        if 'HomeOrAway' not in filtered_df.columns:
            filtered_df['HomeOrAway'] = 'Neutral'  # Default value
            # Where Home Team equals Batting Team (convert to string to avoid Categorical comparison error)
            filtered_df.loc[filtered_df['Home Team'].astype(str) == filtered_df['Bat_Team_y'].astype(str), 'HomeOrAway'] = 'Home'
            # Where Away Team equals Batting Team
            filtered_df.loc[filtered_df['Away Team'].astype(str) == filtered_df['Bat_Team_y'].astype(str), 'HomeOrAway'] = 'Away'
        # Group-based filters using a more efficient approach
        player_stats = filtered_df.groupby('Name').agg({
            'File Name': 'nunique',
            'Runs': 'sum',
            'Out': 'sum',
            'Balls': 'sum',
            'Total_Runs': 'sum',
            'Wickets': 'sum',
            'Team Balls': 'sum'
        })
        # Apply range filters to the player stats
        mask = (
            (player_stats['Runs'] >= runs_range[0]) &
            (player_stats['Runs'] <= runs_range[1]) &
            (player_stats['File Name'] >= matches_range[0]) &
            (player_stats['File Name'] <= matches_range[1])
        )
        # Apply conditional filters only when Out > 0
        non_zero_out = player_stats['Out'] > 0
        if non_zero_out.any():
            avg_mask = (player_stats[non_zero_out]['Runs'] / player_stats[non_zero_out]['Out']).between(avg_range[0], avg_range[1])
            mask.loc[non_zero_out] &= avg_mask
        # Apply conditional filters only when Balls > 0
        non_zero_balls = player_stats['Balls'] > 0
        if non_zero_balls.any():
            sr_mask = ((player_stats[non_zero_balls]['Runs'] / player_stats[non_zero_balls]['Balls']) * 100).between(sr_range[0], sr_range[1])
            mask.loc[non_zero_balls] &= sr_mask
        # Calculate P+ metrics for filtering
        non_zero_wickets = player_stats['Wickets'] > 0
        non_zero_team_balls = player_stats['Team Balls'] > 0
        if non_zero_wickets.any() and non_zero_out.any():
            p_avg = (player_stats[non_zero_wickets & non_zero_out]['Runs'] / player_stats[non_zero_wickets & non_zero_out]['Out']) / (player_stats[non_zero_wickets & non_zero_out]['Total_Runs'] / player_stats[non_zero_wickets & non_zero_out]['Wickets']) * 100
            p_avg_mask = p_avg.between(p_avg_range[0], p_avg_range[1])
            mask.loc[non_zero_wickets & non_zero_out] &= p_avg_mask
        if non_zero_team_balls.any() and non_zero_balls.any():
            p_sr = ((player_stats[non_zero_team_balls & non_zero_balls]['Runs'] / player_stats[non_zero_team_balls & non_zero_balls]['Balls']) / (player_stats[non_zero_team_balls & non_zero_balls]['Total_Runs'] / player_stats[non_zero_team_balls & non_zero_balls]['Team Balls'])) * 100
            p_sr_mask = p_sr.between(p_sr_range[0], p_sr_range[1])
            mask.loc[non_zero_team_balls & non_zero_balls] &= p_sr_mask
        # Get the filtered player list
        filtered_players = player_stats[mask].index
        # Apply player filter to the main dataframe
        filtered_df = filtered_df[filtered_df['Name'].isin(filtered_players)]

        # Create a placeholder for tabs that will be lazily loaded
        main_container = st.container()

        # Dynamic tab generation based on upload mode
        upload_mode = st.session_state.get('upload_mode', 'small')
        full_feature_tabs = [
            "Career", "Format", "Season", "Latest", "Opponent",
            "Location", "Innings", "Position", "Home/Away", "Match Impact",
            "Cumulative", "Block"
        ]
        lite_feature_tabs = [
            "Career", "Format", "Season", "Latest", "Opponent",
            "Location", "Innings", "Position", "Home/Away"
        ]

        if upload_mode == 'large':
            st.info(
                "💡 Performance Mode: Large dataset detected. Disabling memory-intensive tabs "
                "(Match Impact, Cumulative, Block). Upload a smaller batch for full features."
            )
            tabs = main_container.tabs(lite_feature_tabs)
        else:
            tabs = main_container.tabs(full_feature_tabs)
        
        # Career Stats Tab
        with tabs[0]:
            bat_career_df = compute_career_stats(filtered_df)
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">🏏 Career Statistics</h3>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(bat_career_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})
            
            # Skip heavy charts in large upload mode
            if upload_mode != 'large':
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

        # Match Impact / Clutch Performance Tab (only in full mode)
        if upload_mode != 'large':
            with tabs[9]:
                # Try to get match impact stats
                match_impact_result = compute_match_impact_stats(filtered_df, match_df)
                match_impact_df = match_impact_result.get('table', pd.DataFrame())
                chart_data = match_impact_result.get('chart_data', pd.DataFrame())
                
                if not match_impact_df.empty:
                    st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">🎯 Match Impact / Clutch Performance</h3>
                    <p style="color: white !important; margin: 0.5rem 0 0 0; text-align: center; font-size: 0.9rem;">
                        Complete batting statistics broken down by match results - Career vs Won vs Lost vs Draw performance
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add explanatory guide
                st.markdown("""
                <div style="background: rgba(102, 126, 234, 0.1); 
                            padding: 0.8rem; margin: 0.5rem 0; border-radius: 10px; 
                            border-left: 4px solid #667eea;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #667eea;">📊 Statistics Breakdown:</h4>
                    <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.85rem;">
                        <li><strong>Career:</strong> Overall statistics across all matches</li>
                        <li><strong>Won:</strong> Performance in matches where player's team won</li>
                        <li><strong>Lost:</strong> Performance in matches where player's team lost</li>
                        <li><strong>Draw:</strong> Performance in matches that ended in draws</li>
                        <li><strong>Key Metrics:</strong> Matches, Innings, Runs, Average, Strike Rate, High Score, Milestones, etc.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Configure column display with comprehensive formatting
                column_config = {
                    "Name": st.column_config.Column("Name", pinned=True, width=120),
                    "Result": st.column_config.Column("Result", width=80),
                    "Matches": st.column_config.NumberColumn("Matches", format="%d", width=80),
                    "Inns": st.column_config.NumberColumn("Inns", format="%d", width=70),
                    "Runs": st.column_config.NumberColumn("Runs", format="%d", width=80),
                    "HS": st.column_config.NumberColumn("HS", format="%d", width=70),
                    "Avg": st.column_config.NumberColumn("Avg", format="%.2f", width=80),
                    "SR": st.column_config.NumberColumn("SR", format="%.2f", width=80),
                    "50+PI": st.column_config.NumberColumn("50+PI", format="%.2f", width=80),
                    "Boundary%": st.column_config.NumberColumn("Boundary%", format="%.2f", width=90),
                    "50s": st.column_config.NumberColumn("50s", format="%d", width=60),
                    "100s": st.column_config.NumberColumn("100s", format="%d", width=70),
                    "150s": st.column_config.NumberColumn("150s", format="%d", width=70),
                    "200s": st.column_config.NumberColumn("200s", format="%d", width=70),
                    "4s": st.column_config.NumberColumn("4s", format="%d", width=60),
                    "6s": st.column_config.NumberColumn("6s", format="%d", width=60)
                }
                
                st.dataframe(
                    match_impact_df, 
                    use_container_width=True, 
                    hide_index=True, 
                    column_config=column_config
                )
                
                # Create comprehensive bar chart showing all averages first
                st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 0.8rem; margin: 1.5rem 0 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(102, 126, 234, 0.25);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">📊 Complete Average Comparison: Career vs Match Results</h3>
                    <p style="color: white !important; margin: 0.3rem 0 0 0; text-align: center; font-size: 0.8rem;">
                        Compare batting averages across all match outcomes
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Use pre-computed chart data for better performance
                top_players = pd.DataFrame()  # Initialize to empty DataFrame
                
                if not chart_data.empty:
                    # Apply same filtering as scatter plot - require at least 2 matches won AND 2 matches lost
                    # Check if required columns exist first
                    has_required_cols = all(col in chart_data.columns for col in ['Matches_Won', 'Matches_Lost'])
                    
                    if has_required_cols:
                        bar_chart_qualified = chart_data[
                            (chart_data['Matches_Won'] >= 2) & 
                            (chart_data['Matches_Lost'] >= 2)
                        ]
                        
                        # Sort by career runs (descending) and take qualified players
                        if 'Career_Runs' in bar_chart_qualified.columns:
                            bar_chart_qualified = bar_chart_qualified.sort_values('Career_Runs', ascending=False)
                        
                        # Use qualified players for consistency with scatter plot
                        top_players = bar_chart_qualified
                    else:
                        top_players = pd.DataFrame()  # Empty if required columns don't exist
                
                if not top_players.empty:
                    # Create grouped bar chart
                    bar_fig = go.Figure()
                    
                    # Add bars for each category - use proper column access with fallback
                    career_avg = top_players['Avg_Career'] if 'Avg_Career' in top_players.columns else [0] * len(top_players)
                    won_avg = top_players['Avg_Won'] if 'Avg_Won' in top_players.columns else [0] * len(top_players)
                    lost_avg = top_players['Avg_Lost'] if 'Avg_Lost' in top_players.columns else [0] * len(top_players)
                    draw_avg = top_players['Avg_Draw'] if 'Avg_Draw' in top_players.columns else [0] * len(top_players)
                    
                    bar_fig.add_trace(go.Bar(
                        name='Career Average',
                        x=top_players['Name'],
                        y=career_avg,
                        marker_color='#36d1dc',
                        hovertemplate='<b>%{x}</b><br>Career Avg: %{y:.2f}<extra></extra>'
                    ))
                    
                    bar_fig.add_trace(go.Bar(
                        name='Won Average',
                        x=top_players['Name'],
                        y=won_avg,
                        marker_color='#28a745',
                        hovertemplate='<b>%{x}</b><br>Won Avg: %{y:.2f}<extra></extra>'
                    ))
                    
                    bar_fig.add_trace(go.Bar(
                        name='Lost Average',
                        x=top_players['Name'],
                        y=lost_avg,
                        marker_color='#dc3545',
                        hovertemplate='<b>%{x}</b><br>Lost Avg: %{y:.2f}<extra></extra>'
                    ))
                    
                    bar_fig.add_trace(go.Bar(
                        name='Draw Average',
                        x=top_players['Name'],
                        y=draw_avg,
                        marker_color='#ffc107',
                        hovertemplate='<b>%{x}</b><br>Draw Avg: %{y:.2f}<extra></extra>'
                    ))
                    
                    # Update layout
                    bar_fig.update_layout(
                        xaxis_title="Players",
                        yaxis_title="Batting Average",
                        height=600,
                        barmode='group',
                        font=dict(size=12),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        xaxis=dict(
                            tickangle=45,
                            tickmode='linear'
                        )
                    )
                    
                    st.plotly_chart(bar_fig, use_container_width=True, key="comprehensive_avg_comparison")
                    
                    # Add insights below the chart
                    st.markdown("#### 🔍 Key Insights:")
                    
                    # Calculate some interesting statistics
                    if 'Avg_Won' in top_players.columns and 'Avg_Career' in top_players.columns:
                        clutch_performers = top_players[top_players['Avg_Won'] > top_players['Avg_Career']]
                    else:
                        clutch_performers = pd.DataFrame()
                        
                    if 'Avg_Lost' in top_players.columns and 'Avg_Career' in top_players.columns:
                        pressure_performers = top_players[top_players['Avg_Lost'] > top_players['Avg_Career']]
                    else:
                        pressure_performers = pd.DataFrame()
                        
                    if all(col in top_players.columns for col in ['Avg_Won', 'Avg_Lost', 'Avg_Career']):
                        consistent_performers = top_players[
                            (abs(top_players['Avg_Won'] - top_players['Avg_Career']) <= 5) & 
                            (abs(top_players['Avg_Lost'] - top_players['Avg_Career']) <= 5)
                        ]
                    else:
                        consistent_performers = pd.DataFrame()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div style="background: rgba(40, 167, 69, 0.1); padding: 1rem; border-radius: 10px; text-align: center; border: 2px solid #28a745;">
                            <h4 style="color: #28a745; margin: 0;">🎯 Big Game Players</h4>
                            <p style="font-size: 1.2rem; font-weight: bold; margin: 0.5rem 0;">{len(clutch_performers)}</p>
                            <p style="font-size: 0.9rem; margin: 0;">Better when winning</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style="background: rgba(220, 53, 69, 0.1); padding: 1rem; border-radius: 10px; text-align: center; border: 2px solid #dc3545;">
                            <h4 style="color: #dc3545; margin: 0;">⚡ Pressure Players</h4>
                            <p style="font-size: 1.2rem; font-weight: bold; margin: 0.5rem 0;">{len(pressure_performers)}</p>
                            <p style="font-size: 0.9rem; margin: 0;">Better when losing</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div style="background: rgba(54, 209, 220, 0.1); padding: 1rem; border-radius: 10px; text-align: center; border: 2px solid #36d1dc;">
                            <h4 style="color: #36d1dc; margin: 0;">🔄 Consistent Players</h4>
                            <p style="font-size: 1.2rem; font-weight: bold; margin: 0.5rem 0;">{len(consistent_performers)}</p>
                            <p style="font-size: 0.9rem; margin: 0;">Similar across results</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Use pre-computed chart data for scatter plot as well
                # The chart_data already has the Won vs Lost data pivoted
                required_clutch_cols = ['Name', 'Avg_Won', 'Avg_Lost', 'Matches_Won', 'Matches_Lost']
                if not chart_data.empty and all(col in chart_data.columns for col in required_clutch_cols):
                    clutch_data = chart_data[required_clutch_cols].copy()
                else:
                    clutch_data = pd.DataFrame()  # Empty if required columns don't exist
                
                # Remove the duplicate bar chart section - this was duplicated by mistake
                
                # Original scatter plot for Won vs Lost comparison
                if not clutch_data.empty and 'Avg_Won' in clutch_data.columns and 'Avg_Lost' in clutch_data.columns:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 0.8rem; margin: 1.5rem 0 1rem 0; border-radius: 12px; 
                                box-shadow: 0 6px 24px rgba(102, 126, 234, 0.25);
                                border: 1px solid rgba(255, 255, 255, 0.2);">
                        <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">📈 Clutch Performance Analysis: Average in Won vs Lost Matches</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
            # Filter players with sufficient data for meaningful analysis
            # Require at least 2 matches in both categories
                    sufficient_data = clutch_data[
                (clutch_data.get('Matches_Won', 0) >= 2) & 
                (clutch_data.get('Matches_Lost', 0) >= 2) &
                        (clutch_data.get('Avg_Won', 0) > 0) & 
                        (clutch_data.get('Avg_Lost', 0) > 0)
                    ]
                    
                    # Show scatter plot criteria and results
                    st.info(f"""
                    **📈 Scatter Plot Inclusion Criteria:**
                    - ✅ All players included (no run/ranking limits)
                    - ✅ At least 2 matches won AND 2 matches lost
                    - ✅ Positive batting averages in both won and lost matches
                    
                    **📊 Result:** {len(sufficient_data)} of {len(clutch_data)} total players qualify for scatter plot analysis
                    """)
                    
                    if not sufficient_data.empty:
                        # Create clutch performance scatter plot using the same logic as summary cards
                        clutch_fig = go.Figure()
                        
                        # Calculate quartile-based categorization for guaranteed distribution
                        sufficient_data = sufficient_data.copy()
                        sufficient_data['won_diff'] = sufficient_data['Avg_Won'] - sufficient_data.get('Avg_Career', 0)
                        sufficient_data['lost_diff'] = sufficient_data['Avg_Lost'] - sufficient_data.get('Avg_Career', 0)
                        sufficient_data['clutch_score'] = sufficient_data['won_diff'] - sufficient_data['lost_diff']
                        
                        # Sort by clutch score and assign categories based on distribution
                        sufficient_data_sorted = sufficient_data.sort_values('clutch_score', ascending=False)
                        total_players = len(sufficient_data_sorted)
                        
                        # Force distribution: Top 30% = Big Game, Bottom 30% = Pressure, Middle 40% = Consistent
                        big_game_cutoff = int(total_players * 0.3)
                        pressure_cutoff = int(total_players * 0.7)
                        
                        # Use the chart_data directly for scatter plot efficiency
                        for idx, (_, player_row) in enumerate(sufficient_data_sorted.iterrows()):
                            player_name = player_row['Name']
                            won_avg = player_row.get('Avg_Won', 0)
                            lost_avg = player_row.get('Avg_Lost', 0)
                            career_avg = player_row.get('Avg_Career', 0)
                            won_diff = player_row['won_diff']
                            lost_diff = player_row['lost_diff']
                            clutch_score = player_row['clutch_score']
                            
                            # Assign category based on position in sorted list
                            if idx < big_game_cutoff:
                                color = '#28a745'  # Green for Big Game Players
                                category = 'Big Game Player'
                            elif idx >= pressure_cutoff:
                                color = '#dc3545'  # Red for Pressure Players  
                                category = 'Pressure Player'
                            else:
                                color = '#36d1dc'  # Cyan for Consistent Players
                                category = 'Consistent Player'
                            
                            # Show debug info for verification
                            if total_players <= 15:  # Only for small datasets
                                st.write(f"🔍 {player_name}: Clutch Score={clutch_score:.2f}, Category={category}, Rank={idx+1}/{total_players}")
                            
                            clutch_fig.add_trace(go.Scatter(
                                x=[lost_avg],
                                y=[won_avg], 
                                mode='markers+text',
                                text=[player_name],
                                textposition='top center',
                                marker=dict(
                                    size=12,
                                    color=color,
                                    line=dict(width=2, color='white')
                                ),
                                name=player_name,
                                hovertemplate=(
                                    f"<b>{player_name}</b><br><br>"
                                    f"Category: {category}<br>"
                                    f"Career Average: {career_avg:.2f}<br>"
                                    f"Average in Won Matches: {won_avg:.2f}<br>"
                                    f"Average in Lost Matches: {lost_avg:.2f}<br>"
                                    f"Won vs Career Diff: {won_diff:+.2f}<br>"
                                    f"Lost vs Career Diff: {lost_diff:+.2f}<br>"
                                    f"Won Matches: {player_row.get('Matches_Won', 0)}<br>"
                                    f"Lost Matches: {player_row.get('Matches_Lost', 0)}<br>"
                                    "<extra></extra>"
                                ),
                                showlegend=False
                            ))
                        
                        # Add diagonal line (x=y) for equal performance
                        max_val = max(sufficient_data['Avg_Won'].max(), sufficient_data['Avg_Lost'].max())
                        min_val = min(sufficient_data['Avg_Won'].min(), sufficient_data['Avg_Lost'].min())
                        
                        clutch_fig.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            line=dict(dash='dash', color='gray', width=2),
                            name='Equal Performance Line',
                            hoverinfo='skip',
                            showlegend=False
                        ))
                        
                        # Calculate dynamic axis ranges with some padding
                        x_min, x_max = sufficient_data['Avg_Lost'].min(), sufficient_data['Avg_Lost'].max()
                        y_min, y_max = sufficient_data['Avg_Won'].min(), sufficient_data['Avg_Won'].max()
                        
                        # Add 10% padding to the ranges for better visualization
                        x_padding = (x_max - x_min) * 0.1 if x_max > x_min else 5
                        y_padding = (y_max - y_min) * 0.1 if y_max > y_min else 5
                        
                        clutch_fig.update_layout(
                            xaxis_title="Average in Lost Matches",
                            yaxis_title="Average in Won Matches",
                            height=500,
                            font=dict(size=12),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(
                                range=[x_min - x_padding, x_max + x_padding],
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='rgba(128, 128, 128, 0.2)'
                            ),
                            yaxis=dict(
                                range=[y_min - y_padding, y_max + y_padding],
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='rgba(128, 128, 128, 0.2)'
                            )
                        )
                        
                        st.plotly_chart(clutch_fig, use_container_width=True, key="clutch_performance")
                        
                    # Add detailed explanation section
                    st.markdown("---")
                    st.markdown("#### 📚 Understanding Clutch Performance Analysis")
                    
                    # Main container with styling
                    st.markdown("""
                    <div style="background: rgba(102, 126, 234, 0.05); 
                                padding: 1.5rem; margin: 1rem 0; border-radius: 15px; 
                                border-left: 5px solid #667eea;">
                        <h4 style="color: #667eea; margin-top: 0;">🎯 What These Metrics Tell Us:</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Big Game Players section
                    st.markdown("""
                    <div style="margin-bottom: 1rem;">
                        <h5 style="color: #28a745; margin-bottom: 0.5rem;">🎯 Big Game Players / Clutch Performers</h5>
                        <p style="margin: 0; font-size: 0.9rem; line-height: 1.4;">
                            These players <strong>raise their game when it matters most</strong>. They perform better in winning matches, 
                            suggesting they contribute significantly to team victories. These are the players you want batting in crucial 
                            situations - they thrive under pressure and help secure wins.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Pressure Players section
                    st.markdown("""
                    <div style="margin-bottom: 1rem;">
                        <h5 style="color: #dc3545; margin-bottom: 0.5rem;">⚡ Pressure Players / Flat-Track Bullies</h5>
                        <p style="margin: 0; font-size: 0.9rem; line-height: 1.4;">
                            These players actually perform <strong>better in losing matches</strong>. This might indicate they score runs 
                            when the pressure is off or when the match situation is already difficult. While still valuable, 
                            they may struggle to convert good starts into match-winning performances.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Consistent Players section
                    st.markdown("""
                    <div style="margin-bottom: 1rem;">
                        <h5 style="color: #36d1dc; margin-bottom: 0.5rem;">🔄 Consistent Players</h5>
                        <p style="margin: 0; font-size: 0.9rem; line-height: 1.4;">
                            These players maintain <strong>similar performance levels regardless of match outcome</strong>. 
                            They're reliable and steady, providing consistent contributions whether the team wins or loses. 
                            These players form the backbone of any batting lineup.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Average Difference section
                    st.markdown("""
                    <div style="margin-bottom: 1rem;">
                        <h5 style="color: #667eea; margin-bottom: 0.5rem;">📈 Average Difference</h5>
                        <p style="margin: 0; font-size: 0.9rem; line-height: 1.4;">
                            Shows the overall trend across all players. A <strong>positive value</strong> indicates that collectively, 
                            players perform better in winning matches. A negative value would suggest better performance in losses. 
                            The magnitude shows how significant this difference is.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Strategic Insights section
                    st.markdown("""
                    <div style="background: rgba(255, 193, 7, 0.1); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #ffc107;">
                        <h5 style="color: #856404; margin-top: 0;">💡 Strategic Insights:</h5>
                        <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.85rem; line-height: 1.4;">
                            <li><strong>Team Selection:</strong> Clutch performers are ideal for crucial matches and pressure situations</li>
                            <li><strong>Batting Order:</strong> Position clutch players where they can influence close games</li>
                            <li><strong>Match Strategy:</strong> Consistent players provide stability, while clutch players can change game momentum</li>
                            <li><strong>Player Development:</strong> Help pressure players work on converting starts into match-winning contributions</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.info("📊 Not enough data for clutch performance analysis. Players need at least 2 matches in both won and lost categories.")
                
                if match_impact_df.empty:
                    st.warning("⚠️ Match impact analysis requires match result data. Please ensure match data is loaded.")
                    
                    # Provide troubleshooting info
                    if match_df.empty:
                        st.error("**Issue:** Match data is empty. Please re-upload your scorecard files from the Home page.")
                    elif filtered_df.empty:
                        st.error("**Issue:** No batting data available with current filters.")
                    else:
                        st.markdown("### 🔧 Troubleshooting:")
                        common_files = 0
                        if 'File Name' in filtered_df.columns and 'File Name' in match_df.columns:
                            bat_files = set(filtered_df['File Name'].unique())
                            match_files = set(match_df['File Name'].unique())
                            common_files = len(bat_files.intersection(match_files))
                        
                        st.info(f"**Common files between batting and match data:** {common_files}")
                        if common_files == 0:
                            st.error("**Issue:** No matching files between batting and match data")
                        else:
                            st.success(f"Found {common_files} matching files - function should work!")

        # Format Stats Tab  
        with tabs[1]:
            df_format = compute_format_stats(filtered_df)

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
                yearly_format_stats = format_data.groupby('Year').agg({
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
            season_stats_df = compute_season_stats(filtered_df)

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
            yearly_stats = filtered_df.groupby(['Name', 'Year']).agg({
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
                all_players_stats = yearly_stats.groupby('Year').agg({
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

        # Latest Form Analysis Tab
        with tabs[3]:
            # --- Part 1: Overall Last 20 Innings ---
            st.markdown("""
            <div class="section-header">
                <h3>⚡ Recent Form (Last 20 Innings Overall)</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Correctly get the last 20 innings across all formats from the full filtered_df
            latest_innings_df = (filtered_df.sort_values('Date', ascending=False)
                                 .head(20)
                                 .reset_index(drop=True))
            
            if not latest_innings_df.empty:
                # --- Summary Metrics for Overall Recent Form ---
                # --- START OF THE FIX ---
                total_runs = latest_innings_df['Runs'].sum()
                total_balls = latest_innings_df['Balls'].sum()
                total_outs = latest_innings_df['Out'].sum()
                total_innings = len(latest_innings_df)
                total_matches = latest_innings_df['File Name'].nunique()
                total_50s = latest_innings_df['50s'].sum()
                total_100s = latest_innings_df['100s'].sum()
                
                calculated_avg = total_runs / total_outs if total_outs > 0 else 0
                calculated_sr = (total_runs / total_balls * 100) if total_balls > 0 else 0
                # --- END OF THE FIX ---

                st.markdown("<h5>📊 Summary Statistics</h5>", unsafe_allow_html=True)

                # Define custom CSS for the metric grid
                st.markdown("""
                <style>
                .metric-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                    gap: 1rem;
                }
                .metric-box {
                    border: 1px solid #444;
                    border-radius: 8px;
                    padding: 1rem;
                    text-align: center;
                }
                .metric-label {
                    font-size: 0.9rem;
                    color: black;
                    margin-bottom: 0.5rem;
                }
                .metric-value {
                    font-size: 1.75rem;
                    font-weight: bold;
                    color: black;
                }
                </style>
                """, unsafe_allow_html=True)

                # Create the grid using HTML
                metrics_html = f"""
                <div class="metric-grid">
                    <div class="metric-box">
                        <div class="metric-label">Matches</div>
                        <div class="metric-value">{total_matches}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Innings</div>
                        <div class="metric-value">{total_innings}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Outs</div>
                        <div class="metric-value">{total_outs}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Runs</div>
                        <div class="metric-value">{total_runs}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Balls</div>
                        <div class="metric-value">{total_balls}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">50s</div>
                        <div class="metric-value">{total_50s}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">100s</div>
                        <div class="metric-value">{total_100s}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Average</div>
                        <div class="metric-value">{calculated_avg:.2f}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Strike Rate</div>
                        <div class="metric-value">{calculated_sr:.2f}</div>
                    </div>
                </div>
                """
                st.markdown(metrics_html, unsafe_allow_html=True)
                
                # --- Dataframe of Recent Innings ---
                st.markdown("<h5 style='margin-top: 1.5rem;'>📋 Recent Innings Details</h5>", unsafe_allow_html=True)
                display_latest = latest_innings_df[['Name', 'Match_Format', 'Date', 'Bat_Team_y', 'Bowl_Team_y', 'How Out', 'Runs', 'Balls', '4s', '6s']].copy()
                display_latest['Date'] = display_latest['Date'].dt.strftime('%d/%m/%Y')
                
                def style_runs_column(val):
                    if val <= 20: return 'background-color: #ffebee; color: #c62828;'
                    elif 21 <= val <= 49: return 'background-color: #fff3e0; color: #ef6c00;'
                    elif 50 <= val < 100: return 'background-color: #e8f5e8; color: #2e7d32;'
                    elif val >= 100: return 'background-color: #e3f2fd; color: #1565c0;'
                    return ''
                    
                st.dataframe(
                    display_latest.style.applymap(style_runs_column, subset=['Runs']),
                    use_container_width=True, hide_index=True
                )
            else:
                st.info("Not enough data to show recent innings.")

            # --- Part 2: Recent Form by Format ---
            st.markdown("""
            <div class="section-header" style='margin-top: 2rem;'>
                <h3>📊 Recent Form Breakdown by Format</h3>
            </div>
            """, unsafe_allow_html=True)

            sub_tabs = st.tabs(["Last 20 Innings", "Last 40 Innings", "Last 20 Matches"])

            with sub_tabs[0]:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                            box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">📊 Last 20 Innings by Format</h3>
                </div>
                """, unsafe_allow_html=True)
                
                recent_innings_20 = compute_recent_form(filtered_df, by='innings', num_recent=20)
                if not recent_innings_20.empty:
                    st.dataframe(recent_innings_20, use_container_width=True, hide_index=True)
                else:
                    st.info("No data available for this breakdown.")

            with sub_tabs[1]:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                            box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">📊 Last 40 Innings by Format</h3>
                </div>
                """, unsafe_allow_html=True)

                recent_innings_40 = compute_recent_form(filtered_df, by='innings', num_recent=40)
                if not recent_innings_40.empty:
                    st.dataframe(recent_innings_40, use_container_width=True, hide_index=True)
                else:
                    st.info("No data available for this breakdown.")

            with sub_tabs[2]:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                            box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">📊 Last 20 Matches by Format</h3>
                </div>
                """, unsafe_allow_html=True)

                recent_matches_20 = compute_recent_form(filtered_df, by='matches', num_recent=20)
                if not recent_matches_20.empty:
                    st.dataframe(recent_matches_20, use_container_width=True, hide_index=True)
                else:
                    st.info("No data available for this breakdown.")

        # Opponent Stats Tab  
        with tabs[4]:
            opponents_stats_df = compute_opponent_stats(filtered_df)

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
            opponent_avg_df = filtered_df.groupby('Bowl_Team_y').agg(
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
            location_stats_df = compute_location_stats(filtered_df)

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
            location_avg_df = filtered_df.groupby('Home Team').agg(
                Runs=('Runs', 'sum'),
                Out=('Out', 'sum')
            ).reset_index()
            location_avg_df['Avg'] = (location_avg_df['Runs'] / location_avg_df['Out'].replace(0, np.nan)).fillna(0).round(2)
            fig = go.Figure(go.Bar(x=location_avg_df['Home Team'], y=location_avg_df['Avg'], name='Average'))
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
            innings_stats_df = compute_innings_stats(filtered_df)

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
            innings_grouped = filtered_df.groupby('Innings').agg(
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
            position_stats_df = compute_position_stats(filtered_df)
            
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
            position_grouped = filtered_df.groupby('Position').agg(
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
            homeaway_stats_df = compute_homeaway_stats(filtered_df)

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
            
            yearly_homeaway_stats = filtered_df.groupby(['Year', 'HomeOrAway']).agg(
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

        # Cumulative Stats Tab (only in full mode)
        if upload_mode != 'large':
            with tabs[10]:
                cumulative_stats_df = compute_cumulative_stats(filtered_df)
                
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

        # Block Stats Tab (only in full mode)
        if upload_mode != 'large':
            with tabs[11]:
                block_stats_df = compute_block_stats(filtered_df)

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
 
