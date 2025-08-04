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
    pom_counts = filtered_df[filtered_df['Player_of_the_Match'] == filtered_df['Name']].groupby('Name')['File Name'].nunique().reset_index(name='POM')
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
@st.cache_data
def compute_match_impact_stats(filtered_df, match_df):
    """
    Compute comprehensive batting statistics broken down by match results (Won/Lost/Draw).
    Returns data in long format with separate rows for Career, Won, Lost, Draw.
    """
    if match_df.empty or filtered_df.empty:
        return pd.DataFrame()
    
    try:
        # Check required columns
        required_columns = ['File Name', 'Home_Win', 'Away_Won', 'Home_Lost', 'Away_Lost', 'Home_Drawn', 'Away_Drawn', 'Tie']
        if not all(col in match_df.columns for col in required_columns):
            return pd.DataFrame()
        
        # Merge batting data with match results
        merged_df = filtered_df.merge(
            match_df[required_columns],
            on='File Name', 
            how='left'
        )
        
        if merged_df.empty:
            return pd.DataFrame()
        
        # Function to determine match result
        def determine_result(row):
            bat_team = row.get('Bat_Team_y', '')
            home_team = row.get('Home Team', '')
            away_team = row.get('Away Team', '')
            
            if bat_team == home_team:
                if row.get('Home_Win', 0) == 1:
                    return 'Won'
                elif row.get('Home_Lost', 0) == 1:
                    return 'Lost'
                elif row.get('Home_Drawn', 0) == 1:
                    return 'Draw'
                elif row.get('Tie', 0) == 1:
                    return 'Tie'
            elif bat_team == away_team:
                if row.get('Away_Won', 0) == 1:
                    return 'Won'
                elif row.get('Away_Lost', 0) == 1:
                    return 'Lost'
                elif row.get('Away_Drawn', 0) == 1:
                    return 'Draw'
                elif row.get('Tie', 0) == 1:
                    return 'Tie'
            return 'Unknown'
        
        # Apply result determination
        merged_df['Match_Result'] = merged_df.apply(determine_result, axis=1)
        merged_df = merged_df[merged_df['Match_Result'] != 'Unknown']
        
        if merged_df.empty:
            return pd.DataFrame()
        
        # First calculate career stats for each player
        career_stats = filtered_df.groupby('Name').agg({
            'File Name': 'nunique',     # Matches
            'Batted': 'sum',            # Innings  
            'Out': 'sum',               # Times Out
            'Not Out': 'sum',           # Not Outs
            'Balls': 'sum',             # Balls Faced
            'Runs': ['sum', 'max'],     # Runs and High Score
            '4s': 'sum',                # Fours
            '6s': 'sum',                # Sixes  
            '50s': 'sum',               # Fifties
            '100s': 'sum',              # Hundreds
            '150s': 'sum',              # 150s
            '200s': 'sum',              # 200s
            'Total_Runs': 'sum',        # Team Total Runs
            'Wickets': 'sum',           # Team Wickets
            'Team Balls': 'sum'         # Team Balls
        }).round(2)
        
        # Flatten column names for career stats
        career_stats.columns = [
            'Matches', 'Inns', 'Out', 'Not_Out', 'Balls', 'Runs', 'HS', 
            '4s', '6s', '50s', '100s', '150s', '200s', 'Total_Runs', 'Wickets', 'Team_Balls'
        ]
        career_stats = career_stats.reset_index()
        career_stats['Result'] = 'Career'
        
        # Calculate derived metrics for career
        career_stats['Avg'] = (career_stats['Runs'] / career_stats['Out'].replace(0, np.inf)).replace([np.inf, -np.inf], np.nan).round(2).fillna(0)
        career_stats['SR'] = ((career_stats['Runs'] / career_stats['Balls'].replace(0, np.inf)) * 100).replace([np.inf, -np.inf], np.nan).round(2).fillna(0)
        career_stats['BPO'] = (career_stats['Balls'] / career_stats['Out'].replace(0, np.inf)).replace([np.inf, -np.inf], np.nan).round(2).fillna(0)
        career_stats['50+PI'] = (((career_stats['50s'] + career_stats['100s'] + career_stats['150s'] + career_stats['200s']) / career_stats['Inns'].replace(0, np.inf)) * 100).replace([np.inf, -np.inf], np.nan).round(2).fillna(0)
        career_stats['100PI'] = (((career_stats['100s'] + career_stats['150s'] + career_stats['200s']) / career_stats['Inns'].replace(0, np.inf)) * 100).replace([np.inf, -np.inf], np.nan).round(2).fillna(0)
        career_stats['Not_Out%'] = ((career_stats['Not_Out'] / career_stats['Inns'].replace(0, np.inf)) * 100).replace([np.inf, -np.inf], np.nan).round(2).fillna(0)
        career_stats['Boundary%'] = (((career_stats['4s'] * 4 + career_stats['6s'] * 6) / career_stats['Runs'].replace(0, np.inf)) * 100).replace([np.inf, -np.inf], np.nan).round(2).fillna(0)
        
        # Calculate P+ metrics for career
        career_stats['P+Avg'] = ((career_stats['Runs'] / career_stats['Out'].replace(0, np.inf)) / (career_stats['Total_Runs'] / career_stats['Wickets'].replace(0, np.inf)) * 100).replace([np.inf, -np.inf], np.nan).round(2).fillna(0)
        career_stats['P+SR'] = (((career_stats['Runs'] / career_stats['Balls'].replace(0, np.inf)) / (career_stats['Total_Runs'] / career_stats['Team_Balls'].replace(0, np.inf))) * 100).replace([np.inf, -np.inf], np.nan).round(2).fillna(0)
        
        # Now calculate stats by match result
        result_stats = merged_df.groupby(['Name', 'Match_Result']).agg({
            'File Name': 'nunique',     # Matches
            'Batted': 'sum',            # Innings  
            'Out': 'sum',               # Times Out
            'Not Out': 'sum',           # Not Outs
            'Balls': 'sum',             # Balls Faced
            'Runs': ['sum', 'max'],     # Runs and High Score
            '4s': 'sum',                # Fours
            '6s': 'sum',                # Sixes  
            '50s': 'sum',               # Fifties
            '100s': 'sum',              # Hundreds
            '150s': 'sum',              # 150s
            '200s': 'sum',              # 200s
            'Total_Runs': 'sum',        # Team Total Runs
            'Wickets': 'sum',           # Team Wickets
            'Team Balls': 'sum'         # Team Balls
        }).round(2)
        
        # Flatten column names for result stats
        result_stats.columns = [
            'Matches', 'Inns', 'Out', 'Not_Out', 'Balls', 'Runs', 'HS', 
            '4s', '6s', '50s', '100s', '150s', '200s', 'Total_Runs', 'Wickets', 'Team_Balls'
        ]
        result_stats = result_stats.reset_index()
        result_stats = result_stats.rename(columns={'Match_Result': 'Result'})
        
        # Calculate derived metrics for result stats
        result_stats['Avg'] = (result_stats['Runs'] / result_stats['Out'].replace(0, np.inf)).replace([np.inf, -np.inf], np.nan).round(2).fillna(0)
        result_stats['SR'] = ((result_stats['Runs'] / result_stats['Balls'].replace(0, np.inf)) * 100).replace([np.inf, -np.inf], np.nan).round(2).fillna(0)
        result_stats['BPO'] = (result_stats['Balls'] / result_stats['Out'].replace(0, np.inf)).replace([np.inf, -np.inf], np.nan).round(2).fillna(0)
        result_stats['50+PI'] = (((result_stats['50s'] + result_stats['100s'] + result_stats['150s'] + result_stats['200s']) / result_stats['Inns'].replace(0, np.inf)) * 100).replace([np.inf, -np.inf], np.nan).round(2).fillna(0)
        result_stats['100PI'] = (((result_stats['100s'] + result_stats['150s'] + result_stats['200s']) / result_stats['Inns'].replace(0, np.inf)) * 100).replace([np.inf, -np.inf], np.nan).round(2).fillna(0)
        result_stats['Not_Out%'] = ((result_stats['Not_Out'] / result_stats['Inns'].replace(0, np.inf)) * 100).replace([np.inf, -np.inf], np.nan).round(2).fillna(0)
        result_stats['Boundary%'] = (((result_stats['4s'] * 4 + result_stats['6s'] * 6) / result_stats['Runs'].replace(0, np.inf)) * 100).replace([np.inf, -np.inf], np.nan).round(2).fillna(0)
        
        # Calculate P+ metrics for result stats
        result_stats['P+Avg'] = ((result_stats['Runs'] / result_stats['Out'].replace(0, np.inf)) / (result_stats['Total_Runs'] / result_stats['Wickets'].replace(0, np.inf)) * 100).replace([np.inf, -np.inf], np.nan).round(2).fillna(0)
        result_stats['P+SR'] = (((result_stats['Runs'] / result_stats['Balls'].replace(0, np.inf)) / (result_stats['Total_Runs'] / result_stats['Team_Balls'].replace(0, np.inf))) * 100).replace([np.inf, -np.inf], np.nan).round(2).fillna(0)
        
        # Combine career stats and result stats
        all_stats = pd.concat([career_stats, result_stats], ignore_index=True)
        
        # Ensure we have the right order: Career, Won, Lost, Draw for each player
        result_order = ['Career', 'Won', 'Lost', 'Draw']
        all_stats['Result'] = pd.Categorical(all_stats['Result'], categories=result_order, ordered=True)
        
        # Sort by Name and then by Result to get the desired order
        all_stats = all_stats.sort_values(['Name', 'Result']).reset_index(drop=True)
        
        # Fill missing values with 0 for players who don't have all result types
        all_players = all_stats['Name'].unique()
        complete_data = []
        
        for player in all_players:
            player_data = all_stats[all_stats['Name'] == player]
            
            # Ensure we have all result types for each player
            for result_type in result_order:
                existing_row = player_data[player_data['Result'] == result_type]
                
                if not existing_row.empty:
                    complete_data.append(existing_row.iloc[0].to_dict())
                else:
                    # Create a row with zeros if this result type doesn't exist for the player
                    zero_row = {
                        'Name': player,
                        'Result': result_type,
                        'Matches': 0, 'Inns': 0, 'Out': 0, 'Not_Out': 0, 'Balls': 0,
                        'Runs': 0, 'HS': 0, '4s': 0, '6s': 0, '50s': 0, '100s': 0,
                        '150s': 0, '200s': 0, 'Avg': 0, 'SR': 0, 'BPO': 0,
                        '50+PI': 0, '100PI': 0, 'Not_Out%': 0, 'Boundary%': 0,
                        'P+Avg': 0, 'P+SR': 0, 'Total_Runs': 0, 'Wickets': 0, 'Team_Balls': 0
                    }
                    complete_data.append(zero_row)
        
        # Convert back to DataFrame
        final_df = pd.DataFrame(complete_data)
        
        # Sort by total career runs (descending) and then by result type
        career_runs = final_df[final_df['Result'] == 'Career'].set_index('Name')['Runs'].to_dict()
        final_df['Career_Runs_Sort'] = final_df['Name'].map(career_runs)
        final_df = final_df.sort_values(['Career_Runs_Sort', 'Name', 'Result'], ascending=[False, True, True])
        final_df = final_df.drop('Career_Runs_Sort', axis=1).reset_index(drop=True)
        
        # Select and order columns for display
        display_columns = [
            'Name', 'Result', 'Matches', 'Inns', 'Out', 'Not_Out', 'Runs', 'HS', 
            'Avg', 'SR', 'BPO', '4s', '6s', '50s', '100s', '50+PI', '100PI', 
            'Not_Out%', 'Boundary%'
        ]
        
        final_df = final_df[display_columns]
        
        return final_df
        
    except Exception as e:
        st.error(f"Error in compute_match_impact_stats: {e}")
        return pd.DataFrame()

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
    # Sort the DataFrame for cumulative calculations
    df = filtered_df.sort_values(by=['Name', 'Match_Format', 'Date'])

    # Group by the unique match instance
    match_level_df = df.groupby(['Name', 'Match_Format', 'Date', 'File Name']).agg({
        'Batted': 'sum', 'Out': 'sum', 'Balls': 'sum', 'Runs': 'sum', '100s': 'sum'
    }).reset_index()

    # Calculate cumulative stats per player/format
    match_level_df['Cumulative Innings'] = match_level_df.groupby(['Name', 'Match_Format'])['Batted'].cumsum()
    match_level_df['Cumulative Runs'] = match_level_df.groupby(['Name', 'Match_Format'])['Runs'].cumsum()
    match_level_df['Cumulative Balls'] = match_level_df.groupby(['Name', 'Match_Format'])['Balls'].cumsum()
    match_level_df['Cumulative Outs'] = match_level_df.groupby(['Name', 'Match_Format'])['Out'].cumsum()
    match_level_df['Cumulative 100s'] = match_level_df.groupby(['Name', 'Match_Format'])['100s'].cumsum()
    
    # Calculate running averages and rates
    match_level_df['Cumulative Avg'] = (match_level_df['Cumulative Runs'] / match_level_df['Cumulative Outs'].replace(0, np.nan)).fillna(0).round(2)
    match_level_df['Cumulative SR'] = (match_level_df['Cumulative Runs'] / match_level_df['Cumulative Balls'].replace(0, np.nan) * 100).fillna(0).round(2)
    
    return match_level_df.sort_values(by='Date', ascending=False)

@st.cache_data
def compute_block_stats(filtered_df):
    # Use the cumulative stats as a base
    cumulative_df = compute_cumulative_stats(filtered_df)
    
    # Define blocks of 20 matches
    cumulative_df['Match_Block'] = ((cumulative_df['Cumulative Innings'] - 1) // 20)
    
    # Group by blocks and calculate stats for that block
    block_stats_df = cumulative_df.groupby(['Name', 'Match_Format', 'Match_Block']).agg(
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
    block_stats_df['Match_Range'] = block_stats_df['Match_Block'].apply(lambda x: f"{x*20+1}-{x*20+20}")
    block_stats_df['Date_Range'] = block_stats_df['First_Date'].dt.strftime('%d/%m/%Y') + ' to ' + block_stats_df['Last_Date'].dt.strftime('%d/%m/%Y')

    return block_stats_df.sort_values(by=['Name', 'Match_Format', 'Match_Block'])

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
                    bat_df['comp'] = bat_df['comp'].fillna(bat_df['Competition'])
                else:
                    bat_df['comp'] = bat_df['Competition']
            else:
                # Fallback: use Competition if merge fails
                bat_df['comp'] = bat_df['Competition']
                
            # Convert Date to datetime once for all future operations
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
            # Where Home Team equals Batting Team
            bat_df.loc[bat_df['Home Team'] == bat_df['Bat_Team_y'], 'HomeOrAway'] = 'Home'
            # Where Away Team equals Batting Team
            bat_df.loc[bat_df['Away Team'] == bat_df['Bat_Team_y'], 'HomeOrAway'] = 'Away'
            
            st.session_state['processed_bat_df'] = bat_df
        else:
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
        if name_choice and 'All' not in name_choice:
            mask &= filtered_df['Name'].isin(name_choice)
        if bat_team_choice and 'All' not in bat_team_choice:
            mask &= filtered_df['Bat_Team_y'].isin(bat_team_choice)
        if bowl_team_choice and 'All' not in bowl_team_choice:
            mask &= filtered_df['Bowl_Team_y'].isin(bowl_team_choice)
        if match_format_choice and 'All' not in match_format_choice:
            mask &= filtered_df['Match_Format'].isin(match_format_choice)
        if comp_choice and 'All' not in comp_choice:
            mask &= filtered_df['comp'].isin(comp_choice)
        # Apply range filters
        mask &= filtered_df['Year'].between(year_choice[0], year_choice[1])
        mask &= filtered_df['Position'].between(position_choice[0], position_choice[1])
        filtered_df = filtered_df[mask]
        # Ensure HomeOrAway column exists in the filtered DataFrame
        if 'HomeOrAway' not in filtered_df.columns:
            filtered_df['HomeOrAway'] = 'Neutral'  # Default value
            # Where Home Team equals Batting Team
            filtered_df.loc[filtered_df['Home Team'] == filtered_df['Bat_Team_y'], 'HomeOrAway'] = 'Home'
            # Where Away Team equals Batting Team
            filtered_df.loc[filtered_df['Away Team'] == filtered_df['Bat_Team_y'], 'HomeOrAway'] = 'Away'
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
        
        # Create tabs for different views - Removed "Distributions" and "Percentile" tabs for performance
# Create tabs for different views
        tabs = main_container.tabs([
            "Career", "Format", "Season", "Latest", "Opponent", 
            "Location", "Innings", "Position", "Home/Away", "Match Impact",
            "Cumulative", "Block"
        ])
        
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

        # Match Impact / Clutch Performance Tab
        with tabs[9]:
            # Try to get match impact stats
            match_impact_df = compute_match_impact_stats(filtered_df, match_df)
            
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
                
                # Configure column display with proper formatting
                column_config = {
                    "Name": st.column_config.Column("Name", pinned=True, width=120),
                    "Result": st.column_config.Column("Result", width=80),
                    "Runs": st.column_config.NumberColumn("Runs", format="%d"),
                    "Avg": st.column_config.NumberColumn("Avg", format="%.2f"),
                    "SR": st.column_config.NumberColumn("SR", format="%.2f"),
                    "Matches": st.column_config.NumberColumn("Matches", format="%d"),
                    "Inns": st.column_config.NumberColumn("Inns", format="%d"),
                    "HS": st.column_config.NumberColumn("HS", format="%d"),
                    "BPO": st.column_config.NumberColumn("BPO", format="%.2f")
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
                
                # Prepare data for the comprehensive bar chart
                career_data = match_impact_df[match_impact_df['Result'] == 'Career'][['Name', 'Avg', 'Runs']].rename(columns={'Avg': 'Career_Avg'})
                won_data_chart = match_impact_df[match_impact_df['Result'] == 'Won'][['Name', 'Avg', 'Matches']].rename(columns={'Avg': 'Won_Avg'})
                lost_data_chart = match_impact_df[match_impact_df['Result'] == 'Lost'][['Name', 'Avg', 'Matches']].rename(columns={'Avg': 'Lost_Avg'})
                draw_data_chart = match_impact_df[match_impact_df['Result'] == 'Draw'][['Name', 'Avg', 'Matches']].rename(columns={'Avg': 'Draw_Avg'})
                
                # Merge all data
                complete_data = career_data.merge(won_data_chart, on='Name', how='left')
                complete_data = complete_data.merge(lost_data_chart, on='Name', how='left', suffixes=('', '_lost'))
                complete_data = complete_data.merge(draw_data_chart, on='Name', how='left', suffixes=('', '_draw'))
                
                # Fill NaN values with 0 for players without certain results
                complete_data = complete_data.fillna(0)
                
                # Sort by career runs (descending) and take top players for readability
                complete_data = complete_data.sort_values('Runs', ascending=False)
                
                # Limit to top 20 players for better visualization
                top_players = complete_data.head(20) if len(complete_data) > 20 else complete_data
                
                if not top_players.empty:
                    # Create grouped bar chart
                    bar_fig = go.Figure()
                    
                    # Add bars for each category
                    bar_fig.add_trace(go.Bar(
                        name='Career Average',
                        x=top_players['Name'],
                        y=top_players['Career_Avg'],
                        marker_color='#36d1dc',
                        hovertemplate='<b>%{x}</b><br>Career Avg: %{y:.2f}<extra></extra>'
                    ))
                    
                    bar_fig.add_trace(go.Bar(
                        name='Won Average',
                        x=top_players['Name'],
                        y=top_players['Won_Avg'],
                        marker_color='#28a745',
                        hovertemplate='<b>%{x}</b><br>Won Avg: %{y:.2f}<extra></extra>'
                    ))
                    
                    bar_fig.add_trace(go.Bar(
                        name='Lost Average',
                        x=top_players['Name'],
                        y=top_players['Lost_Avg'],
                        marker_color='#dc3545',
                        hovertemplate='<b>%{x}</b><br>Lost Avg: %{y:.2f}<extra></extra>'
                    ))
                    
                    bar_fig.add_trace(go.Bar(
                        name='Draw Average',
                        x=top_players['Name'],
                        y=top_players['Draw_Avg'],
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
                    clutch_performers = top_players[top_players['Won_Avg'] > top_players['Career_Avg']]
                    pressure_performers = top_players[top_players['Lost_Avg'] > top_players['Career_Avg']]
                    consistent_performers = top_players[
                        (abs(top_players['Won_Avg'] - top_players['Career_Avg']) <= 5) & 
                        (abs(top_players['Lost_Avg'] - top_players['Career_Avg']) <= 5)
                    ]
                    
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
                
                # Create clutch performance visualization using the long format data
                # First, we need to pivot the data to get Won vs Lost averages
                won_data = match_impact_df[match_impact_df['Result'] == 'Won'][['Name', 'Avg', 'Matches']].rename(columns={'Avg': 'Avg_Won', 'Matches': 'Matches_Won'})
                lost_data = match_impact_df[match_impact_df['Result'] == 'Lost'][['Name', 'Avg', 'Matches']].rename(columns={'Avg': 'Avg_Lost', 'Matches': 'Matches_Lost'})
                
                # Merge Won and Lost data for comparison
                clutch_data = won_data.merge(lost_data, on='Name', how='inner')
                
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
                    sufficient_data = clutch_data[
                        (clutch_data.get('Matches_Won', 0) >= 2) & 
                        (clutch_data.get('Matches_Lost', 0) >= 2) &
                        (clutch_data.get('Avg_Won', 0) > 0) & 
                        (clutch_data.get('Avg_Lost', 0) > 0)
                    ]
                    
                    if not sufficient_data.empty:
                        # Create clutch performance scatter plot using the same logic as summary cards
                        clutch_fig = go.Figure()
                        
                        # Use the top_players data directly to ensure consistency with summary cards
                        for _, player_top in top_players.iterrows():
                            player_name = player_top['Name']
                            
                            # Check if this player has sufficient scatter plot data
                            if player_name in sufficient_data['Name'].values:
                                # Get the scatter plot coordinates from sufficient_data
                                player_scatter = sufficient_data[sufficient_data['Name'] == player_name].iloc[0]
                                won_avg = player_scatter.get('Avg_Won', 0)
                                lost_avg = player_scatter.get('Avg_Lost', 0)
                                
                                # Use the top_players data for categorization (same as summary cards)
                                career_avg = player_top['Career_Avg']
                                won_avg_top = player_top['Won_Avg']
                                lost_avg_top = player_top['Lost_Avg']
                                
                                # Apply the exact same logic as summary cards
                                is_big_game = won_avg_top > career_avg
                                is_pressure = lost_avg_top > career_avg
                                is_consistent = (abs(won_avg_top - career_avg) <= 5) and (abs(lost_avg_top - career_avg) <= 5)
                                
                                # Priority order: Consistent > Big Game > Pressure > Other
                                # This ensures consistent players are shown as cyan even if they're also big game or pressure
                                if is_consistent:
                                    color = '#36d1dc'  # Cyan for Consistent Players
                                    category = 'Consistent Player'
                                elif is_big_game:
                                    color = '#28a745'  # Green for Big Game Players
                                    category = 'Big Game Player'
                                elif is_pressure:
                                    color = '#dc3545'  # Red for Pressure Players
                                    category = 'Pressure Player'
                                else:
                                    color = '#6c757d'  # Gray for others
                                    category = 'Other'
                                
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
                                        f"Won vs Career Diff: {won_avg_top - career_avg:+.2f}<br>"
                                        f"Lost vs Career Diff: {lost_avg_top - career_avg:+.2f}<br>"
                                        f"Won Matches: {player_scatter.get('Matches_Won', 0)}<br>"
                                        f"Lost Matches: {player_scatter.get('Matches_Lost', 0)}<br>"
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
                            ),
                            annotations=[
                                dict(
                                    x=0.02, y=0.98,
                                    xref='paper', yref='paper',
                                    text="<b>Color Legend:</b><br><span style='color: #28a745;'>■</span> Big Game Players<br><span style='color: #dc3545;'>■</span> Pressure Players<br><span style='color: #36d1dc;'>■</span> Consistent Players",
                                    showarrow=False,
                                    font=dict(size=10),
                                    bgcolor='rgba(255,255,255,0.9)',
                                    bordercolor='gray',
                                    borderwidth=1
                                )
                            ]
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
            else:
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

        # Latest Innings Tab
        with tabs[3]:
            # Create latest innings dataframe
            fresh_latest_df = filtered_df.copy()
            
            # Process the latest innings data
            latest_innings_raw = fresh_latest_df.groupby(['Name', 'Match_Format', 'Date', 'Innings']).agg({
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
            final_latest_df = latest_innings_raw[['Name', 'Match_Format', 'Date', 'Innings', 'Bat Team', 'Bowl Team', 'How Out', 'Runs', 'Balls', '4s', '6s']]
            
            # Calculate stats
            final_latest_df['Out'] = final_latest_df['How Out'].apply(lambda x: 1 if x not in ['not out', 'did not bat', ''] else 0)
            
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
            styled_latest_df = final_latest_df.style.applymap(style_runs_column, subset=['Runs'])
            st.dataframe(styled_latest_df, height=735, use_container_width=True, hide_index=True)

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

        # Cumulative Stats Tab
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

        # Block Stats Tab
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
 
