import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

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

    match_wickets = df.groupby(['Name', 'File Name'])['Bowler_Wkts'].sum().reset_index()
    ten_wickets = match_wickets[match_wickets['Bowler_Wkts'] >= 10].groupby('Name').size().reset_index(name='10W')

    bowlcareer_df = df.groupby('Name').agg({
        'File Name': 'nunique',
        'Bowler_Balls': 'sum',
        'Maidens': 'sum',
        'Bowler_Runs': 'sum',
        'Bowler_Wkts': 'sum'
    }).reset_index()

    bowlcareer_df.columns = ['Name', 'Matches', 'Balls', 'M/D', 'Runs', 'Wickets']

    # --- Safe Calculations ---
    overs_series = (bowlcareer_df['Balls'] // 6) + (bowlcareer_df['Balls'] % 6) / 10
    bowlcareer_df['Overs'] = overs_series.round(1)
    
    wickets_safe = bowlcareer_df['Wickets'].replace(0, np.nan)
    matches_safe = bowlcareer_df['Matches'].replace(0, np.nan)
    overs_safe = bowlcareer_df['Overs'].replace(0, np.nan)
    
    bowlcareer_df['Strike Rate'] = (bowlcareer_df['Balls'] / wickets_safe).round(2).fillna(0)
    bowlcareer_df['Economy Rate'] = (bowlcareer_df['Runs'] / overs_safe).round(2).fillna(0)
    bowlcareer_df['Avg'] = (bowlcareer_df['Runs'] / wickets_safe).round(2).fillna(0)
    bowlcareer_df['WPM'] = (bowlcareer_df['Wickets'] / matches_safe).round(2).fillna(0)
    
    five_wickets = df[df['Bowler_Wkts'] >= 5].groupby('Name').size().reset_index(name='5W')
    bowlcareer_df = bowlcareer_df.merge(five_wickets, on='Name', how='left').fillna({'5W': 0})
    bowlcareer_df = bowlcareer_df.merge(ten_wickets, on='Name', how='left').fillna({'10W': 0})
    
    pom_counts = df[df['Player_of_the_Match'] == df['Name']].groupby('Name')['File Name'].nunique().reset_index(name='POM')
    bowlcareer_df = bowlcareer_df.merge(pom_counts, on='Name', how='left').fillna({'POM': 0})

    for col in ['5W', '10W', 'POM']:
        bowlcareer_df[col] = bowlcareer_df[col].astype(int)

    final_df = bowlcareer_df[[
        'Name', 'Matches', 'Balls', 'Overs', 'M/D', 'Runs', 'Wickets', 'Avg',
        'Strike Rate', 'Economy Rate', '5W', '10W', 'WPM', 'POM'
    ]].sort_values('Wickets', ascending=False)
    
    return final_df

@st.cache_data
def compute_bowl_format_stats(df):
    if df.empty:
        return pd.DataFrame()
    # Group by Name and Match_Format
    format_df = df.groupby(['Name', 'Match_Format']).agg(
        Matches=('File Name', 'nunique'),
        Bowler_Balls=('Bowler_Balls', 'sum'),
        Bowler_Runs=('Bowler_Runs', 'sum'),
        Bowler_Wkts=('Bowler_Wkts', 'sum'),
        Overs=('Overs', 'sum'),
        Maidens=('Maidens', 'sum') if 'Maidens' in df.columns else ('Bowler_Balls', 'sum'),
        FiveW=('FiveW', 'sum') if 'FiveW' in df.columns else ('Bowler_Balls', 'sum'),
        TenW=('TenW', 'sum') if 'TenW' in df.columns else ('Bowler_Balls', 'sum'),
        Avg_Runs_Conceded=('Total_Runs', 'sum') if 'Total_Runs' in df.columns else ('Bowler_Runs', 'sum')
    ).reset_index()
    wickets_safe = format_df['Bowler_Wkts'].replace(0, np.nan)
    balls_safe = format_df['Bowler_Balls'].replace(0, np.nan)
    runs_safe = format_df['Bowler_Runs'].replace(0, np.nan)
    overs_safe = format_df['Overs'].replace(0, np.nan)
    format_df['Strike Rate'] = (format_df['Bowler_Balls'] / wickets_safe).round(2).fillna(0)
    format_df['Economy Rate'] = (format_df['Bowler_Runs'] / (format_df['Bowler_Balls'].replace(0, np.nan) / 6)).round(2).fillna(0)
    format_df['Average'] = (format_df['Bowler_Runs'] / wickets_safe).round(2).fillna(0)
    format_df['WPM'] = (format_df['Bowler_Wkts'] / format_df['Matches'].replace(0, np.nan)).round(2).fillna(0)
    return format_df

@st.cache_data
def compute_bowl_season_stats(df):
    if df.empty:
        return pd.DataFrame()
    # Group by Name and Year
    season_df = df.groupby(['Name', 'Year']).agg(
        Matches=('File Name', 'nunique'),
        Bowler_Balls=('Bowler_Balls', 'sum'),
        Bowler_Runs=('Bowler_Runs', 'sum'),
        Bowler_Wkts=('Bowler_Wkts', 'sum'),
        Overs=('Overs', 'sum'),
        Maidens=('Maidens', 'sum') if 'Maidens' in df.columns else ('Bowler_Balls', 'sum'),
        FiveW=('FiveW', 'sum') if 'FiveW' in df.columns else ('Bowler_Balls', 'sum'),
        TenW=('TenW', 'sum') if 'TenW' in df.columns else ('Bowler_Balls', 'sum'),
        Avg_Runs_Conceded=('Total_Runs', 'sum') if 'Total_Runs' in df.columns else ('Bowler_Runs', 'sum')
    ).reset_index()
    wickets_safe = season_df['Bowler_Wkts'].replace(0, np.nan)
    balls_safe = season_df['Bowler_Balls'].replace(0, np.nan)
    runs_safe = season_df['Bowler_Runs'].replace(0, np.nan)
    overs_safe = season_df['Overs'].replace(0, np.nan)
    season_df['Strike Rate'] = (season_df['Bowler_Balls'] / wickets_safe).round(2).fillna(0)
    season_df['Economy Rate'] = (season_df['Bowler_Runs'] / (season_df['Bowler_Balls'].replace(0, np.nan) / 6)).round(2).fillna(0)
    season_df['Average'] = (season_df['Bowler_Runs'] / wickets_safe).round(2).fillna(0)
    season_df['WPM'] = (season_df['Bowler_Wkts'] / season_df['Matches'].replace(0, np.nan)).round(2).fillna(0)
    return season_df

@st.cache_data
def compute_bowl_opponent_stats(df):
    if df.empty:
        return pd.DataFrame()
    # Calculate statistics dataframe for opponents
    opponent_summary = df.groupby(['Name', 'Bat_Team']).agg({
        'File Name': 'nunique',
        'Bowler_Balls': 'sum',
        'Maidens': 'sum',
        'Bowler_Runs': 'sum',
        'Bowler_Wkts': 'sum'
    }).reset_index()
    opponent_summary.columns = ['Name', 'Opposition', 'Matches', 'Balls', 'M/D', 'Runs', 'Wickets']
    overs_series = (opponent_summary['Balls'] // 6) + (opponent_summary['Balls'] % 6) / 10
    opponent_summary['Overs'] = overs_series.round(1)
    wickets_safe = opponent_summary['Wickets'].replace(0, np.nan)
    matches_safe = opponent_summary['Matches'].replace(0, np.nan)
    overs_safe = opponent_summary['Overs'].replace(0, np.nan)
    opponent_summary['Strike Rate'] = (opponent_summary['Balls'] / wickets_safe).round(2).fillna(0)
    opponent_summary['Economy Rate'] = (opponent_summary['Runs'] / overs_safe).round(2).fillna(0)
    opponent_summary['Average'] = (opponent_summary['Runs'] / wickets_safe).round(2).fillna(0)
    opponent_summary['WPM'] = (opponent_summary['Wickets'] / matches_safe).round(2).fillna(0)
    five_wickets = df[df['Bowler_Wkts'] >= 5].groupby(['Name', 'Bat_Team']).size().reset_index(name='5W')
    opponent_summary = opponent_summary.merge(five_wickets, left_on=['Name', 'Opposition'], right_on=['Name', 'Bat_Team'], how='left').fillna({'5W': 0})
    match_wickets = df.groupby(['Name', 'Bat_Team', 'File Name'])['Bowler_Wkts'].sum().reset_index()
    ten_wickets = match_wickets[match_wickets['Bowler_Wkts'] >= 10].groupby(['Name', 'Bat_Team']).size().reset_index(name='10W')
    opponent_summary = opponent_summary.merge(ten_wickets, left_on=['Name', 'Opposition'], right_on=['Name', 'Bat_Team'], how='left').fillna({'10W': 0})
    opponent_summary['5W'] = opponent_summary['5W'].astype(int)
    opponent_summary['10W'] = opponent_summary['10W'].astype(int)
    final_df = opponent_summary[[
        'Name', 'Opposition', 'Matches', 'Overs', 'M/D', 'Runs', 'Wickets', 'Average',
        'Strike Rate', 'Economy Rate', '5W', '10W', 'WPM'
    ]].sort_values(by='Wickets', ascending=False)
    return final_df

@st.cache_data
def compute_bowl_location_stats(df):
    if df.empty:
        return pd.DataFrame()
    location_summary = df.groupby(['Name', 'Home_Team']).agg({
        'File Name': 'nunique',
        'Bowler_Balls': 'sum',
        'Maidens': 'sum',
        'Bowler_Runs': 'sum',
        'Bowler_Wkts': 'sum'
    }).reset_index()
    location_summary.columns = ['Name', 'Location', 'Matches', 'Balls', 'M/D', 'Runs', 'Wickets']
    overs_series = (location_summary['Balls'] // 6) + (location_summary['Balls'] % 6) / 10
    location_summary['Overs'] = overs_series.round(1)
    wickets_safe = location_summary['Wickets'].replace(0, np.nan)
    matches_safe = location_summary['Matches'].replace(0, np.nan)
    overs_safe = location_summary['Overs'].replace(0, np.nan)
    location_summary['Strike Rate'] = (location_summary['Balls'] / wickets_safe).round(2).fillna(0)
    location_summary['Economy Rate'] = (location_summary['Runs'] / overs_safe).round(2).fillna(0)
    location_summary['Average'] = (location_summary['Runs'] / wickets_safe).round(2).fillna(0)
    location_summary['WPM'] = (location_summary['Wickets'] / matches_safe).round(2).fillna(0)
    five_wickets = df[df['Bowler_Wkts'] >= 5].groupby(['Name', 'Home_Team']).size().reset_index(name='5W')
    location_summary = location_summary.merge(five_wickets, left_on=['Name', 'Location'], right_on=['Name', 'Home_Team'], how='left').fillna({'5W': 0})
    match_wickets = df.groupby(['Name', 'Home_Team', 'File Name'])['Bowler_Wkts'].sum().reset_index()
    ten_wickets = match_wickets[match_wickets['Bowler_Wkts'] >= 10].groupby(['Name', 'Home_Team']).size().reset_index(name='10W')
    location_summary = location_summary.merge(ten_wickets, left_on=['Name', 'Location'], right_on=['Name', 'Home_Team'], how='left').fillna({'10W': 0})
    location_summary['5W'] = location_summary['5W'].astype(int)
    location_summary['10W'] = location_summary['10W'].astype(int)
    final_df = location_summary[[
        'Name', 'Location', 'Matches', 'Overs', 'M/D', 'Runs', 'Wickets', 'Average',
        'Strike Rate', 'Economy Rate', '5W', '10W', 'WPM'
    ]].sort_values(by='Wickets', ascending=False)
    return final_df

@st.cache_data
def compute_bowl_innings_stats(df):
    if df.empty:
        return pd.DataFrame()
    innings_summary = df.groupby(['Name', 'Innings']).agg({
        'File Name': 'nunique',
        'Bowler_Balls': 'sum',
        'Maidens': 'sum',
        'Bowler_Runs': 'sum',
        'Bowler_Wkts': 'sum'
    }).reset_index()
    innings_summary.columns = ['Name', 'Innings', 'Matches', 'Balls', 'M/D', 'Runs', 'Wickets']
    overs_series = (innings_summary['Balls'] // 6) + (innings_summary['Balls'] % 6) / 10
    innings_summary['Overs'] = overs_series.round(1)
    wickets_safe = innings_summary['Wickets'].replace(0, np.nan)
    matches_safe = innings_summary['Matches'].replace(0, np.nan)
    overs_safe = innings_summary['Overs'].replace(0, np.nan)
    innings_summary['Strike Rate'] = (innings_summary['Balls'] / wickets_safe).round(2).fillna(0)
    innings_summary['Economy Rate'] = (innings_summary['Runs'] / overs_safe).round(2).fillna(0)
    innings_summary['Average'] = (innings_summary['Runs'] / wickets_safe).round(2).fillna(0)
    innings_summary['WPM'] = (innings_summary['Wickets'] / matches_safe).round(2).fillna(0)
    five_wickets = df[df['Bowler_Wkts'] >= 5].groupby(['Name', 'Innings']).size().reset_index(name='5W')
    innings_summary = innings_summary.merge(five_wickets, on=['Name', 'Innings'], how='left').fillna({'5W': 0})
    match_wickets = df.groupby(['Name', 'Innings', 'File Name'])['Bowler_Wkts'].sum().reset_index()
    ten_wickets = match_wickets[match_wickets['Bowler_Wkts'] >= 10].groupby(['Name', 'Innings']).size().reset_index(name='10W')
    innings_summary = innings_summary.merge(ten_wickets, on=['Name', 'Innings'], how='left').fillna({'10W': 0})
    innings_summary['5W'] = innings_summary['5W'].astype(int)
    innings_summary['10W'] = innings_summary['10W'].astype(int)
    final_df = innings_summary[[
        'Name', 'Innings', 'Matches', 'Overs', 'M/D', 'Runs', 'Wickets', 'Average',
        'Strike Rate', 'Economy Rate', '5W', '10W', 'WPM'
    ]].sort_values(by='Wickets', ascending=False)
    return final_df

@st.cache_data
def compute_bowl_position_stats(df):
    if df.empty:
        return pd.DataFrame()
    position_summary = df.groupby(['Name', 'Position']).agg({
        'File Name': 'nunique',
        'Bowler_Balls': 'sum',
        'Maidens': 'sum',
        'Bowler_Runs': 'sum',
        'Bowler_Wkts': 'sum'
    }).reset_index()
    position_summary.columns = ['Name', 'Position', 'Matches', 'Balls', 'M/D', 'Runs', 'Wickets']
    overs_series = (position_summary['Balls'] // 6) + (position_summary['Balls'] % 6) / 10
    position_summary['Overs'] = overs_series.round(1)
    wickets_safe = position_summary['Wickets'].replace(0, np.nan)
    matches_safe = position_summary['Matches'].replace(0, np.nan)
    overs_safe = position_summary['Overs'].replace(0, np.nan)
    position_summary['Strike Rate'] = (position_summary['Balls'] / wickets_safe).round(2).fillna(0)
    position_summary['Economy Rate'] = (position_summary['Runs'] / overs_safe).round(2).fillna(0)
    position_summary['Average'] = (position_summary['Runs'] / wickets_safe).round(2).fillna(0)
    position_summary['WPM'] = (position_summary['Wickets'] / matches_safe).round(2).fillna(0)
    five_wickets = df[df['Bowler_Wkts'] >= 5].groupby(['Name', 'Position']).size().reset_index(name='5W')
    position_summary = position_summary.merge(five_wickets, on=['Name', 'Position'], how='left').fillna({'5W': 0})
    match_wickets = df.groupby(['Name', 'Position', 'File Name'])['Bowler_Wkts'].sum().reset_index()
    ten_wickets = match_wickets[match_wickets['Bowler_Wkts'] >= 10].groupby(['Name', 'Position']).size().reset_index(name='10W')
    position_summary = position_summary.merge(ten_wickets, on=['Name', 'Position'], how='left').fillna({'10W': 0})
    position_summary['5W'] = position_summary['5W'].astype(int)
    position_summary['10W'] = position_summary['10W'].astype(int)
    final_df = position_summary[[
        'Name', 'Position', 'Matches', 'Overs', 'M/D', 'Runs', 'Wickets', 'Average',
        'Strike Rate', 'Economy Rate', '5W', '10W', 'WPM'
    ]].sort_values(by='Wickets', ascending=False)
    return final_df

@st.cache_data
def compute_bowl_latest_innings(df):
    if df.empty:
        return pd.DataFrame()
    latest_innings = df.groupby(['Name', 'Match_Format', 'Date', 'Innings']).agg({
        'Bowl_Team': 'first',
        'Bat_Team': 'first',
        'Overs': 'sum',
        'Maidens': 'sum',
        'Bowler_Runs': 'sum',
        'Bowler_Wkts': 'sum',
        'File Name': 'first'
    }).reset_index()
    latest_innings = latest_innings.rename(columns={
        'Match_Format': 'Format',
        'Bowl_Team': 'Team',
        'Bat_Team': 'Opponent',
        'Overs': 'Overs',
        'Maidens': 'Maidens',
        'Bowler_Runs': 'Runs',
        'Bowler_Wkts': 'Wickets',
        'File Name': 'File Name'
    })
    latest_innings['Date'] = pd.to_datetime(latest_innings['Date'], errors='coerce')
    latest_innings = latest_innings.sort_values(by='Date', ascending=False).head(20)
    latest_innings['Date'] = latest_innings['Date'].dt.strftime('%d/%m/%Y')
    return latest_innings[['Name', 'Format', 'Date', 'Innings', 'Team', 'Opponent', 'Overs', 'Maidens', 'Runs', 'Wickets', 'File Name']]

@st.cache_data
def compute_bowl_cumulative_stats(df):
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values(by=['Name', 'Match_Format', 'Date'])
    match_level_df = df.groupby(['Name', 'Match_Format', 'Date', 'File Name']).agg({
        'Bowler_Balls': 'sum', 'Bowler_Runs': 'sum', 'Bowler_Wkts': 'sum'
    }).reset_index()
    # Cumulative Matches: true count of matches per player/format
    match_level_df['Cumulative Matches'] = match_level_df.groupby(['Name', 'Match_Format']).cumcount() + 1
    match_level_df['Cumulative Runs'] = match_level_df.groupby(['Name', 'Match_Format'])['Bowler_Runs'].cumsum()
    match_level_df['Cumulative Balls'] = match_level_df.groupby(['Name', 'Match_Format'])['Bowler_Balls'].cumsum()
    match_level_df['Cumulative Wickets'] = match_level_df.groupby(['Name', 'Match_Format'])['Bowler_Wkts'].cumsum()
    match_level_df['Cumulative Avg'] = (match_level_df['Cumulative Runs'] / match_level_df['Cumulative Wickets'].replace(0, np.nan)).fillna(0).round(2)
    match_level_df['Cumulative SR'] = (match_level_df['Cumulative Balls'] / match_level_df['Cumulative Wickets'].replace(0, np.nan)).fillna(0).round(2)
    match_level_df['Cumulative Econ'] = (match_level_df['Cumulative Runs'] / (match_level_df['Cumulative Balls'].replace(0, np.nan) / 6)).fillna(0).round(2)
    return match_level_df.sort_values(by='Date', ascending=False)

@st.cache_data
def compute_bowl_block_stats(df):
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values(by=['Name', 'Match_Format', 'Date'])
    # Create innings number and innings range
    df['Innings_Number'] = df.groupby(['Name', 'Match_Format']).cumcount() + 1
    df['Innings_Range'] = (((df['Innings_Number'] - 1) // 20) * 20).astype(str) + '-' + (((df['Innings_Number'] - 1) // 20) * 20 + 19).astype(str)
    df['Range_Start'] = ((df['Innings_Number'] - 1) // 20) * 20
    # Group by blocks and calculate statistics
    block_stats_df = df.groupby(['Name', 'Match_Format', 'Innings_Range', 'Range_Start']).agg({
        'Innings_Number': 'count',
        'Bowler_Balls': 'sum',
        'Bowler_Runs': 'sum',
        'Bowler_Wkts': 'sum',
        'Date': ['first', 'last']
    }).reset_index()
    # Flatten the column names
    block_stats_df.columns = ['Name', 'Match_Format', 'Innings_Range', 'Range_Start',
                            'Innings', 'Balls', 'Runs', 'Wickets',
                            'First_Date', 'Last_Date']
    # Calculate statistics for each block
    block_stats_df['Overs'] = (block_stats_df['Balls'] // 6) + (block_stats_df['Balls'] % 6) / 10
    block_stats_df['Average'] = (block_stats_df['Runs'] / block_stats_df['Wickets']).round(2)
    block_stats_df['Strike_Rate'] = (block_stats_df['Balls'] / block_stats_df['Wickets']).round(2)
    block_stats_df['Economy'] = (block_stats_df['Runs'] / block_stats_df['Overs']).round(2)
    # Format dates properly before creating date range
    block_stats_df['First_Date'] = pd.to_datetime(block_stats_df['First_Date']).dt.strftime('%d/%m/%Y')
    block_stats_df['Last_Date'] = pd.to_datetime(block_stats_df['Last_Date']).dt.strftime('%d/%m/%Y')
    # Create date range column
    block_stats_df['Date_Range'] = block_stats_df['First_Date'] + ' to ' + block_stats_df['Last_Date']
    # Sort the DataFrame
    block_stats_df = block_stats_df.sort_values(['Name', 'Match_Format', 'Range_Start'])
    # Select and order final columns
    final_columns = [
        'Name', 'Match_Format', 'Innings_Range', 'Date_Range',
        'Innings', 'Overs', 'Runs', 'Wickets',
        'Average', 'Strike_Rate', 'Economy'
    ]
    block_stats_df = block_stats_df[final_columns]
    # Handle any infinities and NaN values
    block_stats_df = block_stats_df.replace([np.inf, -np.inf], np.nan)
    return block_stats_df

@st.cache_data
def compute_bowl_homeaway_stats(df):
    if df.empty:
        return pd.DataFrame()
    homeaway_stats = df.groupby(['Name', 'HomeOrAway']).agg({
        'File Name': 'nunique',
        'Bowler_Balls': 'sum',
        'Maidens': 'sum',
        'Bowler_Runs': 'sum',
        'Bowler_Wkts': 'sum'
    }).reset_index()
    homeaway_stats.columns = ['Name', 'HomeOrAway', 'Matches', 'Balls', 'M/D', 'Runs', 'Wickets']
    overs_series = (homeaway_stats['Balls'] // 6) + (homeaway_stats['Balls'] % 6) / 10
    homeaway_stats['Overs'] = overs_series.round(1)
    wickets_safe = homeaway_stats['Wickets'].replace(0, np.nan)
    matches_safe = homeaway_stats['Matches'].replace(0, np.nan)
    overs_safe = homeaway_stats['Overs'].replace(0, np.nan)
    homeaway_stats['Strike Rate'] = (homeaway_stats['Balls'] / wickets_safe).round(2).fillna(0)
    homeaway_stats['Economy Rate'] = (homeaway_stats['Runs'] / overs_safe).round(2).fillna(0)
    homeaway_stats['Average'] = (homeaway_stats['Runs'] / wickets_safe).round(2).fillna(0)
    homeaway_stats['WPM'] = (homeaway_stats['Wickets'] / matches_safe).round(2).fillna(0)
    return homeaway_stats

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

        # Calculate range filter statistics
        career_stats = bowl_df.groupby('Name').agg({
            'File Name': 'nunique',
            'Bowler_Wkts': 'sum',
            'Bowler_Runs': 'sum',
            'Bowler_Balls': 'sum'
        }).reset_index()

        career_stats['Avg'] = career_stats['Bowler_Runs'] / career_stats['Bowler_Wkts'].replace(0, np.inf)
        career_stats['SR'] = career_stats['Bowler_Balls'] / career_stats['Bowler_Wkts'].replace(0, np.inf)
        career_stats['Avg'] = career_stats['Avg'].replace([np.inf, -np.inf], np.nan)
        career_stats['SR'] = career_stats['SR'].replace([np.inf, -np.inf], np.nan)

        # Calculate max values
        max_wickets = int(career_stats['Bowler_Wkts'].max())
        max_matches = int(career_stats['File Name'].max())
        max_avg = float(career_stats['Avg'].max())
        max_sr = float(career_stats['SR'].max())

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

###-------------------------------------APPLY FILTERS-------------------------------------###
        # Create filtered dataframe
        filtered_df = bowl_df.copy()

        # Apply basic filters
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

        # Apply year filter (only if Year column exists)
        if 'Year' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Year'].between(year_choice[0], year_choice[1])]
        else:
            st.warning("Year column not available. Year filter will be skipped.")

        # Apply range filters
        filtered_df = filtered_df.groupby('Name').filter(lambda x: 
            wickets_range[0] <= x['Bowler_Wkts'].sum() <= wickets_range[1] and
            matches_range[0] <= x['File Name'].nunique() <= matches_range[1] and
            (avg_range[0] <= (x['Bowler_Runs'].sum() / x['Bowler_Wkts'].sum()) <= avg_range[1] if x['Bowler_Wkts'].sum() > 0 else True) and
            (sr_range[0] <= (x['Bowler_Balls'].sum() / x['Bowler_Wkts'].sum()) <= sr_range[1] if x['Bowler_Wkts'].sum() > 0 else True)
        )

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
            season_df = compute_bowl_season_stats(filtered_df)
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
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
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
                        box-shadow: 0 8px 32px rgba(168, 202, 186, 0.3);
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
                        box-shadow: 0 8px 32px rgba(131, 96, 195, 0.3);
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
            # Handle 'All' selection
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
