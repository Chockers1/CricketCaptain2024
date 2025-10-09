import streamlit as st
import pandas as pd
import numpy as np
import polars as pl
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

# --- Batch Redis Pipeline Helpers ---
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

# Check if bat_df is available in session state
if 'bat_df' in st.session_state:
    bat_df = st.session_state['bat_df']
else:
    bat_df = pd.DataFrame()  # Default to empty DataFrame if not available

# Check if match_df is available in session state
if 'match_df' in st.session_state:
    match_df = st.session_state['match_df']
else:
    match_df = pd.DataFrame()  # Default to empty DataFrame if not available

<<<<<<< HEAD
# Check if player_names is available in session state
if 'player_names' in st.session_state:
    player_names = st.session_state['player_names']
else:
    player_names = []  # Default to empty list if not available

# Check if team_names is available in session state
if 'team_names' in st.session_state:
    team_names = st.session_state['team_names']
else:
    team_names = []  # Default to empty list if not available

# Check if match_formats is available in session state
if 'match_formats' in st.session_state:
    match_formats = st.session_state['match_formats']
else:
    match_formats = []  # Default to empty list if not available

# Check if competitions is available in session state
if 'competitions' in st.session_state:
    competitions = st.session_state['competitions']
else:
    competitions = []  # Default to empty list if not available

# Initialize Redis connection
REDIS_AVAILABLE = False
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()  # Test the connection
    REDIS_AVAILABLE = True
    print("Redis server connected - caching enabled")
except:
    print("Redis server not available - caching disabled")
    redis_client = None

=======
>>>>>>> 7c723bb689ae91ae1eb065d2c05a038af9ae7e4f
# Increase cache expiry to reduce recalculations for unchanged data
CACHE_EXPIRY = timedelta(hours=72)  # Increased from 24 to 72 hours

# Add Streamlit cache decorators for better performance
@st.cache_data(ttl=3600, show_spinner=False)
def optimize_dataframe(df):
    """Optimize DataFrame for faster operations"""
    # Only copy if mutation is required (here, we do mutate)
    optimized_df = df if df.flags.writeable else df.copy()
    # Convert categorical columns to category dtype for faster groupby operations
    categorical_cols = [
        'Name', 'Match_Format', 'Bat_Team_y', 'Bowl_Team_y', 'Home Team', 'Away Team',
        'comp', 'Player_of_the_Match', 'Competition'
    ]
    for col in categorical_cols:
        if col in optimized_df.columns and optimized_df[col].dtype != 'category':
            optimized_df[col] = optimized_df[col].astype('category')
    # Convert numeric columns to appropriate dtypes
    numeric_cols = [
        'Runs', 'Balls', 'Out', 'Not Out', '4s', '6s', 'Year', 'Position', 'Innings',
        'Batted', 'Total_Runs', 'Team Balls', 'Wickets'
    ]
    for col in numeric_cols:
        if col in optimized_df.columns and not pd.api.types.is_integer_dtype(optimized_df[col]):
            optimized_df[col] = pd.to_numeric(optimized_df[col], errors='coerce', downcast='integer')
    return optimized_df

@st.cache_data(ttl=3600, show_spinner=False)
def fast_groupby_agg(df, group_cols, agg_dict):
    """Optimized groupby aggregation with error handling"""
    try:
        # Remove missing columns from agg_dict
        valid_agg = {k: v for k, v in agg_dict.items() if k in df.columns}
        
        if not valid_agg:
            return pd.DataFrame()
        
        result = df.groupby(group_cols, observed=True).agg(valid_agg).reset_index()
        return result
    except Exception as e:
        print(f"Groupby error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def vectorized_calculations(df):
    """Perform all calculations using vectorized operations"""
    if df.empty:
        return df
    # All calculations are vectorized, no .apply or row-wise loops
    result_df = df  # No .copy() needed, as we do not mutate
    result_df['Avg'] = np.where(result_df['Out'] > 0, result_df['Runs'] / result_df['Out'], 0).round(2)
    result_df['SR'] = np.where(result_df['Balls'] > 0, (result_df['Runs'] / result_df['Balls']) * 100, 0).round(2)
    result_df['BPO'] = np.where(result_df['Out'] > 0, result_df['Balls'] / result_df['Out'], 0).round(2)
    if 'Team_Runs' in result_df.columns and 'Wickets' in result_df.columns:
        result_df['Team_Avg'] = np.where(result_df['Wickets'] > 0, result_df['Team_Runs'] / result_df['Wickets'], 0).round(2)
    if 'Team_Runs' in result_df.columns and 'Team Balls' in result_df.columns:
        result_df['Team_SR'] = np.where(result_df['Team Balls'] > 0, (result_df['Team_Runs'] / result_df['Team Balls']) * 100, 0).round(2)
    if 'Team_Avg' in result_df.columns:
        result_df['P+ Avg'] = np.where(result_df['Team_Avg'] > 0, (result_df['Avg'] / result_df['Team_Avg']) * 100, 0).round(2)
    if 'Team_SR' in result_df.columns:
        result_df['P+ SR'] = np.where(result_df['Team_SR'] > 0, (result_df['SR'] / result_df['Team_SR']) * 100, 0).round(2)
    return result_df

def handle_redis_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            return None
    return wrapper

<<<<<<< HEAD
@handle_redis_errors

def cache_dataframe(key, df, expiry=CACHE_EXPIRY):
    """Cache a DataFrame using Redis, using Polars IPC/Parquet for Polars DataFrames, pickle for pandas."""
    if hasattr(df, 'to_pandas') and not isinstance(df, pd.DataFrame):
        # Polars DataFrame: use IPC serialization
        df_bytes = df.to_ipc()
        redis_client.set(key + ':pl', df_bytes, ex=int(expiry.total_seconds()))
        # Store a marker for Polars
        redis_client.set(key + ':type', b'pl', ex=int(expiry.total_seconds()))
    elif isinstance(df, pd.DataFrame):
        # Pandas fallback
        df_bytes = pickle.dumps(df)
        redis_client.set(key + ':pd', df_bytes, ex=int(expiry.total_seconds()))
        redis_client.set(key + ':type', b'pd', ex=int(expiry.total_seconds()))
    else:
        # Fallback for other types
        df_bytes = pickle.dumps(df)
        redis_client.set(key, df_bytes, ex=int(expiry.total_seconds()))

@handle_redis_errors

def get_cached_dataframe(key):
    """Retrieve a cached DataFrame from Redis, using Polars IPC/Parquet for Polars DataFrames, pickle for pandas."""
    dtype = redis_client.get(key + ':type')
    if dtype == b'pl':
        df_bytes = redis_client.get(key + ':pl')
        if df_bytes:
            return pl.read_ipc(df_bytes)
    elif dtype == b'pd':
        df_bytes = redis_client.get(key + ':pd')
        if df_bytes:
            return pickle.loads(df_bytes)
    else:
        # Fallback for legacy keys
        df_bytes = redis_client.get(key)
        if df_bytes:
            return pickle.loads(df_bytes)
    return None

=======
>>>>>>> 7c723bb689ae91ae1eb065d2c05a038af9ae7e4f
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
    """
    if selected_filters is None:
        return ['All'] + sorted(df[column].unique().tolist())

    filtered_df = df  # No .copy() needed, as we do not mutate

    # Apply each active filter
    for filter_col, filter_val in selected_filters.items():
        if filter_val and 'All' not in filter_val and filter_col != column:
            filtered_df = filtered_df[filtered_df[filter_col].isin(filter_val)]

    return ['All'] + sorted(filtered_df[column].unique().tolist())

@st.cache_data(ttl=3600, show_spinner=False)
def preprocess_batting_data(bat_df, match_df):
    """Preprocess batting data using Polars: type conversions, milestone columns, HomeOrAway, comp merge, etc."""
    if bat_df.empty:
        return bat_df
    # Convert to Polars DataFrame
    pl_df = pl.from_pandas(bat_df)
    # Merge comp from match_df if available
    if not match_df.empty and 'File Name' in pl_df.columns and 'comp' in match_df.columns:
        comp_mapping = pl.from_pandas(match_df[['File Name', 'comp']].drop_duplicates())
        pl_df = pl_df.join(comp_mapping, on='File Name', how='left')
        pl_df = pl_df.with_columns([
            pl.when(pl.col('comp').is_null())
              .then(pl.col('Competition'))
              .otherwise(pl.col('comp')).alias('comp')
        ])
    else:
        pl_df = pl_df.with_columns([
            pl.col('Competition').alias('comp')
        ])
    # Date to datetime
    pl_df = pl_df.with_columns([
        pl.col('Date').str.strptime(pl.Date, format='%d %b %Y', strict=False).alias('Date')
    ])
    # Milestone columns
    pl_df = pl_df.with_columns([
        ((pl.col('Runs') >= 50) & (pl.col('Runs') < 100)).cast(pl.Int8).alias('50s'),
        ((pl.col('Runs') >= 100) & (pl.col('Runs') < 150)).cast(pl.Int8).alias('100s'),
        ((pl.col('Runs') >= 150) & (pl.col('Runs') < 200)).cast(pl.Int8).alias('150s'),
        (pl.col('Runs') >= 200).cast(pl.Int8).alias('200s'),
        pl.col('Year').cast(pl.Int32).alias('Year')
    ])
    # HomeOrAway
    pl_df = pl_df.with_columns([
        pl.when(pl.col('Home Team') == pl.col('Bat_Team_y')).then('Home')
         .when(pl.col('Away Team') == pl.col('Bat_Team_y')).then('Away')
         .otherwise('Neutral').alias('HomeOrAway')
    ])
    return pl_df
    """
    Get available options for a column based on current filter selections.
    """
    if selected_filters is None:
        return ['All'] + sorted(df[column].unique().tolist())

    filtered_df = df  # No .copy() needed, as we do not mutate

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
                                   'Boundary%', 'RPM', '<25&Out', '50s', '100s', '150s', '200s',
                                   'Conversion Rate', '<25&OutPI', '50+PI', '100PI', '150PI', '200PI',
                                   'Match+ Avg', 'Match+ SR', 'Team+ Avg', 'Team+ SR', 'Caught%', 'Bowled%', 'LBW%', 
                                   'Run Out%', 'Stumped%', 'Not Out%', 'POM', 'POM Per Match']]
    # Sort the DataFrame by 'Runs' in descending order
    bat_career_df = bat_career_df.sort_values(by='Runs', ascending=False)
    return bat_career_df

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
                        'Avg', 'BPO', 'SR', '4s', '6s', 'BPB', '<25&Out', '50s', '100s', 
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
                                   'Avg', 'BPO', 'SR', '4s', '6s', 'BPB', '<25&Out', '50s', '100s', 
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

    # Timer start
    timer_start = time.time()

    timer_end = None
    elapsed = None
    if 'bat_df' in st.session_state:
        bat_df = st.session_state['bat_df']
        match_df = st.session_state.get('match_df', pd.DataFrame())
        unique_matches = bat_df['File Name'].nunique()
        if unique_matches <= 1:
            timer_end = time.time()
            elapsed = timer_end - timer_start
            st.success(f"⏱️ Batting view loaded in {elapsed:.2f} seconds.")
            st.markdown("""
            <div class="modern-warning">
                <p style="margin: 0; font-weight: 600; color: #856404;">
                    ⚠️ Please upload more than 1 scorecard to use the batting statistics view effectively. 
                    With only one match loaded, statistical analysis and comparisons are limited.
                </p>
            </div>
            """, unsafe_allow_html=True)
            return

        # Use st.session_state to cache preprocessed Polars DataFrame
        if 'pl_df' not in st.session_state:
            @st.cache_data(ttl=3600, show_spinner=False)
            def cached_preprocess(bat_df, match_df):
                return preprocess_batting_data(bat_df, match_df)
            st.session_state['pl_df'] = cached_preprocess(bat_df, match_df)
        pl_df = st.session_state['pl_df']

        # Row limit slider for all tables
        max_rows = st.slider('Max rows to display', min_value=10, max_value=1000, value=100, step=10, key='bat_row_limit')

<<<<<<< HEAD
        # --- Career Tab: Polars end-to-end ---
        pl_career = pl_df.select([
            'Name', 'File Name', 'Batted', 'Out', 'Not Out', 'Balls', 'Runs', '4s', '6s', '50s', '100s', '150s', '200s'
=======
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
            "Location", "Innings", "Position", "Home/Away",
            "Cumulative", "Block"
>>>>>>> 7c723bb689ae91ae1eb065d2c05a038af9ae7e4f
        ])
        @st.cache_data(ttl=3600, show_spinner=False)
        def polars_career_agg(pl_career):
            agg_df = (
                pl_career.groupby('Name').agg([
                    pl.col('File Name').n_unique().alias('Matches'),
                    pl.col('Batted').sum().alias('Inns'),
                    pl.col('Out').sum().alias('Out'),
                    pl.col('Not Out').sum().alias('Not Out'),
                    pl.col('Balls').sum().alias('Balls'),
                    pl.col('Runs').sum().alias('Runs'),
                    pl.col('Runs').max().alias('HS'),
                    pl.col('4s').sum().alias('4s'),
                    pl.col('6s').sum().alias('6s'),
                    pl.col('50s').sum().alias('50s'),
                    pl.col('100s').sum().alias('100s'),
                    pl.col('150s').sum().alias('150s'),
                    pl.col('200s').sum().alias('200s'),
                ])
            )
            agg_df = agg_df.with_columns([
                (pl.col('Runs') / pl.col('Out')).round(2).fill_null(0).alias('Avg'),
                ((pl.col('Runs') / pl.col('Balls')) * 100).round(2).fill_null(0).alias('SR'),
                (pl.col('Balls') / pl.col('Out')).round(2).fill_null(0).alias('BPO'),
            ])
            return agg_df
        agg_df = polars_career_agg(pl_career)
        display_df = agg_df.sort('Runs', descending=True).to_pandas().head(max_rows)

        timer_end = time.time()
        elapsed = timer_end - timer_start
        st.success(f"⏱️ Batting view loaded in {elapsed:.2f} seconds.")
        st.dataframe(display_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})
    else:
        timer_end = time.time()
        elapsed = timer_end - timer_start
        st.success(f"⏱️ Batting view loaded in {elapsed:.2f} seconds.")

@st.cache_data(ttl=3600, show_spinner=False, hash_funcs={dict: lambda x: hash(str(sorted(x.items())))})
def apply_optimized_filters(df, filters):
    """Ultra-fast filtering using vectorized operations and categorical dtypes"""
    
    # Start with the full DataFrame
    filtered_df = df  # No .copy() needed, as we do not mutate
    
    # Create a boolean mask for all filters at once
    mask = pd.Series(True, index=filtered_df.index, dtype=bool)
    
    # Apply categorical filters using .isin() which is much faster for categories
    categorical_filters = {
        'names': 'Name',
        'bat_teams': 'Bat_Team_y', 
        'bowl_teams': 'Bowl_Team_y',
        'formats': 'Match_Format',
        'comp': 'comp'
    }
    
    for filter_key, column in categorical_filters.items():
        filter_values = filters.get(filter_key)
        if filter_values and 'All' not in filter_values and column in filtered_df.columns:
            mask &= filtered_df[column].isin(filter_values)
    
    # Apply numerical range filters using vectorized between operations
    range_filters = {
        'year_range': 'Year',
        'position_range': 'Position', 
        'runs_range': 'Runs'
    }
    
    for filter_key, column in range_filters.items():
        filter_range = filters.get(filter_key)
        if filter_range and column in filtered_df.columns:
            mask &= filtered_df[column].between(filter_range[0], filter_range[1], inclusive='both')
    
    # Apply the combined mask
    filtered_df = filtered_df[mask]
    
    # Optimize HomeOrAway calculation using vectorized operations
    if 'HomeOrAway' not in filtered_df.columns and 'Home Team' in filtered_df.columns:
        # Use numpy.where for vectorized conditional assignment
        home_mask = filtered_df['Home Team'] == filtered_df['Bat_Team_y']
        away_mask = filtered_df['Away Team'] == filtered_df['Bat_Team_y']
        
<<<<<<< HEAD
        filtered_df['HomeOrAway'] = np.where(
            home_mask, 'Home',
            np.where(away_mask, 'Away', 'Neutral')
        )
    
    return filtered_df

@st.cache_data(ttl=3600, show_spinner=False)
def calculate_player_stats_fast(df):
    """Optimized player statistics calculation using single groupby operation"""
    
    if df.empty:
        return pd.DataFrame()
    
    # Single comprehensive aggregation for all needed stats
    agg_dict = {
        'File Name': 'nunique',
        'Runs': ['sum', 'max'],
        'Out': 'sum',
        'Balls': 'sum',
        'Not Out': 'sum',
        '4s': 'sum',
        '6s': 'sum',
        '50s': 'sum', 
        '100s': 'sum',
        '200s': 'sum',
        'Batted': 'sum'
    }
    
    # Optional columns - add only if they exist
    optional_cols = ['Total_Runs', 'Wickets', 'Team Balls', '<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped']
    for col in optional_cols:
        if col in df.columns:
            agg_dict[col] = 'sum'
    
    # Perform single groupby operation
    player_stats = df.groupby('Name', observed=True).agg(agg_dict)
    
    # Flatten multi-level columns
    player_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in player_stats.columns]
    player_stats = player_stats.reset_index()
    
    # Rename columns to standard format
    column_renames = {
        'File Name_nunique': 'Matches',
        'Runs_sum': 'Runs', 
        'Runs_max': 'HS',
        'Out_sum': 'Out',
        'Balls_sum': 'Balls',
        'Not Out_sum': 'Not Out',
        '4s_sum': '4s',
        '6s_sum': '6s',
        '50s_sum': '50s',
        '100s_sum': '100s', 
        '200s_sum': '200s',
        'Batted_sum': 'Inns'
    }
    
    # Add optional column renames
    for col in optional_cols:
        if f'{col}_sum' in player_stats.columns:
            if col == 'Total_Runs':
                column_renames[f'{col}_sum'] = 'Team_Runs'
            elif col == 'Team Balls':
                column_renames[f'{col}_sum'] = 'Team_Balls'
            else:
                column_renames[f'{col}_sum'] = col
    
    player_stats.rename(columns=column_renames, inplace=True)
    
    # Apply vectorized calculations
    player_stats = vectorized_calculations(player_stats)
    
    return player_stats

@st.cache_data(ttl=3600, show_spinner=False)
def apply_advanced_filters_fast(player_stats, filters):
    """Fast advanced filtering on aggregated player stats"""
    
    if player_stats.empty:
        return player_stats
    
    # Create boolean mask for all conditions
    mask = pd.Series(True, index=player_stats.index, dtype=bool)
    
    # Apply range filters
    range_filters = {
        'matches_range': 'Matches',
        'avg_range': 'Avg', 
        'sr_range': 'SR'
    }
    
    for filter_key, column in range_filters.items():
        filter_range = filters.get(filter_key)
        if filter_range and column in player_stats.columns:
            mask &= player_stats[column].between(filter_range[0], filter_range[1], inclusive='both')
    
    # Apply P+ filters if columns exist
    if 'p_avg_range' in filters and 'P+ Avg' in player_stats.columns:
        p_avg_range = filters['p_avg_range']
        mask &= player_stats['P+ Avg'].between(p_avg_range[0], p_avg_range[1], inclusive='both')
    
    if 'p_sr_range' in filters and 'P+ SR' in player_stats.columns:
        p_sr_range = filters['p_sr_range']
        mask &= player_stats['P+ SR'].between(p_sr_range[0], p_sr_range[1], inclusive='both')
    return player_stats[mask]

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
        'comp': comp_choice
    }
    cache_key = generate_cache_key(filters)

    # Apply optimized filtering
    if 'optimized_bat_df' not in st.session_state:
        st.session_state['optimized_bat_df'] = optimize_dataframe(bat_df)
    
    optimized_df = st.session_state['optimized_bat_df']
    filtered_df = apply_optimized_filters(optimized_df, filters)
    
    # Calculate player stats with optimized functions
    player_stats = calculate_player_stats_fast(filtered_df)
    final_player_stats = apply_advanced_filters_fast(player_stats, filters)
    
    # Filter the main dataframe based on selected players
    if not final_player_stats.empty:
        filtered_df = filtered_df[filtered_df['Name'].isin(final_player_stats['Name'])]

    # Create a placeholder for tabs that will be lazily loaded
    main_container = st.container()
    
    # Create tabs for different views - Removed "Distributions" and "Percentile" tabs for performance
    tabs = main_container.tabs([
        "Career", "Format", "Season", "Latest", "Opponent", 
        "Location", "Innings", "Position", "Home/Away",
        "Cumulative", "Block", "Records", "Win/Loss"
    ])
    
    # Career Stats Tab
    with tabs[0]:
        career_cache_key = f"{cache_key}_career_stats"
        bat_career_df = get_cached_dataframe(career_cache_key)
        
        if bat_career_df is None:
                # First, calculate file-level statistics
                file_stats = filtered_df.groupby('File Name').agg({
                    'Runs': 'sum',
                    'Team Balls': 'sum',
                    'Wickets': 'sum',
                    'Total_Runs': 'sum'
                }).reset_index()

                # Calculate match-level metrics
                file_stats['Match_Avg'] = file_stats['Runs'] / file_stats['Wickets'].replace(0, np.nan)
                file_stats['Match_SR'] = (file_stats['Runs'] / file_stats['Team Balls'].replace(0, np.nan)) * 100

                # Calculate average match metrics
                avg_match_stats = {
                    'Match_Avg': file_stats['Match_Avg'].mean(),
                    'Match_SR': file_stats['Match_SR'].mean()
                }

                # Now proceed with career stats calculation
                career_cache_key = f"{cache_key}_career_stats"
                bat_career_df = get_cached_dataframe(career_cache_key)
                
                if bat_career_df is None:
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
                    bat_career_df['Avg'] = (bat_career_df['Runs'] / bat_career_df['Out']).round(2).fillna(0)
                    bat_career_df['SR'] = ((bat_career_df['Runs'] / bat_career_df['Balls']) * 100).round(2).fillna(0)
                    bat_career_df['BPO'] = (bat_career_df['Balls'] / bat_career_df['Out']).round(2).fillna(0)

                    # Calculate new columns for team statistics
                    bat_career_df['Team Avg'] = (bat_career_df['Team Runs'] / bat_career_df['Wickets']).round(2).fillna(0)
                    bat_career_df['Team SR'] = (bat_career_df['Team Runs'] / bat_career_df['Team Balls'] * 100).round(2).fillna(0)

                    # Calculate P+ Avg and P+ SR
                    bat_career_df['Team+ Avg'] = (bat_career_df['Avg'] / bat_career_df['Team Avg'] * 100).round(2).fillna(0)
                    bat_career_df['Team+ SR'] = (bat_career_df['SR'] / bat_career_df['Team SR'] * 100).round(2).fillna(0)

                    # Calculate match comparison metrics
                    bat_career_df['Match+ Avg'] = (bat_career_df['Avg'] / avg_match_avg * 100).round(2)
                    bat_career_df['Match+ SR'] = (bat_career_df['SR'] / avg_match_sr * 100).round(2)

                    # Calculate BPB (Balls Per Boundary)
                    bat_career_df['BPB'] = (bat_career_df['Balls'] / (bat_career_df['4s'] + bat_career_df['6s']).replace(0, 1)).round(2)
                    bat_career_df['Boundary%'] = (((bat_career_df['4s'] * 4) + (bat_career_df['6s'] * 6))   / (bat_career_df['Runs'].replace(0, 1))* 100 ).round(2)
                    bat_career_df['RPM'] = (bat_career_df['Runs'] / bat_career_df['Matches'].replace(0, 1)).round(2)

                    # Calculate new statistics
                    bat_career_df['50+PI'] = (((bat_career_df['50s'] + bat_career_df['100s'] + bat_career_df['150s'] + bat_career_df['200s']) / bat_career_df['Inns']) * 100).round(2).fillna(0)
                    bat_career_df['100PI'] = (((bat_career_df['100s'] + bat_career_df['150s'] + bat_career_df['200s']) / bat_career_df['Inns']) * 100).round(2).fillna(0)
                    bat_career_df['150PI'] = (((bat_career_df['150s'] + bat_career_df['200s']) / bat_career_df['Inns']) * 100).round(2).fillna(0)
                    bat_career_df['200PI'] = ((bat_career_df['200s'] / bat_career_df['Inns']) * 100).round(2).fillna(0)
                    bat_career_df['<25&OutPI'] = ((bat_career_df['<25&Out'] / bat_career_df['Inns']) * 100).round(2).fillna(0)
                    bat_career_df['Conversion Rate'] = ((bat_career_df['100s'] / (bat_career_df['50s'] + bat_career_df['100s']).replace(0, 1)) * 100).round(2)

                    # Calculate dismissal percentages
                    bat_career_df['Caught%'] = ((bat_career_df['Caught'] / bat_career_df['Inns']) * 100).round(2).fillna(0)
                    bat_career_df['Bowled%'] = ((bat_career_df['Bowled'] / bat_career_df['Inns']) * 100).round(2).fillna(0)
                    bat_career_df['LBW%'] = ((bat_career_df['LBW'] / bat_career_df['Inns']) * 100).round(2).fillna(0)
                    bat_career_df['Run Out%'] = ((bat_career_df['Run Out'] / bat_career_df['Inns']) * 100).round(2).fillna(0)
                    bat_career_df['Stumped%'] = ((bat_career_df['Stumped'] / bat_career_df['Inns']) * 100).round(2).fillna(0)
                    bat_career_df['Not Out%'] = ((bat_career_df['Not Out'] / bat_career_df['Inns']) * 100).round(2).fillna(0)

                    # Count Player of the Match awards
                    pom_counts = filtered_df[filtered_df['Player_of_the_Match'] == filtered_df['Name']].groupby('Name')['File Name'].nunique().reset_index(name='POM')
                    bat_career_df = bat_career_df.merge(pom_counts, on='Name', how='left')
                    bat_career_df['POM'] = bat_career_df['POM'].fillna(0).astype(int)
                    bat_career_df['POM Per Match'] = (bat_career_df['POM'] / bat_career_df['Matches'].replace(0, 1)*100).round(2)



                    # Reorder columns and drop Team Avg and Team SR
                    bat_career_df = bat_career_df[['Name', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 
                                                   'Runs', 'HS', 'Avg', 'BPO', 'SR', '4s', '6s', 'BPB',
                                                   'Boundary%', 'RPM', '<25&Out', '50s', '100s', '150s', '200s',
                                                   'Conversion Rate', '<25&OutPI', '50+PI', '100PI', '150PI', '200PI',
                                                   'Match+ Avg', 'Match+ SR', 'Team+ Avg', 'Team+ SR', 'Caught%', 'Bowled%', 'LBW%', 
                                                   'Run Out%', 'Stumped%', 'Not Out%', 'POM', 'POM Per Match']]
                    
                    # Sort the DataFrame by 'Runs' in descending order
                    bat_career_df = bat_career_df.sort_values(by='Runs', ascending=False)
                    
                # Cache the computed career statistics
                cache_dataframe(career_cache_key, bat_career_df)

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
        scatter_cache_key = f"{cache_key}_scatter_data"
        scatter_data = get_cached_dataframe(scatter_cache_key)
        
        if scatter_data is None:
            # Create a new figure for the scatter plot
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

            # Store the figure data in cache
            cache_dataframe(scatter_cache_key, {
                'fig': scatter_fig,
                'data': bat_career_df[['Name', 'Avg', 'SR', 'Runs']].to_dict('records')
            })

        else:
            # Recreate figure from cached data
            scatter_fig = go.Figure()
            for player in scatter_data['data']:
                scatter_fig.add_trace(go.Scatter(
                    x=[player['Avg']],
                    y=[player['SR']],
                    mode='markers+text',
                    text=[player['Name']],
                    textposition='top center',
                    marker=dict(size=10),
                    name=player['Name'],
                    hovertemplate=(
                        f"<b>{player['Name']}</b><br><br>"
                        f"Batting Average: {player['Avg']:.2f}<br>"
                        f"Strike Rate: {player['SR']:.2f}<br>"
                        f"Runs: {player['Runs']}<br>"
                        "<extra></extra>"
                    )
                ))

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
=======
        # Career Stats Tab
        with tabs[0]:
            bat_career_df = compute_career_stats(filtered_df)
            
>>>>>>> 7c723bb689ae91ae1eb065d2c05a038af9ae7e4f
            st.markdown("""
            <div style="background: linear-gradient(135deg, #36d1dc 0%, #5b86e5 100%); 
                        padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                        box-shadow: 0 6px 24px rgba(54, 209, 220, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">📊 Batting Average vs Strike Rate Analysis</h3>
            </div>
            """, unsafe_allow_html=True)
<<<<<<< HEAD
            # Show first plot
            st.plotly_chart(scatter_fig, use_container_width=True)

        with col2:
            # Create new scatter plot for Strike Rate vs Balls Per Out
            sr_bpo_fig = go.Figure()

            # Plot data for each player
            for name in bat_career_df['Name'].unique():
                player_stats = bat_career_df[bat_career_df['Name'] == name]
=======
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
>>>>>>> 7c723bb689ae91ae1eb065d2c05a038af9ae7e4f

                # Get statistics
                strike_rate = player_stats['SR'].iloc[0]
                balls_per_out = player_stats['BPO'].iloc[0]
                runs = player_stats['Runs'].iloc[0]

<<<<<<< HEAD
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
=======
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
>>>>>>> 7c723bb689ae91ae1eb065d2c05a038af9ae7e4f

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
            st.plotly_chart(sr_bpo_fig, use_container_width=True)


<<<<<<< HEAD
        # Format Stats Tab (Polars end-to-end)
        with tabs[1]:
            @st.cache_data(ttl=3600, show_spinner=False)
            def polars_format_agg(pl_df):
                pl_format = pl_df.select([
                    'Name', 'Match_Format', 'File Name', 'Batted', 'Out', 'Not Out', 'Balls', 'Runs', '4s', '6s', '50s', '100s', '200s', '<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 'Total_Runs', 'Overs', 'Wickets', 'Team Balls'
                ])
                pl_format_agg = (
                    pl_format.groupby(['Name', 'Match_Format']).agg([
                        pl.col('File Name').n_unique().alias('Matches'),
                        pl.col('Batted').sum().alias('Inns'),
                        pl.col('Out').sum().alias('Out'),
                        pl.col('Not Out').sum().alias('Not Out'),
                        pl.col('Balls').sum().alias('Balls'),
                        pl.col('Runs').sum().alias('Runs'),
                        pl.col('Runs').max().alias('HS'),
                        pl.col('4s').sum().alias('4s'),
                        pl.col('6s').sum().alias('6s'),
                        pl.col('50s').sum().alias('50s'),
                        pl.col('100s').sum().alias('100s'),
                        pl.col('200s').sum().alias('200s'),
                        pl.col('<25&Out').sum().alias('<25&Out'),
                        pl.col('Caught').sum().alias('Caught'),
                        pl.col('Bowled').sum().alias('Bowled'),
                        pl.col('LBW').sum().alias('LBW'),
                        pl.col('Run Out').sum().alias('Run Out'),
                        pl.col('Stumped').sum().alias('Stumped'),
                        pl.col('Total_Runs').sum().alias('Team Runs'),
                        pl.col('Overs').sum().alias('Overs'),
                        pl.col('Wickets').sum().alias('Wickets'),
                        pl.col('Team Balls').sum().alias('Team Balls'),
                    ])
                )
                pl_format_agg = pl_format_agg.with_columns([
                    (pl.col('Runs') / pl.col('Out')).round(2).fill_null(0).alias('Avg'),
                    ((pl.col('Runs') / pl.col('Balls')) * 100).round(2).fill_null(0).alias('SR'),
                    (pl.col('Balls') / pl.col('Out')).round(2).fill_null(0).alias('BPO'),
                    (pl.col('Team Runs') / pl.col('Wickets')).round(2).fill_null(0).alias('Team Avg'),
                    (pl.col('Team Runs') / pl.col('Team Balls') * 100).round(2).fill_null(0).alias('Team SR'),
                    (pl.col('Avg') / pl.col('Team Avg') * 100).round(2).fill_null(0).alias('P+ Avg'),
                    (pl.col('SR') / pl.col('Team SR') * 100).round(2).fill_null(0).alias('P+ SR'),
                    (pl.col('Balls') / (pl.col('4s') + pl.col('6s')).replace(0, 1)).round(2).fill_null(0).alias('BPB'),
                    (((pl.col('50s') + pl.col('100s')) / pl.col('Inns')) * 100).round(2).fill_null(0).alias('50+PI'),
                    ((pl.col('100s') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('100PI'),
                    ((pl.col('<25&Out') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('<25&OutPI'),
                    ((pl.col('Caught') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Caught%'),
                    ((pl.col('Bowled') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Bowled%'),
                    ((pl.col('LBW') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('LBW%'),
                    ((pl.col('Run Out') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Run Out%'),
                    ((pl.col('Stumped') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Stumped%'),
                    ((pl.col('Not Out') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Not Out%'),
                ])
                return pl_format_agg
            pl_format_agg = polars_format_agg(pl_df)
            df_format = pl_format_agg.sort('Runs', descending=True).to_pandas().head(max_rows)
=======
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
            df_format = compute_format_stats(filtered_df)
>>>>>>> 7c723bb689ae91ae1eb065d2c05a038af9ae7e4f

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
<<<<<<< HEAD
        

        # Season Stats Tab (Polars end-to-end)
        with tabs[2]:
            @st.cache_data(ttl=3600, show_spinner=False)
            def polars_season_agg(pl_df):
                pl_season = pl_df.select([
                    'Name', 'Year', 'File Name', 'Batted', 'Out', 'Not Out', 'Balls', 'Runs', '4s', '6s', '50s', '100s', '200s', '<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 'Total_Runs', 'Overs', 'Wickets', 'Team Balls'
                ])
                pl_season_agg = (
                    pl_season.groupby(['Name', 'Year']).agg([
                        pl.col('File Name').n_unique().alias('Matches'),
                        pl.col('Batted').sum().alias('Inns'),
                        pl.col('Out').sum().alias('Out'),
                        pl.col('Not Out').sum().alias('Not Out'),
                        pl.col('Balls').sum().alias('Balls'),
                        pl.col('Runs').sum().alias('Runs'),
                        pl.col('Runs').max().alias('HS'),
                        pl.col('4s').sum().alias('4s'),
                        pl.col('6s').sum().alias('6s'),
                        pl.col('50s').sum().alias('50s'),
                        pl.col('100s').sum().alias('100s'),
                        pl.col('200s').sum().alias('200s'),
                        pl.col('<25&Out').sum().alias('<25&Out'),
                        pl.col('Caught').sum().alias('Caught'),
                        pl.col('Bowled').sum().alias('Bowled'),
                        pl.col('LBW').sum().alias('LBW'),
                        pl.col('Run Out').sum().alias('Run Out'),
                        pl.col('Stumped').sum().alias('Stumped'),
                        pl.col('Total_Runs').sum().alias('Team Runs'),
                        pl.col('Overs').sum().alias('Overs'),
                        pl.col('Wickets').sum().alias('Wickets'),
                        pl.col('Team Balls').sum().alias('Team Balls'),
                    ])
                )
                pl_season_agg = pl_season_agg.with_columns([
                    (pl.col('Runs') / pl.col('Out')).round(2).fill_null(0).alias('Avg'),
                    ((pl.col('Runs') / pl.col('Balls')) * 100).round(2).fill_null(0).alias('SR'),
                    (pl.col('Balls') / pl.col('Out')).round(2).fill_null(0).alias('BPO'),
                    (pl.col('Team Runs') / pl.col('Wickets')).round(2).fill_null(0).alias('Team Avg'),
                    (pl.col('Team Runs') / pl.col('Team Balls') * 100).round(2).fill_null(0).alias('Team SR'),
                    (pl.col('Avg') / pl.col('Team Avg') * 100).round(2).fill_null(0).alias('P+ Avg'),
                    (pl.col('SR') / pl.col('Team SR') * 100).round(2).fill_null(0).alias('P+ SR'),
                    (pl.col('Balls') / (pl.col('4s') + pl.col('6s')).replace(0, 1)).round(2).fill_null(0).alias('BPB'),
                    (((pl.col('50s') + pl.col('100s')) / pl.col('Inns')) * 100).round(2).fill_null(0).alias('50+PI'),
                    ((pl.col('100s') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('100PI'),
                    ((pl.col('<25&Out') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('<25&OutPI'),
                    ((pl.col('Caught') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Caught%'),
                    ((pl.col('Bowled') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Bowled%'),
                    ((pl.col('LBW') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('LBW%'),
                    ((pl.col('Run Out') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Run Out%'),
                    ((pl.col('Stumped') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Stumped%'),
                    ((pl.col('Not Out') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Not Out%'),
                ])
                return pl_season_agg
            pl_season_agg = polars_season_agg(pl_df)
            df_season = pl_season_agg.sort(['Year', 'Runs'], descending=[True, True]).to_pandas().head(max_rows)
            st.markdown("""
            <div style="background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(67, 206, 162, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">📅 Season Record</h3>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(df_season, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})


    # Opponent Stats Tab (Polars end-to-end)
    with tabs[4]:
        @st.cache_data(ttl=3600, show_spinner=False)
        def polars_opponent_agg(pl_df):
            pl_opponent = pl_df.select([
                'Name', 'Bowl_Team_y', 'File Name', 'Batted', 'Out', 'Not Out', 'Balls', 'Runs', '4s', '6s', '50s', '100s', '200s', '<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 'Total_Runs', 'Overs', 'Wickets', 'Team Balls'
            ])
            pl_opponent_agg = (
                pl_opponent.groupby(['Name', 'Bowl_Team_y']).agg([
                    pl.col('File Name').n_unique().alias('Matches'),
                    pl.col('Batted').sum().alias('Inns'),
                    pl.col('Out').sum().alias('Out'),
                    pl.col('Not Out').sum().alias('Not Out'),
                    pl.col('Balls').sum().alias('Balls'),
                    pl.col('Runs').sum().alias('Runs'),
                    pl.col('Runs').max().alias('HS'),
                    pl.col('4s').sum().alias('4s'),
                    pl.col('6s').sum().alias('6s'),
                    pl.col('50s').sum().alias('50s'),
                    pl.col('100s').sum().alias('100s'),
                    pl.col('200s').sum().alias('200s'),
                    pl.col('<25&Out').sum().alias('<25&Out'),
                    pl.col('Caught').sum().alias('Caught'),
                    pl.col('Bowled').sum().alias('Bowled'),
                    pl.col('LBW').sum().alias('LBW'),
                    pl.col('Run Out').sum().alias('Run Out'),
                    pl.col('Stumped').sum().alias('Stumped'),
                    pl.col('Total_Runs').sum().alias('Team Runs'),
                    pl.col('Overs').sum().alias('Overs'),
                    pl.col('Wickets').sum().alias('Wickets'),
                    pl.col('Team Balls').sum().alias('Team Balls'),
                ])
            )
            pl_opponent_agg = pl_opponent_agg.with_columns([
                (pl.col('Runs') / pl.col('Out')).round(2).fill_null(0).alias('Avg'),
                ((pl.col('Runs') / pl.col('Balls')) * 100).round(2).fill_null(0).alias('SR'),
                (pl.col('Balls') / pl.col('Out')).round(2).fill_null(0).alias('BPO'),
                (pl.col('Team Runs') / pl.col('Wickets')).round(2).fill_null(0).alias('Team Avg'),
                (pl.col('Team Runs') / pl.col('Team Balls') * 100).round(2).fill_null(0).alias('Team SR'),
                (pl.col('Avg') / pl.col('Team Avg') * 100).round(2).fill_null(0).alias('P+ Avg'),
                (pl.col('SR') / pl.col('Team SR') * 100).round(2).fill_null(0).alias('P+ SR'),
                (pl.col('Balls') / (pl.col('4s') + pl.col('6s')).replace(0, 1)).round(2).fill_null(0).alias('BPB'),
                (((pl.col('50s') + pl.col('100s')) / pl.col('Inns')) * 100).round(2).fill_null(0).alias('50+PI'),
                ((pl.col('100s') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('100PI'),
                ((pl.col('<25&Out') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('<25&OutPI'),
                ((pl.col('Caught') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Caught%'),
                ((pl.col('Bowled') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Bowled%'),
                ((pl.col('LBW') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('LBW%'),
                ((pl.col('Run Out') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Run Out%'),
                ((pl.col('Stumped') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Stumped%'),
                ((pl.col('Not Out') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Not Out%'),
            ])
            return pl_opponent_agg
        pl_opponent_agg = polars_opponent_agg(pl_df)
        df_opponent = pl_opponent_agg.sort('Runs', descending=True).to_pandas().head(max_rows)
=======
            
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
        with tabs[9]:
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
        with tabs[10]:
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
>>>>>>> 7c723bb689ae91ae1eb065d2c05a038af9ae7e4f
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); 
                    padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                    box-shadow: 0 8px 32px rgba(247, 151, 30, 0.3);
                    border: 1px solid rgba(255, 255, 255, 0.2);">
            <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">🤝 Opponent Record</h3>
        </div>
        """, unsafe_allow_html=True)
        # Ensure numeric columns are correct dtype for Arrow compatibility
        numeric_cols = [
            'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 'Runs', 'HS', '4s', '6s', '50s', '100s', '200s', '<25&Out',
            'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 'Team Runs', 'Overs', 'Wickets', 'Team Balls',
            'Avg', 'SR', 'BPO', 'Team Avg', 'Team SR', 'P+ Avg', 'P+ SR', 'BPB', '50+PI', '100PI', '<25&OutPI',
            'Caught%', 'Bowled%', 'LBW%', 'Run Out%', 'Stumped%', 'Not Out%'
        ]
        for col in numeric_cols:
            if col in df_opponent.columns:
                df_opponent[col] = pd.to_numeric(df_opponent[col], errors='coerce')
        st.dataframe(df_opponent, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

    # Latest Stats Tab (Polars backend)
    with tabs[3]:
        latest_cache_key = f"{cache_key}_latest_stats"
        df_latest = get_cached_dataframe(latest_cache_key)
        if df_latest is None:
            pl_latest = pl.from_pandas(filtered_df)
            pl_latest_agg = (
                pl_latest.sort('Date', descending=True)
                .groupby('Name')
                .agg([
                    pl.col('Date').first().alias('Latest Date'),
                    pl.col('Runs').first().alias('Latest Runs'),
                    pl.col('Balls').first().alias('Latest Balls'),
                    pl.col('Out').first().alias('Latest Out'),
                    pl.col('4s').first().alias('Latest 4s'),
                    pl.col('6s').first().alias('Latest 6s'),
                    pl.col('Match_Format').first().alias('Latest Format'),
                    pl.col('Bowl_Team_y').first().alias('Latest Opponent'),
                ])
            )
            df_latest = pl_latest_agg.to_pandas()
            cache_dataframe(latest_cache_key, df_latest)
        st.markdown("""
        <div style="background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%); 
                    padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                    box-shadow: 0 8px 32px rgba(0, 198, 255, 0.3);
                    border: 1px solid rgba(255, 255, 255, 0.2);">
            <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">🕒 Latest Innings</h3>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df_latest, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

    # Location Stats Tab (Polars backend)
    with tabs[5]:
        location_cache_key = f"{cache_key}_location_stats"
        df_location = get_cached_dataframe(location_cache_key)
        if df_location is None:
            pl_location = pl.from_pandas(filtered_df)
            pl_location_agg = (
                pl_location.groupby(['Name', 'HomeOrAway']).agg([
                    pl.col('File Name').n_unique().alias('Matches'),
                    pl.col('Batted').sum().alias('Inns'),
                    pl.col('Out').sum().alias('Out'),
                    pl.col('Not Out').sum().alias('Not Out'),
                    pl.col('Balls').sum().alias('Balls'),
                    pl.col('Runs').sum().alias('Runs'),
                    pl.col('Runs').max().alias('HS'),
                    pl.col('4s').sum().alias('4s'),
                    pl.col('6s').sum().alias('6s'),
                    pl.col('50s').sum().alias('50s'),
                    pl.col('100s').sum().alias('100s'),
                    pl.col('200s').sum().alias('200s'),
                    pl.col('<25&Out').sum().alias('<25&Out'),
                    pl.col('Caught').sum().alias('Caught'),
                    pl.col('Bowled').sum().alias('Bowled'),
                    pl.col('LBW').sum().alias('LBW'),
                    pl.col('Run Out').sum().alias('Run Out'),
                    pl.col('Stumped').sum().alias('Stumped'),
                    pl.col('Total_Runs').sum().alias('Team Runs'),
                    pl.col('Overs').sum().alias('Overs'),
                    pl.col('Wickets').sum().alias('Wickets'),
                    pl.col('Team Balls').sum().alias('Team Balls'),
                ])
            )
            pl_location_agg = pl_location_agg.with_columns([
                (pl.col('Runs') / pl.col('Out')).round(2).fill_null(0).alias('Avg'),
                ((pl.col('Runs') / pl.col('Balls')) * 100).round(2).fill_null(0).alias('SR'),
                (pl.col('Balls') / pl.col('Out')).round(2).fill_null(0).alias('BPO'),
                (pl.col('Team Runs') / pl.col('Wickets')).round(2).fill_null(0).alias('Team Avg'),
                (pl.col('Team Runs') / pl.col('Team Balls') * 100).round(2).fill_null(0).alias('Team SR'),
                (pl.col('Avg') / pl.col('Team Avg') * 100).round(2).fill_null(0).alias('P+ Avg'),
                (pl.col('SR') / pl.col('Team SR') * 100).round(2).fill_null(0).alias('P+ SR'),
                (pl.col('Balls') / (pl.col('4s') + pl.col('6s')).replace(0, 1)).round(2).fill_null(0).alias('BPB'),
                (((pl.col('50s') + pl.col('100s')) / pl.col('Inns')) * 100).round(2).fill_null(0).alias('50+PI'),
                ((pl.col('100s') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('100PI'),
                ((pl.col('<25&Out') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('<25&OutPI'),
                ((pl.col('Caught') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Caught%'),
                ((pl.col('Bowled') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Bowled%'),
                ((pl.col('LBW') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('LBW%'),
                ((pl.col('Run Out') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Run Out%'),
                ((pl.col('Stumped') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Stumped%'),
                ((pl.col('Not Out') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Not Out%'),
            ])
            df_location = pl_location_agg.to_pandas()
            df_location = df_location.sort_values(by=["Runs"], ascending=[False])
            cache_dataframe(location_cache_key, df_location)
        st.markdown("""
        <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                    padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                    box-shadow: 0 8px 32px rgba(17, 153, 142, 0.3);
                    border: 1px solid rgba(255, 255, 255, 0.2);">
            <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">🏟️ Location Record</h3>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df_location, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

    # Innings Stats Tab (Polars backend)
    with tabs[6]:
        innings_cache_key = f"{cache_key}_innings_stats"
        df_innings = get_cached_dataframe(innings_cache_key)
        if df_innings is None:
            pl_innings = pl.from_pandas(filtered_df)[[
                'Name', 'Innings', 'File Name', 'Batted', 'Out', 'Not Out', 'Balls', 'Runs', '4s', '6s', '50s', '100s', '200s', '<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 'Total_Runs', 'Overs', 'Wickets', 'Team Balls'
            ]]
            pl_innings_agg = (
                pl_innings.groupby(['Name', 'Innings']).agg([
                    pl.col('File Name').n_unique().alias('Matches'),
                    pl.col('Batted').sum().alias('Inns'),
                    pl.col('Out').sum().alias('Out'),
                    pl.col('Not Out').sum().alias('Not Out'),
                    pl.col('Balls').sum().alias('Balls'),
                    pl.col('Runs').sum().alias('Runs'),
                    pl.col('Runs').max().alias('HS'),
                    pl.col('4s').sum().alias('4s'),
                    pl.col('6s').sum().alias('6s'),
                    pl.col('50s').sum().alias('50s'),
                    pl.col('100s').sum().alias('100s'),
                    pl.col('200s').sum().alias('200s'),
                    pl.col('<25&Out').sum().alias('<25&Out'),
                    pl.col('Caught').sum().alias('Caught'),
                    pl.col('Bowled').sum().alias('Bowled'),
                    pl.col('LBW').sum().alias('LBW'),
                    pl.col('Run Out').sum().alias('Run Out'),
                    pl.col('Stumped').sum().alias('Stumped'),
                    pl.col('Total_Runs').sum().alias('Team Runs'),
                    pl.col('Overs').sum().alias('Overs'),
                    pl.col('Wickets').sum().alias('Wickets'),
                    pl.col('Team Balls').sum().alias('Team Balls'),
                ])
            )
            pl_innings_agg = pl_innings_agg.with_columns([
                (pl.col('Runs') / pl.col('Out')).round(2).fill_null(0).alias('Avg'),
                ((pl.col('Runs') / pl.col('Balls')) * 100).round(2).fill_null(0).alias('SR'),
                (pl.col('Balls') / pl.col('Out')).round(2).fill_null(0).alias('BPO'),
                (pl.col('Team Runs') / pl.col('Wickets')).round(2).fill_null(0).alias('Team Avg'),
                (pl.col('Team Runs') / pl.col('Team Balls') * 100).round(2).fill_null(0).alias('Team SR'),
                (pl.col('Avg') / pl.col('Team Avg') * 100).round(2).fill_null(0).alias('P+ Avg'),
                (pl.col('SR') / pl.col('Team SR') * 100).round(2).fill_null(0).alias('P+ SR'),
                (pl.col('Balls') / (pl.col('4s') + pl.col('6s')).replace(0, 1)).round(2).fill_null(0).alias('BPB'),
                (((pl.col('50s') + pl.col('100s')) / pl.col('Inns')) * 100).round(2).fill_null(0).alias('50+PI'),
                ((pl.col('100s') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('100PI'),
                ((pl.col('<25&Out') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('<25&OutPI'),
                ((pl.col('Caught') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Caught%'),
                ((pl.col('Bowled') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Bowled%'),
                ((pl.col('LBW') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('LBW%'),
                ((pl.col('Run Out') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Run Out%'),
                ((pl.col('Stumped') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Stumped%'),
                ((pl.col('Not Out') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Not Out%'),
            ])
            df_innings = pl_innings_agg.to_pandas()
            df_innings = df_innings.sort_values(by=["Runs"], ascending=[False])
            cache_dataframe(innings_cache_key, df_innings)
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fc5c7d 0%, #6a82fb 100%); 
                    padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                    box-shadow: 0 8px 32px rgba(252, 92, 125, 0.3);
                    border: 1px solid rgba(255, 255, 255, 0.2);">
            <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">🔢 Innings Record</h3>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df_innings, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

    # Position Stats Tab (Polars backend)
    with tabs[7]:
        position_cache_key = f"{cache_key}_position_stats"
        df_position = get_cached_dataframe(position_cache_key)
        if df_position is None:
            pl_position = pl.from_pandas(filtered_df)[[
                'Name', 'Position', 'File Name', 'Batted', 'Out', 'Not Out', 'Balls', 'Runs', '4s', '6s', '50s', '100s', '200s', '<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 'Total_Runs', 'Overs', 'Wickets', 'Team Balls'
            ]]
            pl_position_agg = (
                pl_position.groupby(['Name', 'Position']).agg([
                    pl.col('File Name').n_unique().alias('Matches'),
                    pl.col('Batted').sum().alias('Inns'),
                    pl.col('Out').sum().alias('Out'),
                    pl.col('Not Out').sum().alias('Not Out'),
                    pl.col('Balls').sum().alias('Balls'),
                    pl.col('Runs').sum().alias('Runs'),
                    pl.col('Runs').max().alias('HS'),
                    pl.col('4s').sum().alias('4s'),
                    pl.col('6s').sum().alias('6s'),
                    pl.col('50s').sum().alias('50s'),
                    pl.col('100s').sum().alias('100s'),
                    pl.col('200s').sum().alias('200s'),
                    pl.col('<25&Out').sum().alias('<25&Out'),
                    pl.col('Caught').sum().alias('Caught'),
                    pl.col('Bowled').sum().alias('Bowled'),
                    pl.col('LBW').sum().alias('LBW'),
                    pl.col('Run Out').sum().alias('Run Out'),
                    pl.col('Stumped').sum().alias('Stumped'),
                    pl.col('Total_Runs').sum().alias('Team Runs'),
                    pl.col('Overs').sum().alias('Overs'),
                    pl.col('Wickets').sum().alias('Wickets'),
                    pl.col('Team Balls').sum().alias('Team Balls'),
                ])
            )
            pl_position_agg = pl_position_agg.with_columns([
                (pl.col('Runs') / pl.col('Out')).round(2).fill_null(0).alias('Avg'),
                ((pl.col('Runs') / pl.col('Balls')) * 100).round(2).fill_null(0).alias('SR'),
                (pl.col('Balls') / pl.col('Out')).round(2).fill_null(0).alias('BPO'),
                (pl.col('Team Runs') / pl.col('Wickets')).round(2).fill_null(0).alias('Team Avg'),
                (pl.col('Team Runs') / pl.col('Team Balls') * 100).round(2).fill_null(0).alias('Team SR'),
                (pl.col('Avg') / pl.col('Team Avg') * 100).round(2).fill_null(0).alias('P+ Avg'),
                (pl.col('SR') / pl.col('Team SR') * 100).round(2).fill_null(0).alias('P+ SR'),
                (pl.col('Balls') / (pl.col('4s') + pl.col('6s')).replace(0, 1)).round(2).fill_null(0).alias('BPB'),
                (((pl.col('50s') + pl.col('100s')) / pl.col('Inns')) * 100).round(2).fill_null(0).alias('50+PI'),
                ((pl.col('100s') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('100PI'),
                ((pl.col('<25&Out') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('<25&OutPI'),
                ((pl.col('Caught') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Caught%'),
                ((pl.col('Bowled') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Bowled%'),
                ((pl.col('LBW') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('LBW%'),
                ((pl.col('Run Out') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Run Out%'),
                ((pl.col('Stumped') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Stumped%'),
                ((pl.col('Not Out') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Not Out%'),
            ])
            df_position = pl_position_agg.to_pandas()
            df_position = df_position.sort_values(by=["Runs"], ascending=[False])
            cache_dataframe(position_cache_key, df_position)
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); 
                    padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                    box-shadow: 0 8px 32px rgba(247, 151, 30, 0.3);
                    border: 1px solid rgba(255, 255, 255, 0.2);">
            <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">#️⃣ Position Record</h3>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df_position, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

    # Home/Away Stats Tab (Polars backend)
    with tabs[8]:
        homeaway_cache_key = f"{cache_key}_homeaway_stats"
        df_homeaway = get_cached_dataframe(homeaway_cache_key)
        if df_homeaway is None:
            pl_homeaway = pl.from_pandas(filtered_df)[[
                'Name', 'HomeOrAway', 'File Name', 'Batted', 'Out', 'Not Out', 'Balls', 'Runs', '4s', '6s', '50s', '100s', '200s', '<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 'Total_Runs', 'Overs', 'Wickets', 'Team Balls'
            ]]
            pl_homeaway_agg = (
                pl_homeaway.groupby(['Name', 'HomeOrAway']).agg([
                    pl.col('File Name').n_unique().alias('Matches'),
                    pl.col('Batted').sum().alias('Inns'),
                    pl.col('Out').sum().alias('Out'),
                    pl.col('Not Out').sum().alias('Not Out'),
                    pl.col('Balls').sum().alias('Balls'),
                    pl.col('Runs').sum().alias('Runs'),
                    pl.col('Runs').max().alias('HS'),
                    pl.col('4s').sum().alias('4s'),
                    pl.col('6s').sum().alias('6s'),
                    pl.col('50s').sum().alias('50s'),
                    pl.col('100s').sum().alias('100s'),
                    pl.col('200s').sum().alias('200s'),
                    pl.col('<25&Out').sum().alias('<25&Out'),
                    pl.col('Caught').sum().alias('Caught'),
                    pl.col('Bowled').sum().alias('Bowled'),
                    pl.col('LBW').sum().alias('LBW'),
                    pl.col('Run Out').sum().alias('Run Out'),
                    pl.col('Stumped').sum().alias('Stumped'),
                    pl.col('Total_Runs').sum().alias('Team Runs'),
                    pl.col('Overs').sum().alias('Overs'),
                    pl.col('Wickets').sum().alias('Wickets'),
                    pl.col('Team Balls').sum().alias('Team Balls'),
                ])
            )
            pl_homeaway_agg = pl_homeaway_agg.with_columns([
                (pl.col('Runs') / pl.col('Out')).round(2).fill_null(0).alias('Avg'),
                ((pl.col('Runs') / pl.col('Balls')) * 100).round(2).fill_null(0).alias('SR'),
                (pl.col('Balls') / pl.col('Out')).round(2).fill_null(0).alias('BPO'),
                (pl.col('Team Runs') / pl.col('Wickets')).round(2).fill_null(0).alias('Team Avg'),
                (pl.col('Team Runs') / pl.col('Team Balls') * 100).round(2).fill_null(0).alias('Team SR'),
                (pl.col('Avg') / pl.col('Team Avg') * 100).round(2).fill_null(0).alias('P+ Avg'),
                (pl.col('SR') / pl.col('Team SR') * 100).round(2).fill_null(0).alias('P+ SR'),
                (pl.col('Balls') / (pl.col('4s') + pl.col('6s')).replace(0, 1)).round(2).fill_null(0).alias('BPB'),
                (((pl.col('50s') + pl.col('100s')) / pl.col('Inns')) * 100).round(2).fill_null(0).alias('50+PI'),
                ((pl.col('100s') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('100PI'),
                ((pl.col('<25&Out') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('<25&OutPI'),
                ((pl.col('Caught') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Caught%'),
                ((pl.col('Bowled') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Bowled%'),
                ((pl.col('LBW') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('LBW%'),
                ((pl.col('Run Out') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Run Out%'),
                ((pl.col('Stumped') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Stumped%'),
                ((pl.col('Not Out') / pl.col('Inns')) * 100).round(2).fill_null(0).alias('Not Out%'),
            ])
            df_homeaway = pl_homeaway_agg.to_pandas()
            df_homeaway = df_homeaway.sort_values(by=["Runs"], ascending=[False])
            cache_dataframe(homeaway_cache_key, df_homeaway)
        st.markdown("""
<div style="background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); padding: 1rem; margin: 1rem 0; border-radius: 15px; box-shadow: 0 8px 32px rgba(247, 151, 30, 0.3); border: 1px solid rgba(255, 255, 255, 0.2);"><h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">🏠 Home/Away Record</h3></div>""", unsafe_allow_html=True)
        st.dataframe(df_opponent, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})
        st.markdown("""
<div style="background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); padding: 1rem; margin: 1rem 0; border-radius: 15px; box-shadow: 0 8px 32px rgba(67, 206, 162, 0.3); border: 1px solid rgba(255, 255, 255, 0.2);"><h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">📈 Cumulative Record</h3></div>""", unsafe_allow_html=True)
        st.dataframe(df_cumulative, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

    # Block Stats Tab (Polars backend)
    with tabs[10]:
        block_cache_key = f"{cache_key}_block_stats"
        df_block = get_cached_dataframe(block_cache_key)
        if df_block is None:
            pl_block = pl.from_pandas(filtered_df)
            # Example: block by 5-match intervals (rolling window)
            pl_block = pl_block.sort(['Name', 'Date'])
            pl_block = pl_block.with_columns([
                (pl.col('Runs').rolling_sum(window_size=5).over('Name').alias('Block Runs')),
                (pl.col('Balls').rolling_sum(window_size=5).over('Name').alias('Block Balls')),
                pl.col('Date'),
                pl.col('Name'),
            ])
            df_block = pl_block.select(['Name', 'Date', 'Block Runs', 'Block Balls']).to_pandas()
            cache_dataframe(block_cache_key, df_block)
        st.markdown('<div style="background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); padding: 1rem; margin: 1rem 0; border-radius: 15px; box-shadow: 0 8px 32px rgba(247, 151, 30, 0.3); border: 1px solid rgba(255, 255, 255, 0.2);"><h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">🧱 Block Record (5-match rolling)</h3></div>', unsafe_allow_html=True)
        # Ensure numeric columns are correct dtype for Arrow compatibility
        for col in ['Block Runs', 'Block Balls']:
            if col in df_block.columns:
                df_block[col] = pd.to_numeric(df_block[col], errors='coerce')
        st.dataframe(df_block, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})
