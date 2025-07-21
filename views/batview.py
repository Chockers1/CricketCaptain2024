import streamlit as st
import pandas as pd
import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import plotly.graph_objects as go
import redis
import json
import pickle
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
        if not REDIS_AVAILABLE:
            return None
        try:
            return func(*args, **kwargs)
        except:
            return None
    return wrapper

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

        # --- Career Tab: Polars end-to-end ---
        pl_career = pl_df.select([
            'Name', 'File Name', 'Batted', 'Out', 'Not Out', 'Balls', 'Runs', '4s', '6s', '50s', '100s', '150s', '200s'
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
            st.markdown("""
            <div style="background: linear-gradient(135deg, #36d1dc 0%, #5b86e5 100%); 
                        padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                        box-shadow: 0 6px 24px rgba(54, 209, 220, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">📊 Batting Average vs Strike Rate Analysis</h3>
            </div>
            """, unsafe_allow_html=True)
            # Show first plot
            st.plotly_chart(scatter_fig, use_container_width=True)

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
            st.plotly_chart(sr_bpo_fig, use_container_width=True)


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
