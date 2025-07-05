import streamlit as st
import pandas as pd
import numpy as np
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

# Global flag to track Redis availability
REDIS_AVAILABLE = False

# Try to initialize Redis connection
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
    """Cache a DataFrame using Redis"""
    df_bytes = pickle.dumps(df)
    redis_client.set(key, df_bytes, ex=int(expiry.total_seconds()))

@handle_redis_errors
def get_cached_dataframe(key):
    """Retrieve a cached DataFrame from Redis"""
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

def display_bat_view():
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

        # Try to get filtered data from cache if Redis is available
        filtered_df = get_cached_dataframe(cache_key) if REDIS_AVAILABLE else None
        
        if filtered_df is None:
            # If not in cache or Redis is not available, apply filters
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

            # Cache the filtered DataFrame if Redis is available
            if REDIS_AVAILABLE:
                cache_dataframe(cache_key, filtered_df)

        # Create a placeholder for tabs that will be lazily loaded
        main_container = st.container()
        
        # Create tabs for different views - Added "Win/Loss record" tab
        tabs = main_container.tabs([
            "Career", "Format", "Season", "Latest", "Opponent", 
            "Location", "Innings", "Position", "Home/Away",
            "Cumulative", "Block", "Distributions", "Percentile",
            "Records", "Win/Loss"
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
                file_stats['Match_Avg'] = file_stats['Runs'] / file_stats['Wickets'].replace(0, np.inf)
                file_stats['Match_SR'] = (file_stats['Runs'] / file_stats['Team Balls'].replace(0, np.inf)) * 100

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

        # Format Stats Tab  
        with tabs[1]:
            format_cache_key = f"{cache_key}_format_stats"
            df_format = get_cached_dataframe(format_cache_key)
            
            if df_format is None:
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
                df_format['Avg'] = (df_format['Runs'] / df_format['Out']).round(2).fillna(0)
                df_format['SR'] = ((df_format['Runs'] / df_format['Balls']) * 100).round(2).fillna(0)
                df_format['BPO'] = (df_format['Balls'] / df_format['Out']).round(2).fillna(0)

                # Calculate new columns for team statistics
                df_format['Team Avg'] = (df_format['Team Runs'] / df_format['Wickets']).round(2).fillna(0)
                df_format['Team SR'] = (df_format['Team Runs'] / df_format['Team Balls'] * 100).round(2).fillna(0)

                # Calculate P+ Avg and P+ SR
                df_format['P+ Avg'] = (df_format['Avg'] / df_format['Team Avg'] * 100).round(2).fillna(0)
                df_format['P+ SR'] = (df_format['SR'] / df_format['Team SR'] * 100).round(2).fillna(0)

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
                
                # Cache the computed format statistics
                cache_dataframe(format_cache_key, df_format)
                
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
                    xanchor="center",
                    x=0.5
                )
            )
            
            # Update axes
            fig.update_xaxes(title_text="Year", showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)', 
                             tickmode='linear', dtick=1)  # Ensure only whole years are displayed
            fig.update_yaxes(title_text="Average", showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)', row=1, col=1)
            fig.update_yaxes(title_text="Strike Rate", showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)', row=1, col=2)
            
            # Display the figure
            st.plotly_chart(fig, use_container_width=True)

        # Season Stats Tab
        with tabs[2]:
            # Cache key for season statistics
            season_cache_key = f"{cache_key}_season_stats"
            season_stats_df = get_cached_dataframe(season_cache_key)
            
            if season_stats_df is None:
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
                season_stats_df['Avg'] = (season_stats_df['Runs'] / season_stats_df['Out']).round(2).fillna(0)
                season_stats_df['SR'] = ((season_stats_df['Runs'] / season_stats_df['Balls']) * 100).round(2).fillna(0)
                season_stats_df['BPO'] = (season_stats_df['Balls'] / season_stats_df['Out']).round(2).fillna(0)

                # Calculate new columns for team statistics
                season_stats_df['Team Avg'] = (season_stats_df['Team Runs'] / season_stats_df['Wickets']).round(2).fillna(0)
                season_stats_df['Team SR'] = (season_stats_df['Team Runs'] / season_stats_df['Team Balls'] * 100).round(2).fillna(0)

                # Calculate P+ Avg and P+ SR
                season_stats_df['P+ Avg'] = (season_stats_df['Avg'] / season_stats_df['Team Avg'] * 100).round(2).fillna(0)
                season_stats_df['P+ SR'] = (season_stats_df['SR'] / season_stats_df['Team SR'] * 100).round(2).fillna(0)

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
                
                # Cache the computed season statistics
                cache_dataframe(season_cache_key, season_stats_df)

            # Display the Season Stats
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
            # Cache key for latest innings statistics
            latest_inns_cache_key = f"{cache_key}_latest_innings"
            latest_inns_df = get_cached_dataframe(latest_inns_cache_key)

            if latest_inns_df is None:
                # Create the latest_inns_df by grouping by 'Name', 'Match_Format', 'Date', and 'Innings'
                latest_inns_df = filtered_df.groupby(['Name', 'Match_Format', 'Date', 'Innings']).agg({
                    'Bat_Team_y': 'first',
                    'Bowl_Team_y': 'first',
                    'How Out': 'first',    
                    'Balls': 'sum',        
                    'Runs': ['sum'],       
                    '4s': 'sum',           
                    '6s': 'sum',           
                }).reset_index()

                # Flatten multi-level columns
                latest_inns_df.columns = [
                    'Name', 'Match_Format', 'Date', 'Innings', 'Bat Team', 'Bowl Team', 
                    'How Out', 'Balls', 'Runs', '4s', '6s'
                ]

                # Convert Date to datetime for proper sorting - using a more flexible format
                latest_inns_df['Date'] = pd.to_datetime(latest_inns_df['Date'], format='%d %b %Y')

                # Sort by Date in descending order (newest to oldest)
                latest_inns_df = latest_inns_df.sort_values(by='Date', ascending=False).head(20)

                # Convert Date format to 'dd/mm/yyyy' for display
                latest_inns_df['Date'] = latest_inns_df['Date'].dt.strftime('%d/%m/%Y')

                # Reorder columns to put Runs before Balls
                latest_inns_df = latest_inns_df[['Name', 'Match_Format', 'Date', 'Innings', 'Bat Team', 'Bowl Team', 
                                            'How Out', 'Runs', 'Balls', '4s', '6s']]
                
                # Cache the latest innings data
                cache_dataframe(latest_inns_cache_key, latest_inns_df)

            # Calculate the 'Out' column based on 'How Out'
            latest_inns_df['Out'] = latest_inns_df['How Out'].apply(lambda x: 1 if x not in ['not out', 'did not bat', ''] else 0)

            # Calculate summary statistics for the last 20 innings
            last_20_stats = latest_inns_df.agg({
                'Runs': 'sum',
                'Balls': 'sum',
                '4s': 'sum',
                '6s': 'sum',
                'Innings': 'count',
                'Out': 'sum'
            }).to_dict()

            last_20_stats['Matches'] = latest_inns_df['Date'].nunique()
            last_20_stats['50s'] = latest_inns_df[(latest_inns_df['Runs'] >= 50) & (latest_inns_df['Runs'] <= 99)].shape[0]
            last_20_stats['100s'] = latest_inns_df[latest_inns_df['Runs'] >= 100].shape[0]
            last_20_stats['Average'] = last_20_stats['Runs'] / last_20_stats['Out'] if last_20_stats['Out'] > 0 else 0
            last_20_stats['Strike Rate'] = (last_20_stats['Runs'] / last_20_stats['Balls']) * 100 if last_20_stats['Balls'] > 0 else 0
            last_20_stats['Balls Per Out'] = last_20_stats['Balls'] / last_20_stats['Out'] if last_20_stats['Out'] > 0 else 0

            # Display summary cards
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">⚡ Last 20 Innings</h3>
            </div>
            """, unsafe_allow_html=True)
            col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)


            with col1:
                st.metric("Matches", last_20_stats['Matches'], border=True)
            with col2:
                st.metric("Innings", last_20_stats['Innings'], border=True)
            with col3:
                st.metric("Outs", last_20_stats['Out'], border=True)
            with col4:
                st.metric("Runs", last_20_stats['Runs'], border=True)
            with col5:
                st.metric("Balls", last_20_stats['Balls'], border=True)
            with col6:
                st.metric("50s", last_20_stats['50s'], border=True)
            with col7:
                st.metric("100s", last_20_stats['100s'], border=True)
            with col8:
                st.metric("Average", f"{last_20_stats['Average']:.2f}", border=True)
            with col9:
                st.metric("Strike Rate", f"{last_20_stats['Strike Rate']:.2f}", border=True)

            # Function to apply background color based on Runs
            def color_runs(value):
                if value <= 20:
                    return 'background-color: #DE6A73'  # Light Red
                elif 21 <= value <= 49:
                    return 'background-color: #DEAD68'  # Light Yellow
                elif 50 <= value < 100:
                    return 'background-color: #6977DE'  # Light Blue
                elif value >= 100:
                    return 'background-color: #69DE85'  # Light Green
                return ''  # Default (no background color)

            # Apply conditional formatting to the 'Runs' column
            styled_df = latest_inns_df.style.applymap(color_runs, subset=['Runs'])

            # Display the dataframe
            st.dataframe(styled_df, height=735, use_container_width=True, hide_index=True)

            # Ensure 'Date' column is in datetime format
            filtered_df['Date'] = pd.to_datetime(filtered_df['Date'], format='%d %b %Y')

            # Calculate the last 20 innings for each player
            last_20_innings_all_players = filtered_df.groupby('Name').apply(lambda x: x.nlargest(20, 'Date')).reset_index(drop=True)

            # Display a dataframe with Name, Match_Format, and calculated metrics
            metrics_df = last_20_innings_all_players.groupby(['Name', 'Match_Format']).agg({
                'Date': 'nunique',
                'Innings': 'count',
                'Out': 'sum',
                'Runs': 'sum',
                'Balls': 'sum'
            }).reset_index()

            # Now calculate '50s' and '100s' after the aggregation
            metrics_df['50s'] = last_20_innings_all_players.groupby(['Name', 'Match_Format'])['Runs'].apply(lambda x: ((x >= 50) & (x <= 99)).sum()).reset_index(drop=True)
            metrics_df['100s'] = last_20_innings_all_players.groupby(['Name', 'Match_Format'])['Runs'].apply(lambda x: (x >= 100).sum()).reset_index(drop=True)

            # Calculate Average and Strike Rate
            metrics_df['Average'] = metrics_df['Runs'] / metrics_df['Out']
            metrics_df['Strike Rate'] = (metrics_df['Runs'] / metrics_df['Balls']) * 100

            # Rename columns for clarity
            metrics_df.columns = ['Name', 'Match_Format', 'Matches', 'Innings', 'Outs', 'Runs', 'Balls', '50s', '100s', 'Average', 'Strike Rate']

            # Display the dataframe
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                        box-shadow: 0 6px 24px rgba(250, 112, 154, 0.25);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">📊 Summary Metrics</h3>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        # Opponent Stats Tab  
        with tabs[4]:
            # Cache key for opponents statistics
            opponents_cache_key = f"{cache_key}_opponents_stats"
            opponents_stats_df = get_cached_dataframe(opponents_cache_key)

            if opponents_stats_df is None:
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

                # Cache the opponents statistics data
                cache_dataframe(opponents_cache_key, opponents_stats_df)
            
            # Display the opponents statistics dataframe - this was missing
            st.markdown("""
            <div style="background: linear-gradient(135deg, #a8caba 0%, #5d4e75 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(168, 202, 186, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">🏆 Opponent Statistics</h3>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(opponents_stats_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            # Calculate average runs against opponents with caching
            opponent_avg_cache_key = f"{cache_key}_opponent_averages"
            opponent_stats_df = get_cached_dataframe(opponent_avg_cache_key)

            if opponent_stats_df is None:
                opponent_stats_df = filtered_df.groupby('Bowl_Team_y').agg({
                    'Runs': 'sum',
                    'Out': 'sum',
                    'Balls': 'sum'
                }).reset_index()

                # Calculate Average
                opponent_stats_df['Avg'] = (opponent_stats_df['Runs'] / opponent_stats_df['Out']).round(2)

                # Cache the opponent averages data
                cache_dataframe(opponent_avg_cache_key, opponent_stats_df)

            # Create a bar chart for Average runs against opponents
            bar_chart_cache_key = f"{cache_key}_opponent_bar_chart"
            fig = get_cached_dataframe(bar_chart_cache_key)

            if fig is None:
                fig = go.Figure()

                # Add Average runs trace
                fig.add_trace(
                    go.Bar(x=opponent_stats_df['Bowl_Team_y'], 
                          y=opponent_stats_df['Avg'], 
                          name='Average', 
                          marker_color='#f84e4e')
                )

                # Calculate the appropriate average based on selection
                if 'All' in name_choice and len(name_choice) == 1:
                    # Calculate overall average across all data
                    overall_avg = opponent_stats_df['Avg'].mean()
                else:
                    # Use individual player's average from bat_career_df
                    overall_avg = bat_career_df['Avg'].iloc[0]

                # Add horizontal line for average
                fig.add_trace(
                    go.Scatter(
                        x=[opponent_stats_df['Bowl_Team_y'].iloc[0], opponent_stats_df['Bowl_Team_y'].iloc[-1]],
                        y=[overall_avg, overall_avg],
                        mode='lines+text',
                        name='Average',
                        line=dict(color='black', width=2, dash='dash'),
                        text=[f"{'Overall' if 'All' in name_choice and len(name_choice) == 1 else 'Career'} Average: {overall_avg:.2f}", ''],
                        textposition='top center',
                        showlegend=False
                    )
                )

                # Add markdown title
                st.markdown("""
                <div style="background: linear-gradient(135deg, #a8caba 0%, #5d4e75 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(168, 202, 186, 0.25);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h2 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.4rem; text-align: center;">📈 Average Runs Against Opponents</h2>
                </div>
                """, unsafe_allow_html=True)

                # Update layout
                fig.update_layout(
                    height=500,
                    xaxis_title='Opponent',
                    yaxis_title='Average',
                    margin=dict(l=50, r=50, t=70, b=50),
                    font=dict(size=12),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )

                # Cache the bar chart
                cache_dataframe(bar_chart_cache_key, fig)

            # Display the bar chart
            st.plotly_chart(fig)

        # Location Stats Tab
        with tabs[5]:
            # Cache key for location statistics
            location_cache_key = f"{cache_key}_location_stats"
            opponents_stats_df = get_cached_dataframe(location_cache_key)

            if opponents_stats_df is None:
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
                opponents_stats_df['Avg'] = (opponents_stats_df['Runs'] / opponents_stats_df['Out']).round(2).fillna(0)
                opponents_stats_df['SR'] = ((opponents_stats_df['Runs'] / opponents_stats_df['Balls']) * 100).round(2).fillna(0)
                opponents_stats_df['BPO'] = (opponents_stats_df['Balls'] / opponents_stats_df['Out']).round(2).fillna(0)

                # Calculate new columns for team statistics
                opponents_stats_df['Team Avg'] = (opponents_stats_df['Team Runs'] / opponents_stats_df['Wickets']).round(2).fillna(0)
                opponents_stats_df['Team SR'] = (opponents_stats_df['Team Runs'] / opponents_stats_df['Team Balls'] * 100).round(2).fillna(0)

                # Calculate P+ Avg and P+ SR
                opponents_stats_df['P+ Avg'] = (opponents_stats_df['Avg'] / opponents_stats_df['Team Avg'] * 100).round(2).fillna(0)
                opponents_stats_df['P+ SR'] = (opponents_stats_df['SR'] / opponents_stats_df['Team SR'] * 100).round(2).fillna(0)

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

                # Cache the computed location statistics
                cache_dataframe(location_cache_key, opponents_stats_df)

            st.markdown("""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">📍 Location Statistics</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Display the location dataframe first (full width)
            st.dataframe(opponents_stats_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            # Cache key for location averages
            location_avg_cache_key = f"{cache_key}_location_averages"
            opponent_stats_df = get_cached_dataframe(location_avg_cache_key)

            if opponent_stats_df is None:
                # Calculate average runs against locations
                opponent_stats_df = filtered_df.groupby('Home Team').agg({
                    'Runs': 'sum',
                    'Out': 'sum',
                    'Balls': 'sum'
                }).reset_index()

                # Calculate Average
                opponent_stats_df['Avg'] = (opponent_stats_df['Runs'] / opponent_stats_df['Out']).round(2)

                # Cache the location averages
                cache_dataframe(location_avg_cache_key, opponent_stats_df)

            # Sort by average in descending order
            opponent_stats_df = opponent_stats_df.sort_values(by='Avg', ascending=True)

            # Create a bar chart for Average runs by location
            location_chart_cache_key = f"{cache_key}_location_chart"
            fig = get_cached_dataframe(location_chart_cache_key)

            if fig is None:
                fig = go.Figure()

                # Add Average runs trace
                fig.add_trace(
                    go.Bar(
                        y=opponent_stats_df['Home Team'], 
                        x=opponent_stats_df['Avg'], 
                        name='Average', 
                        marker_color='#f84e4e',
                        orientation='h'
                    )
                )

                # Calculate the appropriate average based on selection
                if 'All' in name_choice and len(name_choice) == 1:
                    # Calculate overall average across all data
                    overall_avg = opponent_stats_df['Avg'].mean()
                else:
                    # Use individual player's average from bat_career_df
                    overall_avg = bat_career_df['Avg'].iloc[0]

                # Add horizontal line for average
                fig.add_trace(
                    go.Scatter(
                        y=[opponent_stats_df['Home Team'].iloc[0], opponent_stats_df['Home Team'].iloc[-1]],
                        x=[overall_avg, overall_avg],
                        mode='lines+text',
                        name='Average',
                        line=dict(color='black', width=2, dash='dash'),
                        text=[f"{'Overall' if 'All' in name_choice and len(name_choice) == 1 else 'Career'} Average: {overall_avg:.2f}", ''],
                        textposition='top center',
                        showlegend=False
                    )
                )

                # Update layout
                fig.update_layout(
                    height=500,
                    yaxis_title='Location',
                    xaxis_title='Average',
                    margin=dict(l=50, r=50, t=0, b=50),  # Adjust top margin to align with table
                    font=dict(size=12),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )

                # Cache the location chart
                cache_dataframe(location_chart_cache_key, fig)

            # Display chart title
            st.markdown("""
            <div style="background: linear-gradient(135deg, #4776e6 0%, #8e54e9 100%); 
                        padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                        box-shadow: 0 6px 24px rgba(71, 118, 230, 0.25);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">📍 Average Runs by Location</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Display the bar chart (full width)
            st.plotly_chart(fig, use_container_width=True)

        # Innings Stats Tab
        with tabs[6]:
            # Cache key for innings statistics
            innings_cache_key = f"{cache_key}_innings_stats"
            innings_stats_df = get_cached_dataframe(innings_cache_key)

            if innings_stats_df is None:
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
                innings_stats_df['Avg'] = (innings_stats_df['Runs'] / innings_stats_df['Out']).round(2).fillna(0)
                innings_stats_df['SR'] = ((innings_stats_df['Runs'] / innings_stats_df['Balls']) * 100).round(2).fillna(0)
                innings_stats_df['BPO'] = (innings_stats_df['Balls'] / innings_stats_df['Out']).round(2).fillna(0)

                # Calculate new columns for team statistics
                innings_stats_df['Team Avg'] = (innings_stats_df['Team Runs'] / innings_stats_df['Wickets']).round(2).fillna(0)
                innings_stats_df['Team SR'] = (innings_stats_df['Team Runs'] / innings_stats_df['Team Balls'] * 100).round(2).fillna(0)

                # Calculate P+ Avg and P+ SR
                innings_stats_df['P+ Avg'] = (innings_stats_df['Avg'] / innings_stats_df['Team Avg'] * 100).round(2).fillna(0)
                innings_stats_df['P+ SR'] = (innings_stats_df['SR'] / innings_stats_df['Team SR'] * 100).round(2).fillna(0)

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

                # Cache the computed innings statistics
                cache_dataframe(innings_cache_key, innings_stats_df)

            # Display the Innings Stats header
            st.markdown("""
            <div style="background: linear-gradient(135deg, #36d1dc 0%, #5b86e5 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(54, 209, 220, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">🎯 Innings Statistics</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Display the full dataframe first
            st.dataframe(innings_stats_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            # Cache key for average runs across innings
            innings_avg_cache_key = f"{cache_key}_innings_averages"
            average_runs_innings_df = get_cached_dataframe(innings_avg_cache_key)

            if average_runs_innings_df is None:
                # Calculate average runs across innings
                average_runs_innings_df = filtered_df.groupby('Innings').agg({
                    'Runs': 'sum',
                    'Out': 'sum',
                    'Balls': 'sum'
                }).reset_index()

                # Calculate Average
                average_runs_innings_df['Avg'] = (average_runs_innings_df['Runs'] / average_runs_innings_df['Out']).round(2)

                # Cache the innings averages
                cache_dataframe(innings_avg_cache_key, average_runs_innings_df)

            # Create a bar chart for Average runs per innings
            innings_chart_cache_key = f"{cache_key}_innings_chart"
            fig = get_cached_dataframe(innings_chart_cache_key)

            if fig is None:
                fig = go.Figure()

                # Add Average runs trace
                fig.add_trace(
                    go.Bar(
                        x=average_runs_innings_df['Innings'], 
                        y=average_runs_innings_df['Avg'], 
                        name='Average', 
                        marker_color='#f84e4e'
                    )
                )

                # Calculate the appropriate average based on selection
                if 'All' in name_choice and len(name_choice) == 1:
                    # Calculate overall average across all innings
                    overall_avg = average_runs_innings_df['Avg'].mean()
                else:
                    # Use individual player's average from bat_career_df
                    overall_avg = bat_career_df['Avg'].iloc[0]

                # Add horizontal line for average
                fig.add_trace(
                    go.Scatter(
                        x=[1, max(average_runs_innings_df['Innings'])],
                        y=[overall_avg, overall_avg],
                        mode='lines+text',
                        name='Average',
                        line=dict(color='black', width=2, dash='dash'),
                        text=[f"{'Overall' if 'All' in name_choice and len(name_choice) == 1 else 'Career'} Average: {overall_avg:.2f}", ''],
                        textposition='top center',
                        showlegend=False
                    )
                )

                # Update layout
                fig.update_layout(
                    height=500,
                    xaxis_title='Innings',
                    yaxis_title='Average',
                    margin=dict(l=50, r=50, t=0, b=50),  # Adjust top margin to align with table
                    font=dict(size=12),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )

                # Update y-axis to show intervals of 5
                fig.update_yaxes(
                    tickmode='linear',
                    dtick=5
                )

                # Update x-axis to show all bar positions
                fig.update_xaxes(
                    tickmode='linear',
                    tick0=0,
                    dtick=1,
                )

                # Cache the innings chart
                cache_dataframe(innings_chart_cache_key, fig)

            # Display chart title
            st.markdown("""
            <div style="background: linear-gradient(135deg, #36d1dc 0%, #5b86e5 100%); 
                        padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                        box-shadow: 0 6px 24px rgba(54, 209, 220, 0.25);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">🎯 Average Runs by Innings Number</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Display the bar chart (full width)
            st.plotly_chart(fig, use_container_width=True)
        
        # Position Stats Tab
        with tabs[7]:
            # Cache key for position statistics
            position_cache_key = f"{cache_key}_position_stats"
            position_stats_df = get_cached_dataframe(position_cache_key)

            if position_stats_df is None:
                #Calculate opponents statistics by Position
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
                position_stats_df['Avg'] = (position_stats_df['Runs'] / position_stats_df['Out']).round(2).fillna(0)
                position_stats_df['SR'] = ((position_stats_df['Runs'] / position_stats_df['Balls']) * 100).round(2).fillna(0)
                position_stats_df['BPO'] = (position_stats_df['Balls'] / position_stats_df['Out']).round(2).fillna(0)

                # Calculate new columns for team statistics
                position_stats_df['Team Avg'] = (position_stats_df['Team Runs'] / position_stats_df['Wickets']).round(2).fillna(0)
                position_stats_df['Team SR'] = (position_stats_df['Team Runs'] / position_stats_df['Team Balls'] * 100).round(2).fillna(0)

                # Calculate P+ Avg and P+ SR
                position_stats_df['P+ Avg'] = (position_stats_df['Avg'] / position_stats_df['Team Avg'] * 100).round(2).fillna(0)
                position_stats_df['P+ SR'] = (position_stats_df['SR'] / position_stats_df['Team SR'] * 100).round(2).fillna(0)

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
                
                # Cache the computed position statistics
                cache_dataframe(position_cache_key, position_stats_df)

            # Display the Position Stats header
            st.markdown("""
            <div style="background: linear-gradient(135deg, #8360c3 0%, #2ebf91 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(131, 96, 195, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">📍 Position Statistics</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Display the full dataframe first
            st.dataframe(position_stats_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            # Cache key for position averages
            position_avg_cache_key = f"{cache_key}_position_averages"
            position_avg_stats_df = get_cached_dataframe(position_avg_cache_key)

            if position_avg_stats_df is None:
                # Calculate average runs by position
                position_avg_stats_df = filtered_df.groupby('Position').agg({
                    'Runs': 'sum',
                    'Out': 'sum',
                    'Balls': 'sum'
                }).reset_index()

                # Calculate Average
                position_avg_stats_df['Avg'] = (position_avg_stats_df['Runs'] / position_avg_stats_df['Out']).round(2)

                # Ensure we have all positions 1-11
                all_positions = pd.DataFrame({'Position': range(1, 12)})
                position_avg_stats_df = pd.merge(all_positions, position_avg_stats_df, on='Position', how='left')
                position_avg_stats_df['Avg'] = position_avg_stats_df['Avg'].fillna(0)
                
                # Cache the position averages
                cache_dataframe(position_avg_cache_key, position_avg_stats_df)

            # Create a bar chart for Average runs by position
            position_chart_cache_key = f"{cache_key}_position_chart"
            fig = get_cached_dataframe(position_chart_cache_key)

            if fig is None:
                fig = go.Figure()

                # Add Average runs trace
                fig.add_trace(
                    go.Bar(
                        y=position_avg_stats_df['Position'], 
                        x=position_avg_stats_df['Avg'], 
                        name='Average', 
                        marker_color='#f84e4e',
                        orientation='h'
                    )
                )

                # Calculate the appropriate average based on selection
                if 'All' in name_choice and len(name_choice) == 1:
                    # Calculate overall average across all positions
                    overall_avg = position_avg_stats_df[position_avg_stats_df['Avg'] > 0]['Avg'].mean()
                else:
                    # Use individual player's average from bat_career_df
                    overall_avg = bat_career_df['Avg'].iloc[0]

                # Add horizontal line for average
                fig.add_trace(
                    go.Scatter(
                        y=[1, 11],  # Show only positions 1-11
                        x=[overall_avg, overall_avg],
                        mode='lines+text',
                        name='Average',
                        line=dict(color='black', width=2, dash='dash'),
                        text=[f"{'Overall' if 'All' in name_choice and len(name_choice) == 1 else 'Career'} Average: {overall_avg:.2f}", ''],
                        textposition='top center',
                        showlegend=False
                    )
                )

                # Update layout
                fig.update_layout(
                    height=500,
                    yaxis_title='Position',
                    xaxis_title='Average',
                    margin=dict(l=50, r=50, t=0, b=50),  # Adjust top margin to align with table
                    font=dict(size=12),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )

                # Update y-axis to show intervals of 1
                fig.update_yaxes(
                    tickmode='linear',
                    dtick=1,
                    range=[0.5, 11.5]  # Add padding on either side of the bars
                )

                # Update x-axis to show intervals of 5
                fig.update_xaxes(
                    tickmode='linear',
                    dtick=5,
                    range=[0, max(position_avg_stats_df['Avg']) + 5]  # Add some padding to the top
                )

                # Cache the position chart
                cache_dataframe(position_chart_cache_key, fig)

            # Display chart title
            st.markdown("""
            <div style="background: linear-gradient(135deg, #8360c3 0%, #2ebf91 100%); 
                        padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                        box-shadow: 0 6px 24px rgba(131, 96, 195, 0.25);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">📍 Average Runs by Batting Position</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Display the bar chart (full width)
            st.plotly_chart(fig, use_container_width=True)

        # Home/Away Stats Tab
        with tabs[8]:
            # Cache key for home/away statistics
            homeaway_cache_key = f"{cache_key}_homeaway_stats"
            homeaway_stats_df = get_cached_dataframe(homeaway_cache_key)

            if homeaway_stats_df is None:
                # Calculate statistics by HomeOrAway designation
                homeaway_stats_df = filtered_df.groupby(['Name', 'HomeOrAway']).agg({
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
                homeaway_stats_df.columns = ['Name', 'HomeOrAway', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 
                                          'Runs', 'HS', '4s', '6s', '50s', '100s', '200s', 
                                          '<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 
                                          'Team Runs', 'Overs', 'Wickets', 'Team Balls']
                
                # Create Career totals (aggregated across all locations)
                career_stats = filtered_df.groupby(['Name']).agg({
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
                
                # Flatten multi-level columns for career stats
                career_stats.columns = ['Name', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 
                                      'Runs', 'HS', '4s', '6s', '50s', '100s', '200s', 
                                      '<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 
                                      'Team Runs', 'Overs', 'Wickets', 'Team Balls']
                
                # Add "Career" as location identifier
                career_stats.insert(1, 'HomeOrAway', 'Career')
                
                # Concatenate career stats with home/away stats
                homeaway_stats_df = pd.concat([career_stats, homeaway_stats_df], ignore_index=True)

                # Calculate average runs per out, strike rate, and balls per out
                homeaway_stats_df['Avg'] = (homeaway_stats_df['Runs'] / homeaway_stats_df['Out']).round(2).fillna(0)
                homeaway_stats_df['SR'] = ((homeaway_stats_df['Runs'] / homeaway_stats_df['Balls']) * 100).round(2).fillna(0)
                homeaway_stats_df['BPO'] = (homeaway_stats_df['Balls'] / homeaway_stats_df['Out']).round(2).fillna(0)

                # Calculate new columns for team statistics
                homeaway_stats_df['Team Avg'] = (homeaway_stats_df['Team Runs'] / homeaway_stats_df['Wickets']).round(2).fillna(0)
                homeaway_stats_df['Team SR'] = (homeaway_stats_df['Team Runs'] / homeaway_stats_df['Team Balls'] * 100).round(2).fillna(0)

                # Calculate P+ Avg and P+ SR
                homeaway_stats_df['P+ Avg'] = (homeaway_stats_df['Avg'] / homeaway_stats_df['Team Avg'] * 100).round(2).fillna(0)
                homeaway_stats_df['P+ SR'] = (homeaway_stats_df['SR'] / homeaway_stats_df['Team SR'] * 100).round(2).fillna(0)

                # Calculate BPB (Balls Per Boundary)
                homeaway_stats_df['BPB'] = (homeaway_stats_df['Balls'] / (homeaway_stats_df['4s'] + homeaway_stats_df['6s']).replace(0, 1)).round(2)

                # Calculate new statistics
                homeaway_stats_df['50+PI'] = (((homeaway_stats_df['50s'] + homeaway_stats_df['100s']) / homeaway_stats_df['Inns']) * 100).round(2).fillna(0)
                homeaway_stats_df['100PI'] = ((homeaway_stats_df['100s'] / homeaway_stats_df['Inns']) * 100).round(2).fillna(0)
                homeaway_stats_df['<25&OutPI'] = ((homeaway_stats_df['<25&Out'] / homeaway_stats_df['Inns']) * 100).round(2).fillna(0)

                # Calculate dismissal percentages
                homeaway_stats_df['Caught%'] = ((homeaway_stats_df['Caught'] / homeaway_stats_df['Inns']) * 100).round(2).fillna(0)
                homeaway_stats_df['Bowled%'] = ((homeaway_stats_df['Bowled'] / homeaway_stats_df['Inns']) * 100).round(2).fillna(0)
                homeaway_stats_df['LBW%'] = ((homeaway_stats_df['LBW'] / homeaway_stats_df['Inns']) * 100).round(2).fillna(0)
                homeaway_stats_df['Run Out%'] = ((homeaway_stats_df['Run Out'] / homeaway_stats_df['Inns']) * 100).round(2).fillna(0)
                homeaway_stats_df['Stumped%'] = ((homeaway_stats_df['Stumped'] / homeaway_stats_df['Inns']) * 100).round(2).fillna(0)
                homeaway_stats_df['Not Out%'] = ((homeaway_stats_df['Not Out'] / homeaway_stats_df['Inns']) * 100).round(2).fillna(0)

                # Reorder columns
                homeaway_stats_df = homeaway_stats_df[['Name', 'HomeOrAway', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 'Runs', 'HS', 
                                                'Avg', 'BPO', 'SR', '4s', '6s', 'BPB', '<25&Out', '50s', '100s', 
                                                '<25&OutPI', '50+PI', '100PI', 'P+ Avg', 'P+ SR', 
                                                'Caught%', 'Bowled%', 'LBW%', 'Run Out%', 'Stumped%', 'Not Out%']]
                
                # Sort by Name and then put Career first, followed by Home, Away, and Neutral
                homeaway_stats_df['HomeOrAway_order'] = homeaway_stats_df['HomeOrAway'].map({'Career': 0, 'Home': 1, 'Away': 2, 'Neutral': 3})
                homeaway_stats_df = homeaway_stats_df.sort_values(by=['Name', 'HomeOrAway_order']).drop('HomeOrAway_order', axis=1)
                
                # Cache the computed home/away statistics
                cache_dataframe(homeaway_cache_key, homeaway_stats_df)

            # Display the Home/Away Stats header
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(255, 126, 95, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">🏠 Home/Away Statistics</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Display the full dataframe first
            st.dataframe(homeaway_stats_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            # Cache key for home/away averages
            homeaway_avg_cache_key = f"{cache_key}_homeaway_averages"
            homeaway_avg_stats_df = get_cached_dataframe(homeaway_avg_cache_key)

            if homeaway_avg_stats_df is None:
                # Calculate average runs by home/away
                homeaway_avg_stats_df = filtered_df.groupby('HomeOrAway').agg({
                    'Runs': 'sum',
                    'Out': 'sum',
                    'Balls': 'sum',
                    'File Name': 'nunique'
                }).reset_index()

                # Calculate averages and strike rates
                homeaway_avg_stats_df['Avg'] = (homeaway_avg_stats_df['Runs'] / homeaway_avg_stats_df['Out']).round(2)
                homeaway_avg_stats_df['SR'] = ((homeaway_avg_stats_df['Runs'] / homeaway_avg_stats_df['Balls']) * 100).round(2)
                homeaway_avg_stats_df['Matches'] = homeaway_avg_stats_df['File Name']
                
                # Cache the home/away averages
                cache_dataframe(homeaway_avg_cache_key, homeaway_avg_stats_df)

            # Create a bar chart for Average runs by home/away
            homeaway_chart_cache_key = f"{cache_key}_homeaway_chart"
            fig = get_cached_dataframe(homeaway_chart_cache_key)

            if fig is None:
                # Create subplots: one for average, one for strike rate
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Average by Home/Away", "Strike Rate by Home/Away"))

                # Add Average bars
                fig.add_trace(
                    go.Bar(
                        x=homeaway_avg_stats_df['HomeOrAway'], 
                        y=homeaway_avg_stats_df['Avg'], 
                        name='Average', 
                        marker_color='#f84e4e',
                        text=homeaway_avg_stats_df['Matches'].apply(lambda x: f"{x} matches"),
                        hovertemplate='%{x}<br>Average: %{y:.2f}<br>%{text}<extra></extra>'
                    ),
                    row=1, col=1
                )

                # Add Strike Rate bars
                fig.add_trace(
                    go.Bar(
                        x=homeaway_avg_stats_df['HomeOrAway'], 
                        y=homeaway_avg_stats_df['SR'], 
                        name='Strike Rate', 
                        marker_color='#4e84f8',
                        text=homeaway_avg_stats_df['Matches'].apply(lambda x: f"{x} matches"),
                        hovertemplate='%{x}<br>Strike Rate: %{y:.2f}<br>%{text}<extra></extra>'
                    ),
                    row=1, col=2
                )

                # Calculate the appropriate averages based on selection
                if 'All' in name_choice and len(name_choice) == 1:
                    # Calculate overall averages across all data
                    overall_avg = homeaway_avg_stats_df['Avg'].mean()
                    overall_sr = homeaway_avg_stats_df['SR'].mean()
                else:
                    # Use individual player's averages from bat_career_df
                    overall_avg = bat_career_df['Avg'].iloc[0]
                    overall_sr = bat_career_df['SR'].iloc[0]

                # Add horizontal line for average
                fig.add_trace(
                    go.Scatter(
                        x=homeaway_avg_stats_df['HomeOrAway'],
                        y=[overall_avg] * len(homeaway_avg_stats_df),
                        mode='lines',
                        name='Career Average',
                        line=dict(color='black', width=2, dash='dash'),
                        showlegend=True
                    ),
                    row=1, col=1
                )

                # Add horizontal line for strike rate
                fig.add_trace(
                    go.Scatter(
                        x=homeaway_avg_stats_df['HomeOrAway'],
                        y=[overall_sr] * len(homeaway_avg_stats_df),
                        mode='lines',
                        name='Career SR',
                        line=dict(color='black', width=2, dash='dash'),
                        showlegend=True
                    ),
                    row=1, col=2
                )

                # Update layout
                fig.update_layout(
                    height=500,
                    margin=dict(l=50, r=50, t=70, b=50),
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
                fig.update_yaxes(title_text="Average", col=1)
                fig.update_yaxes(title_text="Strike Rate", col=2)

                # Cache the home/away chart
                cache_dataframe(homeaway_chart_cache_key, fig)

            # Display chart title
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%); 
                        padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                        box-shadow: 0 6px 24px rgba(255, 126, 95, 0.25);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">🏠 Performance by Home/Away</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Display the bar chart (full width)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add year-by-year home/away comparison charts
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%); 
                        padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                        box-shadow: 0 6px 24px rgba(255, 126, 95, 0.25);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">🏠 Home vs Away Performance Trends by Year</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Cache key for home/away yearly trends
            homeaway_yearly_cache_key = f"{cache_key}_homeaway_yearly"
            homeaway_yearly_fig = get_cached_dataframe(homeaway_yearly_cache_key)
            
            if homeaway_yearly_fig is None:
                # Group data by Year and HomeOrAway to get yearly stats for home and away
                yearly_homeaway_stats = filtered_df.groupby(['Year', 'HomeOrAway']).agg({
                    'Runs': 'sum',
                    'Out': 'sum',
                    'Balls': 'sum',
                    'File Name': 'nunique'
                }).reset_index()
                
                # Calculate metrics
                yearly_homeaway_stats['Average'] = (yearly_homeaway_stats['Runs'] / yearly_homeaway_stats['Out']).round(2).fillna(0)
                yearly_homeaway_stats['Strike_Rate'] = ((yearly_homeaway_stats['Runs'] / yearly_homeaway_stats['Balls']) * 100).round(2).fillna(0)
                yearly_homeaway_stats['Matches'] = yearly_homeaway_stats['File Name']
                
                # Sort by year
                yearly_homeaway_stats = yearly_homeaway_stats.sort_values('Year')
                
                # Create subplots: one for average, one for strike rate
                homeaway_yearly_fig = make_subplots(rows=1, cols=2, 
                                                   subplot_titles=("Average by Year (Home vs Away)", 
                                                                  "Strike Rate by Year (Home vs Away)"))
                
                # Define colors for home, away, and neutral
                colors = {
                    'Home': '#1f77b4',   # Blue
                    'Away': '#d62728',   # Red
                    'Neutral': '#2ca02c'  # Green
                }
                
                # Add traces for Average
                for location in yearly_homeaway_stats['HomeOrAway'].unique():
                    location_data = yearly_homeaway_stats[yearly_homeaway_stats['HomeOrAway'] == location]
                    
                    homeaway_yearly_fig.add_trace(
                        go.Scatter(
                            x=location_data['Year'],
                            y=location_data['Average'],
                            mode='lines+markers',
                            name=f"{location} Avg",
                            line=dict(color=colors.get(location, '#7f7f7f')),
                            marker=dict(size=8),
                            text=location_data['Matches'].apply(lambda x: f"{x} matches"),
                            hovertemplate='Year: %{x}<br>Average: %{y:.2f}<br>%{text}<extra></extra>',
                            legendgroup=location
                        ),
                        row=1, col=1
                    )
                
                # Add traces for Strike Rate
                for location in yearly_homeaway_stats['HomeOrAway'].unique():
                    location_data = yearly_homeaway_stats[yearly_homeaway_stats['HomeOrAway'] == location]
                    
                    homeaway_yearly_fig.add_trace(
                        go.Scatter(
                            x=location_data['Year'],
                            y=location_data['Strike_Rate'],
                            mode='lines+markers',
                            name=f"{location} SR",
                            line=dict(color=colors.get(location, '#7f7f7f'), dash='dot'),
                            marker=dict(size=8),
                            text=location_data['Matches'].apply(lambda x: f"{x} matches"),
                            hovertemplate='Year: %{x}<br>Strike Rate: %{y:.2f}<br>%{text}<extra></extra>',
                            legendgroup=location,
                            showlegend=False  # Don't duplicate legend entries
                        ),
                        row=1, col=2
                    )
                
                # Add career average reference line
                if 'All' in name_choice and len(name_choice) == 1:
                    overall_avg = yearly_homeaway_stats['Average'].mean()
                    overall_sr = yearly_homeaway_stats['Strike_Rate'].mean()
                else:
                    overall_avg = bat_career_df['Avg'].iloc[0]
                    overall_sr = bat_career_df['SR'].iloc[0]
                
                years_range = yearly_homeaway_stats['Year'].unique()
                if len(years_range) >= 2:
                    min_year = min(years_range)
                    max_year = max(years_range)
                    
                    # Add reference line for average
                    homeaway_yearly_fig.add_trace(
                        go.Scatter(
                            x=[min_year, max_year],
                            y=[overall_avg, overall_avg],
                            mode='lines',
                            name='Career Average',
                            line=dict(color='black', width=2, dash='dash'),
                            legendgroup='reference'
                        ),
                        row=1, col=1
                    )
                    
                    # Add reference line for strike rate
                    homeaway_yearly_fig.add_trace(
                        go.Scatter(
                            x=[min_year, max_year],
                            y=[overall_sr, overall_sr],
                            mode='lines',
                            name='Career SR',
                            line=dict(color='black', width=2, dash='dash'),
                            legendgroup='reference',
                            showlegend=False  # Don't duplicate legend entries
                        ),
                        row=1, col=2
                    )
                
                # Update layout
                homeaway_yearly_fig.update_layout(
                    height=500,
                    font=dict(size=12),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    )
                )
                
                # Update axes
                homeaway_yearly_fig.update_xaxes(title_text="Year", showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)',
                                               tickmode='linear', dtick=1)  # Ensure only whole years are displayed
                homeaway_yearly_fig.update_yaxes(title_text="Average", showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)', row=1, col=1)
                homeaway_yearly_fig.update_yaxes(title_text="Strike Rate", showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)', row=1, col=2)
                
                # Cache the home/away yearly chart
                cache_dataframe(homeaway_yearly_cache_key, homeaway_yearly_fig)
            
            # Display the line charts
            st.plotly_chart(homeaway_yearly_fig, use_container_width=True)

        # Cumulative Stats Tab
        with tabs[9]:
            # Cache key for cumulative stats data
            cumulative_cache_key = f"{cache_key}_cumulative_stats"
            cumulative_stats_df = get_cached_dataframe(cumulative_cache_key) if REDIS_AVAILABLE else None

            if cumulative_stats_df is None:
                # Add 'Batted' column - mark 1 for each innings where the player batted
                filtered_df['Batted'] = 1

                # Sort the DataFrame
                filtered_df = filtered_df.sort_values(by=['Name', 'Match_Format', 'Date'])

                # Create the cumulative_stats_df
                cumulative_stats_df = filtered_df.groupby(['Name', 'Match_Format', 'Date']).agg({
                    'File Name': 'nunique',
                    'Batted': 'sum',
                    'Bat_Team_y': 'first',
                    'Bowl_Team_y': 'first',
                    'Out': 'sum',     
                    'Not Out': 'sum', 
                    'Balls': 'sum',   
                    'Runs': 'sum',  
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
                    'Stumped': 'sum'
                }).reset_index()

                # Sort for cumulative calculations
                cumulative_stats_df = cumulative_stats_df.sort_values(by=['Name', 'Match_Format', 'Date'])

                # Create cumulative columns for each name and format combination
                for name in cumulative_stats_df['Name'].unique():
                    for fmt in cumulative_stats_df[cumulative_stats_df['Name'] == name]['Match_Format'].unique():
                        mask = (cumulative_stats_df['Name'] == name) & (cumulative_stats_df['Match_Format'] == fmt)
                        
                        cumulative_stats_df.loc[mask, 'Cumulative Matches'] = cumulative_stats_df.loc[mask, 'Batted'].cumsum()
                        cumulative_stats_df.loc[mask, 'Cumulative Runs'] = cumulative_stats_df.loc[mask, 'Runs'].cumsum()
                        cumulative_stats_df.loc[mask, 'Cumulative 100s'] = cumulative_stats_df.loc[mask, '100s'].cumsum()
                        cumulative_stats_df.loc[mask, 'Cumulative Balls'] = cumulative_stats_df.loc[mask, 'Balls'].cumsum()
                        cumulative_stats_df.loc[mask, 'Cumulative Outs'] = cumulative_stats_df.loc[mask, 'Out'].cumsum()
                        
                        # Calculate running averages and rates
                        cum_outs = cumulative_stats_df.loc[mask, 'Cumulative Outs']
                        cum_runs = cumulative_stats_df.loc[mask, 'Cumulative Runs']
                        cum_balls = cumulative_stats_df.loc[mask, 'Cumulative Balls']
                        cum_matches = cumulative_stats_df.loc[mask, 'Cumulative Matches']
                        
                        # Calculate averages and rates
                        cumulative_stats_df.loc[mask, 'Cumulative Avg'] = (cum_runs / cum_outs.replace(0, np.nan)).fillna(0)
                        cumulative_stats_df.loc[mask, 'Cumulative SR'] = ((cum_runs / cum_balls.replace(0, np.nan)) * 100).fillna(0)
                        cumulative_stats_df.loc[mask, 'Runs per Match'] = (cum_runs / cum_matches.replace(0, np.nan)).fillna(0)

                # Drop unnecessary columns
                columns_to_drop = ['<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped']
                cumulative_stats_df = cumulative_stats_df.drop(columns=columns_to_drop)

                # Sort by Cumulative Matches
                cumulative_stats_df = cumulative_stats_df.sort_values(by='Cumulative Matches', ascending=False)

                if REDIS_AVAILABLE:
                    cache_dataframe(cumulative_cache_key, cumulative_stats_df)

            # Display the cumulative statistics
            st.markdown("""
            <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(17, 153, 142, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">📈 Cumulative Statistics</h3>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(cumulative_stats_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            # Function to generate a random hex color
            def random_color():
                return f'#{random.randint(0, 0xFFFFFF):06x}'

            # Create a dictionary for match format colors dynamically
            color_map = {}

            # Create three columns for the plots
            col1, col2, col3 = st.columns(3)

            # Function to create a single plot
            def create_plot(data, x_col, y_col, title, color_map, show_legend=True):
                fig = go.Figure()
                
                for match_format in data['Match_Format'].unique():
                    format_stats = data[data['Match_Format'] == match_format]
                    
                    for player in format_stats['Name'].unique():
                        legend_name = f"{player} ({match_format})"
                        if legend_name not in color_map:
                            color_map[legend_name] = random_color()
                        
                        player_stats = format_stats[format_stats['Name'] == player]
                        
                        fig.add_trace(go.Scatter(
                            x=player_stats[x_col],
                            y=player_stats[y_col],
                            mode='lines+markers',
                            name=legend_name,
                            marker=dict(color=color_map[legend_name], size=2)
                        ))
                
                fig.update_layout(
                    title=title,
                    height=400,
                    showlegend=show_legend,
                    margin=dict(l=0, r=0, t=40, b=0),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis_title="Matches",
                    yaxis_title=title,
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.2,
                        xanchor="center",
                        x=0.5,
                        bgcolor='rgba(0,0,0,0)'
                    )
                )
                return fig

            # Create and display plots in columns
            with col1:
                fig1 = create_plot(cumulative_stats_df, 'Cumulative Matches', 'Cumulative Avg', 'Cumulative Average', color_map)
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                fig2 = create_plot(cumulative_stats_df, 'Cumulative Matches', 'Cumulative SR', 'Cumulative Strike Rate', color_map)
                st.plotly_chart(fig2, use_container_width=True)

            with col3:
                fig3 = create_plot(cumulative_stats_df, 'Cumulative Matches', 'Cumulative Runs', 'Cumulative Runs', color_map)
                st.plotly_chart(fig3, use_container_width=True)

        # Block Stats Tab
        with tabs[10]:
            # Cache key for block statistics
            block_stats_cache_key = f"{cache_key}_block_stats"
            block_stats_df = get_cached_dataframe(block_stats_cache_key)

            if block_stats_df is None:
                # Ensure 'Date' column is in the proper format for chronological sorting
                filtered_df['Date'] = pd.to_datetime(filtered_df['Date'], format='%d %b %Y').dt.date

                # Add match block column based on Cumulative Matches (now using 20-match ranges)
                cumulative_stats_df['Match_Range'] = (((cumulative_stats_df['Cumulative Matches'] - 1) // 20) * 20).astype(str) + '-' + \
                                                    ((((cumulative_stats_df['Cumulative Matches'] - 1) // 20) * 20 + 19)).astype(str)

                # Add a numeric start range for sorting
                cumulative_stats_df['Range_Start'] = ((cumulative_stats_df['Cumulative Matches'] - 1) // 20) * 20

                # Group by blocks and calculate differences for each statistic
                block_stats_df = cumulative_stats_df.groupby(['Name', 'Match_Format', 'Match_Range', 'Range_Start']).agg({
                    'Cumulative Matches': 'count',            
                    'Out': 'sum',                            
                    'Not Out': 'sum',                        
                    'Balls': 'sum',                          
                    'Runs': 'sum',                           
                    '4s': 'sum',                             
                    '6s': 'sum',                             
                    '50s': 'sum',                            
                    '100s': 'sum',                           
                    '200s': 'sum',                           
                    'Date': ['first', 'last']                
                }).reset_index()

                # Flatten the column names after aggregation
                block_stats_df.columns = ['Name', 'Match_Format', 'Match_Range', 'Range_Start',
                                        'Matches', 'Outs', 'Not_Outs', 'Balls', 'Runs', 
                                        '4s', '6s', '50s', '100s', 
                                        '200s', 'First_Date', 'Last_Date']

                # Format dates
                block_stats_df['First_Date'] = pd.to_datetime(block_stats_df['First_Date']).dt.strftime('%d/%m/%Y')
                block_stats_df['Last_Date'] = pd.to_datetime(block_stats_df['Last_Date']).dt.strftime('%d/%m/%Y')

                # Calculate statistics for each block
                block_stats_df['Batting_Average'] = (block_stats_df['Runs'] / block_stats_df['Outs']).round(2)
                block_stats_df['Strike_Rate'] = ((block_stats_df['Runs'] / block_stats_df['Balls']) * 100).round(2)
                block_stats_df['Innings'] = block_stats_df['Outs'] + block_stats_df['Not_Outs']

                # Create the 'Date_Range' column
                block_stats_df['Date_Range'] = block_stats_df['First_Date'] + ' to ' + block_stats_df['Last_Date']

                # Sort and prepare final DataFrame
                block_stats_df = block_stats_df.sort_values(['Name', 'Match_Format', 'Range_Start'])

                # Select and order final columns for display
                final_columns = [
                    'Name', 'Match_Format', 'Match_Range', 'Date_Range', 
                    'Matches', 'Innings', 'Runs', 'Balls',
                    'Batting_Average', 'Strike_Rate', '4s', '6s',
                    '50s', '100s', '200s', 'Not_Outs'
                ]
                block_stats_df = block_stats_df[final_columns]

                # Handle any infinities or NaN values
                block_stats_df = block_stats_df.replace([np.inf, -np.inf], np.nan)

                # Cache the block statistics
                cache_dataframe(block_stats_cache_key, block_stats_df)

            # Store the final DataFrame
            df_blocks = block_stats_df.copy()

            # Display the block statistics
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(30, 60, 114, 0.4);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">📊 Block Statistics (Groups of 20 Innings)</h3>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(df_blocks, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            # Cache key for batting average chart
            batting_avg_chart_cache_key = f"{cache_key}_batting_avg_chart"
            fig = get_cached_dataframe(batting_avg_chart_cache_key)

            if fig is None:
                # Create a bar chart for Batting Average against Match Range
                fig = go.Figure()

                # Add Batting Average trace
                fig.add_trace(
                    go.Bar(
                        x=block_stats_df['Match_Range'], 
                        y=block_stats_df['Batting_Average'], 
                        name='Batting Average', 
                        marker_color='#f84e4e'
                    )
                )

                # Calculate the appropriate average based on selection
                if 'All' in name_choice and len(name_choice) == 1:
                    # Group by Match_Range and calculate mean Batting_Average
                    range_averages = block_stats_df.groupby('Match_Range')['Batting_Average'].mean()
                    # Calculate overall average across all ranges
                    overall_avg = range_averages.mean()
                else:
                    # Use individual player's average from bat_career_df
                    overall_avg = bat_career_df['Avg'].iloc[0]

                # Add horizontal line for average
                fig.add_trace(
                    go.Scatter(
                        x=[block_stats_df['Match_Range'].iloc[0], block_stats_df['Match_Range'].iloc[-1]],
                        y=[overall_avg, overall_avg],
                        mode='lines+text',
                        name='Average',
                        line=dict(color='black', width=2, dash='dash'),
                        text=[f"{'Range' if 'All' in name_choice and len(name_choice) == 1 else 'Career'} Average: {overall_avg:.2f}", ''],
                        textposition='top center',
                        showlegend=False
                    )
                )

                # Update layout
                fig.update_layout(
                    height=500,
                    xaxis_title='Innings Range',
                    yaxis_title='Batting Average',
                    margin=dict(l=50, r=50, t=70, b=50),
                    font=dict(size=12),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )

                # Update y-axis to show intervals of 5
                fig.update_yaxes(
                    tickmode='linear',
                    dtick=5,
                    range=[0, max(block_stats_df['Batting_Average'].fillna(0)) + 5]  # Add padding and handle NaN values
                )

                # Update x-axis to show all bar positions
                fig.update_xaxes(
                    tickmode='linear',
                    tick0=0,
                    dtick=1
                )

                # Cache the batting average chart
                cache_dataframe(batting_avg_chart_cache_key, fig)

            # Display the bar chart
            st.plotly_chart(fig, key="final_bar_chart")  # Added unique key

        # Renamed from "Visualizations" to "Distributions" Tab
        with tabs[11]:
            # Advanced metrics caching
            advanced_metrics_cache_key = f"{cache_key}_advanced_metrics"
            advanced_metrics = get_cached_dataframe(advanced_metrics_cache_key)

            if advanced_metrics is None:
                # Create three columns for additional metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    # Boundary Rate Analysis
                    boundary_fig = go.Figure()
                    boundary_fig.add_trace(go.Pie(
                        labels=['4s', '6s', 'Other Runs'],
                        values=[filtered_df['4s'].sum() * 4, 
                                filtered_df['6s'].sum() * 6,
                                filtered_df['Runs'].sum() - (filtered_df['4s'].sum() * 4 + filtered_df['6s'].sum() * 6)],
                        hole=.3,
                        marker_colors=['#f84e4e', '#4ef84e', '#4e4ef8']
                    ))
                    boundary_fig.update_layout(
                        height=400,
                        showlegend=True,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                                padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                                box-shadow: 0 6px 24px rgba(250, 112, 154, 0.25);
                                border: 1px solid rgba(255, 255, 255, 0.2);">
                        <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">📊 Run Distribution</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(boundary_fig, use_container_width=True)

                with col2:
                    # Dismissal Analysis
                    dismissal_fig = go.Figure()
                    dismissal_types = ['Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 'Not Out']
                    dismissal_values = [
                        filtered_df['Caught'].sum(),
                        filtered_df['Bowled'].sum(),
                        filtered_df['LBW'].sum(),
                        filtered_df['Run Out'].sum(),
                        filtered_df['Stumped'].sum(),
                        filtered_df['Not Out'].sum()
                    ]
                    dismissal_fig.add_trace(go.Pie(
                        labels=dismissal_types,
                        values=dismissal_values,
                        hole=.3
                    ))
                    dismissal_fig.update_layout(
                        height=400,
                        showlegend=True,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                                box-shadow: 0 6px 24px rgba(240, 147, 251, 0.25);
                                border: 1px solid rgba(255, 255, 255, 0.2);">
                        <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">⚰️ Dismissal Distribution</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(dismissal_fig, use_container_width=True)

                with col3:
                    # Score Range Distribution
                    score_ranges = ['0-24', '25-49', '50-99', '100+']
                    score_values = [
                        filtered_df[filtered_df['Runs'] < 25]['File Name'].count(),
                        filtered_df[(filtered_df['Runs'] >= 25) & (filtered_df['Runs'] < 50)]['File Name'].count(),
                        filtered_df[(filtered_df['Runs'] >= 50) & (filtered_df['Runs'] < 100)]['File Name'].count(),
                        filtered_df[filtered_df['Runs'] >= 100]['File Name'].count()
                    ]
                    score_fig = go.Figure()
                    score_fig.add_trace(go.Pie(
                        labels=score_ranges,
                        values=score_values,
                        hole=.3,
                        marker_colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
                    ))
                    score_fig.update_layout(
                        height=400,
                        showlegend=True,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%); 
                                padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                                box-shadow: 0 6px 24px rgba(78, 205, 196, 0.25);
                                border: 1px solid rgba(255, 255, 255, 0.2);">
                        <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">📈 Score Range Distribution</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(score_fig, use_container_width=True)

                # Cache the advanced metrics
                advanced_metrics = {
                    'boundary_fig': boundary_fig,
                    'dismissal_fig': dismissal_fig,
                    'score_fig': score_fig
                }
                cache_dataframe(advanced_metrics_cache_key, advanced_metrics)
            else:
                # Display cached figures
                st.plotly_chart(advanced_metrics['boundary_fig'], use_container_width=True)
                st.plotly_chart(advanced_metrics['dismissal_fig'], use_container_width=True)
                st.plotly_chart(advanced_metrics['score_fig'], use_container_width=True)

            # Create a section for distribution analysis
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">📊 Distribution Analysis</h3>
            </div>
            """, unsafe_allow_html=True)

            # Create three columns for the distribution charts
            col1, col2, col3 = st.columns(3)

            # Create a color map for players (excluding 'All')
            player_colors = {
                player: f'#{random.randint(0xFF4040, 0xFF9999):06x}'  # Different shades of red
                for player in name_choice if player != 'All'
            }

            # Get the unfiltered data for overall statistics
            overall_df = bat_df.copy()

            with col1:
                # Run Distribution Analysis
                boundary_fig = go.Figure()
                
                # Calculate overall statistics using unfiltered data
                overall_stats = {
                    '4s': (overall_df['4s'].sum() * 4 / overall_df['Runs'].sum() * 100),
                    '6s': (overall_df['6s'].sum() * 6 / overall_df['Runs'].sum() * 100),
                    'Other': (100 - ((overall_df['4s'].sum() * 4 + overall_df['6s'].sum() * 6) / overall_df['Runs'].sum() * 100))
                }

                # Add overall bar (always show this)
                boundary_fig.add_trace(go.Bar(
                    name='Overall',
                    x=['4s', '6s', 'Other Runs'],
                    y=[overall_stats['4s'], overall_stats['6s'], overall_stats['Other']],
                    marker_color='#808080',
                    opacity=0.7
                ))

                # Add selected players with their unique colors
                for player in name_choice:
                    if player != 'All':
                        player_df = filtered_df[filtered_df['Name'] == player]
                        if len(player_df) > 0:
                            fours_pct = player_df['4s'].sum() * 4 / player_df['Runs'].sum() * 100
                            sixes_pct = player_df['6s'].sum() * 6 / player_df['Runs'].sum() * 100
                            other_pct = 100 - (fours_pct + sixes_pct)
                            
                            boundary_fig.add_trace(go.Bar(
                                name=player,
                                x=['4s', '6s', 'Other Runs'],
                                y=[fours_pct, sixes_pct, other_pct],
                                marker_color=player_colors[player]
                            ))

                boundary_fig.update_layout(
                    barmode='group',
                    #title='Run Distribution (%)',
                    height=400,
                    showlegend=True,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis_title='Percentage',
                    yaxis=dict(range=[0, 100])
                )
                st.plotly_chart(boundary_fig, use_container_width=True)

            with col2:
                # Dismissal Analysis
                dismissal_fig = go.Figure()
                dismissal_types = ['Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 'Not Out']
                
                # Calculate overall percentages using unfiltered data
                total_dismissals_overall = sum([overall_df[type].sum() for type in dismissal_types])
                overall_percentages = [overall_df[type].sum() / total_dismissals_overall * 100 for type in dismissal_types]
                
                # Add overall bar (always show this)
                dismissal_fig.add_trace(go.Bar(
                    name='Overall',
                    x=dismissal_types,
                    y=overall_percentages,
                    marker_color='#808080',
                    opacity=0.7
                ))

                # Add selected players with their unique colors
                for player in name_choice:
                    if player != 'All':
                        player_df = filtered_df[filtered_df['Name'] == player]
                        if len(player_df) > 0:  # Fixed curly brace { to colon :
                            player_total = sum([player_df[type].sum() for type in dismissal_types])
                            player_percentages = [player_df[type].sum() / player_total * 100 for type in dismissal_types]
                            
                            dismissal_fig.add_trace(go.Bar(
                                name=player,
                                x=dismissal_types,
                                y=player_percentages,
                                marker_color=player_colors[player]
                            ))

                dismissal_fig.update_layout(
                    barmode='group',
                    #title='Dismissal Distribution (%)',
                    height=400,
                    showlegend=True,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis_title='Percentage',
                    yaxis=dict(range=[0, 100])
                )
                st.plotly_chart(dismissal_fig, use_container_width=True)

            with col3:
                # Score Range Analysis
                score_fig = go.Figure()
                score_ranges = ['0-24', '25-49', '50-99', '100+']
                
                # Calculate overall percentages using unfiltered data
                total_innings_overall = len(overall_df)
                overall_percentages = [
                    len(overall_df[overall_df['Runs'] < 25]) / total_innings_overall * 100,
                    len(overall_df[(overall_df['Runs'] >= 25) & (overall_df['Runs'] < 50)]) / total_innings_overall * 100,
                    len(overall_df[(overall_df['Runs'] >= 50) & (overall_df['Runs'] < 100)]) / total_innings_overall * 100,
                    len(overall_df[overall_df['Runs'] >= 100]) / total_innings_overall * 100
                ]
                
                # Add overall bar (always show this)
                score_fig.add_trace(go.Bar(
                    name='Overall',
                    x=score_ranges,
                    y=overall_percentages,
                    marker_color='#808080',
                    opacity=0.7
                ))

                # Add selected players with their unique colors
                for player in name_choice:
                    if player != 'All':
                        player_df = filtered_df[filtered_df['Name'] == player]
                        if len(player_df) > 0:
                            player_percentages = [
                                len(player_df[player_df['Runs'] < 25]) / len(player_df) * 100,
                                len(player_df[(player_df['Runs'] >= 25) & (player_df['Runs'] < 50)]) / len(player_df) * 100,
                                len(player_df[(player_df['Runs'] >= 50) & (player_df['Runs'] < 100)]) / len(player_df) * 100,
                                len(player_df[player_df['Runs'] >= 100]) / len(player_df) * 100
                            ]
                            
                            score_fig.add_trace(go.Bar(
                                name=player,
                                x=score_ranges,
                                y=player_percentages,
                                marker_color=player_colors[player]
                            ))

                score_fig.update_layout(
                    barmode='group',
                    #title='Score Range Distribution (%)',
                    height=400,
                    showlegend=True,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis_title='Percentage',
                    yaxis=dict(range=[0, 100])
                )
                st.plotly_chart(score_fig, use_container_width=True)

            # Display the bar chart
            #st.plotly_chart(fig)

        # New Percentile tab - moved from Visualizations tab
        with tabs[12]:
            # Create percentile analysis
            st.markdown("""
            <div style="background: linear-gradient(135deg, #a8caba 0%, #5d4e75 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(168, 202, 186, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">📊 Percentile Analysis (Min 10 Matches)</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a copy of df_format for percentile analysis
            percentile_df = df_format.copy()
            
            # Filter for players with 10 or more matches
            percentile_df = percentile_df[percentile_df['Matches'] >= 10]
            
            # Check if there are any players with 10+ matches
            if len(percentile_df) == 0:
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #3b82f6, #1e40af);
                    padding: 1rem;
                    border-radius: 10px;
                    border-left: 4px solid #60a5fa;
                    margin: 1rem 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    <p style="color: white; margin: 0; font-weight: 500;">
                        ℹ️ No players found with 10 or more matches in the current selection.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Calculate percentiles for each format separately
                for format in percentile_df['Match_Format'].unique():
                    format_mask = percentile_df['Match_Format'] == format
                    
                    # Calculate percentiles for each metric
                    percentile_df.loc[format_mask, 'Avg_Percentile'] = percentile_df.loc[format_mask, 'Avg'].rank(pct=True) * 100
                    percentile_df.loc[format_mask, 'BPO_Percentile'] = percentile_df.loc[format_mask, 'BPO'].rank(pct=True) * 100
                    percentile_df.loc[format_mask, 'SR_Percentile'] = percentile_df.loc[format_mask, 'SR'].rank(pct=True) * 100
                    percentile_df.loc[format_mask, '50+PI_Percentile'] = percentile_df.loc[format_mask, '50+PI'].rank(pct=True) * 100
                    percentile_df.loc[format_mask, '100PI_Percentile'] = percentile_df.loc[format_mask, '100PI'].rank(pct=True) * 100

                # Round percentiles to 1 decimal place
                percentile_columns = ['Avg_Percentile', 'BPO_Percentile', 'SR_Percentile', '50+PI_Percentile', '100PI_Percentile']
                for col in percentile_columns:
                    percentile_df[col] = percentile_df[col].round(1)

                # Calculate total percentile score (sum of all percentile columns)
                percentile_df['Total_Score'] = percentile_df[percentile_columns].sum(axis=1)

                # Calculate the total percentile ranking within each format
                for format in percentile_df['Match_Format'].unique():
                    format_mask = percentile_df['Match_Format'] == format
                    percentile_df.loc[format_mask, 'Total_Percentile'] = (
                        percentile_df.loc[format_mask, 'Total_Score'].rank(pct=True) * 100
                    ).round(1)

                # Select and reorder columns for display
                display_columns = ['Name', 'Match_Format', 'Matches', 'Runs', 'Avg', 'BPO', 'SR', '50+PI', '100PI'] + percentile_columns + ['Total_Score', 'Total_Percentile']
                
                # Sort by Total_Percentile in descending order within each format
                percentile_df = percentile_df[display_columns].sort_values(['Match_Format', 'Total_Percentile'], ascending=[True, False])
                
                # Display the percentile analysis
                st.dataframe(percentile_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})
                
                # Add visual representation of percentiles
                st.markdown("""
                <div style="background: linear-gradient(135deg, #a8caba 0%, #5d4e75 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(168, 202, 186, 0.25);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">📊 Percentile Visualization</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Create a radar chart for the top player in each format
                for format in percentile_df['Match_Format'].unique():
                    format_data = percentile_df[percentile_df['Match_Format'] == format]
                    if len(format_data) > 0:
                        top_player = format_data.iloc[0]
                        
                        # Create radar chart
                        fig = go.Figure()
                        
                        # Add the top player's percentiles
                        fig.add_trace(go.Scatterpolar(
                            r=[top_player['Avg_Percentile'], 
                               top_player['BPO_Percentile'], 
                               top_player['SR_Percentile'], 
                               top_player['50+PI_Percentile'],
                               top_player['100PI_Percentile']],
                            theta=['Avg', 'BPO', 'SR', '50+PI', '100PI'],
                            fill='toself',
                            name=f"{top_player['Name']} ({format})"
                        ))
                        
                        # Add a reference line at 50th percentile
                        fig.add_trace(go.Scatterpolar(
                            r=[50, 50, 50, 50, 50],
                            theta=['Avg', 'BPO', 'SR', '50+PI', '100PI'],
                            fill='toself',
                            name='50th Percentile',
                            line=dict(color='gray', dash='dash'),
                            opacity=0.3
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 100]
                                )
                            ),
                            showlegend=True,
                            height=500,
                            title=f"Top Player in {format}",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

        # Records Tab (now Records Tab)
        with tabs[13]:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">🏅 Single Innings Bests</h3>
            </div>
            """, unsafe_allow_html=True)
            # Create columns for layout
            col1, col2 = st.columns(2)

            with col1:
                # --- Top 10 Highest Scores ---
                best_inns_cache_key = f"{cache_key}_best_innings"
                best_inns_df = get_cached_dataframe(best_inns_cache_key)

                if best_inns_df is None:
                    # Select relevant columns and sort by Runs
                    best_inns_df = filtered_df[['Name', 'Runs', 'Balls', 'Bowl_Team_y', 'Year']].copy()
                    best_inns_df = best_inns_df.sort_values(by='Runs', ascending=False).head(10)

                    # Add Rank column
                    best_inns_df.insert(0, 'Rank', range(1, 1 + len(best_inns_df)))

                    # Rename columns
                    best_inns_df = best_inns_df.rename(columns={'Bowl_Team_y': 'Opponent'})

                    # Reorder columns
                    best_inns_df = best_inns_df[['Rank', 'Name', 'Runs', 'Balls', 'Opponent', 'Year']] # Year was already added

                    # Cache the best innings data
                    cache_dataframe(best_inns_cache_key, best_inns_df)

                # Display the Best Innings header
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">🎯 Top 10 Highest Scores</h3>
                </div>
                """, unsafe_allow_html=True)
                # Display the dataframe
                st.dataframe(best_inns_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            with col2:
                # --- Top 10 Best Strike Rate (Innings, min 50 runs) ---
                best_inns_sr_cache_key = f"{cache_key}_best_inns_sr_min50"
                best_inns_sr_df = get_cached_dataframe(best_inns_sr_cache_key)

                if best_inns_sr_df is None:
                    # Filter for innings with at least 50 runs
                    inns_min_50 = filtered_df[filtered_df['Runs'] >= 50].copy()

                    # Calculate SR, handle division by zero
                    inns_min_50['SR'] = ((inns_min_50['Runs'] / inns_min_50['Balls'].replace(0, np.nan)) * 100).round(2)

                    # Select relevant columns and sort by SR
                    best_inns_sr_df = inns_min_50[['Name', 'Runs', 'Balls', 'SR', 'Bowl_Team_y', 'Year']].copy()
                    best_inns_sr_df = best_inns_sr_df.sort_values(by='SR', ascending=False, na_position='last').head(10)

                    # Add Rank column
                    best_inns_sr_df.insert(0, 'Rank', range(1, 1 + len(best_inns_sr_df)))

                    # Rename columns
                    best_inns_sr_df = best_inns_sr_df.rename(columns={'Bowl_Team_y': 'Opponent'})

                    # Reorder columns
                    best_inns_sr_df = best_inns_sr_df[['Rank', 'Name', 'SR', 'Runs', 'Balls', 'Opponent', 'Year']]

                    # Cache the data
                    cache_dataframe(best_inns_sr_cache_key, best_inns_sr_df)

                # Display the header
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">⚡ Top 10 Best SR (Inns, Min 50 Runs)</h3>
                </div>
                """, unsafe_allow_html=True)
                # Display the dataframe
                st.dataframe(best_inns_sr_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})


            # --- Seasonal Bests ---
            st.divider()
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">📊 Seasonal Bests (by Format)</h3>
            </div>
            """, unsafe_allow_html=True)
            col3, col4 = st.columns(2)

            with col3:
                # --- Top 10 Most Runs in a Season by Format ---
                best_season_runs_cache_key = f"{cache_key}_best_season_runs"
                best_season_runs_df = get_cached_dataframe(best_season_runs_cache_key)

                if best_season_runs_df is None:
                    # Group by Name, Year, Match_Format and sum Runs
                    season_runs_df = filtered_df.groupby(['Name', 'Year', 'Match_Format']).agg(
                        Runs=('Runs', 'sum')
                    ).reset_index()

                    # Sort by Runs descending and get top 10
                    best_season_runs_df = season_runs_df.sort_values(by='Runs', ascending=False).head(10)

                    # Add Rank column
                    best_season_runs_df.insert(0, 'Rank', range(1, 1 + len(best_season_runs_df)))

                    # Reorder columns
                    best_season_runs_df = best_season_runs_df[['Rank', 'Name', 'Year', 'Match_Format', 'Runs']]

                    # Cache the data
                    cache_dataframe(best_season_runs_cache_key, best_season_runs_df)

                # Display the header
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">🏆 Top 10 Most Runs (Season)</h3>
                </div>
                """, unsafe_allow_html=True)
                # Display the dataframe
                st.dataframe(best_season_runs_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            with col4:
                # --- Top 10 Best Average Per Season by Format ---
                best_season_avg_cache_key = f"{cache_key}_best_season_avg"
                best_season_avg_df = get_cached_dataframe(best_season_avg_cache_key)

                if best_season_avg_df is None:
                    # Group by Name, Year, Match_Format and sum Runs/Out
                    season_avg_df = filtered_df.groupby(['Name', 'Year', 'Match_Format']).agg(
                        Runs=('Runs', 'sum'),
                        Out=('Out', 'sum')
                    ).reset_index()

                    # Calculate Average, handle division by zero
                    season_avg_df['Average'] = (season_avg_df['Runs'] / season_avg_df['Out'].replace(0, np.nan)).round(2)

                    # Sort by Average descending (NaNs will be pushed to the bottom) and get top 10
                    best_season_avg_df = season_avg_df.sort_values(by='Average', ascending=False, na_position='last').head(10)

                    # Add Rank column
                    best_season_avg_df.insert(0, 'Rank', range(1, 1 + len(best_season_avg_df)))

                    # Reorder columns
                    best_season_avg_df = best_season_avg_df[['Rank', 'Name', 'Year', 'Match_Format', 'Average', 'Runs', 'Out']]

                    # Cache the data
                    cache_dataframe(best_season_avg_cache_key, best_season_avg_df)

                # Display the header
                # Display the header
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">📈 Top 10 Best Average (Season)</h3>
                </div>
                """, unsafe_allow_html=True)
                # Display the dataframe
                st.dataframe(best_season_avg_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            # Create new row
            st.divider() # Add a visual separator
            col5, col6 = st.columns(2)

            with col5:
                # --- Top 10 Best SR Per Season by Format ---
                best_season_sr_cache_key = f"{cache_key}_best_season_sr"
                best_season_sr_df = get_cached_dataframe(best_season_sr_cache_key)

                if best_season_sr_df is None:
                    # Group by Name, Year, Match_Format and sum Runs/Balls
                    season_sr_df = filtered_df.groupby(['Name', 'Year', 'Match_Format']).agg(
                        Runs=('Runs', 'sum'),
                        Balls=('Balls', 'sum')
                    ).reset_index()

                    # Calculate SR, handle division by zero
                    season_sr_df['SR'] = ((season_sr_df['Runs'] / season_sr_df['Balls'].replace(0, np.nan)) * 100).round(2)

                    # Sort by SR descending and get top 10
                    best_season_sr_df = season_sr_df.sort_values(by='SR', ascending=False, na_position='last').head(10)

                    # Add Rank column
                    best_season_sr_df.insert(0, 'Rank', range(1, 1 + len(best_season_sr_df)))

                    # Reorder columns
                    best_season_sr_df = best_season_sr_df[['Rank', 'Name', 'Year', 'Match_Format', 'SR', 'Runs', 'Balls']]

                    # Cache the data
                    cache_dataframe(best_season_sr_cache_key, best_season_sr_df)

                # Display the header
                # Display the header
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">⚡ Top 10 Best SR (Season)</h3>
                </div>
                """, unsafe_allow_html=True)
                # Display the dataframe
                st.dataframe(best_season_sr_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            with col6:
                # --- Top 10 Best BPO Per Season by Format ---
                best_season_bpo_cache_key = f"{cache_key}_best_season_bpo"
                best_season_bpo_df = get_cached_dataframe(best_season_bpo_cache_key)

                if best_season_bpo_df is None:
                    # Group by Name, Year, Match_Format and sum Balls/Out
                    season_bpo_df = filtered_df.groupby(['Name', 'Year', 'Match_Format']).agg(
                        Balls=('Balls', 'sum'),
                        Out=('Out', 'sum')
                    ).reset_index()

                    # Calculate BPO, handle division by zero
                    season_bpo_df['BPO'] = (season_bpo_df['Balls'] / season_bpo_df['Out'].replace(0, np.nan)).round(2)

                    # Sort by BPO descending (higher is better) and get top 10
                    best_season_bpo_df = season_bpo_df.sort_values(by='BPO', ascending=False, na_position='last').head(10)

                    # Add Rank column
                    best_season_bpo_df.insert(0, 'Rank', range(1, 1 + len(best_season_bpo_df)))

                    # Reorder columns
                    best_season_bpo_df = best_season_bpo_df[['Rank', 'Name', 'Year', 'Match_Format', 'BPO', 'Balls', 'Out']]

                    # Cache the data
                    cache_dataframe(best_season_bpo_cache_key, best_season_bpo_df)

                # Display the header
                # Display the header
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">🎯 Top 10 Best BPO (Season)</h3>
                </div>
                """, unsafe_allow_html=True)
                # Display the dataframe
                st.dataframe(best_season_bpo_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            # Create new row
            st.divider()
            col7, col8 = st.columns(2)

            with col7:
                # --- Top 10 Highest Runs Per Match (Season by Format) ---
                best_season_rpm_cache_key = f"{cache_key}_best_season_rpm"
                best_season_rpm_df = get_cached_dataframe(best_season_rpm_cache_key)

                if best_season_rpm_df is None:
                    # Group by Name, Year, Match_Format and aggregate Runs and Matches
                    season_rpm_df = filtered_df.groupby(['Name', 'Year', 'Match_Format']).agg(
                        Runs=('Runs', 'sum'),
                        Matches=('File Name', 'nunique') # Count unique matches
                    ).reset_index()

                    # Calculate Runs Per Match, handle division by zero
                    season_rpm_df['Runs Per Match'] = (season_rpm_df['Runs'] / season_rpm_df['Matches'].replace(0, np.nan)).round(2)

                    # Sort by Runs Per Match descending and get top 10
                    best_season_rpm_df = season_rpm_df.sort_values(by='Runs Per Match', ascending=False, na_position='last').head(10)

                    # Add Rank column
                    best_season_rpm_df.insert(0, 'Rank', range(1, 1 + len(best_season_rpm_df)))

                    # Reorder columns
                    best_season_rpm_df = best_season_rpm_df[['Rank', 'Name', 'Year', 'Match_Format', 'Runs Per Match', 'Runs', 'Matches']]

                    # Cache the data
                    cache_dataframe(best_season_rpm_cache_key, best_season_rpm_df)

                # Display the header
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">💰 Top 10 Runs Per Match (Season)</h3>
                </div>
                """, unsafe_allow_html=True)
                # Display the dataframe
                st.dataframe(best_season_rpm_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            with col8:
                # --- Top 10 Most 50s Per Season by Format ---
                best_season_50s_cache_key = f"{cache_key}_best_season_50s"
                best_season_50s_df = get_cached_dataframe(best_season_50s_cache_key)

                if best_season_50s_df is None:
                    # Ensure '50s' column exists
                    if '50s' not in filtered_df.columns:
                        filtered_df['50s'] = ((filtered_df['Runs'] >= 50) & (filtered_df['Runs'] < 100)).astype(int)

                    # Group by Name, Year, Match_Format and sum 50s
                    season_50s_df = filtered_df.groupby(['Name', 'Year', 'Match_Format']).agg(
                        Fifties=('50s', 'sum')
                    ).reset_index()

                    # Sort by Fifties descending and get top 10
                    best_season_50s_df = season_50s_df.sort_values(by='Fifties', ascending=False).head(10)

                    # Add Rank column
                    best_season_50s_df.insert(0, 'Rank', range(1, 1 + len(best_season_50s_df)))

                    # Reorder columns
                    best_season_50s_df = best_season_50s_df[['Rank', 'Name', 'Year', 'Match_Format', 'Fifties']]

                    # Cache the data
                    cache_dataframe(best_season_50s_cache_key, best_season_50s_df)

                # Display the header
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">🥇 Top 10 Most 50s (Season)</h3>
                </div>
                """, unsafe_allow_html=True)
                # Display the dataframe
                st.dataframe(best_season_50s_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            # Create new row
            st.divider()
            col9, col10 = st.columns(2)

            with col9:
                # --- Top 10 Most 100s Per Season by Format ---
                best_season_100s_cache_key = f"{cache_key}_best_season_100s"
                best_season_100s_df = get_cached_dataframe(best_season_100s_cache_key)

                if best_season_100s_df is None:
                    # Ensure '100s' column exists from pre-processing or calculate it
                    if '100s' not in filtered_df.columns:
                        filtered_df['100s'] = (filtered_df['Runs'] >= 100).astype(int) # Recalculate if missing

                    # Group by Name, Year, Match_Format and sum 100s
                    season_100s_df = filtered_df.groupby(['Name', 'Year', 'Match_Format']).agg(
                        Hundreds=('100s', 'sum') # Use the correct column name '100s'
                    ).reset_index()

                    # Sort by Hundreds descending and get top 10
                    best_season_100s_df = season_100s_df.sort_values(by='Hundreds', ascending=False).head (10)

                    # Add Rank column
                    best_season_100s_df.insert(0, 'Rank', range(1, 1 + len(best_season_100s_df)))

                    # Reorder columns
                    best_season_100s_df = best_season_100s_df[['Rank', 'Name', 'Year', 'Match_Format', 'Hundreds']]

                    # Cache the data
                    cache_dataframe(best_season_100s_cache_key, best_season_100s_df)

                # Display the header
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">💯 Top 10 Most 100s (Season)</h3>
                </div>
                """, unsafe_allow_html=True)
                # Display the dataframe
                st.dataframe(best_season_100s_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            with col10:
                # --- Top 10 Most POM Per Season ---
                best_season_pom_cache_key = f"{cache_key}_best_season_pom"
                best_season_pom_df = get_cached_dataframe(best_season_pom_cache_key)

                if best_season_pom_df is None:
                    # Filter for POM awards and count unique matches per player per year and format
                    pom_df = filtered_df[filtered_df['Player_of_the_Match'] == filtered_df['Name']]
                    # Group by Name, Year, and Match_Format
                    season_pom_df = pom_df.groupby(['Name', 'Year', 'Match_Format'])['File Name'].nunique().reset_index(name='POM')

                    # Sort by POM descending and get top 10
                    best_season_pom_df = season_pom_df.sort_values(by='POM', ascending=False).head(10)

                    # Add Rank column
                    best_season_pom_df.insert(0, 'Rank', range(1, 1 + len(best_season_pom_df)))

                    # Reorder columns - Added Match_Format
                    best_season_pom_df = best_season_pom_df[['Rank', 'Name', 'Year', 'Match_Format', 'POM']]

                    # Cache the data
                    cache_dataframe(best_season_pom_cache_key, best_season_pom_df)

                # Display the header - Updated title
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">🏆 Top 10 Most POM (Season)</h3>
                </div>
                """, unsafe_allow_html=True)
                # Display the dataframe
                st.dataframe(best_season_pom_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            # Create new row
            st.divider()
            col11, col12 = st.columns(2)

            with col11:
                # --- Top 10 Highest Boundary % Per Season (by Format) ---
                best_season_boundary_pct_cache_key = f"{cache_key}_best_season_boundary_pct"
                best_season_boundary_pct_df = get_cached_dataframe(best_season_boundary_pct_cache_key)

                if best_season_boundary_pct_df is None:
                    # Group by Name, Year, Match_Format and aggregate Runs, 4s, 6s
                    season_boundary_df = filtered_df.groupby(['Name', 'Year', 'Match_Format']).agg(
                        Runs=('Runs', 'sum'),
                        Fours=('4s', 'sum'),
                        Sixes=('6s', 'sum')
                    ).reset_index()

                    # Calculate Boundary Runs and Boundary Percentage
                    season_boundary_df['Boundary Runs'] = (season_boundary_df['Fours'] * 4) + (season_boundary_df['Sixes'] * 6)
                    season_boundary_df['Boundary %'] = ((season_boundary_df['Boundary Runs'] / season_boundary_df['Runs'].replace(0, np.nan)) * 100).round(2)

                    # Sort by Boundary % descending and get top 10
                    best_season_boundary_pct_df = season_boundary_df.sort_values(by='Boundary %', ascending=False, na_position='last').head(10)

                    # Add Rank column
                    best_season_boundary_pct_df.insert(0, 'Rank', range(1, 1 + len(best_season_boundary_pct_df)))

                    # Reorder columns
                    best_season_boundary_pct_df = best_season_boundary_pct_df[['Rank', 'Name', 'Year', 'Match_Format', 'Boundary %', 'Boundary Runs', 'Runs']]

                    # Cache the data
                    cache_dataframe(best_season_boundary_pct_cache_key, best_season_boundary_pct_df)

                # Display the header
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">🎯 Top 10 Boundary % (Season)</h3>
                </div>
                """, unsafe_allow_html=True)
                # Display the dataframe
                st.dataframe(best_season_boundary_pct_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})


            # --- Calculate Seasonal Match Averages and SR ---
            # This calculation needs to happen before the Match+ sections, but doesn't need its own column
            seasonal_match_metrics_cache_key = f"{cache_key}_seasonal_match_metrics_by_format"
            seasonal_match_metrics = get_cached_dataframe(seasonal_match_metrics_cache_key)

            if seasonal_match_metrics is None:
                # Calculate file-level stats per year and format (match-level stats)
                file_stats_seasonal = filtered_df.groupby(['File Name', 'Year', 'Match_Format']).agg({ # Added Match_Format
                    'Total_Runs': 'first',
                    'Team Balls': 'first',
                    'Wickets': 'first'
                }).reset_index()

                # Calculate match-level metrics per year and format
                file_stats_seasonal['Match_Avg'] = file_stats_seasonal['Total_Runs'] / file_stats_seasonal['Wickets'].replace(0, np.nan)
                file_stats_seasonal['Match_SR'] = (file_stats_seasonal['Total_Runs'] / file_stats_seasonal['Team Balls'].replace(0, np.nan)) * 100

                # Calculate the average of these match metrics per season and format
                seasonal_match_metrics = file_stats_seasonal.groupby(['Year', 'Match_Format']).agg( # Added Match_Format
                    Avg_Match_Avg=('Match_Avg', 'mean'),
                    Avg_Match_SR=('Match_SR', 'mean')
                ).reset_index()

                # Round the calculated seasonal averages
                seasonal_match_metrics['Avg_Match_Avg'] = seasonal_match_metrics['Avg_Match_Avg'].round(2)
                seasonal_match_metrics['Avg_Match_SR'] = seasonal_match_metrics['Avg_Match_SR'].round(2)

                cache_dataframe(seasonal_match_metrics_cache_key, seasonal_match_metrics)

            with col12:
                # --- Top 10 Best Match+ Avg Per Season ---
                best_season_match_plus_avg_cache_key = f"{cache_key}_best_season_match_plus_avg_by_format"
                best_season_match_plus_avg_df = get_cached_dataframe(best_season_match_plus_avg_cache_key)

                if best_season_match_plus_avg_df is None:
                    # Group player stats by Name, Year, and Match_Format
                    player_seasonal_stats = filtered_df.groupby(['Name', 'Year', 'Match_Format']).agg( # Added Match_Format
                        Runs=('Runs', 'sum'),
                        Out=('Out', 'sum')
                    ).reset_index()

                    # Calculate player's seasonal average
                    player_seasonal_stats['Player_Avg'] = (player_seasonal_stats['Runs'] / player_seasonal_stats['Out'].replace(0, np.nan)).round(2)

                    # Merge with seasonal match metrics on Year and Match_Format
                    merged_stats = pd.merge(player_seasonal_stats, seasonal_match_metrics, on=['Year', 'Match_Format'], how='left') # Merge on both

                    # Calculate Match+ Avg
                    merged_stats['Match+ Avg'] = ((merged_stats['Player_Avg'] / merged_stats['Avg_Match_Avg'].replace(0, np.nan)) * 100).round(2)

                    # Sort by Match+ Avg descending and get top 10
                    best_season_match_plus_avg_df = merged_stats.sort_values(by='Match+ Avg', ascending=False, na_position='last').head(10)

                    # Add Rank column
                    best_season_match_plus_avg_df.insert(0, 'Rank', range(1, 1 + len(best_season_match_plus_avg_df)))

                    # Reorder columns - Added Match_Format
                    best_season_match_plus_avg_df = best_season_match_plus_avg_df[['Rank', 'Name', 'Year', 'Match_Format', 'Match+ Avg', 'Player_Avg', 'Avg_Match_Avg']]

                    # Cache the data
                    cache_dataframe(best_season_match_plus_avg_cache_key, best_season_match_plus_avg_df)

                # Display the header - Updated title
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">📊 Top 10 Best Match+ Avg (Season)</h3>
                </div>
                """, unsafe_allow_html=True)
                # Display the dataframe
                st.dataframe(best_season_match_plus_avg_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            # Create new row for the last seasonal item
            st.divider()
            col13, col14 = st.columns(2) # Use two columns, leave one empty

            with col13:
                # --- Top 10 Best Match+ SR Per Season ---
                best_season_match_plus_sr_cache_key = f"{cache_key}_best_season_match_plus_sr_by_format"
                best_season_match_plus_sr_df = get_cached_dataframe(best_season_match_plus_sr_cache_key)

                if best_season_match_plus_sr_df is None:
                    # Group player stats by Name, Year, and Match_Format
                    player_seasonal_stats_sr = filtered_df.groupby(['Name', 'Year', 'Match_Format']).agg( # Added Match_Format
                        Runs=('Runs', 'sum'),
                        Balls=('Balls', 'sum')
                    ).reset_index()

                    # Calculate player's seasonal SR
                    player_seasonal_stats_sr['Player_SR'] = ((player_seasonal_stats_sr['Runs'] / player_seasonal_stats_sr['Balls'].replace(0, np.nan)) * 100).round(2)

                    # Merge with seasonal match metrics on Year and Match_Format
                    merged_stats_sr = pd.merge(player_seasonal_stats_sr, seasonal_match_metrics, on=['Year', 'Match_Format'], how='left') # Merge on both

                    # Calculate Match+ SR
                    merged_stats_sr['Match+ SR'] = ((merged_stats_sr['Player_SR'] / merged_stats_sr['Avg_Match_SR'].replace(0, np.nan)) * 100).round(2)

                    # Sort by Match+ SR descending and get top 10
                    best_season_match_plus_sr_df = merged_stats_sr.sort_values(by='Match+ SR', ascending=False, na_position='last').head(10)

                    # Add Rank column
                    best_season_match_plus_sr_df.insert(0, 'Rank', range(1, 1 + len(best_season_match_plus_sr_df)))

                    # Reorder columns - Added Match_Format
                    best_season_match_plus_sr_df = best_season_match_plus_sr_df[['Rank', 'Name', 'Year', 'Match_Format', 'Match+ SR', 'Player_SR', 'Avg_Match_SR']]

                    # Cache the data
                    cache_dataframe(best_season_match_plus_sr_cache_key, best_season_match_plus_sr_df)

                # Display the header - Updated title
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">⚡ Top 10 Best Match+ SR (Season)</h3>
                </div>
                """, unsafe_allow_html=True)
                # Display the dataframe
                st.dataframe(best_season_match_plus_sr_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            # col14 is empty


            # --- Career Bests ---
            st.divider()
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">🏆 Career Bests (by Format)</h3>
            </div>
            """, unsafe_allow_html=True)
            col15, col16 = st.columns(2)

            with col15:
                # --- Top 10 Most Runs (Career by Format) ---
                best_career_runs_cache_key = f"{cache_key}_best_career_runs"
                best_career_runs_df = get_cached_dataframe(best_career_runs_cache_key)

                if best_career_runs_df is None:
                    # Use original bat_df for career calculations to ensure we get true career totals
                    # regardless of year filtering applied elsewhere
                    career_df = bat_df.copy()
                    
                    # Apply non-year filters to maintain consistency with other filters
                    if name_choice and 'All' not in name_choice:
                        career_df = career_df[career_df['Name'].isin(name_choice)]
                    if bat_team_choice and 'All' not in bat_team_choice:
                        career_df = career_df[career_df['Bat_Team_y'].isin(bat_team_choice)]
                    if bowl_team_choice and 'All' not in bowl_team_choice:
                        career_df = career_df[career_df['Bowl_Team_y'].isin(bowl_team_choice)]
                    if match_format_choice and 'All' not in match_format_choice:
                        career_df = career_df[career_df['Match_Format'].isin(match_format_choice)]
                    if comp_choice and 'All' not in comp_choice:
                        career_df = career_df[career_df['comp'].isin(comp_choice)]
                        
                    # Group by Name and Match_Format for career total (no Year grouping)
                    career_runs_df = career_df.groupby(['Name', 'Match_Format']).agg({
                        'Runs': 'sum'  # Sum of all runs for each player in each format across all years
                    }).reset_index()
                    
                    # Sort by Runs descending and get top 10
                    best_career_runs_df = career_runs_df.sort_values(by='Runs', ascending=False).head(10)
                    best_career_runs_df.insert(0, 'Rank', range(1, 1 + len(best_career_runs_df)))
                    best_career_runs_df = best_career_runs_df[['Rank', 'Name', 'Match_Format', 'Runs']]
                    cache_dataframe(best_career_runs_cache_key, best_career_runs_df)

                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">🏆 Top 10 Most Runs</h3>
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(best_career_runs_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            with col16:
                # --- Top 10 Best Average (Career by Format, min 10 inns) ---
                best_career_avg_cache_key = f"{cache_key}_best_career_avg_min10inns"
                best_career_avg_df = get_cached_dataframe(best_career_avg_cache_key)

                if best_career_avg_df is None:
                    # Use the same career_df for batting average calculations
                    # This ensures consistency between career stats sections
                    if not 'career_df' in locals():
                        # Use original bat_df for career calculations to ensure we get true career totals
                        # regardless of year filtering applied elsewhere
                        career_df = bat_df.copy()
                        
                        # Apply non-year filters to maintain consistency with other filters
                        if name_choice and 'All' not in name_choice:
                            career_df = career_df[career_df['Name'].isin(name_choice)]
                        if bat_team_choice and 'All' not in bat_team_choice:
                            career_df = career_df[career_df['Bat_Team_y'].isin(bat_team_choice)]
                        if bowl_team_choice and 'All' not in bowl_team_choice:
                            career_df = career_df[career_df['Bowl_Team_y'].isin(bowl_team_choice)]
                        if match_format_choice and 'All' not in match_format_choice:
                            career_df = career_df[career_df['Match_Format'].isin(match_format_choice)]
                        if comp_choice and 'All' not in comp_choice:
                            career_df = career_df[career_df['comp'].isin(comp_choice)]
                    
                    # Group by Name and Match_Format only for true career totals
                    career_avg_df = career_df.groupby(['Name', 'Match_Format']).agg({
                        'Runs': 'sum',
                        'Out': 'sum',
                        'Batted': 'sum'  # Count innings
                    }).reset_index()
                    
                    # Apply min innings filter AFTER aggregation
                    career_avg_df = career_avg_df[career_avg_df['Batted'] >= 10]
                    # Calculate Average, handle division by zero
                    career_avg_df['Average'] = (career_avg_df['Runs'] / career_avg_df['Out'].replace(0, np.nan)).round(2)
                    # Sort and get top 10
                    best_career_avg_df = career_avg_df.sort_values(by='Average', ascending=False, na_position='last').head(10)
                    best_career_avg_df.insert(0, 'Rank', range(1, 1 + len(best_career_avg_df)))
                    best_career_avg_df = best_career_avg_df[['Rank', 'Name', 'Match_Format', 'Average', 'Runs', 'Out', 'Batted']]
                    # Rename column for clarity
                    best_career_avg_df = best_career_avg_df.rename(columns={'Batted': 'Inns'})
                    cache_dataframe(best_career_avg_cache_key, best_career_avg_df)

                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">📈 Top 10 Best Average (Min 10 Inns)</h3>
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(best_career_avg_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            st.divider()
            col17, col18 = st.columns(2)

            with col17:
                # --- Top 10 Best SR (Career by Format, min 500 balls) ---
                best_career_sr_cache_key = f"{cache_key}_best_career_sr_min500b"
                best_career_sr_df = get_cached_dataframe(best_career_sr_cache_key)

                if best_career_sr_df is None:
                    # Ensure we have career_df for consistent career calculations
                    if not 'career_df' in locals():
                        # Use original bat_df for career calculations to ensure we get true career totals
                        # regardless of year filtering applied elsewhere
                        career_df = bat_df.copy()
                        
                        # Apply non-year filters to maintain consistency with other filters
                        if name_choice and 'All' not in name_choice:
                            career_df = career_df[career_df['Name'].isin(name_choice)]
                        if bat_team_choice and 'All' not in bat_team_choice:
                            career_df = career_df[career_df['Bat_Team_y'].isin(bat_team_choice)]
                        if bowl_team_choice and 'All' not in bowl_team_choice:
                            career_df = career_df[career_df['Bowl_Team_y'].isin(bowl_team_choice)]
                        if match_format_choice and 'All' not in match_format_choice:
                            career_df = career_df[career_df['Match_Format'].isin(match_format_choice)]
                        if comp_choice and 'All' not in comp_choice:
                            career_df = career_df[career_df['comp'].isin(comp_choice)]
                    
                    # Group by Name and Match_Format only for career total
                    career_sr_df = career_df.groupby(['Name', 'Match_Format']).agg(
                        Runs=('Runs', 'sum'),
                        Balls=('Balls', 'sum')
                    ).reset_index()
                    # Apply min balls filter AFTER aggregation
                    career_sr_df = career_sr_df[career_sr_df['Balls'] >= 500]
                    career_sr_df['SR'] = ((career_sr_df['Runs'] / career_sr_df['Balls'].replace(0, np.nan)) * 100).round(2)
                    best_career_sr_df = career_sr_df.sort_values(by='SR', ascending=False, na_position='last').head(10)
                    best_career_sr_df.insert(0, 'Rank', range(1, 1 + len(best_career_sr_df)))
                    best_career_sr_df = best_career_sr_df[['Rank', 'Name', 'Match_Format', 'SR', 'Runs', 'Balls']]
                    cache_dataframe(best_career_sr_cache_key, best_career_sr_df)

                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">⚡ Top 10 Best SR (Min 500 Balls)</h3>
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(best_career_sr_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            with col18:
                # --- Top 10 Best BPO (Career by Format, min 10 inns) ---
                best_career_bpo_cache_key = f"{cache_key}_best_career_bpo_min10inns"
                best_career_bpo_df = get_cached_dataframe(best_career_bpo_cache_key)

                if best_career_bpo_df is None:
                    # Ensure we have career_df for consistent career calculations
                    if not 'career_df' in locals():
                        # Use original bat_df for career calculations to ensure we get true career totals
                        # regardless of year filtering applied elsewhere
                        career_df = bat_df.copy()
                        
                        # Apply non-year filters to maintain consistency with other filters
                        if name_choice and 'All' not in name_choice:
                            career_df = career_df[career_df['Name'].isin(name_choice)]
                        if bat_team_choice and 'All' not in bat_team_choice:
                            career_df = career_df[career_df['Bat_Team_y'].isin(bat_team_choice)]
                        if bowl_team_choice and 'All' not in bowl_team_choice:
                            career_df = career_df[career_df['Bowl_Team_y'].isin(bowl_team_choice)]
                        if match_format_choice and 'All' not in match_format_choice:
                            career_df = career_df[career_df['Match_Format'].isin(match_format_choice)]
                        if comp_choice and 'All' not in comp_choice:
                            career_df = career_df[career_df['comp'].isin(comp_choice)]
                    
                    # Group by Name and Match_Format only for career total
                    career_bpo_df = career_df.groupby(['Name', 'Match_Format']).agg(
                        Balls=('Balls', 'sum'),
                        Out=('Out', 'sum'),
                        Inns=('Batted', 'sum')
                    ).reset_index()
                    # Apply min innings filter AFTER aggregation
                    career_bpo_df = career_bpo_df[career_bpo_df['Inns'] >= 10]
                    career_bpo_df['BPO'] = (career_bpo_df['Balls'] / career_bpo_df['Out'].replace(0, np.nan)).round(2)
                    best_career_bpo_df = career_bpo_df.sort_values(by='BPO', ascending=False, na_position='last').head(10)
                    best_career_bpo_df.insert(0, 'Rank', range(1, 1 + len(best_career_bpo_df)))
                    best_career_bpo_df = best_career_bpo_df[['Rank', 'Name', 'Match_Format', 'BPO', 'Balls', 'Out', 'Inns']]
                    cache_dataframe(best_career_bpo_cache_key, best_career_bpo_df)

                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">🎯 Top 10 Best BPO (Min 10 Inns)</h3>
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(best_career_bpo_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            st.divider()
            col19, col20 = st.columns(2)

            with col19:
                # --- Top 10 Highest Runs Per Match (Career by Format, min 10 matches) ---
                best_career_rpm_cache_key = f"{cache_key}_best_career_rpm_min10m"
                best_career_rpm_df = get_cached_dataframe(best_career_rpm_cache_key)

                if best_career_rpm_df is None:
                    # Ensure we have career_df for consistent career calculations
                    if not 'career_df' in locals():
                        # Use original bat_df for career calculations to ensure we get true career totals
                        # regardless of year filtering applied elsewhere
                        career_df = bat_df.copy()
                        
                        # Apply non-year filters to maintain consistency with other filters
                        if name_choice and 'All' not in name_choice:
                            career_df = career_df[career_df['Name'].isin(name_choice)]
                        if bat_team_choice and 'All' not in bat_team_choice:
                            career_df = career_df[career_df['Bat_Team_y'].isin(bat_team_choice)]
                        if bowl_team_choice and 'All' not in bowl_team_choice:
                            career_df = career_df[career_df['Bowl_Team_y'].isin(bowl_team_choice)]
                        if match_format_choice and 'All' not in match_format_choice:
                            career_df = career_df[career_df['Match_Format'].isin(match_format_choice)]
                        if comp_choice and 'All' not in comp_choice:
                            career_df = career_df[career_df['comp'].isin(comp_choice)]
                    
                    # Group by Name and Match_Format only for career total
                    career_rpm_df = career_df.groupby(['Name', 'Match_Format']).agg(
                        Runs=('Runs', 'sum'),
                        Matches=('File Name', 'nunique')
                    ).reset_index()
                    # Apply min matches filter AFTER aggregation
                    career_rpm_df = career_rpm_df[career_rpm_df['Matches'] >= 10]
                    career_rpm_df['Runs Per Match'] = (career_rpm_df['Runs'] / career_rpm_df['Matches'].replace(0, np.nan)).round(2)
                    best_career_rpm_df = career_rpm_df.sort_values(by='Runs Per Match', ascending=False, na_position='last').head(10)
                    best_career_rpm_df.insert(0, 'Rank', range(1, 1 + len(best_career_rpm_df)))
                    best_career_rpm_df = best_career_rpm_df[['Rank', 'Name', 'Match_Format', 'Runs Per Match', 'Runs', 'Matches']]
                    cache_dataframe(best_career_rpm_cache_key, best_career_rpm_df)

                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">💰 Top 10 Runs Per Match (Min 10 Matches)</h3>
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(best_career_rpm_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            with col20:
                # --- Top 10 Most 50s (Career by Format) ---
                best_career_50s_cache_key = f"{cache_key}_best_career_50s"
                best_career_50s_df = get_cached_dataframe(best_career_50s_cache_key)

                if best_career_50s_df is None:
                    # Ensure we have career_df for consistent career calculations
                    if not 'career_df' in locals():
                        # Use original bat_df for career calculations to ensure we get true career totals
                        # regardless of year filtering applied elsewhere
                        career_df = bat_df.copy()
                        
                        # Apply non-year filters to maintain consistency with other filters
                        if name_choice and 'All' not in name_choice:
                            career_df = career_df[career_df['Name'].isin(name_choice)]
                        if bat_team_choice and 'All' not in bat_team_choice:
                            career_df = career_df[career_df['Bat_Team_y'].isin(bat_team_choice)]
                        if bowl_team_choice and 'All' not in bowl_team_choice:
                            career_df = career_df[career_df['Bowl_Team_y'].isin(bowl_team_choice)]
                        if match_format_choice and 'All' not in match_format_choice:
                            career_df = career_df[career_df['Match_Format'].isin(match_format_choice)]
                        if comp_choice and 'All' not in comp_choice:
                            career_df = career_df[career_df['comp'].isin(comp_choice)]
                    
                    # Group by Name and Match_Format only for career total
                    career_50s_df = career_df.groupby(['Name', 'Match_Format']).agg(
                        Fifties=('50s', 'sum')
                    ).reset_index()
                    best_career_50s_df = career_50s_df.sort_values(by='Fifties', ascending=False).head(10)
                    best_career_50s_df.insert(0, 'Rank', range(1, 1 + len(best_career_50s_df)))
                    best_career_50s_df = best_career_50s_df[['Rank', 'Name', 'Match_Format', 'Fifties']]
                    cache_dataframe(best_career_50s_cache_key, best_career_50s_df)

                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">🥇 Top 10 Most 50s</h3>
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(best_career_50s_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            st.divider()
            col21, col22 = st.columns(2)

            with col21:
                # --- Top 10 Most 100s (Career by Format) ---
                best_career_100s_cache_key = f"{cache_key}_best_career_100s"
                best_career_100s_df = get_cached_dataframe(best_career_100s_cache_key)

                if best_career_100s_df is None:
                    # Ensure we have career_df for consistent career calculations
                    if not 'career_df' in locals():
                        # Use original bat_df for career calculations to ensure we get true career totals
                        # regardless of year filtering applied elsewhere
                        career_df = bat_df.copy()
                        
                        # Apply non-year filters to maintain consistency with other filters
                        if name_choice and 'All' not in name_choice:
                            career_df = career_df[career_df['Name'].isin(name_choice)]
                        if bat_team_choice and 'All' not in bat_team_choice:
                            career_df = career_df[career_df['Bat_Team_y'].isin(bat_team_choice)]
                        if bowl_team_choice and 'All' not in bowl_team_choice:
                            career_df = career_df[career_df['Bowl_Team_y'].isin(bowl_team_choice)]
                        if match_format_choice and 'All' not in match_format_choice:
                            career_df = career_df[career_df['Match_Format'].isin(match_format_choice)]
                        if comp_choice and 'All' not in comp_choice:
                            career_df = career_df[career_df['comp'].isin(comp_choice)]
                    
                    # Group by Name and Match_Format only for career total
                    career_100s_df = career_df.groupby(['Name', 'Match_Format']).agg(
                        Hundreds=('100s', 'sum')
                    ).reset_index()
                    best_career_100s_df = career_100s_df.sort_values(by='Hundreds', ascending=False).head(10)
                    best_career_100s_df.insert(0, 'Rank', range(1, 1 + len(best_career_100s_df)))
                    best_career_100s_df = best_career_100s_df[['Rank', 'Name', 'Match_Format', 'Hundreds']]
                    cache_dataframe(best_career_100s_cache_key, best_career_100s_df)

                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">💯 Top 10 Most 100s</h3>
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(best_career_100s_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            with col22:
                # --- Top 10 Most POM (Career by Format) ---
                best_career_pom_cache_key = f"{cache_key}_best_career_pom"
                best_career_pom_df = get_cached_dataframe(best_career_pom_cache_key)

                if best_career_pom_df is None:
                    # Ensure we have career_df for consistent career calculations
                    if not 'career_df' in locals():
                        # Use original bat_df for career calculations to ensure we get true career totals
                        # regardless of year filtering applied elsewhere
                        career_df = bat_df.copy()
                        
                        # Apply non-year filters to maintain consistency with other filters
                        if name_choice and 'All' not in name_choice:
                            career_df = career_df[career_df['Name'].isin(name_choice)]
                        if bat_team_choice and 'All' not in bat_team_choice:
                            career_df = career_df[career_df['Bat_Team_y'].isin(bat_team_choice)]
                        if bowl_team_choice and 'All' not in bowl_team_choice:
                            career_df = career_df[career_df['Bowl_Team_y'].isin(bowl_team_choice)]
                        if match_format_choice and 'All' not in match_format_choice:
                            career_df = career_df[career_df['Match_Format'].isin(match_format_choice)]
                        if comp_choice and 'All' not in comp_choice:
                            career_df = career_df[career_df['comp'].isin(comp_choice)]
                    
                    # Filter for POM awards first using career_df
                    pom_df_career = career_df[career_df['Player_of_the_Match'] == career_df['Name']]
                    # Group by Name and Match_Format only for career total
                    career_pom_df = pom_df_career.groupby(['Name', 'Match_Format'])['File Name'].nunique().reset_index(name='POM')
                    best_career_pom_df = career_pom_df.sort_values(by='POM', ascending=False).head(10)
                    best_career_pom_df.insert(0, 'Rank', range(1, 1 + len(best_career_pom_df)))
                    best_career_pom_df = best_career_pom_df[['Rank', 'Name', 'Match_Format', 'POM']]
                    cache_dataframe(best_career_pom_cache_key, best_career_pom_df)

                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">🏆 Top 10 Most POM</h3>
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(best_career_pom_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            st.divider()
            col23, col24 = st.columns(2)

            with col23:
                # --- Top 10 Highest Boundary % (Career by Format, min 500 runs) ---
                best_career_boundary_pct_cache_key = f"{cache_key}_best_career_boundary_pct_min500r"
                best_career_boundary_pct_df = get_cached_dataframe(best_career_boundary_pct_cache_key)

                if best_career_boundary_pct_df is None:
                    # Ensure we have career_df for consistent career calculations
                    if not 'career_df' in locals():
                        # Use original bat_df for career calculations to ensure we get true career totals
                        # regardless of year filtering applied elsewhere
                        career_df = bat_df.copy()
                        
                        # Apply non-year filters to maintain consistency with other filters
                        if name_choice and 'All' not in name_choice:
                            career_df = career_df[career_df['Name'].isin(name_choice)]
                        if bat_team_choice and 'All' not in bat_team_choice:
                            career_df = career_df[career_df['Bat_Team_y'].isin(bat_team_choice)]
                        if bowl_team_choice and 'All' not in bowl_team_choice:
                            career_df = career_df[career_df['Bowl_Team_y'].isin(bowl_team_choice)]
                        if match_format_choice and 'All' not in match_format_choice:
                            career_df = career_df[career_df['Match_Format'].isin(match_format_choice)]
                        if comp_choice and 'All' not in comp_choice:
                            career_df = career_df[career_df['comp'].isin(comp_choice)]
                    
                    # Group by Name and Match_Format only for career total
                    career_boundary_df = career_df.groupby(['Name', 'Match_Format']).agg(
                        Runs=('Runs', 'sum'),
                        Fours=('4s', 'sum'),
                        Sixes=('6s', 'sum')
                    ).reset_index()
                    # Apply min runs filter AFTER aggregation
                    career_boundary_df = career_boundary_df[career_boundary_df['Runs'] >= 500]
                    career_boundary_df['Boundary Runs'] = (career_boundary_df['Fours'] * 4) + (career_boundary_df['Sixes'] * 6)
                    career_boundary_df['Boundary %'] = ((career_boundary_df['Boundary Runs'] / career_boundary_df['Runs'].replace(0, np.nan)) * 100).round(2)
                    best_career_boundary_pct_df = career_boundary_df.sort_values(by='Boundary %', ascending=False, na_position='last').head(10)
                    best_career_boundary_pct_df.insert(0, 'Rank', range(1, 1 + len(best_career_boundary_pct_df)))
                    best_career_boundary_pct_df = best_career_boundary_pct_df[['Rank', 'Name', 'Match_Format', 'Boundary %', 'Boundary Runs', 'Runs']]
                    cache_dataframe(best_career_boundary_pct_cache_key, best_career_boundary_pct_df)

                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">🎯 Top 10 Boundary % (Min 500 Runs)</h3>
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(best_career_boundary_pct_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})


            # --- Calculate Overall Match Averages and SR per Format ---
            # This calculation remains the same - it's based on all matches in the dataset per format
            overall_match_metrics_cache_key = f"{cache_key}_overall_match_metrics_by_format"
            overall_match_metrics = get_cached_dataframe(overall_match_metrics_cache_key)

            if overall_match_metrics is None:
                # Use the original unfiltered bat_df for overall calculations
                file_stats_overall = bat_df.groupby(['File Name', 'Match_Format']).agg({
                    'Total_Runs': 'first',
                    'Team Balls': 'first',
                    'Wickets': 'first'
                }).reset_index()
                file_stats_overall['Match_Avg'] = file_stats_overall['Total_Runs'] / file_stats_overall['Wickets'].replace(0, np.nan)
                file_stats_overall['Match_SR'] = (file_stats_overall['Total_Runs'] / file_stats_overall['Team Balls'].replace(0, np.nan)) * 100
                # Group by format to get the average match avg/sr across all matches of that format
                overall_match_metrics = file_stats_overall.groupby(['Match_Format']).agg(
                    Overall_Avg_Match_Avg=('Match_Avg', 'mean'),
                    Overall_Avg_Match_SR=('Match_SR', 'mean')
                ).reset_index()
                overall_match_metrics['Overall_Avg_Match_Avg'] = overall_match_metrics['Overall_Avg_Match_Avg'].round(2)
                overall_match_metrics['Overall_Avg_Match_SR'] = overall_match_metrics['Overall_Avg_Match_SR'].round(2)
                cache_dataframe(overall_match_metrics_cache_key, overall_match_metrics)


            with col24:
                # --- Top 10 Best Match+ Avg (Career by Format, min 10 inns) ---
                best_career_match_plus_avg_cache_key = f"{cache_key}_best_career_match_plus_avg_min10inns"
                best_career_match_plus_avg_df = get_cached_dataframe(best_career_match_plus_avg_cache_key)

                if best_career_match_plus_avg_df is None:
                    # Ensure we have career_df for consistent career calculations
                    if not 'career_df' in locals():
                        # Use original bat_df for career calculations to ensure we get true career totals
                        # regardless of year filtering applied elsewhere
                        career_df = bat_df.copy()
                        
                        # Apply non-year filters to maintain consistency with other filters
                        if name_choice and 'All' not in name_choice:
                            career_df = career_df[career_df['Name'].isin(name_choice)]
                        if bat_team_choice and 'All' not in bat_team_choice:
                            career_df = career_df[career_df['Bat_Team_y'].isin(bat_team_choice)]
                        if bowl_team_choice and 'All' not in bowl_team_choice:
                            career_df = career_df[career_df['Bowl_Team_y'].isin(bowl_team_choice)]
                        if match_format_choice and 'All' not in match_format_choice:
                            career_df = career_df[career_df['Match_Format'].isin(match_format_choice)]
                        if comp_choice and 'All' not in comp_choice:
                            career_df = career_df[career_df['comp'].isin(comp_choice)]
                    
                    # Group by Name and Match_Format only for career total
                    player_career_stats_avg = career_df.groupby(['Name', 'Match_Format']).agg(
                        Runs=('Runs', 'sum'),
                        Out=('Out', 'sum'),
                        Inns=('Batted', 'sum')
                    ).reset_index()
                    # Apply min innings filter AFTER aggregation
                    player_career_stats_avg = player_career_stats_avg[player_career_stats_avg['Inns'] >= 10]
                    player_career_stats_avg['Player_Avg'] = (player_career_stats_avg['Runs'] / player_career_stats_avg['Out'].replace(0, np.nan)).round(2)
                    # Merge with overall format metrics
                    merged_career_avg = pd.merge(player_career_stats_avg, overall_match_metrics, on='Match_Format', how='left')
                    merged_career_avg['Match+ Avg'] = ((merged_career_avg['Player_Avg'] / merged_career_avg['Overall_Avg_Match_Avg'].replace(0, np.nan)) * 100).round(2)
                    best_career_match_plus_avg_df = merged_career_avg.sort_values(by='Match+ Avg', ascending=False, na_position='last').head(10)
                    best_career_match_plus_avg_df.insert(0, 'Rank', range(1, 1 + len(best_career_match_plus_avg_df)))
                    best_career_match_plus_avg_df = best_career_match_plus_avg_df[['Rank', 'Name', 'Match_Format', 'Match+ Avg', 'Player_Avg', 'Overall_Avg_Match_Avg', 'Inns']]
                    cache_dataframe(best_career_match_plus_avg_cache_key, best_career_match_plus_avg_df)

                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">📊 Top 10 Best Match+ Avg (Min 10 Inns)</h3>
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(best_career_match_plus_avg_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            st.divider()
            col25, col26 = st.columns(2) # Use two columns, leave one empty

            with col25:
                # --- Top 10 Best Match+ SR (Career by Format, min 500 balls) ---
                best_career_match_plus_sr_cache_key = f"{cache_key}_best_career_match_plus_sr_min500b"
                best_career_match_plus_sr_df = get_cached_dataframe(best_career_match_plus_sr_cache_key)

                if best_career_match_plus_sr_df is None:
                    # Ensure we have career_df for consistent career calculations
                    if not 'career_df' in locals():
                        # Use original bat_df for career calculations to ensure we get true career totals
                        # regardless of year filtering applied elsewhere
                        career_df = bat_df.copy()
                        
                        # Apply non-year filters to maintain consistency with other filters
                        if name_choice and 'All' not in name_choice:
                            career_df = career_df[career_df['Name'].isin(name_choice)]
                        if bat_team_choice and 'All' not in bat_team_choice:
                            career_df = career_df[career_df['Bat_Team_y'].isin(bat_team_choice)]
                        if bowl_team_choice and 'All' not in bowl_team_choice:
                            career_df = career_df[career_df['Bowl_Team_y'].isin(bowl_team_choice)]
                        if match_format_choice and 'All' not in match_format_choice:
                            career_df = career_df[career_df['Match_Format'].isin(match_format_choice)]
                        if comp_choice and 'All' not in comp_choice:
                            career_df = career_df[career_df['comp'].isin(comp_choice)]
                    
                    # Group by Name and Match_Format only for career total
                    player_career_stats_sr = career_df.groupby(['Name', 'Match_Format']).agg(
                        Runs=('Runs', 'sum'),
                        Balls=('Balls', 'sum')
                    ).reset_index()
                    # Apply min balls filter AFTER aggregation
                    player_career_stats_sr = player_career_stats_sr[player_career_stats_sr['Balls'] >= 500]
                    player_career_stats_sr['Player_SR'] = ((player_career_stats_sr['Runs'] / player_career_stats_sr['Balls'].replace(0, np.nan)) * 100).round(2)
                    # Merge with overall format metrics
                    merged_career_sr = pd.merge(player_career_stats_sr, overall_match_metrics, on='Match_Format', how='left')
                    merged_career_sr['Match+ SR'] = ((merged_career_sr['Player_SR'] / merged_career_sr['Overall_Avg_Match_SR'].replace(0, np.nan)) * 100).round(2)
                    best_career_match_plus_sr_df = merged_career_sr.sort_values(by='Match+ SR', ascending=False, na_position='last').head(10)
                    best_career_match_plus_sr_df.insert(0, 'Rank', range(1, 1 + len(best_career_match_plus_sr_df)))
                    best_career_match_plus_sr_df = best_career_match_plus_sr_df[['Rank', 'Name', 'Match_Format', 'Match+ SR', 'Player_SR', 'Overall_Avg_Match_SR', 'Balls']]
                    cache_dataframe(best_career_match_plus_sr_cache_key, best_career_match_plus_sr_df)

                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">⚡ Top 10 Best Match+ SR (Min 500 Balls)</h3>
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(best_career_match_plus_sr_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            # col26 is empty

        # Win/Loss record Tab
        with tabs[14]:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">🏆 Player Win/Loss Record</h3>
            </div>
            """, unsafe_allow_html=True)

            # Cache key for win/loss data
            wl_cache_key = f"{cache_key}_wl_data"
            wl_df = get_cached_dataframe(wl_cache_key)

            if wl_df is None:
                if not filtered_df.empty and not match_df.empty and 'File Name' in filtered_df.columns and 'File Name' in match_df.columns:
                    # 1. Get unique File Names from the filtered batting data
                    unique_files_df = filtered_df[['File Name']].drop_duplicates()

                    # 2. Merge unique File Names with match_df to get unique match results
                    match_results_df = pd.merge(unique_files_df, match_df, on='File Name', how='left', suffixes=('', '_match_orig'))

                    # 3. Merge the original filtered_df (with player names) onto the unique match results
                    wl_df = pd.merge(filtered_df, match_results_df, on='File Name', how='left', suffixes=('_bat', '_match'))

                    # Cache the merged dataframe
                    if REDIS_AVAILABLE:
                        cache_dataframe(wl_cache_key, wl_df)
                else:
                    wl_df = pd.DataFrame()  # Create empty df if merge is not possible

            if not wl_df.empty:
                # --- Create winloss_df summary ---
                try:
                    # Drop duplicates at the File Name and Name level before aggregation
                    # This ensures each player's participation in a match is counted only once for win/loss stats
                    wl_agg_base = wl_df[['Name', 'File Name', 'HomeOrAway', 'Home_Win', 'Away_Won', 'Home_Drawn', 'Away_Drawn', 'Tie', 'Innings_Win']].drop_duplicates()

                    winloss_df = wl_agg_base.groupby('Name').agg(
                        Home_Matches=pd.NamedAgg(column='HomeOrAway', aggfunc=lambda x: (x == 'Home').sum()),
                        Away_Matches=pd.NamedAgg(column='HomeOrAway', aggfunc=lambda x: (x == 'Away').sum()),
                        Home_Win=pd.NamedAgg(column='Home_Win', aggfunc=lambda x: ((wl_agg_base.loc[x.index, 'HomeOrAway'] == 'Home') & (x == 1)).sum()),
                        Away_Win=pd.NamedAgg(column='Away_Won', aggfunc=lambda x: ((wl_agg_base.loc[x.index, 'HomeOrAway'] == 'Away') & (x == 1)).sum()),
                        Home_Lost=pd.NamedAgg(column='Away_Won', aggfunc=lambda x: ((wl_agg_base.loc[x.index, 'HomeOrAway'] == 'Home') & (x == 1)).sum()),
                        Away_Lost=pd.NamedAgg(column='Home_Win', aggfunc=lambda x: ((wl_agg_base.loc[x.index, 'HomeOrAway'] == 'Away') & (x == 1)).sum()),
                        Home_Draw=pd.NamedAgg(column='Home_Drawn', aggfunc=lambda x: ((wl_agg_base.loc[x.index, 'HomeOrAway'] == 'Home') & (x == 1)).sum()),
                        Away_Draw=pd.NamedAgg(column='Away_Drawn', aggfunc=lambda x: ((wl_agg_base.loc[x.index, 'HomeOrAway'] == 'Away') & (x == 1)).sum()),
                        Home_Tie=pd.NamedAgg(column='Tie', aggfunc=lambda x: ((wl_agg_base.loc[x.index, 'HomeOrAway'] == 'Home') & (x == 1)).sum()),
                        Away_Tie=pd.NamedAgg(column='Tie', aggfunc=lambda x: ((wl_agg_base.loc[x.index, 'HomeOrAway'] == 'Away') & (x == 1)).sum()),
                        Home_Innings_Win=pd.NamedAgg(column='Innings_Win', aggfunc=lambda x: ((wl_agg_base.loc[x.index, 'HomeOrAway'] == 'Home') & (wl_agg_base.loc[x.index, 'Home_Win'] == 1) & (x == 1)).sum()),
                        Away_Innings_Win=pd.NamedAgg(column='Innings_Win', aggfunc=lambda x: ((wl_agg_base.loc[x.index, 'HomeOrAway'] == 'Away') & (wl_agg_base.loc[x.index, 'Away_Won'] == 1) & (x == 1)).sum())
                    ).reset_index()

                    # Calculate combined columns
                    winloss_df['Total Matches'] = winloss_df['Home_Matches'] + winloss_df['Away_Matches']
                    winloss_df['Total Win'] = winloss_df['Home_Win'] + winloss_df['Away_Win']
                    winloss_df['Total Lost'] = winloss_df['Home_Lost'] + winloss_df['Away_Lost']
                    winloss_df['Total Draw'] = winloss_df['Home_Draw'] + winloss_df['Away_Draw']
                    winloss_df['Total Tie'] = winloss_df['Home_Tie'] + winloss_df['Away_Tie']
                    winloss_df['Total Innings Win'] = winloss_df['Home_Innings_Win'] + winloss_df['Away_Innings_Win']

                    # Calculate percentage columns, handling division by zero
                    winloss_df['Win %'] = (winloss_df['Total Win'] / winloss_df['Total Matches'].replace(0, np.nan) * 100).fillna(0).astype(int)
                    winloss_df['Lost %'] = (winloss_df['Total Lost'] / winloss_df['Total Matches'].replace(0, np.nan) * 100).fillna(0).astype(int)
                    winloss_df['Draw %'] = (winloss_df['Total Draw'] / winloss_df['Total Matches'].replace(0, np.nan) * 100).fillna(0).astype(int)

                    # Reorder columns for display: Name, Totals (with percentages), Home, Away
                    winloss_df = winloss_df[
                        ['Name', 'Total Matches', 'Total Win', 'Win %', 'Total Lost', 'Lost %', 'Total Draw', 'Draw %', 'Total Tie', 'Total Innings Win',
                        'Home_Matches', 'Home_Win', 'Home_Lost', 'Home_Draw', 'Home_Tie', 'Home_Innings_Win',
                        'Away_Matches', 'Away_Win', 'Away_Lost', 'Away_Draw', 'Away_Tie', 'Away_Innings_Win']
                    ]

                    #st.markdown("#### Win/Loss Summary by Player (winloss_df)")
                    # Display up to 50 rows with increased height
                    st.dataframe(
                        winloss_df,
                        use_container_width=True,
                        hide_index=True,
                        height=800,
                        column_config={
                            "Name": st.column_config.Column("Name", pinned=True),
                            "Win %": st.column_config.NumberColumn("Win %", format="%d%%"),
                            "Lost %": st.column_config.NumberColumn("Lost %", format="%d%%"),
                            "Draw %": st.column_config.NumberColumn("Draw %", format="%d%%"),
                        }
                    )
                    st.divider()  # Add separator
                except KeyError as e:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #ef4444, #dc2626);
                        padding: 1rem;
                        border-radius: 10px;
                        border-left: 4px solid #f87171;
                        margin: 1rem 0;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    ">
                        <p style="color: white; margin: 0; font-weight: 500;">
                            ❌ Error creating win/loss summary: Missing column - {e}. Please ensure the match data includes win/loss/draw/tie information.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #ef4444, #dc2626);
                        padding: 1rem;
                        border-radius: 10px;
                        border-left: 4px solid #f87171;
                        margin: 1rem 0;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    ">
                        <p style="color: white; margin: 0; font-weight: 500;">
                            ❌ An unexpected error occurred while creating the win/loss summary: {e}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # --- Display wl_df_display (original merged data with columns dropped) ---
                columns_to_display = ['Name', 'Home Team', 'Away Team', 'Match_Result', 'HomeOrAway', 'Home_Win', 'Away_Won', 'Home_Drawn', 'Away_Drawn', 'Tie', 'Innings_Win']
                available_cols = [col for col in columns_to_display if col in wl_df.columns]
                

# ...existing code...

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
