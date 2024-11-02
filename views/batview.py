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

CACHE_EXPIRY = timedelta(hours=24)

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
    sorted_filters = dict(sorted(filters.items()))
    return f"bat_stats_{hash(json.dumps(sorted_filters))}"

@handle_redis_errors
def clear_all_caches():
    """Clear all Redis caches"""
    redis_client.flushall()
    st.success("All caches cleared successfully")

def display_bat_view():
    # Custom CSS for styling
    st.markdown("""
    <style>
    /* Table styling */
    table { color: black; }
    thead tr th {
        background-color: #f04f53 !important;
        color: white !important;
    }
    tbody tr:nth-child(even) { background-color: #f0f2f6; }
    tbody tr:nth-child(odd) { background-color: white; }

    /* Multiselect and slider styling */
    div[data-baseweb="tag"] {
        background-color: #f04f53 !important;
    }
    .css-1p0q8wb, .css-eg6t2j {
        background-color: #f04f53 !important;
    }
    .stSlider p {
        color: #f04f53 !important;
    }
    .css-e8gw43 {
        background-color: #f04f53 !important;
    }
    .css-1cpxqw2 {
        background-color: #f04f53 !important;
        color: white !important;
    }
    
    /* Selection color */
    ::selection {
        background-color: #f04f53 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='color:#f04f53; text-align: center;'>Batting Statistics</h1>", unsafe_allow_html=True)
    
    # Add cache clearing button in sidebar (only if Redis is available)
    if REDIS_AVAILABLE and st.sidebar.button("Clear Caches"):
        clear_all_caches()
    
    # Check if bat_df is available in session state
    if 'bat_df' in st.session_state:
        bat_df = st.session_state['bat_df']

        # Create filters at the top of the page
        names = ['All'] + sorted(bat_df['Name'].unique().tolist())
        bat_teams = ['All'] + sorted(bat_df['Bat_Team_y'].unique().tolist())
        bowl_teams = ['All'] + sorted(bat_df['Bowl_Team_y'].unique().tolist())
        years = sorted(bat_df['Year'].astype(int).unique().tolist())
        positions = ['All'] + sorted(bat_df['Position'].unique().tolist())
        match_formats = ['All'] + sorted(bat_df['Match_Format'].unique().tolist())

        # Create filters at the top of the page
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            name_choice = st.multiselect('Name:', names, default='All')
        with col2:
            bat_team_choice = st.multiselect('Batting Team:', bat_teams, default='All')
        with col3:
            bowl_team_choice = st.multiselect('Bowling Team:', bowl_teams, default='All')
        with col4:
            match_format_choice = st.multiselect('Format:', match_formats, default='All')

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
        col5, col6, col7, col8, col9, col10 = st.columns(6)

        with col5:
            st.markdown("<p style='text-align: center;'>Choose Year:</p>", unsafe_allow_html=True)
            year_choice = st.slider('', 
                   min_value=min(years),
                   max_value=max(years),
                   value=(min(years), max(years)),
                   label_visibility='collapsed')
        with col6:
            st.markdown("<p style='text-align: center;'>Choose Position:</p>", unsafe_allow_html=True)
            position_choice = st.slider('', 
                       min_value=1, 
                       max_value=11, 
                       value=(1, 11),
                       label_visibility='collapsed')

        with col7:
            st.markdown("<p style='text-align: center;'>Runs Range</p>", unsafe_allow_html=True)
            runs_range = st.slider('', 
                                min_value=1, 
                                max_value=max_runs, 
                                value=(1, max_runs),
                                label_visibility='collapsed')

        with col8:
            st.markdown("<p style='text-align: center;'>Matches Range</p>", unsafe_allow_html=True)
            matches_range = st.slider('', 
                                    min_value=1, 
                                    max_value=max_matches, 
                                    value=(1, max_matches),
                                    label_visibility='collapsed')

        with col9:
            st.markdown("<p style='text-align: center;'>Average Range</p>", unsafe_allow_html=True)
            avg_range = st.slider('', 
                                min_value=0.0, 
                                max_value=max_avg, 
                                value=(0.0, max_avg),
                                label_visibility='collapsed')

        with col10:
            st.markdown("<p style='text-align: center;'>Strike Rate Range</p>", unsafe_allow_html=True)
            sr_range = st.slider('', 
                                min_value=0.0, 
                                max_value=600.0, 
                                value=(0.0, 600.0),
                                label_visibility='collapsed')

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
            'sr_range': sr_range
        }
        cache_key = generate_cache_key(filters)

        # Try to get filtered data from cache if Redis is available
        filtered_df = get_cached_dataframe(cache_key) if REDIS_AVAILABLE else None
        
        if filtered_df is None:
            # If not in cache or Redis is not available, apply filters
            filtered_df = bat_df.copy()
            filtered_df['Year'] = filtered_df['Year'].astype(int)

            if name_choice and 'All' not in name_choice:
                filtered_df = filtered_df[filtered_df['Name'].isin(name_choice)]
            if bat_team_choice and 'All' not in bat_team_choice:
                filtered_df = filtered_df[filtered_df['Bat_Team_y'].isin(bat_team_choice)]
            if bowl_team_choice and 'All' not in bowl_team_choice:
                filtered_df = filtered_df[filtered_df['Bowl_Team_y'].isin(bowl_team_choice)]
            if match_format_choice and 'All' not in match_format_choice:
                filtered_df = filtered_df[filtered_df['Match_Format'].isin(match_format_choice)]

            filtered_df = filtered_df[filtered_df['Year'].between(year_choice[0], year_choice[1])]
            filtered_df = filtered_df[filtered_df['Position'].between(position_choice[0], position_choice[1])]

            filtered_df = filtered_df.groupby('Name').filter(lambda x: 
                runs_range[0] <= x['Runs'].sum() <= runs_range[1] and
                matches_range[0] <= x['File Name'].nunique() <= matches_range[1] and
                (avg_range[0] <= (x['Runs'].sum() / x['Out'].sum()) <= avg_range[1] if x['Out'].sum() > 0 else True) and
                (sr_range[0] <= ((x['Runs'].sum() / x['Balls'].sum()) * 100) <= sr_range[1] if x['Balls'].sum() > 0 else True)
            )
            
            # Cache the filtered DataFrame if Redis is available
            if REDIS_AVAILABLE:
                cache_dataframe(cache_key, filtered_df)

######---------------------------------------CAREER STATS TAB-------------------------------

######---------------------------------------CAREER STATS TAB-------------------------------

        # Calculate milestone innings based on run ranges with caching
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

            # Flatten multi-level columns
            bat_career_df.columns = ['Name', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 
                                     'Runs', 'HS', '4s', '6s', '50s', '100s', '150s', '200s', 
                                     '<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 
                                     'Team Runs', 'Overs', 'Wickets', 'Team Balls']

            # Calculate average runs per out, strike rate, and balls per out
            bat_career_df['Avg'] = (bat_career_df['Runs'] / bat_career_df['Out']).round(2).fillna(0)
            bat_career_df['SR'] = ((bat_career_df['Runs'] / bat_career_df['Balls']) * 100).round(2).fillna(0)
            bat_career_df['BPO'] = (bat_career_df['Balls'] / bat_career_df['Out']).round(2).fillna(0)

            # Calculate new columns for team statistics
            bat_career_df['Team Avg'] = (bat_career_df['Team Runs'] / bat_career_df['Wickets']).round(2).fillna(0)
            bat_career_df['Team SR'] = (bat_career_df['Team Runs'] / bat_career_df['Team Balls'] * 100).round(2).fillna(0)

            # Calculate P+ Avg and P+ SR
            bat_career_df['P+ Avg'] = (bat_career_df['Avg'] / bat_career_df['Team Avg'] * 100).round(2).fillna(0)
            bat_career_df['P+ SR'] = (bat_career_df['SR'] / bat_career_df['Team SR'] * 100).round(2).fillna(0)

            # Calculate BPB (Balls Per Boundary)
            bat_career_df['BPB'] = (bat_career_df['Balls'] / (bat_career_df['4s'] + bat_career_df['6s']).replace(0, 1)).round(2)

            # Calculate new statistics
            bat_career_df['50+PI'] = (((bat_career_df['50s'] + bat_career_df['100s'] + bat_career_df['150s'] + bat_career_df['200s']) / bat_career_df['Inns']) * 100).round(2).fillna(0)
            bat_career_df['100PI'] = (((bat_career_df['100s'] + bat_career_df['150s'] + bat_career_df['200s']) / bat_career_df['Inns']) * 100).round(2).fillna(0)
            bat_career_df['150PI'] = (((bat_career_df['150s'] + bat_career_df['200s']) / bat_career_df['Inns']) * 100).round(2).fillna(0)
            bat_career_df['200PI'] = ((bat_career_df['200s'] / bat_career_df['Inns']) * 100).round(2).fillna(0)
            bat_career_df['<25&OutPI'] = ((bat_career_df['<25&Out'] / bat_career_df['Inns']) * 100).round(2).fillna(0)

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

            # Reorder columns and drop Team Avg and Team SR
            bat_career_df = bat_career_df[['Name', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 'Runs', 'HS', 
                                       'Avg', 'BPO', 'SR', '4s', '6s', 'BPB', '<25&Out', '50s', '100s', '150s', '200s',
                                       '<25&OutPI', '50+PI', '100PI', '150PI', '200PI', 'P+ Avg', 'P+ SR', 
                                       'Caught%', 'Bowled%', 'LBW%', 'Run Out%', 'Stumped%', 'Not Out%', 'POM']]
            
            # Sort the DataFrame by 'Runs' in descending order
            bat_career_df = bat_career_df.sort_values(by='Runs', ascending=False)
            
            # Cache the computed career statistics
            cache_dataframe(career_cache_key, bat_career_df)

        # Display the filtered and aggregated career statistics
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Career Statistics</h3>", unsafe_allow_html=True)
        st.dataframe(bat_career_df, use_container_width=True, hide_index=True)
        
#################----------------SCATTER CHART----------------------------##########################

        # Cache key for scatter plot data
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

        # Display the title using Streamlit's markdown
        st.markdown(
            "<h3 style='color:#f04f53; text-align: center;'>Batting Average vs Strike Rate Analysis</h3>",
            unsafe_allow_html=True
        )

        # Show plot
        st.plotly_chart(scatter_fig)
###---------------------------------------------FORMAT STATS-------------------------------------------------------------------###
###---------------------------------------------FORMAT STATS-------------------------------------------------------------------###

        # Cache key for format statistics
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

        # Display the filtered and aggregated format statistics
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Format Record</h3>", unsafe_allow_html=True)
        st.dataframe(df_format, use_container_width=True, hide_index=True)


 
###---------------------------------------------SESAON STATS-------------------------------------------------------------------###
 
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
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Season Statistics</h3>", unsafe_allow_html=True)
        st.dataframe(season_stats_df, use_container_width=True, hide_index=True)

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

        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Average & Strike Rate Per Season</h3>", unsafe_allow_html=True)

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
        fig.update_xaxes(title_text="Year", gridcolor='lightgray')
        fig.update_yaxes(title_text="Average", gridcolor='lightgray', col=1)
        fig.update_yaxes(title_text="Strike Rate", gridcolor='lightgray', col=2)

        st.plotly_chart(fig)
######---------------------------------------LATEST INNINGS STATS TAB-------------------------------

######---------------------------------------LATEST INNINGS STATS TAB-------------------------------

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

            # Convert Date to datetime for proper sorting
            latest_inns_df['Date'] = pd.to_datetime(latest_inns_df['Date'])

            # Sort by Date in descending order (newest to oldest)
            latest_inns_df = latest_inns_df.sort_values(by='Date', ascending=False).head(15)

            # Convert Date format to 'dd/mm/yyyy' for display
            latest_inns_df['Date'] = latest_inns_df['Date'].dt.strftime('%d/%m/%Y')

            # Reorder columns to put Runs before Balls
            latest_inns_df = latest_inns_df[['Name', 'Match_Format', 'Date', 'Innings', 'Bat Team', 'Bowl Team', 
                                           'How Out', 'Runs', 'Balls', '4s', '6s']]
            
            # Cache the latest innings data
            cache_dataframe(latest_inns_cache_key, latest_inns_df)

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

        # Display the Latest Innings Stats with conditional formatting
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Last 15 Innings</h3>", unsafe_allow_html=True)

        # Display the dataframe
        st.dataframe(styled_df, height=575, use_container_width=True, hide_index=True)
   

###---------------------------------------------OPPONENTS STATS-------------------------------------------------------------------###
###---------------------------------------------OPPONENTS STATS-------------------------------------------------------------------###
        
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
            st.markdown("<h2 style='color:#f04f53; text-align: center;'>Average Runs Against Opponents</h2>", unsafe_allow_html=True)

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

###---------------------------------------------LOCATION STATS-------------------------------------------------------------------###

###---------------------------------------------LOCATION STATS-------------------------------------------------------------------###

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

        # Display the location Stats
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Location Statistics</h3>", unsafe_allow_html=True)
        st.dataframe(opponents_stats_df, use_container_width=True, hide_index=True)

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

        # Create a bar chart for Average runs by location
        location_chart_cache_key = f"{cache_key}_location_chart"
        fig = get_cached_dataframe(location_chart_cache_key)

        if fig is None:
            fig = go.Figure()

            # Add Average runs trace
            fig.add_trace(
                go.Bar(
                    x=opponent_stats_df['Home Team'], 
                    y=opponent_stats_df['Avg'], 
                    name='Average', 
                    marker_color='#f84e4e'
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
                    x=[opponent_stats_df['Home Team'].iloc[0], opponent_stats_df['Home Team'].iloc[-1]],
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
                xaxis_title='Location',
                yaxis_title='Average',
                margin=dict(l=50, r=50, t=70, b=50),
                font=dict(size=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )

            # Cache the location chart
            cache_dataframe(location_chart_cache_key, fig)

        # Display the bar chart
        st.plotly_chart(fig)
 
 ###---------------------------------------------INNINGS STATS-------------------------------------------------------------------###
###---------------------------------------------INNINGS STATS-------------------------------------------------------------------###

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
            
            innings_stats_df = innings_stats_df.sort_values(by='Runs', ascending=False)

            # Cache the computed innings statistics
            cache_dataframe(innings_cache_key, innings_stats_df)

        # Display the Innings Stats
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Innings Statistics</h3>", unsafe_allow_html=True)
        st.dataframe(innings_stats_df, use_container_width=True, hide_index=True)

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
                margin=dict(l=50, r=50, t=70, b=50),
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

        # Display the bar chart
        st.plotly_chart(fig)
    
 ###---------------------------------------------POSITION STATS-------------------------------------------------------------------###
###---------------------------------------------POSITION STATS-------------------------------------------------------------------###

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
            
            position_stats_df = position_stats_df.sort_values(by='Runs', ascending=False)
            
            # Cache the computed position statistics
            cache_dataframe(position_cache_key, position_stats_df)

        # Display the Position Stats
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Position Statistics</h3>", unsafe_allow_html=True)
        st.dataframe(position_stats_df, use_container_width=True, hide_index=True)

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
                    x=position_avg_stats_df['Position'], 
                    y=position_avg_stats_df['Avg'], 
                    name='Average', 
                    marker_color='#f84e4e'
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
                    x=[1, 11],  # Show only positions 1-11
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
                xaxis_title='Position',
                yaxis_title='Average',
                margin=dict(l=50, r=50, t=70, b=50),
                font=dict(size=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )

            # Update y-axis to show intervals of 5
            fig.update_yaxes(
                tickmode='linear',
                dtick=5,
                range=[0, max(position_avg_stats_df['Avg']) + 5]  # Add some padding to the top
            )

            # Update x-axis to show only positions 1-11
            fig.update_xaxes(
                tickmode='linear',
                tick0=1,  # Start at position 1
                dtick=1,  # Show every position
                range=[0.5, 11.5]  # Add padding on either side of the bars
            )

            # Cache the position chart
            cache_dataframe(position_chart_cache_key, fig)

        # Display the bar chart
        st.plotly_chart(fig)


###--------------------------------------CUMULATIVE STATS CHARTS------------------------------------------#######
###--------------------------------------CUMULATIVE STATS CHARTS------------------------------------------#######

        # Convert the 'Date' column to datetime format for proper chronological sorting
        filtered_df['Date'] = pd.to_datetime(filtered_df['Date'], format='%d %b %Y').dt.date

        # Create a cache key for the cumulative stats data
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
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Cumulative Statistics</h3>", unsafe_allow_html=True)
        st.dataframe(cumulative_stats_df, use_container_width=True, hide_index=True)

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
 ###--------------------------------------BLOCK STATS------------------------------------------#######

        # Cache key for block statistics
        block_stats_cache_key = f"{cache_key}_block_stats"
        block_stats_df = get_cached_dataframe(block_stats_cache_key)

        if block_stats_df is None:
            # Ensure 'Date' column is in the proper format for chronological sorting
            filtered_df['Date'] = pd.to_datetime(filtered_df['Date'], format='%d %b %Y').dt.date

            # Add match block column based on Cumulative Matches
            cumulative_stats_df['Match_Range'] = (((cumulative_stats_df['Cumulative Matches'] - 1) // 10) * 10).astype(str) + '-' + \
                                                ((((cumulative_stats_df['Cumulative Matches'] - 1) // 10) * 10 + 9)).astype(str)

            # Add a numeric start range for sorting
            cumulative_stats_df['Range_Start'] = ((cumulative_stats_df['Cumulative Matches'] - 1) // 10) * 10

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
                                    'Fours', 'Sixes', 'Fifties', 'Hundreds', 
                                    'Double_Hundreds', 'First_Date', 'Last_Date']

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
                'Batting_Average', 'Strike_Rate', 'Fours', 'Sixes',
                'Fifties', 'Hundreds', 'Double_Hundreds', 'Not_Outs'
            ]
            block_stats_df = block_stats_df[final_columns]

            # Handle any infinities or NaN values
            block_stats_df = block_stats_df.replace([np.inf, -np.inf], np.nan)

            # Cache the block statistics
            cache_dataframe(block_stats_cache_key, block_stats_df)

        # Store the final DataFrame
        df_blocks = block_stats_df.copy()

        # Display the block statistics
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Block Statistics (Groups of 10 Matches)</h3>", unsafe_allow_html=True)
        st.dataframe(df_blocks, use_container_width=True, hide_index=True)

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
                xaxis_title='Match Range',
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
        st.plotly_chart(fig)

# Call the function to display the batting view
display_bat_view()