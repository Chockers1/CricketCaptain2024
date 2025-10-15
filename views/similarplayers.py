import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import random
import plotly.graph_objects as go
import json
from datetime import timedelta
from functools import wraps
import time
import subprocess
import seaborn as sns
import matplotlib.pyplot as plt

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

/* Filter Card Styling */
.filter-card {
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    padding: 20px;
    border-radius: 15px;
    margin: 15px 0;
    box-shadow: 0 8px 25px rgba(168, 237, 234, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

/* Table styling */
table { 
    color: black; 
    width: 100%; 
}

thead tr th {
    background-color: #f04f53 !important;
    color: white !important;
}

tbody tr:nth-child(even) { 
    background-color: #f0f2f6; 
}

tbody tr:nth-child(odd) { 
    background-color: white; 
}

/* Custom metric cards */
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 15px;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 10px 0;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    margin: 0;
}

.metric-label {
    font-size: 0.9rem;
    opacity: 0.9;
    margin: 0;
}
</style>
""", unsafe_allow_html=True)

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

def get_filtered_options(df, column, current_filters):
    """Get available options for a filter based on current selections"""
    temp_df = df.copy()
    
    for filter_col, filter_values in current_filters.items():
        if filter_col != column and filter_values and 'All' not in filter_values:
            temp_df = temp_df[temp_df[filter_col].isin(filter_values)]
    
    available_options = ['All'] + sorted(temp_df[column].dropna().unique().tolist())
    return available_options

def display_similar_players_view():
    # Display main header
    st.markdown("""
        <div class="main-header">
            <h1>🔍 Similar Players Analysis</h1>
            <p style="color: white; opacity: 0.9; font-size: 1.1rem; margin: 10px 0 0 0;">
                Find players with comparable career statistics and performance profiles
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Custom CSS for main tabs
    st.markdown("""
        <style>
        .main-tabs .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
            justify-content: center;
            margin: 20px auto;
        }
        .main-tabs .stTabs [data-baseweb="tab"] {
            font-size: 1.2rem;
            font-weight: bold;
            padding: 15px 30px;
            border-radius: 12px;
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            color: white;
            border: none;
            min-width: 250px;
        }
        .main-tabs .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #f04f53 0%, #f5576c 100%);
        }
        .main-tabs .stTabs [data-baseweb="tab"]:hover {
            background: linear-gradient(135deg, #81C784 0%, #66BB6A 100%);
            transform: translateY(-2px);
            transition: all 0.3s ease;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Main tabs for Batting and Bowling analysis
    st.markdown('<div class="main-tabs">', unsafe_allow_html=True)
    main_tabs = st.tabs(["🏏 Batting Similarity", "⚾ Bowling Similarity"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    with main_tabs[0]:  # Batting Similarity Tab
        # Check if batting data exists
        if 'bat_df' not in st.session_state or st.session_state['bat_df'].empty:
            st.error("❌ Please upload batting data first from the Home page.")
            return

        bat_df = st.session_state['bat_df'].copy()
        
        # Convert Year to int to avoid comparison issues
        bat_df['Year'] = pd.to_numeric(bat_df['Year'], errors='coerce').fillna(0).astype(int)
        
        # Initialize session state for filters
        if 'similar_filter_state' not in st.session_state:
            st.session_state.similar_filter_state = {
                'match_format': ['All'],
                'bat_team': ['All'],
                'bowl_team': ['All'],
                'position': ['All'],
                'year': [int(bat_df['Year'].min()), int(bat_df['Year'].max())]
            }
    
        # Add position key if it doesn't exist (for existing sessions)
        if 'position' not in st.session_state.similar_filter_state:
            st.session_state.similar_filter_state['position'] = ['All']
    
        # Create filter section
        st.markdown('<div class="filter-card">', unsafe_allow_html=True)
        st.markdown("### 🎛️ Filter Options")
        
        # Get current filter selections
        selected_filters = {
            'Match_Format': st.session_state.similar_filter_state['match_format'] if 'All' not in st.session_state.similar_filter_state['match_format'] else [],
            'Bat_Team_y': st.session_state.similar_filter_state['bat_team'] if 'All' not in st.session_state.similar_filter_state['bat_team'] else [],
            'Bowl_Team_y': st.session_state.similar_filter_state['bowl_team'] if 'All' not in st.session_state.similar_filter_state['bowl_team'] else [],
            'Position': st.session_state.similar_filter_state['position'] if 'All' not in st.session_state.similar_filter_state['position'] else []
        }
    
        # Filter controls
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            available_formats = get_filtered_options(bat_df, 'Match_Format', 
                {k: v for k, v in selected_filters.items() if k != 'Match_Format' and 'All' not in v})
            match_format_choice = st.multiselect('Match Format:', 
                                              available_formats, 
                                              default=[fmt for fmt in st.session_state.similar_filter_state['match_format'] if fmt in available_formats])
            if match_format_choice != st.session_state.similar_filter_state['match_format']:
                st.session_state.similar_filter_state['match_format'] = match_format_choice
    
        with col2:
            available_bat_teams = get_filtered_options(bat_df, 'Bat_Team_y', 
                {k: v for k, v in selected_filters.items() if k != 'Bat_Team_y' and 'All' not in v})
            bat_team_choice = st.multiselect('Batting Team:', 
                                           available_bat_teams,
                                           default=[team for team in st.session_state.similar_filter_state['bat_team'] if team in available_bat_teams])
            if bat_team_choice != st.session_state.similar_filter_state['bat_team']:
                st.session_state.similar_filter_state['bat_team'] = bat_team_choice
    
        with col3:
            available_bowl_teams = get_filtered_options(bat_df, 'Bowl_Team_y', 
                {k: v for k, v in selected_filters.items() if k != 'Bowl_Team_y' and 'All' not in v})
            bowl_team_choice = st.multiselect('Bowling Team:', 
                                            available_bowl_teams,
                                            default=[team for team in st.session_state.similar_filter_state['bowl_team'] if team in available_bowl_teams])
            if bowl_team_choice != st.session_state.similar_filter_state['bowl_team']:
                st.session_state.similar_filter_state['bowl_team'] = bowl_team_choice
    
        with col4:
            # Check if Position column exists
            if 'Position' in bat_df.columns:
                available_positions = get_filtered_options(bat_df, 'Position', 
                    {k: v for k, v in selected_filters.items() if k != 'Position' and 'All' not in v})
                position_choice = st.multiselect('Batting Position:', 
                                               available_positions,
                                               default=[pos for pos in st.session_state.similar_filter_state['position'] if pos in available_positions])
                if position_choice != st.session_state.similar_filter_state['position']:
                    st.session_state.similar_filter_state['position'] = position_choice
            else:
                position_choice = ['All']  # Default to All if no Position column
                st.empty()  # Leave empty if no Position column
    
        with col5:
            # Get min/max years safely
            min_year = int(bat_df['Year'].min()) if not bat_df['Year'].empty else 2020
            max_year = int(bat_df['Year'].max()) if not bat_df['Year'].empty else 2025
            
            # Ensure the session state year range is within bounds
            if (st.session_state.similar_filter_state['year'][0] < min_year or 
                st.session_state.similar_filter_state['year'][1] > max_year):
                st.session_state.similar_filter_state['year'] = [min_year, max_year]
            
            year_range = st.slider('Year Range:', 
                                 min_year, 
                                 max_year,
                                 st.session_state.similar_filter_state['year'])
            if year_range != st.session_state.similar_filter_state['year']:
                st.session_state.similar_filter_state['year'] = year_range
    
        st.markdown('</div>', unsafe_allow_html=True)
    
        # Apply filters
        filtered_df = bat_df.copy()
        
        if 'All' not in match_format_choice and match_format_choice:
            filtered_df = filtered_df[filtered_df['Match_Format'].isin(match_format_choice)]
        
        if 'All' not in bat_team_choice and bat_team_choice:
            filtered_df = filtered_df[filtered_df['Bat_Team_y'].isin(bat_team_choice)]
            
        if 'All' not in bowl_team_choice and bowl_team_choice:
            filtered_df = filtered_df[filtered_df['Bowl_Team_y'].isin(bowl_team_choice)]
        
        # Apply position filter if Position column exists
        if 'Position' in bat_df.columns and 'All' not in position_choice and position_choice:
            filtered_df = filtered_df[filtered_df['Position'].isin(position_choice)]
        
        filtered_df = filtered_df[filtered_df['Year'].between(year_range[0], year_range[1])]
    
        if filtered_df.empty:
            st.warning("No data matches the current filter criteria.")
            return
    
        # Create tabs like in batview.py
        tabs = st.tabs(["Career"])
        
        # Career Stats Tab (similar to batview.py)
        with tabs[0]:
            # Get career stats for display
            career_stats = compute_career_stats(filtered_df)
            
            if career_stats.empty:
                st.warning("No career statistics available with current filters.")
                return
    
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">🏏 Career Statistics</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(career_stats, use_container_width=True, hide_index=True, 
                        column_config={"Name": st.column_config.Column("Name", pinned=True)})
    
            # Player Selection for Similarity Analysis
            st.markdown("### 👤 Select Reference Player")
            
            player_names = sorted(career_stats['Name'].unique())
            selected_player = st.selectbox("Choose a player to find similar players:", player_names)
            
            if selected_player:
                # Get reference player stats
                ref_stats = career_stats[career_stats['Name'] == selected_player].iloc[0]
                
                # Display reference player stats
                st.markdown(f"### 📊 Reference Player: **{selected_player}**")
                
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{ref_stats['Matches']}</p>
                        <p class="metric-label">Matches</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{ref_stats['Runs']}</p>
                        <p class="metric-label">Runs</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{ref_stats['Avg']:.2f}</p>
                        <p class="metric-label">Average</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{ref_stats['SR']:.2f}</p>
                        <p class="metric-label">Strike Rate</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col5:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{ref_stats['BPO']:.2f}</p>
                        <p class="metric-label">Balls Per Out</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col6:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{ref_stats['100PI']:.2f}%</p>
                        <p class="metric-label">100 Per Innings</p>
                    </div>
                    """, unsafe_allow_html=True)
    
                # Similarity calculation
                st.markdown("### 🎯 Similarity Criteria")
                
                # Mode selection
                similarity_mode = st.radio(
                    "Select Similarity Method:",
                    ["Tolerance Mode", "Distance Mode"],
                    index=0,
                    horizontal=True,
                    help="Tolerance: Filter by custom ranges | Distance: Find mathematically closest players"
                )
                
                st.markdown("---")
                
                if similarity_mode == "Tolerance Mode":
                    # Tolerance mode - existing sliders
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        avg_tolerance = st.slider("Average Tolerance (±)", 0.0, 20.0, 5.0, 0.5)
                    with col2:
                        sr_tolerance = st.slider("Strike Rate Tolerance (±)", 0.0, 30.0, 10.0, 1.0)
                    with col3:
                        bpo_tolerance = st.slider("Balls Per Out Tolerance (±)", 0.0, 20.0, 5.0, 0.5)
                    with col4:
                        min_matches = st.slider("Minimum Matches", 1, 100, 10, 1)
                    with col5:
                        pi100_tolerance = st.slider("100PI Tolerance (±)", 0.0, 20.0, 5.0, 0.1)
                
                else:  # Distance Mode
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        min_matches = st.slider("Minimum Matches", 1, 100, 10, 1)
                    with col2:
                        top_n = st.slider("Top N Similar Players", 3, 25, 10, 1)
                    with col3:
                        weight_preset = st.selectbox(
                            "Weight Preset",
                            ["Balanced", "Average Heavy", "Strike Rate Heavy"],
                            help="Balanced: Equal weights | Average Heavy: Emphasizes consistency | Strike Rate Heavy: Emphasizes aggression"
                        )
                    
                    # Weight configuration based on preset
                    if weight_preset == "Balanced":
                        weights = {'Avg': 0.25, 'SR': 0.25, 'BPO': 0.25, '100PI': 0.25}
                    elif weight_preset == "Average Heavy":
                        weights = {'Avg': 0.4, 'SR': 0.2, 'BPO': 0.2, '100PI': 0.2}
                    else:  # Strike Rate Heavy
                        weights = {'Avg': 0.2, 'SR': 0.4, 'BPO': 0.2, '100PI': 0.2}
                    
                    st.info(f"**Weights:** Average: {weights['Avg']:.0%} | Strike Rate: {weights['SR']:.0%} | BPO: {weights['BPO']:.0%} | 100PI: {weights['100PI']:.0%}")
    
                # Find similar players based on selected mode
                if similarity_mode == "Tolerance Mode":
                    # Tolerance mode - existing logic
                    similar_players = career_stats[
                        (career_stats['Name'] != selected_player) &
                        (career_stats['Matches'] >= min_matches) &
                        (abs(career_stats['Avg'] - ref_stats['Avg']) <= avg_tolerance) &
                        (abs(career_stats['SR'] - ref_stats['SR']) <= sr_tolerance) &
                        (abs(career_stats['BPO'] - ref_stats['BPO']) <= bpo_tolerance) &
                        (abs(career_stats['100PI'] - ref_stats['100PI']) <= pi100_tolerance)
                    ].copy()
    
                    # Calculate similarity score for tolerance mode
                    if not similar_players.empty:
                        similar_players['Avg_Diff'] = abs(similar_players['Avg'] - ref_stats['Avg'])
                        similar_players['SR_Diff'] = abs(similar_players['SR'] - ref_stats['SR'])
                        similar_players['BPO_Diff'] = abs(similar_players['BPO'] - ref_stats['BPO'])
                        similar_players['100PI_Diff'] = abs(similar_players['100PI'] - ref_stats['100PI'])
                        
                        # Calculate similarity score based on all 4 criteria (weighted)
                        similar_players['Similarity_Score'] = (
                            100 - (similar_players['Avg_Diff'] / avg_tolerance * 35) - 
                            (similar_players['SR_Diff'] / sr_tolerance * 25) -
                            (similar_players['BPO_Diff'] / bpo_tolerance * 25) -
                            (similar_players['100PI_Diff'] / pi100_tolerance * 15)
                        ).round(2)
                        similar_players = similar_players.sort_values('Similarity_Score', ascending=False)
                
                else:  # Distance Mode
                    # Filter eligible players
                    eligible_players = career_stats[
                        (career_stats['Name'] != selected_player) &
                        (career_stats['Matches'] >= min_matches)
                    ].copy()
                    
                    if not eligible_players.empty:
                        # Calculate z-scores for normalization
                        metrics = ['Avg', 'SR', 'BPO', '100PI']
                        z_scores = {}
                        
                        for metric in metrics:
                            mean_val = eligible_players[metric].mean()
                            std_val = eligible_players[metric].std()
                            if std_val > 0:
                                z_scores[f'{metric}_z'] = (eligible_players[metric] - mean_val) / std_val
                                ref_z = (ref_stats[metric] - mean_val) / std_val
                            else:
                                z_scores[f'{metric}_z'] = eligible_players[metric] * 0  # All zeros if no variation
                                ref_z = 0
                            z_scores[f'ref_{metric}_z'] = ref_z
                        
                        # Calculate weighted Euclidean distance
                        eligible_players['Distance'] = np.sqrt(
                            weights['Avg'] * (z_scores['Avg_z'] - z_scores['ref_Avg_z'])**2 +
                            weights['SR'] * (z_scores['SR_z'] - z_scores['ref_SR_z'])**2 +
                            weights['BPO'] * (z_scores['BPO_z'] - z_scores['ref_BPO_z'])**2 +
                            weights['100PI'] * (z_scores['100PI_z'] - z_scores['ref_100PI_z'])**2
                        )
                        
                        # Convert distance to similarity score (0-100 scale)
                        eligible_players['Similarity_Score'] = (100 * (1 / (1 + eligible_players['Distance']))).round(2)
                        
                        # Get top N most similar players
                        similar_players = eligible_players.nsmallest(top_n, 'Distance').copy()
                        similar_players = similar_players.sort_values('Similarity_Score', ascending=False)
                    else:
                        similar_players = pd.DataFrame()
    
                # Display results
                if similarity_mode == "Tolerance Mode":
                    st.markdown(f"### 🔍 Players Similar to {selected_player} (Tolerance Mode)")
                else:
                    st.markdown(f"### 🔍 Top {top_n if not similar_players.empty else 0} Most Similar Players to {selected_player} (Distance Mode)")
                
                if similar_players.empty:
                    if similarity_mode == "Tolerance Mode":
                        st.info("No similar players found with the current criteria. Try adjusting the tolerance values.")
                    else:
                        st.info("No eligible players found. Try lowering the minimum matches requirement.")
                else:
                    if similarity_mode == "Tolerance Mode":
                        st.write(f"Found **{len(similar_players)}** players within tolerance ranges:")
                    else:
                        st.write(f"**Top {len(similar_players)}** most similar players using weighted distance analysis:")
                    
                    # Display similar players table
                    display_cols = ['Name', 'Matches', 'Runs', 'Avg', 'SR', 'BPO', '100PI', '50s', '100s', '150s', '200s', 'Similarity_Score']
                    similar_display = similar_players[display_cols].copy()
                    
                    # Create progress bar column (convert to 0-1 scale for proper display)
                    similar_display['Similarity %'] = similar_display['Similarity_Score'] / 100  # Convert to 0-1 scale for progress bar
                    
                    # Reorder columns 
                    display_cols_final = ['Name', 'Matches', 'Runs', 'Avg', 'SR', 'BPO', '100PI', '50s', '100s', '150s', '200s', 'Similarity %']
                    similar_display = similar_display[display_cols_final]
                    
                    st.dataframe(similar_display, use_container_width=True, hide_index=True,
                                column_config={
                                    "Name": st.column_config.Column("Name", pinned=True),
                                    "Similarity %": st.column_config.ProgressColumn("Similarity %", min_value=0.0, max_value=1.0)
                                })
    
                    # Advanced Visualizations
                    if len(similar_players) >= 1:
                        st.markdown("---")
                        
                        # Custom CSS for centered tabs
                        st.markdown("""
                            <style>
                            .stTabs [data-baseweb="tab-list"] {
                                gap: 8px;
                                justify-content: center;
                                margin: 0 auto;
                                max-width: 100%;
                            }
                            .stTabs [data-baseweb="tab"] {
                                flex: 1;
                                text-align: center;
                                font-weight: bold;
                                min-width: 200px;
                                padding: 10px 16px;
                                border-radius: 8px;
                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                color: white;
                                border: none;
                            }
                            .stTabs [data-baseweb="tab"][aria-selected="true"] {
                                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                                color: white;
                            }
                            .stTabs [data-baseweb="tab"]:hover {
                                background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                                color: #333;
                            }
                            </style>
                        """, unsafe_allow_html=True)
                        
                        # Tab layout for different visualizations
                        viz_tabs = st.tabs(["📈 Similarity Chart", "🎯 Radar Chart", "📊 Metric Differences", "🔗 Correlation Heatmap"])
                        
                        with viz_tabs[0]:  # Similarity Visualization
                            st.markdown("#### 📈 Average vs Strike Rate Analysis")
                            st.write("Interactive scatter plot showing similarity relationships")
                            
                            # Create scatter plot
                            fig = go.Figure()
                            
                            # Add reference player
                            fig.add_trace(go.Scatter(
                                x=[ref_stats['Avg']],
                                y=[ref_stats['SR']],
                                mode='markers+text',
                                text=[selected_player],
                                textposition='top center',
                                marker=dict(size=15, color='red', symbol='star'),
                                name='Reference Player',
                                hovertemplate=(
                                    f"<b>{selected_player}</b> (Reference)<br><br>"
                                    f"Average: {ref_stats['Avg']:.2f}<br>"
                                    f"Strike Rate: {ref_stats['SR']:.2f}<br>"
                                    f"Runs: {ref_stats['Runs']}<br>"
                                    f"Matches: {ref_stats['Matches']}<br>"
                                    "<extra></extra>"
                                )
                            ))
                            
                            # Add similar players
                            fig.add_trace(go.Scatter(
                                x=similar_players['Avg'],
                                y=similar_players['SR'],
                                mode='markers+text',
                                text=similar_players['Name'],
                                textposition='top center',
                                marker=dict(
                                    size=10, 
                                    color=similar_players['Similarity_Score'],
                                    colorscale='Viridis',
                                    colorbar=dict(title="Similarity %"),
                                    showscale=True
                                ),
                                name='Similar Players',
                                hovertemplate=(
                                    "<b>%{text}</b><br><br>"
                                    "Average: %{x:.2f}<br>"
                                    "Strike Rate: %{y:.2f}<br>"
                                    "Similarity: %{marker.color:.1f}%<br>"
                                    "<extra></extra>"
                                )
                            ))
                            
                            # Update layout
                            fig.update_layout(
                                title="Average vs Strike Rate - Similarity Analysis",
                                xaxis_title="Batting Average",
                                yaxis_title="Strike Rate",
                                height=600,
                                font=dict(size=12),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                showlegend=True
                            )
                            
                            # Add mode-specific annotations and boundaries
                            if similarity_mode == "Tolerance Mode":
                                # Add tolerance boundaries (showing Average vs Strike Rate tolerance zone)
                                fig.add_shape(
                                    type="rect",
                                    x0=ref_stats['Avg'] - avg_tolerance,
                                    x1=ref_stats['Avg'] + avg_tolerance,
                                    y0=ref_stats['SR'] - sr_tolerance,
                                    y1=ref_stats['SR'] + sr_tolerance,
                                    line=dict(color="rgba(255,0,0,0.5)", width=2, dash="dash"),
                                    fillcolor="rgba(255,0,0,0.1)"
                                )
                                
                                # Add annotation about multi-criteria filtering
                                fig.add_annotation(
                                    text=f"Tolerance Ranges: Avg(±{avg_tolerance}), SR(±{sr_tolerance}), BPO(±{bpo_tolerance}), 100PI(±{pi100_tolerance})",
                                    xref="paper", yref="paper",
                                    x=0.5, y=1.02, xanchor='center', yanchor='bottom',
                                    showarrow=False,
                                    font=dict(size=10, color="gray"),
                                    bgcolor="rgba(255,255,255,0.8)",
                                    bordercolor="gray",
                                    borderwidth=1
                                )
                            else:  # Distance Mode
                                # Add annotation about distance-based selection
                                weight_text = f"Weights: Avg({weights['Avg']:.0%}), SR({weights['SR']:.0%}), BPO({weights['BPO']:.0%}), 100PI({weights['100PI']:.0%})"
                                fig.add_annotation(
                                    text=f"Top {top_n} by Weighted Distance | {weight_text}",
                                    xref="paper", yref="paper",
                                    x=0.5, y=1.02, xanchor='center', yanchor='bottom',
                                    showarrow=False,
                                    font=dict(size=10, color="gray"),
                                    bgcolor="rgba(255,255,255,0.8)",
                                    bordercolor="gray",
                                    borderwidth=1
                                )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with viz_tabs[1]:  # Radar Chart
                            st.markdown("#### 🏆 Skill Profile Comparison")
                            st.write("Compare player profiles across all similarity metrics")
                            
                            # Prepare data for radar chart
                            metrics = ['Avg', 'SR', 'BPO', '100PI']
                            
                            # Get top 3 similar players for radar
                            top_players = similar_players.head(3) if len(similar_players) >= 3 else similar_players
                            
                            # Normalize metrics (0-100 scale relative to all eligible players)
                            if similarity_mode == "Distance Mode":
                                eligible_for_radar = career_stats[career_stats['Matches'] >= min_matches]
                            else:
                                eligible_for_radar = career_stats[career_stats['Matches'] >= min_matches]
                            
                            normalized_data = {}
                            for metric in metrics:
                                min_val = eligible_for_radar[metric].min()
                                max_val = eligible_for_radar[metric].max()
                                
                                # Normalize reference player
                                ref_norm = ((ref_stats[metric] - min_val) / (max_val - min_val)) * 100 if max_val != min_val else 50
                                normalized_data[f'ref_{metric}'] = ref_norm
                                
                                # Normalize similar players
                                for idx, row in top_players.iterrows():
                                    player_norm = ((row[metric] - min_val) / (max_val - min_val)) * 100 if max_val != min_val else 50
                                    if row['Name'] not in normalized_data:
                                        normalized_data[row['Name']] = {}
                                    normalized_data[row['Name']][metric] = player_norm
                            
                            # Create radar chart
                            radar_fig = go.Figure()
                            
                            # Add reference player
                            ref_values = [normalized_data[f'ref_{metric}'] for metric in metrics]
                            ref_values += [ref_values[0]]  # Close the polygon
                            radar_fig.add_trace(go.Scatterpolar(
                                r=ref_values,
                                theta=metrics + [metrics[0]],
                                fill='toself',
                                name=f'{selected_player} (Reference)',
                                line_color='red',
                                fillcolor='rgba(255,0,0,0.1)'
                            ))
                            
                            # Add similar players with different colors
                            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                            for i, (idx, row) in enumerate(top_players.iterrows()):
                                if i < len(colors):
                                    values = [normalized_data[row['Name']][metric] for metric in metrics]
                                    values += [values[0]]  # Close the polygon
                                    radar_fig.add_trace(go.Scatterpolar(
                                        r=values,
                                        theta=metrics + [metrics[0]],
                                        fill='toself',
                                        name=f"{row['Name']} ({row['Similarity_Score']:.1f}%)",
                                        line_color=colors[i],
                                        fillcolor='rgba(31,119,180,0.1)' if i == 0 else 'rgba(255,127,14,0.1)' if i == 1 else 'rgba(44,160,44,0.1)'
                                    ))
                            
                            radar_fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 100],
                                        tickmode='array',
                                        tickvals=[20, 40, 60, 80, 100],
                                        ticktext=['20%', '40%', '60%', '80%', '100%']
                                    )),
                                showlegend=True,
                                title="Normalized Skill Profile (0-100% relative to player pool)",
                                height=500,
                                font=dict(size=12)
                            )
                            
                            st.plotly_chart(radar_fig, use_container_width=True)
                            
                            # Show actual values table
                            with st.expander("📋 View Actual Metric Values"):
                                radar_data = []
                                radar_data.append({
                                    'Player': f"{selected_player} (Reference)",
                                    'Average': f"{ref_stats['Avg']:.2f}",
                                    'Strike Rate': f"{ref_stats['SR']:.2f}",
                                    'Balls Per Out': f"{ref_stats['BPO']:.2f}",
                                    '100PI': f"{ref_stats['100PI']:.2f}",
                                    'Similarity': "100.0%"
                                })
                                
                                for idx, row in top_players.iterrows():
                                    radar_data.append({
                                        'Player': row['Name'],
                                        'Average': f"{row['Avg']:.2f}",
                                        'Strike Rate': f"{row['SR']:.2f}",
                                        'Balls Per Out': f"{row['BPO']:.2f}",
                                        '100PI': f"{row['100PI']:.2f}",
                                        'Similarity': f"{row['Similarity_Score']:.1f}%"
                                    })
                                
                                st.dataframe(pd.DataFrame(radar_data), use_container_width=True, hide_index=True)
                        
                        with viz_tabs[2]:  # Feature Difference Bar Chart
                            st.markdown("#### 🎯 Metric Differences vs Reference Player")
                            st.write("Green bars = higher than reference | Red bars = lower than reference")
                            
                            # Prepare difference data
                            top_diff_players = similar_players.head(5) if len(similar_players) >= 5 else similar_players
                            
                            diff_data = []
                            for idx, row in top_diff_players.iterrows():
                                for metric in metrics:
                                    diff_value = row[metric] - ref_stats[metric]
                                    diff_data.append({
                                        'Player': row['Name'],
                                        'Metric': metric,
                                        'Difference': diff_value,
                                        'Color': 'Positive' if diff_value >= 0 else 'Negative',
                                        'Similarity': row['Similarity_Score']
                                    })
                            
                            diff_df = pd.DataFrame(diff_data)
                            
                            # Create horizontal bar chart
                            bar_fig = px.bar(
                                diff_df, 
                                x='Difference', 
                                y='Player', 
                                color='Color',
                                facet_col='Metric',
                                color_discrete_map={'Positive': '#2ca02c', 'Negative': '#d62728'},
                                title="Metric Differences vs Reference Player",
                                labels={'Difference': 'Difference from Reference', 'Player': 'Similar Players'}
                            )
                            
                            bar_fig.update_layout(
                                height=400,
                                showlegend=True,
                                font=dict(size=10)
                            )
                            bar_fig.update_xaxes(title_text="Difference", zeroline=True, zerolinecolor='black')
                            bar_fig.update_yaxes(title_text="")
                            
                            st.plotly_chart(bar_fig, use_container_width=True)
                            
                            # Show difference table
                            with st.expander("📊 View Numeric Differences"):
                                pivot_diff = diff_df.pivot(index='Player', columns='Metric', values='Difference')
                                pivot_diff = pivot_diff.round(2)
                                
                                # Add similarity scores
                                similarity_dict = dict(zip(top_diff_players['Name'], top_diff_players['Similarity_Score']))
                                pivot_diff['Similarity %'] = [similarity_dict[player] for player in pivot_diff.index]
                                
                                # Reorder columns
                                pivot_diff = pivot_diff[['Avg', 'SR', 'BPO', '100PI', 'Similarity %']]
                                
                                st.dataframe(
                                    pivot_diff.style.format({
                                        'Avg': '{:.2f}', 'SR': '{:.2f}', 'BPO': '{:.2f}', 
                                        '100PI': '{:.2f}', 'Similarity %': '{:.1f}%'
                                    }).background_gradient(subset=['Avg', 'SR', 'BPO', '100PI'], cmap='RdYlGn'),
                                    use_container_width=True
                                )
                        
                        with viz_tabs[3]:  # Correlation Heatmap
                            st.markdown("#### 🔗 Metric Correlation Analysis")
                            st.write("Understanding how similarity metrics relate to each other")
                            
                            # Prepare correlation data using all eligible players
                            if similarity_mode == "Distance Mode":
                                corr_data = career_stats[career_stats['Matches'] >= min_matches][metrics]
                            else:
                                corr_data = career_stats[career_stats['Matches'] >= min_matches][metrics]
                            
                            # Calculate Spearman correlation
                            correlation_matrix = corr_data.corr(method='spearman')
                            
                            # Create heatmap using plotly
                            heatmap_fig = px.imshow(
                                correlation_matrix,
                                text_auto=True,
                                aspect="auto",
                                color_continuous_scale="RdBu_r",
                                title="Spearman Correlation Matrix of Similarity Metrics",
                                labels=dict(color="Correlation Coefficient")
                            )
                            
                            heatmap_fig.update_layout(
                                height=400,
                                font=dict(size=12),
                                xaxis_title="",
                                yaxis_title=""
                            )
                            
                            st.plotly_chart(heatmap_fig, use_container_width=True)
                            
                            # Correlation insights
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**🔍 Key Correlations:**")
                                # Find strongest correlations (excluding diagonal)
                                corr_flat = correlation_matrix.values
                                np.fill_diagonal(corr_flat, 0)  # Remove diagonal
                                
                                # Get indices of max absolute correlation
                                max_corr_idx = np.unravel_index(np.argmax(np.abs(corr_flat)), corr_flat.shape)
                                max_corr_val = corr_flat[max_corr_idx]
                                max_corr_pair = (correlation_matrix.index[max_corr_idx[0]], correlation_matrix.columns[max_corr_idx[1]])
                                
                                # Get indices of min correlation (most negative)
                                min_corr_idx = np.unravel_index(np.argmin(corr_flat), corr_flat.shape)
                                min_corr_val = corr_flat[min_corr_idx]
                                min_corr_pair = (correlation_matrix.index[min_corr_idx[0]], correlation_matrix.columns[min_corr_idx[1]])
                                
                                st.write(f"**Strongest:** {max_corr_pair[0]} ↔ {max_corr_pair[1]} ({max_corr_val:.3f})")
                                st.write(f"**Weakest:** {min_corr_pair[0]} ↔ {min_corr_pair[1]} ({min_corr_val:.3f})")
                            
                            with col2:
                                st.markdown("**💡 Interpretation:**")
                                st.write("• **1.0**: Perfect positive correlation")
                                st.write("• **0.0**: No correlation") 
                                st.write("• **-1.0**: Perfect negative correlation")
                                st.write("• **|r| > 0.7**: Strong relationship")
        
    with main_tabs[1]:  # Bowling Similarity Tab
        # Check if bowling data exists
        if 'bowl_df' not in st.session_state or st.session_state['bowl_df'].empty:
            st.error("❌ Please upload bowling data first from the Home page.")
            return

        bowl_df = st.session_state['bowl_df'].copy()
        
        # Convert Date to Year for filtering
        if 'Date' in bowl_df.columns:
            try:
                bowl_df['Year'] = pd.to_datetime(bowl_df['Date'], errors='coerce').dt.year
                # Fill any NaN years with a default value
                bowl_df['Year'] = bowl_df['Year'].fillna(2024).astype(int)
            except:
                bowl_df['Year'] = 2024  # Default year if date parsing fails
        else:
            # If no Date column, try to extract year from filename or use default
            bowl_df['Year'] = 2024  # Default year if no date information
        
        # Initialize session state for bowling filters
        if 'bowl_similar_filter_state' not in st.session_state:
            min_year = int(bowl_df['Year'].min()) if not bowl_df['Year'].empty else 2020
            max_year = int(bowl_df['Year'].max()) if not bowl_df['Year'].empty else 2025
            st.session_state.bowl_similar_filter_state = {
                'match_format': ['All'],
                'bowl_team': ['All'],
                'bat_team': ['All'],
                'position': ['All'],
                'year': [min_year, max_year]
            }
        elif 'position' not in st.session_state.bowl_similar_filter_state:
            st.session_state.bowl_similar_filter_state['position'] = ['All']
        
        # Create bowling filter section
        st.markdown('<div class="filter-card">', unsafe_allow_html=True)
        st.markdown("### 🎛️ Bowling Filter Options")
        
        # Get current bowling filter selections
        bowl_selected_filters = {
            'Match_Format': st.session_state.bowl_similar_filter_state['match_format'] if 'All' not in st.session_state.bowl_similar_filter_state['match_format'] else [],
            'Bowl_Team': st.session_state.bowl_similar_filter_state['bowl_team'] if 'All' not in st.session_state.bowl_similar_filter_state['bowl_team'] else [],
            'Bat_Team': st.session_state.bowl_similar_filter_state['bat_team'] if 'All' not in st.session_state.bowl_similar_filter_state['bat_team'] else [],
            'Position': st.session_state.bowl_similar_filter_state['position'] if 'All' not in st.session_state.bowl_similar_filter_state['position'] else []
        }

        bowl_position_choice = st.session_state.bowl_similar_filter_state['position']

        # Bowling filter controls
        bowl_col1, bowl_col2, bowl_col3, bowl_col4, bowl_col5 = st.columns(5)
        
        with bowl_col1:
            bowl_available_formats = get_filtered_options(bowl_df, 'Match_Format', 
                {k: v for k, v in bowl_selected_filters.items() if k != 'Match_Format' and v})
            bowl_match_format_choice = st.multiselect('Match Format:', 
                                              bowl_available_formats, 
                                              default=[fmt for fmt in st.session_state.bowl_similar_filter_state['match_format'] if fmt in bowl_available_formats],
                                              key='bowl_format')
            if bowl_match_format_choice != st.session_state.bowl_similar_filter_state['match_format']:
                st.session_state.bowl_similar_filter_state['match_format'] = bowl_match_format_choice

        with bowl_col2:
            bowl_available_bowl_teams = get_filtered_options(bowl_df, 'Bowl_Team', 
                {k: v for k, v in bowl_selected_filters.items() if k != 'Bowl_Team' and v})
            bowl_bowl_team_choice = st.multiselect('Bowling Team:', 
                                           bowl_available_bowl_teams,
                                           default=[team for team in st.session_state.bowl_similar_filter_state['bowl_team'] if team in bowl_available_bowl_teams],
                                           key='bowl_bowl_team')
            if bowl_bowl_team_choice != st.session_state.bowl_similar_filter_state['bowl_team']:
                st.session_state.bowl_similar_filter_state['bowl_team'] = bowl_bowl_team_choice

        with bowl_col3:
            bowl_available_bat_teams = get_filtered_options(bowl_df, 'Bat_Team', 
                {k: v for k, v in bowl_selected_filters.items() if k != 'Bat_Team' and v})
            bowl_bat_team_choice = st.multiselect('Batting Team:', 
                                          bowl_available_bat_teams,
                                          default=[team for team in st.session_state.bowl_similar_filter_state['bat_team'] if team in bowl_available_bat_teams],
                                          key='bowl_bat_team')
            if bowl_bat_team_choice != st.session_state.bowl_similar_filter_state['bat_team']:
                st.session_state.bowl_similar_filter_state['bat_team'] = bowl_bat_team_choice

        with bowl_col4:
            if 'Position' in bowl_df.columns:
                bowl_available_positions = get_filtered_options(
                    bowl_df,
                    'Position',
                    {k: v for k, v in bowl_selected_filters.items() if k != 'Position' and v}
                )
                bowl_position_choice = st.multiselect(
                    'Bowling Position:',
                    bowl_available_positions,
                    default=[pos for pos in st.session_state.bowl_similar_filter_state['position'] if pos in bowl_available_positions],
                    key='bowl_position'
                )
                if bowl_position_choice != st.session_state.bowl_similar_filter_state['position']:
                    st.session_state.bowl_similar_filter_state['position'] = bowl_position_choice
            else:
                bowl_position_choice = ['All']
                st.empty()

        with bowl_col5:
            # Get min/max years safely for bowling
            bowl_min_year = int(bowl_df['Year'].min()) if not bowl_df['Year'].empty else 2020
            bowl_max_year = int(bowl_df['Year'].max()) if not bowl_df['Year'].empty else 2025
            
            # Ensure the session state year range is within bounds
            if (st.session_state.bowl_similar_filter_state['year'][0] < bowl_min_year or 
                st.session_state.bowl_similar_filter_state['year'][1] > bowl_max_year):
                st.session_state.bowl_similar_filter_state['year'] = [bowl_min_year, bowl_max_year]
            
            bowl_year_range = st.slider('Year Range:', 
                                 bowl_min_year, 
                                 bowl_max_year,
                                 st.session_state.bowl_similar_filter_state['year'],
                                 key='bowl_year')
            if bowl_year_range != st.session_state.bowl_similar_filter_state['year']:
                st.session_state.bowl_similar_filter_state['year'] = bowl_year_range

        st.markdown('</div>', unsafe_allow_html=True)

        # Apply bowling filters
        bowl_filtered_df = bowl_df.copy()
        
        if 'All' not in bowl_match_format_choice and bowl_match_format_choice:
            bowl_filtered_df = bowl_filtered_df[bowl_filtered_df['Match_Format'].isin(bowl_match_format_choice)]
        
        if 'All' not in bowl_bowl_team_choice and bowl_bowl_team_choice:
            bowl_filtered_df = bowl_filtered_df[bowl_filtered_df['Bowl_Team'].isin(bowl_bowl_team_choice)]
            
        if 'All' not in bowl_bat_team_choice and bowl_bat_team_choice:
            bowl_filtered_df = bowl_filtered_df[bowl_filtered_df['Bat_Team'].isin(bowl_bat_team_choice)]

        if 'Position' in bowl_filtered_df.columns and 'All' not in bowl_position_choice and bowl_position_choice:
            bowl_filtered_df = bowl_filtered_df[bowl_filtered_df['Position'].isin(bowl_position_choice)]
        
        bowl_filtered_df = bowl_filtered_df[bowl_filtered_df['Year'].between(bowl_year_range[0], bowl_year_range[1])]

        # Check if we have data after filtering
        if bowl_filtered_df.empty:
            st.warning("No bowling statistics available with current filters.")
            return

        # Compute bowling career statistics using bowlview.py function
        def compute_bowl_career_stats_local(df):
            if df.empty:
                return pd.DataFrame()
            
            try:
                # Basic aggregations
                bowl_career_stats = df.groupby('Name').agg({
                    'File Name': 'nunique',  # Matches
                    'Bowler_Balls': 'sum',   # Balls
                    'Bowler_Runs': 'sum',    # Runs
                    'Bowler_Wkts': 'sum',    # Wickets
                    'Maidens': 'sum' if 'Maidens' in df.columns else lambda x: 0
                }).reset_index()
                
                bowl_career_stats.columns = ['Name', 'Matches', 'Balls', 'Runs', 'Wickets', 'Maidens']
                
                # Calculate bowling statistics
                bowl_career_stats['Overs'] = ((bowl_career_stats['Balls'] / 6) + (bowl_career_stats['Balls'] % 6) / 10).round(1)
                bowl_career_stats['Avg'] = (bowl_career_stats['Runs'] / bowl_career_stats['Wickets'].replace(0, np.nan)).round(2).fillna(0)
                bowl_career_stats['Strike Rate'] = (bowl_career_stats['Balls'] / bowl_career_stats['Wickets'].replace(0, np.nan)).round(2).fillna(0)
                bowl_career_stats['Economy Rate'] = (bowl_career_stats['Runs'] / (bowl_career_stats['Balls'] / 6).replace(0, np.nan)).round(2).fillna(0)
                bowl_career_stats['WPM'] = (bowl_career_stats['Wickets'] / bowl_career_stats['Matches'].replace(0, np.nan)).round(2).fillna(0)
                
                # Calculate 5W and 10W
                five_wickets = df[df['Bowler_Wkts'] >= 5].groupby('Name').size().reset_index(name='5W')
                bowl_career_stats = bowl_career_stats.merge(five_wickets, on='Name', how='left')
                bowl_career_stats['5W'] = bowl_career_stats['5W'].fillna(0)
                
                match_wickets = df.groupby(['Name', 'File Name'])['Bowler_Wkts'].sum().reset_index()
                ten_wickets = match_wickets[match_wickets['Bowler_Wkts'] >= 10].groupby('Name').size().reset_index(name='10W')
                bowl_career_stats = bowl_career_stats.merge(ten_wickets, on='Name', how='left')
                bowl_career_stats['10W'] = bowl_career_stats['10W'].fillna(0)
                
                # Player of the Match
                if 'Player_of_the_Match' in df.columns:
                    pom_counts = df[df['Player_of_the_Match'] == df['Name']].groupby('Name')['File Name'].nunique().reset_index(name='POM')
                    bowl_career_stats = bowl_career_stats.merge(pom_counts, on='Name', how='left')
                    bowl_career_stats['POM'] = bowl_career_stats['POM'].fillna(0)
                else:
                    bowl_career_stats['POM'] = 0
                
                # Calculate POM Per Match
                bowl_career_stats['POM Per Match'] = (bowl_career_stats['POM'] / bowl_career_stats['Matches'].replace(0, 1)*100).round(2)
                
                return bowl_career_stats.sort_values('Wickets', ascending=False)
            except Exception as e:
                st.warning(f"Could not compute bowling career stats: {e}")
                return pd.DataFrame()

        # Calculate bowling career statistics
        bowl_career_stats = compute_bowl_career_stats_local(bowl_filtered_df)

        if bowl_career_stats.empty:
            st.warning("No bowling career statistics available with current filters.")
            return

        # Display bowling career statistics tab
        bowl_tabs = st.tabs(["Career"])
        
        with bowl_tabs[0]:
            st.markdown("""
                <div class="section-header">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">🏏 Bowling Career Statistics</h3>
                </div>
                """, unsafe_allow_html=True)
            
            # Display bowling career statistics dataframe
            st.dataframe(bowl_career_stats[['Name', 'Matches', 'Overs', 'Runs', 'Wickets', 'Avg', 'Strike Rate', 'Economy Rate', 'WPM', '5W', '10W', 'POM', 'POM Per Match']], 
                        use_container_width=True, hide_index=True,
                        column_config={
                            "Name": st.column_config.Column("Name", pinned=True)
                        })

            # Add similarity analysis section
            st.markdown("---")
            st.markdown("### 🔍 Select Reference Bowler")
            
            # Bowler selection
            bowl_selected_player = st.selectbox('Choose a bowler to find similar players:', 
                                        bowl_career_stats['Name'].tolist(),
                                        key='bowl_player_select')

            if bowl_selected_player:
                bowl_ref_stats = bowl_career_stats[bowl_career_stats['Name'] == bowl_selected_player].iloc[0]

                # Reference bowler display
                st.markdown(f"""
                <div class="section-header">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">📊 Reference Bowler: {bowl_selected_player}</h3>
                </div>
                """, unsafe_allow_html=True)

                bowl_col1, bowl_col2, bowl_col3, bowl_col4, bowl_col5, bowl_col6 = st.columns(6)
                with bowl_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{bowl_ref_stats['Matches']}</p>
                        <p class="metric-label">Matches</p>
                    </div>
                    """, unsafe_allow_html=True)
                with bowl_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{bowl_ref_stats['Wickets']}</p>
                        <p class="metric-label">Wickets</p>
                    </div>
                    """, unsafe_allow_html=True)
                with bowl_col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{bowl_ref_stats['Avg']:.2f}</p>
                        <p class="metric-label">Average</p>
                    </div>
                    """, unsafe_allow_html=True)
                with bowl_col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{bowl_ref_stats['Strike Rate']:.2f}</p>
                        <p class="metric-label">Strike Rate</p>
                    </div>
                    """, unsafe_allow_html=True)
                with bowl_col5:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{bowl_ref_stats['Economy Rate']:.2f}</p>
                        <p class="metric-label">Economy Rate</p>
                    </div>
                    """, unsafe_allow_html=True)
                with bowl_col6:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{bowl_ref_stats['WPM']:.2f}</p>
                        <p class="metric-label">Wickets per Match</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Bowling similarity calculation
                st.markdown("### 🎯 Bowling Similarity Criteria")
                
                bowling_similarity_mode = st.radio(
                    "Select Bowling Similarity Method:",
                    ["Tolerance Mode", "Distance Mode"],
                    index=0,
                    horizontal=True,
                    help="Tolerance: Filter by custom ranges | Distance: Find mathematically closest bowlers",
                    key='bowl_similarity_mode'
                )
                
                st.markdown("---")
                
                if bowling_similarity_mode == "Tolerance Mode":
                    # Tolerance mode - bowling sliders
                    bowl_col1, bowl_col2, bowl_col3, bowl_col4 = st.columns(4)
                    with bowl_col1:
                        bowl_avg_tolerance = st.slider("Average Tolerance (±)", 0.0, 10.0, 2.0, 0.1, key='bowl_avg_tol')
                    with bowl_col2:
                        bowl_sr_tolerance = st.slider("Strike Rate Tolerance (±)", 0.0, 30.0, 10.0, 1.0, key='bowl_sr_tol')
                    with bowl_col3:
                        bowl_econ_tolerance = st.slider("Economy Rate Tolerance (±)", 0.0, 3.0, 0.5, 0.1, key='bowl_econ_tol')
                    with bowl_col4:
                        bowl_min_matches = st.slider("Minimum Matches", 1, 50, 5, 1, key='bowl_min_matches')
                
                else:  # Distance Mode
                    bowl_col1, bowl_col2, bowl_col3 = st.columns(3)
                    with bowl_col1:
                        bowl_min_matches = st.slider("Minimum Matches", 1, 50, 5, 1, key='bowl_min_matches_dist')
                    with bowl_col2:
                        bowl_top_n = st.slider("Top N Similar Bowlers", 3, 25, 10, 1, key='bowl_top_n')
                    with bowl_col3:
                        bowl_weight_preset = st.selectbox(
                            "Weight Preset",
                            ["Balanced", "Average Heavy", "Economy Heavy", "Strike Rate Heavy"],
                            help="Balanced: Equal weights | Average Heavy: Emphasizes consistency | Economy Heavy: Emphasizes economy | Strike Rate Heavy: Emphasizes wicket-taking ability",
                            key='bowl_weights'
                        )
                    
                    # Weight configuration based on preset
                    if bowl_weight_preset == "Balanced":
                        bowl_weights = {'Avg': 0.33, 'Strike Rate': 0.33, 'Economy Rate': 0.34}
                    elif bowl_weight_preset == "Average Heavy":
                        bowl_weights = {'Avg': 0.5, 'Strike Rate': 0.25, 'Economy Rate': 0.25}
                    elif bowl_weight_preset == "Economy Heavy":
                        bowl_weights = {'Avg': 0.25, 'Strike Rate': 0.25, 'Economy Rate': 0.5}
                    else:  # Strike Rate Heavy
                        bowl_weights = {'Avg': 0.25, 'Strike Rate': 0.5, 'Economy Rate': 0.25}
                    
                    st.info(f"**Weights:** Average: {bowl_weights['Avg']:.0%} | Strike Rate: {bowl_weights['Strike Rate']:.0%} | Economy Rate: {bowl_weights['Economy Rate']:.0%}")

                # Find similar bowlers based on selected mode
                if bowling_similarity_mode == "Tolerance Mode":
                    # Tolerance mode - existing logic
                    bowl_similar_players = bowl_career_stats[
                        (bowl_career_stats['Name'] != bowl_selected_player) &
                        (bowl_career_stats['Matches'] >= bowl_min_matches) &
                        (abs(bowl_career_stats['Avg'] - bowl_ref_stats['Avg']) <= bowl_avg_tolerance) &
                        (abs(bowl_career_stats['Strike Rate'] - bowl_ref_stats['Strike Rate']) <= bowl_sr_tolerance) &
                        (abs(bowl_career_stats['Economy Rate'] - bowl_ref_stats['Economy Rate']) <= bowl_econ_tolerance)
                    ].copy()

                    # Calculate similarity score for tolerance mode
                    if not bowl_similar_players.empty:
                        bowl_similar_players['Avg_Diff'] = abs(bowl_similar_players['Avg'] - bowl_ref_stats['Avg'])
                        bowl_similar_players['SR_Diff'] = abs(bowl_similar_players['Strike Rate'] - bowl_ref_stats['Strike Rate'])
                        bowl_similar_players['Econ_Diff'] = abs(bowl_similar_players['Economy Rate'] - bowl_ref_stats['Economy Rate'])
                        
                        # Calculate similarity score based on 3 bowling criteria
                        bowl_similar_players['Similarity_Score'] = (
                            100 - (bowl_similar_players['Avg_Diff'] / bowl_avg_tolerance * 35) - 
                            (bowl_similar_players['SR_Diff'] / bowl_sr_tolerance * 32.5) -
                            (bowl_similar_players['Econ_Diff'] / bowl_econ_tolerance * 32.5)
                        ).round(2)
                        bowl_similar_players = bowl_similar_players.sort_values('Similarity_Score', ascending=False)
                
                else:  # Distance Mode
                    # Filter eligible bowlers
                    bowl_eligible_players = bowl_career_stats[
                        (bowl_career_stats['Name'] != bowl_selected_player) &
                        (bowl_career_stats['Matches'] >= bowl_min_matches)
                    ].copy()
                    
                    if not bowl_eligible_players.empty:
                        # Calculate z-scores for normalization
                        bowl_metrics = ['Avg', 'Strike Rate', 'Economy Rate']
                        bowl_z_scores = {}
                        
                        for metric in bowl_metrics:
                            mean_val = bowl_eligible_players[metric].mean()
                            std_val = bowl_eligible_players[metric].std()
                            if std_val > 0:
                                bowl_z_scores[f'{metric}_z'] = (bowl_eligible_players[metric] - mean_val) / std_val
                                bowl_ref_z = (bowl_ref_stats[metric] - mean_val) / std_val
                            else:
                                bowl_z_scores[f'{metric}_z'] = bowl_eligible_players[metric] * 0  # All zeros if no variation
                                bowl_ref_z = 0
                            bowl_z_scores[f'ref_{metric}_z'] = bowl_ref_z
                        
                        # Calculate weighted Euclidean distance
                        bowl_eligible_players['Distance'] = np.sqrt(
                            bowl_weights['Avg'] * (bowl_z_scores['Avg_z'] - bowl_z_scores['ref_Avg_z'])**2 +
                            bowl_weights['Strike Rate'] * (bowl_z_scores['Strike Rate_z'] - bowl_z_scores['ref_Strike Rate_z'])**2 +
                            bowl_weights['Economy Rate'] * (bowl_z_scores['Economy Rate_z'] - bowl_z_scores['ref_Economy Rate_z'])**2
                        )
                        
                        # Convert distance to similarity score (0-100 scale)
                        bowl_eligible_players['Similarity_Score'] = (100 * (1 / (1 + bowl_eligible_players['Distance']))).round(2)
                        
                        # Get top N most similar bowlers
                        bowl_similar_players = bowl_eligible_players.nsmallest(bowl_top_n, 'Distance').copy()
                        bowl_similar_players = bowl_similar_players.sort_values('Similarity_Score', ascending=False)
                    else:
                        bowl_similar_players = pd.DataFrame()

                # Display bowling results
                if bowling_similarity_mode == "Tolerance Mode":
                    st.markdown(f"### 🔍 Bowlers Similar to {bowl_selected_player} (Tolerance Mode)")
                else:
                    st.markdown(f"### 🔍 Top {bowl_top_n if not bowl_similar_players.empty else 0} Most Similar Bowlers to {bowl_selected_player} (Distance Mode)")
                
                if bowl_similar_players.empty:
                    if bowling_similarity_mode == "Tolerance Mode":
                        st.info("No similar bowlers found with the current criteria. Try adjusting the tolerance values.")
                    else:
                        st.info("No eligible bowlers found. Try lowering the minimum matches requirement.")
                else:
                    if bowling_similarity_mode == "Tolerance Mode":
                        st.write(f"Found **{len(bowl_similar_players)}** bowlers within tolerance ranges:")
                    else:
                        st.write(f"**Top {len(bowl_similar_players)}** most similar bowlers using weighted distance analysis:")

                    # Display similar bowlers table
                    bowl_display_cols = ['Name', 'Matches', 'Wickets', 'Avg', 'Strike Rate', 'Economy Rate', '5W', '10W', 'WPM', 'Similarity_Score']
                    bowl_similar_display = bowl_similar_players[bowl_display_cols].copy()
                    
                    # Create progress bar column (convert to 0-1 scale for proper display)
                    bowl_similar_display['Similarity %'] = bowl_similar_display['Similarity_Score'] / 100  # Convert to 0-1 scale for progress bar
                    
                    # Reorder columns 
                    bowl_display_cols_final = ['Name', 'Matches', 'Wickets', 'Avg', 'Strike Rate', 'Economy Rate', '5W', '10W', 'WPM', 'Similarity %']
                    bowl_similar_display = bowl_similar_display[bowl_display_cols_final]
                    
                    st.dataframe(bowl_similar_display, use_container_width=True, hide_index=True,
                                column_config={
                                    "Name": st.column_config.Column("Name", pinned=True),
                                    "Similarity %": st.column_config.ProgressColumn("Similarity %", min_value=0.0, max_value=1.0)
                                })

                    # Advanced Bowling Visualizations
                    if len(bowl_similar_players) >= 1:
                        st.markdown("---")

                        # Reuse custom CSS to center visualization tabs
                        st.markdown("""
                            <style>
                            .stTabs [data-baseweb="tab-list"] {
                                gap: 8px;
                                justify-content: center;
                                margin: 0 auto;
                                max-width: 100%;
                            }
                            .stTabs [data-baseweb="tab"] {
                                flex: 1;
                                text-align: center;
                                font-weight: bold;
                                min-width: 200px;
                                padding: 10px 16px;
                                border-radius: 8px;
                                background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
                                color: white;
                                border: none;
                            }
                            .stTabs [data-baseweb="tab"][aria-selected="true"] {
                                background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
                                color: white;
                            }
                            .stTabs [data-baseweb="tab"]:hover {
                                background: linear-gradient(135deg, #8fd3f4 0%, #84fab0 100%);
                                color: #333;
                            }
                            </style>
                        """, unsafe_allow_html=True)

                        bowl_viz_tabs = st.tabs(["🎯 Similarity Map", "🎛️ Radar Chart", "📊 Metric Differences", "🔗 Correlation Heatmap"])

                        # Shared bowling metrics
                        bowl_metrics = ['Avg', 'Strike Rate', 'Economy Rate', 'WPM']

                        with bowl_viz_tabs[0]:
                            st.markdown("#### 🎯 Average vs Strike Rate")
                            st.write("Reference bowler highlighted; bubble colour shows similarity percentage")

                            fig = go.Figure()

                            fig.add_trace(go.Scatter(
                                x=[bowl_ref_stats['Avg']],
                                y=[bowl_ref_stats['Strike Rate']],
                                mode='markers+text',
                                text=[bowl_selected_player],
                                textposition='top center',
                                marker=dict(size=15, color='red', symbol='star'),
                                name='Reference Bowler',
                                hovertemplate=(
                                    f"<b>{bowl_selected_player}</b> (Reference)<br><br>"
                                    f"Average: {bowl_ref_stats['Avg']:.2f}<br>"
                                    f"Strike Rate: {bowl_ref_stats['Strike Rate']:.2f}<br>"
                                    f"Economy: {bowl_ref_stats['Economy Rate']:.2f}<br>"
                                    f"Wickets: {bowl_ref_stats['Wickets']}<br>"
                                    f"Matches: {bowl_ref_stats['Matches']}<br>"
                                    "<extra></extra>"
                                )
                            ))

                            fig.add_trace(go.Scatter(
                                x=bowl_similar_players['Avg'],
                                y=bowl_similar_players['Strike Rate'],
                                mode='markers+text',
                                text=bowl_similar_players['Name'],
                                textposition='top center',
                                marker=dict(
                                    size=10,
                                    color=bowl_similar_players['Similarity_Score'],
                                    colorscale='Viridis',
                                    colorbar=dict(title="Similarity %"),
                                    showscale=True
                                ),
                                name='Similar Bowlers',
                                hovertemplate=(
                                    "<b>%{text}</b><br><br>"
                                    "Average: %{x:.2f}<br>"
                                    "Strike Rate: %{y:.2f}<br>"
                                    "Economy: %{customdata[0]:.2f}<br>"
                                    "Similarity: %{marker.color:.1f}%<br>"
                                    "<extra></extra>"
                                ),
                                customdata=np.column_stack([
                                    bowl_similar_players['Economy Rate']
                                ])
                            ))

                            fig.update_layout(
                                title="Average vs Strike Rate - Bowling Similarity",
                                xaxis_title="Bowling Average",
                                yaxis_title="Strike Rate",
                                height=600,
                                font=dict(size=12),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                showlegend=True
                            )

                            if bowling_similarity_mode == "Tolerance Mode":
                                fig.add_shape(
                                    type="rect",
                                    x0=bowl_ref_stats['Avg'] - bowl_avg_tolerance,
                                    x1=bowl_ref_stats['Avg'] + bowl_avg_tolerance,
                                    y0=bowl_ref_stats['Strike Rate'] - bowl_sr_tolerance,
                                    y1=bowl_ref_stats['Strike Rate'] + bowl_sr_tolerance,
                                    line=dict(color="rgba(255,0,0,0.5)", width=2, dash="dash"),
                                    fillcolor="rgba(255,0,0,0.1)"
                                )
                                fig.add_annotation(
                                    text=(
                                        f"Tolerance Ranges: Avg(±{bowl_avg_tolerance}), "
                                        f"Strike Rate(±{bowl_sr_tolerance}), Economy(±{bowl_econ_tolerance})"
                                    ),
                                    xref="paper", yref="paper",
                                    x=0.5, y=1.02, xanchor='center', yanchor='bottom',
                                    showarrow=False,
                                    font=dict(size=10, color="gray"),
                                    bgcolor="rgba(255,255,255,0.8)",
                                    bordercolor="gray",
                                    borderwidth=1
                                )
                            else:
                                weight_text = (
                                    f"Weights: Avg({bowl_weights['Avg']:.0%}), "
                                    f"Strike Rate({bowl_weights['Strike Rate']:.0%}), "
                                    f"Economy({bowl_weights['Economy Rate']:.0%})"
                                )
                                fig.add_annotation(
                                    text=f"Top {len(bowl_similar_players)} by Weighted Distance | {weight_text}",
                                    xref="paper", yref="paper",
                                    x=0.5, y=1.02, xanchor='center', yanchor='bottom',
                                    showarrow=False,
                                    font=dict(size=10, color="gray"),
                                    bgcolor="rgba(255,255,255,0.8)",
                                    bordercolor="gray",
                                    borderwidth=1
                                )

                            st.plotly_chart(fig, use_container_width=True)

                        with bowl_viz_tabs[1]:
                            st.markdown("#### 🎛️ Skill Profile Comparison")
                            st.write("Normalized radar chart across core bowling metrics")

                            top_bowlers = bowl_similar_players.head(3) if len(bowl_similar_players) >= 3 else bowl_similar_players

                            eligible_bowlers = bowl_career_stats[bowl_career_stats['Matches'] >= bowl_min_matches]
                            normalized = {}

                            for metric in bowl_metrics:
                                min_val = eligible_bowlers[metric].min()
                                max_val = eligible_bowlers[metric].max()

                                ref_norm = ((bowl_ref_stats[metric] - min_val) / (max_val - min_val) * 100) if max_val != min_val else 50
                                normalized[f'ref_{metric}'] = ref_norm

                                for _, row in top_bowlers.iterrows():
                                    player_norm = ((row[metric] - min_val) / (max_val - min_val) * 100) if max_val != min_val else 50
                                    normalized.setdefault(row['Name'], {})[metric] = player_norm

                            radar_fig = go.Figure()

                            ref_values = [normalized[f'ref_{metric}'] for metric in bowl_metrics]
                            ref_values += [ref_values[0]]
                            radar_fig.add_trace(go.Scatterpolar(
                                r=ref_values,
                                theta=bowl_metrics + [bowl_metrics[0]],
                                fill='toself',
                                name=f"{bowl_selected_player} (Reference)",
                                line_color='red',
                                fillcolor='rgba(255,0,0,0.1)'
                            ))

                            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                            for i, (_, row) in enumerate(top_bowlers.iterrows()):
                                values = [normalized[row['Name']][metric] for metric in bowl_metrics]
                                values += [values[0]]
                                radar_fig.add_trace(go.Scatterpolar(
                                    r=values,
                                    theta=bowl_metrics + [bowl_metrics[0]],
                                    fill='toself',
                                    name=f"{row['Name']} ({row['Similarity_Score']:.1f}%)",
                                    line_color=colors[i % len(colors)],
                                    fillcolor=f"rgba(31,119,180,{0.15 + i*0.05})"
                                ))

                            radar_fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 100],
                                        tickmode='array',
                                        tickvals=[20, 40, 60, 80, 100],
                                        ticktext=['20%', '40%', '60%', '80%', '100%']
                                    )),
                                showlegend=True,
                                title="Normalized Bowling Metrics (0-100%)",
                                height=500,
                                font=dict(size=12)
                            )

                            st.plotly_chart(radar_fig, use_container_width=True)

                            with st.expander("📋 View Actual Bowling Metrics"):
                                radar_rows = [{
                                    'Player': f"{bowl_selected_player} (Reference)",
                                    'Average': f"{bowl_ref_stats['Avg']:.2f}",
                                    'Strike Rate': f"{bowl_ref_stats['Strike Rate']:.2f}",
                                    'Economy Rate': f"{bowl_ref_stats['Economy Rate']:.2f}",
                                    'Wickets per Match': f"{bowl_ref_stats['WPM']:.2f}",
                                    'Similarity': "100.0%"
                                }]

                                for _, row in top_bowlers.iterrows():
                                    radar_rows.append({
                                        'Player': row['Name'],
                                        'Average': f"{row['Avg']:.2f}",
                                        'Strike Rate': f"{row['Strike Rate']:.2f}",
                                        'Economy Rate': f"{row['Economy Rate']:.2f}",
                                        'Wickets per Match': f"{row['WPM']:.2f}",
                                        'Similarity': f"{row['Similarity_Score']:.1f}%"
                                    })

                                st.dataframe(pd.DataFrame(radar_rows), use_container_width=True, hide_index=True)

                        with bowl_viz_tabs[2]:
                            st.markdown("#### 📊 Differences vs Reference Bowler")
                            st.write("Positive values favour the similar bowler; negative values favour the reference")

                            diff_players = bowl_similar_players.head(5) if len(bowl_similar_players) >= 5 else bowl_similar_players

                            diff_records = []
                            for _, row in diff_players.iterrows():
                                for metric in bowl_metrics:
                                    diff_val = row[metric] - bowl_ref_stats[metric]
                                    diff_records.append({
                                        'Player': row['Name'],
                                        'Metric': metric,
                                        'Difference': diff_val,
                                        'Color': 'Positive' if diff_val >= 0 else 'Negative',
                                        'Similarity': row['Similarity_Score']
                                    })

                            diff_df = pd.DataFrame(diff_records)

                            diff_fig = px.bar(
                                diff_df,
                                x='Difference',
                                y='Player',
                                color='Color',
                                facet_col='Metric',
                                color_discrete_map={'Positive': '#2ca02c', 'Negative': '#d62728'},
                                title="Metric Differences vs Reference Bowler",
                                labels={'Difference': 'Difference from Reference', 'Player': 'Similar Bowlers'}
                            )

                            diff_fig.update_layout(height=420, showlegend=True, font=dict(size=10))
                            diff_fig.update_xaxes(title_text="Difference", zeroline=True, zerolinecolor='black')
                            diff_fig.update_yaxes(title_text="")

                            st.plotly_chart(diff_fig, use_container_width=True)

                            with st.expander("📊 Numeric Differences"):
                                pivot = diff_df.pivot(index='Player', columns='Metric', values='Difference').round(2)
                                similarity_lookup = dict(zip(diff_players['Name'], diff_players['Similarity_Score']))
                                pivot['Similarity %'] = [similarity_lookup[name] for name in pivot.index]
                                pivot = pivot[['Avg', 'Strike Rate', 'Economy Rate', 'WPM', 'Similarity %']]

                                st.dataframe(
                                    pivot.style.format({
                                        'Avg': '{:.2f}',
                                        'Strike Rate': '{:.2f}',
                                        'Economy Rate': '{:.2f}',
                                        'WPM': '{:.2f}',
                                        'Similarity %': '{:.1f}%'
                                    }).background_gradient(subset=['Avg', 'Strike Rate', 'Economy Rate', 'WPM'], cmap='RdYlGn'),
                                    use_container_width=True
                                )

                        with bowl_viz_tabs[3]:
                            st.markdown("#### 🔗 Bowling Metric Correlations")
                            st.write("Spearman correlation across eligible bowlers")

                            corr_source = bowl_career_stats[bowl_career_stats['Matches'] >= bowl_min_matches]
                            corr_matrix = corr_source[bowl_metrics[:-1]].corr(method='spearman')

                            heatmap = px.imshow(
                                corr_matrix,
                                text_auto='.2f',
                                color_continuous_scale='RdBu',
                                zmin=-1,
                                zmax=1,
                                title="Spearman Correlation of Bowling Metrics"
                            )

                            heatmap.update_layout(height=420, font=dict(size=11))
                            st.plotly_chart(heatmap, use_container_width=True)

                            strongest = corr_matrix.stack().sort_values(key=lambda s: abs(s), ascending=False)
                            if not strongest.empty:
                                # remove self correlations
                                strongest = strongest[strongest.index.get_level_values(0) != strongest.index.get_level_values(1)]
                            if not strongest.empty:
                                top_pair = strongest.index[0]
                                st.info(f"Strongest relationship: {top_pair[0]} ↔ {top_pair[1]} ({strongest.iloc[0]:.2f})")

# Call the function to display the similar players view
display_similar_players_view()
