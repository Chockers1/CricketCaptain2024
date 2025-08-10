# Imports and Setup
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

# Modern CSS for beautiful UI - Full styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #f04f53 0%, #f5576c 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    .uploadedFile {
        border-radius: 10px;
        border: 2px dashed #f04f53;
        padding: 20px;
    }
    
    .modern-card {
        background: linear-gradient(135deg, #fff0f1 0%, #f8f9fa 100%);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(240,79,83,0.1);
        margin: 20px 0;
        text-align: center;
    }
    
    .filter-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    
    .result-banner {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        color: white;
        font-weight: bold;
        font-size: 18px;
    }
    
    .success-banner {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        color: white;
        font-weight: bold;
    }
    
    .warning-banner {
        background: linear-gradient(135deg, #ff9500 0%, #ffad33 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        color: white;
        font-weight: bold;
    }
    
    .stats-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 30px 0 20px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        color: white;
        font-weight: bold;
        font-size: 24px;
    }
    
    .filter-section {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    h1, h2, h3 {
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    table { 
        color: black; 
        width: 100%; 
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    thead tr th {
        background-color: #f04f53 !important;
        color: white !important;
        font-weight: bold;
        padding: 12px;
    }
    tbody tr:nth-child(even) { 
        background-color: #f8f9fa; 
    }
    tbody tr:nth-child(odd) { 
        background-color: white; 
    }
    tbody tr:hover {
        background-color: #fff0f1;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        border-radius: 8px;
        border: 2px solid #f0f2f6;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within,
    .stMultiSelect > div > div:focus-within {
        border-color: #f04f53;
        box-shadow: 0 0 0 3px rgba(240,79,83,0.1);
    }
    
    .stSlider [data-baseweb="slider-track"] {
        background: linear-gradient(90deg, #f04f53 0%, #f5576c 100%);
    }
    
    .stSlider p { 
        color: #f04f53 !important; 
    }
    
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #fff0f1 0%, #f8f9fa 100%);
        border: 1px solid #f04f53;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .plot-container {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

def display_ar_view():
    # Modern page header
    st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 30px; border-radius: 20px; margin: 20px 0; text-align: center;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
            <h1 style="color: white; margin: 0; font-size: 2.5rem;">
                🏏 All Rounder Statistics
            </h1>
            <p style="color: white; margin: 10px 0 0 0; font-size: 1.2rem; opacity: 0.9;">
                Comprehensive batting and bowling performance analysis
            </p>
        </div>
    """, unsafe_allow_html=True)

    if 'bat_df' in st.session_state and 'bowl_df' in st.session_state:
        # Make copies of original dataframes
        bat_df = st.session_state['bat_df'].copy()
        bowl_df = st.session_state['bowl_df'].copy()

        # Check if only one scorecard is loaded
        unique_matches_bat = bat_df['File Name'].nunique()
        unique_matches_bowl = bowl_df['File Name'].nunique()
        if unique_matches_bat <= 1 or unique_matches_bowl <= 1:
            st.markdown("""
                <div class="warning-banner">
                    ⚠️ Please upload more than 1 scorecard to use the all-rounder statistics view effectively. 
                    With only one match loaded, statistical analysis and comparisons are limited.
                </div>
            """, unsafe_allow_html=True)
            return

        # Create Year columns from Date with safer date parsing
        try:
            # Try to parse dates with the correct format
            bat_df['Date'] = pd.to_datetime(bat_df['Date'], format='%d %b %Y', errors='coerce')
            bowl_df['Date'] = pd.to_datetime(bowl_df['Date'], format='%d %b %Y', errors='coerce')
        except:
            # If that fails, try with dayfirst=True
            bat_df['Date'] = pd.to_datetime(bat_df['Date'], dayfirst=True, errors='coerce')
            bowl_df['Date'] = pd.to_datetime(bowl_df['Date'], dayfirst=True, errors='coerce')

        # Extract years from the parsed dates
        bat_df['Year'] = bat_df['Date'].dt.year
        bowl_df['Year'] = bowl_df['Date'].dt.year

        # Convert Year columns to integers and handle any NaN values
        bat_df['Year'] = pd.to_numeric(bat_df['Year'], errors='coerce').fillna(0).astype(int)
        bowl_df['Year'] = pd.to_numeric(bowl_df['Year'], errors='coerce').fillna(0).astype(int)

        # Get filter options (exclude year 0 from the years list)
        names = ['All'] + sorted(bat_df['Name'].unique().tolist())
        bat_teams = ['All'] + sorted(bat_df['Bat_Team_y'].unique().tolist())
        bowl_teams = ['All'] + sorted(bat_df['Bowl_Team_y'].unique().tolist())
        match_formats = ['All'] + sorted(bat_df['Match_Format'].unique().tolist())
        years = sorted(list(set(bat_df['Year'].unique()) | set(bowl_df['Year'].unique())))
        years = [year for year in years if year != 0]  # Remove year 0 if present

        # Check if we have any valid years
        if not years:
            st.markdown("""
                <div class="warning-banner">
                    ❌ No valid dates found in the data.
                </div>
            """, unsafe_allow_html=True)
            return

        # Create first row of filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            name_choice = st.multiselect('👤 Name:', names, default='All')
        with col2:
            bat_team_choice = st.multiselect('🏏 Batting Team:', bat_teams, default='All')
        with col3:
            bowl_team_choice = st.multiselect('🏏 Bowling Team:', bowl_teams, default='All')
        with col4:
            match_format_choice = st.multiselect('🏆 Format:', match_formats, default='All')

        # First apply basic filters to get initial stats
        filtered_bat_df = bat_df.copy()
        filtered_bowl_df = bowl_df.copy()
        
        # Apply name and format filters
        if 'All' not in name_choice:
            filtered_bat_df = filtered_bat_df[filtered_bat_df['Name'].isin(name_choice)]
            filtered_bowl_df = filtered_bowl_df[filtered_bowl_df['Name'].isin(name_choice)]
        if 'All' not in match_format_choice:
            filtered_bat_df = filtered_bat_df[filtered_bat_df['Match_Format'].isin(match_format_choice)]
            filtered_bowl_df = filtered_bowl_df[filtered_bowl_df['Match_Format'].isin(match_format_choice)]
        if 'All' not in bat_team_choice:
            filtered_bat_df = filtered_bat_df[filtered_bat_df['Bat_Team_y'].isin(bat_team_choice)]
        if 'All' not in bowl_team_choice:
            filtered_bat_df = filtered_bat_df[filtered_bat_df['Bowl_Team_y'].isin(bowl_team_choice)]

        # Calculate initial career stats for max values
        initial_stats = pd.merge(
            filtered_bat_df.groupby('Name').agg({
                'File Name': 'nunique',
                'Runs': 'sum',
                'Out': 'sum',
                'Balls': 'sum'
            }).reset_index(),
            filtered_bowl_df.groupby('Name').agg({
                'Bowler_Runs': 'sum',
                'Bowler_Wkts': 'sum',
                'Bowler_Balls': 'sum'
            }).reset_index(),
            on='Name',
            how='outer'
        ).fillna(0)

        # Calculate max values for filters
        max_matches = int(initial_stats['File Name'].max())
        max_runs = int(initial_stats['Runs'].max())
        max_wickets = int(initial_stats['Bowler_Wkts'].max())

        # Calculate averages for max values
        initial_stats['Bat Avg'] = (initial_stats['Runs'] / initial_stats['Out'].replace(0, np.inf)).round(2)
        initial_stats['Bat SR'] = ((initial_stats['Runs'] / initial_stats['Balls'].replace(0, np.inf)) * 100).round(2)
        initial_stats['Bowl Avg'] = np.where(
            initial_stats['Bowler_Wkts'] > 0,
            (initial_stats['Bowler_Runs'] / initial_stats['Bowler_Wkts']).round(2),
            np.inf
        )
        initial_stats['Bowl SR'] = np.where(
            initial_stats['Bowler_Wkts'] > 0,
            (initial_stats['Bowler_Balls'] / initial_stats['Bowler_Wkts']).round(2),
            np.inf
        )

        # Replace infinities with NaN for max calculations
        initial_stats = initial_stats.replace([np.inf, -np.inf], np.nan)

        max_bat_avg = float(initial_stats['Bat Avg'].max())
        max_bat_sr = float(initial_stats['Bat SR'].max())
        max_bowl_avg = float(initial_stats['Bowl Avg'].max())
        max_bowl_sr = float(initial_stats['Bowl SR'].max())

        # Create range filter columns with modern styling
        st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
        col5, col6, col7, col8, col9, col10, col11, col12 = st.columns(8)

        with col5:
            st.markdown("<p style='text-align: center; font-weight: bold; color: #f04f53;'>📅 Choose Year:</p>", unsafe_allow_html=True)
            if len(years) == 1:
                st.markdown(f"<p style='text-align: center; font-size: 1.2rem; color: #2c3e50;'>{years[0]}</p>", unsafe_allow_html=True)
                year_choice = (years[0], years[0])
            else:
                year_choice = st.slider('Year Range', 
                    min_value=min(years),
                    max_value=max(years),
                    value=(min(years), max(years)))

        with col6:
            st.markdown("<p style='text-align: center; font-weight: bold; color: #f04f53;'>🎯 Matches Range</p>", unsafe_allow_html=True)
            matches_range = st.slider('Matches', 
                                    min_value=1, 
                                    max_value=max_matches, 
                                    value=(1, max_matches))

        with col7:
            st.markdown("<p style='text-align: center; font-weight: bold; color: #f04f53;'>🏃 Runs Range</p>", unsafe_allow_html=True)
            runs_range = st.slider('Runs', 
                                min_value=1, 
                                max_value=max_runs, 
                                value=(1, max_runs))

        with col8:
            st.markdown("<p style='text-align: center; font-weight: bold; color: #f04f53;'>🎳 Wickets Range</p>", unsafe_allow_html=True)
            wickets_range = st.slider('Wickets', 
                                    min_value=0, 
                                    max_value=max_wickets, 
                                    value=(0, max_wickets))

        with col9:
            st.markdown("<p style='text-align: center; font-weight: bold; color: #f04f53;'>📊 Batting Average</p>", unsafe_allow_html=True)
            bat_avg_range = st.slider('Batting Avg', 
                                    min_value=0.0, 
                                    max_value=max_bat_avg, 
                                    value=(0.0, max_bat_avg))

        with col10:
            st.markdown("<p style='text-align: center; font-weight: bold; color: #f04f53;'>📈 Bowling Average</p>", unsafe_allow_html=True)
            bowl_avg_range = st.slider('Bowling Avg', 
                                    min_value=0.0, 
                                    max_value=max_bowl_avg, 
                                    value=(0.0, max_bowl_avg))

        with col11:
            st.markdown("<p style='text-align: center; font-weight: bold; color: #f04f53;'>⚡ Batting SR</p>", unsafe_allow_html=True)
            bat_sr_range = st.slider('Batting SR', 
                                    min_value=0.0, 
                                    max_value=max_bat_sr, 
                                    value=(0.0, max_bat_sr))

        with col12:
            st.markdown("<p style='text-align: center; font-weight: bold; color: #f04f53;'>💨 Bowling SR</p>", unsafe_allow_html=True)
            bowl_sr_range = st.slider('Bowling SR', 
                                    min_value=0.0, 
                                    max_value=max_bowl_sr, 
                                    value=(0.0, max_bowl_sr))

        # Apply year filter
        filtered_bat_df = filtered_bat_df[
            filtered_bat_df['Year'].between(year_choice[0], year_choice[1])
        ]
        filtered_bowl_df = filtered_bowl_df[
            filtered_bowl_df['Year'].between(year_choice[0], year_choice[1])
        ]

        # Ensure categorical-safe comparison for POM (cast to string)
        if 'Player_of_the_Match' in filtered_bat_df.columns:
            filtered_bat_df['Player_of_the_Match'] = filtered_bat_df['Player_of_the_Match'].astype(str)
        if 'Name' in filtered_bat_df.columns:
            filtered_bat_df['Name'] = filtered_bat_df['Name'].astype(str)

        # Get POM count after all filters
        pom_count = filtered_bat_df[
            filtered_bat_df['Player_of_the_Match'] == filtered_bat_df['Name']
        ].groupby('Name').agg(
            POM=('File Name', 'nunique')
        ).reset_index()

        # Calculate final career stats
        filtered_df = pd.merge(
            filtered_bat_df,
            filtered_bowl_df,
            on=['File Name', 'Name', 'Innings'],
            how='outer',
            suffixes=('_bat', '_bowl')
        )

        career_stats = filtered_df.groupby('Name').agg({
            'File Name': 'nunique',
            'Batted': 'sum',
            'Out': 'sum',
            'Balls': 'sum',
            'Runs': 'sum',
            '50s': 'sum',
            '100s': 'sum',
            'Bowler_Balls': 'sum',
            'Bowler_Runs': 'sum',
            'Bowler_Wkts': 'sum',
            '5Ws': 'sum',
            '10Ws': 'sum'
        }).reset_index()

        career_stats.columns = ['Name', 'Matches', 'Inns', 'Outs', 'Balls', 'Runs', 
                              'Fifties', 'Hundreds', 'Bowl_Balls', 'Bowl_Runs', 
                              'Wickets', 'FiveWickets', 'TenWickets']

        # Add POM to career stats
        career_stats = career_stats.merge(pom_count, on='Name', how='left')
        career_stats['POM'] = career_stats['POM'].fillna(0).astype(int)

        # Calculate career stat metrics
        career_stats['Bat Avg'] = (career_stats['Runs'] / career_stats['Outs']).round(2)
        career_stats['Bat SR'] = ((career_stats['Runs'] / career_stats['Balls']) * 100).round(2)
        career_stats['Overs'] = (career_stats['Bowl_Balls'] // 6) + (career_stats['Bowl_Balls'] % 6) / 10
        career_stats['Bowl Avg'] = np.where(
            career_stats['Wickets'] > 0,
            (career_stats['Bowl_Runs'] / career_stats['Wickets']).round(2),
            np.inf
        )
        career_stats['Econ'] = np.where(
            career_stats['Overs'] > 0,
            (career_stats['Bowl_Runs'] / career_stats['Overs']).round(2),
            0
        )
        career_stats['Bowl SR'] = np.where(
            career_stats['Wickets'] > 0,
            (career_stats['Bowl_Balls'] / career_stats['Wickets']).round(2),
            np.inf
        )

        # Apply all statistical filters
        filtered_stats = career_stats[
            (career_stats['Matches'].between(matches_range[0], matches_range[1])) &
            (career_stats['Runs'].between(runs_range[0], runs_range[1])) &
            (career_stats['Wickets'].between(wickets_range[0], wickets_range[1])) &
            (career_stats['Bat Avg'].between(bat_avg_range[0], bat_avg_range[1])) &
            (career_stats['Bat SR'].between(bat_sr_range[0], bat_sr_range[1])) &
            ((career_stats['Bowl SR'].between(bowl_sr_range[0], bowl_sr_range[1])) | 
             (career_stats['Wickets'] == 0)) &
            ((career_stats['Bowl Avg'].between(bowl_avg_range[0], bowl_avg_range[1])) |
             (career_stats['Wickets'] == 0))
        ]

        # Clean up display columns
        filtered_stats = filtered_stats.drop(columns=['Balls', 'Bowl_Balls', 'TenWickets', 'Inns', 'Outs'])
        filtered_stats = filtered_stats[[ 
            'Name', 'Matches', 'Runs', 'Bat Avg', 'Bat SR', 
            'Fifties', 'Hundreds', 'Overs',
            'Bowl_Runs', 'Wickets', 'FiveWickets', 
            'Bowl Avg', 'Econ', 'Bowl SR', 'POM'
        ]]
        filtered_stats = filtered_stats.replace([np.inf, -np.inf], np.nan)

        # Display Career Statistics with modern header
        st.markdown("""
            <div style="background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); 
                        padding: 20px; border-radius: 15px; text-align: center; margin: 30px 0 20px 0;
                        box-shadow: 0 8px 25px rgba(0,0,0,0.15); color: white; font-weight: bold; font-size: 24px;">
                📊 Career Statistics
            </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(filtered_stats, use_container_width=True, hide_index=True)
###-------------------------------------SEASON STATS----------------------------------------###
        # First get separate batting and bowling season stats
        batting_season_stats = filtered_bat_df.groupby(['Name', 'Year']).agg({
            'File Name': 'nunique',
            'Batted': 'sum',
            'Out': 'sum',
            'Balls': 'sum',
            'Runs': 'sum',
            '50s': 'sum',
            '100s': 'sum'
        }).reset_index()

        # Calculate bowling season stats
        bowling_season_stats = filtered_bowl_df.groupby(['Name', 'Year']).agg({
            'Bowler_Balls': 'sum',
            'Bowler_Runs': 'sum',
            'Bowler_Wkts': 'sum',
            '5Ws': 'sum',
            '10Ws': 'sum'
        }).reset_index()

        # Merge batting and bowling season stats
        season_stats = batting_season_stats.merge(
            bowling_season_stats, 
            on=['Name', 'Year'], 
            how='outer'
        )

        # Get POM count by season
        season_pom_count = filtered_bat_df[
            filtered_bat_df['Player_of_the_Match'] == filtered_bat_df['Name']
        ].groupby(['Name', 'Year']).agg(
            POM=('File Name', 'nunique')
        ).reset_index()

        # Merge POM count with season stats
        season_stats = season_stats.merge(season_pom_count, on=['Name', 'Year'], how='left')
        season_stats['POM'] = season_stats['POM'].fillna(0).astype(int)

        # Fill NaN values with 0 for numerical columns
        numeric_columns = ['File Name', 'Batted', 'Out', 'Balls', 'Runs', '50s', '100s',
                         'Bowler_Balls', 'Bowler_Runs', 'Bowler_Wkts', '5Ws', '10Ws']
        season_stats[numeric_columns] = season_stats[numeric_columns].fillna(0)

        # Rename columns for clarity
        season_stats = season_stats.rename(columns={
            'File Name': 'Matches',
            'Batted': 'Inns',
            'Out': 'Outs',
            '50s': 'Fifties',
            '100s': 'Hundreds',
            'Bowler_Balls': 'Bowl_Balls',
            'Bowler_Runs': 'Bowl_Runs',
            'Bowler_Wkts': 'Wickets',
            '5Ws': 'FiveWickets',
            '10Ws': 'TenWickets'
        })

        # Calculate additional metrics
        season_stats['Bat Avg'] = (season_stats['Runs'] / season_stats['Outs']).round(2)
        season_stats['Bat SR'] = ((season_stats['Runs'] / season_stats['Balls']) * 100).round(2)
        season_stats['Overs'] = (season_stats['Bowl_Balls'] // 6) + (season_stats['Bowl_Balls'] % 6) / 10
        
        # Handle bowling averages and strike rates
        season_stats['Bowl Avg'] = np.where(
            season_stats['Wickets'] > 0,
            (season_stats['Bowl_Runs'] / season_stats['Wickets']).round(2),
            np.inf
        )
        
        season_stats['Econ'] = np.where(
            season_stats['Overs'] > 0,
            (season_stats['Bowl_Runs'] / season_stats['Overs']).round(2),
            0
        )
        
        season_stats['Bowl SR'] = np.where(
            season_stats['Wickets'] > 0,
            (season_stats['Bowl_Balls'] / season_stats['Wickets']).round(2),
            np.inf
        )

        # Apply filters to season stats
        filtered_season_stats = season_stats[
            (season_stats['Matches'].between(matches_range[0], matches_range[1])) &
            (season_stats['Runs'].between(runs_range[0], runs_range[1])) &
            (season_stats['Wickets'].between(wickets_range[0], wickets_range[1])) &
            (season_stats['Bat Avg'].between(bat_avg_range[0], bat_avg_range[1])) &
            (season_stats['Bat SR'].between(bat_sr_range[0], bat_sr_range[1])) &
            ((season_stats['Bowl SR'].between(bowl_sr_range[0], bowl_sr_range[1])) | 
             (season_stats['Wickets'] == 0)) &
            ((season_stats['Bowl Avg'].between(bowl_avg_range[0], bowl_avg_range[1])) |
             (season_stats['Wickets'] == 0))
        ]

        # Clean up and order columns
        filtered_season_stats = filtered_season_stats.drop(
            columns=['Balls', 'Bowl_Balls', 'TenWickets', 'Inns', 'Outs']
        )

        filtered_season_stats = filtered_season_stats[[ 
            'Name', 'Year', 'Matches', 'Runs', 'Bat Avg', 'Bat SR', 
            'Fifties', 'Hundreds', 'Overs',
            'Bowl_Runs', 'Wickets', 'FiveWickets', 
            'Bowl Avg', 'Econ', 'Bowl SR', 'POM'
        ]]

        # Replace infinite values with NaN for display
        filtered_season_stats = filtered_season_stats.replace([np.inf, -np.inf], np.nan)

        # Display Season Statistics with modern header
        st.markdown("""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 20px; border-radius: 15px; text-align: center; margin: 30px 0 20px 0;
                        box-shadow: 0 8px 25px rgba(0,0,0,0.15); color: white; font-weight: bold; font-size: 24px;">
                📅 Season Statistics
            </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(filtered_season_stats, use_container_width=True, hide_index=True)

###-------------------------------------GRAPHS----------------------------------------###
        # Modern graphs section header
        st.markdown("""
            <div style="background: linear-gradient(135deg, #ff9500 0%, #ffad33 100%); 
                        padding: 20px; border-radius: 15px; text-align: center; margin: 30px 0 20px 0;
                        box-shadow: 0 8px 25px rgba(0,0,0,0.15); color: white; font-weight: bold; font-size: 24px;">
                📈 Performance Charts
            </div>
        """, unsafe_allow_html=True)
        
        # Get individual players
        individual_players = [name for name in name_choice if name != 'All']

        # Create subplots for Batting and Bowling Averages with proper titles
        fig = make_subplots(
            rows=1, cols=2, 
            subplot_titles=("Batting and Bowling Average Per Year", ""),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Handle 'All' selection
        if 'All' in name_choice:
            # Ensure we're using Year from original dataframes
            all_bat_stats = filtered_bat_df.groupby('Year').agg({
                'Runs': 'sum',
                'Out': 'sum'
            }).reset_index()
            
            all_bowl_stats = filtered_bowl_df.groupby('Year').agg({
                'Bowler_Runs': 'sum',
                'Bowler_Wkts': 'sum'
            }).reset_index()

            all_players_stats = all_bat_stats.merge(all_bowl_stats, on='Year', how='outer').fillna(0)
            all_players_stats['Bat_Avg'] = (all_players_stats['Runs'] / all_players_stats['Out']).round(2)
            all_players_stats['Bowl_Avg'] = (all_players_stats['Bowler_Runs'] / all_players_stats['Bowler_Wkts']).round(2)

            all_color = '#f84e4e' if not individual_players else 'black'

            # Make sure we have valid years for plotting
            valid_years = all_players_stats[
                (all_players_stats['Year'] >= year_choice[0]) & 
                (all_players_stats['Year'] <= year_choice[1])
            ]

            fig.add_trace(go.Bar(
                x=valid_years['Year'], 
                y=valid_years['Bat_Avg'], 
                name='All Players',
                legendgroup='All',
                marker_color=all_color
            ), row=1, col=1)

            fig.add_trace(go.Bar(
                x=valid_years['Year'], 
                y=valid_years['Bowl_Avg'], 
                name='All Players',
                legendgroup='All',
                marker_color=all_color,
                showlegend=False
            ), row=1, col=2)

        # Add individual player traces from season stats directly
        for i, name in enumerate(individual_players):
            player_stats = filtered_season_stats[filtered_season_stats['Name'] == name]
            color = '#f84e4e' if i == 0 else f'#{random.randint(0, 0xFFFFFF):06x}'

            fig.add_trace(go.Bar(
                x=player_stats['Year'], 
                y=player_stats['Bat Avg'], 
                name=name,
                legendgroup=name,
                marker_color=color
            ), row=1, col=1)

            fig.add_trace(go.Bar(
                x=player_stats['Year'], 
                y=player_stats['Bowl Avg'], 
                name=name,
                legendgroup=name,
                marker_color=color,
                showlegend=False
            ), row=1, col=2)

        for i, name in enumerate(individual_players):
            player_stats = filtered_season_stats[filtered_season_stats['Name'] == name]
            color = '#f84e4e' if i == 0 else f'#{random.randint(0, 0xFFFFFF):06x}'

            fig.add_trace(go.Bar(
                x=player_stats['Year'],
                y=player_stats['Runs'],
                name=name,
                legendgroup=name,
                marker_color=color
            ), row=1, col=1)

            fig.add_trace(go.Bar(
                x=player_stats['Year'],
                y=player_stats['Wickets'],
                name=name,
                legendgroup=name,
                marker_color=color,
                showlegend=False
            ), row=1, col=2)

        # Update layout for charts with modern styling
        fig.update_layout(
            showlegend=True,
            height=500,
            font=dict(size=12),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            barmode='group',
            title_font_size=20,
            title_font_color='#2c3e50',
            legend=dict(
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            )
        )

        # Update y-axis titles for both subplots
        fig.update_yaxes(title_text="Average", row=1, col=1)
        fig.update_yaxes(title_text="Average", row=1, col=2)

        # Update axes for second graph
        fig.update_xaxes(
            range=[year_choice[0] - 0.5, year_choice[1] + 0.5],
            tickmode='linear',
            dtick=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)'
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)'
        )

        # Display the chart in a styled container
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

###################-------------------------------------GRAPHS----------------------------------------#############

        # Scatter plot of Batting Average vs Bowling Average with Name labels
        if 'All' in name_choice:
            # If 'All' is selected, include all players from the filtered dataset
            all_players = filtered_season_stats['Name'].unique()
        else:
            all_players = individual_players  # Use selected individual players

        if not all_players.any():
            st.markdown("""
                <div class="warning-banner">
                    ❌ No players available to display.
                </div>
            """, unsafe_allow_html=True)
        else:
            # Create a new figure for the scatter plot
            scatter_fig = go.Figure()

            # Loop through each player and plot their Batting and Bowling Averages
            for i, name in enumerate(all_players):
                player_stats = filtered_season_stats[filtered_season_stats['Name'] == name]

                # Calculate player's Batting and Bowling Averages
                batting_avg = player_stats['Bat Avg'].mean()  # Averaging across seasons if there are multiple entries
                bowling_avg = player_stats['Bowl Avg'].mean()  # Averaging across seasons if there are multiple entries
                total_runs = player_stats['Runs'].sum()  # Sum of runs
                total_wickets = player_stats['Wickets'].sum()  # Sum of wickets

                # Add scatter point for the player with custom hover info
                scatter_fig.add_trace(go.Scatter(
                    x=[batting_avg], 
                    y=[bowling_avg], 
                    mode='markers+text', 
                    text=[name],
                    textposition='top center',
                    marker=dict(size=10),
                    name=name,
                    hovertemplate=(
                        f"<b>{name}</b><br><br>"
                        "Batting Average: %{x:.2f}<br>"
                        "Bowling Average: %{y:.2f}<br>"
                        "Runs: " + str(total_runs) + "<br>"
                        "Wickets: " + str(total_wickets) + "<extra></extra>"
                    )
                ))
            # Display the scatter plot title and chart
            st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 20px; border-radius: 15px; text-align: center; margin: 30px 0 20px 0;
                            box-shadow: 0 8px 25px rgba(0,0,0,0.15); color: white; font-weight: bold; font-size: 24px;">
                    🎯 Batting Average vs Bowling Average
                </div>
            """, unsafe_allow_html=True)

            # Update layout for scatter plot with modern styling
            scatter_fig.update_layout(
                xaxis_title="Batting Average",
                yaxis_title="Bowling Average",
                height=500,
                font=dict(size=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False,  # No need for legend, player names will be shown as labels
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.2)'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.2)'
                )
            )

            # Show scatter plot in styled container
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.plotly_chart(scatter_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="warning-banner">
                ❌ Required data not found. Please ensure you have processed the scorecards.
            </div>
        """, unsafe_allow_html=True)



# No need for the if __name__ == "__main__" part
display_ar_view()
