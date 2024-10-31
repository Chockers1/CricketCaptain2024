import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random


def parse_date(date_str):
    """Helper function to parse dates in multiple formats"""
    try:
        for fmt in ['%d/%m/%Y', '%d %b %Y', '%Y-%m-%d']:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except ValueError:
                continue
        return pd.to_datetime(date_str)
    except Exception:
        return pd.NaT

def display_team_view():
    st.markdown("<h1 style='color:#f04f53; text-align: center;'>Team Statistics</h1>", unsafe_allow_html=True)

    # Custom CSS for styling
    st.markdown("""
    <style>
    table { color: black; width: 100%; }
    thead tr th {
        background-color: #f04f53 !important;
        color: white !important;
    }
    tbody tr:nth-child(even) { background-color: #f0f2f6; }
    tbody tr:nth-child(odd) { background-color: white; }
    </style>
    """, unsafe_allow_html=True)

    # Check if required DataFrames exist in session state
    if 'bat_df' in st.session_state and 'bowl_df' in st.session_state:
        bat_df = st.session_state['bat_df'].copy()
        bowl_df = st.session_state['bowl_df'].copy()

        # Convert Year to integer type for both DataFrames
        if 'Year' not in bat_df.columns and 'Date' in bat_df.columns:
            bat_df['Year'] = pd.to_datetime(bat_df['Date']).dt.year
        elif 'Year' not in bat_df.columns:
            bat_df['Year'] = pd.Timestamp.now().year
            
        if 'Year' not in bowl_df.columns and 'Date' in bowl_df.columns:
            bowl_df['Year'] = pd.to_datetime(bowl_df['Date']).dt.year
        elif 'Year' not in bowl_df.columns:
            bowl_df['Year'] = pd.Timestamp.now().year

        # Ensure Year is integer type
        bat_df['Year'] = pd.to_numeric(bat_df['Year'], errors='coerce').fillna(0).astype(int)
        bowl_df['Year'] = pd.to_numeric(bowl_df['Year'], errors='coerce').fillna(0).astype(int)

        # Get filter options
        match_formats = ['All'] + sorted(bat_df['Match_Format'].unique().tolist())
        years = sorted(list(set(bat_df['Year'].unique()) | set(bowl_df['Year'].unique())))
        years = [year for year in years if year != 0]
        
        if not years:
            years = [pd.Timestamp.now().year]

        # Get unique teams
        bat_teams = ['All'] + sorted(set(list(bat_df['Bat_Team_y'].unique()) + list(bowl_df['Bat_Team'].unique())))
        bowl_teams = ['All'] + sorted(set(list(bat_df['Bowl_Team_y'].unique()) + list(bowl_df['Bowl_Team'].unique())))

        # Create the filters row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            bat_team_choice = st.multiselect('Bat Team', bat_teams, default='All', key='bat_team_filter')

        with col2:
            bowl_team_choice = st.multiselect('Bowl Team', bowl_teams, default='All', key='bowl_team_filter')

        with col3:
            match_format_choice = st.multiselect('Format', match_formats, default='All', key='match_format_filter')

        with col4:
            st.markdown("<p style='margin-bottom: 5px;'>Year</p>", unsafe_allow_html=True)
            year_choice = st.slider('',
                                min_value=min(years),
                                max_value=max(years),
                                value=(min(years), max(years)),
                                key='year_slider',
                                label_visibility='collapsed')

        # Create filtered DataFrames based on selections
        filtered_bat_df = bat_df.copy()
        filtered_bowl_df = bowl_df.copy()

        # Apply filters
        if 'All' not in bat_team_choice:
            filtered_bat_df = filtered_bat_df[filtered_bat_df['Bat_Team_y'].isin(bat_team_choice)]
            filtered_bowl_df = filtered_bowl_df[filtered_bowl_df['Bat_Team'].isin(bat_team_choice)]

        if 'All' not in bowl_team_choice:
            filtered_bat_df = filtered_bat_df[filtered_bat_df['Bowl_Team_y'].isin(bowl_team_choice)]
            filtered_bowl_df = filtered_bowl_df[filtered_bowl_df['Bowl_Team'].isin(bowl_team_choice)]

        if 'All' not in match_format_choice:
            filtered_bat_df = filtered_bat_df[filtered_bat_df['Match_Format'].isin(match_format_choice)]
            filtered_bowl_df = filtered_bowl_df[filtered_bowl_df['Match_Format'].isin(match_format_choice)]

        filtered_bat_df = filtered_bat_df[filtered_bat_df['Year'].between(year_choice[0], year_choice[1])]
        filtered_bowl_df = filtered_bowl_df[filtered_bowl_df['Year'].between(year_choice[0], year_choice[1])]


    ###############################################################################
    # Section 2: Page Header and Styling
    ###############################################################################
        st.markdown("<h1 style='color:#f04f53; text-align: center;'>Team Stats</h1>", unsafe_allow_html=True)

        # Custom CSS for styling
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

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            width: 100%;
        }
        .stTabs [data-baseweb="tab"] {
            flex-grow: 1;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)

    # Create tabs
    tab1, tab2 = st.tabs(["Batting Statistics", "Bowling Statistics"])

    # Batting Statistics Tab
    with tab1:
        # Initial merged stats
        initial_merged_df = pd.merge(
            filtered_bat_df,
            filtered_bowl_df,
            on=['File Name', 'Innings'],
            how='outer',
            suffixes=('_bat', '_bowl')
        )

        # Aggregating statistics
        initial_stats = initial_merged_df.groupby('Bat_Team_y').agg({
            'File Name': 'nunique',
            'Batted': 'sum',
            'Out': 'sum',
            'Balls': 'sum',
            'Runs': 'sum',
            'Bowler_Balls': 'sum',
            'Bowler_Runs': 'sum',
            'Bowler_Wkts': 'sum'
        }).reset_index()

        # Calculate team batting statistics
        bat_Team_df = filtered_bat_df.groupby('Bat_Team_y').agg({
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
        bat_Team_df.columns = ['Team', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls',
                                'Runs', 'HS', '4s', '6s', '50s', '100s', '200s',
                                '<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped',
                                'Team Runs', 'Overs', 'Wickets', 'Team Balls']

        # Calculate metrics
        bat_Team_df['Batting Average'] = (bat_Team_df['Runs'] / bat_Team_df['Out']).round(2).fillna(0)
        bat_Team_df['Batting Strike Rate'] = ((bat_Team_df['Runs'] / bat_Team_df['Balls']) * 100).round(2).fillna(0)
        bat_Team_df['Balls Per Out'] = (bat_Team_df['Balls'] / bat_Team_df['Out']).round(2).fillna(0)
        bat_Team_df['Team Batting Average'] = (bat_Team_df['Team Runs'] / bat_Team_df['Wickets']).round(2).fillna(0)
        bat_Team_df['Team Batting Strike Rate'] = (bat_Team_df['Team Runs'] / bat_Team_df['Team Balls'] * 100).round(2).fillna(0)
        bat_Team_df['P+ Batting Average'] = (bat_Team_df['Batting Average'] / bat_Team_df['Team Batting Average'] * 100).round(2).fillna(0)
        bat_Team_df['P+ Batting Strike Rate'] = (bat_Team_df['Batting Strike Rate'] / bat_Team_df['Team Batting Strike Rate'] * 100).round(2).fillna(0)
        bat_Team_df['Balls Per Boundary'] = (bat_Team_df['Balls'] / (bat_Team_df['4s'] + bat_Team_df['6s']).replace(0, 1)).round(2)
        bat_Team_df['50+ Scores %'] = (((bat_Team_df['50s'] + bat_Team_df['100s']) / bat_Team_df['Inns']) * 100).round(2).fillna(0)
        bat_Team_df['100s %'] = ((bat_Team_df['100s'] / bat_Team_df['Inns']) * 100).round(2).fillna(0)
        bat_Team_df['<25 & Out %'] = ((bat_Team_df['<25&Out'] / bat_Team_df['Inns']) * 100).round(2).fillna(0)
        bat_Team_df['Caught %'] = ((bat_Team_df['Caught'] / bat_Team_df['Inns']) * 100).round(2).fillna(0)
        bat_Team_df['Bowled %'] = ((bat_Team_df['Bowled'] / bat_Team_df['Inns']) * 100).round(2).fillna(0)
        bat_Team_df['LBW %'] = ((bat_Team_df['LBW'] / bat_Team_df['Inns']) * 100).round(2).fillna(0)

        # Display Career Stats
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Career Statistics</h3>", unsafe_allow_html=True)
        st.dataframe(bat_Team_df, use_container_width=True, hide_index=True)
    ######################_--------------scatter-----------------#######################

        # Create a new figure for the scatter plot
        scatter_fig = go.Figure()

        # Plot data for each team
        for team in bat_Team_df['Team'].unique():
            team_stats = bat_Team_df[bat_Team_df['Team'] == team]
            
            # Get team statistics
            batting_avg = team_stats['Batting Average'].iloc[0]
            strike_rate = team_stats['Batting Strike Rate'].iloc[0]
            runs = team_stats['Runs'].iloc[0]
            
            # Add scatter point for the team
            scatter_fig.add_trace(go.Scatter(
                x=[batting_avg],
                y=[strike_rate],
                mode='markers+text',
                text=[team],
                textposition='top center',
                marker=dict(size=10),
                name=team,
                hovertemplate=(
                    f"{team}<br>"
                    f"Batting Average: {batting_avg:.2f}<br>"
                    f"Strike Rate: {strike_rate:.2f}<br>"
                    f"Runs: {runs}<br>"
                )
            ))

        # Display the title using Streamlit's markdown
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Team Batting Average v Strike Rate</h3>", unsafe_allow_html=True)


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

        # Show plot
        st.plotly_chart(scatter_fig)   


    ######################_--------------SEASON STATS-----------------#######################
        # Calculate season statistics for teams
        bat_team_season_df = filtered_bat_df.groupby(['Bat_Team_y', 'Year']).agg({
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
        bat_team_season_df.columns = ['Team', 'Year', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 
                                    'Runs', 'HS', '4s', '6s', '50s', '100s', '200s', 
                                    '<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 
                                    'Team Runs', 'Overs', 'Wickets', 'Team Balls']

        # Calculate season metrics
        bat_team_season_df['Avg'] = (bat_team_season_df['Runs'] / bat_team_season_df['Out']).round(2).fillna(0)
        bat_team_season_df['SR'] = ((bat_team_season_df['Runs'] / bat_team_season_df['Balls']) * 100).round(2).fillna(0)
        bat_team_season_df['BPO'] = (bat_team_season_df['Balls'] / bat_team_season_df['Out']).round(2).fillna(0)
        bat_team_season_df['Team Avg'] = (bat_team_season_df['Team Runs'] / bat_team_season_df['Wickets']).round(2).fillna(0)
        bat_team_season_df['Team SR'] = (bat_team_season_df['Team Runs'] / bat_team_season_df['Team Balls'] * 100).round(2).fillna(0)
        bat_team_season_df['P+ Avg'] = (bat_team_season_df['Avg'] / bat_team_season_df['Team Avg'] * 100).round(2).fillna(0)
        bat_team_season_df['P+ SR'] = (bat_team_season_df['SR'] / bat_team_season_df['Team SR'] * 100).round(2).fillna(0)
        bat_team_season_df['BPB'] = (bat_team_season_df['Balls'] / (bat_team_season_df['4s'] + bat_team_season_df['6s']).replace(0, 1)).round(2)
        bat_team_season_df['50+PI'] = (((bat_team_season_df['50s'] + bat_team_season_df['100s']) / bat_team_season_df['Inns']) * 100).round(2).fillna(0)
        bat_team_season_df['100PI'] = ((bat_team_season_df['100s'] / bat_team_season_df['Inns']) * 100).round(2).fillna(0)
        bat_team_season_df['<25&OutPI'] = ((bat_team_season_df['<25&Out'] / bat_team_season_df['Inns']) * 100).round(2).fillna(0)

        # Calculate dismissal percentages
        bat_team_season_df['Caught%'] = ((bat_team_season_df['Caught'] / bat_team_season_df['Inns']) * 100).round(2).fillna(0)
        bat_team_season_df['Bowled%'] = ((bat_team_season_df['Bowled'] / bat_team_season_df['Inns']) * 100).round(2).fillna(0)
        bat_team_season_df['LBW%'] = ((bat_team_season_df['LBW'] / bat_team_season_df['Inns']) * 100).round(2).fillna(0)
        bat_team_season_df['Run Out%'] = ((bat_team_season_df['Run Out'] / bat_team_season_df['Inns']) * 100).round(2).fillna(0)
        bat_team_season_df['Stumped%'] = ((bat_team_season_df['Stumped'] / bat_team_season_df['Inns']) * 100).round(2).fillna(0)
        bat_team_season_df['Not Out%'] = ((bat_team_season_df['Not Out'] / bat_team_season_df['Inns']) * 100).round(2).fillna(0)

        # Reorder and sort
        bat_team_season_df = bat_team_season_df[['Team', 'Year', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 'Runs', 'HS', 
                                                'Avg', 'BPO', 'SR', '4s', '6s', 'BPB', '<25&Out', '50s', '100s', 
                                                '<25&OutPI', '50+PI', '100PI', 'P+ Avg', 'P+ SR', 
                                                'Caught%', 'Bowled%', 'LBW%', 'Run Out%', 'Stumped%', 'Not Out%']]
        bat_team_season_df = bat_team_season_df.sort_values(by=['Year', 'Runs'], ascending=[False, False])   

        # Display Season Stats
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Season Statistics</h3>", unsafe_allow_html=True)
        st.dataframe(bat_team_season_df, use_container_width=True, hide_index=True)

        # Create a new figure for the scatter plot
        scatter_fig = go.Figure()

        # For each row in bat_team_season_df (this will include all team-year combinations)
        for _, row in bat_team_season_df.iterrows():
            # Add scatter point for the team-year combination
            scatter_fig.add_trace(go.Scatter(
                x=[row['Avg']],
                y=[row['SR']],
                mode='markers+text',
                text=[f"{row['Team']} ({row['Year']})"],  # Add year to the label
                textposition='top center', 
                marker=dict(size=10),
                name=f"{row['Team']} {row['Year']}",
                hovertemplate=(
                    f"{row['Team']} ({row['Year']})<br>"
                    f"Batting Average: {row['Avg']:.3f}<br>"
                    f"Strike Rate: {row['SR']:.2f}<br>"
                )
            ))

        # Display the title using Streamlit's markdown

        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Team Batting Average v Strike Rate Season</h3>", unsafe_allow_html=True)


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

        # Show plot
        st.plotly_chart(scatter_fig, key="team_season_scatter")

        ###---------------------------------------------OPPONENTS STATS-------------------------------------------------------------------###        
        # Calculate opponents statistics
        bat_team_opponent_df = filtered_bat_df.groupby(['Bat_Team_y', 'Bowl_Team_y']).agg({
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
        bat_team_opponent_df.columns = ['Team', 'Opponent', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 
                                        'Runs', 'HS', '4s', '6s', '50s', '100s', '200s', 
                                        '<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 
                                        'Team Runs', 'Overs', 'Wickets', 'Team Balls']

        # Calculate opponent averages
        opponent_stats_df = filtered_bat_df.groupby('Bowl_Team_y').agg({
            'Runs': 'sum',
            'Out': 'sum',
            'Balls': 'sum'
        }).reset_index()
        opponent_stats_df['Avg'] = (opponent_stats_df['Runs'] / opponent_stats_df['Out']).round(2)

        # Create bar chart
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=opponent_stats_df['Bowl_Team_y'], 
                y=opponent_stats_df['Avg'], 
                name='Average', 
                marker_color='#f84e4e'
            )
        )

        # Display Opponents Stats
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Opposition Statistics</h3>", unsafe_allow_html=True)
        st.dataframe(bat_team_opponent_df, use_container_width=True, hide_index=True)
        st.plotly_chart(fig)

##############----------------------LOCATION STATS----------------###############
        # First create the location DataFrame
        bat_team_location_df = filtered_bat_df.groupby(['Bat_Team_y', 'Home Team']).agg({
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

        # Then flatten the columns
        bat_team_location_df.columns = ['Team', 'Location', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 
                                        'Runs', 'HS', '4s', '6s', '50s', '100s', '200s', 
                                        '<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 
                                        'Team Runs', 'Overs', 'Wickets', 'Team Balls']

        # Calculate metrics for location stats
        bat_team_location_df['Avg'] = (bat_team_location_df['Runs'] / bat_team_location_df['Out']).round(2).fillna(0)
        bat_team_location_df['SR'] = ((bat_team_location_df['Runs'] / bat_team_location_df['Balls']) * 100).round(2).fillna(0)
        bat_team_location_df['BPO'] = (bat_team_location_df['Balls'] / bat_team_location_df['Out']).round(2).fillna(0)
        bat_team_location_df['Team Avg'] = (bat_team_location_df['Team Runs'] / bat_team_location_df['Wickets']).round(2).fillna(0)
        bat_team_location_df['Team SR'] = (bat_team_location_df['Team Runs'] / bat_team_location_df['Team Balls'] * 100).round(2).fillna(0)
        bat_team_location_df['P+ Avg'] = (bat_team_location_df['Avg'] / bat_team_location_df['Team Avg'] * 100).round(2).fillna(0)
        bat_team_location_df['P+ SR'] = (bat_team_location_df['SR'] / bat_team_location_df['Team SR'] * 100).round(2).fillna(0)
        bat_team_location_df['BPB'] = (bat_team_location_df['Balls'] / (bat_team_location_df['4s'] + bat_team_location_df['6s']).replace(0, 1)).round(2)
        bat_team_location_df['50+PI'] = (((bat_team_location_df['50s'] + bat_team_location_df['100s']) / bat_team_location_df['Inns']) * 100).round(2).fillna(0)
        bat_team_location_df['100PI'] = ((bat_team_location_df['100s'] / bat_team_location_df['Inns']) * 100).round(2).fillna(0)
        bat_team_location_df['<25&OutPI'] = ((bat_team_location_df['<25&Out'] / bat_team_location_df['Inns']) * 100).round(2).fillna(0)

        # Calculate dismissal percentages
        bat_team_location_df['Caught%'] = ((bat_team_location_df['Caught'] / bat_team_location_df['Inns']) * 100).round(2).fillna(0)
        bat_team_location_df['Bowled%'] = ((bat_team_location_df['Bowled'] / bat_team_location_df['Inns']) * 100).round(2).fillna(0)
        bat_team_location_df['LBW%'] = ((bat_team_location_df['LBW'] / bat_team_location_df['Inns']) * 100).round(2).fillna(0)
        bat_team_location_df['Run Out%'] = ((bat_team_location_df['Run Out'] / bat_team_location_df['Inns']) * 100).round(2).fillna(0)
        bat_team_location_df['Stumped%'] = ((bat_team_location_df['Stumped'] / bat_team_location_df['Inns']) * 100).round(2).fillna(0)
        bat_team_location_df['Not Out%'] = ((bat_team_location_df['Not Out'] / bat_team_location_df['Inns']) * 100).round(2).fillna(0)

        # Reorder columns and sort
        bat_team_location_df = bat_team_location_df[['Team', 'Location', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 'Runs', 'HS', 
                                                    'Avg', 'BPO', 'SR', '4s', '6s', 'BPB', '<25&Out', '50s', '100s', 
                                                    '<25&OutPI', '50+PI', '100PI', 'P+ Avg', 'P+ SR', 
                                                    'Caught%', 'Bowled%', 'LBW%', 'Run Out%', 'Stumped%', 'Not Out%']]
        bat_team_location_df = bat_team_location_df.sort_values(by='Runs', ascending=False)

        # Display Location Stats
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Location Statistics</h3>", unsafe_allow_html=True)
        st.dataframe(bat_team_location_df, use_container_width=True, hide_index=True)

        # Calculate location-based team averages
        location_stats_df = bat_team_location_df.groupby('Location').agg({
            'Runs': 'sum',
            'Out': 'sum',
            'Balls': 'sum'
        }).reset_index()
        location_stats_df['Avg'] = (location_stats_df['Runs'] / location_stats_df['Out']).round(2)

        # Create bar chart for average runs per out by location
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=location_stats_df['Location'], 
                y=location_stats_df['Avg'], 
                name='Average', 
                marker_color='#f84e4e'
            )
        )

        # Display location-based statistics
        #st.markdown("<h3 style='color:#f04f53; text-align: center;'>Location-Based Batting Averages</h3>", unsafe_allow_html=True)
        #st.dataframe(location_stats_df, use_container_width=True, hide_index=True)
        st.plotly_chart(fig)





        ####################----------------------POSITION STATS----------------------#########################
        # Calculate position statistics
        bat_team_position_df = filtered_bat_df.groupby(['Bat_Team_y', 'Position']).agg({
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

        # Flatten multi-level columns for position stats
        bat_team_position_df.columns = ['Team', 'Position', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 
                                        'Runs', 'HS', '4s', '6s', '50s', '100s', '200s', 
                                        '<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 
                                        'Team Runs', 'Overs', 'Wickets', 'Team Balls']

        # Calculate metrics for position stats
        bat_team_position_df['Avg'] = (bat_team_position_df['Runs'] / bat_team_position_df['Out']).round(2).fillna(0)
        bat_team_position_df['SR'] = ((bat_team_position_df['Runs'] / bat_team_position_df['Balls']) * 100).round(2).fillna(0)
        bat_team_position_df['BPO'] = (bat_team_position_df['Balls'] / bat_team_position_df['Out']).round(2).fillna(0)
        bat_team_position_df['Team Avg'] = (bat_team_position_df['Team Runs'] / bat_team_position_df['Wickets']).round(2).fillna(0)
        bat_team_position_df['Team SR'] = (bat_team_position_df['Team Runs'] / bat_team_position_df['Team Balls'] * 100).round(2).fillna(0)
        bat_team_position_df['P+ Avg'] = (bat_team_position_df['Avg'] / bat_team_position_df['Team Avg'] * 100).round(2).fillna(0)
        bat_team_position_df['P+ SR'] = (bat_team_position_df['SR'] / bat_team_position_df['Team SR'] * 100).round(2).fillna(0)
        bat_team_position_df['BPB'] = (bat_team_position_df['Balls'] / (bat_team_position_df['4s'] + bat_team_position_df['6s']).replace(0, 1)).round(2)
        bat_team_position_df['50+PI'] = (((bat_team_position_df['50s'] + bat_team_position_df['100s']) / bat_team_position_df['Inns']) * 100).round(2).fillna(0)
        bat_team_position_df['100PI'] = ((bat_team_position_df['100s'] / bat_team_position_df['Inns']) * 100).round(2).fillna(0)
        bat_team_position_df['<25&OutPI'] = ((bat_team_position_df['<25&Out'] / bat_team_position_df['Inns']) * 100).round(2).fillna(0)

        # Calculate dismissal percentages for position stats
        bat_team_position_df['Caught%'] = ((bat_team_position_df['Caught'] / bat_team_position_df['Inns']) * 100).round(2).fillna(0)
        bat_team_position_df['Bowled%'] = ((bat_team_position_df['Bowled'] / bat_team_position_df['Inns']) * 100).round(2).fillna(0)
        bat_team_position_df['LBW%'] = ((bat_team_position_df['LBW'] / bat_team_position_df['Inns']) * 100).round(2).fillna(0)
        bat_team_position_df['Run Out%'] = ((bat_team_position_df['Run Out'] / bat_team_position_df['Inns']) * 100).round(2).fillna(0)
        bat_team_position_df['Stumped%'] = ((bat_team_position_df['Stumped'] / bat_team_position_df['Inns']) * 100).round(2).fillna(0)
        bat_team_position_df['Not Out%'] = ((bat_team_position_df['Not Out'] / bat_team_position_df['Inns']) * 100).round(2).fillna(0)

        # Reorder columns and sort for position stats
        bat_team_position_df = bat_team_position_df[['Team', 'Position', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 'Runs', 'HS', 
                                                    'Avg', 'BPO', 'SR', '4s', '6s', 'BPB', '<25&Out', '50s', '100s', 
                                                    '<25&OutPI', '50+PI', '100PI', 'P+ Avg', 'P+ SR', 
                                                    'Caught%', 'Bowled%', 'LBW%', 'Run Out%', 'Stumped%', 'Not Out%']]
        bat_team_position_df = bat_team_position_df.sort_values(by='Runs', ascending=False)

        # Display Position Stats
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Position Statistics</h3>", unsafe_allow_html=True)
        st.dataframe(bat_team_position_df, use_container_width=True, hide_index=True)


###-------------------------------------BOWLING STATS-------------------------------------###

    # Bowling  Statistics Tab
    with tab2:
        # Calculate career statistics
        match_wickets = filtered_bowl_df.groupby(['Bowl_Team', 'File Name'])['Bowler_Wkts'].sum().reset_index()
        ten_wickets = match_wickets[match_wickets['Bowler_Wkts'] >= 10].groupby('Bowl_Team').size().reset_index(name='10W')

        bowl_team_df = filtered_bowl_df.groupby('Bowl_Team').agg({
            'File Name': 'nunique',
            'Bowler_Balls': 'sum',
            'Maidens': 'sum',
            'Bowler_Runs': 'sum',
            'Bowler_Wkts': 'sum'
        }).reset_index()

        bowl_team_df.columns = ['Bowl_Team', 'Matches', 'Balls', 'M/D', 'Runs', 'Wickets']

        # Calculate career metrics
        bowl_team_df['Overs'] = (bowl_team_df['Balls'] // 6) + (bowl_team_df['Balls'] % 6) / 10
        bowl_team_df['Strike Rate'] = (bowl_team_df['Balls'] / bowl_team_df['Wickets']).round(2)
        bowl_team_df['Economy Rate'] = (bowl_team_df['Runs'] / bowl_team_df['Overs']).round(2)
        bowl_team_df['Avg'] = (bowl_team_df['Runs'] / bowl_team_df['Wickets']).round(2)

        # Add additional statistics
        five_wickets = filtered_bowl_df[filtered_bowl_df['Bowler_Wkts'] >= 5].groupby('Bowl_Team').size().reset_index(name='5W')
        bowl_team_df = bowl_team_df.merge(five_wickets, on='Bowl_Team', how='left')
        bowl_team_df['5W'] = bowl_team_df['5W'].fillna(0).astype(int)

        bowl_team_df = bowl_team_df.merge(ten_wickets, on='Bowl_Team', how='left')
        bowl_team_df['10W'] = bowl_team_df['10W'].fillna(0).astype(int)

        bowl_team_df['WPM'] = (bowl_team_df['Wickets'] / bowl_team_df['Matches']).round(2)

        pom_counts = filtered_bowl_df[filtered_bowl_df['Player_of_the_Match'] == filtered_bowl_df['Bowl_Team']].groupby('Bowl_Team')['File Name'].nunique().reset_index(name='POM')
        bowl_team_df = bowl_team_df.merge(pom_counts, on='Bowl_Team', how='left')
        bowl_team_df['POM'] = bowl_team_df['POM'].fillna(0).astype(int)

        bowl_team_df = bowl_team_df.replace([np.inf, -np.inf], np.nan)

        # Final column ordering
        bowl_team_df = bowl_team_df[[
            'Bowl_Team', 'Matches', 'Balls', 'Overs', 'M/D', 'Runs', 'Wickets', 'Avg',
            'Strike Rate', 'Economy Rate', '5W', '10W', 'WPM', 'POM'
        ]]

        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Bowling Statistics</h3>", unsafe_allow_html=True)
        st.dataframe(bowl_team_df, use_container_width=True, hide_index=True)

        ###########--------------------------- SCATTER PLOT GRAPH--------------------------###################

        # Create a new figure for the scatter plot
        scatter_fig = go.Figure()

        for name in bowl_team_df['Bowl_Team'].unique():
            team_stats = bowl_team_df[bowl_team_df['Bowl_Team'] == name]
            
            # Calculate Economy Rate
            total_runs = team_stats['Runs'].iloc[0]
            total_balls = team_stats['Balls'].iloc[0]
            economy_rate = (total_runs / (total_balls / 6)).round(2)  # Total runs per over

            # Calculate Strike Rate (total balls per wicket)
            wickets = team_stats['Wickets'].iloc[0]
            strike_rate = (total_balls / wickets).round(2) if wickets > 0 else "N/A"  # Avoid division by zero

            # Format the hovertemplate based on strike rate availability
            strike_rate_display = f"{strike_rate:.2f}" if isinstance(strike_rate, float) else strike_rate
            
            # Add scatter point for the team
            scatter_fig.add_trace(go.Scatter(
                x=[economy_rate],
                y=[strike_rate if isinstance(strike_rate, float) else None],
                mode='markers+text',
                text=[name],
                textposition='top center',
                marker=dict(size=10),
                name=name,
                hovertemplate=(
                    f"<b>{name}</b><br><br>"
                    f"Economy Rate: {economy_rate:.2f}<br>"
                    f"Strike Rate: {strike_rate_display}<br>"
                    f"Wickets: {wickets}<br>"
                    "<extra></extra>"
                )
            ))

        # Display the title using Streamlit's markdown
        st.markdown(
            "<h3 style='color:#f04f53; text-align: center;'>Economy Rate vs Strike Rate Analysis</h3>",
            unsafe_allow_html=True
        )

        # Update layout
        scatter_fig.update_layout(
            xaxis_title="Economy Rate",
            yaxis_title="Strike Rate",
            height=500,
            font=dict(size=12),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )

        # Show plot
        st.plotly_chart(scatter_fig)
    ###------------------------------------- SEASON STATS -------------------------------------###

        # Group by 'Bowl_Team' and 'Year' to calculate season statistics
        match_wickets = filtered_bowl_df.groupby(['Bowl_Team', 'Year', 'File Name'])['Bowler_Wkts'].sum().reset_index()
        ten_wickets = match_wickets[match_wickets['Bowler_Wkts'] >= 10].groupby(['Bowl_Team', 'Year']).size().reset_index(name='10W')

        bowl_team_season_df = filtered_bowl_df.groupby(['Bowl_Team', 'Year']).agg({
            'File Name': 'nunique',
            'Bowler_Balls': 'sum',
            'Maidens': 'sum',
            'Bowler_Runs': 'sum',
            'Bowler_Wkts': 'sum'
        }).reset_index()

        bowl_team_season_df.columns = ['Bowl_Team', 'Year', 'Matches', 'Balls', 'M/D', 'Runs', 'Wickets']

        # Calculate season metrics
        bowl_team_season_df['Overs'] = (bowl_team_season_df['Balls'] // 6) + (bowl_team_season_df['Balls'] % 6) / 10
        bowl_team_season_df['Strike Rate'] = (bowl_team_season_df['Balls'] / bowl_team_season_df['Wickets']).round(2)
        bowl_team_season_df['Economy Rate'] = (bowl_team_season_df['Runs'] / bowl_team_season_df['Overs']).round(2)
        bowl_team_season_df['Avg'] = (bowl_team_season_df['Runs'] / bowl_team_season_df['Wickets']).round(2)

        # Add additional statistics
        five_wickets = filtered_bowl_df[filtered_bowl_df['Bowler_Wkts'] >= 5].groupby(['Bowl_Team', 'Year']).size().reset_index(name='5W')
        bowl_team_season_df = bowl_team_season_df.merge(five_wickets, on=['Bowl_Team', 'Year'], how='left')
        bowl_team_season_df['5W'] = bowl_team_season_df['5W'].fillna(0).astype(int)

        bowl_team_season_df = bowl_team_season_df.merge(ten_wickets, on=['Bowl_Team', 'Year'], how='left')
        bowl_team_season_df['10W'] = bowl_team_season_df['10W'].fillna(0).astype(int)

        bowl_team_season_df['WPM'] = (bowl_team_season_df['Wickets'] / bowl_team_season_df['Matches']).round(2)

        pom_counts = filtered_bowl_df[filtered_bowl_df['Player_of_the_Match'] == filtered_bowl_df['Bowl_Team']].groupby(['Bowl_Team', 'Year'])['File Name'].nunique().reset_index(name='POM')
        bowl_team_season_df = bowl_team_season_df.merge(pom_counts, on=['Bowl_Team', 'Year'], how='left')
        bowl_team_season_df['POM'] = bowl_team_season_df['POM'].fillna(0).astype(int)

        bowl_team_season_df = bowl_team_season_df.replace([np.inf, -np.inf], np.nan)

        # Final column ordering
        bowl_team_season_df = bowl_team_season_df[[
            'Bowl_Team', 'Year', 'Matches', 'Balls', 'Overs', 'M/D', 'Runs', 'Wickets', 'Avg',
            'Strike Rate', 'Economy Rate', '5W', '10W', 'WPM', 'POM'
        ]]

        # Display in Streamlit with the title "Season Stats"
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Team Bowling Season Stats</h3>", unsafe_allow_html=True)
        st.dataframe(bowl_team_season_df, use_container_width=True, hide_index=True)


        # Create a new figure for the scatter plot
        scatter_fig = go.Figure()

        # For each row in bowl_team_season_df (this will include all team-year combinations)
        for _, row in bowl_team_season_df.iterrows():
            # Calculate Economy Rate and Strike Rate
            economy_rate = round((row['Runs'] / (row['Balls'] / 6)), 2)  # Total runs per over
            strike_rate = round((row['Balls'] / row['Wickets']), 2) if row['Wickets'] > 0 else "N/A"  # Avoid division by zero

            # Format the hovertemplate based on strike rate availability
            strike_rate_display = f"{strike_rate:.2f}" if isinstance(strike_rate, float) else strike_rate
            
            # Add scatter point for the team-year combination
            scatter_fig.add_trace(go.Scatter(
                x=[economy_rate],
                y=[strike_rate if isinstance(strike_rate, float) else None],
                mode='markers+text',
                text=[f"{row['Bowl_Team']} ({row['Year']})"],  # Changed from 'Team' to 'Bowl_Team'
                textposition='top center',
                marker=dict(size=10),
                name=f"{row['Bowl_Team']} {row['Year']}", # Changed from 'Team' to 'Bowl_Team'
                hovertemplate=(
                    f"<b>{row['Bowl_Team']} ({row['Year']})</b><br><br>" # Changed from 'Team' to 'Bowl_Team'
                    f"Economy Rate: {economy_rate:.2f}<br>"
                    f"Strike Rate: {strike_rate_display}<br>"
                    f"Wickets: {row['Wickets']}<br>"
                    "<extra></extra>"
                )
            ))

        # Display the title using Streamlit's markdown
        st.markdown(
            "<h3 style='color:#f04f53; text-align: center;'>Economy Rate vs Strike Rate Analysis per Season</h3>",
            unsafe_allow_html=True
        )

        # Update layout
        scatter_fig.update_layout(
            xaxis_title="Economy Rate",
            yaxis_title="Strike Rate",
            height=500,
            font=dict(size=12),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )

        # Show plot with unique key
        st.plotly_chart(scatter_fig, key="bowl_season_scatter")

    ###------------------------------------- OPPONENT STATS -------------------------------------###

        # Calculate statistics dataframe for opponents
        bowl_team_opponent_df = filtered_bowl_df.groupby(['Bowl_Team', 'Bat_Team']).agg({
            'File Name': 'nunique',      # Matches
            'Bowler_Balls': 'sum',
            'Maidens': 'sum',
            'Bowler_Runs': 'sum',
            'Bowler_Wkts': 'sum'
        }).reset_index()

        bowl_team_opponent_df.columns = ['Bowl_Team', 'Opposition', 'Matches', 'Balls', 'M/D', 'Runs', 'Wickets']

        # Calculate metrics
        bowl_team_opponent_df['Overs'] = (bowl_team_opponent_df['Balls'] // 6) + (bowl_team_opponent_df['Balls'] % 6) / 10
        bowl_team_opponent_df['Strike Rate'] = (bowl_team_opponent_df['Balls'] / bowl_team_opponent_df['Wickets']).round(2)
        bowl_team_opponent_df['Economy Rate'] = (bowl_team_opponent_df['Runs'] / bowl_team_opponent_df['Overs']).round(2)
        bowl_team_opponent_df['WPM'] = (bowl_team_opponent_df['Wickets'] / bowl_team_opponent_df['Matches']).round(2)

        # Count 5W innings
        five_wickets = filtered_bowl_df[filtered_bowl_df['Bowler_Wkts'] >= 5].groupby(['Bowl_Team', 'Bat_Team']).size().reset_index(name='5W')
        bowl_team_opponent_df = bowl_team_opponent_df.merge(five_wickets, left_on=['Bowl_Team', 'Opposition'], right_on=['Bowl_Team', 'Bat_Team'], how='left')
        bowl_team_opponent_df['5W'] = bowl_team_opponent_df['5W'].fillna(0).astype(int)

        # Count 10W matches
        match_wickets = filtered_bowl_df.groupby(['Bowl_Team', 'Bat_Team', 'File Name'])['Bowler_Wkts'].sum().reset_index()
        ten_wickets = match_wickets[match_wickets['Bowler_Wkts'] >= 10].groupby(['Bowl_Team', 'Bat_Team']).size().reset_index(name='10W')
        bowl_team_opponent_df = bowl_team_opponent_df.merge(ten_wickets, left_on=['Bowl_Team', 'Opposition'], right_on=['Bowl_Team', 'Bat_Team'], how='left')
        bowl_team_opponent_df['10W'] = bowl_team_opponent_df['10W'].fillna(0).astype(int)

        # Handle infinities and NaNs
        bowl_team_opponent_df = bowl_team_opponent_df.replace([np.inf, -np.inf], np.nan)

        # Final column ordering
        bowl_team_opponent_df = bowl_team_opponent_df[[
            'Bowl_Team', 'Opposition', 'Matches', 'Balls', 'Overs', 'M/D', 'Runs', 'Wickets',
            'Strike Rate', 'Economy Rate', '5W', '10W', 'WPM'
        ]]

        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Opposition Statistics</h3>", unsafe_allow_html=True)
        st.dataframe(bowl_team_opponent_df, use_container_width=True, hide_index=True)

        # Extract unique teams from bat_team_position_df
        team_choice = bat_team_position_df['Team'].unique().tolist()

        # Create opponent averages graph
        fig = go.Figure()

        # List of individual teams, excluding 'All'
        individual_teams = [team for team in team_choice if team != 'All']

        # Always calculate and show 'All' stats if 'All' is selected
        if 'All' in team_choice:
            # Calculate aggregate stats for all teams
            all_opponent_stats = bowl_team_opponent_df.groupby(['Opposition']).agg({
                'Runs': 'sum',   
                'Wickets': 'sum',  
            }).reset_index()

            # Calculate bowling average
            all_opponent_stats['Average'] = (all_opponent_stats['Runs'] / all_opponent_stats['Wickets']).round(2)

            # Add trace for all teams
            fig.add_trace(
                go.Bar(
                    x=all_opponent_stats['Opposition'],
                    y=all_opponent_stats['Average'],
                    name='All Teams',
                    marker_color='#FF4B4B',  # Streamlit's default red color
                    text=all_opponent_stats['Wickets'],
                    textposition='auto'
                )
            )

        # Calculate individual team stats
        individual_stats = bowl_team_opponent_df.groupby(['Opposition']).agg({
            'Runs': 'sum',
            'Wickets': 'sum',
        }).reset_index()

        # Calculate average for each team
        individual_stats['Average'] = (individual_stats['Runs'] / individual_stats['Wickets']).round(2)

        # Add traces for each individually selected team
        for team in individual_teams:
            team_data = individual_stats[individual_stats['Opposition'] == team]
            
            fig.add_trace(
                go.Bar(
                    x=team_data['Opposition'],
                    y=team_data['Average'],
                    name=team,
                    marker_color='#FF4B4B',  # Streamlit's default red color
                    text=team_data['Wickets'],
                    textposition='auto'
                )
            )

        # Update layout
        fig.update_layout(
            showlegend=False,
            height=500,
            xaxis_title="Opposition",
            yaxis_title="Average",
            font=dict(size=12),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis={'categoryorder': 'total ascending'},
            barmode='group'
        )

        # Show gridlines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

        # Display the chart
        st.plotly_chart(fig)


    ###-------------------------------------LOCATION STATS-------------------------------------###
        # Calculate statistics dataframe for locations
        location_summary = filtered_bowl_df.groupby(['Bowl_Team', 'Home_Team']).agg({
            'File Name': 'nunique',      # Matches
            'Bowler_Balls': 'sum',
            'Maidens': 'sum',
            'Bowler_Runs': 'sum',
            'Bowler_Wkts': 'sum'
        }).reset_index()

        location_summary.columns = ['Team', 'Location', 'Matches', 'Balls', 'M/D', 'Runs', 'Wickets']

        # Calculate metrics
        location_summary['Overs'] = (location_summary['Balls'] // 6) + (location_summary['Balls'] % 6) / 10
        location_summary['Strike Rate'] = (location_summary['Balls'] / location_summary['Wickets']).round(2)
        location_summary['Economy Rate'] = (location_summary['Runs'] / location_summary['Overs']).round(2)
        location_summary['WPM'] = (location_summary['Wickets'] / location_summary['Matches']).round(2)

        # Count 5W innings
        five_wickets = filtered_bowl_df[filtered_bowl_df['Bowler_Wkts'] >= 5].groupby(['Bowl_Team', 'Home_Team']).size().reset_index(name='5W')

        # Ensure you merge with the correct columns
        location_summary = location_summary.merge(
            five_wickets,
            left_on=['Team', 'Location'],  # Use new names
            right_on=['Bowl_Team', 'Home_Team'],  # Keep these as they were originally
            how='left'
        )
        location_summary['5W'] = location_summary['5W'].fillna(0).astype(int)

        # Count 10W matches
        match_wickets = filtered_bowl_df.groupby(['Bowl_Team', 'Home_Team', 'File Name'])['Bowler_Wkts'].sum().reset_index()
        ten_wickets = match_wickets[match_wickets['Bowler_Wkts'] >= 10].groupby(['Bowl_Team', 'Home_Team']).size().reset_index(name='10W')

        # Merge for 10W with updated column names
        location_summary = location_summary.merge(
            ten_wickets,
            left_on=['Team', 'Location'],
            right_on=['Bowl_Team', 'Home_Team'],  # Keep this consistent
            how='left'
        )
        location_summary['10W'] = location_summary['10W'].fillna(0).astype(int)

        # Handle infinities and NaNs
        location_summary = location_summary.replace([np.inf, -np.inf], np.nan)

        # Final column ordering
        location_summary = location_summary[[
            'Team', 'Location', 'Matches', 'Balls', 'Overs', 'M/D', 'Runs', 'Wickets',
            'Strike Rate', 'Economy Rate', '5W', '10W', 'WPM'
        ]]

        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Location Statistics</h3>", unsafe_allow_html=True)
        st.dataframe(location_summary, use_container_width=True, hide_index=True)

        # Create location averages graph
        fig = go.Figure()

        # Get unique teams from location_summary
        team_choice = location_summary['Team'].unique().tolist()
        # Add 'All' option
        team_choice = ['All'] + team_choice

        # List of individual teams, excluding 'All'
        individual_teams = [team for team in team_choice if team != 'All']

        # Always calculate and show 'All' stats
        if 'All' in team_choice:
            # Calculate aggregate stats for all teams
            all_location_stats = location_summary.groupby(['Location']).agg({
                'Runs': 'sum',   
                'Wickets': 'sum',  
            }).reset_index()

            # Calculate bowling average
            all_location_stats['Average'] = (all_location_stats['Runs'] / all_location_stats['Wickets']).round(2)

            # Add trace for all teams
            fig.add_trace(
                go.Bar(
                    x=all_location_stats['Location'],
                    y=all_location_stats['Average'],
                    name='All Teams',
                    marker_color='#FF4B4B',  # Streamlit's default red color
                    text=all_location_stats['Wickets'],
                    textposition='auto'
                )
            )

        # Add traces for each individually selected team
        for team in individual_teams:
            team_data = location_summary[location_summary['Team'] == team]
            
            fig.add_trace(
                go.Bar(
                    x=team_data['Location'],
                    y=(team_data['Runs'] / team_data['Wickets']).round(2),
                    name=team,
                    marker_color='#FF4B4B',  # Streamlit's default red color
                    text=team_data['Wickets'],
                    textposition='auto'
                )
            )

        # Update layout
        fig.update_layout(
            showlegend=False,
            height=500,
            xaxis_title="Location",
            yaxis_title="Average",
            font=dict(size=12),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis={'categoryorder': 'total ascending'},
            barmode='group'
        )

        # Show gridlines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

        # Display the chart
        st.plotly_chart(fig, key="location_average_chart")





# No need for the if __name__ == "__main__" part
display_team_view()
