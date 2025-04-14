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

        # Convert dates to datetime with safer parsing
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
        match_formats = ['All'] + sorted(bat_df['Match_Format'].unique().tolist())
        years = sorted(list(set(bat_df['Year'].unique()) | set(bowl_df['Year'].unique())))
        years = [year for year in years if year != 0]  # Remove year 0 if present
        
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
            if len(years) == 1:
                st.markdown(f"<p style='text-align: center;'>{years[0]}</p>", unsafe_allow_html=True)
                year_choice = (years[0], years[0])
            else:
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

    # Batting  Statistics Tab
    with tab1:
        # Create subtabs for different batting statistics views
        batting_subtabs = st.tabs(["Career", "Season", "Opposition", "Location", "Position"])
        
        # Career Statistics Subtab
        with batting_subtabs[0]:
            # Initial merged stats
            initial_merged_df = pd.merge(
                filtered_bat_df,
                filtered_bowl_df,
                on=['File Name', 'Innings'],
                how='outer',
                suffixes=('_bat', '_bowl')
            )

            # Calculate milestone innings based on run ranges
            filtered_bat_df['50s'] = ((filtered_bat_df['Runs'] >= 50) & (filtered_bat_df['Runs'] < 100)).astype(int)
            filtered_bat_df['100s'] = ((filtered_bat_df['Runs'] >= 100) & (filtered_bat_df['Runs'] < 150)).astype(int)
            filtered_bat_df['150s'] = ((filtered_bat_df['Runs'] >= 150) & (filtered_bat_df['Runs'] < 200)).astype(int)
            filtered_bat_df['200s'] = (filtered_bat_df['Runs'] >= 200).astype(int)

            # Create the bat_team_career_df by grouping by 'Bat_Team_y'
            bat_team_career_df = filtered_bat_df.groupby('Bat_Team_y').agg({
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
                '150s': 'sum',
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
            bat_team_career_df.columns = ['Team', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 
                                        'Runs', 'HS', '4s', '6s', '50s', '100s', '150s', '200s', 
                                        '<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 
                                        'Team Runs', 'Overs', 'Wickets', 'Team Balls']

            # Calculate average runs per out, strike rate, and balls per out
            bat_team_career_df['Avg'] = (bat_team_career_df['Runs'] / bat_team_career_df['Out']).round(2).fillna(0)
            bat_team_career_df['SR'] = ((bat_team_career_df['Runs'] / bat_team_career_df['Balls']) * 100).round(2).fillna(0)
            bat_team_career_df['BPO'] = (bat_team_career_df['Balls'] / bat_team_career_df['Out']).round(2).fillna(0)

            # Calculate team statistics
            bat_team_career_df['Team Avg'] = (bat_team_career_df['Team Runs'] / bat_team_career_df['Wickets']).round(2).fillna(0)
            bat_team_career_df['Team SR'] = (bat_team_career_df['Team Runs'] / bat_team_career_df['Team Balls'] * 100).round(2).fillna(0)

            # Calculate P+ Avg and P+ SR
            bat_team_career_df['P+ Avg'] = (bat_team_career_df['Avg'] / bat_team_career_df['Team Avg'] * 100).round(2).fillna(0)
            bat_team_career_df['P+ SR'] = (bat_team_career_df['SR'] / bat_team_career_df['Team SR'] * 100).round(2).fillna(0)

            # Calculate BPB (Balls Per Boundary)
            bat_team_career_df['BPB'] = (bat_team_career_df['Balls'] / (bat_team_career_df['4s'] + bat_team_career_df['6s']).replace(0, 1)).round(2)

            # Calculate percentage statistics
            bat_team_career_df['50+PI'] = (((bat_team_career_df['50s'] + bat_team_career_df['100s'] + bat_team_career_df['150s'] + bat_team_career_df['200s']) / bat_team_career_df['Inns']) * 100).round(2).fillna(0)
            bat_team_career_df['100PI'] = (((bat_team_career_df['100s'] + bat_team_career_df['150s'] + bat_team_career_df['200s']) / bat_team_career_df['Inns']) * 100).round(2).fillna(0)
            bat_team_career_df['150PI'] = (((bat_team_career_df['150s'] + bat_team_career_df['200s']) / bat_team_career_df['Inns']) * 100).round(2).fillna(0)
            bat_team_career_df['200PI'] = ((bat_team_career_df['200s'] / bat_team_career_df['Inns']) * 100).round(2).fillna(0)
            bat_team_career_df['<25&OutPI'] = ((bat_team_career_df['<25&Out'] / bat_team_career_df['Inns']) * 100).round(2).fillna(0)

            # Calculate dismissal percentages
            bat_team_career_df['Caught%'] = ((bat_team_career_df['Caught'] / bat_team_career_df['Inns']) * 100).round(2).fillna(0)
            bat_team_career_df['Bowled%'] = ((bat_team_career_df['Bowled'] / bat_team_career_df['Inns']) * 100).round(2).fillna(0)
            bat_team_career_df['LBW%'] = ((bat_team_career_df['LBW'] / bat_team_career_df['Inns']) * 100).round(2).fillna(0)
            bat_team_career_df['Run Out%'] = ((bat_team_career_df['Run Out'] / bat_team_career_df['Inns']) * 100).round(2).fillna(0)
            bat_team_career_df['Stumped%'] = ((bat_team_career_df['Stumped'] / bat_team_career_df['Inns']) * 100).round(2).fillna(0)
            bat_team_career_df['Not Out%'] = ((bat_team_career_df['Not Out'] / bat_team_career_df['Inns']) * 100).round(2).fillna(0)

            # Reorder columns (removed Wins and Win% from the list)
            bat_team_career_df = bat_team_career_df[['Team', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 'Runs', 'HS', 
                                        'Avg', 'BPO', 'SR', '4s', '6s', 'BPB', '<25&Out', '50s', '100s', '150s', '200s',
                                        '<25&OutPI', '50+PI', '100PI', '150PI', '200PI', 'P+ Avg', 'P+ SR', 
                                        'Caught%', 'Bowled%', 'LBW%', 'Run Out%', 'Stumped%', 'Not Out%']]

            # Sort the DataFrame by 'Runs' in descending order
            bat_team_career_df = bat_team_career_df.sort_values(by='Runs', ascending=False)

            # Display the filtered and aggregated team career statistics
            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Team Career Statistics</h3>", unsafe_allow_html=True)
            st.markdown(
                """
                <style>
                /* Make the first column of any table sticky */
                .stDataFrame table tbody tr :first-child, 
                .stDataFrame table thead tr :first-child {
                    position: sticky;
                    left: 0;
                    background: white;
                    z-index: 1;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.dataframe(
                bat_team_career_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Team": st.column_config.Column("Team", pinned=True)
                }
            )

            # Create a new figure for the scatter plot
            scatter_fig = go.Figure()

            # Plot data for each team
            for team in bat_team_career_df['Team'].unique():
                team_stats = bat_team_career_df[bat_team_career_df['Team'] == team]
                
                # Get team statistics
                batting_avg = team_stats['Avg'].iloc[0]
                strike_rate = team_stats['SR'].iloc[0]
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
            st.plotly_chart(scatter_fig, use_container_width=True)

        # Season Stats Subtab
        with batting_subtabs[1]:
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
            st.dataframe(
                bat_team_season_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Team": st.column_config.Column("Team", pinned=True)
                }
            )

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
            st.plotly_chart(scatter_fig, key="team_season_scatter", use_container_width=True)

        # Opposition Stats Subtab
        with batting_subtabs[2]:
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

            # Calculate metrics
            bat_team_opponent_df['Avg'] = (bat_team_opponent_df['Runs'] / bat_team_opponent_df['Out']).round(2).fillna(0)
            bat_team_opponent_df['SR'] = ((bat_team_opponent_df['Runs'] / bat_team_opponent_df['Balls']) * 100).round(2).fillna(0)
            bat_team_opponent_df['BPO'] = (bat_team_opponent_df['Balls'] / bat_team_opponent_df['Out']).round(2).fillna(0)
            bat_team_opponent_df['Team Avg'] = (bat_team_opponent_df['Team Runs'] / bat_team_opponent_df['Wickets']).round(2).fillna(0)
            bat_team_opponent_df['Team SR'] = (bat_team_opponent_df['Team Runs'] / bat_team_opponent_df['Team Balls'] * 100).round(2).fillna(0)
            bat_team_opponent_df['P+ Avg'] = (bat_team_opponent_df['Avg'] / bat_team_opponent_df['Team Avg'] * 100).round(2).fillna(0)
            bat_team_opponent_df['P+ SR'] = (bat_team_opponent_df['SR'] / bat_team_opponent_df['Team SR'] * 100).round(2).fillna(0)
            
            # Calculate dismissal percentages and other metrics
            bat_team_opponent_df['BPB'] = (bat_team_opponent_df['Balls'] / (bat_team_opponent_df['4s'] + bat_team_opponent_df['6s']).replace(0, 1)).round(2)
            bat_team_opponent_df['50+PI'] = (((bat_team_opponent_df['50s'] + bat_team_opponent_df['100s']) / bat_team_opponent_df['Inns']) * 100).round(2).fillna(0)
            bat_team_opponent_df['100PI'] = ((bat_team_opponent_df['100s'] / bat_team_opponent_df['Inns']) * 100).round(2).fillna(0)
            bat_team_opponent_df['<25&OutPI'] = ((bat_team_opponent_df['<25&Out'] / bat_team_opponent_df['Inns']) * 100).round(2).fillna(0)
            
            # Sort for better presentation
            bat_team_opponent_df = bat_team_opponent_df.sort_values(by=['Team', 'Runs'], ascending=[True, False])

            # Display Opponents Stats
            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Opposition Statistics</h3>", unsafe_allow_html=True)
            st.dataframe(
                bat_team_opponent_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Team": st.column_config.Column("Team", pinned=True)
                }
            )

            # Calculate opponent averages
            opponent_stats_df = filtered_bat_df.groupby('Bowl_Team_y').agg({
                'Runs': 'sum',
                'Out': 'sum',
                'Balls': 'sum'
            }).reset_index()
            opponent_stats_df['Avg'] = (opponent_stats_df['Runs'] / opponent_stats_df['Out']).round(2)
            opponent_stats_df['SR'] = ((opponent_stats_df['Runs'] / opponent_stats_df['Balls']) * 100).round(2)

            # Create bar chart for opponent stats
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=opponent_stats_df['Bowl_Team_y'], 
                    y=opponent_stats_df['Avg'], 
                    name='Average', 
                    marker_color='#f84e4e',
                    text=opponent_stats_df['Avg'].round(2),
                    textposition='auto'
                )
            )

            # Update layout for opponent stats chart
            fig.update_layout(
                showlegend=True,
                height=500,
                xaxis_title="Opposition Team",
                yaxis_title="Average",
                font=dict(size=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis={'categoryorder': 'total ascending'}
            )

            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

            st.plotly_chart(fig, use_container_width=True)

        # Location Stats Subtab
        with batting_subtabs[3]:
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
            st.dataframe(
                bat_team_location_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Team": st.column_config.Column("Team", pinned=True)
                }
            )

            # Calculate location-based team averages
            location_stats_df = bat_team_location_df.groupby('Location').agg({
                'Runs': 'sum',
                'Out': 'sum',
                'Balls': 'sum'
            }).reset_index()
            location_stats_df['Avg'] = (location_stats_df['Runs'] / location_stats_df['Out']).round(2)
            location_stats_df['SR'] = ((location_stats_df['Runs'] / location_stats_df['Balls']) * 100).round(2)

            # Create bar chart for average runs per out by location
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=location_stats_df['Location'], 
                    y=location_stats_df['Avg'], 
                    name='Average', 
                    marker_color='#f84e4e',
                    text=location_stats_df['Avg'].round(2),
                    textposition='auto'
                )
            )

            # Update layout for location chart
            fig.update_layout(
                showlegend=False,
                height=500,
                xaxis_title='Location',
                yaxis_title='Average',
                font=dict(size=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis={'categoryorder': 'total ascending'}
            )

            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

            st.plotly_chart(fig, use_container_width=True)

        # Position Stats Subtab
        with batting_subtabs[4]:
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
            bat_team_position_df = bat_team_position_df.sort_values(by=['Team', 'Position'])

            # Display Position Stats
            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Position Statistics</h3>", unsafe_allow_html=True)
            st.dataframe(
                bat_team_position_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Team": st.column_config.Column("Team", pinned=True)
                }
            )

            # Create position averages graph - switch to horizontal bar chart to better show position data
            fig = go.Figure()

            # Calculate position-based team averages
            position_stats_df = bat_team_position_df.groupby('Position').agg({
                'Runs': 'sum',
                'Out': 'sum',
                'Inns': 'sum'
            }).reset_index()
            position_stats_df['Avg'] = (position_stats_df['Runs'] / position_stats_df['Out']).round(2)
            position_stats_df['Position'] = position_stats_df['Position'].astype(int)

            # Sort by position for the graph
            position_stats_df = position_stats_df.sort_values('Position')

            # Create bar chart for average by position
            fig.add_trace(
                go.Bar(
                    y=position_stats_df['Position'],  # Switch to Y axis for horizontal bars
                    x=position_stats_df['Avg'],       # Switch to X axis for values
                    orientation='h',                   # Make bars horizontal
                    name='Average', 
                    marker_color='#f84e4e',
                    text=position_stats_df['Avg'].round(2),
                    textposition='auto'
                )
            )

            # Update layout for position chart
            fig.update_layout(
                showlegend=False,
                height=500,
                yaxis_title='Position',  # Switched axis titles
                xaxis_title='Average',
                font=dict(size=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis={
                    'categoryorder':'array',
                    'categoryarray': sorted(position_stats_df['Position'].unique()),
                    'dtick': 1
                }
            )

            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

            st.plotly_chart(fig, use_container_width=True)

###-------------------------------------BOWLING STATS-------------------------------------###

    # Bowling Statistics Tab
    with tab2:
        # Create subtabs for different bowling statistics views
        bowling_subtabs = st.tabs(["Career", "Season", "Opposition", "Location", "Team Position"])
        
        # Career Statistics Subtab
        with bowling_subtabs[0]:
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
            
            # Sort by bowling average for better presentation
            bowl_team_df = bowl_team_df.sort_values(by='Avg')

            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Team Bowling Career Statistics</h3>", unsafe_allow_html=True)
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
            st.plotly_chart(scatter_fig, use_container_width=True)

        # Season Stats Subtab
        with bowling_subtabs[1]:
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
            
            # Sort by Year (descending) and then by Average
            bowl_team_season_df = bowl_team_season_df.sort_values(by=['Year', 'Avg'], ascending=[False, True])

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
                    text=[f"{row['Bowl_Team']} ({row['Year']})"],
                    textposition='top center',
                    marker=dict(size=10),
                    name=f"{row['Bowl_Team']} {row['Year']}",
                    hovertemplate=(
                        f"<b>{row['Bowl_Team']} ({row['Year']})</b><br><br>"
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
            st.plotly_chart(scatter_fig, key="bowl_season_scatter", use_container_width=True)
            
            # Create a bar chart to show wickets per year
            fig = go.Figure()
            
            # Group by Team and Year to get total wickets
            wickets_per_year = filtered_bowl_df.groupby(['Bowl_Team', 'Year'])['Bowler_Wkts'].sum().reset_index()
            
            # Get unique teams
            unique_teams = wickets_per_year['Bowl_Team'].unique()
            
            # Add a trace for each team
            for team in unique_teams:
                team_data = wickets_per_year[wickets_per_year['Bowl_Team'] == team]
                fig.add_trace(go.Bar(
                    x=team_data['Year'],
                    y=team_data['Bowler_Wkts'],
                    name=team
                ))
            
            # Update layout
            fig.update_layout(
                title="Wickets Taken by Team per Year",
                xaxis_title="Year",
                yaxis_title="Wickets",
                barmode='group',
                height=500,
                font=dict(size=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=True
            )
            
            # Add gridlines
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
            
            # Show plot
            st.plotly_chart(fig, use_container_width=True)

        # Opposition Stats Subtab
        with bowling_subtabs[2]:
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
            bowl_team_opponent_df['Avg'] = (bowl_team_opponent_df['Runs'] / bowl_team_opponent_df['Wickets']).round(2)
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
                'Bowl_Team', 'Opposition', 'Matches', 'Balls', 'Overs', 'M/D', 'Runs', 'Wickets', 'Avg',
                'Strike Rate', 'Economy Rate', '5W', '10W', 'WPM'
            ]]
            
            # Sort by bowling team and then average
            bowl_team_opponent_df = bowl_team_opponent_df.sort_values(by=['Bowl_Team', 'Avg'])

            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Opposition Statistics</h3>", unsafe_allow_html=True)
            st.dataframe(
                bowl_team_opponent_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Bowl_Team": st.column_config.Column("Team", pinned=True)
                }
            )

            # Extract unique teams from bowl_team_opponent_df
            team_choice = bowl_team_opponent_df['Bowl_Team'].unique().tolist()

            # Create opponent averages graph
            fig = go.Figure()

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
                    marker_color='#FF4B4B',
                    text=all_opponent_stats['Wickets'],
                    textposition='auto'
                )
            )

            # Update layout
            fig.update_layout(
                title="Average Against Opposition Teams",
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
            st.plotly_chart(fig, use_container_width=True)

        # Location Stats Subtab
        with bowling_subtabs[3]:
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
            location_summary['Avg'] = (location_summary['Runs'] / location_summary['Wickets']).round(2)
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
                'Team', 'Location', 'Matches', 'Balls', 'Overs', 'M/D', 'Runs', 'Wickets', 'Avg',
                'Strike Rate', 'Economy Rate', '5W', '10W', 'WPM'
            ]]
            
            # Sort by team and then average
            location_summary = location_summary.sort_values(by=['Team', 'Avg'])

            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Location Statistics</h3>", unsafe_allow_html=True)
            st.dataframe(
                location_summary,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Team": st.column_config.Column("Team", pinned=True)
                }
            )

            # Create location averages graph
            fig = go.Figure()

            # Get unique teams from location_summary
            team_choice = location_summary['Team'].unique().tolist()

            # Calculate aggregate stats for all locations
            all_location_stats = location_summary.groupby(['Location']).agg({
                'Runs': 'sum',   
                'Wickets': 'sum',  
            }).reset_index()

            # Calculate bowling average
            all_location_stats['Average'] = (all_location_stats['Runs'] / all_location_stats['Wickets']).round(2)

            # Add trace for all locations
            fig.add_trace(
                go.Bar(
                    x=all_location_stats['Location'],
                    y=all_location_stats['Average'],
                    name='All Teams',
                    marker_color='#FF4B4B',
                    text=all_location_stats['Average'].round(2),
                    textposition='auto'
                )
            )

            # Update layout
            fig.update_layout(
                title="Average by Location",
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
            st.plotly_chart(fig, key="location_average_chart", use_container_width=True)

        # Team Position Stats Subtab
        with bowling_subtabs[4]:
            # Calculate statistics dataframe for team and position
            team_position_summary = filtered_bowl_df.groupby(['Bowl_Team', 'Position']).agg({
                'File Name': 'nunique',      # Matches
                'Bowler_Balls': 'sum',
                'Maidens': 'sum',
                'Bowler_Runs': 'sum',
                'Bowler_Wkts': 'sum'
            }).reset_index()

            team_position_summary.columns = ['Bowl_Team', 'Position', 'Matches', 'Balls', 'M/D', 'Runs', 'Wickets']

            # Calculate metrics
            team_position_summary['Overs'] = (team_position_summary['Balls'] // 6) + (team_position_summary['Balls'] % 6) / 10
            team_position_summary['Strike Rate'] = (team_position_summary['Balls'] / team_position_summary['Wickets']).round(2)
            team_position_summary['Economy Rate'] = (team_position_summary['Runs'] / team_position_summary['Overs']).round(2)
            team_position_summary['Avg'] = (team_position_summary['Runs'] / team_position_summary['Wickets']).round(2)
            team_position_summary['WPM'] = (team_position_summary['Wickets'] / team_position_summary['Matches']).round(2)

            # Count 5W innings - FIX: Use Bowl_Team instead of Team
            five_wickets = filtered_bowl_df[filtered_bowl_df['Bowler_Wkts'] >= 5].groupby(['Bowl_Team', 'Position']).size().reset_index(name='5W')
            team_position_summary = team_position_summary.merge(five_wickets, on=['Bowl_Team', 'Position'], how='left')
            team_position_summary['5W'] = team_position_summary['5W'].fillna(0).astype(int)

            # Count 10W matches - FIX: Use Bowl_Team instead of Team
            match_wickets = filtered_bowl_df.groupby(['Bowl_Team', 'Position', 'File Name'])['Bowler_Wkts'].sum().reset_index()
            ten_wickets = match_wickets[match_wickets['Bowler_Wkts'] >= 10].groupby(['Bowl_Team', 'Position']).size().reset_index(name='10W')
            team_position_summary = team_position_summary.merge(ten_wickets, on=['Bowl_Team', 'Position'], how='left')
            team_position_summary['10W'] = team_position_summary['10W'].fillna(0).astype(int)

            # Handle infinities and NaNs
            team_position_summary = team_position_summary.replace([np.inf, -np.inf], np.nan)

            # Final column ordering
            team_position_summary = team_position_summary[[
                'Bowl_Team', 'Position', 'Matches', 'Balls', 'Overs', 'M/D', 'Runs', 'Wickets', 'Avg',
                'Strike Rate', 'Economy Rate', '5W', '10W', 'WPM'
            ]]
            
            # Sort by Team name and Position
            team_position_summary = team_position_summary.sort_values(by=['Bowl_Team', 'Position'])

            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Team Position Statistics</h3>", unsafe_allow_html=True)
            st.dataframe(
                team_position_summary,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Bowl_Team": st.column_config.Column("Team", pinned=True)
                }
            )

            # Create a visualization for bowling averages by team position
            fig = go.Figure()

            # Get unique teams from team_position_summary
            unique_teams = sorted(team_position_summary['Bowl_Team'].unique())
            
            # Create color scale for teams
            colors = [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in range(len(unique_teams))]
            
            # Add the first team in the main color
            if len(unique_teams) > 0:
                colors[0] = '#f84e4e'  # Streamlit red for the first team
            
            # Create traces for each team
            for i, team in enumerate(unique_teams):
                team_data = team_position_summary[team_position_summary['Bowl_Team'] == team]
                team_data = team_data.sort_values('Position')
                
                fig.add_trace(go.Scatter(
                    x=team_data['Position'],
                    y=team_data['Avg'],
                    mode='lines+markers',
                    name=team,
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=8),
                    hovertemplate=(
                        f"Team: {team}<br>"
                        "Position: %{x}<br>"
                        "Average: %{y:.2f}<br>"
                        "<extra></extra>"
                    )
                ))

            # Update layout
            fig.update_layout(
                title="Bowling Average by Team Position",
                showlegend=True,
                height=500,
                xaxis_title="Bowling Position",
                yaxis_title="Average",
                font=dict(size=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis={
                    'tickmode': 'linear',
                    'tick0': 1,
                    'dtick': 1,
                    'range': [0.5, 11.5]  # Positions 1-11
                },
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            # Add gridlines
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a bar chart view for easier comparison of positions across teams
            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Average by Position for Each Team</h3>", unsafe_allow_html=True)
            
            # Let user select a position to compare across teams
            selected_position = st.selectbox(
                'Select Position to Compare',
                sorted(team_position_summary['Position'].unique()),
                index=0
            )
            
            # Filter data for the selected position
            position_data = team_position_summary[team_position_summary['Position'] == selected_position]
            position_data = position_data.sort_values('Avg')  # Sort by average for better visualization
            
            # Create bar chart
            pos_fig = go.Figure()
            pos_fig.add_trace(go.Bar(
                x=position_data['Bowl_Team'],
                y=position_data['Avg'],
                marker_color='#f84e4e',
                text=position_data['Wickets'],
                textposition='auto',
                hovertemplate=(
                    "Team: %{x}<br>"
                    "Average: %{y:.2f}<br>"
                    "Wickets: %{text}<br>"
                    "<extra></extra>"
                )
            ))
            
            # Update layout
            pos_fig.update_layout(
                showlegend=False,
                height=400,
                xaxis_title="Team",
                yaxis_title="Average",
                font=dict(size=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            pos_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
            pos_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
            
            # Display bar chart
            st.plotly_chart(pos_fig, use_container_width=True)

# No need for the if __name__ == "__main__" part
display_team_view()
