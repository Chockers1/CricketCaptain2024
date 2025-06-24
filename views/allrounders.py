# Imports and Setup
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

def display_ar_view():    # Custom styling
    st.markdown("""
    <style>
    .stSlider p { color: #f04f53 !important; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 style='color:#f04f53; text-align: center;'>All Rounder Statistics</h1>", unsafe_allow_html=True)

    if 'bat_df' in st.session_state and 'bowl_df' in st.session_state:
        # Make copies of original dataframes
        bat_df = st.session_state['bat_df'].copy()
        bowl_df = st.session_state['bowl_df'].copy()

        # Check if only one scorecard is loaded
        unique_matches_bat = bat_df['File Name'].nunique()
        unique_matches_bowl = bowl_df['File Name'].nunique()
        if unique_matches_bat <= 1 or unique_matches_bowl <= 1:
            st.warning("⚠️ Please upload more than 1 scorecard to use the all-rounder statistics view effectively. With only one match loaded, statistical analysis and comparisons are limited.")
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
            st.error("No valid dates found in the data.")
            return

        # Create first row of filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            name_choice = st.multiselect('Name:', names, default='All')
        with col2:
            bat_team_choice = st.multiselect('Batting Team:', bat_teams, default='All')
        with col3:
            bowl_team_choice = st.multiselect('Bowling Team:', bowl_teams, default='All')
        with col4:
            match_format_choice = st.multiselect('Format:', match_formats, default='All')

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

        # Create range filter columns
        col5, col6, col7, col8, col9, col10, col11, col12 = st.columns(8)

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
                    label_visibility='collapsed')

        with col6:
            st.markdown("<p style='text-align: center;'>Matches Range</p>", unsafe_allow_html=True)
            matches_range = st.slider('', 
                                    min_value=1, 
                                    max_value=max_matches, 
                                    value=(1, max_matches),
                                    label_visibility='collapsed')

        with col7:
            st.markdown("<p style='text-align: center;'>Runs Range</p>", unsafe_allow_html=True)
            runs_range = st.slider('', 
                                min_value=1, 
                                max_value=max_runs, 
                                value=(1, max_runs),
                                label_visibility='collapsed')

        with col8:
            st.markdown("<p style='text-align: center;'>Wickets Range</p>", unsafe_allow_html=True)
            wickets_range = st.slider('', 
                                    min_value=0, 
                                    max_value=max_wickets, 
                                    value=(0, max_wickets),
                                    label_visibility='collapsed')

        with col9:
            st.markdown("<p style='text-align: center;'>Batting Average</p>", unsafe_allow_html=True)
            bat_avg_range = st.slider('', 
                                    min_value=0.0, 
                                    max_value=max_bat_avg, 
                                    value=(0.0, max_bat_avg),
                                    label_visibility='collapsed')

        with col10:
            st.markdown("<p style='text-align: center;'>Bowling Average</p>", unsafe_allow_html=True)
            bowl_avg_range = st.slider('', 
                                    min_value=0.0, 
                                    max_value=max_bowl_avg, 
                                    value=(0.0, max_bowl_avg),
                                    label_visibility='collapsed')

        with col11:
            st.markdown("<p style='text-align: center;'>Batting SR</p>", unsafe_allow_html=True)
            bat_sr_range = st.slider('', 
                                    min_value=0.0, 
                                    max_value=max_bat_sr, 
                                    value=(0.0, max_bat_sr),
                                    label_visibility='collapsed')

        with col12:
            st.markdown("<p style='text-align: center;'>Bowling SR</p>", unsafe_allow_html=True)
            bowl_sr_range = st.slider('', 
                                    min_value=0.0, 
                                    max_value=max_bowl_sr, 
                                    value=(0.0, max_bowl_sr),
                                    label_visibility='collapsed')

        # Apply year filter
        filtered_bat_df = filtered_bat_df[
            filtered_bat_df['Year'].between(year_choice[0], year_choice[1])
        ]
        filtered_bowl_df = filtered_bowl_df[
            filtered_bowl_df['Year'].between(year_choice[0], year_choice[1])
        ]

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

        # Display Career Statistics
        st.markdown(
            "<h3 style='color:#f04f53; text-align: center;'>Career Statistics</h3>", 
            unsafe_allow_html=True
        )
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

        # Display Season Statistics
        st.markdown(
            "<h3 style='color:#f04f53; text-align: center;'>Season Statistics</h3>", 
            unsafe_allow_html=True
        )
        st.dataframe(filtered_season_stats, use_container_width=True, hide_index=True)

###-------------------------------------GRAPHS----------------------------------------###
        # Get individual players
        individual_players = [name for name in name_choice if name != 'All']

        # Create subplots for Batting and Bowling Averages
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Batting Average", "Bowling Average"))

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

        # Update layout for second graph
        fig.update_layout(
            showlegend=True,
            height=500,
            yaxis_title="Runs",
            yaxis2_title="Wickets",
            font=dict(size=12),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            barmode='group'
        )

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

        st.plotly_chart(fig)

###################-------------------------------------GRAPHS----------------------------------------#############

        # Scatter plot of Batting Average vs Bowling Average with Name labels
        if 'All' in name_choice:
            # If 'All' is selected, include all players from the filtered dataset
            all_players = filtered_season_stats['Name'].unique()
        else:
            all_players = individual_players  # Use selected individual players

        if not all_players.any():
            st.error("No players available to display.")
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
            # Display the title using Streamlit's markdown
            st.markdown(
                "<h3 style='color:#f04f53; text-align: center;'>Batting Average vs Bowling Average</h3>", 
                unsafe_allow_html=True
            )

            # Update layout for scatter plot
            scatter_fig.update_layout(
                xaxis_title="Batting Average",
                yaxis_title="Bowling Average",
                height=500,
                font=dict(size=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False  # No need for legend, player names will be shown as labels
            )

            # Show scatter plot
            st.plotly_chart(scatter_fig)
    else:
        st.error("Required data not found. Please ensure you have processed the scorecards.")



# No need for the if __name__ == "__main__" part
display_ar_view()