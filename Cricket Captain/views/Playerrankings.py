import streamlit as st 
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

def calculate_batter_rating_per_match(df):
    """
    Calculate batting rating with 1 point per run plus milestone and strike rate bonuses/penalties
    """
    stats = df.copy()
    
    # Calculate strike rate
    stats['Strike_Rate'] = (stats['Runs'] / stats['Balls']) * 100
    
    # Base rating is 1 point per run
    stats['Base_Score'] = stats['Runs']
    
    # Calculate milestone bonuses
    def calculate_bonus(row):
        runs = row['Runs']
        sr = row['Strike_Rate']
        
        # Initialize bonus
        bonus = 0
        
        # Milestone bonuses
        if 100 <= runs < 150:
            bonus += 50  # Bonus for century
        elif 150 <= runs < 200:
            bonus += 75  # Bonus for 150+
        elif runs >= 200:
            bonus += 100  # Bonus for double century
            
        # Strike rate bonuses/penalties
        if runs >= 40:  # Only apply SR bonus/penalty for substantial innings
            if sr >= 75:
                bonus += 25  # Bonus for good scoring rate
            elif sr <= 40:
                bonus -= 25  # Penalty for very slow scoring
                
        return bonus
    
    # Apply bonuses
    stats['Bonus'] = stats.apply(calculate_bonus, axis=1)
    
    # Calculate final rating
    stats['Batting_Rating'] = stats['Base_Score'] + stats['Bonus']
        
    return stats[['File Name', 'Name', 'Batting_Rating', 'Runs', 'Strike_Rate']]

def calculate_bowler_rating_per_match(df):
    """
    Calculate bowling rating with base points for wickets and maidens, 
    plus bonuses for milestone wickets and economy rate
    """
    stats = df.copy()
    
    # Calculate basic metrics
    stats['Overs'] = stats['Bowler_Balls'] / 6
    stats['Economy'] = (stats['Bowler_Runs'] / stats['Overs']).replace(float('inf'), 0)
    
    # Base points: 20 points per wicket and 2 points per maiden
    stats['Base_Score'] = (stats['Bowler_Wkts'] * 20) + (stats['Maidens'] * 2)
    
    def calculate_wicket_bonus(wickets):
        """Calculate bonus points for wicket milestones"""
        if wickets == 10:
            return 260
        elif wickets == 9:
            return 200
        elif wickets == 8:
            return 150
        elif wickets == 7:
            return 100
        elif wickets == 6:
            return 75
        elif wickets == 5:
            return 50
        elif wickets == 4:
            return 35
        elif wickets == 3:
            return 20
        else:
            return 0
    
    def calculate_economy_bonus(row):
        """Calculate bonus/penalty for economy rate if bowled 10+ overs"""
        if row['Overs'] >= 10:
            if row['Economy'] <= 2.5:
                return 25  # Bonus for good economy
            elif row['Economy'] >= 4.5:
                return -25  # Penalty for poor economy
        return 0
    
    # Apply bonuses
    stats['Wicket_Bonus'] = stats['Bowler_Wkts'].apply(calculate_wicket_bonus)
    stats['Economy_Bonus'] = stats.apply(calculate_economy_bonus, axis=1)
    
    # Calculate final rating
    stats['Bowling_Rating'] = stats['Base_Score'] + stats['Wicket_Bonus'] + stats['Economy_Bonus']
    
    return stats[['File Name', 'Name', 'Bowling_Rating', 'Bowler_Wkts', 'Economy', 'Maidens']]

def display_ar_view():
    # Add the title
    st.markdown("<h1 style='color:#f04f53; text-align: center;'>Player Rankings</h1>", unsafe_allow_html=True)
    
    # Custom styling
    st.markdown("""<style>
    .stSlider p { color: #f04f53 !important; }
    table { color: black; width: 100%; }
    thead tr th {
        background-color: #f04f53 !important;
        color: white !important;
    }
    tbody tr:nth-child(even) { background-color: #f2f2f2; }
    tbody tr:nth-child(odd) { background-color: white; }
    .block-container {
        max-width: 100%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    </style>""", unsafe_allow_html=True)
    
    # Check if required DataFrames exist in session state
    if 'bat_df' not in st.session_state or 'bowl_df' not in st.session_state:
        st.error("Required data not found in session state")
        return

    bat_df = st.session_state['bat_df'].copy()
    bowl_df = st.session_state['bowl_df'].copy()

    # Convert Date column to datetime
    if 'Date' in bat_df.columns:
        bat_df['Date'] = pd.to_datetime(bat_df['Date'], errors='coerce')
        bat_df['Year'] = bat_df['Date'].dt.year

    # Ensure Year is integer type
    bat_df['Year'] = pd.to_numeric(bat_df['Year'], errors='coerce').fillna(0).astype(int)

    # Rename Bat_Team_y to Team for clarity
    bat_df = bat_df.rename(columns={'Bat_Team_y': 'Team'})

    # Process batting data for rankings
    if 'Runs' in bat_df.columns:
        player_ranking_match = bat_df.groupby(['Year', 'File Name', 'Name', 'Team']).agg({
            'Runs': 'sum',
            'Balls': 'sum',
            '4s': 'sum',
            '6s': 'sum',
            'Batted': 'sum',
            'Out': 'sum',
            '50s': 'sum',
            '100s': 'sum',
            'Date': 'first'
        }).reset_index()

        # Calculate milestone counts
        temp_df = bat_df.groupby(['File Name', 'Name'])['Runs'].agg(list).reset_index()
        temp_df['200s'] = temp_df['Runs'].apply(lambda x: sum(1 for r in x if r >= 200))
        temp_df['300s_plus'] = temp_df['Runs'].apply(lambda x: sum(1 for r in x if r >= 300))

        # Merge milestone counts
        player_ranking_match = pd.merge(
            player_ranking_match,
            temp_df[['File Name', 'Name', '200s', '300s_plus']],
            on=['File Name', 'Name'],
            how='left'
        )
        
        # Fill NaN values
        player_ranking_match[['200s', '300s_plus']] = player_ranking_match[['200s', '300s_plus']].fillna(0)
    else:
        st.error("'Runs' column not found in batting data")
        return

    # Process bowling data
    bowling_stats = bowl_df.groupby(['File Name', 'Name']).agg({
        'Bowler_Balls': 'sum',
        'Maidens': 'sum',
        'Bowler_Runs': 'sum',
        'Bowler_Wkts': 'sum',
        '5Ws': 'sum',
    }).reset_index()

    # Merge bowling stats into player_ranking_match
    player_ranking_match = pd.merge(
        player_ranking_match,
        bowling_stats,
        on=['File Name', 'Name'],
        how='outer'
    )

    # Fill NaN values
    bowling_columns = ['Bowler_Balls', 'Maidens', 'Bowler_Runs', 'Bowler_Wkts', '5Ws']
    batting_columns = ['Runs', 'Balls', '4s', '6s', 'Batted', 'Out', '50s', '100s', '200s', '300s_plus']
    player_ranking_match[bowling_columns] = player_ranking_match[bowling_columns].fillna(0)
    player_ranking_match[batting_columns] = player_ranking_match[batting_columns].fillna(0)

    # Add match format if available
    if 'match_df' in st.session_state:
        match_df = st.session_state['match_df'].copy()
        format_info = match_df[['File Name', 'Match_Format']].drop_duplicates()
        player_ranking_match = pd.merge(
            player_ranking_match,
            format_info,
            on='File Name',
            how='left'
        )

    # Calculate ratings
    batting_ratings = calculate_batter_rating_per_match(player_ranking_match)
    bowling_ratings = calculate_bowler_rating_per_match(player_ranking_match)

    # Merge ratings
    player_ranking_match = pd.merge(
        player_ranking_match,
        batting_ratings[['File Name', 'Name', 'Batting_Rating']],
        on=['File Name', 'Name'],
        how='left'
    )

    player_ranking_match = pd.merge(
        player_ranking_match,
        bowling_ratings[['File Name', 'Name', 'Bowling_Rating']],
        on=['File Name', 'Name'],
        how='left'
    )

    # Sort and add match number
    player_ranking_match = player_ranking_match.sort_values(['Date', 'Name'])
    player_ranking_match['Match Number'] = player_ranking_match.groupby(['Name', 'Match_Format']).cumcount() + 1

    # Round ratings
    rating_columns = ['Batting_Rating', 'Bowling_Rating']
    player_ranking_match[rating_columns] = player_ranking_match[rating_columns].round(2)

    # Create tabs
    # Add this CSS before creating the tabs
    st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            justify-content: space-between;
        }
        .stTabs [data-baseweb="tab"] {
            width: 200px;
            white-space: pre-wrap;
            text-align: center;
        }
        .stTabs [data-baseweb="tab-border"] {
            display: none;
        }
    </style>""", unsafe_allow_html=True)

    # Then create your tabs
    tab1, tab2, tab3, tab4 = st.tabs(["#1 Ranking", "Batting Rankings", "Bowling Rankings", "All-Rounder Rankings"])
    with tab1:

# Create yearly summary of best performers
        st.markdown("<h3 style='text-align: center; color:#f04f53;'>Yearly Best Performers</h3>", unsafe_allow_html=True)
        
        # First group by Year and Name to get totals per player per year
        batting_by_player = player_ranking_match.groupby(['Year', 'Name'])['Batting_Rating'].sum().reset_index()
        # Then find the player with max total for each year
        best_batting = batting_by_player.groupby('Year').apply(
            lambda x: f"{x.loc[x['Batting_Rating'].idxmax(), 'Name']} - {x['Batting_Rating'].max():.0f}" 
            if x['Batting_Rating'].max() > 0 else "-"
        ).reset_index(name='Best_Batting')

        # Same for bowling
        bowling_by_player = player_ranking_match.groupby(['Year', 'Name'])['Bowling_Rating'].sum().reset_index()
        best_bowling = bowling_by_player.groupby('Year').apply(
            lambda x: f"{x.loc[x['Bowling_Rating'].idxmax(), 'Name']} - {x['Bowling_Rating'].max():.0f}" 
            if x['Bowling_Rating'].max() > 0 else "-"
        ).reset_index(name='Best_Bowling')

        # Calculate AR Rating for yearly summary
        yearly_ar = player_ranking_match.copy()
        yearly_ar['Batting_RPG'] = yearly_ar.groupby(['Year', 'Name'])['Batting_Rating'].transform('sum') / \
                                  yearly_ar.groupby(['Year', 'Name'])['File Name'].transform('nunique')
        yearly_ar['Bowling_RPG'] = yearly_ar.groupby(['Year', 'Name'])['Bowling_Rating'].transform('sum') / \
                                  yearly_ar.groupby(['Year', 'Name'])['File Name'].transform('nunique')
        
        # Apply AR qualification criteria
        min_batting_rpg = 20
        min_bowling_rpg = 20
        qualified = (yearly_ar['Batting_RPG'] >= min_batting_rpg) & \
                   (yearly_ar['Bowling_RPG'] >= min_bowling_rpg)
        
        # Calculate AR_Rating only for qualified players
        yearly_ar['AR_Rating'] = 0.0
        yearly_ar.loc[qualified, 'AR_Rating'] = \
            yearly_ar.loc[qualified, 'Batting_Rating'] + yearly_ar.loc[qualified, 'Bowling_Rating']

        # Group AR ratings by player and year to get totals
        ar_by_player = yearly_ar.groupby(['Year', 'Name'])['AR_Rating'].sum().reset_index()
        best_ar = ar_by_player.groupby('Year').apply(
            lambda x: f"{x.loc[x['AR_Rating'].idxmax(), 'Name']} - {x['AR_Rating'].max():.0f}" 
            if x['AR_Rating'].max() > 0 else "-"
        ).reset_index(name='Best_AllRounder')

        # Combine all summaries
        yearly_summary = pd.merge(best_batting, best_bowling, on='Year', how='outer')
        yearly_summary = pd.merge(yearly_summary, best_ar, on='Year', how='outer')
        
        # Sort by Year in descending order
        yearly_summary = yearly_summary.sort_values('Year', ascending=False)

# Display the summary with all rows visible
        st.dataframe(
            yearly_summary,
            use_container_width=True,hide_index=True,
            height=(len(yearly_summary) + 1) * 35,  # Calculate height based on number of rows
            column_config={
                "Year": st.column_config.NumberColumn(format="%d"),
                "Best_Batting": "Batting",
                "Best_Bowling": "Bowling",
                "Best_AllRounder": "All-Rounder"
            }
        )

    with tab2:
            # Create batting rankings
            batting_rankings = player_ranking_match.groupby(['Year', 'Name', 'Team']).agg({
                'File Name': 'nunique',  # Count matches
                'Batting_Rating': 'sum',
                'Runs': 'sum'
            }).reset_index()

            # Rename columns
            batting_rankings = batting_rankings.rename(columns={
                'File Name': 'Matches',
                'Batting_Rating': 'Rating',
                'Runs': 'Total_Runs'
            })

            # Calculate RPG
            batting_rankings['RPG'] = batting_rankings['Rating'] / batting_rankings['Matches']
            
            # Add rank within each year
            batting_rankings['Rank'] = batting_rankings.groupby('Year')['Rating'].rank(method='dense', ascending=False).astype(int)
            
            # Sort by Year and Rank
            batting_rankings = batting_rankings.sort_values(['Year', 'Rank'])

            # Round numeric columns and ensure Year is integer
            numeric_cols = ['Rating', 'RPG', 'Total_Runs']
            batting_rankings[numeric_cols] = batting_rankings[numeric_cols].round(2)
            batting_rankings['Year'] = batting_rankings['Year'].astype(int)

            # Add filters for batting rankings
            st.markdown("<h3 style='text-align: center; color:#f04f53;'>Filters</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                selected_names_bat = st.multiselect(
                    "Filter by Player Name",
                    options=sorted(batting_rankings['Name'].unique()),
                    key="batting_names"
                )
            with col2:
                selected_teams_bat = st.multiselect(
                    "Filter by Team",
                    options=sorted(batting_rankings['Team'].unique()),
                    key="batting_teams"
                )

            # Apply filters
            filtered_batting = batting_rankings.copy()
            if selected_names_bat:
                filtered_batting = filtered_batting[filtered_batting['Name'].isin(selected_names_bat)]
            if selected_teams_bat:
                filtered_batting = filtered_batting[filtered_batting['Team'].isin(selected_teams_bat)]

            # Sort by Year (descending) and Rank (ascending)
            filtered_batting = filtered_batting.sort_values(['Year', 'Rank'], ascending=[False, True])

            st.markdown("<h3 style='text-align: center; color:#f04f53;'>Batting Rankings</h3>", unsafe_allow_html=True)
            st.dataframe(
                filtered_batting,
                use_container_width=True,
                hide_index=True,
                height=850,  # This will show approximately 25 rows
                column_config={
                    "Year": st.column_config.NumberColumn(format="%d"),
                    "Rank": st.column_config.NumberColumn(format="%d")
                }
            )
# Create rating per game trend graph
            st.markdown("<h3 style='text-align: center; color:#f04f53;'>Rating Per Game Trends</h3>", unsafe_allow_html=True)
            fig1 = go.Figure()

            trend_data = filtered_batting.sort_values('Year')
            
            for player in trend_data['Name'].unique():
                player_data = trend_data[trend_data['Name'] == player]
                
                fig1.add_trace(go.Scatter(
                    x=player_data['Year'],
                    y=player_data['RPG'],
                    name=player,
                    mode='lines+markers',
                    line=dict(width=2),
                    hovertemplate="Year: %{x}<br>RPG: %{y:.2f}<br>Name: " + player + "<br>Team: %{customdata}<extra></extra>",
                    customdata=player_data['Team']
                ))

            fig1.update_layout(
                xaxis_title="Year",
                yaxis_title="Rating Per Game",
                hovermode='closest',
                showlegend=False,  # Remove legend
                margin=dict(l=20, r=20, t=40, b=40),  # Reduced bottom margin since no legend
                height=650,
                plot_bgcolor='rgba(255,255,255,0)',
                paper_bgcolor='rgba(255,255,255,0)',
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.2)',
                    gridwidth=1
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.2)',
                    gridwidth=1
                )
            )

            st.plotly_chart(fig1, use_container_width=True)

            # Create rank trend graph
            st.markdown("<h3 style='text-align: center; color:#f04f53;'>Rank Trends</h3>", unsafe_allow_html=True)
            fig2 = go.Figure()

            for player in trend_data['Name'].unique():
                player_data = trend_data[trend_data['Name'] == player]
                player_data = player_data.sort_values('Year')
                
                fig2.add_trace(go.Scatter(
                    x=player_data['Year'],
                    y=player_data['Rank'],
                    name=player,
                    mode='lines+markers',
                    line=dict(width=2),
                    hovertemplate="Year: %{x}<br>Rank: %{y}<br>Name: " + player + "<br>Team: %{customdata}<extra></extra>",
                    customdata=player_data['Team']
                ))

            fig2.update_layout(
                xaxis_title="Year",
                yaxis_title="Rank",
                hovermode='closest',
                showlegend=False,  # Remove legend
                margin=dict(l=20, r=20, t=40, b=40),  # Reduced bottom margin since no legend
                height=650,
                plot_bgcolor='rgba(255,255,255,0)',
                paper_bgcolor='rgba(255,255,255,0)',
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.2)',
                    gridwidth=1
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.2)',
                    gridwidth=1,
                    autorange="reversed"
                )
            )

            st.plotly_chart(fig2, use_container_width=True)

#########-------------------TAB 3 BOWLING RANKINGS------------------################

    with tab3:
        # Create bowling rankings
        bowling_rankings = player_ranking_match.groupby(['Year', 'Name', 'Team']).agg({
            'File Name': 'nunique',  # Count matches
            'Bowling_Rating': 'sum',
            'Bowler_Wkts': 'sum'
        }).reset_index()

        # Rename columns
        bowling_rankings = bowling_rankings.rename(columns={
            'File Name': 'Matches',
            'Bowler_Wkts': 'Total_Wickets'
        })

        # Calculate PPG and ranks by year
        bowling_rankings['PPG'] = bowling_rankings['Bowling_Rating'] / bowling_rankings['Matches']
        
        # Calculate ranks within each year based on Bowling_Rating
        bowling_rankings['Rank'] = bowling_rankings.groupby('Year')['Bowling_Rating'].rank(method='min', ascending=False)
        
        bowling_rankings = bowling_rankings.sort_values(['Year', 'Rank'])
        
        # Round numeric columns
        numeric_cols = ['Bowling_Rating', 'PPG', 'Total_Wickets']
        bowling_rankings[numeric_cols] = bowling_rankings[numeric_cols].round(2)

        # Add filters for bowling rankings
        st.markdown("<h3 style='text-align: center; color:#f04f53;'>Filters</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            selected_names_bowl = st.multiselect(
                "Filter by Player Name",
                options=sorted(bowling_rankings['Name'].unique()),
                key="bowling_names"
            )
        with col2:
            selected_teams_bowl = st.multiselect(
                "Filter by Team",
                options=sorted(bowling_rankings['Team'].unique()),
                key="bowling_teams"
            )

        # Apply filters
        filtered_bowling = bowling_rankings.copy()
        if selected_names_bowl:
            filtered_bowling = filtered_bowling[filtered_bowling['Name'].isin(selected_names_bowl)]
        if selected_teams_bowl:
            filtered_bowling = filtered_bowling[filtered_bowling['Team'].isin(selected_teams_bowl)]

        st.markdown("<h3 style='text-align: center; color:#f04f53;'>Bowling Rankings</h3>", unsafe_allow_html=True)
        st.dataframe(
            filtered_bowling, 
            use_container_width=True, 
            hide_index=True,
            height=850  # This will show approximately 25 rows
        )

        # Create Rating Per Game trend graph
        st.markdown("<h3 style='text-align: center; color:#f04f53;'>Rating Per Game Trends</h3>", unsafe_allow_html=True)
        fig1 = go.Figure()
        
        trend_data = filtered_bowling.sort_values('Year')
        
        for player in trend_data['Name'].unique():
            player_data = trend_data[trend_data['Name'] == player]
            fig1.add_trace(go.Scatter(
                x=player_data['Year'],
                y=player_data['PPG'],
                name=player,
                mode='lines+markers',
                line=dict(width=2),
                hovertemplate="Year: %{x}<br>RPG: %{y:.2f}<br>Name: " + player + "<br>Team: %{customdata}<extra></extra>",
                customdata=player_data['Team']
            ))
        
        fig1.update_layout(
            xaxis_title="Year",
            yaxis_title="Rating Per Game",
            hovermode='closest',
            showlegend=False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=20, r=20, t=40, b=200),
            height=650,
            plot_bgcolor='rgba(255,255,255,0)',
            paper_bgcolor='rgba(255,255,255,0)',
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                gridwidth=1
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                gridwidth=1
            )
        )
        
        st.plotly_chart(fig1, use_container_width=True)

        # Create Rank trend graph
        st.markdown("<h3 style='text-align: center; color:#f04f53;'>Rank Trends</h3>", unsafe_allow_html=True)
        fig2 = go.Figure()
        
        for player in trend_data['Name'].unique():
            player_data = trend_data[trend_data['Name'] == player]
            player_data = player_data.sort_values('Year')
            
            fig2.add_trace(go.Scatter(
                x=player_data['Year'],
                y=player_data['Rank'],
                name=player,
                mode='lines+markers',
                line=dict(width=2),
                hovertemplate="Year: %{x}<br>Rank: %{y}<br>Name: " + player + "<br>Team: %{customdata}<extra></extra>",
                customdata=player_data['Team']
            ))
        
        fig2.update_layout(
            xaxis_title="Year",
            yaxis_title="Rank",
            hovermode='closest',
            showlegend=False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=20, r=20, t=40, b=200),
            height=650,
            plot_bgcolor='rgba(255,255,255,0)',
            paper_bgcolor='rgba(255,255,255,0)',
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                gridwidth=1
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                gridwidth=1,
                autorange="reversed"  # Make rank 1 appear at the top
            )
        )
        
        st.plotly_chart(fig2, use_container_width=True)

############------------------ tAB 4 ALL ROUNDERS-------------------##################

# Add a new tab for All-Rounders
    with tab4:
        # Create all-rounder rankings by combining batting and bowling data
        all_rounder_rankings = player_ranking_match.groupby(['Year', 'Name', 'Team']).agg({
            'File Name': 'nunique',  # Count matches
            'Batting_Rating': 'sum',
            'Bowling_Rating': 'sum',
            'Runs': 'sum',
            'Bowler_Wkts': 'sum'
        }).reset_index()

        # Rename columns
        all_rounder_rankings = all_rounder_rankings.rename(columns={
            'File Name': 'Matches',
            'Bowler_Wkts': 'Total_Wickets'
        })

        # Calculate Rating Per Game for both disciplines
        all_rounder_rankings['Batting_RPG'] = all_rounder_rankings['Batting_Rating'] / all_rounder_rankings['Matches']
        all_rounder_rankings['Bowling_RPG'] = all_rounder_rankings['Bowling_Rating'] / all_rounder_rankings['Matches']

        # Calculate All-Rounder Rating
        min_batting_rpg = 20  # Minimum batting rating per game to qualify
        min_bowling_rpg = 20  # Minimum bowling rating per game to qualify

        # Create qualification mask
        qualified = (all_rounder_rankings['Batting_RPG'] >= min_batting_rpg) & \
                (all_rounder_rankings['Bowling_RPG'] >= min_bowling_rpg)

        # Calculate combined rating only for qualified players
        all_rounder_rankings['AR_Rating'] = 0.0  # Initialize with 0
        all_rounder_rankings.loc[qualified, 'AR_Rating'] = \
            (all_rounder_rankings.loc[qualified, 'Batting_Rating'] + 
            all_rounder_rankings.loc[qualified, 'Bowling_Rating'])

        # Calculate AR Rating Per Game
        all_rounder_rankings['AR_RPG'] = all_rounder_rankings['AR_Rating'] / all_rounder_rankings['Matches']

        # Filter out non-qualified players before ranking
        qualified_rankings = all_rounder_rankings[all_rounder_rankings['AR_Rating'] > 0].copy()

        # Calculate ranks within each year based on AR_Rating for qualified players only
        qualified_rankings['Rank'] = qualified_rankings.groupby('Year')['AR_Rating'].rank(method='min', ascending=False)

        # Round numeric columns
        numeric_cols = ['Batting_Rating', 'Bowling_Rating', 'AR_Rating', 'Batting_RPG', 'Bowling_RPG', 'AR_RPG']
        qualified_rankings[numeric_cols] = qualified_rankings[numeric_cols].round(2)

        # Add filters for all-rounder rankings
        st.markdown("<h3 style='text-align: center; color:#f04f53;'>Filters</h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            # Sort years in descending order
            years = sorted(qualified_rankings['Year'].unique(), reverse=True)
            selected_years_ar = st.multiselect(
                "Filter by Year",
                options=years,
                key="ar_years"
            )

        with col2:
            selected_names_ar = st.multiselect(
                "Filter by Player Name",
                options=sorted(qualified_rankings['Name'].unique()),
                key="ar_names"
            )

        with col3:
            selected_teams_ar = st.multiselect(
                "Filter by Team",
                options=sorted(qualified_rankings['Team'].unique()),
                key="ar_teams"
            )

        # Apply filters
        filtered_ar = qualified_rankings.copy()

        if selected_years_ar:
            filtered_ar = filtered_ar[filtered_ar['Year'].isin(selected_years_ar)]
        if selected_names_ar:
            filtered_ar = filtered_ar[filtered_ar['Name'].isin(selected_names_ar)]
        if selected_teams_ar:
            filtered_ar = filtered_ar[filtered_ar['Team'].isin(selected_teams_ar)]

        # Final sort by Year (descending) and Rank (ascending)
        filtered_ar = filtered_ar.sort_values(['Year', 'Rank'], ascending=[False, True])

        # Display rankings
        st.markdown("<h3 style='text-align: center; color:#f04f53;'>All-Rounder Rankings</h3>", unsafe_allow_html=True)
        
        # Select columns to display
        display_columns = [
            'Year', 'Rank', 'Name', 'Team', 'Matches', 
            'Batting_RPG', 'Bowling_RPG', 'AR_RPG',
            'Runs', 'Total_Wickets', 'AR_Rating'
        ]
        
        st.dataframe(
            filtered_ar[display_columns],
            use_container_width=True,
            hide_index=True,
            height=850,  # This will show approximately 25 rows
            column_config={
                "Year": st.column_config.NumberColumn(format="%d"),
                "Rank": st.column_config.NumberColumn(format="%d"),
                "Batting_RPG": st.column_config.NumberColumn("Batting RPG", format="%.2f"),
                "Bowling_RPG": st.column_config.NumberColumn("Bowling RPG", format="%.2f"),
                "AR_RPG": st.column_config.NumberColumn("AR RPG", format="%.2f"),
                "AR_Rating": st.column_config.NumberColumn("AR Rating", format="%.2f")
            }
        )

        # Create Rating Per Game trend graph
        st.markdown("<h3 style='text-align: center; color:#f04f53;'>All-Rounder Rating Per Game Trends</h3>", unsafe_allow_html=True)
        fig1 = go.Figure()
        
        trend_data = filtered_ar.sort_values('Year')
        
        for player in trend_data['Name'].unique():
            player_data = trend_data[trend_data['Name'] == player]
            fig1.add_trace(go.Scatter(
                x=player_data['Year'],
                y=player_data['AR_RPG'],
                name=player,
                mode='lines+markers',
                line=dict(width=2),
                hovertemplate="Year: %{x}<br>RPG: %{y:.2f}<br>Name: " + player + "<br>Team: %{customdata}<extra></extra>",
                customdata=player_data['Team']
            ))
        
        fig1.update_layout(
            xaxis_title="Year",
            yaxis_title="All-Rounder Rating Per Game",
            hovermode='closest',
            showlegend=False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=20, r=20, t=40, b=200),
            height=650,
            plot_bgcolor='rgba(255,255,255,0)',
            paper_bgcolor='rgba(255,255,255,0)',
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                gridwidth=1
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                gridwidth=1
            )
        )
        
        st.plotly_chart(fig1, use_container_width=True)

        # Create Rank trend graph
        st.markdown("### Rank Trends")
        fig2 = go.Figure()
        
        for player in trend_data['Name'].unique():
            player_data = trend_data[trend_data['Name'] == player]
            player_data = player_data.sort_values('Year')
            
            fig2.add_trace(go.Scatter(
                x=player_data['Year'],
                y=player_data['Rank'],
                name=player,
                mode='lines+markers',
                line=dict(width=2),
                hovertemplate="Year: %{x}<br>Rank: %{y}<br>Name: " + player + "<br>Team: %{customdata}<extra></extra>",
                customdata=player_data['Team']
            ))
        
        fig2.update_layout(
            xaxis_title="Year",
            yaxis_title="Rank",
            hovermode='closest',
            showlegend=False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=20, r=20, t=40, b=200),
            height=650,
            plot_bgcolor='rgba(255,255,255,0)',
            paper_bgcolor='rgba(255,255,255,0)',
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                gridwidth=1
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                gridwidth=1,
                autorange="reversed"  # Make rank 1 appear at the top
            )
        )
        
        st.plotly_chart(fig2, use_container_width=True)

# Call the function to display
display_ar_view()