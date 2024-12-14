import streamlit as st 
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_batter_rating_per_match(df):
    """Calculate batting rating with bonuses"""
    stats = df.copy()
    stats['Strike_Rate'] = (stats['Runs'] / stats['Balls']) * 100
    stats['Base_Score'] = stats['Runs']
    
    def calculate_bonus(row):
        runs = row['Runs']
        sr = row['Strike_Rate']
        bonus = 0
        
        if 100 <= runs < 150:
            bonus += 50  # Century bonus
        elif 150 <= runs < 200:
            bonus += 75  # 150+ bonus
        elif runs >= 200:
            bonus += 100  # Double century bonus
            
        if runs >= 40:  # Only apply SR bonus/penalty for substantial innings
            if sr >= 75:
                bonus += 25  # Good scoring rate bonus
            elif sr <= 40:
                bonus -= 25  # Slow scoring penalty
                
        return bonus
    
    stats['Bonus'] = stats.apply(calculate_bonus, axis=1)
    stats['Batting_Rating'] = stats['Base_Score'] + stats['Bonus']
    return stats

def calculate_bowler_rating_per_match(df):
    """Calculate bowling rating with bonuses"""
    stats = df.copy()
    stats['Overs'] = stats['Bowler_Balls'] / 6
    stats['Economy'] = (stats['Bowler_Runs'] / stats['Overs']).replace(float('inf'), 0)
    stats['Base_Score'] = (stats['Bowler_Wkts'] * 20) + (stats['Maidens'] * 2)
    
    def calculate_wicket_bonus(wickets):
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
        return 0
    
    def calculate_economy_bonus(row):
        if row['Overs'] >= 10:
            if row['Economy'] <= 2.5:
                return 25
            elif row['Economy'] >= 4.5:
                return -25
        return 0
    
    stats['Wicket_Bonus'] = stats['Bowler_Wkts'].apply(calculate_wicket_bonus)
    stats['Economy_Bonus'] = stats.apply(calculate_economy_bonus, axis=1)
    stats['Bowling_Rating'] = stats['Base_Score'] + stats['Wicket_Bonus'] + stats['Economy_Bonus']
    
    return stats

def calculate_peak_ratings(rankings_df):
    """Calculate peak ratings and duration at peak for each player"""
    peak_data = rankings_df.groupby('Name').agg({
        'Rating': ['max', 'mean'],
        'Year': ['min', 'max', 'count']
    }).reset_index()
    
    peak_data.columns = ['Name', 'Peak_Rating', 'Avg_Rating', 'Start_Year', 'End_Year', 'Years_Active']
    return peak_data

def display_number_one_rankings(bat_df, bowl_df):
    """Display #1 Rankings tab content"""
    # Original yearly best performers section
    st.markdown("<h3 style='text-align: center; color:#f04f53;'>Yearly Best Performers</h3>", unsafe_allow_html=True)
    
    # Calculate yearly bests (existing code remains the same)
    batting_by_player = bat_df.groupby(['Year', 'Name'])['Batting_Rating'].sum().reset_index()
    best_batting = batting_by_player.groupby('Year').apply(
        lambda x: f"{x.loc[x['Batting_Rating'].idxmax(), 'Name']} - {x['Batting_Rating'].max():.0f}" 
        if x['Batting_Rating'].max() > 0 else "-"
    ).reset_index(name='Best_Batting')

    bowling_by_player = bowl_df.groupby(['Year', 'Name'])['Bowling_Rating'].sum().reset_index()
    best_bowling = bowling_by_player.groupby('Year').apply(
        lambda x: f"{x.loc[x['Bowling_Rating'].idxmax(), 'Name']} - {x['Bowling_Rating'].max():.0f}" 
        if x['Bowling_Rating'].max() > 0 else "-"
    ).reset_index(name='Best_Bowling')

    # Calculate AR ratings
    yearly_ar = pd.merge(
        batting_by_player,
        bowling_by_player,
        on=['Year', 'Name'],
        how='outer'
    ).fillna(0)
    
    yearly_ar['AR_Rating'] = yearly_ar['Batting_Rating'] + yearly_ar['Bowling_Rating']
    best_ar = yearly_ar.groupby('Year').apply(
        lambda x: f"{x.loc[x['AR_Rating'].idxmax(), 'Name']} - {x['AR_Rating'].max():.0f}" 
        if x['AR_Rating'].max() > 0 else "-"
    ).reset_index(name='Best_AllRounder')

    # Combine summaries
    yearly_summary = pd.merge(best_batting, best_bowling, on='Year', how='outer')
    yearly_summary = pd.merge(yearly_summary, best_ar, on='Year', how='outer')
    yearly_summary = yearly_summary.sort_values('Year', ascending=False)

    # Display yearly summary
    st.dataframe(
        yearly_summary,
        use_container_width=True,
        hide_index=True,
        height=(len(yearly_summary) + 1) * 35,
        column_config={
            "Year": st.column_config.NumberColumn(format="%d"),
            "Best_Batting": "Batting",
            "Best_Bowling": "Bowling",
            "Best_AllRounder": "All-Rounder"
        }
    )



    # Peak Achievements
    st.markdown("<h3 style='text-align: center; color:#f04f53;'>Peak Rating Achievements</h3>", unsafe_allow_html=True)
    
    # Calculate peak ratings
    batting_peaks = batting_by_player.groupby('Name')['Batting_Rating'].max().reset_index()
    bowling_peaks = bowling_by_player.groupby('Name')['Bowling_Rating'].max().reset_index()
    ar_peaks = yearly_ar.groupby('Name')['AR_Rating'].max().reset_index()

    # Combine peak ratings
    peaks = pd.merge(batting_peaks, bowling_peaks, on='Name', how='outer', suffixes=('_Bat', '_Bowl'))
    peaks = pd.merge(peaks, ar_peaks, on='Name', how='outer')
    peaks = peaks.fillna(0)
    peaks = peaks.sort_values('AR_Rating', ascending=False)

    st.dataframe(
        peaks,
        use_container_width=True,
        hide_index=True,
        height=400,
        column_config={
            "Name": "Player",
            "Batting_Rating_Bat": "Peak Batting Rating",
            "Bowling_Rating_Bowl": "Peak Bowling Rating",
            "AR_Rating": "Peak All-Rounder Rating"
        }
    )

    # Number of times ranked #1
    st.markdown("<h3 style='text-align: center; color:#f04f53;'>Number of Times Ranked #1</h3>", unsafe_allow_html=True)
    
    # Calculate #1 rankings
    batting_tops = batting_by_player.copy()
    batting_tops['Rank'] = batting_tops.groupby('Year')['Batting_Rating'].rank(method='min', ascending=False)
    batting_no1 = batting_tops[batting_tops['Rank'] == 1].groupby('Name').size().reset_index(name='Batting_#1')

    bowling_tops = bowling_by_player.copy()
    bowling_tops['Rank'] = bowling_tops.groupby('Year')['Bowling_Rating'].rank(method='min', ascending=False)
    bowling_no1 = bowling_tops[bowling_tops['Rank'] == 1].groupby('Name').size().reset_index(name='Bowling_#1')

    ar_tops = yearly_ar.copy()
    ar_tops['Rank'] = ar_tops.groupby('Year')['AR_Rating'].rank(method='min', ascending=False)
    ar_no1 = ar_tops[ar_tops['Rank'] == 1].groupby('Name').size().reset_index(name='AllRounder_#1')

    # Combine all #1 rankings
    all_no1 = pd.merge(batting_no1, bowling_no1, on='Name', how='outer')
    all_no1 = pd.merge(all_no1, ar_no1, on='Name', how='outer')
    all_no1 = all_no1.fillna(0)
    all_no1[['Batting_#1', 'Bowling_#1', 'AllRounder_#1']] = all_no1[['Batting_#1', 'Bowling_#1', 'AllRounder_#1']].astype(int)
    all_no1['Total_#1'] = all_no1['Batting_#1'] + all_no1['Bowling_#1'] + all_no1['AllRounder_#1']
    all_no1 = all_no1.sort_values('Total_#1', ascending=False)

    st.dataframe(
        all_no1,
        use_container_width=True,
        hide_index=True,
        height=400,
        column_config={
            "Name": "Player",
            "Batting_#1": "Batting #1",
            "Bowling_#1": "Bowling #1",
            "AllRounder_#1": "All-Rounder #1",
            "Total_#1": "Total #1"
        }
    )

    # Hall of Fame Criteria
    st.markdown("<h3 style='text-align: center; color:#f04f53;'>Hall of Fame Status</h3>", unsafe_allow_html=True)
    
    def calculate_hof_score(row):
        score = 0
        # Points for #1 rankings
        score += row['Batting_#1'] * 20
        score += row['Bowling_#1'] * 20
        score += row['AllRounder_#1'] * 15
        return score

    # Calculate Hall of Fame scores
    hof_data = all_no1.copy()
    hof_data['HOF_Score'] = hof_data.apply(calculate_hof_score, axis=1)
    
    # Create HOF Status column
    def get_hof_status(score):
        if score >= 100:
            return "Hall of Fame"
        else:
            return f"{score}% to HOF"
            
    hof_data['HOF_Status'] = hof_data['HOF_Score'].apply(get_hof_status)
    hof_data = hof_data.sort_values('HOF_Score', ascending=False)

    st.dataframe(
        hof_data[['Name', 'HOF_Score', 'HOF_Status']],
        use_container_width=True,
        hide_index=True,
        height=400,
        column_config={
            "Name": "Player",
            "HOF_Score": "Hall of Fame Score",
            "HOF_Status": "Status"
        }
    )


def display_batting_rankings(bat_df):
    """Display Batting Rankings tab content"""
    # Use the correct team column from batting data
    if 'Team' not in bat_df.columns:
        if 'Bat_Team' in bat_df.columns:
            bat_df['Team'] = bat_df['Bat_Team']
        elif 'Batting_Team' in bat_df.columns:
            bat_df['Team'] = bat_df['Batting_Team']
        elif 'Bat_Team_y' in bat_df.columns:
            bat_df['Team'] = bat_df['Bat_Team_y']
        else:
            bat_df['Team'] = 'Unknown'

    # Calculate additional batting statistics
    bat_df['Average'] = bat_df.groupby(['Year', 'Name'])['Runs'].transform('mean')
    bat_df['Strike_Rate'] = (bat_df['Runs'] / bat_df['Balls']) * 100
    bat_df['Centuries'] = bat_df['Runs'].apply(lambda x: 1 if x >= 100 else 0)
    bat_df['Double_Centuries'] = bat_df['Runs'].apply(lambda x: 1 if x >= 200 else 0)

    batting_rankings = bat_df.groupby(['Year', 'Name', 'Team']).agg({
        'File Name': 'nunique',
        'Batting_Rating': 'sum',
        'Runs': 'sum',
        'Average': 'mean',
        'Strike_Rate': 'mean',
        'Centuries': 'sum',
        'Double_Centuries': 'sum',
        '4s': 'sum',
        '6s': 'sum'
    }).reset_index()

    batting_rankings = batting_rankings.rename(columns={
        'File Name': 'Matches',
        'Batting_Rating': 'Rating',
        'Runs': 'Total_Runs'
    })

    batting_rankings['RPG'] = batting_rankings['Rating'] / batting_rankings['Matches']
    batting_rankings['Rank'] = batting_rankings.groupby('Year')['Rating'].rank(method='dense', ascending=False).astype(int)
    batting_rankings = batting_rankings.sort_values(['Year', 'Rank'])

    numeric_cols = ['Rating', 'RPG', 'Total_Runs', 'Average', 'Strike_Rate']
    batting_rankings[numeric_cols] = batting_rankings[numeric_cols].round(2)
    batting_rankings['Year'] = batting_rankings['Year'].astype(int)

    # Filters section
    st.markdown("<h3 style='text-align: center; color:#f04f53;'>Filters</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_names = st.multiselect(
            "Filter by Player Name",
            options=sorted(batting_rankings['Name'].unique()),
            key="batting_names"
        )
    
    with col2:
        selected_teams = st.multiselect(
            "Filter by Team",
            options=sorted(batting_rankings['Team'].unique()),
            key="batting_teams"
        )
        
    with col3:
        min_matches = st.number_input(
            "Minimum Season Matches",
            min_value=1,
            max_value=int(batting_rankings['Matches'].max()),
            value=1
        )

    filtered_batting = batting_rankings.copy()
    if selected_names:
        filtered_batting = filtered_batting[filtered_batting['Name'].isin(selected_names)]
    if selected_teams:
        filtered_batting = filtered_batting[filtered_batting['Team'].isin(selected_teams)]
    filtered_batting = filtered_batting[filtered_batting['Matches'] >= min_matches]
    filtered_batting = filtered_batting.sort_values(['Year', 'Rank'], ascending=[False, True])

    # Career Statistics Summary
    st.markdown("<h3 style='text-align: center; color:#f04f53;'>Career Statistics</h3>", unsafe_allow_html=True)
    
    career_stats = filtered_batting.groupby('Name').agg({
        'Matches': 'sum',
        'Total_Runs': 'sum',
        'Average': 'mean',
        'Strike_Rate': 'mean',
        'Centuries': 'sum',
        'Double_Centuries': 'sum',
        '4s': 'sum',
        '6s': 'sum',
        'Rating': 'mean'
    }).round(2)
    
    career_stats = career_stats.sort_values('Rating', ascending=False)
    
    st.dataframe(
        career_stats,
        use_container_width=True,
        hide_index=False,
        height=400,
        column_config={
            "Name": "Player",
            "Rating": "Average Season Rating",
            "Total_Runs": "Career Runs",
            "Average": "Batting Average",
            "Strike_Rate": "Strike Rate",
            "Centuries": "100s",
            "Double_Centuries": "200s",
            "4s": "Fours",
            "6s": "Sixes"
        }
    )

    # Yearly Rankings Table
    st.markdown("<h3 style='text-align: center; color:#f04f53;'>Yearly Rankings</h3>", unsafe_allow_html=True)
    st.dataframe(
        filtered_batting,
        use_container_width=True,
        hide_index=True,
        height=850,
        column_config={
            "Year": st.column_config.NumberColumn(format="%d"),
            "Rank": st.column_config.NumberColumn(format="%d"),
            "Strike_Rate": "Strike Rate",
            "Centuries": "100s",
            "Double_Centuries": "200s"
        }
    )

    # Create a row for both trend graphs
    st.markdown("<h3 style='text-align: center; color:#f04f53;'>Performance Trends</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating per game trend graph
        fig1 = go.Figure()
        
        trend_data = filtered_batting.sort_values('Year')
        show_legend = len(selected_names) > 0  # Only show legend if players are selected
        
        for player in trend_data['Name'].unique():
            player_data = trend_data[trend_data['Name'] == player]
            fig1.add_trace(go.Scatter(
                x=player_data['Year'],
                y=player_data['RPG'],
                name=player,
                mode='lines+markers',
                line=dict(width=2),
                hovertemplate="Year: %{x}<br>RPG: %{y:.2f}<br>Name: " + player + "<br>Team: %{customdata}<extra></extra>",
                customdata=player_data['Team'],
                showlegend=show_legend
            ))

        fig1.update_layout(
            title="Rating Per Game Trends",
            xaxis_title="Year",
            yaxis_title="Rating Per Game",
            hovermode='closest',
            showlegend=show_legend,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            margin=dict(l=20, r=20, t=40, b=40),
            height=500,
            plot_bgcolor='rgba(255,255,255,0)',
            paper_bgcolor='rgba(255,255,255,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1)
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Rank trend graph
        fig2 = go.Figure()
        
        for player in trend_data['Name'].unique():
            player_data = trend_data[trend_data['Name'] == player]
            fig2.add_trace(go.Scatter(
                x=player_data['Year'],
                y=player_data['Rank'],
                name=player,
                mode='lines+markers',
                line=dict(width=2),
                hovertemplate="Year: %{x}<br>Rank: %{y}<br>Name: " + player + "<br>Team: %{customdata}<extra></extra>",
                customdata=player_data['Team'],
                showlegend=show_legend
            ))

        fig2.update_layout(
            title="Ranking Trends",
            xaxis_title="Year",
            yaxis_title="Rank",
            hovermode='closest',
            showlegend=show_legend,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            margin=dict(l=20, r=20, t=40, b=40),
            height=500,
            plot_bgcolor='rgba(255,255,255,0)',
            paper_bgcolor='rgba(255,255,255,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1,
                      autorange="reversed")
        )

        st.plotly_chart(fig2, use_container_width=True)

    # Remove the Batting Milestones Analysis and Team Performance Breakdown sections
    # End of the function


def display_bowling_rankings(bowl_df):
    """Display Bowling Rankings tab content"""
    # First ensure we have the Team information
    if 'Bowl_Team' in bowl_df.columns:
        bowl_df['Team'] = bowl_df['Bowl_Team']
    elif 'Bowling_Team' in bowl_df.columns:
        bowl_df['Team'] = bowl_df['Bowling_Team']
    elif 'Team' not in bowl_df.columns:
        bowl_df['Team'] = 'Unknown'

    # Calculate bowling statistics correctly
    bowl_df['Overs'] = bowl_df['Bowler_Balls'] / 6
    bowl_df['Average'] = bowl_df.groupby(['Year', 'Name'])['Bowler_Runs'].transform('sum') / \
                        bowl_df.groupby(['Year', 'Name'])['Bowler_Wkts'].transform('sum')
    bowl_df['Strike_Rate'] = (bowl_df.groupby(['Year', 'Name'])['Bowler_Balls'].transform('sum') / \
                             bowl_df.groupby(['Year', 'Name'])['Bowler_Wkts'].transform('sum'))
    bowl_df['Economy'] = (bowl_df.groupby(['Year', 'Name'])['Bowler_Runs'].transform('sum') * 6) / \
                        bowl_df.groupby(['Year', 'Name'])['Bowler_Balls'].transform('sum')
    
    # Handle division by zero
    bowl_df['Average'] = bowl_df['Average'].replace([float('inf'), float('-inf')], 0)
    bowl_df['Strike_Rate'] = bowl_df['Strike_Rate'].replace([float('inf'), float('-inf')], 0)
    bowl_df['Economy'] = bowl_df['Economy'].replace([float('inf'), float('-inf')], 0)

    # Calculate milestone counts
    bowl_df['Five_Wickets'] = bowl_df['Bowler_Wkts'].apply(lambda x: 1 if x >= 5 else 0)
    bowl_df['Ten_Wickets'] = bowl_df['Bowler_Wkts'].apply(lambda x: 1 if x >= 10 else 0)

    # Create bowling rankings
    bowling_rankings = bowl_df.groupby(['Year', 'Name', 'Team']).agg({
        'File Name': 'nunique',
        'Bowling_Rating': 'sum',
        'Bowler_Wkts': 'sum',
        'Average': 'mean',
        'Strike_Rate': 'mean',
        'Economy': 'mean',
        'Five_Wickets': 'sum',
        'Ten_Wickets': 'sum',
        'Maidens': 'sum'
    }).reset_index()

    bowling_rankings = bowling_rankings.rename(columns={
        'File Name': 'Matches',
        'Bowling_Rating': 'Rating',
        'Bowler_Wkts': 'Total_Wickets'
    })

    bowling_rankings['RPG'] = bowling_rankings['Rating'] / bowling_rankings['Matches']
    bowling_rankings['Rank'] = bowling_rankings.groupby('Year')['Rating'].rank(method='min', ascending=False)

    numeric_cols = ['Rating', 'RPG', 'Total_Wickets', 'Average', 'Strike_Rate', 'Economy']
    bowling_rankings[numeric_cols] = bowling_rankings[numeric_cols].round(2)

    # Filters section
    st.markdown("<h3 style='text-align: center; color:#f04f53;'>Filters</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_names = st.multiselect(
            "Filter by Player Name",
            options=sorted(bowling_rankings['Name'].unique()),
            key="bowling_names"
        )
    
    with col2:
        selected_teams = st.multiselect(
            "Filter by Team",
            options=sorted(bowling_rankings['Team'].unique()),
            key="bowling_teams"
        )
        
    with col3:
        min_matches = st.number_input(
            "Minimum Matches",
            min_value=1,
            max_value=int(bowling_rankings['Matches'].max()),
            value=1
        )

    # Apply filters
    filtered_bowling = bowling_rankings.copy()
    if selected_names:
        filtered_bowling = filtered_bowling[filtered_bowling['Name'].isin(selected_names)]
    if selected_teams:
        filtered_bowling = filtered_bowling[filtered_bowling['Team'].isin(selected_teams)]
    filtered_bowling = filtered_bowling[filtered_bowling['Matches'] >= min_matches]
    filtered_bowling = filtered_bowling.sort_values(['Year', 'Rank'], ascending=[False, True])

    # Career Statistics Summary
    st.markdown("<h3 style='text-align: center; color:#f04f53;'>Career Statistics</h3>", unsafe_allow_html=True)
    
    career_stats = filtered_bowling.groupby('Name').agg({
        'Matches': 'sum',
        'Total_Wickets': 'sum',
        'Average': 'mean',
        'Strike_Rate': 'mean',
        'Economy': 'mean',
        'Five_Wickets': 'sum',
        'Ten_Wickets': 'sum',
        'Maidens': 'sum',
        'Rating': 'mean'
    }).round(2)
    
    career_stats = career_stats.sort_values('Rating', ascending=False)
    
    st.dataframe(
        career_stats,
        use_container_width=True,
        hide_index=False,
        height=400,
        column_config={
            "Name": "Player",
            "Rating": "Average Season Rating",
            "Total_Wickets": "Career Wickets",
            "Average": "Bowling Average",
            "Strike_Rate": "Strike Rate",
            "Economy": "Economy Rate",
            "Five_Wickets": "5 Wicket Hauls",
            "Ten_Wickets": "10 Wicket Matches",
            "Maidens": "Maiden Overs"
        }
    )

    # Yearly Rankings Table
    st.markdown("<h3 style='text-align: center; color:#f04f53;'>Yearly Rankings</h3>", unsafe_allow_html=True)
    st.dataframe(
        filtered_bowling,
        use_container_width=True,
        hide_index=True,
        height=850,
        column_config={
            "Year": st.column_config.NumberColumn(format="%d"),
            "Rank": st.column_config.NumberColumn(format="%d"),
            "Average": "Bowling Average",
            "Strike_Rate": "Strike Rate",
            "Economy": "Economy Rate",
            "Five_Wickets": "5WI",
            "Ten_Wickets": "10WM"
        }
    )

    # Create a row for both trend graphs
    st.markdown("<h3 style='text-align: center; color:#f04f53;'>Performance Trends</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating per game trend graph
        fig1 = go.Figure()
        
        trend_data = filtered_bowling.sort_values('Year')
        show_legend = len(selected_names) > 0  # Only show legend if players are selected
        
        for player in trend_data['Name'].unique():
            player_data = trend_data[trend_data['Name'] == player]
            fig1.add_trace(go.Scatter(
                x=player_data['Year'],
                y=player_data['RPG'],
                name=player,
                mode='lines+markers',
                line=dict(width=2),
                hovertemplate="Year: %{x}<br>RPG: %{y:.2f}<br>Name: " + player + "<br>Team: %{customdata}<extra></extra>",
                customdata=player_data['Team'],
                showlegend=show_legend
            ))

        fig1.update_layout(
            title="Rating Per Game Trends",
            xaxis_title="Year",
            yaxis_title="Rating Per Game",
            hovermode='closest',
            showlegend=show_legend,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            margin=dict(l=20, r=20, t=40, b=40),
            height=500,
            plot_bgcolor='rgba(255,255,255,0)',
            paper_bgcolor='rgba(255,255,255,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1)
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Rank trend graph
        fig2 = go.Figure()
        
        for player in trend_data['Name'].unique():
            player_data = trend_data[trend_data['Name'] == player]
            fig2.add_trace(go.Scatter(
                x=player_data['Year'],
                y=player_data['Rank'],
                name=player,
                mode='lines+markers',
                line=dict(width=2),
                hovertemplate="Year: %{x}<br>Rank: %{y}<br>Name: " + player + "<br>Team: %{customdata}<extra></extra>",
                customdata=player_data['Team'],
                showlegend=show_legend
            ))

        fig2.update_layout(
            title="Ranking Trends",
            xaxis_title="Year",
            yaxis_title="Rank",
            hovermode='closest',
            showlegend=show_legend,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            margin=dict(l=20, r=20, t=40, b=40),
            height=500,
            plot_bgcolor='rgba(255,255,255,0)',
            paper_bgcolor='rgba(255,255,255,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1,
                      autorange="reversed")
        )

        st.plotly_chart(fig2, use_container_width=True)

def display_allrounder_rankings(bat_df, bowl_df):
    """Display All-Rounder Rankings tab content"""
    # Create all-rounder rankings
    all_rounder_rankings = bat_df.groupby(['Year', 'Name', 'Team']).agg({
        'File Name': 'nunique',
        'Batting_Rating': 'sum',
        'Runs': 'sum'
    }).reset_index()

    # Merge bowling stats
    bowling_stats = bowl_df.groupby(['Year', 'Name']).agg({
        'Bowling_Rating': 'sum',
        'Bowler_Wkts': 'sum'
    }).reset_index()

    all_rounder_rankings = pd.merge(
        all_rounder_rankings,
        bowling_stats,
        on=['Year', 'Name'],
        how='outer'
    )

    # Calculate RPG for both disciplines
    all_rounder_rankings['Batting_RPG'] = all_rounder_rankings['Batting_Rating'] / all_rounder_rankings['File Name']
    all_rounder_rankings['Bowling_RPG'] = all_rounder_rankings['Bowling_Rating'] / all_rounder_rankings['File Name']

    # Apply qualification criteria
    qualified = (all_rounder_rankings['Batting_RPG'] >= 20) & (all_rounder_rankings['Bowling_RPG'] >= 20)
    
    all_rounder_rankings['AR_Rating'] = 0.0
    all_rounder_rankings.loc[qualified, 'AR_Rating'] = \
        all_rounder_rankings.loc[qualified, 'Batting_Rating'] + all_rounder_rankings.loc[qualified, 'Bowling_Rating']

    all_rounder_rankings['AR_RPG'] = all_rounder_rankings['AR_Rating'] / all_rounder_rankings['File Name']

    # Calculate ranks for qualified players
    qualified_rankings = all_rounder_rankings[all_rounder_rankings['AR_Rating'] > 0].copy()
    qualified_rankings['Rank'] = qualified_rankings.groupby('Year')['AR_Rating'].rank(method='min', ascending=False)

    # Round numeric columns
    numeric_cols = ['Batting_Rating', 'Bowling_Rating', 'AR_Rating', 'Batting_RPG', 'Bowling_RPG', 'AR_RPG']
    qualified_rankings[numeric_cols] = qualified_rankings[numeric_cols].round(2)

    # Add filters
    st.markdown("<h3 style='text-align: center; color:#f04f53;'>Filters</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        years = sorted(qualified_rankings['Year'].unique(), reverse=True)
        selected_years = st.multiselect(
            "Filter by Year",
            options=years,
            key="ar_years"
        )

    with col2:
        selected_names = st.multiselect(
            "Filter by Player Name",
            options=sorted(qualified_rankings['Name'].unique()),
            key="ar_names"
        )

    with col3:
        selected_teams = st.multiselect(
            "Filter by Team",
            options=sorted(qualified_rankings['Team'].unique()),
            key="ar_teams"
        )

    # Apply filters
    filtered_ar = qualified_rankings.copy()

    if selected_years:
        filtered_ar = filtered_ar[filtered_ar['Year'].isin(selected_years)]
    if selected_names:
        filtered_ar = filtered_ar[filtered_ar['Name'].isin(selected_names)]
    if selected_teams:
        filtered_ar = filtered_ar[filtered_ar['Team'].isin(selected_teams)]

    filtered_ar = filtered_ar.sort_values(['Year', 'Rank'], ascending=[False, True])

    # Display rankings
    st.markdown("<h3 style='text-align: center; color:#f04f53;'>All-Rounder Rankings</h3>", unsafe_allow_html=True)
    
    display_columns = [
        'Year', 'Rank', 'Name', 'Team', 'File Name', 
        'Batting_RPG', 'Bowling_RPG', 'AR_RPG',
        'Runs', 'Bowler_Wkts', 'AR_Rating'
    ]
    
    st.dataframe(
        filtered_ar[display_columns],
        use_container_width=True,
        hide_index=True,
        height=850,
        column_config={
            "Year": st.column_config.NumberColumn(format="%d"),
            "Rank": st.column_config.NumberColumn(format="%d"),
            "File Name": "Matches",
            "Batting_RPG": "Batting RPG",
            "Bowling_RPG": "Bowling RPG",
            "AR_RPG": "AR RPG",
            "Bowler_Wkts": "Total Wickets",
            "AR_Rating": "AR Rating"
        }
    )

    # Create a row for both trend graphs
    st.markdown("<h3 style='text-align: center; color:#f04f53;'>Performance Trends</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating per game trend graph
        fig1 = go.Figure()
        
        trend_data = filtered_ar.sort_values('Year')
        show_legend = len(selected_names) > 0  # Only show legend if players are selected
        
        for player in trend_data['Name'].unique():
            player_data = trend_data[trend_data['Name'] == player]
            fig1.add_trace(go.Scatter(
                x=player_data['Year'],
                y=player_data['AR_RPG'],
                name=player,
                mode='lines+markers',
                line=dict(width=2),
                hovertemplate="Year: %{x}<br>RPG: %{y:.2f}<br>Name: " + player + "<br>Team: %{customdata}<extra></extra>",
                customdata=player_data['Team'],
                showlegend=show_legend
            ))

        fig1.update_layout(
            title="Rating Per Game Trends",
            xaxis_title="Year",
            yaxis_title="Rating Per Game",
            hovermode='closest',
            showlegend=show_legend,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            margin=dict(l=20, r=20, t=40, b=40),
            height=500,
            plot_bgcolor='rgba(255,255,255,0)',
            paper_bgcolor='rgba(255,255,255,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1)
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Rank trend graph
        fig2 = go.Figure()
        
        for player in trend_data['Name'].unique():
            player_data = trend_data[trend_data['Name'] == player]
            fig2.add_trace(go.Scatter(
                x=player_data['Year'],
                y=player_data['Rank'],
                name=player,
                mode='lines+markers',
                line=dict(width=2),
                hovertemplate="Year: %{x}<br>Rank: %{y}<br>Name: " + player + "<br>Team: %{customdata}<extra></extra>",
                customdata=player_data['Team'],
                showlegend=show_legend
            ))

        fig2.update_layout(
            title="Ranking Trends",
            xaxis_title="Year",
            yaxis_title="Rank",
            hovermode='closest',
            showlegend=show_legend,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            margin=dict(l=20, r=20, t=40, b=40),
            height=500,
            plot_bgcolor='rgba(255,255,255,0)',
            paper_bgcolor='rgba(255,255,255,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1,
                      autorange="reversed")
        )

        st.plotly_chart(fig2, use_container_width=True)

def display_ar_view():
    """Main function to display all rankings views"""
    st.markdown("<h1 style='color:#f04f53; text-align: center;'>Player Rankings</h1>", unsafe_allow_html=True)
    
    # Add styling
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
    /* Tab styling */
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
    
    # Rest of the function remains the same...

    
    if 'bat_df' not in st.session_state or 'bowl_df' not in st.session_state:
        st.error("Required data not found in session state")
        return

    # Get the dataframes
    bat_df = st.session_state['bat_df'].copy()
    bowl_df = st.session_state['bowl_df'].copy()

    # Process dates and add Year column
    if 'Date' in bat_df.columns:
        bat_df['Date'] = pd.to_datetime(bat_df['Date'], errors='coerce')
        bat_df['Year'] = bat_df['Date'].dt.year

    # Add Year to bowling dataframe using File Name as key
    bowl_df = pd.merge(
        bowl_df,
        bat_df[['File Name', 'Year']].drop_duplicates(),
        on='File Name',
        how='left'
    )

    # Calculate ratings
    bat_df = calculate_batter_rating_per_match(bat_df)
    bowl_df = calculate_bowler_rating_per_match(bowl_df)

    # Ensure Year is integer type
    bat_df['Year'] = pd.to_numeric(bat_df['Year'], errors='coerce').fillna(0).astype(int)
    bowl_df['Year'] = pd.to_numeric(bowl_df['Year'], errors='coerce').fillna(0).astype(int)

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "#1 Rankings", 
        "Batting Rankings", 
        "Bowling Rankings", 
        "All-Rounder Rankings"
    ])

    with tab1:
        display_number_one_rankings(bat_df, bowl_df)
        
    with tab2:
        display_batting_rankings(bat_df)
        
    with tab3:
        display_bowling_rankings(bowl_df)
        
    with tab4:
        display_allrounder_rankings(bat_df, bowl_df)

# Call the function to display
display_ar_view()
