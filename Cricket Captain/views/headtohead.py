import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def parse_date(date_str):
    """Helper function to parse dates in multiple formats"""
    try:
        # Try different date formats
        for fmt in ['%d/%m/%Y', '%d %b %Y', '%Y-%m-%d']:
            try:
                return pd.to_datetime(date_str, format=fmt).date()  # Added .date() to remove time
            except ValueError:
                continue
        # If none of the specific formats work, let pandas try to infer the format
        return pd.to_datetime(date_str).date()  # Added .date() to remove time
    except Exception:
        return pd.NaT

# Page Header
st.markdown("<h1 style='color:#f04f53; text-align: center;'>Records</h1>", unsafe_allow_html=True)

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

# Get unique formats from all dataframes
all_formats = set()
all_teams = set()

if 'game_df' in st.session_state:
    all_formats.update(st.session_state['game_df']['Match_Format'].unique())
if 'bat_df' in st.session_state:
    all_formats.update(st.session_state['bat_df']['Match_Format'].unique())
if 'bowl_df' in st.session_state:
    all_formats.update(st.session_state['bowl_df']['Match_Format'].unique())
if 'match_df' in st.session_state:
    all_formats.update(st.session_state['match_df']['Match_Format'].unique())
    # Get unique teams from both Home_Team and Away_Team
    match_df = st.session_state['match_df']
    all_teams.update(match_df['Home_Team'].unique())
    all_teams.update(match_df['Away_Team'].unique())

# Create columns for filters
col1, col2, col3 = st.columns(3)

with col1:
    # Format filter
    formats = ['All'] + sorted(list(all_formats))
    format_choice = st.multiselect('Format:', formats, default='All', key='global_format_filter')

with col2:
    # Team filter
    teams = ['All'] + sorted(list(all_teams))
    team_choice = st.multiselect('Team:', teams, default='All', key='team_filter')

with col3:
    # Opponent filter
    opponent_choice = st.multiselect('Opponent:', teams, default='All', key='opponent_filter')

# Enhanced filter function to handle all three filters
def filter_by_all(df):
    filtered_df = df.copy()
    
    # Format filter
    if 'All' not in format_choice:
        filtered_df = filtered_df[filtered_df['Match_Format'].isin(format_choice)]
    
    # Team filter
    if 'All' not in team_choice:
        filtered_df = filtered_df[filtered_df['Team'].isin(team_choice)]
    
    # Opponent filter
    if 'All' not in opponent_choice:
        filtered_df = filtered_df[filtered_df['Opponent'].isin(opponent_choice)]
    
    return filtered_df

#########====================CREATE HEAD TO HEAD TABLE===================######################
if 'match_df' in st.session_state:
    match_df = st.session_state['match_df']
    
    # Create two versions of each match: one for home team and one for away team
    home_stats = match_df[['Home_Team', 'Away_Team', 'Home_Win', 'Home_Lost', 'Home_Drawn', 'Match_Format']].copy()
    away_stats = match_df[['Home_Team', 'Away_Team', 'Home_Win', 'Home_Lost', 'Home_Drawn', 'Match_Format']].copy()
    
    # Prepare home team records
    home_stats.columns = ['Team', 'Opponent', 'Won', 'Lost', 'Drawn', 'Match_Format']
    
    # Prepare away team records (need to flip Win/Loss since they're from home team perspective)
    away_stats.columns = ['Opponent', 'Team', 'Lost', 'Won', 'Drawn', 'Match_Format']
    
    # Combine both perspectives
    all_matches = pd.concat([home_stats, away_stats], ignore_index=True)
    
    # Apply filters before aggregation
    filtered_matches = filter_by_all(all_matches)
    
    # Group by Team and Opponent to get aggregate statistics
    head2headrecord_df = filtered_matches.groupby(['Team', 'Opponent']).agg({
        'Won': 'sum',
        'Lost': 'sum',
        'Drawn': 'sum'
    }).reset_index()
    
    # Calculate total matches and percentages
    head2headrecord_df['Matches'] = (head2headrecord_df['Won'] + 
                                    head2headrecord_df['Lost'] + 
                                    head2headrecord_df['Drawn'])
    
    head2headrecord_df['Win_Percentage'] = (head2headrecord_df['Won'] / 
                                           head2headrecord_df['Matches'] * 100).round(1)
    
    head2headrecord_df['Loss_Percentage'] = (head2headrecord_df['Lost'] / 
                                            head2headrecord_df['Matches'] * 100).round(1)
    
    head2headrecord_df['Draw_Percentage'] = (head2headrecord_df['Drawn'] / 
                                            head2headrecord_df['Matches'] * 100).round(1)
    
    # Sort by number of matches and win percentage
    head2headrecord_df = head2headrecord_df.sort_values(['Matches', 'Win_Percentage'], 
                                                       ascending=[False, False])
    
    # Display the head-to-head records
    st.markdown("<h1 style='color:#f04f53; text-align: center;'>Head to Head Records<h1>", unsafe_allow_html=True)

    st.dataframe(head2headrecord_df, use_container_width=True, hide_index=True)
    
    # Store in session state for future use
    st.session_state['head2headrecord_df'] = head2headrecord_df
    
    # Display raw matches table
    st.markdown("<h1 style='color:#f04f53; text-align: center;'>All Matches<h1>", unsafe_allow_html=True)
    
    # Create a filtered and sorted version of the raw matches with selected columns
    raw_matches = match_df.copy()
    raw_matches['Date'] = raw_matches['Date'].apply(parse_date)  # Ensure dates are parsed
    
    # Select and rename columns
    raw_matches = raw_matches[['Date', 'Home_Team', 'Away_Team', 'Competition', 'Match_Format', 'Player_of_the_Match', 'Margin']]
    raw_matches.columns = ['Date', 'Home Team', 'Away Team', 'Competition', 'Format', 'POM', 'Margin']
    
    # Apply format filter to raw matches
    if 'All' not in format_choice:
        raw_matches = raw_matches[raw_matches['Format'].isin(format_choice)]
    
    # Apply team filters to raw matches (checking both Home Team and Away Team)
    if 'All' not in team_choice:
        raw_matches = raw_matches[
            (raw_matches['Home Team'].isin(team_choice)) | 
            (raw_matches['Away Team'].isin(team_choice))
        ]
    
    if 'All' not in opponent_choice:
        raw_matches = raw_matches[
            (raw_matches['Home Team'].isin(opponent_choice)) | 
            (raw_matches['Away Team'].isin(opponent_choice))
        ]
    
    # Sort by date (newest to oldest)
    raw_matches = raw_matches.sort_values('Date', ascending=False)
    
    # Display the filtered and sorted matches
    st.dataframe(raw_matches, use_container_width=True, hide_index=True)
    
else:
    st.info("No match records available for head-to-head analysis.")