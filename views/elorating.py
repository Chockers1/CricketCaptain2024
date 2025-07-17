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



# Modern CSS for beautiful UI - Full styling with enhanced elements
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
    
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #f0f2f6;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #f04f53;
        box-shadow: 0 0 0 3px rgba(240,79,83,0.1);
    }
    
    .stMultiSelect > div > div {
        border-radius: 8px;
        border: 2px solid #f0f2f6;
        transition: all 0.3s ease;
    }
    
    .stMultiSelect > div > div:focus-within {
        border-color: #f04f53;
        box-shadow: 0 0 0 3px rgba(240,79,83,0.1);
    }
    
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        width: 100%;
    }
    .stTabs [data-baseweb="tab"] {
        flex-grow: 1;
        text-align: center;
        font-weight: bold;
        font-size: 1.1em;
        border-radius: 8px 8px 0 0;
        background: #fff0f1;
        margin: 0 2px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #f04f53 0%, #f5576c 100%);
        color: white !important;
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* Beautiful scrollbar for dataframes */
    .stDataFrame [data-testid="stDataFrameResizeHandle"] {
        background-color: #f04f53;
    }
    
    /* Enhanced plotly charts */
    .js-plotly-plot {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
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

# Create beautiful header section with purple gradient background
st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px 20px; border-radius: 20px; text-align: center; 
                margin-bottom: 2rem; box-shadow: 0 8px 25px rgba(0,0,0,0.15);'>
        <h1 style='color: white; font-size: 2.2rem; font-weight: 700; margin: 0 0 8px 0;'>
            🔍 ELO Rankings
        </h1>
        <p style='color: white; font-size: 1rem; margin: 0; opacity: 0.9;'>
            Filter by format, team, and opponent to analyze ELO ratings
        </p>
    </div>
""", unsafe_allow_html=True)

# Create columns for filters
col1, col2, col3 = st.columns(3)

with col1:
    formats = ['All'] + sorted(list(all_formats))
    format_choice = st.multiselect('Format:', formats, default='All', key='global_format_filter')

# Create dynamic lists based on selected formats
if 'All' not in format_choice:
    relevant_matches = st.session_state['match_df'][
        st.session_state['match_df']['Match_Format'].isin(format_choice)
    ] if 'match_df' in st.session_state else pd.DataFrame()
    dynamic_teams = sorted(
        set(relevant_matches['Home_Team'].unique()) | set(relevant_matches['Away_Team'].unique())
    )
else:
    dynamic_teams = sorted(list(all_teams))

with col2:
    teams = ['All'] + dynamic_teams
    team_choice = st.multiselect('Team:', teams, default='All', key='team_filter')

# Create dynamic opponents based on selected teams (and format if not 'All')
if 'All' not in team_choice:
    relevant_opponents = st.session_state['match_df'][
        (st.session_state['match_df']['Home_Team'].isin(team_choice)) |
        (st.session_state['match_df']['Away_Team'].isin(team_choice))
    ] if 'match_df' in st.session_state else pd.DataFrame()
    if 'All' not in format_choice:
        relevant_opponents = relevant_opponents[
            relevant_opponents['Match_Format'].isin(format_choice)
        ]
    dynamic_opponents = sorted(
        set(relevant_opponents['Home_Team'].unique()) | set(relevant_opponents['Away_Team'].unique())
    )
else:
    dynamic_opponents = dynamic_teams

with col3:
    opponents = ['All'] + dynamic_opponents
    opponent_choice = st.multiselect('Opponent:', opponents, default='All', key='opponent_filter')

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

# Build an initially unfiltered DataFrame
if 'match_df' in st.session_state:
    base_df = st.session_state['match_df'].copy()
else:
    base_df = pd.DataFrame()

# Apply selected filters to narrow down base_df for dynamic choices:
if 'All' not in st.session_state.get('global_format_filter', []):
    base_df = base_df[base_df['Match_Format'].isin(st.session_state['global_format_filter'])]
if 'All' not in st.session_state.get('team_filter', []):
    base_df = base_df[
        (base_df['Home_Team'].isin(st.session_state['team_filter'])) |
        (base_df['Away_Team'].isin(st.session_state['team_filter']))
    ]
if 'All' not in st.session_state.get('opponent_filter', []):
    base_df = base_df[
        (base_df['Home_Team'].isin(st.session_state['opponent_filter'])) |
        (base_df['Away_Team'].isin(st.session_state['opponent_filter']))
    ]

# Once base_df is filtered with current selections, build dynamic options:
filtered_formats = sorted(base_df['Match_Format'].unique()) if not base_df.empty else []
filtered_teams = sorted(
    set(base_df['Home_Team'].unique()) | set(base_df['Away_Team'].unique())
) if not base_df.empty else []

#########====================CREATE HEAD TO HEAD TABLE===================######################
if 'match_df' in st.session_state:
    match_df = st.session_state['match_df'].copy()
    
    # Convert to datetime for proper sorting
    match_df['Date_Sort'] = match_df['Date'].apply(parse_date)
    # Sort DataFrame by date
    match_df_sorted = match_df.sort_values('Date_Sort')
    
    # Initialize ELO ratings - now a nested dictionary {format: {team: rating}}
    initial_elo = 1000
    elo_ratings = {}
    k_factor = 32

    def expected_score(elo_a, elo_b):
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    def update_elo(winner_elo, loser_elo, k_factor, result):
        expected_win = expected_score(winner_elo, loser_elo)
        new_winner_elo = winner_elo + k_factor * (result - expected_win)
        new_loser_elo = loser_elo + k_factor * ((1 - result) - (1 - expected_win))
        return round(new_winner_elo, 2), round(new_loser_elo, 2)

    # Add ELO columns
    match_df_sorted['Home_Elo_Start'] = 0
    match_df_sorted['Away_Elo_Start'] = 0
    match_df_sorted['Home_Elo_End'] = 0
    match_df_sorted['Away_Elo_End'] = 0
    
    # Add match number per format and overall match number
    match_df_sorted['Match_Number'] = range(1, len(match_df_sorted) + 1)
    match_df_sorted['Format_Number'] = match_df_sorted.groupby('Match_Format').cumcount() + 1

    # Calculate ELO ratings for each match
    for index, row in match_df_sorted.iterrows():
        home_team = row['Home_Team']
        away_team = row['Away_Team']
        match_format = row['Match_Format']
        
        # Initialize format dictionary if new
        if match_format not in elo_ratings:
            elo_ratings[match_format] = {}
        
        # Initialize ELO ratings if team is new in this format
        if home_team not in elo_ratings[match_format]:
            elo_ratings[match_format][home_team] = initial_elo
        if away_team not in elo_ratings[match_format]:
            elo_ratings[match_format][away_team] = initial_elo
        
        # Get starting ELO ratings for this format
        home_elo_start = elo_ratings[match_format][home_team]
        away_elo_start = elo_ratings[match_format][away_team]
        
        # Record starting ELO ratings
        match_df_sorted.at[index, 'Home_Elo_Start'] = home_elo_start
        match_df_sorted.at[index, 'Away_Elo_Start'] = away_elo_start
        
        # Determine match result
        if row['Home_Win']:
            home_result = 1
            away_result = 0
        elif row['Home_Lost']:
            home_result = 0
            away_result = 1
        elif row['Home_Drawn']:
            home_result = 0.5
            away_result = 0.5
        else:
            # In case of any other result
            home_result = 0.5
            away_result = 0.5
        
        # Update ELO ratings
        new_home_elo, new_away_elo = update_elo(home_elo_start, away_elo_start, k_factor, home_result)
        
        # Record updated ELO ratings
        match_df_sorted.at[index, 'Home_Elo_End'] = new_home_elo
        match_df_sorted.at[index, 'Away_Elo_End'] = new_away_elo
        
        # Update ELO ratings in dictionary for this format
        elo_ratings[match_format][home_team] = new_home_elo
        elo_ratings[match_format][away_team] = new_away_elo
    
    # Create separate home and away dataframes with ELO information
    home_df = match_df_sorted[['Date', 'Home_Team', 'Away_Team', 'Home_Elo_Start', 'Home_Elo_End', 
                              'Away_Elo_Start', 'Away_Elo_End', 'Match_Format', 'Match_Number', 
                              'Format_Number', 'Margin']].copy()
    home_df.rename(columns={
        'Home_Team': 'Team',
        'Away_Team': 'Opponent',
        'Home_Elo_Start': 'Team_Elo_Start',
        'Home_Elo_End': 'Team_Elo_End',
        'Away_Elo_Start': 'Opponent_Elo_Start',
        'Away_Elo_End': 'Opponent_Elo_End'
    }, inplace=True)
    home_df['Location'] = 'Home'
    home_df['Elo_Change'] = home_df['Team_Elo_End'] - home_df['Team_Elo_Start']

    away_df = match_df_sorted[['Date', 'Home_Team', 'Away_Team', 'Home_Elo_Start', 'Home_Elo_End', 
                              'Away_Elo_Start', 'Away_Elo_End', 'Match_Format', 'Match_Number', 
                              'Format_Number', 'Margin']].copy()
    away_df.rename(columns={
        'Away_Team': 'Team',
        'Home_Team': 'Opponent',
        'Away_Elo_Start': 'Team_Elo_Start',
        'Away_Elo_End': 'Team_Elo_End',
        'Home_Elo_Start': 'Opponent_Elo_Start',
        'Home_Elo_End': 'Opponent_Elo_End'
    }, inplace=True)
    # For away games, we need to adjust the margin to be from the away team's perspective
    away_df['Margin'] = away_df['Margin'].apply(lambda x: x if pd.isna(x) else str(x).replace('runs', 'runs').replace('wickets', 'wickets') if 'wickets' in str(x) else 
                                               (('-' + str(x)) if not str(x).startswith('-') else str(x)[1:]))
    away_df['Location'] = 'Away'
    away_df['Elo_Change'] = away_df['Team_Elo_End'] - away_df['Team_Elo_Start']

    # When creating elo_combined_df, ensure dates are properly parsed
    home_df['Date'] = home_df['Date'].apply(parse_date)
    away_df['Date'] = away_df['Date'].apply(parse_date)
    elo_combined_df = pd.concat([home_df, away_df], ignore_index=True)
    
    # Store the combined dataframe in session state
    st.session_state['elo_df'] = elo_combined_df
    st.session_state['match_df_with_elo'] = match_df_sorted
    st.session_state['current_elo_ratings'] = elo_ratings

    # Apply filters to elo_combined_df
    filtered_elo_df = elo_combined_df.copy()
    
    # Format filter
    if 'All' not in format_choice:
        filtered_elo_df = filtered_elo_df[filtered_elo_df['Match_Format'].isin(format_choice)]
    
    # Team filter
    if 'All' not in team_choice:
        team_mask = (filtered_elo_df['Team'].isin(team_choice))
        filtered_elo_df = filtered_elo_df[team_mask]
    
    # Opponent filter
    if 'All' not in opponent_choice:
        opponent_mask = (filtered_elo_df['Opponent'].isin(opponent_choice))
        filtered_elo_df = filtered_elo_df[opponent_mask]

    filtered_elo_df['Diff'] = filtered_elo_df['Team_Elo_Start'] - filtered_elo_df['Opponent_Elo_Start']

# Define team colors
team_colors = {
    'Afghanistan': '#0033A0',
    'Australia': '#FFD700',
    'Bangladesh': '#006A4E',
    'Canada': '#FF0000',
    'England': '#00247D',
    'India': '#FF9933',
    'Ireland': '#009A44',
    'Namibia': '#0033A0',
    'Nepal': '#DC143C',
    'Netherlands': '#FF6600',
    'New Zealand': '#000000',
    'Oman': '#FF0000',
    'Pakistan': '#006600',
    'Papua New Guinea': '#FFD700',
    'Scotland': '#800080',
    'South Africa': '#006600',
    'Sri Lanka': '#0033A0',
    'USA': '#B22234',
    'Uganda': '#FFCC00',
    'West Indies': '#800000',
    'Zimbabwe': '#EF3340'
}

# Filter and display current ELO ratings
current_ratings = []
for format_name, teams in elo_ratings.items():
    # Skip if format is filtered out
    if 'All' not in format_choice and format_name not in format_choice:
        continue
        
    for team, rating in teams.items():
        # Skip if team is filtered out
        if 'All' not in team_choice and team not in team_choice:
            continue
            
        # Get team's historical Elo ratings for this format
        team_history = filtered_elo_df[
            (filtered_elo_df['Team'] == team) & 
            (filtered_elo_df['Match_Format'] == format_name)
        ]
        
        # Calculate max and min Elo from both start and end ratings
        all_ratings = pd.concat([
            team_history['Team_Elo_Start'],
            team_history['Team_Elo_End']
        ])
        
        max_elo = all_ratings.max() if not all_ratings.empty else rating
        min_elo = all_ratings.min() if not all_ratings.empty else rating
            
        current_ratings.append({
            'Format': format_name,
            'Team': team,
            'Current_ELO': rating,
            'Max_ELO': max_elo,
            'Min_ELO': min_elo
        })

if current_ratings:
    current_ratings_df = pd.DataFrame(current_ratings)
    
    # Ensure no null values in critical columns
    current_ratings_df = current_ratings_df.dropna(subset=['Current_ELO', 'Format', 'Team'])
    
    # Add rank by grouping on format
    current_ratings_df['Rank'] = current_ratings_df.groupby('Format')['Current_ELO'] \
        .rank(method='dense', ascending=False).astype(int)

    # Remove Max_ELO and Min_ELO
    if 'Max_ELO' in current_ratings_df.columns:
        current_ratings_df.drop(columns=['Max_ELO', 'Min_ELO'], inplace=True)

    # Sort by Format and Rank
    current_ratings_df = current_ratings_df.sort_values(['Format', 'Rank'])

    # Beautiful section banner
    st.markdown("""
        <div class="result-banner">
            🏆 Current ELO Rankings
        </div>
    """, unsafe_allow_html=True)

    # Apply conditional formatting to the entire row
    def apply_row_formatting(row):
        try:
            if pd.isna(row['Rank']) or row['Rank'] is None:
                return [''] * len(row)
            rank_val = int(row['Rank'])
            if rank_val == 1:
                return ['background-color: gold; color: black;'] * len(row)
            elif rank_val == 2:
                return ['background-color: silver; color: black;'] * len(row)
            elif rank_val == 3:
                return ['background-color: #cd7f32; color: black;'] * len(row)  # Bronze color
            return [''] * len(row)
        except (ValueError, TypeError):
            return [''] * len(row)

    styled_current_ratings_df = current_ratings_df.style.apply(apply_row_formatting, axis=1).format({
        'Current_ELO': '{:.2f}'
    }, na_rep='')

    st.dataframe(
        styled_current_ratings_df,
        hide_index=True,
        use_container_width=True
    )

    # Create beautiful summary cards for #1 ranked teams in each format
    st.markdown("""
        <div style="margin: 30px 0;">
            <h3 style="text-align: center; color: #f04f53; margin-bottom: 20px;">🥇 Current #1 Rankings by Format</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Get the #1 ranked team for each format
    format_leaders = current_ratings_df[current_ratings_df['Rank'] == 1].copy()
    
    # Create cards in a grid layout - always 3 per row for smaller, more compact cards
    if not format_leaders.empty:
        cols = st.columns(3)
        
        for idx, (_, leader) in enumerate(format_leaders.iterrows()):
            col_idx = idx % 3
            with cols[col_idx]:
                # Get team color for accent
                team_color = team_colors.get(leader['Team'], '#f04f53')
                
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); 
                                padding: 15px; border-radius: 12px; box-shadow: 0 6px 20px rgba(255,215,0,0.3); 
                                margin: 10px 0; text-align: center; border: 2px solid #FFD700; 
                                min-height: 160px; max-height: 160px; position: relative;">
                        <div style="position: absolute; top: 8px; right: 8px; 
                                    background: rgba(255,255,255,0.9); border-radius: 50%; 
                                    width: 30px; height: 30px; display: flex; align-items: center; justify-content: center;">
                            <span style="font-size: 1.2em;">🏆</span>
                        </div>
                        <div style="margin-top: 8px;">
                            <h4 style="color: #8B4513; margin: 0 0 12px 0; font-weight: bold; font-size: 1em;">{leader['Format']}</h4>
                        </div>
                        <div style="background: rgba(255,255,255,0.95); padding: 15px; border-radius: 8px; 
                                    box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 0;">
                            <h3 style="color: {team_color}; margin: 0 0 10px 0; font-size: 1.2em; font-weight: bold;">{leader['Team']}</h3>
                            <div style="text-align: center; margin-top: 8px;">
                                <div style="color: #666; font-weight: bold; font-size: 0.9em; margin-bottom: 4px;">ELO Rating</div>
                                <div style="color: {team_color}; font-size: 1.8em; font-weight: bold; line-height: 1;">{leader['Current_ELO']:.1f}</div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    # Add clear separation before the matrix
    st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)

    # Create a pivot table showing ranks for all formats
    rank_matrix = current_ratings_df.pivot(
        index='Team',
        columns='Format',
        values='Rank'
    ).fillna('')  # Use empty string instead of '-'

    # Convert non-empty values to integers
    for col in rank_matrix.columns:
        rank_matrix[col] = rank_matrix[col].apply(lambda x: int(x) if x != '' else x)

    # Custom sort function to put blank values at bottom and ensure proper numerical sorting
    def sort_with_blanks_at_bottom(df):
        # For each row, count number of non-blank values and calculate average rank
        df['sort_score'] = df.apply(lambda x: (
            sum(1 for v in x if v == ''),  # Count of blank values (higher count means lower priority)
            sum(v for v in x if v != '')/sum(1 for v in x if v != '') if sum(1 for v in x if v != '') > 0 else float('inf')
        ), axis=1)
        sorted_df = df.sort_values('sort_score').drop('sort_score', axis=1)
        return sorted_df

    # Sort the matrix
    rank_matrix = sort_with_blanks_at_bottom(rank_matrix)

    # Ensure proper numerical sorting for columns
    rank_matrix = rank_matrix.apply(lambda col: pd.to_numeric(col, errors='coerce')).sort_values(by=list(rank_matrix.columns), na_position='last')

    # Sort by 'Team' column alphabetically
    rank_matrix = rank_matrix.sort_index()

    # Apply conditional formatting
    def apply_formatting(val):
        if pd.isna(val) or val is None or val == '':
            return ''
        try:
            val = int(val)
            if val == 1:
                return 'background-color: gold; color: black;'
            elif val == 2:
                return 'background-color: silver; color: black;'
            elif val == 3:
                return 'background-color: #cd7f32; color: black;'  # Bronze color
            return ''
        except (ValueError, TypeError):
            return ''

    styled_rank_matrix = rank_matrix.style.applymap(apply_formatting).format(precision=0, na_rep='')

    # Beautiful section banner for rankings by format
    st.markdown("""
        <div class="result-banner">
            🥇 Team Rankings Matrix
        </div>
    """, unsafe_allow_html=True)
    
    st.dataframe(
        styled_rank_matrix,
        use_container_width=True,
        height=20 * 35  # Set height to display 18 rows (approx. 35 pixels per row)
    )

#####===================GRAPH=====================#####
# Beautiful section banner for progression
st.markdown("""
    <div class="result-banner">
        📈 ELO Ratings Progression
    </div>
""", unsafe_allow_html=True)
graph_df = filtered_elo_df.copy()

# Convert Date to datetime
graph_df['Date'] = pd.to_datetime(graph_df['Date'])

# Create a complete timeline for all format/team combinations
min_date = graph_df['Date'].min()
max_date = graph_df['Date'].max()

# Get all months in the date range
all_months = pd.date_range(
    start=pd.Timestamp(min_date).replace(day=1),
    end=pd.Timestamp(max_date) + pd.offsets.MonthEnd(1),
    freq='M'
)

# Create all possible format/team/month combinations
format_teams = graph_df[['Match_Format', 'Team']].drop_duplicates()
timeline_data = []

for _, row in format_teams.iterrows():
    format_name = row['Match_Format']
    team_name = row['Team']
    
    # Get this team's match data
    team_matches = graph_df[
        (graph_df['Match_Format'] == format_name) & 
        (graph_df['Team'] == team_name)
    ].sort_values('Date')
    
    if not team_matches.empty:
        last_rating = None
        for month in all_months:
            # Find matches in this month
            month_matches = team_matches[
                (team_matches['Date'].dt.year == month.year) & 
                (team_matches['Date'].dt.month == month.month)
            ]
            
            if not month_matches.empty:
                # Use the last rating from this month
                rating = month_matches.iloc[-1]['Team_Elo_End']
                last_rating = rating
            elif last_rating is not None:
                # Use the last known rating
                rating = last_rating
            else:
                # Use the first rating we'll see
                first_match = team_matches.iloc[0]
                rating = first_match['Team_Elo_Start']
                last_rating = rating
            
            timeline_data.append({
                'Date': month,
                'Match_Format': format_name,
                'Team': team_name,
                'Rating': rating
            })

# Convert to DataFrame
timeline_df = pd.DataFrame(timeline_data)

# Create the plot
fig = go.Figure()

# Get unique format-team combinations from timeline_df
format_teams = timeline_df.groupby(['Match_Format', 'Team']).size()
num_traces = len(format_teams)

for name, group in timeline_df.groupby(['Match_Format', 'Team']):
    format_name, team_name = name
    color = team_colors.get(team_name, None)
    fig.add_trace(go.Scatter(
        x=group['Date'],
        y=group['Rating'],
        name=f"{team_name} ({format_name})",
        mode='lines+markers',
        line=dict(
            shape='spline',
            smoothing=0.3,
            color=color
        ),
        marker=dict(size=8),
        hovertemplate=
        f"{team_name} ({format_name})<br>" +
        "Date: %{x|%b %Y}<br>" +
        "ELO: %{y:.2f}<br>" +
        "<extra></extra>"
    ))

legend_rows = max(1, round(num_traces / 3))

fig.update_layout(
    xaxis_title="Date (Year-Month)",
    yaxis_title="ELO Rating",
    hovermode='closest',
    height=600 + (legend_rows * 30),
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-(0.1 + (legend_rows-1)*0.1),
        xanchor="center",
        x=0.5,
        traceorder="normal",
        font=dict(size=10),
        itemwidth=40,
        itemsizing="constant"
    )
)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', tickformat="%b %Y")
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
fig.update_layout(margin=dict(b=50 + (legend_rows * 20)))

st.plotly_chart(fig, use_container_width=True)

# ELO Rating Distribution
st.markdown("""
    <div class="result-banner">
        📊 ELO Rating Distribution
    </div>
""", unsafe_allow_html=True)

if 'elo_df' in st.session_state:
    # Use the already filtered dataframe
    dist_df = filtered_elo_df.copy()

    fig = go.Figure()
    for team in sorted(dist_df['Team'].unique()):
        team_ratings = dist_df[dist_df['Team'] == team]['Team_Elo_End']
        color = team_colors.get(team, None)
        fig.add_trace(go.Box(
            y=team_ratings,
            name=team,
            boxpoints='outliers',
            jitter=0.3,
            pointpos=-1.8,
            marker_color=color
        ))

    fig.update_layout(
        yaxis_title="ELO Rating",
        height=500,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)



############===================number 1 per format==============================###############
if 'elo_df' in st.session_state:
    try:
        # Create a copy of the DataFrame
        elo_df = st.session_state['elo_df'].copy()
        
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
        
        # Convert Date to datetime using our parse_date function
        elo_df['Date'] = elo_df['Date'].apply(parse_date)
        
        # Remove any invalid dates
        invalid_dates = elo_df['Date'].isna().sum()
        if invalid_dates > 0:
            elo_df = elo_df.dropna(subset=['Date'])
        
        if len(elo_df) == 0:
            st.error("No valid dates found after parsing")
            st.stop()
        
        # Convert dates to datetime for period calculations
        elo_df['Month_End'] = pd.to_datetime(elo_df['Date']).dt.to_period('M').dt.to_timestamp(how='end')
        
        # Get min and max dates
        min_date = elo_df['Date'].min()
        max_date = elo_df['Date'].max()
        
        # Create date range for all months
        date_range = pd.date_range(
            start=pd.to_datetime(min_date).replace(day=1),
            end=pd.to_datetime(max_date),
            freq='M'
        )
        
        # Initialize results dictionary
        monthly_leaders = {date: {} for date in date_range}
        
        # Filter for selected format if specified
        if 'All' not in format_choice:
            elo_df = elo_df[elo_df['Match_Format'].isin(format_choice)]
        
        # For each format and month, find the team with highest Elo
        for format_name in elo_df['Match_Format'].unique():
            format_data = elo_df[elo_df['Match_Format'] == format_name]
            
            for month_end in date_range:
                # Convert month_end to date for comparison
                month_end_date = month_end.date()
                
                # Get all matches up to this month end
                matches_until_month = format_data[format_data['Date'] <= month_end_date]
                
                if not matches_until_month.empty:
                    # Group by team and get the latest Elo rating for each
                    latest_ratings = matches_until_month.sort_values('Date').groupby('Team').last()
                    
                    # Find the team with max rating
                    if not latest_ratings.empty:
                        max_team = latest_ratings.loc[latest_ratings['Team_Elo_End'].idxmax()]
                        monthly_leaders[month_end][format_name] = (
                            max_team.name,  # Team name
                            max_team['Team_Elo_End']  # Elo rating
                        )
        
        # Create a formatted DataFrame
        formatted_data = []
        for month, format_data in monthly_leaders.items():
            row_dict = {'Month': month.strftime('%m-%Y')}
            row_dict['Sort_Date'] = month
            for format_name in elo_df['Match_Format'].unique():
                if format_name in format_data:
                    team, elo = format_data[format_name]
                    row_dict[format_name] = f"{team} - {elo:.1f}"
                else:
                    row_dict[format_name] = ""
            formatted_data.append(row_dict)
        
        # Create DataFrame only if we have data
        if formatted_data:
            monthly_summary_df = pd.DataFrame(formatted_data)
            
            # Sort by date descending and drop the sort column
            monthly_summary_df = monthly_summary_df.sort_values('Sort_Date', ascending=False).drop('Sort_Date', axis=1)
            
            # Display the monthly summary
            st.markdown("""
                <div class="result-banner">
                    📅 Monthly ELO Leaders by Format
                </div>
            """, unsafe_allow_html=True)
            st.dataframe(
                monthly_summary_df,
                hide_index=True,
                use_container_width=True,
                height=600
            )
            
            # Calculate number of months as #1 for each team per format
            leader_counts = {format_name: {} for format_name in elo_df['Match_Format'].unique()}
            
            # Process each row in monthly_summary_df
            for _, row in monthly_summary_df.iterrows():
                for format_name in elo_df['Match_Format'].unique():
                    if format_name in row and row[format_name]:  # Check if there's data for this format
                        team = row[format_name].split(' - ')[0]  # Extract team name before the hyphen
                        if team not in leader_counts[format_name]:
                            leader_counts[format_name][team] = 0
                        leader_counts[format_name][team] += 1
            
            # Convert to DataFrame
            summary_rows = []
            for format_name, teams in leader_counts.items():
                for team, count in teams.items():
                    summary_rows.append({
                        'Format': format_name,
                        'Team': team,
                        'Months at #1': count
                    })
            
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                
                # Sort by Format and then by number of months (descending)
                summary_df = summary_df.sort_values(['Format', 'Months at #1'], ascending=[True, False])
                
                # Display the summary
                st.markdown("""
                    <div class="result-banner">
                        👑 Total Months as #1 by Team and Format
                    </div>
                """, unsafe_allow_html=True)
                st.dataframe(
                    summary_df,
                    hide_index=True,
                    use_container_width=True,
                    height=600
                )
            else:
                st.warning("No summary data available for the selected filters")
        else:
            st.warning("No data available for the selected filters")
            
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.write("Debug - Sample of dates:", elo_df['Date'].head().tolist())
        st.write("Debug - Date column type:", elo_df['Date'].dtype)
        st.stop()

#################### ALL TIME HIGHEST ELO RATINGS #####################

if 'elo_df' in st.session_state:
    try:
        # Create a copy of the DataFrame
        elo_df_highest = st.session_state['elo_df'].copy()
        
        # Apply filters
        if 'All' not in format_choice:
            elo_df_highest = elo_df_highest[elo_df_highest['Match_Format'].isin(format_choice)]
        
        if 'All' not in team_choice:
            elo_df_highest = elo_df_highest[elo_df_highest['Team'].isin(team_choice)]
        
        if 'All' not in opponent_choice:
            elo_df_highest = elo_df_highest[elo_df_highest['Opponent'].isin(opponent_choice)]
        
        # Convert Date to datetime using our parse_date function
        elo_df_highest['Date'] = elo_df_highest['Date'].apply(parse_date)
        
        # Remove any invalid dates
        elo_df_highest = elo_df_highest.dropna(subset=['Date'])
        
        if len(elo_df_highest) > 0:
            # Convert dates to datetime for month formatting
            elo_df_highest['Date_dt'] = pd.to_datetime(elo_df_highest['Date'])
            elo_df_highest['Month_Year'] = elo_df_highest['Date_dt'].dt.strftime('%m-%Y')
            
            # Create a list of all ELO ratings with their details
            highest_ratings = []
            
            for index, row in elo_df_highest.iterrows():
                highest_ratings.append({
                    'Format': row['Match_Format'],
                    'Team': row['Team'],
                    'Month': row['Month_Year'],
                    'ELO': row['Team_Elo_End'],
                    'Date': row['Date'],
                    'Opponent': row['Opponent']
                })
            
            if highest_ratings:
                highest_ratings_df = pd.DataFrame(highest_ratings)
                
                # Sort by ELO (descending) to show highest ratings first
                highest_ratings_df = highest_ratings_df.sort_values('ELO', ascending=False)
                
                # Format the ELO column to 2 decimal places
                highest_ratings_df['ELO'] = highest_ratings_df['ELO'].round(2)
                
                # Display the all-time highest ratings
                st.markdown("""
                    <div class="result-banner">
                        🏆 All-Time Highest ELO Ratings
                    </div>
                """, unsafe_allow_html=True)
                
                # Apply conditional formatting for top 3 ratings
                def highlight_top_ratings(row):
                    # Get the top 3 ELO ratings overall
                    top_3_ratings = highest_ratings_df.nlargest(3, 'ELO')['ELO'].tolist()
                    if len(top_3_ratings) >= 1 and row['ELO'] == top_3_ratings[0]:
                        return ['background-color: #FFD700; color: black;'] * len(row)  # Gold
                    elif len(top_3_ratings) >= 2 and row['ELO'] == top_3_ratings[1]:
                        return ['background-color: #C0C0C0; color: black;'] * len(row)  # Silver
                    elif len(top_3_ratings) >= 3 and row['ELO'] == top_3_ratings[2]:
                        return ['background-color: #cd7f32; color: black;'] * len(row)  # Bronze
                    return [''] * len(row)
                
                # Display only the relevant columns
                display_cols = ['Format', 'Team', 'Month', 'ELO', 'Opponent']
                styled_highest_df = highest_ratings_df[display_cols].style.apply(highlight_top_ratings, axis=1).format({
                    'ELO': '{:.2f}'
                })
                
                st.dataframe(
                    styled_highest_df,
                    hide_index=True,
                    use_container_width=True,
                    height=600
                )
                
                # Create summary cards for the top 3 highest ratings overall
                st.markdown("""
                    <div style="margin: 30px 0;">
                        <h3 style="text-align: center; color: #f04f53; margin-bottom: 20px;">🥇 Top 3 Highest ELO Ratings Ever</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # Get the top 3 highest ratings overall
                top_3_peaks = highest_ratings_df.nlargest(3, 'ELO')
                
                # Create cards in a grid layout
                if not top_3_peaks.empty:
                    cols = st.columns(3)
                    
                    for idx, (_, peak) in enumerate(top_3_peaks.iterrows()):
                        with cols[idx]:
                            # Get team color for accent
                            team_color = team_colors.get(peak['Team'], '#f04f53')
                            
                            # Determine medal color based on ranking
                            if idx == 0:
                                medal_color = '#FFD700'  # Gold
                                medal_emoji = '🥇'
                            elif idx == 1:
                                medal_color = '#C0C0C0'  # Silver
                                medal_emoji = '🥈'
                            else:
                                medal_color = '#cd7f32'  # Bronze
                                medal_emoji = '🥉'
                            
                            st.markdown(f"""
                                <div style="background: linear-gradient(135deg, {medal_color} 0%, #FFA500 100%); 
                                            padding: 15px; border-radius: 12px; box-shadow: 0 6px 20px rgba(255,215,0,0.3); 
                                            margin: 10px 0; text-align: center; border: 2px solid {medal_color}; 
                                            min-height: 200px; max-height: 200px; position: relative;">
                                    <div style="position: absolute; top: 8px; right: 8px; 
                                                background: rgba(255,255,255,0.9); border-radius: 50%; 
                                                width: 30px; height: 30px; display: flex; align-items: center; justify-content: center;">
                                        <span style="font-size: 1.2em;">{medal_emoji}</span>
                                    </div>
                                    <div style="margin-top: 8px;">
                                        <h4 style="color: #8B4513; margin: 0 0 8px 0; font-weight: bold; font-size: 0.9em;">{peak['Format']}</h4>
                                    </div>
                                    <div style="background: rgba(255,255,255,0.95); padding: 15px; border-radius: 8px; 
                                                box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 0;">
                                        <h3 style="color: {team_color}; margin: 0 0 8px 0; font-size: 1.1em; font-weight: bold;">{peak['Team']}</h3>
                                        <div style="text-align: center; margin-top: 8px;">
                                            <div style="color: #666; font-weight: bold; font-size: 0.85em; margin-bottom: 4px;">Peak ELO Rating</div>
                                            <div style="color: {team_color}; font-size: 1.6em; font-weight: bold; line-height: 1; margin-bottom: 4px;">{peak['ELO']:.2f}</div>
                                            <div style="color: #666; font-size: 0.8em; font-weight: normal; margin-bottom: 2px;">{peak['Month']}</div>
                                            <div style="color: #666; font-size: 0.75em; font-weight: normal;">vs {peak['Opponent']}</div>
                                        </div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
            else:
                st.warning("No highest rating data available for the selected filters")
        else:
            st.warning("No valid data available for the selected filters")
            
    except Exception as e:
        st.error(f"Error processing highest ratings data: {str(e)}")

#################### UPDATES #####################



# Add after the head-to-head analysis
st.markdown("""
    <div class="result-banner">
        📊 Performance Metrics
    </div>
""", unsafe_allow_html=True)

if 'elo_df' in st.session_state:
    pm_df = filtered_elo_df.copy()
    metrics = []
    format_team_pairs = pm_df[['Match_Format', 'Team']].drop_duplicates()
    for _, row in format_team_pairs.iterrows():
        fmt = row['Match_Format']
        team = row['Team']
        team_data = pm_df[(pm_df['Match_Format'] == fmt) & (pm_df['Team'] == team)]
        if len(team_data) == 0:
            continue
        metrics.append({
            'Format': fmt,
            'Team': team,
            'Matches': len(team_data),
            'Avg ELO': round(team_data['Team_Elo_End'].mean(), 2),
            'Biggest Win': round(team_data['Elo_Change'].max(), 2),
            'Biggest Loss': round(team_data['Elo_Change'].min(), 2)
        })

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df[['Format', 'Team', 'Matches', 'Avg ELO', 'Biggest Win', 'Biggest Loss']]

    st.dataframe(
        metrics_df.sort_values(['Format', 'Avg ELO'], ascending=[True, False]),
        hide_index=True,
        use_container_width=True
    )


#########################new features------------------

# Display the filtered ELO Ratings Dataset with beautiful styling
st.markdown("""
    <div class="result-banner">
        📊 ELO Ratings Match by Match
    </div>
""", unsafe_allow_html=True)

# Sort the DataFrame first
sorted_filtered_elo_df = filtered_elo_df.sort_values(['Match_Format', 'Date', 'Match_Number'])

# Select the columns to display and ensure no null values in critical columns
display_df = sorted_filtered_elo_df[['Date', 'Match_Format', 'Team', 'Opponent', 
                                     'Location', 'Team_Elo_Start', 'Opponent_Elo_Start', 'Diff', 'Elo_Change',
                                     'Team_Elo_End', 'Opponent_Elo_End', 
                                     'Format_Number', 'Margin']].copy()

# Fill any NaN values in numeric columns
numeric_cols = ['Team_Elo_Start', 'Opponent_Elo_Start', 'Diff', 'Elo_Change', 'Team_Elo_End', 'Opponent_Elo_End', 'Format_Number']
for col in numeric_cols:
    if col in display_df.columns:
        display_df[col] = display_df[col].fillna(0)

# Apply conditional formatting with softer colors
def color_elo_change(val):
    if pd.isna(val) or val is None:
        return ''
    try:
        val = float(val)
        color = '#d4edda' if val > 0 else '#f8d7da' if val < 0 else '#fff3cd'
        return f'background-color: {color}'
    except (ValueError, TypeError):
        return ''

styled_display_df = display_df.style.applymap(color_elo_change, subset=['Elo_Change', 'Diff']).format({
    'Team_Elo_Start': '{:.2f}',
    'Opponent_Elo_Start': '{:.2f}',
    'Diff': '{:.2f}',
    'Elo_Change': '{:.2f}',
    'Team_Elo_End': '{:.2f}',
    'Opponent_Elo_End': '{:.2f}'
}, na_rep='')

st.dataframe(
    styled_display_df,
    hide_index=True,
    use_container_width=True
)

# Add after the Head-to-Head Analysis
st.markdown("""
    <div class="result-banner">
        📈 ELO Rating Volatility
    </div>
""", unsafe_allow_html=True)

if 'elo_df' in st.session_state:
    vol_df = filtered_elo_df.copy()
    volatility_data = []
    for team in sorted(vol_df['Team'].unique()):
        team_data = vol_df[vol_df['Team'] == team]
        
        volatility_data.append({
            'Team': team,
            'Std Dev': round(team_data['Team_Elo_End'].std(), 2),
            'Avg Change': round(abs(team_data['Elo_Change']).mean(), 2),
            'Max Swing': round(max(abs(team_data['Elo_Change'])), 2),
            'Rating Range': round(team_data['Team_Elo_End'].max() - team_data['Team_Elo_End'].min(), 2)
        })
    
    volatility_df = pd.DataFrame(volatility_data)
    
    # Create bar chart
    fig = go.Figure()
    
    for metric in ['Std Dev', 'Avg Change', 'Max Swing']:
        fig.add_trace(go.Bar(
            name=metric,
            x=volatility_df['Team'],
            y=volatility_df[metric],
            text=volatility_df[metric],
            textposition='auto',
        ))
    
    fig.update_layout(
        barmode='group',
        height=500,
        #title="Elo Rating Volatility Metrics by Team",
        xaxis_title="Team",
        yaxis_title="Rating Points"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed table
    st.dataframe(
        volatility_df.sort_values('Std Dev', ascending=False),
        hide_index=True,
        use_container_width=True
    )

# Add Metrics Explanation Section
st.markdown("""
    <div class="result-banner">
        📚 Understanding the Metrics
    </div>
""", unsafe_allow_html=True)

# Enhanced explanation cards with clean, professional design - all same height
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); 
                    margin: 20px 0; text-align: center; border: 1px solid #e9ecef; height: 450px; display: flex; flex-direction: column;">
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 15px;">
                <span style="font-size: 2em; margin-right: 10px;">📊</span>
                <h3 style="color: #f04f53; margin: 0; font-weight: bold;">Performance Metrics</h3>
            </div>
            <div style="text-align: left; max-width: 600px; margin: 0 auto; flex-grow: 1; display: flex; flex-direction: column; justify-content: center;">
                <div style="margin-bottom: 10px; padding: 10px; border: 1px solid #f04f53; border-radius: 8px; background: #fafafa;">
                    <strong style="color: #f04f53;">Matches:</strong> <span style="color: #666;">Total games played by the team</span>
                </div>
                <div style="margin-bottom: 10px; padding: 10px; border: 1px solid #f04f53; border-radius: 8px; background: #fafafa;">
                    <strong style="color: #f04f53;">Avg ELO:</strong> <span style="color: #666;">Team's average rating across all matches</span>
                </div>
                <div style="margin-bottom: 10px; padding: 10px; border: 1px solid #38ab2f; border-radius: 8px; background: #fafafa;">
                    <strong style="color: #38ab2f;">Biggest Win:</strong> <span style="color: #666;">Largest rating gain in a single match</span>
                </div>
                <div style="margin-bottom: 0; padding: 10px; border: 1px solid #dc3545; border-radius: 8px; background: #fafafa;">
                    <strong style="color: #dc3545;">Biggest Loss:</strong> <span style="color: #666;">Largest rating drop in a single match</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); 
                    margin: 20px 0; text-align: center; border: 1px solid #e9ecef; height: 450px; display: flex; flex-direction: column;">
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 15px;">
                <span style="font-size: 2em; margin-right: 10px;">📈</span>
                <h3 style="color: #f04f53; margin: 0; font-weight: bold;">ELO Distribution</h3>
            </div>
            <p style="color: #666; font-style: italic; margin-bottom: 15px; text-align: center;">Box plot visualization components:</p>
            <div style="text-align: left; max-width: 600px; margin: 0 auto; flex-grow: 1; display: flex; flex-direction: column; justify-content: center;">
                <div style="margin-bottom: 8px; padding: 8px; border: 1px solid #667eea; border-radius: 6px; background: #fafafa;">
                    <strong style="color: #667eea;">Middle Line:</strong> <span style="color: #666;">Median ELO rating</span>
                </div>
                <div style="margin-bottom: 8px; padding: 8px; border: 1px solid #764ba2; border-radius: 6px; background: #fafafa;">
                    <strong style="color: #764ba2;">Box:</strong> <span style="color: #666;">50% of ratings (25th-75th percentile)</span>
                </div>
                <div style="margin-bottom: 8px; padding: 8px; border: 1px solid #f04f53; border-radius: 6px; background: #fafafa;">
                    <strong style="color: #f04f53;">Whiskers:</strong> <span style="color: #666;">Full range (excluding outliers)</span>
                </div>
                <div style="margin-bottom: 0; padding: 8px; border: 1px solid #ffc107; border-radius: 6px; background: #fafafa;">
                    <strong style="color: #ffc107;">Points:</strong> <span style="color: #666;">Outlier ratings (unusual values)</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); 
                    margin: 20px 0; text-align: center; border: 1px solid #e9ecef; height: 450px; display: flex; flex-direction: column;">
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 15px;">
                <span style="font-size: 2em; margin-right: 10px;">📊</span>
                <h3 style="color: #f04f53; margin: 0; font-weight: bold;">Rating Volatility</h3>
            </div>
            <p style="color: #666; font-style: italic; margin-bottom: 15px; text-align: center;">Consistency and variation metrics:</p>
            <div style="text-align: left; max-width: 600px; margin: 0 auto; flex-grow: 1; display: flex; flex-direction: column; justify-content: center;">
                <div style="margin-bottom: 10px; padding: 10px; border: 1px solid #28a745; border-radius: 8px; background: #fafafa;">
                    <strong style="color: #28a745;">Std Dev:</strong> <span style="color: #666;">Lower = more consistent performance</span>
                </div>
                <div style="margin-bottom: 10px; padding: 10px; border: 1px solid #17a2b8; border-radius: 8px; background: #fafafa;">
                    <strong style="color: #17a2b8;">Avg Change:</strong> <span style="color: #666;">Typical rating change per match</span>
                </div>
                <div style="margin-bottom: 10px; padding: 10px; border: 1px solid #ffc107; border-radius: 8px; background: #fafafa;">
                    <strong style="color: #ffc107;">Max Swing:</strong> <span style="color: #666;">Largest single-match change</span>
                </div>
                <div style="margin-bottom: 0; padding: 10px; border: 1px solid #6610f2; border-radius: 8px; background: #fafafa;">
                    <strong style="color: #6610f2;">Rating Range:</strong> <span style="color: #666;">Highest minus lowest rating</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); 
                    margin: 20px 0; text-align: center; border: 1px solid #e9ecef; height: 450px; display: flex; flex-direction: column;">
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 15px;">
                <span style="font-size: 2em; margin-right: 10px;">👑</span>
                <h3 style="color: #f04f53; margin: 0; font-weight: bold;">Monthly Leaders</h3>
            </div>
            <p style="color: #666; font-style: italic; margin-bottom: 15px; text-align: center;">Leadership tracking over time:</p>
            <div style="text-align: left; max-width: 600px; margin: 0 auto; flex-grow: 1; display: flex; flex-direction: column; justify-content: center;">
                <div style="margin-bottom: 8px; padding: 8px; border: 1px solid #ffd700; border-radius: 6px; background: #fafafa;">
                    <strong style="color: #ffd700;">Team:</strong> <span style="color: #666;">Leading team for each format</span>
                </div>
                <div style="margin-bottom: 8px; padding: 8px; border: 1px solid #f04f53; border-radius: 6px; background: #fafafa;">
                    <strong style="color: #f04f53;">Rating:</strong> <span style="color: #666;">Highest ELO achieved that month</span>
                </div>
                <div style="margin-bottom: 0; padding: 8px; border: 1px solid #38ab2f; border-radius: 6px; background: #fafafa;">
                    <strong style="color: #38ab2f;">Duration:</strong> <span style="color: #666;">Time maintaining #1 position</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Enhanced ELO Calculation section with clean, professional design
st.markdown("---")
st.markdown("""
    <div style="background: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); 
                margin: 30px 0; text-align: center; border: 1px solid #e9ecef;">
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
            <span style="font-size: 3em; margin-right: 15px;">ℹ️</span>
            <h2 style="color: #f04f53; margin: 0; font-weight: bold;">About ELO Calculations</h2>
        </div>
        <p style="color: #666; font-size: 1.1em; text-align: center; margin-bottom: 25px; font-style: italic;">
            The ELO rating system uses these key components:
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px; max-width: 800px; margin: 0 auto;">
            <div style="padding: 15px; background: #fafafa; border-radius: 12px; border: 2px solid #28a745;">
                <strong style="color: #28a745; font-size: 1.1em;">Match Outcome</strong><br>
                <span style="color: #666;">Win/Loss/Draw result determines rating change direction</span>
            </div>
            <div style="padding: 15px; background: #fafafa; border-radius: 12px; border: 2px solid #17a2b8;">
                <strong style="color: #17a2b8; font-size: 1.1em;">Rating Difference</strong><br>
                <span style="color: #666;">Expected vs actual performance affects magnitude</span>
            </div>
            <div style="padding: 15px; background: #fafafa; border-radius: 12px; border: 2px solid #ffc107;">
                <strong style="color: #ffc107; font-size: 1.1em;">K-Factor: 32</strong><br>
                <span style="color: #666;">Maximum possible rating change per match</span>
            </div>
            <div style="padding: 15px; background: #fafafa; border-radius: 12px; border: 2px solid #6610f2;">
                <strong style="color: #6610f2; font-size: 1.1em;">Format Specific</strong><br>
                <span style="color: #666;">Separate ratings maintained for each match format</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Add footer with additional information
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                border-radius: 15px; margin: 20px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
        <p style='color: #666; font-size: 0.9em; margin: 0;'>
            📊 ELO Rating Analysis • Cricket Captain 2025 Stats Pack
        </p>
    </div>
""", unsafe_allow_html=True)
