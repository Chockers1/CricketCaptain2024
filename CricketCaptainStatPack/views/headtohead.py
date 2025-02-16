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
st.markdown("<h1 style='color:#f04f53; text-align: center;'>Head to Head</h1>", unsafe_allow_html=True)

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

# Create tabs (UPDATED)
tabs = st.tabs(["Match", "Series", "Tournaments"])

# Existing Match Tab code
with tabs[0]:
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
        
        # Format Date columns to dd/mm/yyyy after parsing
        raw_matches['Date'] = raw_matches['Date'].apply(
            lambda d: d.strftime('%d/%m/%Y') if not pd.isnull(d) else ''
        )
        
        # Display the filtered and sorted matches
        st.dataframe(raw_matches, use_container_width=True, hide_index=True)
        
        # Add metrics if a team is selected
        if 'All' not in team_choice and team_choice:
            total_matches = len(raw_matches)
            total_wins = sum(1 for margin in raw_matches['Margin'] if any(team in margin for team in team_choice) and 'won' in margin)
            total_losses = sum(1 for margin in raw_matches['Margin'] if all(team not in margin for team in team_choice) and 'won' in margin)
            total_draws = sum(1 for margin in raw_matches['Margin'] if 'drawn' in margin)
            win_percentage = (total_wins / total_matches) * 100 if total_matches > 0 else 0
            loss_percentage = (total_losses / total_matches) * 100 if total_matches > 0 else 0

            st.markdown("<h1 style='color:#f04f53; text-align: center;'>Team Record</h1>", unsafe_allow_html=True)

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Matches", total_matches, border=True)
            with col2:
                st.metric("Won", total_wins, border=True)
            with col3:
                st.metric("Lost", total_losses, border=True)
            with col4:
                st.metric("Win %", f"{win_percentage:.2f}%", border=True)
            with col5:
                st.metric("Lost %", f"{loss_percentage:.2f}%", border=True)
    else:
        st.info("No match records available for head-to-head analysis.")

    if 'match_df' in st.session_state:
        df = st.session_state['match_df'].copy()
        df['Date'] = df['Date'].apply(parse_date)
        
        # Create separate records for home and away perspectives
        home_stats = df[['Home_Team', 'Away_Team', 'Home_Win', 'Home_Lost', 'Home_Drawn', 'Match_Format']].copy()
        away_stats = df[['Home_Team', 'Away_Team', 'Home_Win', 'Home_Lost', 'Home_Drawn', 'Match_Format']].copy()
        
        home_stats.columns = ['Team', 'Opponent', 'Won', 'Lost', 'Drawn', 'Match_Format']
        away_stats.columns = ['Opponent', 'Team', 'Lost', 'Won', 'Drawn', 'Match_Format']
        
        combined = pd.concat([home_stats, away_stats], ignore_index=True)
        
        # Apply the filters defined on the page
        filtered_combined = filter_by_all(combined)
        
        # Aggregate the records by format
        agg_df = filtered_combined.groupby('Match_Format').agg({
            'Won': 'sum',
            'Lost': 'sum',
            'Drawn': 'sum'
        }).reset_index()
        
        agg_df['Matches'] = agg_df['Won'] + agg_df['Lost'] + agg_df['Drawn']
        agg_df['Win%'] = (agg_df['Won'] / agg_df['Matches'] * 100).round(1)
        agg_df['Draw%'] = (agg_df['Drawn'] / agg_df['Matches'] * 100).round(1)
        agg_df['Lost%'] = (agg_df['Lost'] / agg_df['Matches'] * 100).round(1)
        
        # Order and rename columns as required
        summary_df = agg_df[['Match_Format', 'Matches', 'Won', 'Drawn', 'Lost', 'Win%', 'Draw%', 'Lost%']].rename(
            columns={'Match_Format': 'Format'}
        )
        
        st.markdown("<h2 style='color:#f04f53; text-align: center;'>Match Summary by Format</h2>", unsafe_allow_html=True)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    ###################

    # Form Guide
    if 'match_df' in st.session_state and 'All' not in team_choice:
        st.markdown("<h1 style='color:#f04f53; text-align: center;'>Form Guide</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#f04f53; text-align: center; margin-top: -15px; font-size: 0.9em;'>← Latest</h3>", unsafe_allow_html=True)
        
        # Style for the form indicators with improved layout
        form_styles = """
        <style>
        .form-container {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background: white;
            border-radius: 8px;
            margin: 10px auto;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            max-width: 100%;
            width: 100%;
        }
        .form-indicator {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            color: white;
            font-weight: bold;
            margin: 0 4px;
            font-size: 16px;
            flex-shrink: 0;
            position: relative;
        }
        .form-indicator:hover .tooltip {
            visibility: visible;
            opacity: 1;
        }
        .tooltip {
            visibility: hidden;
            background-color: rgba(0, 0, 0, 0.75);
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 8px;
            position: absolute;
            z-index: 1;
            bottom: 125%; /* Position above the indicator */
            left: 50%;
            margin-left: -75px;
            opacity: 0;
            transition: opacity 0.3s;
            width: 150px;
            font-size: 12px;
            line-height: 1.4;
        }
        .tooltip::after {
            content: "";
            position: absolute;
            top: 100%; /* Arrow at the bottom */
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: rgba(0, 0, 0, 0.75) transparent transparent transparent;
        }
        .form-indicators-container {
            display: flex;
            flex-wrap: nowrap;
            justify-content: center;
            gap: 8px;
            flex: 1;
            padding: 0 20px;
        }
        .team-name {
            font-weight: bold;
            width: 150px;
            font-size: 16px;
            text-align: right;
            padding-right: 20px;
            flex-shrink: 0;
        }
        .win {
            background-color: #28a745;
        }
        .draw {
            background-color: #ffc107;
            color: black;
        }
        .loss {
            background-color: #dc3545;
        }
        </style>
        """
        st.markdown(form_styles, unsafe_allow_html=True)

        for team in team_choice:
            # Use recent 20 matches for the team
            team_matches = raw_matches[
                (raw_matches['Home Team'] == team) | 
                (raw_matches['Away Team'] == team)
            ].copy().head(20)

            # Create form indicators
            form_indicators = []
            for _, match in team_matches.iterrows():
                margin = match['Margin']
                is_home = match['Home Team'] == team
                
                # Get the date for the tooltip
                date_val = pd.to_datetime(match['Date'], dayfirst=True, errors='coerce')
                if pd.notnull(date_val):
                    date_str = date_val.strftime('%Y-%m-%d')
                else:
                    date_str = ''
                opponent = match['Away Team'] if is_home else match['Home Team']
                tooltip = f"<b>Date:</b> {date_str}<br><b>Margin:</b> {margin}"
                
                if 'won by' in margin:
                    winning_team = margin.split(' won')[0]
                    if winning_team == team:
                        form_indicators.append(f'<div class="form-indicator win"><span class="tooltip">{tooltip}</span>W</div>')
                    else:
                        form_indicators.append(f'<div class="form-indicator loss"><span class="tooltip">{tooltip}</span>L</div>')
                else:
                    form_indicators.append(f'<div class="form-indicator draw"><span class="tooltip">{tooltip}</span>D</div>')

            # Display header with team and format on one line and form indicators beneath
            if form_indicators:  # Only display if there are matches
                header_text = f"{team} - {fmt}" if 'fmt' in globals() else team
                form_html = f"""
                <div class="form-container">
                    <div style="text-align: center; font-weight: bold; font-size: 1.2em;">
                        {header_text}
                    </div>
                    <div class="form-indicators-container">
                        {''.join(reversed(form_indicators))}
                    </div>
                </div>
                """
                st.markdown(form_html, unsafe_allow_html=True)

            # Add a row per format only if "All" is selected
            if 'All' in format_choice:
                for fm in formats:
                    if fm != 'All':
                        sub_matches = raw_matches[
                            ((raw_matches['Home Team'] == team) | (raw_matches['Away Team'] == team)) &
                            (raw_matches['Format'] == fm)
                        ].copy().head(20)

                        # Create form indicators for sub_matches
                        form_indicators = []
                        for _, match in sub_matches.iterrows():
                            margin = match['Margin']
                            is_home = match['Home Team'] == team
                            
                            # Get the date for the tooltip
                            date_val = pd.to_datetime(match['Date'], dayfirst=True, errors='coerce')
                            if pd.notnull(date_val):
                                date_str = date_val.strftime('%Y-%m-%d')
                            else:
                                date_str = ''
                            opponent = match['Away Team'] if is_home else match['Home Team']
                            tooltip = f"<b>Date:</b> {date_str}<br><b>Margin:</b> {margin}"
                            
                            if 'won by' in margin:
                                winning_team = margin.split(' won')[0]
                                if winning_team == team:
                                    form_indicators.append(f'<div class="form-indicator win"><span class="tooltip">{tooltip}</span>W</div>')
                                else:
                                    form_indicators.append(f'<div class="form-indicator loss"><span class="tooltip">{tooltip}</span>L</div>')
                            else:
                                form_indicators.append(f'<div class="form-indicator draw"><span class="tooltip">{tooltip}</span>D</div>')

                        # Calculate record for the specific format
                        total = len(sub_matches)
                        wins = sub_matches['Margin'].apply(lambda m: 1 if str(m).startswith(team) else 0).sum()
                        draws = sub_matches['Margin'].apply(lambda m: 1 if 'drawn' in str(m).lower() else 0).sum()
                        losses = total - wins - draws
                        record_str = f"Pld {total}, W {wins}, L {losses}, D {draws}"

                        # Display team name and form for the specific format
                        if form_indicators:  # Only display if there are matches
                            form_html = f"""
                            <div class="form-container">
                                <span class="team-name">{team} ({fm}) {record_str}</span>
                                <div class="form-indicators-container">
                                    {''.join(reversed(form_indicators))}
                                </div>
                            </div>
                            """
                            st.markdown(form_html, unsafe_allow_html=True)


    else:
        if 'match_df' in st.session_state:
            st.info("Select a team in the filter to see their form guide.")

    #######
    if 'match_df' in st.session_state and 'All' not in team_choice:
        for team in team_choice:
            team_matches = raw_matches[
                (raw_matches['Home Team'] == team) | 
                (raw_matches['Away Team'] == team)
            ].head(50)[::-1]  # Last 50 matches, reversed to go from old to recent
            
            # Create win/loss/draw data for plotting
            results = []
            colors = []
            for _, match in team_matches.iterrows():
                if 'drawn' in match['Margin'].lower():
                    results.append(0.5)  # Draw
                    colors.append('#ffc107')  # Amber
                elif match['Margin'].startswith(team):
                    results.append(1)    # Win
                    colors.append('#28a745')  # Green
                else:
                    results.append(0)    # Loss
                    colors.append('#dc3545')  # Red
            
            # Create line chart
            fig = go.Figure()
            
            # Add the main line with custom shape
            fig.add_trace(go.Scatter(
                y=results,
                mode='lines',
                name='Performance',
                line=dict(
                    shape='spline',  # Makes the line more fluid
                    smoothing=0.8,   # Adjusts the smoothness
                    width=2,
                    color='#666666'  # Neutral gray for the line
                )
            ))
            
            # Add colored markers on top
            fig.add_trace(go.Scatter(
                y=results,
                mode='markers',
                marker=dict(
                    size=10,
                    color=colors,
                    line=dict(
                        width=2,
                        color='white'
                    )
                ),
                showlegend=False
            ))
            
            fig.update_layout(
                #title=f"{team} - Performance Trend",
                yaxis=dict(
                    ticktext=["Loss", "Draw", "Win"],
                    tickvals=[0, 0.5, 1],
                    range=[-0.1, 1.1],
                    gridcolor='lightgray'
                ),
                xaxis=dict(
                    title="Last 50 Matches (Old → Recent)",
                    gridcolor='lightgray'
                ),
                plot_bgcolor='white',
                showlegend=False,
                height=300
            )
            
            # Add markdown-style team name as a title
            st.markdown(f"<h1 style='color:#f04f53; text-align: center;'>{team} - Performance Trend </h1>", unsafe_allow_html=True)
            
            st.plotly_chart(fig, use_container_width=True)

    ###########

    # Add before the Form Guide
    if 'match_df' in st.session_state and 'All' not in team_choice:
        for team in team_choice:
            team_matches = raw_matches[
                (raw_matches['Home Team'] == team) | 
                (raw_matches['Away Team'] == team)
            ].sort_values('Date')  # Ensure matches are sorted by date
            
            # Initialize streak tracking variables
            current_win_streak = 0
            current_unbeaten_streak = 0
            current_loss_streak = 0
            current_winless_streak = 0
            
            longest_win_streak = 0
            longest_unbeaten_streak = 0
            longest_loss_streak = 0
            longest_winless_streak = 0
            
            # Initialize date tracking
            win_streak_start_date = None
            win_streak_end_date = None
            unbeaten_streak_start_date = None
            unbeaten_streak_end_date = None
            loss_streak_start_date = None
            loss_streak_end_date = None
            winless_streak_start_date = None
            winless_streak_end_date = None
            
            longest_win_streak_start_date = None
            longest_win_streak_end_date = None
            longest_unbeaten_streak_start_date = None
            longest_unbeaten_streak_end_date = None
            longest_loss_streak_start_date = None
            longest_loss_streak_end_date = None
            longest_winless_streak_start_date = None
            longest_winless_streak_end_date = None

            # Calculate streaks
            for _, match in team_matches.iterrows():
                margin = match['Margin']
                date = match['Date']
                
                is_win = margin.startswith(team)
                is_draw = 'drawn' in margin.lower()
                is_loss = not (is_win or is_draw)
                
                # Win streak tracking
                if is_win:
                    if current_win_streak == 0:
                        win_streak_start_date = date
                    current_win_streak += 1
                    win_streak_end_date = date
                    
                    if current_win_streak > longest_win_streak:
                        longest_win_streak = current_win_streak
                        longest_win_streak_start_date = win_streak_start_date
                        longest_win_streak_end_date = win_streak_end_date
                    
                    # Reset loss streak
                    current_loss_streak = 0
                    loss_streak_start_date = None
                    loss_streak_end_date = None
                else:
                    current_win_streak = 0
                    win_streak_start_date = None
                    win_streak_end_date = None
                
                # Unbeaten streak tracking
                if is_win or is_draw:
                    if current_unbeaten_streak == 0:
                        unbeaten_streak_start_date = date
                    current_unbeaten_streak += 1
                    unbeaten_streak_end_date = date
                    
                    if current_unbeaten_streak > longest_unbeaten_streak:
                        longest_unbeaten_streak = current_unbeaten_streak
                        longest_unbeaten_streak_start_date = unbeaten_streak_start_date
                        longest_unbeaten_streak_end_date = unbeaten_streak_end_date
                else:
                    current_unbeaten_streak = 0
                    unbeaten_streak_start_date = None
                    unbeaten_streak_end_date = None
                
                # Loss streak tracking
                if is_loss:
                    if current_loss_streak == 0:
                        loss_streak_start_date = date
                    current_loss_streak += 1
                    loss_streak_end_date = date
                    
                    if current_loss_streak > longest_loss_streak:
                        longest_loss_streak = current_loss_streak
                        longest_loss_streak_start_date = loss_streak_start_date
                        longest_loss_streak_end_date = loss_streak_end_date
                else:
                    current_loss_streak = 0
                    loss_streak_start_date = None
                    loss_streak_end_date = None
                
                # Winless streak tracking
                if is_loss or is_draw:
                    if current_winless_streak == 0:
                        winless_streak_start_date = date
                    current_winless_streak += 1
                    winless_streak_end_date = date
                    
                    if current_winless_streak > longest_winless_streak:
                        longest_winless_streak = current_winless_streak
                        longest_winless_streak_start_date = winless_streak_start_date
                        longest_winless_streak_end_date = winless_streak_end_date
                else:
                    current_winless_streak = 0
                    winless_streak_start_date = None
                    winless_streak_end_date = None

            def format_streak_date(date_val):
                """Helper function to format streak dates"""
                if isinstance(date_val, str):
                    date_val = pd.to_datetime(date_val, errors='coerce')
                return date_val.strftime('%Y-%m-%d') if pd.notnull(date_val) else 'N/A'

            # Display streaks in columns
            col1, col2 = st.columns(2)

            # Current Streaks
            with col1:
                st.markdown(
                    f"<h2 style='color:#f04f53; text-align: center;'>Current Streaks</h2>",
                    unsafe_allow_html=True,
                )
                current_streaks_data = {
                    "Metric": [
                        "Win Streak",
                        "Win Streak Dates",
                        "Unbeaten Run",
                        "Unbeaten Run Dates",
                        "Loss Streak",
                        "Loss Streak Dates",
                        "Winless Run",
                        "Winless Run Dates",
                    ],
                    "Value": [
                        current_win_streak,
                        f"{format_streak_date(win_streak_start_date)} to {format_streak_date(win_streak_end_date)}",
                        current_unbeaten_streak,
                        f"{format_streak_date(unbeaten_streak_start_date)} to {format_streak_date(unbeaten_streak_end_date)}",
                        current_loss_streak,
                        f"{format_streak_date(loss_streak_start_date)} to {format_streak_date(loss_streak_end_date)}",
                        current_winless_streak,
                        f"{format_streak_date(winless_streak_start_date)} to {format_streak_date(winless_streak_end_date)}",
                    ],
                }
                current_streaks_df = pd.DataFrame(current_streaks_data)
                st.dataframe(current_streaks_df, hide_index=True, use_container_width=True)

            # Longest Streaks
            with col2:
                st.markdown(
                    f"<h2 style='color:#f04f53; text-align: center;'>Longest Streaks</h2>",
                    unsafe_allow_html=True,
                )
                longest_streaks_data = {
                    "Metric": [
                        "Win Streak",
                        "Win Streak Dates",
                        "Unbeaten Run",
                        "Unbeaten Run Dates",
                        "Loss Streak",
                        "Loss Streak Dates",
                        "Winless Run",
                        "Winless Run Dates",
                    ],
                    "Value": [
                        longest_win_streak,
                        f"{format_streak_date(longest_win_streak_start_date)} to {format_streak_date(longest_win_streak_end_date)}",
                        longest_unbeaten_streak,
                        f"{format_streak_date(longest_unbeaten_streak_start_date)} to {format_streak_date(longest_unbeaten_streak_end_date)}",
                        longest_loss_streak,
                        f"{format_streak_date(longest_loss_streak_start_date)} to {format_streak_date(longest_loss_streak_end_date)}",
                        longest_winless_streak,
                        f"{format_streak_date(longest_winless_streak_start_date)} to {format_streak_date(longest_winless_streak_end_date)}",
                    ],
                }
                longest_streaks_df = pd.DataFrame(longest_streaks_data)
                st.dataframe(longest_streaks_df, hide_index=True, use_container_width=True)

    ################## updated graphics ############


    # Add a win percentage by year chart
    if 'match_df' in st.session_state and 'All' not in team_choice:
        st.markdown("<h1 style='color:#f04f53; text-align: center;'>Win Percentage by Year</h1>", unsafe_allow_html=True)
        
        for team in team_choice:
            # Get matches for the team and ensure Date is datetime
            team_matches = raw_matches[
                (raw_matches['Home Team'] == team) | 
                (raw_matches['Away Team'] == team)
            ].copy()
            
            # Extract year using string operations since Date is already in datetime format
            team_matches['Year'] = pd.to_datetime(team_matches['Date'], dayfirst=True).apply(lambda x: x.year)
            
            # Calculate win percentage by year
            yearly_stats = []
            for year in sorted(team_matches['Year'].unique()):
                year_matches = team_matches[team_matches['Year'] == year]
                wins = sum(
                    (year_matches['Home Team'] == team) & (year_matches['Margin'].str.startswith(team)) |
                    (year_matches['Away Team'] == team) & (year_matches['Margin'].str.startswith(team))
                )
                total = len(year_matches)
                win_pct = round((wins / total * 100), 2) if total > 0 else 0
                yearly_stats.append({
                    'Year': year,
                    'Win_Percentage': win_pct,
                    'Total_Matches': total
                })
            
            yearly_df = pd.DataFrame(yearly_stats)
            
            # Create line chart
            fig = go.Figure()
            
            # Add win percentage line
            fig.add_trace(go.Scatter(
                x=yearly_df['Year'],
                y=yearly_df['Win_Percentage'],
                mode='lines+markers',
                name='Win %',
                line=dict(color='#28a745', width=3),
                marker=dict(size=8)
            ))
            
            # Add total matches as bar chart
            fig.add_trace(go.Bar(
                x=yearly_df['Year'],
                y=yearly_df['Total_Matches'],
                name='Total Matches',
                yaxis='y2',
                opacity=0.3,
                marker_color='#666666'
            ))
            
            fig.update_layout(
                #title=f"{team} - Yearly Performance",
                yaxis=dict(
                    title="Win Percentage",
                    ticksuffix="%",
                    range=[0, 100]
                ),
                yaxis2=dict(
                    title="Total Matches",
                    overlaying='y',
                    side='right'
                ),
                hovermode='x unified',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)


    if 'match_df' in st.session_state and 'All' not in team_choice:

        # Similar styling as Form Guide
        opponent_form_styles = """
        <style>
        .opponent-form-container {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background: white;
            border-radius: 8px;
            margin: 10px auto;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            max-width: 100%;
            width: 100%;
            flex-direction: column;
        }
        .opponent-name {
            font-weight: bold;
            padding: 10px 0;
        }
        .outings-container {
            display: flex;
            gap: 4px;
            flex-wrap: wrap;
            justify-content: center;
        }
        .outing-indicator {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            font-weight: bold;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }
        .outing-indicator:hover .tooltip {
            visibility: visible;
            opacity: 1;
        }
        .tooltip {
            visibility: hidden;
            background-color: rgba(0, 0, 0, 0.75);
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 8px;
            position: absolute;
            z-index: 1;
            bottom: 125%; /* Position above the indicator */
            left: 50%;
            margin-left: -75px;
            opacity: 0;
            transition: opacity 0.3s;
            width: 150px;
            font-size: 12px;
            line-height: 1.4;
        }
        .tooltip::after {
            content: "";
            position: absolute;
            top: 100%; /* Arrow at the bottom */
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: rgba(0, 0, 0, 0.75) transparent transparent transparent;
        }
        .win { background-color: #28a745; }
        .draw { background-color: #ffc107; color: black; }
        .loss { background-color: #dc3545; }
        </style>
        """
        st.markdown(opponent_form_styles, unsafe_allow_html=True)

        for team in team_choice:
            st.markdown(f"<h2 style='color:#f04f53; text-align: center;'>{team} - Last 20 Outings vs Each Opponent</h2>", unsafe_allow_html=True)
            st.markdown("<h3 style='color:#f04f53; text-align: center; margin-top: -15px; font-size: 0.9em;'>Latest →</h3>", unsafe_allow_html=True)

            # Filter matches for this team
            team_data = raw_matches[
                (raw_matches['Home Team'] == team) | (raw_matches['Away Team'] == team)
            ]

            # Get opponents the team has faced
            opponents = set(team_data['Home Team'].unique()) | set(team_data['Away Team'].unique())
            opponents.discard(team)

            for opponent in sorted(opponents):
                # Get last 20 matches vs this opponent
                mask = (
                    ((team_data['Home Team'] == team) & (team_data['Away Team'] == opponent)) |
                    ((team_data['Home Team'] == opponent) & (team_data['Away Team'] == team))
                )
                last_ten = team_data[mask].sort_values('Date', ascending=False).head(20)
                
                if not last_ten.empty:
                    outings = []
                    w_count, l_count, d_count = 0, 0, 0
                    for _, mtch in last_ten.iterrows():
                        margin = mtch['Margin'].lower()
                        date_val = pd.to_datetime(mtch['Date'], dayfirst=True, errors='coerce')
                        if pd.notnull(date_val):
                            date_str = date_val.strftime('%Y-%m-%d')
                        else:
                            date_str = ''
                        format = mtch['Format']
                        tooltip = f"<b>Date:</b> {date_str}<br><b>Margin:</b> {margin}<br><b>Format:</b> {format}"
                        if margin.startswith(team.lower()):
                            outings.append(f"<div class='outing-indicator win'><span class='tooltip'>{tooltip}</span>W</div>")
                            w_count += 1
                        elif 'drawn' in margin:
                            outings.append(f"<div class='outing-indicator draw'><span class='tooltip'>{tooltip}</span>D</div>")
                            d_count += 1
                        else:
                            outings.append(f"<div class='outing-indicator loss'><span class='tooltip'>{tooltip}</span>L</div>")
                            l_count += 1
                    
                    html_block = f"""
                    <div class="opponent-form-container">
                        <div class="opponent-name">
                            {team} - {(fm if 'fm' in globals() else 'All')} | Opponent: {opponent} (W {w_count}, L {l_count}, D {d_count})
                        </div>
                        <div class="outings-container">
                            {''.join(reversed(outings))}
                        </div>
                    </div>
                    """
                    st.markdown(html_block, unsafe_allow_html=True)

                # Add per-format rows
                for fm in format_choice:
                    if fm != 'All':
                        sub_data = team_data[
                            ((team_data['Home Team'] == team) & (team_data['Away Team'] == opponent)) |
                            ((team_data['Home Team'] == opponent) & (team_data['Away Team'] == team)) &
                            (team_data['Format'] == fm)
                        ].sort_values('Date', ascending=False).head(20)
                        
                        if not sub_data.empty:
                            outings = []
                            w_count, l_count, d_count = 0, 0, 0
                            for _, mtch in sub_data.iterrows():
                                margin = mtch['Margin'].lower()
                                date_val = pd.to_datetime(mtch['Date'], dayfirst=True, errors='coerce')
                                if pd.notnull(date_val):
                                    date_str = date_val.strftime('%Y-%m-%d')
                                else:
                                    date_str = ''
                                format = mtch['Format']
                                tooltip = f"<b>Date:</b> {date_str}<br><b>Margin:</b> {margin}<br><b>Format:</b> {format}"
                                if margin.startswith(team.lower()):
                                    outings.append(f"<div class='outing-indicator win'><span class='tooltip'>{tooltip}</span>W</div>")
                                    w_count += 1
                                elif 'drawn' in margin:
                                    outings.append(f"<div class='outing-indicator draw'><span class='tooltip'>{tooltip}</span>D</div>")
                                    d_count += 1
                                else:
                                    outings.append(f"<div class='outing-indicator loss'><span class='tooltip'>{tooltip}</span>L</div>")
                                    l_count += 1
                            
                            html_block = f"""
                            <div class="opponent-form-container">
                                <div class="opponent-name">Opponent: {opponent} ({fm}) (W {w_count}, L {l_count}, D {d_count})</div>
                                <div class="outings-container">{''.join(reversed(outings))}</div>
                            </div>
                            """
                            st.markdown(html_block, unsafe_allow_html=True)

    # ...existing code...
    if 'match_df' in st.session_state and 'All' not in team_choice:
        # ...existing code...
        for team in team_choice:
            # ...existing code for "All" row...
            # last_ten = ... # existing code

            # Loop over formats
            for fm in format_choice:
                if fm != 'All':
                    sub_data = team_data[
                        (team_data['Format'] == fm)
                    ].sort_values('Date', ascending=False).head(20)
                    # ...build & display for sub_data...

    if 'match_df' in st.session_state and 'All' not in team_choice:
    # Add Form Guide Record summary for each selected team
        for team in team_choice:
            team_matches = raw_matches[(raw_matches['Home Team'] == team) | (raw_matches['Away Team'] == team)]
            total = len(team_matches)
            wins = team_matches['Margin'].apply(lambda m: 1 if str(m).startswith(team) else 0).sum()
            draws = team_matches['Margin'].apply(lambda m: 1 if 'drawn' in str(m).lower() else 0).sum()
            losses = total - wins - draws
            st.markdown(f"<h3 style='text-align:center;'>{team} Record: P{total}, W{wins}, L{losses}, D{draws}</h3>", unsafe_allow_html=True)

            # Add a row per format record when "All" is selected
            if 'All' in format_choice:
                for fm in [f for f in formats if f != "All"]:
                    team_format_matches = raw_matches[
                        ((raw_matches['Home Team'] == team) | (raw_matches['Away Team'] == team)) &
                        (raw_matches['Format'] == fm)
                    ]
                    total = len(team_format_matches)
                    wins = team_format_matches['Margin'].apply(lambda m: 1 if str(m).startswith(team) else 0).sum()
                    draws = team_format_matches['Margin'].apply(lambda m: 1 if 'drawn' in str(m).lower() else 0).sum()
                    losses = total - wins - draws
                    st.markdown(
                        f"<p style='text-align:center;'>{team} ({fm}): Pld {total}, W {wins}, L {losses}, D {draws}</p>",
                        unsafe_allow_html=True
                    )

################### SERIES TAB #################################
with tabs[1]:
    # Add validation for team selection
    if ('All' in team_choice and len(team_choice) > 1) or (len(team_choice) > 1 and 'All' not in team_choice):
        st.error("Please select either 'All' or one team to view the series data.")
    else:
        # Always show Series Data section
        if 'match_df' in st.session_state:
            st.markdown("<h1 style='color:#f04f53; text-align: center;'>Series Data</h1>", unsafe_allow_html=True)

            match_df = st.session_state['match_df']
            series_df = match_df.copy()
            series_df['Date'] = pd.to_datetime(series_df['Date'], dayfirst=True).dt.date

            # Apply format filter
            if 'All' not in format_choice:
                series_df = series_df[series_df['Match_Format'].isin(format_choice)]

            # Apply team and opponent filters
            if 'All' not in team_choice:
                series_df = series_df[
                    (series_df['Home_Team'].isin(team_choice)) | 
                    (series_df['Away_Team'].isin(team_choice))
                ]

            if 'All' not in opponent_choice:
                series_df = series_df[
                    (series_df['Home_Team'].isin(opponent_choice)) | 
                    (series_df['Away_Team'].isin(opponent_choice))
                ]

            def is_part_of_series(competition: str) -> bool:
                """Check if competition is a numbered series match."""
                parts = competition.lower().split()
                return parts[0] in ['1st', '2nd', '3rd', '4th', '5th']

            def get_series_number(competition: str) -> int:
                """Extract the numeric indicator from competition name."""
                number_map = {'1st': 1, '2nd': 2, '3rd': 3, '4th': 4, '5th': 5}
                parts = competition.lower().split()
                return number_map.get(parts[0], 1)

            # Filter out competitions we regard for series
            potential_series = [
                '1st Test Match','2nd Test Match','3rd Test Match','4th Test Match','5th Test Match',
                '1st One Day International','2nd One Day International','3rd One Day International',
                '4th One Day International','5th One Day International',
                '1st 20 Over International','2nd 20 Over International','3rd 20 Over International',
                '4th 20 Over International','5th 20 Over International',
                'Only One Day International','Only 20 Over International','Only Test Match',
                'Test Championship Final'
            ]
            series_df = series_df[series_df['Competition'].isin(potential_series)].copy()
            series_df = series_df.sort_values('Date', ascending=True)

            series_list = []
            for _, match in series_df.iterrows():
                comp = match['Competition']
                match_date = pd.to_datetime(match['Date'], dayfirst=True)
                if is_part_of_series(comp):
                    # Group all matches in this series within 60 days, same teams & format
                    subset = series_df[
                        (series_df['Home_Team'] == match['Home_Team']) &
                        (series_df['Away_Team'] == match['Away_Team']) &
                        (series_df['Match_Format'] == match['Match_Format']) &
                        (abs(pd.to_datetime(series_df['Date'], dayfirst=True) - match_date).dt.days <= 60)
                    ]
                    if not subset.empty:
                        info = {
                            'Start_Date': min(pd.to_datetime(subset['Date'], dayfirst=True)).date(),
                            'End_Date': max(pd.to_datetime(subset['Date'], dayfirst=True)).date(),
                            'Home_Team': match['Home_Team'],
                            'Away_Team': match['Away_Team'],
                            'Match_Format': match['Match_Format'],
                            'Games_Played': len(subset),
                            'Total_Home_Wins': subset['Home_Win'].sum(),
                            'Total_Away_Wins': subset['Home_Lost'].sum(),
                            'Total_Draws': subset['Home_Drawn'].sum(),
                            'Show_Matches': False  # Add default value for checkbox
                        }
                        key = f"{info['Home_Team']}_{info['Away_Team']}_{info['Match_Format']}_{info['Start_Date']}"
                        if not any(x.get('key') == key for x in series_list):
                            info['key'] = key
                            series_list.append(info)
                elif comp in [
                    'Only One Day International','Only 20 Over International','Only Test Match','Test Championship Final'
                ]:
                    info = {
                        'Start_Date': match_date.date(),
                        'End_Date': match_date.date(),
                        'Home_Team': match['Home_Team'],
                        'Away_Team': match['Away_Team'],
                        'Match_Format': match['Match_Format'],
                        'Games_Played': 1,
                        'Total_Home_Wins': match['Home_Win'],
                        'Total_Away_Wins': match['Home_Lost'],
                        'Total_Draws': match['Home_Drawn'],
                        'Show_Matches': False  # Add default value for checkbox
                    }
                    info['key'] = f"{info['Home_Team']}_{info['Away_Team']}_{info['Match_Format']}_{info['Start_Date']}"
                    series_list.append(info)

            series_grouped = pd.DataFrame(series_list)
            if series_list:
                series_grouped = pd.DataFrame(series_list).drop_duplicates('key').drop('key', axis=1)
                series_grouped = series_grouped.sort_values('Start_Date')
                series_grouped['Series'] = range(1, len(series_grouped) + 1)

                def determine_series_winner(row):
                    if row['Total_Home_Wins'] > row['Total_Away_Wins']:
                        return f"{row['Home_Team']} won {row['Total_Home_Wins']}-{row['Total_Away_Wins']}"
                    elif row['Total_Away_Wins'] > row['Total_Home_Wins']:
                        return f"{row['Away_Team']} won {row['Total_Away_Wins']}-{row['Total_Home_Wins']}"
                    else:
                        if row['Total_Draws'] > 0:
                            return f"Series Drawn {row['Total_Home_Wins']}-{row['Total_Away_Wins']}"
                        return f"Series Tied {row['Total_Home_Wins']}-{row['Total_Away_Wins']}"

                series_grouped['Series_Result'] = series_grouped.apply(determine_series_winner, axis=1)

                # Configure columns including checkbox
                edited_df = st.data_editor(
                    series_grouped,
                    column_config={
                        "Show_Matches": st.column_config.CheckboxColumn(
                            "Show Matches",
                            help="Show matches for this series",
                            default=False,
                        )
                    },
                    use_container_width=True,
                    hide_index=True
                )

                # Get series with checked boxes
                checked_series = edited_df[edited_df['Show_Matches']]

                if not checked_series.empty:
                    st.markdown("<h1 style='color:#f04f53; text-align: center;'>Selected Series Matches</h1>", unsafe_allow_html=True)
                    
                    # Create empty list to store matches from all selected series
                    all_selected_matches = []

                    # Filter matches for each selected series
                    for _, row in checked_series.iterrows():
                        series_matches = series_df[
                            (series_df['Home_Team'] == row['Home_Team']) &
                            (series_df['Away_Team'] == row['Away_Team']) &
                            (series_df['Match_Format'] == row['Match_Format']) &
                            (pd.to_datetime(series_df['Date'], dayfirst=True).dt.date >= row['Start_Date']) &
                            (pd.to_datetime(series_df['Date'], dayfirst=True).dt.date <= row['End_Date'])
                        ][['Date', 'Competition', 'Match_Format', 'Home_Team', 'Away_Team', 'Player_of_the_Match', 'Margin']]
                        all_selected_matches.append(series_matches)

                    # Combine all selected series matches
                    if all_selected_matches:
                        combined_matches = pd.concat(all_selected_matches)
                        combined_matches = combined_matches.sort_values('Date')
                        st.dataframe(combined_matches, use_container_width=True, hide_index=True)



        # Always show Head to Head Series Records
        st.markdown("<h1 style='color:#f04f53; text-align: center;'>Head to Head Series Records</h1>", unsafe_allow_html=True)

        # Create a dataframe for series records
        if not series_grouped.empty:
            # Prepare home team records 
            home_stats = series_grouped[['Home_Team', 'Away_Team', 'Match_Format', 'Series_Result']].copy()
            # Extract wins based on Series_Result
            home_stats['Won'] = home_stats.apply(lambda x: 1 if x['Series_Result'].startswith(x['Home_Team']) else 0, axis=1)
            home_stats['Lost'] = home_stats.apply(lambda x: 1 if x['Series_Result'].startswith(x['Away_Team']) else 0, axis=1)
            home_stats['Draw'] = home_stats.apply(lambda x: 1 if 'Drawn' in x['Series_Result'] or 'Tied' in x['Series_Result'] else 0, axis=1)
            home_stats = home_stats[['Home_Team', 'Away_Team', 'Match_Format', 'Won', 'Lost', 'Draw']]
            home_stats.columns = ['Team', 'Opponent', 'Match_Format', 'Won', 'Lost', 'Draw']

            # Prepare away team records
            away_stats = series_grouped[['Away_Team', 'Home_Team', 'Match_Format', 'Series_Result']].copy()
            # Extract wins based on Series_Result
            away_stats['Won'] = away_stats.apply(lambda x: 1 if x['Series_Result'].startswith(x['Away_Team']) else 0, axis=1)
            away_stats['Lost'] = away_stats.apply(lambda x: 1 if x['Series_Result'].startswith(x['Home_Team']) else 0, axis=1)
            away_stats['Draw'] = away_stats.apply(lambda x: 1 if 'Drawn' in x['Series_Result'] or 'Tied' in x['Series_Result'] else 0, axis=1)
            away_stats = away_stats[['Away_Team', 'Home_Team', 'Match_Format', 'Won', 'Lost', 'Draw']]
            away_stats.columns = ['Team', 'Opponent', 'Match_Format', 'Won', 'Lost', 'Draw']

            # Combine both perspectives
            all_series = pd.concat([home_stats, away_stats])

            # Group by Team, Opponent and Match_Format
            h2h_series = all_series.groupby(['Team', 'Opponent', 'Match_Format']).agg({
                'Won': 'sum',
                'Lost': 'sum', 
                'Draw': 'sum'
            }).reset_index()

            # Add total series column
            h2h_series['Series'] = h2h_series['Won'] + h2h_series['Lost'] + h2h_series['Draw']

            # Sort by number of series and wins
            h2h_series = h2h_series.sort_values(['Series', 'Won'], ascending=[False, False])

            # Display the table
            st.dataframe(h2h_series, use_container_width=True, hide_index=True)
        else:
            st.info("No series data available for head-to-head analysis.")

        # Add metrics if a team is selected
        if 'All' not in team_choice and team_choice and not series_grouped.empty:
            total_series = len(series_grouped)
            total_wins = series_grouped['Series_Result'].apply(lambda x: 1 if x.startswith(team) else 0).sum()
            total_losses = series_grouped['Series_Result'].apply(lambda x: 1 if not x.startswith(team) and 'Drawn' not in x and 'Tied' not in x else 0).sum()
            total_draws = series_grouped['Series_Result'].apply(lambda x: 1 if 'Drawn' in x or 'Tied' in x else 0).sum()
            win_percentage = (total_wins / total_series) * 100 if total_series > 0 else 0
            loss_percentage = (total_losses / total_series) * 100 if total_series > 0 else 0

            st.markdown("<h1 style='color:#f04f53; text-align: center;'>Team Series Record</h1>", unsafe_allow_html=True)

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Series", total_series, border=True)
            with col2:
                st.metric("Won", total_wins, border=True)
            with col3:
                st.metric("Lost", total_losses, border=True)
            with col4:
                st.metric("Win %", f"{win_percentage:.2f}%", border=True)
            with col5:
                st.metric("Lost %", f"{loss_percentage:.2f}%", border=True)


                

    # Only show Form Guide and Performance Trend if specific team(s) selected
    if 'match_df' in st.session_state and team_choice and 'All' not in team_choice and not series_grouped.empty:
        # Performance Trend

        for team in team_choice:
            # Series Form Guide
            st.markdown("<h1 style='color:#f04f53; text-align: center;'>Series Form Guide</h1>", unsafe_allow_html=True)
            st.markdown("<h3 style='color:#f04f53; text-align: center; margin-top: -15px; font-size: 0.9em;'>Latest →</h3>", unsafe_allow_html=True)
            
            # All formats row first
            team_series = series_grouped[
                (series_grouped['Home_Team'] == team) | 
                (series_grouped['Away_Team'] == team)
            ].sort_values('Start_Date', ascending=False).head(20)

            # Calculate team record for all formats
            total_series = len(team_series)
            win_count = team_series['Series_Result'].apply(lambda x: 1 if x.startswith(team) else 0).sum()
            draw_count = team_series['Series_Result'].apply(lambda x: 1 if ('Drawn' in x or 'Tied' in x) else 0).sum()
            loss_count = total_series - win_count - draw_count
            record_str = f"P{total_series}, W{win_count}, L{loss_count}, D{draw_count}"
            
            if not team_series.empty:
                form_indicators = []
                for _, series in team_series.iterrows():
                    is_home = series['Home_Team'] == team
                    opponent = series['Away_Team'] if is_home else series['Home_Team']
                    result = series['Series_Result']
                    
                    tooltip = (f"<b>Dates:</b> {series['Start_Date']} to {series['End_Date']}<br>"
                            f"<b>vs {opponent}</b><br>"
                            f"<b>Format:</b> {series['Match_Format']}<br>"
                            f"<b>Result:</b> {result}")
                    
                    if result.startswith(team):
                        form_indicators.append(f'<div class="form-indicator win"><span class="tooltip">{tooltip}</span>W</div>')
                    elif 'Drawn' in result or 'Tied' in result:
                        form_indicators.append(f'<div class="form-indicator draw"><span class="tooltip">{tooltip}</span>D</div>')
                    else:
                        form_indicators.append(f'<div class="form-indicator loss"><span class="tooltip">{tooltip}</span>L</div>')

                if form_indicators:
                    form_html = f"""
                    <div class="form-container">
                        <span class="team-name">{team} {record_str}</span>
                        <div class="form-indicators-container">
                            {''.join(reversed(form_indicators))}
                        </div>
                    </div>
                    """
                    st.markdown(form_html, unsafe_allow_html=True)

            # Add rows for each format
            for fmt in format_choice if 'All' not in format_choice else series_grouped['Match_Format'].unique():
                format_series = series_grouped[
                    ((series_grouped['Home_Team'] == team) | 
                    (series_grouped['Away_Team'] == team)) &
                    (series_grouped['Match_Format'] == fmt)
                ].sort_values('Start_Date', ascending=False).head(20)

                # Calculate team record for this format
                total_series_fmt = len(format_series)
                win_count_fmt = format_series['Series_Result'].apply(lambda x: 1 if x.startswith(team) else 0).sum()
                draw_count_fmt = format_series['Series_Result'].apply(lambda x: 1 if ('Drawn' in x or 'Tied' in x) else 0).sum()
                loss_count_fmt = total_series_fmt - win_count_fmt - draw_count_fmt
                record_str_fmt = f"P{total_series_fmt}, W{win_count_fmt}, L{loss_count_fmt}, D{draw_count_fmt}"
                
                if not format_series.empty:
                    form_indicators = []
                    for _, series in format_series.iterrows():
                        is_home = series['Home_Team'] == team
                        opponent = series['Away_Team'] if is_home else series['Home_Team']
                        result = series['Series_Result']
                        
                        tooltip = (f"<b>Dates:</b> {series['Start_Date']} to {series['End_Date']}<br>"
                                f"<b>vs {opponent}</b><br>"
                                f"<b>Format:</b> {series['Match_Format']}<br>"
                                f"<b>Result:</b> {result}")
                        
                        if result.startswith(team):
                            form_indicators.append(f'<div class="form-indicator win"><span class="tooltip">{tooltip}</span>W</div>')
                        elif 'Drawn' in result or 'Tied' in result:
                            form_indicators.append(f'<div class="form-indicator draw"><span class="tooltip">{tooltip}</span>D</div>')
                        else:
                            form_indicators.append(f'<div class="form-indicator loss"><span class="tooltip">{tooltip}</span>L</div>')

                    if form_indicators:
                        form_html = f"""
                        <div class="form-container">
                            <span class="team-name">{team} ({fmt}) {record_str_fmt}</span>
                            <div class="form-indicators-container">
                                {''.join(reversed(form_indicators))}
                            </div>
                        </div>
                        """
                        st.markdown(form_html, unsafe_allow_html=True)

            st.markdown(f"<h1 style='color:#f04f53; text-align: center;'>{team} - Series Performance Trend</h1>", unsafe_allow_html=True)


            team_series_trend = series_grouped[
                (series_grouped['Home_Team'] == team) | 
                (series_grouped['Away_Team'] == team)
            ].sort_values('Start_Date').tail(50)
            
            if not team_series_trend.empty:
                results = []
                colors = []
                for _, series in team_series_trend.iterrows():
                    if series['Series_Result'].startswith(team):
                        results.append(1)    # Win
                        colors.append('#28a745')
                    elif 'Drawn' in series['Series_Result'] or 'Tied' in series['Series_Result']:
                        results.append(0.5)  # Draw
                        colors.append('#ffc107')
                    else:
                        results.append(0)    # Loss
                        colors.append('#dc3545')
                
                fig = go.Figure()
                
                # Add the main line
                fig.add_trace(go.Scatter(
                    y=results,
                    mode='lines',
                    name='Performance',
                    line=dict(shape='spline', smoothing=0.8, width=2, color='#666666')
                ))
                
                # Add colored markers
                fig.add_trace(go.Scatter(
                    y=results,
                    mode='markers',
                    marker=dict(size=10, color=colors, line=dict(width=2, color='white')),
                    showlegend=False
                ))
                
                fig.update_layout(
                    yaxis=dict(
                        ticktext=["Loss", "Draw", "Win"],
                        tickvals=[0, 0.5, 1],
                        range=[-0.1, 1.1],
                        gridcolor='lightgray'
                    ),
                    xaxis=dict(
                        title="Last 50 Series (Recent ← Old)",
                        gridcolor='lightgray'
                    ),
                    plot_bgcolor='white',
                    showlegend=False,
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)

        # Add a win percentage by year chart for series data
        if 'match_df' in st.session_state and 'All' not in team_choice:
            st.markdown("<h1 style='color:#f04f53; text-align: center;'>Win Percentage by Year (Series)</h1>", unsafe_allow_html=True)
            
            for team in team_choice:
                # Get series for the team and ensure Date is datetime 
                team_series = series_grouped[
                    (series_grouped['Home_Team'] == team) | 
                    (series_grouped['Away_Team'] == team)
                ].copy()
                
                # Extract year using string operations since Date is already in datetime format
                team_series['Year'] = pd.to_datetime(team_series['Start_Date'], dayfirst=True).apply(lambda x: x.year)
                
                # Create line chart
                fig = go.Figure()
                
                # Calculate and plot win percentage for each format
                formats = format_choice if 'All' not in format_choice else team_series['Match_Format'].unique()
                format_colors = {
                    'Test Match': '#28a745',
                    'One Day International': '#dc3545', 
                    '20 Over International': '#ffc107'
                }
                
                for fmt in formats:
                    # Filter for this format
                    format_series = team_series[team_series['Match_Format'] == fmt]
                    
                    # Calculate win percentage by year for this format
                    yearly_stats = []
                    for year in sorted(format_series['Year'].unique()):
                        year_series = format_series[format_series['Year'] == year]
                        wins = sum(
                            (year_series['Home_Team'] == team) & (year_series['Series_Result'].str.startswith(team)) |
                            (year_series['Away_Team'] == team) & (year_series['Series_Result'].str.startswith(team))
                        )
                        total = len(year_series)
                        win_pct = round((wins / total * 100), 2) if total > 0 else 0
                        yearly_stats.append({
                            'Year': year,
                            'Win_Percentage': win_pct,
                            'Total_Series': total
                        })
                    
                    yearly_df = pd.DataFrame(yearly_stats)
                    
                    if not yearly_df.empty:
                        # Add win percentage line for this format
                        fig.add_trace(go.Scatter(
                            x=yearly_df['Year'],
                            y=yearly_df['Win_Percentage'],
                            mode='lines+markers',
                            name=f'{fmt} Win %',
                            line=dict(color=format_colors.get(fmt, '#666666'), width=3),
                            marker=dict(size=8)
                        ))
            
            fig.update_layout(
                xaxis=dict(
                title="Year",
                tickmode='linear',
                tick0=team_series['Year'].min(),
                dtick=1,
                gridcolor='lightgray'
                ),
                yaxis=dict(
                title="Win Percentage",
                ticksuffix="%",
                range=[0, 100]
                ),
                hovermode='x unified',
                showlegend=True,
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)

        # South Africa - Last 20 Outings vs Each Opponent (Series)
        if 'match_df' in st.session_state and 'South Africa' in team_choice:
            st.markdown(f"<h2 style='color:#f04f53; text-align: center;'>{team} - Last 20 Outings vs Each Opponent (Series)</h2>", unsafe_allow_html=True)

            st.markdown("<h3 style='color:#f04f53; text-align: center; margin-top: -15px; font-size: 0.9em;'>Latest →</h3>", unsafe_allow_html=True)

            # Similar styling as Form Guide
            opponent_form_styles = """
            <style>
            .opponent-form-container {
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
                background: white;
                border-radius: 8px;
                margin: 10px auto;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                max-width: 100%;
                width: 100%;
                flex-direction: column;
            }
            .opponent-name {
                font-weight: bold;
                padding: 10px 0;
            }
            .outings-container {
                display: flex;
                gap: 4px;
                flex-wrap: wrap;
                justify-content: center;
            }
            .outing-indicator {
                width: 30px;
                height: 30px;
                border-radius: 50%;
                font-weight: bold;
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                position: relative;
            }
            .outing-indicator:hover .tooltip {
                visibility: visible;
                opacity: 1;
            }
            .tooltip {
                visibility: hidden;
                background-color: rgba(0, 0, 0, 0.75);
                color: #fff;
                text-align: center;
                border-radius: 6px;
                padding: 8px;
                position: absolute;
                z-index: 1;
                bottom: 125%; /* Position above the indicator */
                left: 50%;
                margin-left: -75px;
                opacity: 0;
                transition: opacity 0.3s;
                width: 150px;
                font-size: 12px;
                line-height: 1.4;
            }
            .tooltip::after {
                content: "";
                position: absolute;
                top: 100%; /* Arrow at the bottom */
                left: 50%;
                margin-left: -5px;
                border-width: 5px;
                border-style: solid;
                border-color: rgba(0, 0, 0, 0.75) transparent transparent transparent;
            }
            .win { background-color: #28a745; }
            .draw { background-color: #ffc107; color: black; }
            .loss { background-color: #dc3545; }
            </style>
            """
            st.markdown(opponent_form_styles, unsafe_allow_html=True)

            # Filter series for South Africa
            team_data = series_grouped[
                (series_grouped['Home_Team'] == 'South Africa') | 
                (series_grouped['Away_Team'] == 'South Africa')
            ]

            # Get opponents South Africa has faced
            opponents = set(team_data['Home_Team'].unique()) | set(team_data['Away_Team'].unique())
            opponents.discard('South Africa')

            for opponent in sorted(opponents):
                # Get last 20 series vs this opponent
                mask = (
                    ((team_data['Home_Team'] == 'South Africa') & (team_data['Away_Team'] == opponent)) |
                    ((team_data['Home_Team'] == opponent) & (team_data['Away_Team'] == 'South Africa'))
                )
                last_twenty = team_data[mask].sort_values('Start_Date', ascending=False).head(20)
                
                if not last_twenty.empty:
                    outings = []
                    w_count, l_count, d_count = 0, 0, 0
                    for _, series in last_twenty.iterrows():
                        result = series['Series_Result'].lower()
                        start_date = series['Start_Date'].strftime('%Y-%m-%d')
                        end_date = series['End_Date'].strftime('%Y-%m-%d')
                        format = series['Match_Format']
                        tooltip = f"<b>Dates:</b> {start_date} to {end_date}<br><b>Result:</b> {series['Series_Result']}<br><b>Format:</b> {format}"
                        if result.startswith('south africa'):
                            outings.append(f"<div class='outing-indicator win'><span class='tooltip'>{tooltip}</span>W</div>")
                            w_count += 1
                        elif 'drawn' in result or 'tied' in result:
                            outings.append(f"<div class='outing-indicator draw'><span class='tooltip'>{tooltip}</span>D</div>")
                            d_count += 1
                        else:
                            outings.append(f"<div class='outing-indicator loss'><span class='tooltip'>{tooltip}</span>L</div>")
                            l_count += 1
                    
                    html_block = f"""
                    <div class="opponent-form-container">
                        <div class="opponent-name">Opponent: {opponent} (W {w_count}, L {l_count}, D {d_count})</div>
                        <div class="outings-container">{''.join(reversed(outings))}</div>
                    </div>
                    """
                    st.markdown(html_block, unsafe_allow_html=True)

                # Add per-format rows
                for fm in format_choice:
                    if fm != 'All':
                        sub_data = team_data[
                            (team_data['Match_Format'] == fm) & mask
                        ].sort_values('Start_Date', ascending=False).head(20)
                        
                        if not sub_data.empty:
                            outings = []
                            w_count, l_count, d_count = 0, 0, 0
                            for _, series in sub_data.iterrows():
                                result = series['Series_Result'].lower()
                                start_date = series['Start_Date'].strftime('%Y-%m-%d')
                                end_date = series['End_Date'].strftime('%Y-%m-%d')
                                format = series['Match_Format']
                                tooltip = f"<b>Dates:</b> {start_date} to {end_date}<br><b>Result:</b> {series['Series_Result']}<br><b>Format:</b> {format}"
                                if result.startswith('south africa'):
                                    outings.append(f"<div class='outing-indicator win'><span class='tooltip'>{tooltip}</span>W</div>")
                                    w_count += 1
                                elif 'drawn' in result or 'tied' in result:
                                    outings.append(f"<div class='outing-indicator draw'><span class='tooltip'>{tooltip}</span>D</div>")
                                    d_count += 1
                                else:
                                    outings.append(f"<div class='outing-indicator loss'><span class='tooltip'>{tooltip}</span>L</div>")
                                    l_count += 1
                            
                            html_block = f"""
                            <div class="opponent-form-container">
                                <div class="opponent-name">Opponent: {opponent} ({fm}) (W {w_count}, L {l_count}, D {d_count})</div>
                                <div class="outings-container">{''.join(reversed(outings))}</div>
                            </div>
                            """
                            st.markdown(html_block, unsafe_allow_html=True)

# New Tournaments Tab code
with tabs[2]:
    st.markdown("<h1 style='color:#f04f53; text-align: center;'>Tournaments</h1>", unsafe_allow_html=True)
    if 'match_df' in st.session_state:
        # Apply page filters to match_df for Tournaments tab
        tournaments_df = st.session_state['match_df'].copy()
        if 'All' not in format_choice:
            tournaments_df = tournaments_df[tournaments_df['Match_Format'].isin(format_choice)]
        if 'All' not in team_choice:
            tournaments_df = tournaments_df[
                (tournaments_df['Home_Team'].isin(team_choice)) | 
                (tournaments_df['Away_Team'].isin(team_choice))
            ]
        if 'All' not in opponent_choice:
            tournaments_df = tournaments_df[
                (tournaments_df['Home_Team'].isin(opponent_choice)) | 
                (tournaments_df['Away_Team'].isin(opponent_choice))
            ]
    
        #st.markdown("<h2 style='color:#f04f53; text-align: center;'>Raw Match Data</h2>", unsafe_allow_html=True)
        #st.dataframe(tournaments_df, use_container_width=True)
        
        finals_competitions = [
            "Test Championship Final",
            "World Cup 20 - Final",
            "World Cup - Final",
            "Champions Cup - Final"
        ]
        finals = tournaments_df[tournaments_df['Competition'].isin(finals_competitions)].copy()
        if not finals.empty:
            finals['Date'] = pd.to_datetime(finals['Date'], dayfirst=True).apply(lambda dt: dt.strftime("%d/%m/%Y"))
            #st.markdown("<h2 style='color:#f04f53; text-align: center;'>All Finals</h2>", unsafe_allow_html=True)
            #st.dataframe(finals, use_container_width=True, hide_index=True)
        else:
            st.info("No finals data available.")
        
        # Add World Cup Progress tracker
        #st.markdown("<h2 style='color:#f04f53; text-align: center;'>ODI World Cup Matches</h2>", unsafe_allow_html=True)
    
    #########ODI WORLD CUP PROGRESS ####################

    # Create a new tab for ODI World Cup Progress
    odiwc_progress_df = st.session_state['match_df'].copy()

    # Filter out non-ODI World Cup matches
    odiwc_progress_df = odiwc_progress_df[odiwc_progress_df['comp'].str.contains('ODI World Cup')]

    # Extract the year from the date
    odiwc_progress_df['Year'] = pd.to_datetime(odiwc_progress_df['Date'], dayfirst=True).dt.year

    #st.dataframe(odiwc_progress_df, use_container_width=True)

    # unique year and world cup 
    years = odiwc_progress_df['Year'].unique()
    #st.dataframe(years, use_container_width=True)

    teams = list(set(odiwc_progress_df['Home_Team'].unique()) | set(odiwc_progress_df['Away_Team'].unique()))    
    teams = list(set(odiwc_progress_df['Home_Team'].unique()) | set(odiwc_progress_df['Away_Team'].unique()))
    teams.sort()
    # put the unique teams list into a DataFrame with one column named 'A'
    teams_df = pd.DataFrame(teams, columns=['A'])
    #st.dataframe(teams_df, use_container_width=True, hide_index=True)

    unique_years = sorted(odiwc_progress_df['Year'].unique())
    year_columns = st.columns(len(unique_years))
    #for col, year in zip(year_columns, unique_years):
        #col.markdown(f"<h3 style='text-align: center;'>{year}</h3>")

    #st.markdown("<h2 style='color:#f04f53; text-align: center;'>ODI World Cup Matches Matrix</h2>", unsafe_allow_html=True)
    wc_teams = pd.concat([
        odiwc_progress_df[['Year','Home_Team']].rename(columns={'Home_Team': 'Team'}),
        odiwc_progress_df[['Year','Away_Team']].rename(columns={'Away_Team': 'Team'})
    ], ignore_index=True)

    wc_matrix = wc_teams.groupby(['Team','Year']).size().unstack(fill_value=0)
    #st.dataframe(wc_matrix, use_container_width=True)

    def map_stage_to_code(stage: str) -> str:
        """Convert stage name to code with numeric rank (lower is better)"""
        if "Final" in stage and "Semi" not in stage:
            return ("F", 1)  # Final is best
        elif "Semi-Final" in stage:
            return ("SF", 2)  # Semi-Final is second best
        elif "Group" in stage:
            return ("GRP", 3)  # Group stage is worst
        return ("?", 4)  # Unknown stages ranked last

    # Process each team's matches to find best stage per year
    team_results = []
    for team in teams:
        team_matches = odiwc_progress_df[
            (team == odiwc_progress_df['Home_Team']) | 
            (team == odiwc_progress_df['Away_Team'])
        ]
        
        for year in unique_years:
            year_matches = team_matches[team_matches['Year'] == year]
            if not year_matches.empty:
                stages = [map_stage_to_code(comp) for comp in year_matches['Competition']]
                best_stage = min(stages, key=lambda x: x[1])
                if best_stage[0] == "F":
                    # Pick one final match row
                    final_row = year_matches.iloc[0]
                    # Check if it was a draw first
                    if 'drawn' in str(final_row["Match_Result"]).lower():
                        best_code = "W"  # Both teams get W for a draw
                    else:
                        # Check if team won or lost
                        if team in str(final_row["Match_Result"]):
                            best_code = "W"
                        else:
                            best_code = "RU"
                else:
                    best_code = best_stage[0]
                team_results.append({
                    'Team': team,
                    'Year': year,
                    'Stage': best_code
                })

    # Convert results to matrix format
    stage_matrix = pd.DataFrame(team_results).pivot(
        index='Team', 
        columns='Year', 
        values='Stage'
    ).fillna('')

    st.markdown("<h2 style='color:#f04f53; text-align: center;'>ODI World Cup Best Stage Reached</h2>", unsafe_allow_html=True)

    def highlight_stage(row):
        # row is a Series with year columns as keys, each value is a stage code.
        return [
            "background-color: gold; color: black;" if cell == "W" else
            "background-color: lightcoral; color: black;" if cell == "GRP" else 
            "background-color: silver; color: black;" if cell == "RU" else
            "background-color: #cd7f32; color: black;" if cell == "SF" else ""
            for cell in row
        ]

    # Display the styled table
    styled_stage_matrix = stage_matrix.style.apply(highlight_stage, axis=1)
    st.dataframe(styled_stage_matrix, use_container_width=True)

    # Format 'Date' back to 'dd/mm/yyyy' for display purposes
    odiwc_progress_df['Date'] = pd.to_datetime(odiwc_progress_df['Date'], dayfirst=True, errors='coerce').apply(
        lambda d: d.strftime('%d/%m/%Y') if pd.notnull(d) else ''
    )

# T20 World Cup Best Stage Reached section
# Create a copy of match_df and filter for T20 World Cup matches where comp equals "T20 World Cup"
t20wc_df = st.session_state['match_df'].copy()
t20wc_df = t20wc_df[t20wc_df['comp'] == "T20 World Cup"]
t20wc_df['Year'] = pd.to_datetime(t20wc_df['Date'], dayfirst=True, errors='coerce').dt.year

# New: Display filtered T20 World Cup dataset
#st.markdown("<h2 style='color:#f04f53; text-align: center;'>Filtered T20 World Cup Matches</h2>", unsafe_allow_html=True)
#st.dataframe(t20wc_df, use_container_width=True)

# Retrieve unique teams for this competition from both Home_Team and Away_Team
teams_t20 = pd.concat([t20wc_df['Home_Team'], t20wc_df['Away_Team']]).unique()

t20_stage_results = []
for team in teams_t20:
    for year in sorted(t20wc_df['Year'].dropna().unique()):
        team_matches = t20wc_df[
            (((t20wc_df['Home_Team'] == team) | (t20wc_df['Away_Team'] == team))
             & (t20wc_df['Year'] == year))
        ]
        stage = ""
        best = 0  # numeric precedence for stage
        # Determine the stage based on the latest round played
        for _, match in team_matches.iterrows():
            comp = match['Competition']
            if "Final" in comp and "Semi-Final" not in comp:
                current = 4
                # Check for drawn match first
                if 'drawn' in str(match['Match_Result']).lower():
                    code = "W"  # Both teams get W for a draw
                else:
                    # Check if team won or lost
                    if team in str(match['Match_Result']):
                        code = "W"
                    else:
                        code = "RU"
            elif "Semi-Final" in comp:
                current = 3
                code = "SF"
            elif "Super Eight" in comp:
                current = 2
                code = "S8"
            elif "Group Match" in comp:
                current = 1
                code = "GRP"
            else:
                current = 0
                code = ""
            if current > best:
                best = current
                stage = code
        t20_stage_results.append({'Team': team, 'Year': year, 'Stage': stage})

# Create a DataFrame and ensure required columns exist
t20_stage_df = pd.DataFrame(t20_stage_results)
if t20_stage_df.empty or 'Team' not in t20_stage_df.columns:
    t20_stage_df = pd.DataFrame(columns=['Team', 'Year', 'Stage'])

t20_stage_matrix = t20_stage_df.pivot(
    index='Team', 
    columns='Year', 
    values='Stage'
).fillna("")

def highlight_t20_stage(row):
    return [
        "background-color: gold; color: black;" if cell == "W" else
        "background-color: silver; color: black;" if cell == "RU" else
        "background-color: #cd7f32; color: black;" if cell == "SF" else
        "background-color: lightblue; color: black;" if cell == "S8" else
        "background-color: lightcoral; color: black;" if cell == "GRP" else ""
        for cell in row
    ]

st.markdown("<h2 style='color:#f04f53; text-align: center;'>T20 World Cup Progress</h2>", unsafe_allow_html=True)
styled_t20_stage_matrix = t20_stage_matrix.style.apply(highlight_t20_stage, axis=1)
st.dataframe(styled_t20_stage_matrix, use_container_width=True)

# Test Championship Progress Tracker
wtc_df = st.session_state['match_df'].copy()
wtc_df = wtc_df[wtc_df['comp'] == "Test Championship Final"]
wtc_df['Year'] = pd.to_datetime(wtc_df['Date'], dayfirst=True, errors='coerce').dt.year

# Fixed list of Test Championship teams
wtc_teams = [
    'Australia', 'Bangladesh', 'England', 'India', 'New Zealand', 
    'Pakistan', 'South Africa', 'Sri Lanka', 'West Indies'
]

# New: Display filtered Test Championship dataset
#st.markdown("<h2 style='color:#f04f53; text-align: center;'>World Test Championship Progress</h2>", unsafe_allow_html=True)
##st.dataframe(wtc_df, use_container_width=True)

wtc_stage_results = []
for team in wtc_teams:
    for year in sorted(wtc_df['Year'].dropna().unique()):
        team_matches = wtc_df[
            (((wtc_df['Home_Team'] == team) | (wtc_df['Away_Team'] == team))
             & (wtc_df['Year'] == year))
        ]
        stage = "GRP"  # Default to group stage
        # If team appears in a final, check for draw first
        if not team_matches.empty:
            match = team_matches.iloc[0]  # Get the final match
            if 'drawn' in str(match['Match_Result']).lower():
                stage = "W"  # Both teams get W for a draw
            else:
                stage = "W" if team in str(match['Match_Result']) else "RU"  # W if team in result, RU otherwise
        wtc_stage_results.append({'Team': team, 'Year': year, 'Stage': stage})

# Create matrix
wtc_stage_df = pd.DataFrame(wtc_stage_results)
wtc_stage_matrix = wtc_stage_df.pivot(
    index='Team', 
    columns='Year', 
    values='Stage'
).fillna("GRP")

def highlight_wtc_stage(row):
    return [
        "background-color: gold; color: black;" if cell == "W" else
        "background-color: silver; color: black;" if cell == "RU" else
        "background-color: lightcoral; color: black;" if cell == "GRP" else ""
        for cell in row
    ]

st.markdown("<h2 style='color:#f04f53; text-align: center;'>World Test Championship Progress</h2>", unsafe_allow_html=True)
styled_wtc_stage_matrix = wtc_stage_matrix.style.apply(highlight_wtc_stage, axis=1)
st.dataframe(styled_wtc_stage_matrix, use_container_width=True)

# ...existing code...
