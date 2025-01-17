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

###################

# Form Guide
if 'match_df' in st.session_state and 'All' not in team_choice:
    st.markdown("<h1 style='color:#f04f53; text-align: center;'>Form Guide</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#f04f53; text-align: center; margin-top: -15px; font-size: 0.9em;'>Latest →</h3>", unsafe_allow_html=True)
    
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
        # "All Format" row (unchanged logic)
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
            date = match['Date'].strftime('%Y-%m-%d')
            opponent = match['Away Team'] if is_home else match['Home Team']
            tooltip = f"{date} vs {opponent}"
            
            if 'won by' in margin:
                winning_team = margin.split(' won')[0]
                if winning_team == team:
                    form_indicators.append(f'<div class="form-indicator win" title="{tooltip}">W</div>')
                else:
                    form_indicators.append(f'<div class="form-indicator loss" title="{tooltip}">L</div>')
            else:
                form_indicators.append(f'<div class="form-indicator draw" title="{tooltip}">D</div>')

        # Display team name and form
        if form_indicators:  # Only display if there are matches
            form_html = f"""
            <div class="form-container">
                <span class="team-name">{team}</span>
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
                        date = match['Date'].strftime('%Y-%m-%d')
                        opponent = match['Away Team'] if is_home else match['Home Team']
                        tooltip = f"{date} vs {opponent}"
                        
                        if 'won by' in margin:
                            winning_team = margin.split(' won')[0]
                            if winning_team == team:
                                form_indicators.append(f'<div class="form-indicator win" title="{tooltip}">W</div>')
                            else:
                                form_indicators.append(f'<div class="form-indicator loss" title="{tooltip}">L</div>')
                        else:
                            form_indicators.append(f'<div class="form-indicator draw" title="{tooltip}">D</div>')

                    # Display team name and form for the specific format
                    if form_indicators:  # Only display if there are matches
                        form_html = f"""
                        <div class="form-container">
                            <span class="team-name">{team} ({fm})</span>
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
                    f"{win_streak_start_date.strftime('%Y-%m-%d') if win_streak_start_date else 'N/A'} to {win_streak_end_date.strftime('%Y-%m-%d') if win_streak_end_date else 'N/A'}",
                    current_unbeaten_streak,
                    f"{unbeaten_streak_start_date.strftime('%Y-%m-%d') if unbeaten_streak_start_date else 'N/A'} to {unbeaten_streak_end_date.strftime('%Y-%m-%d') if unbeaten_streak_end_date else 'N/A'}",
                    current_loss_streak,
                    f"{loss_streak_start_date.strftime('%Y-%m-%d') if loss_streak_start_date else 'N/A'} to {loss_streak_end_date.strftime('%Y-%m-%d') if loss_streak_end_date else 'N/A'}",
                    current_winless_streak,
                    f"{winless_streak_start_date.strftime('%Y-%m-%d') if winless_streak_start_date else 'N/A'} to {winless_streak_end_date.strftime('%Y-%m-%d') if winless_streak_end_date else 'N/A'}",
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
                    f"{longest_win_streak_start_date.strftime('%Y-%m-%d') if longest_win_streak_start_date else 'N/A'} to {longest_win_streak_end_date.strftime('%Y-%m-%d') if longest_win_streak_end_date else 'N/A'}",
                    longest_unbeaten_streak,
                    f"{longest_unbeaten_streak_start_date.strftime('%Y-%m-%d') if longest_unbeaten_streak_start_date else 'N/A'} to {longest_unbeaten_streak_end_date.strftime('%Y-%m-%d') if longest_unbeaten_streak_end_date else 'N/A'}",
                    longest_loss_streak,
                    f"{longest_loss_streak_start_date.strftime('%Y-%m-%d') if longest_loss_streak_start_date else 'N/A'} to {longest_loss_streak_end_date.strftime('%Y-%m-%d') if longest_loss_streak_end_date else 'N/A'}",
                    longest_winless_streak,
                    f"{longest_winless_streak_start_date.strftime('%Y-%m-%d') if longest_winless_streak_start_date else 'N/A'} to {longest_winless_streak_end_date.strftime('%Y-%m-%d') if longest_winless_streak_end_date else 'N/A'}",
                ],
            }
            longest_streaks_df = pd.DataFrame(longest_streaks_data)
            st.dataframe(longest_streaks_df, hide_index=True, use_container_width=True)

################## updated graphics ############

# Add a head-to-head comparison heatmap
if 'match_df' in st.session_state and len(team_choice) >= 2:
    st.markdown("<h1 style='color:#f04f53; text-align: center;'>Head-to-Head Comparison Heatmap</h1>", unsafe_allow_html=True)
    
    # Create a matrix of win percentages between teams
    teams = sorted(list(all_teams))
    matrix_data = np.zeros((len(teams), len(teams)))
    
    for i, team1 in enumerate(teams):
        for j, team2 in enumerate(teams):
            if team1 != team2:
                matches = head2headrecord_df[
                    (head2headrecord_df['Team'] == team1) & 
                    (head2headrecord_df['Opponent'] == team2)
                ]
                if not matches.empty:
                    matrix_data[i][j] = matches.iloc[0]['Win_Percentage']
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix_data,
        x=teams,
        y=teams,
        colorscale='RdYlGn',  # Red to Yellow to Green colorscale
        text=np.round(matrix_data, 1).astype(str) + '%',
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        showscale=True,
        colorbar=dict(title='Win %')
    ))
    
    fig.update_layout(
        height=600,
        xaxis_title="Opponent",
        yaxis_title="Team",
        xaxis={'tickangle': 45}
    )
    
    st.plotly_chart(fig, use_container_width=True)

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
        team_matches['Year'] = pd.to_datetime(team_matches['Date']).apply(lambda x: x.year)
        
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
                    if margin.startswith(team.lower()):
                        outings.append("<div class='outing-indicator win'>W</div>")
                        w_count += 1
                    elif 'drawn' in margin:
                        outings.append("<div class='outing-indicator draw'>D</div>")
                        d_count += 1
                    else:
                        outings.append("<div class='outing-indicator loss'>L</div>")
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
                        (team_data['Format'] == fm) & mask
                    ].sort_values('Date', ascending=False).head(20)
                    
                    if not sub_data.empty:
                        outings = []
                        w_count, l_count, d_count = 0, 0, 0
                        for _, mtch in sub_data.iterrows():
                            margin = mtch['Margin'].lower()
                            if margin.startswith(team.lower()):
                                outings.append("<div class='outing-indicator win'>W</div>")
                                w_count += 1
                            elif 'drawn' in margin:
                                outings.append("<div class='outing-indicator draw'>D</div>")
                                d_count += 1
                            else:
                                outings.append("<div class='outing-indicator loss'>L</div>")
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
# ...existing code...


