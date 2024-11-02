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
                return pd.to_datetime(date_str, format=fmt)  # Removed .date()
            except ValueError:
                continue
        # If none of the specific formats work, let pandas try to infer the format
        return pd.to_datetime(date_str)  # Removed .date()
    except Exception:
        return pd.NaT

# Page Header
st.markdown("<h1 style='color:#f04f53; text-align: center;'>Elo Ratings</h1>", unsafe_allow_html=True)

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
    match_df = st.session_state['match_df'].copy()
    
    # Convert to datetime for proper sorting
    match_df['Date_Sort'] = pd.to_datetime(match_df['Date'])
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

# Combine the dataframes
    elo_combined_df = pd.concat([home_df, away_df], ignore_index=True)
    
    # Store the combined dataframe in session state for later use
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

    # Display the filtered ELO Ratings Dataset
    st.markdown("<h3 style='color:#f04f53; text-align: center;'>Elo Ratings Match by Match</h3>", unsafe_allow_html=True)
    st.dataframe(
        filtered_elo_df.sort_values(['Match_Format', 'Date', 'Match_Number'])[['Date', 'Match_Format', 'Team', 'Opponent', 
                                                                            'Location', 'Team_Elo_Start', 'Team_Elo_End',
                                                                            'Elo_Change', 'Opponent_Elo_Start', 'Opponent_Elo_End', 
                                                                            'Format_Number', 'Match_Number', 'Margin']],
        hide_index=True,
        use_container_width=True
    )

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

if current_ratings:  # Only create and display if we have ratings after filtering
    current_ratings_df = pd.DataFrame(current_ratings)
    
    st.markdown("<h3 style='color:#f04f53; text-align: center;'>Current Elo Ratings</h3>", unsafe_allow_html=True)
    st.dataframe(
        current_ratings_df.sort_values(['Format', 'Current_ELO'], ascending=[True, False])[
            ['Format', 'Team', 'Current_ELO', 'Max_ELO', 'Min_ELO']
        ],
        hide_index=True,
        use_container_width=True
    )
#####===================GRAPH=====================#####
    # Create line graph for ELO progression
    st.markdown("<h3 style='color:#f04f53; text-align: center;'>Elo Ratings Progression</h3>", unsafe_allow_html=True)
    
    # Filter data for the graph based on user selections
    graph_df = filtered_elo_df.copy()
    
    # Create the line plot
    fig = go.Figure()
    
    # Get unique combinations of Format and Team
    format_team_combinations = graph_df.groupby(['Match_Format', 'Team']).size().reset_index()[['Match_Format', 'Team']]
    
    # Add a line for each Format-Team combination
    for _, row in format_team_combinations.iterrows():
        format_name = row['Match_Format']
        team_name = row['Team']
        
        # Filter data for this format and team
        team_data = graph_df[
            (graph_df['Match_Format'] == format_name) & 
            (graph_df['Team'] == team_name)
        ].sort_values('Format_Number')  # Sort by Format_Number which is already in correct order
        
        # Add line to plot
        fig.add_trace(go.Scatter(
            x=team_data['Format_Number'],
            y=team_data['Team_Elo_End'],
            name=f"{team_name} ({format_name})",
            mode='lines+markers',
            hovertemplate=
            f"{team_name} ({format_name})<br>" +
            "Date: %{customdata[0]}<br>" +
            "Opponent: %{customdata[1]}<br>" +
            "Format Match: %{x}<br>" +
            "ELO: %{y:.2f}<br>" +
            "ELO Change: %{customdata[2]:+.2f}<br>" +
            "Margin: %{customdata[3]}<br>" +
            "<extra></extra>",
            customdata=list(zip(
                team_data['Date'],
                team_data['Opponent'],
                team_data['Elo_Change'].round(2),
                team_data['Margin']
            ))
        ))

    # Calculate how many rows we need for the legend
    num_traces = len(format_team_combinations)
    legend_rows = max(1, round(num_traces / 3))  # Assuming we want roughly 3 items per row
    
    # Update layout with dynamic legend positioning
    fig.update_layout(
        xaxis_title="Format Match Number",
        yaxis_title="ELO Rating",
        hovermode='closest',
        height=600 + (legend_rows * 30),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-(0.1 + (legend_rows-1)*0.1),  # Dynamic y position based on number of rows
            xanchor="center",
            x=0.5,
            traceorder="normal",
            font=dict(size=10),
            itemwidth=40,
            itemsizing="constant"
        )
    )

    # Add gridlines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

    # Add more white space at bottom of figure for legend
    fig.update_layout(margin=dict(b=50 + (legend_rows * 20)))

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
############===================number 1 per format==============================###############
############===================number 1 per format==============================###############
if 'elo_df' in st.session_state:
    # Create a copy of the DataFrame
    elo_df = st.session_state['elo_df'].copy()
    
    # Convert Date to datetime using the correct format for abbreviated month names
    try:
        if not pd.api.types.is_datetime64_any_dtype(elo_df['Date']):
            elo_df['Date'] = pd.to_datetime(elo_df['Date'], format='%d %b %Y')
        
        # Create end of month date for each row
        elo_df['Month_End'] = elo_df['Date'].dt.to_period('M').dt.to_timestamp(how='end')
        
        # Get min and max dates
        min_date = elo_df['Date'].min()
        max_date = elo_df['Date'].max()
        
        # Create date range for all months
        date_range = pd.date_range(
            start=min_date.replace(day=1),
            end=max_date,
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
                # Get all matches up to this month end
                matches_until_month = format_data[format_data['Date'] <= month_end]
                
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
            # Add a sortable date column
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
            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Monthly Elo Leaders by Format</h3>", 
                        unsafe_allow_html=True)
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
            
            summary_df = pd.DataFrame(summary_rows)
            
            # Sort by Format and then by number of months (descending)
            summary_df = summary_df.sort_values(['Format', 'Months at #1'], ascending=[True, False])
            
            # Display the summary
            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Total Months as #1 by Team and Format</h3>", 
                        unsafe_allow_html=True)
            st.dataframe(
                summary_df,
                hide_index=True,
                use_container_width=True,
                height=600
            )
        else:
            st.warning("No data available for the selected filters")
            
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.stop()