# Imports and Setup
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random


# Modern CSS for beautiful UI - Full Home.py styling
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
    
    .uploadedFile {
        border-radius: 10px;
        border: 2px dashed #f04f53;
        padding: 20px;
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
    
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def process_dataframes():
    """Process all dataframes"""
    if 'bat_df' in st.session_state:
        # OPTIMIZATION: Use reference instead of copy - saves memory
        bat_df = st.session_state['bat_df']  # No .copy() needed for read operations
        # Ensure proper date parsing for bat_df
        try:
            bat_df['Date'] = pd.to_datetime(bat_df['Date'], format='%d %b %Y', errors='coerce')
        except:
            bat_df['Date'] = pd.to_datetime(bat_df['Date'], dayfirst=True, errors='coerce')
    else:
        bat_df = None

    if 'bowl_df' in st.session_state:
        bowl_df = st.session_state['bowl_df'].copy()
        # Ensure proper date parsing for bowl_df
        try:
            bowl_df['Date'] = pd.to_datetime(bowl_df['Date'], format='%d %b %Y', errors='coerce')
        except:
            bowl_df['Date'] = pd.to_datetime(bowl_df['Date'], dayfirst=True, errors='coerce')
    else:
        bowl_df = None

    if 'match_df' in st.session_state:
        match_df = st.session_state['match_df'].copy()
        # Ensure proper date parsing for match_df
        try:
            match_df['Date'] = pd.to_datetime(match_df['Date'], format='%d %b %Y', errors='coerce')
        except:
            match_df['Date'] = pd.to_datetime(match_df['Date'], dayfirst=True, errors='coerce')
    else:
        match_df = None

    if 'game_df' in st.session_state:
        game_df = st.session_state['game_df'].copy()
        # Ensure proper date parsing for game_df
        try:
            game_df['Date'] = pd.to_datetime(game_df['Date'], format='%d %b %Y', errors='coerce')
        except:
            game_df['Date'] = pd.to_datetime(game_df['Date'], dayfirst=True, errors='coerce')
    else:
        game_df = None

    return bat_df, bowl_df, match_df, game_df

# Get the dataframes and create scorecard copies
bat_df, bowl_df, match_df, game_df = process_dataframes()
sc_bat_df = bat_df.copy() if bat_df is not None else None
sc_bowl_df = bowl_df.copy() if bowl_df is not None else None
sc_match_df = match_df.copy() if match_df is not None else None
sc_game_df = game_df.copy() if game_df is not None else None

# Remove raw match data display

# Standardize column names and create unique IDs
if sc_bat_df is not None:
    # Standardize team names
    sc_bat_df = sc_bat_df.rename(columns={
        'Home Team': 'Home_Team',
        'Away Team': 'Away_Team'
    })
    # Create unique ID for batting entries
    sc_bat_df['Match_Innings_ID'] = sc_bat_df['File Name'].astype(str) + '_' + sc_bat_df['Innings'].astype(str)

if sc_bowl_df is not None:
    # Create unique ID for bowling entries
    sc_bowl_df['Match_Innings_ID'] = sc_bowl_df['File Name'].astype(str) + '_' + sc_bowl_df['Innings'].astype(str)

if sc_game_df is not None:
    # Create unique ID for game entries
    sc_game_df['Match_Innings_ID'] = sc_game_df['File Name'].astype(str) + '_' + sc_game_df['Innings'].astype(str)

# Now we can join data using either:
# 1. 'File Name' for match-level data
# 2. 'Match_Innings_ID' for innings-level data

def get_match_data(file_name):
    """Get all data for a specific match"""
    if sc_match_df is not None:
        match_info = sc_match_df[sc_match_df['File Name'] == file_name].iloc[0]
        game_info = sc_game_df[sc_game_df['File Name'] == file_name]
        bat_info = sc_bat_df[sc_bat_df['File Name'] == file_name]
        bowl_info = sc_bowl_df[sc_bowl_df['File Name'] == file_name]
        return match_info, game_info, bat_info, bowl_info
    return None

def get_innings_data(match_innings_id):
    """Get all data for a specific innings"""
    if all(df is not None for df in [sc_game_df, sc_bat_df, sc_bowl_df]):
        game_info = sc_game_df[sc_game_df['Match_Innings_ID'] == match_innings_id]
        bat_info = sc_bat_df[sc_bat_df['Match_Innings_ID'] == match_innings_id]
        bowl_info = sc_bowl_df[sc_bowl_df['Match_Innings_ID'] == match_innings_id]
        return game_info, bat_info, bowl_info
    return None

# Create match selection interface with modern purple gradient banner
st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px; border-radius: 15px; text-align: center; margin: 20px 0;
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);'>
        <h1 style='color: white; font-size: 2.5rem; margin-bottom: 15px; font-weight: bold;'>
            🔍 Scorecards
        </h1>
        <p style='color: rgba(255,255,255,0.9); font-size: 1.2rem; margin-bottom: 0;'>
            Use the filters below to find your scorecard and select it in the table below.
        </p>
    </div>
""", unsafe_allow_html=True)

if sc_match_df is not None:
    def get_filtered_matches(home=None, away=None, format=None, player=None, date=None):
        """Get filtered matches based on current selections"""
        filtered = sc_match_df.copy()
        
        if home and home != 'All':
            filtered = filtered[filtered['Home_Team'] == home]
        if away and away != 'All':
            filtered = filtered[filtered['Away_Team'] == away]
        if format and format != 'All':
            filtered = filtered[filtered['Match_Format'] == format]
        if player and player != 'All':
            player_matches = sc_bat_df[sc_bat_df['Name'] == player]['File Name'].unique()
            filtered = filtered[filtered['File Name'].isin(player_matches)]
        if date and date != 'All':
            filtered = filtered[filtered['Date'].dt.strftime('%d/%m/%Y') == date]
        
        return filtered

    # Create filters in columns
    col1, col2 = st.columns(2)
    col3, col4, col5 = st.columns(3)
    
    # Get initially filtered matches
    filtered_matches = get_filtered_matches()
    
    with col1:
        home_teams = ['All'] + sorted(filtered_matches['Home_Team'].unique().tolist())
        home_team = st.selectbox('Home Team:', home_teams, index=0)
    
    # Update filtered matches for away team options
    filtered_matches = get_filtered_matches(home_team)
    
    with col2:
        away_teams = ['All'] + sorted(filtered_matches['Away_Team'].unique().tolist())
        away_team = st.selectbox('Away Team:', away_teams, index=0)
    
    # Update filtered matches for format options
    filtered_matches = get_filtered_matches(home_team, away_team)
    
    with col3:
        formats = ['All'] + sorted(filtered_matches['Match_Format'].unique().tolist())
        match_format = st.selectbox('Format:', formats, index=0)
    
    # Update filtered matches for player options
    filtered_matches = get_filtered_matches(home_team, away_team, match_format)
    
    with col4:
        if sc_bat_df is not None:
            player_files = filtered_matches['File Name'].unique()
            available_players = sc_bat_df[sc_bat_df['File Name'].isin(player_files)]['Name'].unique()
            names = ['All'] + sorted(available_players.tolist())
            player_name = st.selectbox('Player Name:', names, index=0)
    
    # Final filter update for dates
    filtered_matches = get_filtered_matches(home_team, away_team, match_format, player_name)
    
    with col5:
        dates = ['All'] + sorted(filtered_matches['Date'].dt.strftime('%d/%m/%Y').unique().tolist())
        match_date = st.selectbox('Date:', dates)
    
    # Set the final selected matches
    selected_match = get_filtered_matches(home_team, away_team, match_format, player_name, match_date)

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

    # Display selected matches with modern styling
    if not selected_match.empty:
        st.markdown("""
            <div class="result-banner">
                📋 Selected Match Details
            </div>
        """, unsafe_allow_html=True)
        
        # Select and rename columns for display
        display_columns = {
            'File Name': 'File Name',
            'Home_Team': 'Home Team',
            'Away_Team': 'Away Team',
            'Date': 'Date',
            'Competition': 'Competition',
            'Match_Format': 'Match Format',
            'Player_of_the_Match': 'Player of the Match',
            'Match_Result': 'Match Result'
        }
        
        # Ensure all required columns exist
        available_columns = [col for col in display_columns.keys() if col in selected_match.columns]
        selected_match = selected_match[available_columns].copy()
        
        # Rename available columns
        rename_dict = {col: display_columns[col] for col in available_columns}
        selected_match = selected_match.rename(columns=rename_dict)
        
        # Keep original datetime for sorting and create display date
        if 'Date' in selected_match.columns:
            selected_match = selected_match.sort_values('Date', ascending=False)
            selected_match['Display_Date'] = selected_match['Date'].dt.strftime('%d/%m/%Y')
            
            # Create display version with formatted date
            display_match = selected_match.copy()
            display_match['Date'] = display_match['Display_Date']
            display_match = display_match.drop('Display_Date', axis=1)
        else:
            display_match = selected_match.copy()

        # Reset index to ensure clean dataframe
        display_match = display_match.reset_index(drop=True)
        
        # Add a "Select" column typed as bool
        display_match["Select"] = False
        display_match["Select"] = display_match["Select"].astype(bool)

        # Single call to st.data_editor with error handling
        try:
            edited_matches = st.data_editor(
                display_match,
                column_config={
                    "Select": st.column_config.CheckboxColumn(label="Select", default=False)
                },
                use_container_width=True,
                hide_index=True,
                key="match_editor"
            )
        except Exception as e:
            st.error(f"Error displaying match data: {str(e)}")
            # Fallback to regular dataframe display
            st.dataframe(display_match, use_container_width=True, hide_index=True)
            edited_matches = display_match

        # Capture and store the selected file
        selected_indices = edited_matches[edited_matches["Select"]].index
        if len(selected_indices) == 1:
            row = edited_matches.loc[selected_indices[0]]
            st.session_state['selected_match_file'] = row["File Name"]
        else:
            st.session_state.pop('selected_match_file', None)

        # Initialize display variables
        file_name = None
        display_match = False

        # Handle row selection using session state
        if 'selected_match_file' in st.session_state:
            file_name = st.session_state['selected_match_file']
            # Verify the file still exists in current filtered set
            if file_name in selected_match['File Name'].values:
                display_match = True
            else:
                del st.session_state['selected_match_file']

        # Display match details if we have a valid file_name
        if display_match and file_name:
            # Clear selection if filters change
            if (not selected_match['File Name'].isin([file_name]).any()):
                if 'selected_match_file' in st.session_state:
                    del st.session_state['selected_match_file']
                file_name = selected_match['File Name'].iloc[0]

            # Get corresponding game data
            innings_data = sc_game_df[sc_game_df['File Name'] == file_name].sort_values('Innings')
            
            # Get match result details from match_df
            match_details = sc_match_df[sc_match_df['File Name'] == file_name][
                ['Match_Result', 'Innings_Win', 'Margin_Runs', 'Margin_Wickets']
            ].iloc[0]
            
            # Add match details to innings_data
            innings_data = innings_data.assign(
                Match_Result=match_details['Match_Result'],
                Innings_Win=match_details['Innings_Win'],
                Margin_Runs=match_details['Margin_Runs'],
                Margin_Wickets=match_details['Margin_Wickets']
            )

            # Check for follow-on scenario
            follow_on = False
            if innings_data.iloc[0:2].shape[0] >= 2:  # Make sure we have at least 2 innings
                first_batting_team = innings_data.iloc[0]['Bat_Team']
                second_batting_team = innings_data.iloc[1]['Bat_Team']
                
                # Case 1: First batting team wins by wickets
                if (pd.notna(match_details['Margin_Wickets']) and 
                    match_details['Margin_Wickets'] > 0 and 
                    first_batting_team in match_details['Match_Result']):
                    follow_on = True
                
                # Case 2: Second batting team wins by runs
                elif (pd.notna(match_details['Margin_Runs']) and 
                      match_details['Margin_Runs'] > 0 and 
                      second_batting_team in match_details['Match_Result']):
                    follow_on = True

            # Get corresponding game data and sort by display order
            innings_data = sc_game_df[sc_game_df['File Name'] == file_name].copy()
            
            # If follow on, modify the display order
            if follow_on:
                innings_data['Display_Order'] = innings_data['Innings'].map({1: 1, 2: 2, 3: 4, 4: 3})
            else:
                innings_data['Display_Order'] = innings_data['Innings']
                
            innings_data = innings_data.sort_values('Display_Order')
            

            
            # Add Match Result first with slightly smaller text
            selected_row = selected_match[selected_match["File Name"] == file_name]
            match_result = selected_row["Match Result"].iloc[0]
            pom = selected_row["Player of the Match"].iloc[0]

            # Match highlights in a modern card
            st.markdown("""
                <div class="modern-card">
                    <h3 style='color:#f04f53; margin-bottom: 20px;'>🏆 Match Highlights</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Add Match Result in styled format
            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 15px; border-radius: 10px; text-align: center; margin: 15px 0;
                            box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                    <span style='color: white; font-weight: bold; font-size: 18px;'>
                        🎯 Match Result: {match_result}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Add Player of the Match in styled format
            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            padding: 12px; border-radius: 10px; text-align: center; margin: 10px 0;
                            box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                    <span style='color: white; font-weight: bold; font-size: 16px;'>
                        🌟 Player of the Match: {pom}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Get top run scorer with modern styling
            match_batting = sc_bat_df[sc_bat_df['File Name'] == file_name]
            if not match_batting.empty:
                top_scorer = match_batting.loc[match_batting['Runs'].idxmax()]
                st.markdown(
                    f"""
                    <div style='background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); 
                                padding: 12px; border-radius: 10px; text-align: center; margin: 10px 0;
                                box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                        <span style='color: white; font-weight: bold; font-size: 16px;'>
                            🏏 Highest Score: {top_scorer['Name']} ({top_scorer['Runs']} runs)
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Get top wicket taker with modern styling
            match_bowling = sc_bowl_df[sc_bowl_df['File Name'] == file_name]
            if not match_bowling.empty:
                top_bowler = match_bowling.loc[match_bowling['Bowler_Wkts'].idxmax()]
                st.markdown(
                    f"""
                    <div style='background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%); 
                                padding: 12px; border-radius: 10px; text-align: center; margin: 10px 0;
                                box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                        <span style='color: white; font-weight: bold; font-size: 16px;'>
                            🎳 Best Bowling: {top_bowler['Name']} ({top_bowler['Bowler_Wkts']}-{top_bowler['Bowler_Runs']})
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            for idx, innings in innings_data.iterrows():
                if match_details['Innings_Win'] == 1:
                    innings_num = {1: "1st", 2: "2nd", 3: "3rd"}[innings['Innings']]
                else:
                    innings_num = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}[innings['Display_Order']]
                
                following_on_text = " Following On" if follow_on and innings['Display_Order'] == 3 else ""
                
                # Get the match format
                match_format_value = selected_row["Match Format"].iloc[0]

                lead_trail_text = ""
                if innings['Innings'] == 2 and match_format_value in ["Test Match", "First Class"]:
                    first_innings_score = innings_data[innings_data['Innings'] == 1]['Total_Runs'].values[0]
                    second_innings_score = innings['Total_Runs']
                    if second_innings_score > first_innings_score:
                        lead_trail_text = f" lead by {second_innings_score - first_innings_score} runs"
                    else:
                        lead_trail_text = f" trail by {first_innings_score - second_innings_score} runs"
                
                # Beautiful innings header card
                innings_gradient = {
                    "1st": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                    "2nd": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)", 
                    "3rd": "linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%)",
                    "4th": "linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%)"
                }
                
                gradient = innings_gradient.get(innings_num, "linear-gradient(135deg, #667eea 0%, #764ba2 100%)")
                
                st.markdown(
                    f"""
                    <div style='background: {gradient}; 
                                padding: 20px; border-radius: 15px; text-align: center; margin: 20px 0;
                                box-shadow: 0 8px 25px rgba(0,0,0,0.15);'>
                        <div style='color: white; font-weight: bold; font-size: 20px; margin-bottom: 8px;'>
                            🏏 {innings_num} Innings{following_on_text}
                        </div>
                        <div style='color: white; font-size: 24px; font-weight: bold; margin-bottom: 8px;'>
                            {innings['Bat_Team']} - {innings['Total_Runs']}/{innings['Wickets']}
                        </div>
                        <div style='color: rgba(255,255,255,0.9); font-size: 16px;'>
                            📊 Overs: {innings['Overs']} | Run Rate: {innings['Run_Rate']}{lead_trail_text}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Get batting details for this innings
                match_innings_id = innings['Match_Innings_ID']
                batting_details = sc_bat_df[sc_bat_df['Match_Innings_ID'] == match_innings_id]
                bowling_details = sc_bowl_df[sc_bowl_df['Match_Innings_ID'] == match_innings_id]
                
                if not batting_details.empty:
                    # Add beautiful batting section header
                    st.markdown("""
                        <div style='background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); 
                                    padding: 12px; border-radius: 10px; text-align: center; margin: 15px 0 10px 0;
                                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                            <span style='color: white; font-weight: bold; font-size: 16px;'>
                                🏏 Batting Scorecard
                            </span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Display batting details with enhanced styling
                    batting_details = batting_details[['Position', 'Name', 'How Out', 'Runs', 'Balls', '4s', '6s', 'Boundary Runs', 'Strike Rate']]
                    batting_details = batting_details.sort_values('Position')
                    
                    # Apply custom styling to the dataframe
                    st.markdown("""
                        <style>
                        .batting-table {
                            border-radius: 15px;
                            overflow: hidden;
                            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                            border: none;
                        }
                        .batting-table thead tr th {
                            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%) !important;
                            color: white !important;
                            font-weight: bold;
                            font-size: 14px;
                            padding: 15px 8px;
                            text-align: center;
                            border: none;
                        }
                        .batting-table tbody tr {
                            transition: all 0.3s ease;
                            border: none;
                        }
                        .batting-table tbody tr:hover {
                            background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%) !important;
                            transform: scale(1.02);
                        }
                        .batting-table tbody tr td {
                            padding: 12px 8px;
                            text-align: center;
                            border: none;
                            font-weight: 500;
                        }
                        </style>
                    """, unsafe_allow_html=True)
                    
                    st.dataframe(
                        batting_details, 
                        use_container_width=True, 
                        hide_index=True, 
                        height=425,
                        column_config={
                            "Position": st.column_config.NumberColumn("Pos", width="small"),
                            "Name": st.column_config.TextColumn("Batsman", width="medium"),
                            "How Out": st.column_config.TextColumn("How Out", width="large"),
                            "Runs": st.column_config.NumberColumn("Runs", width="small"),
                            "Balls": st.column_config.NumberColumn("Balls", width="small"),
                            "4s": st.column_config.NumberColumn("4s", width="small"),
                            "6s": st.column_config.NumberColumn("6s", width="small"),
                            "Boundary Runs": st.column_config.NumberColumn("Boundary", width="small"),
                            "Strike Rate": st.column_config.NumberColumn("SR", width="small", format="%.1f")
                        }
                    )

                if not bowling_details.empty:
                    # Add beautiful bowling section header
                    st.markdown("""
                        <div style='background: linear-gradient(135deg, #FF6B6B 0%, #EE5A6F 100%); 
                                    padding: 12px; border-radius: 10px; text-align: center; margin: 25px 0 10px 0;
                                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                            <span style='color: white; font-weight: bold; font-size: 16px;'>
                                🎳 Bowling Figures
                            </span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Display bowling details with enhanced styling
                    bowling_details = bowling_details[['Position', 'Name', 'Bowler_Overs', 'Maidens', 'Bowler_Runs', 'Bowler_Wkts', 'Bowler_Econ']]
                    bowling_details = bowling_details.rename(columns={
                        'Bowler_Overs': 'Overs',
                        'Maidens': 'Maidens/Dots',
                        'Bowler_Runs': 'Runs',
                        'Bowler_Wkts': 'Wickets',
                        'Bowler_Econ': 'Economy Rate'
                    })
                    bowling_details = bowling_details.sort_values('Position')
                    
                    # Apply custom styling for bowling table
                    st.markdown("""
                        <style>
                        .bowling-table {
                            border-radius: 15px;
                            overflow: hidden;
                            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                            border: none;
                        }
                        .bowling-table thead tr th {
                            background: linear-gradient(135deg, #FF6B6B 0%, #EE5A6F 100%) !important;
                            color: white !important;
                            font-weight: bold;
                            font-size: 14px;
                            padding: 15px 8px;
                            text-align: center;
                            border: none;
                        }
                        .bowling-table tbody tr {
                            transition: all 0.3s ease;
                            border: none;
                        }
                        .bowling-table tbody tr:hover {
                            background: linear-gradient(135deg, #ffe8e8 0%, #fff0f0 100%) !important;
                            transform: scale(1.02);
                        }
                        .bowling-table tbody tr td {
                            padding: 12px 8px;
                            text-align: center;
                            border: none;
                            font-weight: 500;
                        }
                        </style>
                    """, unsafe_allow_html=True)
                    
                    st.dataframe(
                        bowling_details, 
                        use_container_width=True, 
                        hide_index=True, 
                        height=300,
                        column_config={
                            "Position": st.column_config.NumberColumn("Pos", width="small"),
                            "Name": st.column_config.TextColumn("Bowler", width="medium"),
                            "Overs": st.column_config.NumberColumn("Overs", width="small", format="%.1f"),
                            "Maidens/Dots": st.column_config.NumberColumn("M/Dots", width="small"),
                            "Runs": st.column_config.NumberColumn("Runs", width="small"),
                            "Wickets": st.column_config.NumberColumn("Wkts", width="small"),
                            "Economy Rate": st.column_config.NumberColumn("Econ", width="small", format="%.2f")
                        }
                    )
            
def scorecard():
    pass

 
