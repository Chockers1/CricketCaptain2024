# Imports and Setup
import streamlit as st
import pandas as pd
import numpy as np

def display_comparison_view():
    # Custom styling
    st.markdown("""
    <style>
    .stSlider p { color: #f04f53 !important; }
    .comparison-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 10px;
        margin-top: 20px;
    }
    .comparison-row {
        display: flex;
        justify-content: space-between;
        width: 100%;
        padding: 5px 0;
        border-bottom: 1px solid #ddd;
    }
    .comparison-column {
        flex: 1;
        text-align: center;
    }
    .comparison-metric {
        flex: 1;
        text-align: center;
        font-weight: bold;
        color: #f04f53;
    }
    .player-name {
        text-align: center;
        font-weight: bold;
        color: #f04f53;
        margin-bottom: 10px;
    }
    .highlight-green {
        background-color: #d4edda;
    }
    .highlight-red {
        background-color: #f8d7da;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 style='color:#f04f53; text-align: center;'>Player Comparison</h1>", unsafe_allow_html=True)

    if 'bat_df' in st.session_state and 'bowl_df' in st.session_state:
        # Make copies of original dataframes
        bat_df = st.session_state['bat_df'].copy()
        bowl_df = st.session_state['bowl_df'].copy()

        # Create Year columns from Date with safer date parsing
        try:
            bat_df['Date'] = pd.to_datetime(bat_df['Date'], format='%d %b %Y', errors='coerce')
            bowl_df['Date'] = pd.to_datetime(bowl_df['Date'], format='%d %b %Y', errors='coerce')
        except:
            bat_df['Date'] = pd.to_datetime(bat_df['Date'], dayfirst=True, errors='coerce')
            bowl_df['Date'] = pd.to_datetime(bowl_df['Date'], dayfirst=True, errors='coerce')

        bat_df['Year'] = bat_df['Date'].dt.year
        bowl_df['Year'] = bowl_df['Date'].dt.year

        bat_df['Year'] = pd.to_numeric(bat_df['Year'], errors='coerce').fillna(0).astype(int)
        bowl_df['Year'] = pd.to_numeric(bowl_df['Year'], errors='coerce').fillna(0).astype(int)

        names = sorted(bat_df['Name'].unique().tolist())
        match_formats = ['All'] + sorted(bat_df['Match_Format'].unique().tolist())

        col1, col2, col3 = st.columns(3)
        
        with col1:
            player1_choice = st.selectbox('Player 1:', names)
        with col2:
            player2_choice = st.selectbox('Player 2:', names)
        with col3:
            match_format_choice = st.selectbox('Format:', match_formats, index=0)

        filtered_bat_df = bat_df.copy()
        filtered_bowl_df = bowl_df.copy()

        if match_format_choice != 'All':
            filtered_bat_df = filtered_bat_df[filtered_bat_df['Match_Format'] == match_format_choice]
            filtered_bowl_df = filtered_bowl_df[filtered_bowl_df['Match_Format'] == match_format_choice]

        def calculate_stats(player_name):
            bat_stats = filtered_bat_df[filtered_bat_df['Name'] == player_name]
            # Make sure we're only getting bowling stats for when the player was bowling
            bowl_stats = filtered_bowl_df[
                (filtered_bowl_df['Name'] == player_name) & 
                (filtered_bowl_df['Bowler_Balls'] > 0)
            ]

            career_stats = pd.merge(
                bat_stats.groupby('Name').agg({
                    'File Name': 'nunique',
                    'Runs': 'sum',
                    'Out': 'sum',
                    'Balls': 'sum',
                    '50s': 'sum',
                    '100s': 'sum'
                }).reset_index(),
                bowl_stats.groupby('Name').agg({
                    'Bowler_Runs': 'sum',
                    'Bowler_Wkts': 'sum',  # Changed to Bowler_Wkts
                    'Bowler_Balls': 'sum',
                    '5Ws': 'sum'
                }).reset_index(),
                on='Name',
                how='outer'
            ).fillna(0)

            career_stats['Matches'] = career_stats['File Name']
            career_stats['Bat Average'] = (career_stats['Runs'] / career_stats['Out'].replace(0, np.inf)).round(2)
            career_stats['Bat Strike Rate'] = ((career_stats['Runs'] / career_stats['Balls'].replace(0, np.inf)) * 100).round(2)
            career_stats['Balls Per Out'] = (career_stats['Balls'] / career_stats['Out'].replace(0, np.inf)).round(2)
            career_stats['Runs Per Match'] = (career_stats['Runs'] / career_stats['Matches']).round(2)
            career_stats['Overs'] = (career_stats['Bowler_Balls'] // 6) + (career_stats['Bowler_Balls'] % 6) / 10
            career_stats['Wickets'] = career_stats['Bowler_Wkts']  # Use Bowler_Wkts directly
            career_stats['Bowl Average'] = np.where(
                career_stats['Bowler_Wkts'] > 0,
                (career_stats['Bowler_Runs'] / career_stats['Bowler_Wkts']).round(2),
                np.inf
            )
            career_stats['Economy Rate'] = np.where(
                career_stats['Overs'] > 0,
                (career_stats['Bowler_Runs'] / career_stats['Overs']).round(2),
                0
            )
            career_stats['Bowl Strike Rate'] = np.where(
                career_stats['Bowler_Wkts'] > 0,
                (career_stats['Bowler_Balls'] / career_stats['Bowler_Wkts']).round(2),
                np.inf
            )
            career_stats['Wickets Per Match'] = (career_stats['Wickets'] / career_stats['Matches']).round(2)

            pom_count = bat_stats[bat_stats['Player_of_the_Match'] == player_name].groupby('Name').agg(
                POM=('File Name', 'nunique')
            ).reset_index()

            career_stats = career_stats.merge(pom_count, on='Name', how='left')
            career_stats['POM'] = career_stats['POM'].fillna(0).astype(int)

            return career_stats

        player1_stats = calculate_stats(player1_choice)
        player2_stats = calculate_stats(player2_choice)

        st.markdown("<div class='comparison-container'>", unsafe_allow_html=True)
        
        st.markdown(f"<div class='comparison-row'><div class='comparison-column'>{player1_choice}</div><div class='comparison-metric'></div><div class='comparison-column'>{player2_choice}</div></div>", unsafe_allow_html=True)
        
        metrics = ["Matches", "Runs", "Bat Average", "Bat Strike Rate", "Balls Per Out", "Runs Per Match", 
                  "50s", "100s", "Overs", "Wickets", "Bowl Average", "Economy Rate", "Bowl Strike Rate", 
                  "Wickets Per Match", "5Ws", "POM"]
        for metric in metrics:
            player1_value = player1_stats[metric].values[0]
            player2_value = player2_stats[metric].values[0]
            
            if metric in ["Matches", "Runs", "Bat Average", "Bat Strike Rate", "Balls Per Out", "Runs Per Match", 
                         "50s", "100s", "Overs", "Wickets", "Wickets Per Match", "5Ws", "POM"]:
                player1_class = "highlight-green" if player1_value > player2_value else "highlight-red"
                player2_class = "highlight-green" if player2_value > player1_value else "highlight-red"
            else:
                player1_class = "highlight-green" if player1_value < player2_value else "highlight-red"
                player2_class = "highlight-green" if player2_value < player1_value else "highlight-red"

            if metric == "Economy Rate" and player1_value == 0.0 and player1_stats['Overs'].values[0] == 0.0:
                player1_class = "highlight-red"
            if metric == "Economy Rate" and player2_value == 0.0 and player2_stats['Overs'].values[0] == 0.0:
                player2_class = "highlight-red"

            st.markdown(f"""
            <div class='comparison-row'>
                <div class='comparison-column {player1_class}'>{player1_value}</div>
                <div class='comparison-metric'>{metric}</div>
                <div class='comparison-column {player2_class}'>{player2_value}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.error("Required data not found. Please ensure you have processed the scorecards.")

# No need for the if __name__ == "__main__" part
display_comparison_view()
