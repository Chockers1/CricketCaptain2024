# Imports and Setup
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

            bowl_stats_per_file = bowl_stats.groupby('File Name')['Bowler_Wkts'].sum().reset_index()
            bowl_stats_per_file['TenW'] = np.where(bowl_stats_per_file['Bowler_Wkts'] >= 10, 1, 0)
            tenW_count = bowl_stats_per_file['TenW'].sum()
            career_stats['10Ws'] = tenW_count

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
            career_stats['100s Per Match'] = (
                (career_stats['100s'] / career_stats['Matches'].replace(0, np.inf)) * 100
            ).round(2)
            career_stats['5Ws Per Match'] = (
                (career_stats['5Ws'] / career_stats['Matches'].replace(0, np.inf)) * 100
            ).round(2)

            pom_count = bat_stats[bat_stats['Player_of_the_Match'] == player_name].groupby('Name').agg(
                POM=('File Name', 'nunique')
            ).reset_index()

            career_stats = career_stats.merge(pom_count, on='Name', how='left')
            career_stats['POM'] = career_stats['POM'].fillna(0).astype(int)
            career_stats['POM Per Match'] = (
                (career_stats['POM'] / career_stats['Matches'].replace(0, np.inf)) * 100
            ).round(2)

            return career_stats

        player1_stats = calculate_stats(player1_choice)
        player2_stats = calculate_stats(player2_choice)

        st.markdown("<div class='comparison-container'>", unsafe_allow_html=True)
        
        st.markdown(f"<div class='comparison-row'><div class='comparison-column'>{player1_choice}</div><div class='comparison-metric'></div><div class='comparison-column'>{player2_choice}</div></div>", unsafe_allow_html=True)
        
        metrics = [
            "Matches",
            "Runs",
            "Bat Average",
            "Bat Strike Rate",
            "Balls Per Out",
            "Runs Per Match",
            "50s",
            "100s",
            "100s Per Match",
            "Overs",
            "Wickets",
            "Bowl Average",
            "Economy Rate",
            "Bowl Strike Rate",
            "Wickets Per Match",
            "5Ws",
            "5Ws Per Match",
            "10Ws",
            "POM",
            "POM Per Match"
        ]
        for metric in metrics:
            player1_value = player1_stats[metric].values[0]
            player2_value = player2_stats[metric].values[0]
            
            if metric in [
                "Matches",
                "Runs",
                "Bat Average",
                "Bat Strike Rate",
                "Balls Per Out",
                "Runs Per Match",
                "50s",
                "100s",
                "100s Per Match",
                "Overs",
                "Wickets",
                "Wickets Per Match",
                "5Ws",
                "5Ws Per Match",
                "10Ws",
                "POM",
                "POM Per Match"
            ]:
                player1_class = "highlight-green" if player1_value > player2_value else "highlight-red"
                player2_class = "highlight-green" if player2_value > player1_value else "highlight-red"
            else:
                player1_class = "highlight-green" if player1_value < player2_value else "highlight-red"
                player2_class = "highlight-green" if player2_value < player1_value else "highlight-red"

            if metric == "Economy Rate" and player1_value == 0.0 and player1_stats['Overs'].values[0] == 0.0:
                player1_class = "highlight-red"
            if metric == "Economy Rate" and player2_value == 0.0 and player2_stats['Overs'].values[0] == 0.0:
                player2_class = "highlight-red"

            if metric in ["100s Per Match", "5Ws Per Match", "POM Per Match"]:
                p1_display = f"{player1_value}%"
                p2_display = f"{player2_value}%"
            else:
                p1_display = player1_value
                p2_display = player2_value

            st.markdown(f"""
            <div class='comparison-row'>
                <div class='comparison-column {player1_class}'>{p1_display}</div>
                <div class='comparison-metric'>{metric}</div>
                <div class='comparison-column {player2_class}'>{p2_display}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Add Radar Chart
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Player Comparison Radar</h3>", unsafe_allow_html=True)
        
        # Select metrics for radar chart
        radar_metrics = ["Bat Average", "Bat Strike Rate", "Balls Per Out", "Bowl Average", "Bowl Strike Rate"]
        
        fig = go.Figure()
        
        # Add trace for player 1
        fig.add_trace(go.Scatterpolar(
            r=[player1_stats[m].values[0] for m in radar_metrics],
            theta=radar_metrics,
            fill='toself',
            name=player1_choice
        ))
        
        # Add trace for player 2
        fig.add_trace(go.Scatterpolar(
            r=[player2_stats[m].values[0] for m in radar_metrics],
            theta=radar_metrics,
            fill='toself',
            name=player2_choice
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max([max([player1_stats[m].values[0] for m in radar_metrics]),
                                 max([player2_stats[m].values[0] for m in radar_metrics])])]
                )),
            showlegend=True,
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Career Progression Chart
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Career Progression</h3>", unsafe_allow_html=True)
        
        # Create year-by-year stats for both players
        def get_yearly_stats(player_name):
            bat_yearly = filtered_bat_df[filtered_bat_df['Name'] == player_name].groupby('Year').agg({
                'Runs': 'sum',
                'Out': 'sum'
            }).reset_index()
            
            bowl_yearly = filtered_bowl_df[filtered_bowl_df['Name'] == player_name].groupby('Year').agg({
                'Bowler_Wkts': 'sum'
            }).reset_index()
            
            return pd.merge(bat_yearly, bowl_yearly, on='Year', how='outer').fillna(0)

        player1_yearly = get_yearly_stats(player1_choice)
        player2_yearly = get_yearly_stats(player2_choice)

        # Create subplot with two y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces for runs
        fig.add_trace(
            go.Scatter(x=player1_yearly['Year'], y=player1_yearly['Runs'],
                      name=f"{player1_choice} Runs", line=dict(color='#f04f53')),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=player2_yearly['Year'], y=player2_yearly['Runs'],
                      name=f"{player2_choice} Runs", line=dict(dash='dash', color='#f04f53')),
            secondary_y=False,
        )

        # Add traces for wickets
        fig.add_trace(
            go.Scatter(x=player1_yearly['Year'], y=player1_yearly['Bowler_Wkts'],
                      name=f"{player1_choice} Wickets", line=dict(color='#6c24c0')),
            secondary_y=True,
        )
        
        fig.add_trace(
            go.Scatter(x=player2_yearly['Year'], y=player2_yearly['Bowler_Wkts'],
                      name=f"{player2_choice} Wickets", line=dict(dash='dash', color='#6c24c0')),
            secondary_y=True,
        )

        # Update layout
        fig.update_layout(
            title_text="Runs and Wickets by Year",
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )

        # Update axes
        fig.update_xaxes(title_text="Year", showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
        fig.update_yaxes(title_text="Runs", secondary_y=False, showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
        fig.update_yaxes(title_text="Wickets", secondary_y=True, showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Required data not found. Please ensure you have processed the scorecards.")

# No need for the if __name__ == "__main__" part
display_comparison_view()
