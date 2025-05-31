# Imports and Setup
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def display_comparison_view():
    # Enhanced styling with modern design
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        background: linear-gradient(135deg, #f04f53 0%, #e73a3e 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(240, 79, 83, 0.3);
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    
    .player-selector {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border: 1px solid rgba(240, 79, 83, 0.1);
    }
    
    .comparison-container {
        background: white;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        overflow: hidden;
        margin: 2rem 0;
        border: 1px solid rgba(240, 79, 83, 0.1);
    }
    
    .comparison-header {
        background: linear-gradient(135deg, #f04f53 0%, #e73a3e 100%);
        padding: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .player-header {
        color: white;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.3rem;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
        flex: 1;
        text-align: center;
    }
    
    .vs-divider {
        background: white;
        color: #f04f53;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.2rem;
        font-family: 'Inter', sans-serif;
        margin: 0 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    .comparison-body {
        padding: 0;
    }
    
    .comparison-row {
        display: flex;
        align-items: center;
        transition: all 0.3s ease;
        border-bottom: 1px solid rgba(240, 79, 83, 0.1);
    }
    
    .comparison-row:hover {
        background: rgba(240, 79, 83, 0.02);
        transform: translateY(-1px);
    }
    
    .comparison-row:last-child {
        border-bottom: none;
    }
    
    .comparison-column {
        flex: 1;
        padding: 1rem;
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 1rem;
        border-radius: 8px;
        margin: 5px;
        transition: all 0.3s ease;
    }
    
    .comparison-metric {
        flex: 1.2;
        text-align: center;
        font-weight: 600;
        color: #f04f53;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        padding: 1rem;
        background: rgba(240, 79, 83, 0.05);
        margin: 5px 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .highlight-green {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(21, 87, 36, 0.15);
        border: 1px solid rgba(21, 87, 36, 0.2);
    }
    
    .highlight-red {
        background: linear-gradient(135deg, #f8d7da 0%, #f1b0b7 100%);
        color: #721c24;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(114, 28, 36, 0.15);
        border: 1px solid rgba(114, 28, 36, 0.2);
    }
    
    .chart-container {
        background: white;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(240, 79, 83, 0.1);
    }
    
    .chart-title {
        color: #f04f53;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(240, 79, 83, 0.2);
    }
    
    .stSlider p { 
        color: #f04f53 !important; 
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
    }
    
    .stSelectbox label {
        color: #f04f53 !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
    }
    
    .metric-category {
        background: linear-gradient(135deg, #6c24c0 0%, #5a1ea0 100%);
        color: white;
        padding: 0.8rem;
        margin: 5px 0;
        text-align: center;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.9rem;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        gap: 0;
        background: white;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Enhanced header
    st.markdown("""
    <div class='main-header'>
        <h1>🏏 Player Comparison</h1>
        <p>Compare cricket players' career statistics and performance metrics</p>
    </div>
    """, unsafe_allow_html=True)

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

        # Enhanced player selector
        st.markdown("<div class='player-selector'>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            player1_choice = st.selectbox('🏏 Player 1:', names)
        with col2:
            player2_choice = st.selectbox('🏏 Player 2:', names)
        with col3:
            match_format_choice = st.selectbox('📊 Format:', match_formats, index=0)
        st.markdown("</div>", unsafe_allow_html=True)

        filtered_bat_df = bat_df.copy()
        filtered_bowl_df = bowl_df.copy()

        if match_format_choice != 'All':
            filtered_bat_df = filtered_bat_df[filtered_bat_df['Match_Format'] == match_format_choice]
            filtered_bowl_df = filtered_bowl_df[filtered_bowl_df['Match_Format'] == match_format_choice]

        def calculate_stats(player_name):
            bat_stats = filtered_bat_df[filtered_bat_df['Name'] == player_name]
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
                    'Bowler_Wkts': 'sum',
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
            career_stats['Wickets'] = career_stats['Bowler_Wkts']
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
            career_stats['50+ Scores Per Match'] = (
                (career_stats['50s'] + career_stats['100s']) 
                / career_stats['Matches'].replace(0, np.inf) 
                * 100
            ).round(2)
            career_stats['50+ Scores'] = career_stats['50s'] + career_stats['100s']

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

        # Enhanced comparison table with categories
        st.markdown("<div class='comparison-container'>", unsafe_allow_html=True)
        
        # Header
        st.markdown(f"""
        <div class='comparison-header'>
            <div class='player-header'>{player1_choice}</div>
            <div class='vs-divider'>VS</div>
            <div class='player-header'>{player2_choice}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='comparison-body'>", unsafe_allow_html=True)
        
        # Organize metrics by category
        metric_categories = {
            "🏏 General": ["Matches"],
            "🔥 Batting Performance": [
                "Runs", "Bat Average", "Bat Strike Rate", "Balls Per Out", "Runs Per Match"
            ],
            "🎯 Batting Milestones": [
                "50+ Scores", "50s", "100s", "50+ Scores Per Match", "100s Per Match"
            ],
            "⚡ Bowling Performance": [
                "Overs", "Wickets", "Bowl Average", "Economy Rate", "Bowl Strike Rate", "Wickets Per Match"
            ],
            "🏆 Bowling Milestones": [
                "5Ws", "5Ws Per Match", "10Ws"
            ],
            "🌟 Awards": [
                "POM", "POM Per Match"
            ]
        }
        
        for category, metrics in metric_categories.items():
            st.markdown(f"<div class='metric-category'>{category}</div>", unsafe_allow_html=True)
            
            for metric in metrics:
                player1_value = player1_stats[metric].values[0]
                player2_value = player2_stats[metric].values[0]
                
                if metric in [
                    "Matches", "Runs", "Bat Average", "Bat Strike Rate", "Balls Per Out",
                    "Runs Per Match", "50+ Scores", "50s", "100s", "50+ Scores Per Match",
                    "100s Per Match", "Overs", "Wickets", "Wickets Per Match", "5Ws",
                    "5Ws Per Match", "10Ws", "POM", "POM Per Match"
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

                if metric in ["100s Per Match", "5Ws Per Match", "POM Per Match", "50+ Scores Per Match"]:
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

        st.markdown("</div></div>", unsafe_allow_html=True)

        # Enhanced Radar Chart
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>📊 Performance Radar</div>", unsafe_allow_html=True)
        
        radar_metrics = ["Bat Average", "Bat Strike Rate", "Balls Per Out", "Bowl Average", "Bowl Strike Rate"]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=[player1_stats[m].values[0] for m in radar_metrics],
            theta=radar_metrics,
            fill='toself',
            name=player1_choice,
            line=dict(color='#f04f53', width=3),
            fillcolor='rgba(240, 79, 83, 0.2)'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=[player2_stats[m].values[0] for m in radar_metrics],
            theta=radar_metrics,
            fill='toself',
            name=player2_choice,
            line=dict(color='#6c24c0', width=3),
            fillcolor='rgba(108, 36, 192, 0.2)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max([max([player1_stats[m].values[0] for m in radar_metrics]),
                                 max([player2_stats[m].values[0] for m in radar_metrics])])],
                    gridcolor='rgba(128, 128, 128, 0.2)',
                    linecolor='rgba(128, 128, 128, 0.2)'
                ),
                bgcolor='rgba(0,0,0,0)',
                angularaxis=dict(
                    gridcolor='rgba(128, 128, 128, 0.2)',
                    linecolor='rgba(128, 128, 128, 0.2)'
                )
            ),
            showlegend=True,
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', size=12),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Enhanced Career Progression Chart
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>📈 Career Progression</div>", unsafe_allow_html=True)
        
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

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Enhanced traces with better styling
        fig.add_trace(
            go.Scatter(x=player1_yearly['Year'], y=player1_yearly['Runs'],
                      name=f"{player1_choice} Runs", 
                      line=dict(color='#f04f53', width=3),
                      mode='lines+markers',
                      marker=dict(size=8)),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=player2_yearly['Year'], y=player2_yearly['Runs'],
                      name=f"{player2_choice} Runs", 
                      line=dict(color='#f04f53', width=3, dash='dash'),
                      mode='lines+markers',
                      marker=dict(size=8)),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=player1_yearly['Year'], y=player1_yearly['Bowler_Wkts'],
                      name=f"{player1_choice} Wickets", 
                      line=dict(color='#6c24c0', width=3),
                      mode='lines+markers',
                      marker=dict(size=8)),
            secondary_y=True,
        )
        
        fig.add_trace(
            go.Scatter(x=player2_yearly['Year'], y=player2_yearly['Bowler_Wkts'],
                      name=f"{player2_choice} Wickets", 
                      line=dict(color='#6c24c0', width=3, dash='dash'),
                      mode='lines+markers',
                      marker=dict(size=8)),
            secondary_y=True,
        )

        fig.update_layout(
            title=dict(
                text="Runs and Wickets by Year",
                font=dict(family='Inter', size=16, color='#f04f53')
            ),
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified',
            font=dict(family='Inter'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            )
        )

        fig.update_xaxes(
            title_text="Year", 
            showgrid=True, 
            gridwidth=1, 
            gridcolor='rgba(128, 128, 128, 0.2)',
            title_font=dict(family='Inter', color='#333')
        )
        fig.update_yaxes(
            title_text="Runs", 
            secondary_y=False, 
            showgrid=True, 
            gridwidth=1, 
            gridcolor='rgba(128, 128, 128, 0.2)',
            title_font=dict(family='Inter', color='#f04f53')
        )
        fig.update_yaxes(
            title_text="Wickets", 
            secondary_y=True, 
            showgrid=True, 
            gridwidth=1, 
            gridcolor='rgba(128, 128, 128, 0.2)',
            title_font=dict(family='Inter', color='#6c24c0')
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #ff6b6b, #ffa500); border-radius: 15px; color: white;'>
            <h2>⚠️ Data Not Available</h2>
            <p>Required data not found. Please ensure you have processed the scorecards.</p>
        </div>
        """, unsafe_allow_html=True)

display_comparison_view()
