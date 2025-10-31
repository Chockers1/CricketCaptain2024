# Imports and Setup
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def display_comparison_view():    # Clean modern styling consistent with home page
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        margin: 0;
    }
    
    .main-header p {
        color: white;
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
      .player-selector {
        background: transparent;
        padding: 0.5rem 0;
        border-radius: 0;
        box-shadow: none;
        margin-bottom: 1rem;
        border: none;
    }
      .comparison-container {
        background: transparent;
        border-radius: 0;
        box-shadow: none;
        overflow: hidden;
        margin: 1rem 0;
        border: none;
    }
    
    .comparison-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
      .player-header {
        color: white;
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 600;
        font-size: 1.3rem;
        flex: 1;
        text-align: center;
    }
      .vs-divider {
        background: white;
        color: #667eea;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.2rem;
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
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
        border-bottom: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    .comparison-row:hover {
        background: rgba(102, 126, 234, 0.02);
        transform: translateY(-1px);
    }
    
    .comparison-row:last-child {
        border-bottom: none;
    }
    
    .comparison-column {
        flex: 1;
        padding: 1rem;
        text-align: center;
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
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
        color: #667eea;
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 0.95rem;
        padding: 1rem;
        background: rgba(102, 126, 234, 0.05);
        margin: 5px 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .highlight-green {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(86, 171, 47, 0.15);
        border: 1px solid rgba(86, 171, 47, 0.2);
    }
    
    .highlight-red {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(255, 107, 107, 0.15);
        border: 1px solid rgba(255, 107, 107, 0.2);
    }
      .chart-container {
        background: transparent;
        border-radius: 0;
        box-shadow: none;
        padding: 1rem;
        margin: 1rem 0;
        border: none;
    }
    
    .chart-title {
        color: #667eea;
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 600;
        font-size: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.2);
    }    /* Use Streamlit's default styling with purple accent */
    :root {
        --primary-color: #764ba2;
        --background-color: #ffffff;
        --secondary-background-color: #f0f2f6;
        --text-color: #262730;
    }
    
    /* Simple purple accent for sliders */
    .stSlider div[data-baseweb="slider"] div[role="slider"] {
        background-color: #764ba2 !important;
    }
    
    .stSlider div[data-baseweb="slider"] div[role="slider"] ~ div {
        background-color: #764ba2 !important;
    }
    
    .metric-category {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.8rem;
        margin: 5px 0;
        text-align: center;
        font-weight: 600;
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.9rem;
    }    .stats-grid {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        gap: 0;
        background: transparent;
        border-radius: 0;
        overflow: hidden;
        box-shadow: none;
        margin: 1rem 0;
    }
    
    /* Remove all default Streamlit container styling */
    div[data-testid="stVerticalBlock"] {
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
    }
    
    div[data-testid="stHorizontalBlock"] {
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
    }
    
    .streamlit-expanderHeader {
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
    }
    
    .streamlit-expanderContent {
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
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

        METRIC_COLUMNS = [
            "Matches", "Runs", "Bat Average", "Bat Strike Rate", "Balls Per Out", "Runs Per Match",
            "50+ Scores", "50s", "100s", "50+ Scores Per Match", "100s Per Match",
            "Overs", "Wickets", "Bowl Average", "Economy Rate", "Bowl Strike Rate", "Wickets Per Match",
            "5Ws", "5Ws Per Match", "10Ws", "POM", "POM Per Match"
        ]
        SUPPORT_COLUMNS = ["Out", "Balls", "Bowler_Runs", "Bowler_Balls"]

        def build_player_stats(bat_source: pd.DataFrame, bowl_source: pd.DataFrame) -> pd.DataFrame:
            bat_subset = bat_source.copy()
            bowl_subset = bowl_source.copy()

            bat_numeric_cols = ['Runs', 'Out', 'Balls', '50s', '100s']
            for col in bat_numeric_cols:
                if col in bat_subset.columns:
                    bat_subset[col] = pd.to_numeric(bat_subset[col], errors='coerce').fillna(0)

            bowl_numeric_cols = ['Bowler_Runs', 'Bowler_Wkts', 'Bowler_Balls', '5Ws']
            for col in bowl_numeric_cols:
                if col in bowl_subset.columns:
                    bowl_subset[col] = pd.to_numeric(bowl_subset[col], errors='coerce').fillna(0)

            match_frames = []
            if {'Name', 'File Name'}.issubset(bat_subset.columns):
                match_frames.append(bat_subset[['Name', 'File Name']])
            if {'Name', 'File Name'}.issubset(bowl_subset.columns):
                match_frames.append(bowl_subset[['Name', 'File Name']])

            if match_frames:
                matches = (
                    pd.concat(match_frames, ignore_index=True)
                    .dropna()
                    .drop_duplicates()
                    .groupby('Name', observed=False)['File Name']
                    .nunique()
                    .reset_index(name='Matches')
                )
            else:
                matches = pd.DataFrame(columns=['Name', 'Matches'])

            batting = pd.DataFrame(columns=['Name', 'Runs', 'Out', 'Balls', '50s', '100s'])
            if {'Name', 'Runs', 'Out', 'Balls', '50s', '100s'}.issubset(bat_subset.columns):
                batting = (
                    bat_subset.groupby('Name', observed=False)[['Runs', 'Out', 'Balls', '50s', '100s']]
                    .sum()
                    .reset_index()
                )

            bowling_filtered = bowl_subset
            if 'Bowler_Balls' in bowling_filtered.columns:
                bowling_filtered = bowling_filtered[bowling_filtered['Bowler_Balls'] > 0]

            bowling = pd.DataFrame(columns=['Name', 'Bowler_Runs', 'Bowler_Wkts', 'Bowler_Balls', '5Ws'])
            if {'Name', 'Bowler_Runs', 'Bowler_Wkts', 'Bowler_Balls'}.issubset(bowling_filtered.columns):
                agg_dict = {
                    'Bowler_Runs': 'sum',
                    'Bowler_Wkts': 'sum',
                    'Bowler_Balls': 'sum'
                }
                if '5Ws' in bowling_filtered.columns:
                    agg_dict['5Ws'] = 'sum'
                bowling = (
                    bowling_filtered
                    .groupby('Name', observed=False)[list(agg_dict.keys())]
                    .sum()
                    .reset_index()
                )
                if '5Ws' not in bowling.columns:
                    bowling['5Ws'] = 0

            ten_w = pd.DataFrame(columns=['Name', '10Ws'])
            if {'Name', 'File Name', 'Bowler_Wkts'}.issubset(bowling_filtered.columns):
                ten_w = (
                    bowling_filtered.groupby(['Name', 'File Name'], observed=False)['Bowler_Wkts']
                    .sum()
                    .ge(10)
                    .groupby('Name', observed=False)
                    .sum()
                    .reset_index(name='10Ws')
                )

            pom = pd.DataFrame(columns=['Name', 'POM'])
            if {'Name', 'File Name', 'Player_of_the_Match'}.issubset(bat_subset.columns):
                pom_series = bat_subset['Player_of_the_Match'].astype('string').fillna('')
                name_series = bat_subset['Name'].astype('string').fillna('')
                pom_mask = pom_series != ''
                pom_mask &= pom_series == name_series
                if pom_mask.any():
                    pom = (
                        bat_subset.loc[pom_mask]
                        .groupby('Name', observed=False)['File Name']
                        .nunique()
                        .reset_index(name='POM')
                    )

            stats = matches
            for df in [batting, bowling, ten_w, pom]:
                stats = stats.merge(df, on='Name', how='outer') if not stats.empty or not df.empty else df

            if stats.empty:
                empty_cols = METRIC_COLUMNS + SUPPORT_COLUMNS
                return pd.DataFrame(columns=empty_cols, index=pd.Index([], name='Name'))

            stats = stats.fillna(0)
            if 'Bowler_Wkts' in stats.columns:
                stats = stats.rename(columns={'Bowler_Wkts': 'Wickets'})
            else:
                stats['Wickets'] = 0

            stats['50+ Scores'] = stats.get('50s', 0) + stats.get('100s', 0)

            stats['Matches'] = pd.to_numeric(stats.get('Matches', 0), errors='coerce').fillna(0).astype(int)
            stats['Runs'] = pd.to_numeric(stats.get('Runs', 0), errors='coerce').fillna(0)
            stats['Out'] = pd.to_numeric(stats.get('Out', 0), errors='coerce').fillna(0)
            stats['Balls'] = pd.to_numeric(stats.get('Balls', 0), errors='coerce').fillna(0)
            stats['50s'] = pd.to_numeric(stats.get('50s', 0), errors='coerce').fillna(0)
            stats['100s'] = pd.to_numeric(stats.get('100s', 0), errors='coerce').fillna(0)
            stats['Wickets'] = pd.to_numeric(stats.get('Wickets', 0), errors='coerce').fillna(0)
            stats['5Ws'] = pd.to_numeric(stats.get('5Ws', 0), errors='coerce').fillna(0)
            stats['10Ws'] = pd.to_numeric(stats.get('10Ws', 0), errors='coerce').fillna(0)
            stats['POM'] = pd.to_numeric(stats.get('POM', 0), errors='coerce').fillna(0)
            stats['Bowler_Runs'] = pd.to_numeric(stats.get('Bowler_Runs', 0), errors='coerce').fillna(0)
            stats['Bowler_Balls'] = pd.to_numeric(stats.get('Bowler_Balls', 0), errors='coerce').fillna(0)

            balls_total = stats['Bowler_Balls'].round().astype(int)
            overs_whole = (balls_total // 6).astype(int)
            overs_partial = (balls_total % 6) / 10
            stats['Overs'] = overs_whole + overs_partial

            with np.errstate(divide='ignore', invalid='ignore'):
                stats['Bat Average'] = np.where(stats['Out'] > 0, stats['Runs'] / stats['Out'], np.inf)
                stats['Balls Per Out'] = np.where(stats['Out'] > 0, stats['Balls'] / stats['Out'], np.inf)
                stats['Bat Strike Rate'] = np.where(stats['Balls'] > 0, (stats['Runs'] / stats['Balls']) * 100, 0)
                stats['Runs Per Match'] = np.where(stats['Matches'] > 0, stats['Runs'] / stats['Matches'], 0)
                stats['Bowl Average'] = np.where(stats['Wickets'] > 0, stats['Bowler_Runs'] / stats['Wickets'], np.inf)
                stats['Economy Rate'] = np.where(stats['Overs'] > 0, stats['Bowler_Runs'] / stats['Overs'], 0)
                stats['Bowl Strike Rate'] = np.where(stats['Wickets'] > 0, stats['Bowler_Balls'] / stats['Wickets'], np.inf)
                stats['Wickets Per Match'] = np.where(stats['Matches'] > 0, stats['Wickets'] / stats['Matches'], 0)
                stats['100s Per Match'] = np.where(stats['Matches'] > 0, (stats['100s'] / stats['Matches']) * 100, 0)
                stats['5Ws Per Match'] = np.where(stats['Matches'] > 0, (stats['5Ws'] / stats['Matches']) * 100, 0)
                stats['50+ Scores Per Match'] = np.where(stats['Matches'] > 0, (stats['50+ Scores'] / stats['Matches']) * 100, 0)
                stats['POM Per Match'] = np.where(stats['Matches'] > 0, (stats['POM'] / stats['Matches']) * 100, 0)

            stats['Bat Average'] = np.round(stats['Bat Average'], 2)
            stats['Balls Per Out'] = np.round(stats['Balls Per Out'], 2)
            stats['Bat Strike Rate'] = np.round(stats['Bat Strike Rate'], 2)
            stats['Runs Per Match'] = np.round(stats['Runs Per Match'], 2)
            stats['Bowl Average'] = np.round(stats['Bowl Average'], 2)
            stats['Economy Rate'] = np.round(stats['Economy Rate'], 2)
            stats['Bowl Strike Rate'] = np.round(stats['Bowl Strike Rate'], 2)
            stats['Wickets Per Match'] = np.round(stats['Wickets Per Match'], 2)
            stats['100s Per Match'] = np.round(stats['100s Per Match'], 2)
            stats['5Ws Per Match'] = np.round(stats['5Ws Per Match'], 2)
            stats['50+ Scores Per Match'] = np.round(stats['50+ Scores Per Match'], 2)
            stats['POM Per Match'] = np.round(stats['POM Per Match'], 2)

            column_order = METRIC_COLUMNS + SUPPORT_COLUMNS
            column_order += [col for col in ['Wickets'] if col not in column_order]
            for col in column_order:
                if col not in stats.columns:
                    stats[col] = 0

            stats = stats[['Name'] + column_order]
            stats = stats.set_index('Name')
            stats = stats.replace([np.nan, -0.0], 0)
            return stats

        player_stats_df = build_player_stats(filtered_bat_df, filtered_bowl_df)
        if player_stats_df.empty:
            base_columns = METRIC_COLUMNS + SUPPORT_COLUMNS + ['Wickets']
            default_stats = pd.Series({col: 0.0 for col in base_columns})
        else:
            default_stats = pd.Series(0.0, index=player_stats_df.columns)

        def get_player_stats(player_name: str) -> pd.Series:
            if player_name in player_stats_df.index:
                return player_stats_df.loc[player_name].reindex(default_stats.index, fill_value=0.0)
            return default_stats.copy()

        player1_stats = get_player_stats(player1_choice)
        player2_stats = get_player_stats(player2_choice)

        def get_metric_value(stats_series: pd.Series, metric: str) -> float:
            value = stats_series.get(metric, 0.0)
            if isinstance(value, (pd.Series, np.ndarray, list, tuple)):
                value = value[0] if len(value) > 0 else 0.0
            if pd.isna(value):
                return 0.0
            return float(value)

        percentage_metrics = {"50+ Scores Per Match", "100s Per Match", "5Ws Per Match", "POM Per Match"}
        integer_metrics = {"Matches", "Runs", "50+ Scores", "50s", "100s", "Wickets", "5Ws", "10Ws", "POM"}
        higher_is_better = {
            "Matches", "Runs", "Bat Average", "Bat Strike Rate", "Balls Per Out", "Runs Per Match",
            "50+ Scores", "50s", "100s", "50+ Scores Per Match", "100s Per Match",
            "Overs", "Wickets", "Wickets Per Match", "5Ws", "5Ws Per Match", "10Ws", "POM", "POM Per Match"
        }

        def format_metric_value(value: float, metric: str) -> str:
            if np.isinf(value):
                return "∞%" if metric in percentage_metrics else "∞"
            if abs(value) < 1e-9:
                value = 0.0
            if metric in integer_metrics:
                return str(int(round(value)))
            if metric in percentage_metrics:
                formatted = f"{value:.2f}".rstrip('0').rstrip('.')
                return f"{formatted}%"
            formatted = f"{value:.2f}".rstrip('0').rstrip('.')
            return formatted

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
                player1_value = get_metric_value(player1_stats, metric)
                player2_value = get_metric_value(player2_stats, metric)

                if np.isinf(player1_value) and np.isinf(player2_value):
                    values_equal = True
                elif np.isinf(player1_value) or np.isinf(player2_value):
                    values_equal = False
                else:
                    values_equal = np.isclose(player1_value, player2_value, equal_nan=True)

                player1_class = ""
                player2_class = ""

                if metric in higher_is_better:
                    if not values_equal:
                        player1_class = "highlight-green" if player1_value > player2_value else "highlight-red"
                        player2_class = "highlight-green" if player2_value > player1_value else "highlight-red"
                else:
                    if not values_equal:
                        player1_class = "highlight-green" if player1_value < player2_value else "highlight-red"
                        player2_class = "highlight-green" if player2_value < player1_value else "highlight-red"

                if metric == "Economy Rate":
                    if get_metric_value(player1_stats, "Overs") == 0:
                        player1_class = "highlight-red"
                    if get_metric_value(player2_stats, "Overs") == 0:
                        player2_class = "highlight-red"

                p1_display = format_metric_value(player1_value, metric)
                p2_display = format_metric_value(player2_value, metric)

                st.markdown(f"""
                <div class='comparison-row'>
                    <div class='comparison-column {player1_class}'>{p1_display}</div>
                    <div class='comparison-metric'>{metric}</div>
                    <div class='comparison-column {player2_class}'>{p2_display}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

        # Comparison Summary Section
        st.markdown("<div class='comparison-container'>", unsafe_allow_html=True)
        st.markdown("""
        <div class='comparison-header'>
            <div class='player-header'>Category Comparison Summary</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='comparison-body'>", unsafe_allow_html=True)
        
        # Calculate category scores
        def calculate_category_scores():
            scores = {
                "Batting Performance": [0, 0],
                "Batting Milestones": [0, 0], 
                "Bowling Performance": [0, 0],
                "Bowling Milestones": [0, 0],
                "Awards": [0, 0]
            }
            
            def compare_high(metric: str, category: str) -> None:
                p1_val = get_metric_value(player1_stats, metric)
                p2_val = get_metric_value(player2_stats, metric)
                if np.isinf(p1_val) and np.isinf(p2_val):
                    return
                if not (np.isinf(p1_val) or np.isinf(p2_val)):
                    if np.isclose(p1_val, p2_val, equal_nan=True):
                        return
                if p1_val > p2_val:
                    scores[category][0] += 1
                elif p2_val > p1_val:
                    scores[category][1] += 1

            def compare_low(metric: str, category: str) -> None:
                p1_val = get_metric_value(player1_stats, metric)
                p2_val = get_metric_value(player2_stats, metric)

                if metric == "Economy Rate":
                    overs1 = get_metric_value(player1_stats, "Overs")
                    overs2 = get_metric_value(player2_stats, "Overs")
                    if overs1 == 0 and overs2 == 0:
                        return
                    if overs1 == 0:
                        scores[category][1] += 1
                        return
                    if overs2 == 0:
                        scores[category][0] += 1
                        return

                if np.isinf(p1_val) and np.isinf(p2_val):
                    return
                if np.isinf(p1_val):
                    scores[category][1] += 1
                    return
                if np.isinf(p2_val):
                    scores[category][0] += 1
                    return
                if np.isclose(p1_val, p2_val, equal_nan=True):
                    return
                if p1_val < p2_val:
                    scores[category][0] += 1
                elif p2_val < p1_val:
                    scores[category][1] += 1

            # Batting Performance metrics (higher is better)
            batting_perf_metrics = ["Runs", "Bat Average", "Bat Strike Rate", "Balls Per Out", "Runs Per Match"]
            for metric in batting_perf_metrics:
                compare_high(metric, "Batting Performance")
            
            # Batting Milestones (higher is better)
            batting_milestone_metrics = ["50+ Scores", "50s", "100s", "50+ Scores Per Match", "100s Per Match"]
            for metric in batting_milestone_metrics:
                compare_high(metric, "Batting Milestones")
            
            # Bowling Performance metrics (mix of higher/lower is better)
            bowling_perf_metrics = ["Overs", "Wickets", "Bowl Average", "Economy Rate", "Bowl Strike Rate", "Wickets Per Match"]
            for metric in bowling_perf_metrics:
                if metric in ["Overs", "Wickets", "Wickets Per Match"]:
                    compare_high(metric, "Bowling Performance")
                else:
                    compare_low(metric, "Bowling Performance")
            
            # Bowling Milestones (higher is better)
            bowling_milestone_metrics = ["5Ws", "5Ws Per Match", "10Ws"]
            for metric in bowling_milestone_metrics:
                compare_high(metric, "Bowling Milestones")
            
            # Awards (higher is better)
            award_metrics = ["POM", "POM Per Match"]
            for metric in award_metrics:
                compare_high(metric, "Awards")
            
            return scores
        
        category_scores = calculate_category_scores()
        
        # Display category comparisons
        for category, (p1_score, p2_score) in category_scores.items():
            p1_class = "highlight-green" if p1_score > p2_score else ("highlight-red" if p1_score < p2_score else "")
            p2_class = "highlight-green" if p2_score > p1_score else ("highlight-red" if p2_score < p1_score else "")
            
            st.markdown(f"""
            <div class='comparison-row'>
                <div class='comparison-column {p1_class}'>{p1_score}</div>
                <div class='comparison-metric'>{category}</div>
                <div class='comparison-column {p2_class}'>{p2_score}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Combined totals
        p1_total = category_scores["Batting Performance"][0] + category_scores["Bowling Performance"][0]
        p2_total = category_scores["Batting Performance"][1] + category_scores["Bowling Performance"][1]
        
        p1_combined_class = "highlight-green" if p1_total > p2_total else ("highlight-red" if p1_total < p2_total else "")
        p2_combined_class = "highlight-green" if p2_total > p1_total else ("highlight-red" if p2_total < p1_total else "")
        
        st.markdown(f"""
        <div class='comparison-row'>
            <div class='comparison-column {p1_combined_class}'>{p1_total}</div>
            <div class='comparison-metric'>Combined Performance</div>
            <div class='comparison-column {p2_combined_class}'>{p2_total}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)

        # Enhanced Radar Chart
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>📊 Performance Radar</div>", unsafe_allow_html=True)
        
        radar_metrics = ["Bat Average", "Bat Strike Rate", "Balls Per Out", "Bowl Average", "Bowl Strike Rate"]
        
        radar_player1 = [get_metric_value(player1_stats, metric) for metric in radar_metrics]
        radar_player2 = [get_metric_value(player2_stats, metric) for metric in radar_metrics]

        radar_player1 = [value if np.isfinite(value) else 0 for value in radar_player1]
        radar_player2 = [value if np.isfinite(value) else 0 for value in radar_player2]

        combined_radar_values = radar_player1 + radar_player2
        max_radar_value = max(combined_radar_values) if combined_radar_values else 1
        if max_radar_value <= 0:
            max_radar_value = 1
        radial_max = max_radar_value * 1.1

        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=radar_player1,
            theta=radar_metrics,
            fill='toself',
            name=player1_choice,
            line=dict(color='#667eea', width=3),
            fillcolor='rgba(102, 126, 234, 0.2)'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=radar_player2,
            theta=radar_metrics,
            fill='toself',
            name=player2_choice,
            line=dict(color='#764ba2', width=3),
            fillcolor='rgba(118, 75, 162, 0.2)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, radial_max],
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
            font=dict(family='SF Pro Display', size=12),
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

        fig = make_subplots(specs=[[{"secondary_y": True}]])        # Enhanced traces with better styling using Home.py colors
        fig.add_trace(
            go.Scatter(x=player1_yearly['Year'], y=player1_yearly['Runs'],
                      name=f"{player1_choice} Runs", 
                      line=dict(color='#667eea', width=3),
                      mode='lines+markers',
                      marker=dict(size=8)),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=player2_yearly['Year'], y=player2_yearly['Runs'],
                      name=f"{player2_choice} Runs", 
                      line=dict(color='#764ba2', width=3, dash='dash'),
                      mode='lines+markers',
                      marker=dict(size=8)),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=player1_yearly['Year'], y=player1_yearly['Bowler_Wkts'],
                      name=f"{player1_choice} Wickets", 
                      line=dict(color='#f093fb', width=3),
                      mode='lines+markers',
                      marker=dict(size=8)),
            secondary_y=True,
        )
        
        fig.add_trace(
            go.Scatter(x=player2_yearly['Year'], y=player2_yearly['Bowler_Wkts'],
                      name=f"{player2_choice} Wickets", 
                      line=dict(color='#f5576c', width=3, dash='dash'),
                      mode='lines+markers',
                      marker=dict(size=8)),
            secondary_y=True,
        )

        fig.update_layout(
            title=dict(
                text="Runs and Wickets by Year",
                font=dict(family='SF Pro Display', size=16, color='#4e73df')
            ),
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified',
            font=dict(family='SF Pro Display'),
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
            title_font=dict(family='SF Pro Display', color='#5a5c69')
        )
        fig.update_yaxes(
            title_text="Runs", 
            secondary_y=False, 
            showgrid=True, 
            gridwidth=1, 
            gridcolor='rgba(128, 128, 128, 0.2)',
            title_font=dict(family='SF Pro Display', color='#4e73df')
        )
        fig.update_yaxes(
            title_text="Wickets", 
            secondary_y=True, 
            showgrid=True, 
            gridwidth=1, 
            gridcolor='rgba(128, 128, 128, 0.2)',
            title_font=dict(family='SF Pro Display', color='#1cc88a')
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
