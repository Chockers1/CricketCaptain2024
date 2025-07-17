import streamlit as st 
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# Modern CSS for beautiful UI - Enhanced styling
def apply_modern_styling():
    """Apply modern CSS styling to the page"""
    st.markdown("""
    <style>
        .main > div {
            padding-top: 1rem;
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
        
        .metrics-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 15px;
            margin: 15px 0;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            color: white;
        }
        
        .achievement-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            margin: 10px 0;
            box-shadow: 0 6px 20px rgba(0,0,0,0.1);
            color: white;
            font-weight: bold;
        }
        
        h1, h2, h3, h4 {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            font-weight: 700;
        }
        
        .main-title {
            background: linear-gradient(135deg, #f04f53, #667eea);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 2rem;
        }
        
        .section-header {
            background: linear-gradient(135deg, #f04f53, #f5576c);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            font-size: 1.8rem;
            font-weight: 700;
            margin: 2rem 0 1rem 0;
        }
        
        table { 
            color: black; 
            width: 100%; 
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        thead tr th {
            background: linear-gradient(135deg, #f04f53 0%, #f5576c 100%) !important;
            color: white !important;
            font-weight: bold;
            padding: 12px;
            text-align: center;
        }
        
        tbody tr:nth-child(even) { 
            background-color: #f8f9fa; 
        }
        tbody tr:nth-child(odd) { 
            background-color: white; 
        }
        tbody tr:hover {
            background: linear-gradient(135deg, #fff0f1 0%, #f8f9fa 100%);
            transition: all 0.3s ease;
            transform: scale(1.01);
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
        
        .stNumberInput > div > div {
            border-radius: 8px;
            border: 2px solid #f0f2f6;
            transition: all 0.3s ease;
        }
        
        .stNumberInput > div > div:focus-within {
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
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            border-radius: 15px;
            padding: 8px;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .stTabs [data-baseweb="tab"] {
            flex-grow: 1;
            text-align: center;
            background: transparent;
            color: #2c3e50;
            font-weight: 700;
            border-radius: 10px;
            margin: 2px;
            transition: all 0.3s ease;
            padding: 12px 16px;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(255,255,255,0.4);
            transform: translateY(-1px);
        }
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #f04f53 0%, #f5576c 100%);
            color: white;
            box-shadow: 0 4px 12px rgba(240,79,83,0.3);
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
        
        /* Animation for elements */
        .stDataFrame, .metric-card, .modern-card {
            animation: fadeInUp 0.6s ease-out;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Rating indicators */
        .rating-high {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            padding: 4px 8px;
            border-radius: 6px;
            font-weight: bold;
        }
        
        .rating-medium {
            background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
            color: black;
            padding: 4px 8px;
            border-radius: 6px;
            font-weight: bold;
        }
        
        .rating-low {
            background: linear-gradient(135deg, #dc3545 0%, #e83e8c 100%);
            color: white;
            padding: 4px 8px;
            border-radius: 6px;
            font-weight: bold;
        }
        
        /* Container styling */
        .block-container {
            max-width: 100%;
            padding-top: 1rem;
            padding-right: 1rem;
            padding-left: 1rem;
            padding-bottom: 1rem;
        }
        
        /* Custom spacing */
        .custom-spacer {
            height: 2rem;
        }
        
    </style>
    """, unsafe_allow_html=True)

def calculate_batter_rating_per_match(df):
    """Calculate batting rating with bonuses"""
    stats = df.copy()
    stats['Strike_Rate'] = (stats['Runs'] / stats['Balls']) * 100
    stats['Base_Score'] = stats['Runs']
    
    def calculate_bonus(row):
        runs = row['Runs']
        sr = row['Strike_Rate']
        bonus = 0
        
        if 100 <= runs < 150:
            bonus += 50  # Century bonus
        elif 150 <= runs < 200:
            bonus += 75  # 150+ bonus
        elif runs >= 200:
            bonus += 100  # Double century bonus
            
        if runs >= 40:  # Only apply SR bonus/penalty for substantial innings
            if sr >= 75:
                bonus += 25  # Good scoring rate bonus
            elif sr <= 40:
                bonus -= 25  # Slow scoring penalty
                
        return bonus
    
    stats['Bonus'] = stats.apply(calculate_bonus, axis=1)
    stats['Batting_Rating'] = stats['Base_Score'] + stats['Bonus']
    return stats

def calculate_bowler_rating_per_match(df):
    """Calculate bowling rating with bonuses"""
    stats = df.copy()
    stats['Overs'] = stats['Bowler_Balls'] / 6
    stats['Economy'] = (stats['Bowler_Runs'] / stats['Overs']).replace(float('inf'), 0)
    stats['Base_Score'] = (stats['Bowler_Wkts'] * 20) + (stats['Maidens'] * 2)
    
    def calculate_wicket_bonus(wickets):
        if wickets == 10:
            return 260
        elif wickets == 9:
            return 200
        elif wickets == 8:
            return 150
        elif wickets == 7:
            return 100
        elif wickets == 6:
            return 75
        elif wickets == 5:
            return 50
        elif wickets == 4:
            return 35
        elif wickets == 3:
            return 20
        return 0
    
    def calculate_economy_bonus(row):
        if row['Overs'] >= 10:
            if row['Economy'] <= 2.5:
                return 25
            elif row['Economy'] >= 4.5:
                return -25
        return 0
    
    stats['Wicket_Bonus'] = stats['Bowler_Wkts'].apply(calculate_wicket_bonus)
    stats['Economy_Bonus'] = stats.apply(calculate_economy_bonus, axis=1)
    stats['Bowling_Rating'] = stats['Base_Score'] + stats['Wicket_Bonus'] + stats['Economy_Bonus']
    
    return stats

def calculate_peak_ratings(rankings_df):
    """Calculate peak ratings and duration at peak for each player"""
    peak_data = rankings_df.groupby('Name').agg({
        'Rating': ['max', 'mean'],
        'Year': ['min', 'max', 'count']
    }).reset_index()
    
    peak_data.columns = ['Name', 'Peak_Rating', 'Avg_Rating', 'Start_Year', 'End_Year', 'Years_Active']
    return peak_data

def display_number_one_rankings(bat_df, bowl_df):
    """Display #1 Rankings tab content"""
    
    # Get all unique formats from both dataframes
    all_formats = sorted(set(bat_df['Match_Format'].unique()) | set(bowl_df['Match_Format'].unique()))
    
    # Use the format filter from session state if it exists
    selected_format = st.session_state.get('global_format_filter', [])

    # Filter dataframes by format if selected
    filtered_bat_df = bat_df.copy()
    filtered_bowl_df = bowl_df.copy()
    
    if selected_format:
        filtered_bat_df = filtered_bat_df[filtered_bat_df['Match_Format'].isin(selected_format)]
        filtered_bowl_df = filtered_bowl_df[filtered_bowl_df['Match_Format'].isin(selected_format)]

    # Original yearly best performers section with modern styling
    st.markdown("""
        <div style="text-align: center; margin: 30px 0;">
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                       padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                <h2 style="margin: 0; font-size: 1.8em; font-weight: bold;">
                    🎯 Yearly Best Performers
                </h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                    Top performers in each category by year
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Calculate yearly bests using filtered data with robust error handling
    try:
        # Initialize empty dataframes
        best_batting = pd.DataFrame(columns=['Year', 'Match_Format', 'Best_Batting'])
        best_bowling = pd.DataFrame(columns=['Year', 'Match_Format', 'Best_Bowling'])
        best_ar = pd.DataFrame(columns=['Year', 'Match_Format', 'Best_AllRounder'])
        yearly_ar = pd.DataFrame()

        # Best batting by year
        if not filtered_bat_df.empty:
            clean_bat = filtered_bat_df.dropna(subset=['Year', 'Match_Format', 'Name', 'Batting_Rating'])
            clean_bat = clean_bat[clean_bat['Name'].str.strip() != '']
            if not clean_bat.empty:
                batting_by_player = clean_bat.groupby(['Year', 'Match_Format', 'Name'])['Batting_Rating'].sum().reset_index()
                best_batting = batting_by_player.groupby(['Year', 'Match_Format']).apply(
                    lambda x: f"{x.loc[x['Batting_Rating'].idxmax(), 'Name']} - {x['Batting_Rating'].max():.0f}" 
                    if x['Batting_Rating'].max() > 0 else "-"
                ).reset_index(name='Best_Batting')

        # Best bowling by year
        if not filtered_bowl_df.empty:
            clean_bowl = filtered_bowl_df.dropna(subset=['Year', 'Match_Format', 'Name', 'Bowling_Rating'])
            clean_bowl = clean_bowl[clean_bowl['Name'].str.strip() != '']
            if not clean_bowl.empty:
                bowling_by_player = clean_bowl.groupby(['Year', 'Match_Format', 'Name'])['Bowling_Rating'].sum().reset_index()
                best_bowling = bowling_by_player.groupby(['Year', 'Match_Format']).apply(
                    lambda x: f"{x.loc[x['Bowling_Rating'].idxmax(), 'Name']} - {x['Bowling_Rating'].max():.0f}" 
                    if x['Bowling_Rating'].max() > 0 else "-"
                ).reset_index(name='Best_Bowling')

        # Calculate AR ratings with Match_Format
        if 'batting_by_player' in locals() and 'bowling_by_player' in locals() and not batting_by_player.empty and not bowling_by_player.empty:
            yearly_ar = pd.merge(
                batting_by_player,
                bowling_by_player,
                on=['Year', 'Match_Format', 'Name'],
                how='outer'
            ).fillna(0)
            
            yearly_ar['AR_Rating'] = yearly_ar['Batting_Rating'] + yearly_ar['Bowling_Rating']
            best_ar = yearly_ar.groupby(['Year', 'Match_Format']).apply(
                lambda x: f"{x.loc[x['AR_Rating'].idxmax(), 'Name']} - {x['AR_Rating'].max():.0f}" 
                if x['AR_Rating'].max() > 0 else "-"
            ).reset_index(name='Best_AllRounder')

        # Combine summaries safely
        if not best_batting.empty or not best_bowling.empty or not best_ar.empty:
            yearly_summary = pd.merge(best_batting, best_bowling, on=['Year', 'Match_Format'], how='outer')
            yearly_summary = pd.merge(yearly_summary, best_ar, on=['Year', 'Match_Format'], how='outer')
            yearly_summary = yearly_summary.fillna("-")
            yearly_summary = yearly_summary.sort_values(['Year', 'Match_Format'], ascending=[False, True])

            if not yearly_summary.empty:
                st.dataframe(
                    yearly_summary,
                    use_container_width=True,
                    hide_index=True,
                    height=min(400, (len(yearly_summary) + 1) * 35),
                    column_config={
                        "Year": st.column_config.NumberColumn("📅 Year", format="%d"),
                        "Match_Format": st.column_config.TextColumn("📊 Format", width="small"),
                        "Best_Batting": st.column_config.TextColumn("🏏 Batting Champion", width="large"),
                        "Best_Bowling": st.column_config.TextColumn("⚡ Bowling Champion", width="large"), 
                        "Best_AllRounder": st.column_config.TextColumn("🌟 All-Rounder Champion", width="large")
                    }
                )
            else:
                st.info("No yearly summary data available for the selected filters.")
        else:
            st.info("No yearly summary data available for the selected filters.")
            batting_by_player = pd.DataFrame()
            bowling_by_player = pd.DataFrame()
            yearly_ar = pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error calculating yearly summary: {str(e)}")
        batting_by_player = pd.DataFrame()
        bowling_by_player = pd.DataFrame()
        yearly_ar = pd.DataFrame()
    st.markdown('</div>', unsafe_allow_html=True)

    # Peak Achievements with modern styling
    st.markdown("""
        <div style="text-align: center; margin: 30px 0;">
            <div style="background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%); 
                       padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                <h2 style="margin: 0; font-size: 1.8em; font-weight: bold;">
                    ⭐ Peak Rating Achievements
                </h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                    Highest ratings achieved by each player
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Calculate peak ratings by format with robust error handling
    try:
        batting_peaks = pd.DataFrame(columns=['Name', 'Match_Format', 'Batting_Rating'])
        bowling_peaks = pd.DataFrame(columns=['Name', 'Match_Format', 'Bowling_Rating'])
        ar_peaks = pd.DataFrame(columns=['Name', 'Match_Format', 'AR_Rating'])

        if not batting_by_player.empty and 'Batting_Rating' in batting_by_player.columns:
            clean_batting = batting_by_player.dropna(subset=['Name', 'Match_Format', 'Batting_Rating'])
            clean_batting = clean_batting[clean_batting['Name'].str.strip() != '']
            if not clean_batting.empty:
                batting_peaks = clean_batting.groupby(['Name', 'Match_Format'])['Batting_Rating'].max().reset_index()

        if not bowling_by_player.empty and 'Bowling_Rating' in bowling_by_player.columns:
            clean_bowling = bowling_by_player.dropna(subset=['Name', 'Match_Format', 'Bowling_Rating'])
            clean_bowling = clean_bowling[clean_bowling['Name'].str.strip() != '']
            if not clean_bowling.empty:
                bowling_peaks = clean_bowling.groupby(['Name', 'Match_Format'])['Bowling_Rating'].max().reset_index()

        if not yearly_ar.empty and 'AR_Rating' in yearly_ar.columns:
            clean_ar = yearly_ar.dropna(subset=['Name', 'Match_Format', 'AR_Rating'])
            clean_ar = clean_ar[clean_ar['Name'].str.strip() != '']
            if not clean_ar.empty:
                ar_peaks = clean_ar.groupby(['Name', 'Match_Format'])['AR_Rating'].max().reset_index()

        # Combine peak ratings safely
        if not batting_peaks.empty or not bowling_peaks.empty or not ar_peaks.empty:
            peaks = pd.merge(batting_peaks, bowling_peaks, on=['Name', 'Match_Format'], how='outer', suffixes=('_Bat', '_Bowl'))
            peaks = pd.merge(peaks, ar_peaks, on=['Name', 'Match_Format'], how='outer')
            peaks = peaks.fillna(0)
            
            # Clean and validate final data
            peaks = peaks.dropna(subset=['Name', 'Match_Format'])
            peaks = peaks[peaks['Name'].str.strip() != '']
            
            if not peaks.empty:
                # Ensure numeric columns
                numeric_cols = ['Batting_Rating_Bat', 'Bowling_Rating_Bowl', 'AR_Rating']
                for col in numeric_cols:
                    if col in peaks.columns:
                        peaks[col] = pd.to_numeric(peaks[col], errors='coerce').fillna(0)
                
                peaks = peaks.sort_values(['Match_Format', 'AR_Rating'], ascending=[True, False])

                st.dataframe(
                    peaks,
                    use_container_width=True,
                    hide_index=True,
                    height=min(400, (len(peaks) + 1) * 35),
                    column_config={
                        "Name": st.column_config.TextColumn("🏆 Player", width="medium"),
                        "Match_Format": st.column_config.TextColumn("📊 Format", width="small"),
                        "Batting_Rating_Bat": st.column_config.NumberColumn("🏏 Peak Batting Rating", format="%.0f"),
                        "Bowling_Rating_Bowl": st.column_config.NumberColumn("⚡ Peak Bowling Rating", format="%.0f"),
                        "AR_Rating": st.column_config.NumberColumn("🌟 Peak All-Rounder Rating", format="%.0f")
                    }
                )
            else:
                st.info("No peak rating data available for the selected filters.")
        else:
            st.info("No peak rating data available for the selected filters.")
            
    except Exception as e:
        st.error(f"Error calculating peak ratings: {str(e)}")

    # Number of times ranked #1 with modern styling
    st.markdown("""
        <div style="text-align: center; margin: 30px 0;">
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                       padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                <h2 style="margin: 0; font-size: 1.8em; font-weight: bold;">
                    👑 Number of Times Ranked #1
                </h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                    Championship titles across all categories
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Calculate #1 rankings by format with robust error handling
    try:
        # Initialize empty dataframes with proper structure
        batting_no1 = pd.DataFrame(columns=['Name', 'Match_Format', 'Batting_#1'])
        bowling_no1 = pd.DataFrame(columns=['Name', 'Match_Format', 'Bowling_#1'])
        ar_no1 = pd.DataFrame(columns=['Name', 'Match_Format', 'AllRounder_#1'])
        
        # Ensure dataframes exist and have required columns
        if not batting_by_player.empty and 'Batting_Rating' in batting_by_player.columns and 'Name' in batting_by_player.columns:
            batting_tops = batting_by_player.copy()
            # Clean data before ranking
            batting_tops = batting_tops.dropna(subset=['Name', 'Year', 'Match_Format', 'Batting_Rating'])
            batting_tops = batting_tops[batting_tops['Name'].str.strip() != '']
            if not batting_tops.empty:
                batting_tops['Rank'] = batting_tops.groupby(['Year', 'Match_Format'])['Batting_Rating'].rank(method='min', ascending=False)
                batting_no1 = batting_tops[batting_tops['Rank'] == 1].groupby(['Name', 'Match_Format']).size().reset_index(name='Batting_#1')

        if not bowling_by_player.empty and 'Bowling_Rating' in bowling_by_player.columns and 'Name' in bowling_by_player.columns:
            bowling_tops = bowling_by_player.copy()
            # Clean data before ranking
            bowling_tops = bowling_tops.dropna(subset=['Name', 'Year', 'Match_Format', 'Bowling_Rating'])
            bowling_tops = bowling_tops[bowling_tops['Name'].str.strip() != '']
            if not bowling_tops.empty:
                bowling_tops['Rank'] = bowling_tops.groupby(['Year', 'Match_Format'])['Bowling_Rating'].rank(method='min', ascending=False)
                bowling_no1 = bowling_tops[bowling_tops['Rank'] == 1].groupby(['Name', 'Match_Format']).size().reset_index(name='Bowling_#1')

        if not yearly_ar.empty and 'AR_Rating' in yearly_ar.columns and 'Name' in yearly_ar.columns:
            ar_tops = yearly_ar.copy()
            # Clean data before ranking
            ar_tops = ar_tops.dropna(subset=['Name', 'Year', 'Match_Format', 'AR_Rating'])
            ar_tops = ar_tops[ar_tops['Name'].str.strip() != '']
            if not ar_tops.empty:
                ar_tops['Rank'] = ar_tops.groupby(['Year', 'Match_Format'])['AR_Rating'].rank(method='min', ascending=False)
                ar_no1 = ar_tops[ar_tops['Rank'] == 1].groupby(['Name', 'Match_Format']).size().reset_index(name='AllRounder_#1')

        # Combine all #1 rankings with safe merging
        if not batting_no1.empty or not bowling_no1.empty or not ar_no1.empty:
            all_no1 = pd.merge(batting_no1, bowling_no1, on=['Name', 'Match_Format'], how='outer')
            all_no1 = pd.merge(all_no1, ar_no1, on=['Name', 'Match_Format'], how='outer')
            all_no1 = all_no1.fillna(0)
            
            # Ensure columns exist and are properly typed
            required_cols = ['Batting_#1', 'Bowling_#1', 'AllRounder_#1']
            for col in required_cols:
                if col not in all_no1.columns:
                    all_no1[col] = 0
                all_no1[col] = pd.to_numeric(all_no1[col], errors='coerce').fillna(0).astype(int)
            
            all_no1['Total_#1'] = all_no1['Batting_#1'] + all_no1['Bowling_#1'] + all_no1['AllRounder_#1']
            all_no1 = all_no1.sort_values(['Match_Format', 'Total_#1'], ascending=[True, False])

            # Final data validation before display
            if not all_no1.empty and 'Name' in all_no1.columns:
                all_no1 = all_no1.dropna(subset=['Name'])
                all_no1 = all_no1[all_no1['Name'].str.strip() != '']
                
                if not all_no1.empty:
                    st.dataframe(
                        all_no1,
                        use_container_width=True,
                        hide_index=True,
                        height=min(400, (len(all_no1) + 1) * 35),
                        column_config={
                            "Name": st.column_config.TextColumn("🏆 Player", width="medium"),
                            "Match_Format": st.column_config.TextColumn("📊 Format", width="small"),
                            "Batting_#1": st.column_config.NumberColumn("🏏 Batting #1", format="%d"),
                            "Bowling_#1": st.column_config.NumberColumn("⚡ Bowling #1", format="%d"), 
                            "AllRounder_#1": st.column_config.NumberColumn("🌟 All-Rounder #1", format="%d"),
                            "Total_#1": st.column_config.NumberColumn("👑 Total #1", format="%d")
                        }
                    )
                else:
                    st.info("No #1 ranking data available for the selected filters.")
                    all_no1 = pd.DataFrame()  # Reset for Hall of Fame section
            else:
                st.info("No #1 ranking data available for the selected filters.")
                all_no1 = pd.DataFrame()  # Reset for Hall of Fame section
        else:
            st.info("No #1 ranking data available for the selected filters.")
            all_no1 = pd.DataFrame()  # Create empty dataframe for Hall of Fame section
            
    except Exception as e:
        st.error(f"Error calculating #1 rankings: {str(e)}")
        all_no1 = pd.DataFrame()  # Create empty dataframe for Hall of Fame section

    # Hall of Fame Criteria with modern styling
    st.markdown("""
        <div style="text-align: center; margin: 30px 0;">
            <div style="background: linear-gradient(135deg, #a8caba 0%, #5d4e75 100%); 
                       padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                <h2 style="margin: 0; font-size: 1.8em; font-weight: bold;">
                    🏛️ Hall of Fame Status
                </h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                    Elite performers recognition based on career rating points
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Hall of Fame thresholds by format
    HOF_THRESHOLDS = {
        'hall_of_fame': 7500,      # Standard HOF threshold
        'elite': 10000,             # Elite status
        'legendary': 12500          # Legendary status
    }
    
    def get_hof_status(points):
        """Determine Hall of Fame status based on total rating points"""
        if points >= HOF_THRESHOLDS['legendary']:
            return "🌟 Legend", "legendary"
        elif points >= HOF_THRESHOLDS['elite']:
            return "💎 Elite", "elite"
        elif points >= HOF_THRESHOLDS['hall_of_fame']:
            return "🏛️ Hall of Fame", "hall_of_fame"
        else:
            return "📊 Active", "active"

    # Calculate Hall of Fame status based on total career rating points
    try:
        # Calculate career totals for each player by format
        hof_data = []
        
        # Process batting data
        if not batting_by_player.empty and 'Batting_Rating' in batting_by_player.columns:
            batting_career = batting_by_player.groupby(['Name', 'Match_Format'])['Batting_Rating'].sum().reset_index()
            batting_career = batting_career.rename(columns={'Batting_Rating': 'Batting_Points'})
            hof_data.append(batting_career)
        
        # Process bowling data  
        if not bowling_by_player.empty and 'Bowling_Rating' in bowling_by_player.columns:
            bowling_career = bowling_by_player.groupby(['Name', 'Match_Format'])['Bowling_Rating'].sum().reset_index()
            bowling_career = bowling_career.rename(columns={'Bowling_Rating': 'Bowling_Points'})
            hof_data.append(bowling_career)
            
        # Process all-rounder data
        if not yearly_ar.empty and 'AR_Rating' in yearly_ar.columns:
            ar_career = yearly_ar.groupby(['Name', 'Match_Format'])['AR_Rating'].sum().reset_index()
            ar_career = ar_career.rename(columns={'AR_Rating': 'AllRounder_Points'})
            hof_data.append(ar_career)

        if hof_data:
            # Merge all career totals
            career_totals = hof_data[0]
            for df in hof_data[1:]:
                career_totals = pd.merge(career_totals, df, on=['Name', 'Match_Format'], how='outer')
            
            # Fill NaN values with 0
            career_totals = career_totals.fillna(0)
            
            # Calculate total points for each discipline
            if 'Batting_Points' not in career_totals.columns:
                career_totals['Batting_Points'] = 0
            if 'Bowling_Points' not in career_totals.columns:
                career_totals['Bowling_Points'] = 0
            if 'AllRounder_Points' not in career_totals.columns:
                career_totals['AllRounder_Points'] = 0
                
            # Determine highest category achieved for overall HOF status
            career_totals['Max_Points'] = career_totals[['Batting_Points', 'Bowling_Points', 'AllRounder_Points']].max(axis=1)
            career_totals[['HOF_Status', 'HOF_Level']] = career_totals['Max_Points'].apply(
                lambda x: pd.Series(get_hof_status(x))
            )
            
            # Add specific discipline HOF status
            career_totals[['Batting_HOF', 'Batting_Level']] = career_totals['Batting_Points'].apply(
                lambda x: pd.Series(get_hof_status(x))
            )
            career_totals[['Bowling_HOF', 'Bowling_Level']] = career_totals['Bowling_Points'].apply(
                lambda x: pd.Series(get_hof_status(x))
            )
            career_totals[['AR_HOF', 'AR_Level']] = career_totals['AllRounder_Points'].apply(
                lambda x: pd.Series(get_hof_status(x))
            )
            
            # Filter to show only HOF players (at least 10k points in any discipline)
            hof_players = career_totals[career_totals['Max_Points'] >= HOF_THRESHOLDS['hall_of_fame']]
            
            if not hof_players.empty:
                # Sort by format and max points
                hof_players = hof_players.sort_values(['Match_Format', 'Max_Points'], ascending=[True, False])
                
                # Display HOF explanation
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                               padding: 15px; border-radius: 10px; margin: 20px 0; color: white;">
                        <h4 style="margin: 0 0 10px 0;">🏛️ Hall of Fame Criteria</h4>
                        <p style="margin: 0; font-size: 0.9em;">
                            <strong>🏛️ Hall of Fame:</strong> 7500+ career rating points | 
                            <strong>💎 Elite:</strong> 10,000+ points | 
                            <strong>🌟 Legendary:</strong> 12,500+ points
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.dataframe(
                    hof_players[['Name', 'Match_Format', 'Batting_Points', 'Bowling_Points', 
                               'AllRounder_Points', 'Max_Points', 'HOF_Status']],
                    use_container_width=True,
                    hide_index=True,
                    height=min(500, (len(hof_players) + 1) * 35),
                    column_config={
                        "Name": st.column_config.TextColumn("�️ Player", width="medium"),
                        "Match_Format": st.column_config.TextColumn("📊 Format", width="small"),
                        "Batting_Points": st.column_config.NumberColumn("🏏 Batting Points", format="%.0f"),
                        "Bowling_Points": st.column_config.NumberColumn("⚡ Bowling Points", format="%.0f"),
                        "AllRounder_Points": st.column_config.NumberColumn("🌟 AR Points", format="%.0f"),
                        "Max_Points": st.column_config.NumberColumn("🎯 Highest Points", format="%.0f"),
                        "HOF_Status": st.column_config.TextColumn("�️ HOF Status", width="medium")
                    }
                )
                
                # Show summary stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_hof = len(hof_players)
                    st.metric("🏛️ Total HOF Players", total_hof)
                with col2:
                    elite_count = len(hof_players[hof_players['HOF_Level'] == 'elite'])
                    st.metric("💎 Elite Players", elite_count)
                with col3:
                    legend_count = len(hof_players[hof_players['HOF_Level'] == 'legendary'])
                    st.metric("🌟 Legendary Players", legend_count)
                    
            else:
                st.info(f"No players have achieved Hall of Fame status (minimum {HOF_THRESHOLDS['hall_of_fame']:,} career rating points) for the selected filters.")
        else:
            st.info("No career rating data available to calculate Hall of Fame status.")
            
    except Exception as e:
        st.error(f"Error calculating Hall of Fame data: {str(e)}")
        st.info("Please try adjusting your filters or check the data.")

    # ...rest of the function remains unchanged...

def display_batting_rankings(bat_df):
    """Display Batting Rankings tab content"""
    # Apply global format filter if it exists
    selected_format = st.session_state.get('global_format_filter', [])
    if selected_format:
        bat_df = bat_df[bat_df['Match_Format'].isin(selected_format)]
    
    # Use the correct team column from batting data
    if 'Team' not in bat_df.columns:
        if 'Bat_Team' in bat_df.columns:
            bat_df['Team'] = bat_df['Bat_Team']
        elif 'Batting_Team' in bat_df.columns:
            bat_df['Team'] = bat_df['Batting_Team']
        elif 'Bat_Team_y' in bat_df.columns:
            bat_df['Team'] = bat_df['Bat_Team_y']
        else:
            bat_df['Team'] = 'Unknown'

    # Calculate additional batting statistics
    bat_df['Average'] = bat_df.groupby(['Year', 'Name'])['Runs'].transform('mean')
    bat_df['Strike_Rate'] = (bat_df['Runs'] / bat_df['Balls']) * 100
    bat_df['Centuries'] = bat_df['Runs'].apply(lambda x: 1 if x >= 100 else 0)
    bat_df['Double_Centuries'] = bat_df['Runs'].apply(lambda x: 1 if x >= 200 else 0)

    batting_rankings = bat_df.groupby(['Year', 'Name', 'Team', 'Match_Format']).agg({
        'File Name': 'nunique',
        'Batting_Rating': 'sum',
        'Runs': 'sum',
        'Average': 'mean',
        'Strike_Rate': 'mean',
        'Centuries': 'sum',
        'Double_Centuries': 'sum',
        '4s': 'sum',
        '6s': 'sum'
    }).reset_index()

    batting_rankings = batting_rankings.rename(columns={
        'File Name': 'Matches',
        'Batting_Rating': 'Rating',
        'Runs': 'Total_Runs'
    })

    batting_rankings['RPG'] = batting_rankings['Rating'] / batting_rankings['Matches']
    batting_rankings['Rank'] = batting_rankings.groupby('Year')['Rating'].rank(method='dense', ascending=False).astype(int)
    batting_rankings = batting_rankings.sort_values(['Year', 'Rank'])

    numeric_cols = ['Rating', 'RPG', 'Total_Runs', 'Average', 'Strike_Rate']
    batting_rankings[numeric_cols] = batting_rankings[numeric_cols].round(2)
    batting_rankings['Year'] = batting_rankings['Year'].astype(int)

    # Filters section with modern styling
    st.markdown("""
        <div style="text-align: center; margin: 30px 0;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                       padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                <h2 style="margin: 0; font-size: 1.8em; font-weight: bold;">
                    🔧 Filters & Controls
                </h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                    Customize your batting analysis
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        selected_names = st.multiselect(
            "👤 Filter by Player Name",
            options=sorted(batting_rankings['Name'].unique()),
            key="batting_names",
            help="Select specific players to analyze"
        )
    
    with col2:
        selected_teams = st.multiselect(
            "🏏 Filter by Team",
            options=sorted(batting_rankings['Team'].unique()),
            key="batting_teams",
            help="Select teams to filter by"
        )
    
    with col3:
        selected_formats = st.multiselect(
            "📊 Filter by Format",
            options=sorted(batting_rankings['Match_Format'].unique()),
            key="batting_formats",
            help="Choose match formats"
        )
        
    with col4:
        min_matches = st.number_input(
            "📈 Minimum Season Matches",
            min_value=1,
            max_value=int(batting_rankings['Matches'].max()),
            value=1,
            help="Minimum matches per season"
        )

    filtered_batting = batting_rankings.copy()
    if selected_names:
        filtered_batting = filtered_batting[filtered_batting['Name'].isin(selected_names)]
    if selected_teams:
        filtered_batting = filtered_batting[filtered_batting['Team'].isin(selected_teams)]
    if selected_formats:
        filtered_batting = filtered_batting[filtered_batting['Match_Format'].isin(selected_formats)]
    filtered_batting = filtered_batting[filtered_batting['Matches'] >= min_matches]
    filtered_batting = filtered_batting.sort_values(['Year', 'Rank'], ascending=[False, True])

    # Career Statistics Summary - Modified to include total Rating
    st.markdown("""
        <div style="text-align: center; margin: 30px 0;">
            <div style="background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%); 
                       padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                <h2 style="margin: 0; font-size: 1.8em; font-weight: bold;">
                    📊 Career Statistics Summary
                </h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                    Complete career performance metrics
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    career_stats = filtered_batting.groupby(['Name', 'Match_Format']).agg({
        'Matches': 'sum',
        'Total_Runs': 'sum',
        'Average': 'mean',
        'Strike_Rate': 'mean',
        'Centuries': 'sum',
        'Double_Centuries': 'sum',
        '4s': 'sum',
        '6s': 'sum',
        'Rating': ['sum', 'mean']  # Now calculating both sum and mean
    }).round(2)
    
    # Flatten the column names
    career_stats.columns = ['Matches', 'Total_Runs', 'Average', 'Strike_Rate', 
                          'Centuries', 'Double_Centuries', '4s', '6s', 
                          'Total_Rating', 'Avg_Rating']
    
    career_stats = career_stats.sort_values(['Name', 'Match_Format'])
    career_stats = career_stats.reset_index()
    
    # Display Career Stats safely
    try:
        if not career_stats.empty:
            st.dataframe(
                career_stats,
                use_container_width=True,
                hide_index=True,
                height=min(400, (len(career_stats) + 1) * 35),
                column_config={
                    "Name": st.column_config.TextColumn("🏆 Player", width="medium"),
                    "Match_Format": st.column_config.TextColumn("📊 Format", width="small"),
                    "Total_Rating": st.column_config.NumberColumn("⭐ Total Rating Points", format="%.0f"),
                    "Avg_Rating": st.column_config.NumberColumn("📈 Average Season Rating", format="%.1f"),
                    "Total_Runs": st.column_config.NumberColumn("🏏 Career Runs", format="%.0f"),
                    "Average": st.column_config.NumberColumn("📊 Batting Average", format="%.1f"),
                    "Strike_Rate": st.column_config.NumberColumn("⚡ Strike Rate", format="%.1f"),
                    "Centuries": st.column_config.NumberColumn("💯 100s", format="%.0f"),
                    "Double_Centuries": st.column_config.NumberColumn("🔥 200s", format="%.0f"),
                    "4s": st.column_config.NumberColumn("4️⃣ Fours", format="%.0f"),
                    "6s": st.column_config.NumberColumn("6️⃣ Sixes", format="%.0f")
                }
            )
        else:
            st.info("No career statistics available for the selected filters.")
    except Exception as e:
        st.error(f"Error displaying career statistics: {str(e)}")

    # Yearly Rankings Table
    st.markdown("""
        <div style="text-align: center; margin: 30px 0;">
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                       padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                <h2 style="margin: 0; font-size: 1.8em; font-weight: bold;">
                    📋 Yearly Rankings
                </h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                    Season-by-season batting rankings and performance
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    # Display Yearly Rankings safely
    try:
        if not filtered_batting.empty:
            st.dataframe(
                filtered_batting,
                use_container_width=True,
                hide_index=True,
                height=min(850, (len(filtered_batting) + 1) * 35),
                column_config={
                    "Year": st.column_config.NumberColumn("📅 Year", format="%d"),
                    "Rank": st.column_config.NumberColumn("🏆 Rank", format="%d"),
                    "Name": st.column_config.TextColumn("🏏 Player", width="medium"),
                    "Match_Format": st.column_config.TextColumn("📊 Format", width="small"),
                    "Strike_Rate": st.column_config.NumberColumn("⚡ Strike Rate", format="%.1f"),
                    "Centuries": st.column_config.NumberColumn("💯 100s", format="%.0f"),
                    "Double_Centuries": st.column_config.NumberColumn("🔥 200s", format="%.0f")
                }
            )
        else:
            st.info("No yearly rankings data available for the selected filters.")
    except Exception as e:
        st.error(f"Error displaying yearly rankings: {str(e)}")

    # Create a row for both trend graphs
    st.markdown("""
        <div style="text-align: center; margin: 30px 0;">
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                       padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                <h2 style="margin: 0; font-size: 1.8em; font-weight: bold;">
                    📈 Batting Performance Trends
                </h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                    Rating and ranking trends over time
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating per game trend graph
        fig1 = go.Figure()
        
        trend_data = filtered_batting.sort_values('Year')
        show_legend = len(selected_names) > 0  # Only show legend if players are selected
        
        for player in trend_data['Name'].unique():
            player_data = trend_data[trend_data['Name'] == player]
            fig1.add_trace(go.Scatter(
                x=player_data['Year'],
                y=player_data['RPG'],
                name=player,
                mode='lines+markers',
                line=dict(width=2),
                hovertemplate="Year: %{x}<br>RPG: %{y:.2f}<br>Name: " + player + "<br>Team: %{customdata}<extra></extra>",
                customdata=player_data['Team'],
                showlegend=show_legend
            ))

        fig1.update_layout(
            title="Rating Per Game Trends",
            xaxis_title="Year",
            yaxis_title="Rating Per Game",
            hovermode='closest',
            showlegend=show_legend,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            margin=dict(l=20, r=20, t=40, b=40),
            height=500,
            plot_bgcolor='rgba(255,255,255,0)',
            paper_bgcolor='rgba(255,255,255,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1)
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Rank trend graph
        fig2 = go.Figure()
        
        for player in trend_data['Name'].unique():
            player_data = trend_data[trend_data['Name'] == player]
            fig2.add_trace(go.Scatter(
                x=player_data['Year'],
                y=player_data['Rank'],
                name=player,
                mode='lines+markers',
                line=dict(width=2),
                hovertemplate="Year: %{x}<br>Rank: %{y}<br>Name: " + player + "<br>Team: %{customdata}<extra></extra>",
                customdata=player_data['Team'],
                showlegend=show_legend
            ))

        fig2.update_layout(
            title={
                'text': "📈 Ranking Trends",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'family': 'SF Pro Display', 'color': '#2c3e50'}
            },
            xaxis_title="📅 Year",
            yaxis_title="🏆 Rank",
            hovermode='closest',
            showlegend=show_legend,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(240,79,83,0.3)",
                borderwidth=1
            ),
            margin=dict(l=20, r=20, t=60, b=40),
            height=500,
            plot_bgcolor='rgba(248,249,250,0.8)',
            paper_bgcolor='rgba(255,255,255,0)',
            xaxis=dict(
                showgrid=True, 
                gridcolor='rgba(240,79,83,0.2)', 
                gridwidth=1,
                title_font={'color': '#2c3e50', 'size': 14}
            ),
            yaxis=dict(
                showgrid=True, 
                gridcolor='rgba(240,79,83,0.2)', 
                gridwidth=1,
                autorange="reversed",
                title_font={'color': '#2c3e50', 'size': 14}
            )
        )

        st.plotly_chart(fig2, use_container_width=True)

    # Remove the Batting Milestones Analysis and Team Performance Breakdown sections
    # End of the function


def display_bowling_rankings(bowl_df):
    """Display Bowling Rankings tab content"""
    # Apply global format filter if it exists
    selected_format = st.session_state.get('global_format_filter', [])
    if selected_format:
        bowl_df = bowl_df[bowl_df['Match_Format'].isin(selected_format)]
    
    # First ensure we have the Team information
    if 'Bowl_Team' in bowl_df.columns:
        bowl_df['Team'] = bowl_df['Bowl_Team']
    elif 'Bowling_Team' in bowl_df.columns:
        bowl_df['Team'] = bowl_df['Bowling_Team']
    elif 'Team' not in bowl_df.columns:
        bowl_df['Team'] = 'Unknown'

    # Calculate bowling statistics correctly
    bowl_df['Overs'] = bowl_df['Bowler_Balls'] / 6
    bowl_df['Average'] = bowl_df.groupby(['Year', 'Name'])['Bowler_Runs'].transform('sum') / \
                        bowl_df.groupby(['Year', 'Name'])['Bowler_Wkts'].transform('sum')
    bowl_df['Strike_Rate'] = (bowl_df.groupby(['Year', 'Name'])['Bowler_Balls'].transform('sum') / \
                             bowl_df.groupby(['Year', 'Name'])['Bowler_Wkts'].transform('sum'))
    bowl_df['Economy'] = (bowl_df.groupby(['Year', 'Name'])['Bowler_Runs'].transform('sum') * 6) / \
                        bowl_df.groupby(['Year', 'Name'])['Bowler_Balls'].transform('sum')
    
    # Handle division by zero
    bowl_df['Average'] = bowl_df['Average'].replace([float('inf'), float('-inf')], 0)
    bowl_df['Strike_Rate'] = bowl_df['Strike_Rate'].replace([float('inf'), float('-inf')], 0)
    bowl_df['Economy'] = bowl_df['Economy'].replace([float('inf'), float('-inf')], 0)

    # Calculate milestone counts
    bowl_df['Five_Wickets'] = bowl_df['Bowler_Wkts'].apply(lambda x: 1 if x >= 5 else 0)
    bowl_df['Ten_Wickets'] = bowl_df['Bowler_Wkts'].apply(lambda x: 1 if x >= 10 else 0)

    # Create bowling rankings
    bowling_rankings = bowl_df.groupby(['Year', 'Name', 'Team', 'Match_Format']).agg({
        'File Name': 'nunique',
        'Bowling_Rating': 'sum',
        'Bowler_Wkts': 'sum',
        'Average': 'mean',
        'Strike_Rate': 'mean',
        'Economy': 'mean',
        'Five_Wickets': 'sum',
        'Ten_Wickets': 'sum',
        'Maidens': 'sum'
    }).reset_index()

    bowling_rankings = bowling_rankings.rename(columns={
        'File Name': 'Matches',
        'Bowling_Rating': 'Rating',
        'Bowler_Wkts': 'Total_Wickets'
    })

    bowling_rankings['RPG'] = bowling_rankings['Rating'] / bowling_rankings['Matches']
    bowling_rankings['Rank'] = bowling_rankings.groupby('Year')['Rating'].rank(method='min', ascending=False)

    numeric_cols = ['Rating', 'RPG', 'Total_Wickets', 'Average', 'Strike_Rate', 'Economy']
    bowling_rankings[numeric_cols] = bowling_rankings[numeric_cols].round(2)

    # Filters section with modern styling
    st.markdown("""
        <div style="text-align: center; margin: 30px 0;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                       padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                <h2 style="margin: 0; font-size: 1.8em; font-weight: bold;">
                    🔧 Filters & Controls
                </h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                    Customize your bowling analysis
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        selected_names = st.multiselect(
            "👤 Filter by Player Name",
            options=sorted(bowling_rankings['Name'].unique()),
            key="bowling_names",
            help="Select specific players to analyze"
        )
    
    with col2:
        selected_teams = st.multiselect(
            "🏏 Filter by Team",
            options=sorted(bowling_rankings['Team'].unique()),
            key="bowling_teams",
            help="Select teams to filter by"
        )
    
    with col3:
        selected_formats = st.multiselect(
            "📊 Filter by Format",
            options=sorted(bowling_rankings['Match_Format'].unique()),
            key="bowling_formats",
            help="Choose match formats"
        )
        
    with col4:
        min_matches = st.number_input(
            "📈 Minimum Matches",
            min_value=1,
            max_value=int(bowling_rankings['Matches'].max()),
            value=1,
            help="Minimum matches per season"
        )

    # Apply filters
    filtered_bowling = bowling_rankings.copy()
    if selected_names:
        filtered_bowling = filtered_bowling[filtered_bowling['Name'].isin(selected_names)]
    if selected_teams:
        filtered_bowling = filtered_bowling[filtered_bowling['Team'].isin(selected_teams)]
    if selected_formats:
        filtered_bowling = filtered_bowling[filtered_bowling['Match_Format'].isin(selected_formats)]
    filtered_bowling = filtered_bowling[filtered_bowling['Matches'] >= min_matches]
    filtered_bowling = filtered_bowling.sort_values(['Year', 'Rank'], ascending=[False, True])

    # Career Statistics Summary - Modified to include total Rating
    st.markdown("""
        <div style="text-align: center; margin: 30px 0;">
            <div style="background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%); 
                       padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                <h2 style="margin: 0; font-size: 1.8em; font-weight: bold;">
                    📊 Career Statistics Summary
                </h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                    Complete career bowling performance metrics
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    career_stats = filtered_bowling.groupby(['Name', 'Match_Format']).agg({
        'Matches': 'sum',
        'Total_Wickets': 'sum',
        'Average': 'mean',
        'Strike_Rate': 'mean',
        'Economy': 'mean',
        'Five_Wickets': 'sum',
        'Ten_Wickets': 'sum',
        'Maidens': 'sum',
        'Rating': ['sum', 'mean']  # Now calculating both sum and mean
    }).round(2)
    
    # Flatten the column names
    career_stats.columns = ['Matches', 'Total_Wickets', 'Average', 'Strike_Rate',
                          'Economy', 'Five_Wickets', 'Ten_Wickets', 'Maidens',
                          'Total_Rating', 'Avg_Rating']
    
    career_stats = career_stats.sort_values(['Name', 'Match_Format'])
    career_stats = career_stats.reset_index()
    
    # Display Career Stats safely
    try:
        if not career_stats.empty:
            st.dataframe(
                career_stats,
                use_container_width=True,
                hide_index=True,
                height=min(400, (len(career_stats) + 1) * 35),
                column_config={
                    "Name": st.column_config.TextColumn("🏆 Player", width="medium"),
                    "Match_Format": st.column_config.TextColumn("📊 Format", width="small"),
                    "Total_Rating": st.column_config.NumberColumn("⭐ Total Rating Points", format="%.0f"),
                    "Avg_Rating": st.column_config.NumberColumn("📈 Average Season Rating", format="%.1f"),
                    "Total_Wickets": st.column_config.NumberColumn("⚡ Career Wickets", format="%.0f"),
                    "Average": st.column_config.NumberColumn("📊 Bowling Average", format="%.1f"),
                    "Strike_Rate": st.column_config.NumberColumn("🎯 Strike Rate", format="%.1f"),
                    "Economy": st.column_config.NumberColumn("💰 Economy Rate", format="%.2f"),
                    "Five_Wickets": st.column_config.NumberColumn("🔥 5 Wicket Hauls", format="%.0f"),
                    "Ten_Wickets": st.column_config.NumberColumn("💯 10 Wicket Matches", format="%.0f"),
                    "Maidens": st.column_config.NumberColumn("🛡️ Maiden Overs", format="%.0f")
                }
            )
        else:
            st.info("No career statistics available for the selected filters.")
    except Exception as e:
        st.error(f"Error displaying career statistics: {str(e)}")

    # Yearly Rankings Table
    st.markdown("""
        <div style="text-align: center; margin: 30px 0;">
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                       padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                <h2 style="margin: 0; font-size: 1.8em; font-weight: bold;">
                    📋 Yearly Rankings
                </h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                    Season-by-season bowling rankings and performance
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    # Display Yearly Rankings safely
    try:
        if not filtered_bowling.empty:
            st.dataframe(
                filtered_bowling,
                use_container_width=True,
                hide_index=True,
                height=min(850, (len(filtered_bowling) + 1) * 35),
                column_config={
                    "Year": st.column_config.NumberColumn("📅 Year", format="%d"),
                    "Rank": st.column_config.NumberColumn("🏆 Rank", format="%d"),
                    "Name": st.column_config.TextColumn("⚡ Player", width="medium"),
                    "Match_Format": st.column_config.TextColumn("📊 Format", width="small"),
                    "Average": st.column_config.NumberColumn("📊 Bowling Average", format="%.1f"),
                    "Strike_Rate": st.column_config.NumberColumn("🎯 Strike Rate", format="%.1f"),
                    "Economy": st.column_config.NumberColumn("💰 Economy Rate", format="%.2f"),
                    "Five_Wickets": st.column_config.NumberColumn("🔥 5WI", format="%.0f"),
                    "Ten_Wickets": st.column_config.NumberColumn("💯 10WM", format="%.0f")
                }
            )
        else:
            st.info("No yearly rankings data available for the selected filters.")
    except Exception as e:
        st.error(f"Error displaying yearly rankings: {str(e)}")

    # Create a row for both trend graphs
    st.markdown("""
        <div style="text-align: center; margin: 30px 0;">
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                       padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                <h2 style="margin: 0; font-size: 1.8em; font-weight: bold;">
                    📈 Bowling Performance Trends
                </h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                    Rating and ranking trends over time
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating per game trend graph
        fig1 = go.Figure()
        
        trend_data = filtered_bowling.sort_values('Year')
        show_legend = len(selected_names) > 0  # Only show legend if players are selected
        
        for player in trend_data['Name'].unique():
            player_data = trend_data[trend_data['Name'] == player]
            fig1.add_trace(go.Scatter(
                x=player_data['Year'],
                y=player_data['RPG'],
                name=player,
                mode='lines+markers',
                line=dict(width=2),
                hovertemplate="Year: %{x}<br>RPG: %{y:.2f}<br>Name: " + player + "<br>Team: %{customdata}<extra></extra>",
                customdata=player_data['Team'],
                showlegend=show_legend
            ))

        fig1.update_layout(
            title="Rating Per Game Trends",
            xaxis_title="Year",
            yaxis_title="Rating Per Game",
            hovermode='closest',
            showlegend=show_legend,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            margin=dict(l=20, r=20, t=40, b=40),
            height=500,
            plot_bgcolor='rgba(255,255,255,0)',
            paper_bgcolor='rgba(255,255,255,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1)
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Rank trend graph
        fig2 = go.Figure()
        
        for player in trend_data['Name'].unique():
            player_data = trend_data[trend_data['Name'] == player]
            fig2.add_trace(go.Scatter(
                x=player_data['Year'],
                y=player_data['Rank'],
                name=player,
                mode='lines+markers',
                line=dict(width=2),
                hovertemplate="Year: %{x}<br>Rank: %{y}<br>Name: " + player + "<br>Team: %{customdata}<extra></extra>",
                customdata=player_data['Team'],
                showlegend=show_legend
            ))

        fig2.update_layout(
            title="Ranking Trends",
            xaxis_title="Year",
            yaxis_title="Rank",
            hovermode='closest',
            showlegend=show_legend,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            margin=dict(l=20, r=20, t=40, b=40),
            height=500,
            plot_bgcolor='rgba(255,255,255,0)',
            paper_bgcolor='rgba(255,255,255,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1,
                      autorange="reversed")
        )

        st.plotly_chart(fig2, use_container_width=True)

def display_allrounder_rankings(bat_df, bowl_df):
    """Display All-Rounder Rankings tab content"""
    # Apply global format filter if it exists
    selected_format = st.session_state.get('global_format_filter', [])
    if selected_format:
        bat_df = bat_df[bat_df['Match_Format'].isin(selected_format)]
        bowl_df = bowl_df[bowl_df['Match_Format'].isin(selected_format)]
    
    # Check if dataframes are empty after filtering
    if bat_df.empty or bowl_df.empty:
        st.warning("No data available for the selected format(s). Please select different formats or check your data.")
        return
    
    # Ensure Team column exists in batting dataframe
    if 'Team' not in bat_df.columns:
        if 'Bat_Team' in bat_df.columns:
            bat_df['Team'] = bat_df['Bat_Team']
        elif 'Batting_Team' in bat_df.columns:
            bat_df['Team'] = bat_df['Batting_Team']
        elif 'Bat_Team_y' in bat_df.columns:
            bat_df['Team'] = bat_df['Bat_Team_y']
        else:
            bat_df['Team'] = 'Unknown'
    
    # Ensure Team column exists in bowling dataframe
    if 'Team' not in bowl_df.columns:
        if 'Bowl_Team' in bowl_df.columns:
            bowl_df['Team'] = bowl_df['Bowl_Team']
        elif 'Bowling_Team' in bowl_df.columns:
            bowl_df['Team'] = bowl_df['Bowling_Team']
        elif 'Bowl_Team_y' in bowl_df.columns:
            bowl_df['Team'] = bowl_df['Bowl_Team_y']
        else:
            bowl_df['Team'] = 'Unknown'
    
    try:
        # Create all-rounder rankings
        all_rounder_rankings = bat_df.groupby(['Year', 'Name', 'Team', 'Match_Format']).agg({
            'File Name': 'nunique',
            'Batting_Rating': 'sum',
            'Runs': 'sum'
        }).reset_index()
    except Exception as e:
        st.error(f"Error creating batting rankings: {str(e)}")
        st.info("Please check that the required columns exist in your data.")
        return

    try:
        # Merge bowling stats
        bowling_stats = bowl_df.groupby(['Year', 'Name', 'Match_Format']).agg({
            'Bowling_Rating': 'sum',
            'Bowler_Wkts': 'sum'
        }).reset_index()

        # Modify the merge to include Match_Format
        all_rounder_rankings = pd.merge(
            all_rounder_rankings,
            bowling_stats,
            on=['Year', 'Name', 'Match_Format'],
            how='outer'
        )
    except Exception as e:
        st.error(f"Error merging bowling statistics: {str(e)}")
        st.info("Continuing with batting data only...")
        # Create empty bowling columns if merge fails
        all_rounder_rankings['Bowling_Rating'] = 0
        all_rounder_rankings['Bowler_Wkts'] = 0

    # Fill NaN values for missing data
    all_rounder_rankings = all_rounder_rankings.fillna(0)
    
    # Ensure Team column exists after merge
    if 'Team' not in all_rounder_rankings.columns:
        all_rounder_rankings['Team'] = 'Unknown'
    
    # Ensure File Name column has proper values (can't be 0 for RPG calculation)
    all_rounder_rankings['File Name'] = all_rounder_rankings['File Name'].replace(0, 1)

    # Calculate RPG for both disciplines
    all_rounder_rankings['Batting_RPG'] = all_rounder_rankings['Batting_Rating'] / all_rounder_rankings['File Name']
    all_rounder_rankings['Bowling_RPG'] = all_rounder_rankings['Bowling_Rating'] / all_rounder_rankings['File Name']

    # Apply qualification criteria
    qualified = (all_rounder_rankings['Batting_RPG'] >= 20) & (all_rounder_rankings['Bowling_RPG'] >= 20)
    
    all_rounder_rankings['AR_Rating'] = 0.0
    all_rounder_rankings.loc[qualified, 'AR_Rating'] = \
        all_rounder_rankings.loc[qualified, 'Batting_Rating'] + all_rounder_rankings.loc[qualified, 'Bowling_Rating']

    all_rounder_rankings['AR_RPG'] = all_rounder_rankings['AR_Rating'] / all_rounder_rankings['File Name']

    # Calculate ranks for qualified players
    qualified_rankings = all_rounder_rankings[all_rounder_rankings['AR_Rating'] > 0].copy()
    qualified_rankings['Rank'] = qualified_rankings.groupby('Year')['AR_Rating'].rank(method='min', ascending=False)

    # Round numeric columns
    numeric_cols = ['Batting_Rating', 'Bowling_Rating', 'AR_Rating', 'Batting_RPG', 'Bowling_RPG', 'AR_RPG']
    qualified_rankings[numeric_cols] = qualified_rankings[numeric_cols].round(2)

    # Add filters
    st.markdown("""
        <div style="text-align: center; margin: 30px 0;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                       padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                <h2 style="margin: 0; font-size: 1.8em; font-weight: bold;">
                    🔧 Filters & Controls
                </h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                    Customize your all-rounder analysis
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        selected_names = st.multiselect(
            "Filter by Player Name",
            options=sorted(qualified_rankings['Name'].unique()),
            key="ar_names"
        )

    with col2:
        selected_teams = st.multiselect(
            "Filter by Team",
            options=sorted(qualified_rankings['Team'].unique()),
            key="ar_teams"
        )
    
    with col3:
        selected_formats = st.multiselect(
            "Filter by Format",
            options=sorted(qualified_rankings['Match_Format'].unique()),
            key="ar_formats"
        )

    with col4:
        years = sorted(qualified_rankings['Year'].unique(), reverse=True)
        selected_years = st.multiselect(
            "Filter by Year",
            options=years,
            key="ar_years"
        )

    # Apply filters
    filtered_ar = qualified_rankings.copy()

    if selected_years:
        filtered_ar = filtered_ar[filtered_ar['Year'].isin(selected_years)]
    if selected_names:
        filtered_ar = filtered_ar[filtered_ar['Name'].isin(selected_names)]
    if selected_teams:
        filtered_ar = filtered_ar[filtered_ar['Team'].isin(selected_teams)]
    if selected_formats:
        filtered_ar = filtered_ar[filtered_ar['Match_Format'].isin(selected_formats)]

    filtered_ar = filtered_ar.sort_values(['Year', 'Rank'], ascending=[False, True])

    # Display rankings - Modified to include Format in grouping
    st.markdown("""
        <div style="text-align: center; margin: 30px 0;">
            <div style="background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%); 
                       padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                <h2 style="margin: 0; font-size: 1.8em; font-weight: bold;">
                    🌟 All-Rounder Rankings
                </h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                    Complete all-rounder performance metrics
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # First calculate career stats by format
    career_stats = filtered_ar.groupby(['Name', 'Match_Format']).agg({
        'File Name': 'sum',
        'Runs': 'sum',
        'Bowler_Wkts': 'sum',
        'Batting_RPG': 'mean',
        'Bowling_RPG': 'mean',
        'AR_RPG': 'mean',
        'AR_Rating': 'mean'
    }).round(2)
    
    career_stats = career_stats.sort_values(['Name', 'Match_Format'])
    career_stats = career_stats.reset_index()
    
    # Display Career Stats safely
    try:
        if not career_stats.empty:
            st.dataframe(
                career_stats,
                use_container_width=True,
                hide_index=True,
                height=min(400, (len(career_stats) + 1) * 35),
                column_config={
                    "Name": st.column_config.TextColumn("🏆 Player", width="medium"),
                    "Match_Format": st.column_config.TextColumn("📊 Format", width="small"),
                    "File Name": st.column_config.NumberColumn("📝 Total Matches", format="%.0f"),
                    "Runs": st.column_config.NumberColumn("🏏 Career Runs", format="%.0f"),
                    "Bowler_Wkts": st.column_config.NumberColumn("⚡ Career Wickets", format="%.0f"),
                    "Batting_RPG": st.column_config.NumberColumn("🏏 Avg Batting RPG", format="%.1f"),
                    "Bowling_RPG": st.column_config.NumberColumn("⚡ Avg Bowling RPG", format="%.1f"),
                    "AR_RPG": st.column_config.NumberColumn("🌟 Avg AR RPG", format="%.1f"),
                    "AR_Rating": st.column_config.NumberColumn("📈 Average AR Rating", format="%.1f")
                }
            )
        else:
            st.info("No career statistics available for the selected filters.")
    except Exception as e:
        st.error(f"Error displaying career statistics: {str(e)}")

    # Then display the regular rankings table
    st.markdown("""
        <div style="text-align: center; margin: 30px 0;">
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                       padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                <h2 style="margin: 0; font-size: 1.8em; font-weight: bold;">
                    📋 Yearly Rankings
                </h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                    Season-by-season all-rounder rankings and performance
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    display_columns = [
        'Year', 'Rank', 'Name', 'Team', 'File Name', 
        'Batting_RPG', 'Bowling_RPG', 'AR_RPG',
        'Runs', 'Bowler_Wkts', 'AR_Rating'
    ]
    
    # Display Yearly Rankings safely
    try:
        if not filtered_ar.empty and all(col in filtered_ar.columns for col in display_columns):
            st.dataframe(
                filtered_ar[display_columns],
                use_container_width=True,
                hide_index=True,
                height=min(850, (len(filtered_ar) + 1) * 35),
                column_config={
                    "Year": st.column_config.NumberColumn("📅 Year", format="%d"),
                    "Rank": st.column_config.NumberColumn("🏆 Rank", format="%d"),
                    "Name": st.column_config.TextColumn("🌟 Player", width="medium"),
                    "Team": st.column_config.TextColumn("🏛️ Team", width="small"),
                    "Match_Format": st.column_config.TextColumn("📊 Format", width="small"),
                    "File Name": st.column_config.NumberColumn("📝 Matches", format="%.0f"),
                    "Batting_RPG": st.column_config.NumberColumn("🏏 Batting RPG", format="%.1f"),
                    "Bowling_RPG": st.column_config.NumberColumn("⚡ Bowling RPG", format="%.1f"),
                    "AR_RPG": st.column_config.NumberColumn("🌟 AR RPG", format="%.1f"),
                    "Runs": st.column_config.NumberColumn("🏏 Runs", format="%.0f"),
                    "Bowler_Wkts": st.column_config.NumberColumn("⚡ Total Wickets", format="%.0f"),
                    "AR_Rating": st.column_config.NumberColumn("📈 AR Rating", format="%.1f")
                }
            )
        else:
            st.info("No yearly rankings data available for the selected filters.")
    except Exception as e:
        st.error(f"Error displaying yearly rankings: {str(e)}")

    # Create a row for both trend graphs
    st.markdown("""
        <div style="text-align: center; margin: 30px 0;">
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                       padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                <h2 style="margin: 0; font-size: 1.8em; font-weight: bold;">
                    📈 All-Rounder Performance Trends
                </h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                    Rating and ranking trends over time
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating per game trend graph
        fig1 = go.Figure()
        
        trend_data = filtered_ar.sort_values('Year')
        show_legend = len(selected_names) > 0  # Only show legend if players are selected
        
        for player in trend_data['Name'].unique():
            player_data = trend_data[trend_data['Name'] == player]
            fig1.add_trace(go.Scatter(
                x=player_data['Year'],
                y=player_data['AR_RPG'],
                name=player,
                mode='lines+markers',
                line=dict(width=2),
                hovertemplate="Year: %{x}<br>RPG: %{y:.2f}<br>Name: " + player + "<br>Team: %{customdata}<extra></extra>",
                customdata=player_data['Team'],
                showlegend=show_legend
            ))

        fig1.update_layout(
            title="Rating Per Game Trends",
            xaxis_title="Year",
            yaxis_title="Rating Per Game",
            hovermode='closest',
            showlegend=show_legend,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            margin=dict(l=20, r=20, t=40, b=40),
            height=500,
            plot_bgcolor='rgba(255,255,255,0)',
            paper_bgcolor='rgba(255,255,255,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1)
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Rank trend graph
        fig2 = go.Figure()
        
        for player in trend_data['Name'].unique():
            player_data = trend_data[trend_data['Name'] == player]
            fig2.add_trace(go.Scatter(
                x=player_data['Year'],
                y=player_data['Rank'],
                name=player,
                mode='lines+markers',
                line=dict(width=2),
                hovertemplate="Year: %{x}<br>Rank: %{y}<br>Name: " + player + "<br>Team: %{customdata}<extra></extra>",
                customdata=player_data['Team'],
                showlegend=show_legend
            ))

        fig2.update_layout(
            title="Ranking Trends",
            xaxis_title="Year",
            yaxis_title="Rank",
            hovermode='closest',
            showlegend=show_legend,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            margin=dict(l=20, r=20, t=40, b=40),
            height=500,
            plot_bgcolor='rgba(255,255,255,0)',
            paper_bgcolor='rgba(255,255,255,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1,
                      autorange="reversed")
        )

        st.plotly_chart(fig2, use_container_width=True)

def display_ar_view():
    """Main function to display all rankings views"""
    # Apply modern styling
    apply_modern_styling()
    
    # Add proper top spacing to match other pages
    st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
    
    # Beautiful main title with purple gradient background banner
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px 20px; border-radius: 20px; text-align: center; 
                margin-bottom: 2rem; box-shadow: 0 8px 25px rgba(0,0,0,0.15);'>
        <h1 style='color: white; font-size: 2.2rem; font-weight: 700; margin: 0 0 8px 0;'>
            📊 Player Rankings
        </h1>
        <p style='color: white; font-size: 1rem; margin: 0; opacity: 0.9;'>
            Cricket rankings system based on the cricket draft algorithm
        </p>
    </div>
    """, unsafe_allow_html=True)

    
    # Check for required data
    if 'bat_df' not in st.session_state or 'bowl_df' not in st.session_state:
        st.markdown("""
        <div class='achievement-card'>
            <h3>⚠️ Data Not Available</h3>
            <p>Required batting and bowling data not found in session state.</p>
            <p>Please load data from the main application first.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Get the dataframes
    bat_df = st.session_state['bat_df'].copy()
    bowl_df = st.session_state['bowl_df'].copy()

    def extract_date(text):
        """Extract date from match description string"""
        # If already a timestamp/datetime, return as is
        if isinstance(text, (pd.Timestamp, datetime.datetime)):
            return text
            
        if pd.isna(text):
            return pd.NaT
        
        # Find the last dash and get everything after it
        if ' - ' in text:
            date_part = text.split(' - ')[-1].strip()
        else:
            date_part = text.strip()
            
        # Try parsing with different formats
        try:
            # Try the common formats in your data
            for fmt in ['%d %b %Y', '%d/%m/%Y', '%d %B %Y']:
                try:
                    return pd.to_datetime(date_part, format=fmt)
                except ValueError:
                    continue
            # If none of the specific formats work, try general parsing
            return pd.to_datetime(date_part)
        except:
            return pd.NaT

    # Process dates and add Year column with robust date parsing
    if 'Date' in bat_df.columns:
        bat_df['Date'] = bat_df['Date'].apply(extract_date)
        bat_df['Year'] = bat_df['Date'].dt.year
        # Remove any rows where Year is null or 0
        bat_df = bat_df[bat_df['Year'].notna() & (bat_df['Year'] != 0)]

    # Process bowling dataframe dates
    if 'Date' in bowl_df.columns:
        bowl_df['Date'] = bowl_df['Date'].apply(extract_date)
        bowl_df['Year'] = bowl_df['Date'].dt.year
    
    # Merge with batting years as backup
    bowl_df = pd.merge(
        bowl_df,
        bat_df[['File Name', 'Year']].drop_duplicates(),
        on='File Name',
        how='left'
    )
    
    # If there are two Year columns, use the first non-null value
    if 'Year_x' in bowl_df.columns and 'Year_y' in bowl_df.columns:
        bowl_df['Year'] = bowl_df['Year_x'].combine_first(bowl_df['Year_y'])
        bowl_df = bowl_df.drop(['Year_x', 'Year_y'], axis=1)
    
    # Remove any rows where Year is null or 0
    bowl_df = bowl_df[bowl_df['Year'].notna() & (bowl_df['Year'] != 0)]

    # Calculate ratings
    bat_df = calculate_batter_rating_per_match(bat_df)
    bowl_df = calculate_bowler_rating_per_match(bowl_df)

    # Ensure Year is integer type
    bat_df['Year'] = pd.to_numeric(bat_df['Year'], errors='coerce').fillna(0).astype(int)
    bowl_df['Year'] = pd.to_numeric(bowl_df['Year'], errors='coerce').fillna(0).astype(int)

    # Global Format Filter (above navigation)
    st.markdown('<div class="custom-spacer"></div>', unsafe_allow_html=True)
    
    # Get all unique formats from both dataframes
    all_formats = sorted(set(bat_df['Match_Format'].unique()) | set(bowl_df['Match_Format'].unique()))
    selected_format = st.multiselect(
        "📊 Select Format(s) to Analyze",
        options=all_formats,
        key="global_format_filter",
        help="Choose one or more formats to filter all rankings and analysis"
    )

    # Points System Explanation - Collapsible Section
    with st.expander("📊 **Points System Explanation - Format Specific**", expanded=False):
        # Create tabs for different formats
        format_tab1, format_tab2, format_tab3, format_tab4 = st.tabs([
            "🏏 Test/First Class", "🥎 One Day/List A", "⚡ T20", "🌟 All-Rounder"
        ])
        
        with format_tab1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); 
                        padding: 20px; border-radius: 15px; margin: 10px 0;">
                <h3 style="color: white; text-align: center; margin-bottom: 20px;">🏏 Test Match / First Class Points</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style="background: rgba(255,255,255,0.95); padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <h4 style="color: #2c3e50; margin-bottom: 10px;">🏏 Batting System</h4>
                    <p style="margin: 5px 0;"><strong>• Base Score:</strong> 1 point per run</p>
                    <p style="margin: 5px 0;"><strong>• Century (100-149):</strong> +50 bonus points</p>
                    <p style="margin: 5px 0;"><strong>• 150+ Score:</strong> +75 bonus points</p>
                    <p style="margin: 5px 0;"><strong>• Double Century (200+):</strong> +100 bonus points</p>
                    <p style="margin: 5px 0;"><strong>• Strike Rate Bonus (40+ runs):</strong></p>
                    <p style="margin: 5px 0 5px 20px;">- Fast scoring (≥50 SR): +25 points</p>
                    <p style="margin: 5px 0 5px 20px;">- Very slow (≤30 SR): -25 points</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style="background: rgba(255,255,255,0.95); padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <h4 style="color: #2c3e50; margin-bottom: 10px;">⚡ Bowling System</h4>
                    <p style="margin: 5px 0;"><strong>• Base Score:</strong> 20 points per wicket + 2 points per maiden</p>
                    <p style="margin: 5px 0;"><strong>• Wicket Bonuses:</strong> 3W(+20), 4W(+35), 5W(+50), 6W(+75), 7W(+100), 8W(+150), 9W(+200), 10W(+260)</p>
                    <p style="margin: 5px 0;"><strong>• Economy Bonus (15+ overs):</strong></p>
                    <p style="margin: 5px 0 5px 20px;">- Excellent (≤2.0): +25 points</p>
                    <p style="margin: 5px 0 5px 20px;">- Expensive (≥3.5): -25 points</p>
                </div>
            """, unsafe_allow_html=True)
        
        with format_tab2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); 
                        padding: 20px; border-radius: 15px; margin: 10px 0;">
                <h3 style="color: white; text-align: center; margin-bottom: 20px;">🥎 One Day / List A Points</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style="background: rgba(255,255,255,0.95); padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <h4 style="color: #3498db; margin-bottom: 10px;">🏏 Batting System</h4>
                    <p style="margin: 5px 0;"><strong>• Base Score:</strong> 1 point per run</p>
                    <p style="margin: 5px 0;"><strong>• Century (100-149):</strong> +50 bonus points</p>
                    <p style="margin: 5px 0;"><strong>• 150+ Score:</strong> +75 bonus points</p>
                    <p style="margin: 5px 0;"><strong>• Strike Rate Bonus (40+ runs):</strong></p>
                    <p style="margin: 5px 0 5px 20px;">- High scoring (≥85 SR): +25 points</p>
                    <p style="margin: 5px 0 5px 20px;">- Slow scoring (≤65 SR): -25 points</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style="background: rgba(255,255,255,0.95); padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <h4 style="color: #3498db; margin-bottom: 10px;">⚡ Bowling System</h4>
                    <p style="margin: 5px 0;"><strong>• Base Score:</strong> 20 points per wicket + 2 points per maiden/dot over</p>
                    <p style="margin: 5px 0;"><strong>• Wicket Bonuses:</strong> 3W(+20), 4W(+35), 5W(+50), 6W(+75)</p>
                    <p style="margin: 5px 0;"><strong>• Economy Bonus (8+ overs):</strong></p>
                    <p style="margin: 5px 0 5px 20px;">- Excellent (≤3.5): +25 points</p>
                    <p style="margin: 5px 0 5px 20px;">- Expensive (≥6.0): -25 points</p>
                </div>
            """, unsafe_allow_html=True)
        
        with format_tab3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); 
                        padding: 20px; border-radius: 15px; margin: 10px 0;">
                <h3 style="color: white; text-align: center; margin-bottom: 20px;">⚡ T20 Points</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style="background: rgba(255,255,255,0.95); padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <h4 style="color: #e74c3c; margin-bottom: 10px;">🏏 Batting System</h4>
                    <p style="margin: 5px 0;"><strong>• Base Score:</strong> 1 point per run</p>
                    <p style="margin: 5px 0;"><strong>• Half Century (50-99):</strong> +30 bonus points</p>
                    <p style="margin: 5px 0;"><strong>• Century (100+):</strong> +50 bonus points</p>
                    <p style="margin: 5px 0;"><strong>• Strike Rate Bonus (25+ runs):</strong></p>
                    <p style="margin: 5px 0 5px 20px;">- Explosive (≥140 SR): +25 points</p>
                    <p style="margin: 5px 0 5px 20px;">- Slow (≤110 SR): -25 points</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style="background: rgba(255,255,255,0.95); padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <h4 style="color: #e74c3c; margin-bottom: 10px;">⚡ Bowling System</h4>
                    <p style="margin: 5px 0;"><strong>• Base Score:</strong> 20 points per wicket + 3 points per dot ball over</p>
                    <p style="margin: 5px 0;"><strong>• Wicket Bonuses:</strong> 3W(+20), 4W(+35), 5W(+50)</p>
                    <p style="margin: 5px 0;"><strong>• Economy Bonus (3+ overs):</strong></p>
                    <p style="margin: 5px 0 5px 20px;">- Outstanding (≤6.0): +25 points</p>
                    <p style="margin: 5px 0 5px 20px;">- Expensive (≥10.0): -25 points</p>
                </div>
            """, unsafe_allow_html=True)
        
        with format_tab4:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%); 
                        padding: 20px; border-radius: 15px; margin: 10px 0;">
                <h3 style="color: white; text-align: center; margin-bottom: 20px;">🌟 All-Rounder System</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style="background: rgba(255,255,255,0.95); padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <h4 style="color: #4ecdc4; margin-bottom: 10px;">Qualification Criteria (Format Adjusted)</h4>
                    <p style="margin: 5px 0;"><strong>• Test/FC:</strong> Min 15 RPG in both batting and bowling</p>
                    <p style="margin: 5px 0;"><strong>• ODI/List A:</strong> Min 20 RPG in both batting and bowling</p>
                    <p style="margin: 5px 0;"><strong>• T20:</strong> Min 25 RPG in both batting and bowling</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style="background: rgba(255,255,255,0.95); padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <h4 style="color: #4ecdc4; margin-bottom: 10px;">Total Rating Calculation</h4>
                    <p style="margin: 5px 0;"><strong>• All-Rounder Rating = Batting Rating + Bowling Rating</strong></p>
                    <p style="margin: 5px 0;"><strong>• Format-specific thresholds account for different scoring patterns</strong></p>
                    <p style="margin: 5px 0;"><em>Only qualified players (meeting both RPG thresholds) are ranked</em></p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                        padding: 15px; border-radius: 10px; margin: 10px 0; text-align: center;">
                <p style="margin: 0; color: #666; font-style: italic;">
                    <strong>Note:</strong> Each format has adjusted thresholds and bonuses to reflect the different nature of play.
                    T20 emphasizes strike rates and economy, ODI balances both, while Test cricket rewards consistency and big scores.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Create modern tabs with enhanced styling
    st.markdown('<div class="custom-spacer"></div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs([
        "👑 #1 Rankings", 
        "🏏 Batting Rankings", 
        "⚡ Bowling Rankings", 
        "🌟 All-Rounder Rankings"
    ])

    with tab1:
        display_number_one_rankings(bat_df, bowl_df)
        
    with tab2:
        display_batting_rankings(bat_df)
        
    with tab3:
        display_bowling_rankings(bowl_df)
        
    with tab4:
        display_allrounder_rankings(bat_df, bowl_df)

# Call the function to display with modern styling
if __name__ == "__main__":
    display_ar_view()
 
