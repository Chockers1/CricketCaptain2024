import streamlit as st 
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
try:
    import polars as pl
except Exception:
    pl = None

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

def _show_table_with_cap(
    df: pd.DataFrame,
    name: str | None = None,
    top_n: int = 5000,
    force_full: bool = False,
) -> pd.DataFrame:
    """Return a display DataFrame, optionally capped in large mode.

    - If upload_mode=='large' and df is big, we cap to top_n for display only.
    - If force_full=True, do not cap.
    """
    if df is None or len(df) == 0:
        return df
    mode = st.session_state.get('upload_mode', 'small')
    if not force_full and mode == 'large' and len(df) > top_n:
        if name:
            st.info(f"Large dataset mode: showing first {top_n:,} rows for {name}.")
        return df.head(top_n)
    return df

def _sanitize_for_polars(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    # Coerce Date to datetime where present
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # Cast categoricals or objects with nested types to strings
    for c in df.columns:
        if pd.api.types.is_categorical_dtype(df[c]):
            df[c] = df[c].astype(str)
        elif df[c].dtype == object:
            sample = df[c].dropna().head(1)
            if not sample.empty and isinstance(sample.iloc[0], (list, dict, set, tuple)):
                df[c] = df[c].astype(str)
    return df

@st.cache_data
def calculate_batter_rating_per_match(df):
    """Calculate batting rating with bonuses"""
    stats = df.copy()
    # Vectorized strike rate and base
    stats['Strike_Rate'] = (pd.to_numeric(stats['Runs'], errors='coerce').fillna(0) /
                            pd.to_numeric(stats['Balls'], errors='coerce').replace(0, np.nan)) * 100
    stats['Strike_Rate'] = stats['Strike_Rate'].fillna(0)
    stats['Base_Score'] = pd.to_numeric(stats['Runs'], errors='coerce').fillna(0)
    runs = stats['Base_Score']
    sr = stats['Strike_Rate']
    # Century tier bonuses
    bonus = np.select(
        [runs.ge(200), runs.ge(150), runs.ge(100)],
        [100, 75, 50],
        default=0
    )
    # SR bonus/penalty when substantial innings
    sr_bonus = np.where(runs.ge(40) & sr.ge(75), 25, 0)
    sr_pen = np.where(runs.ge(40) & sr.le(40), -25, 0)
    stats['Bonus'] = bonus + sr_bonus + sr_pen
    stats['Batting_Rating'] = stats['Base_Score'] + stats['Bonus']
    return stats

@st.cache_data
def calculate_bowler_rating_per_match(df):
    """Calculate bowling rating with bonuses"""
    stats = df.copy()
    balls = pd.to_numeric(stats.get('Bowler_Balls', 0), errors='coerce').fillna(0)
    runs = pd.to_numeric(stats.get('Bowler_Runs', 0), errors='coerce').fillna(0)
    wkts = pd.to_numeric(stats.get('Bowler_Wkts', 0), errors='coerce').fillna(0)
    maid = pd.to_numeric(stats.get('Maidens', stats.get('Bowler_Maidens', 0)), errors='coerce').fillna(0)
    overs = balls / 6.0
    econ = runs / overs.replace(0, np.nan)
    econ = econ.fillna(0)
    stats['Overs'] = overs
    stats['Economy'] = econ
    stats['Base_Score'] = wkts * 20 + maid * 2
    # Wicket bonus mapping
    bonus_map = {10:260,9:200,8:150,7:100,6:75,5:50,4:35,3:20}
    stats['Wicket_Bonus'] = wkts.map(bonus_map).fillna(0)
    # Economy bonus when enough overs
    econ_bonus = np.where(overs.ge(10) & econ.le(2.5), 25, 0)
    econ_pen = np.where(overs.ge(10) & econ.ge(4.5), -25, 0)
    stats['Economy_Bonus'] = econ_bonus + econ_pen
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
                if pl is not None:
                    p = pl.from_pandas(_sanitize_for_polars(clean_bat))
                    batting_by_player = (
                        p.group_by(['Year','Match_Format','Name']).agg(pl.col('Batting_Rating').sum().alias('Batting_Rating'))
                    ).to_pandas()
                else:
                    batting_by_player = clean_bat.groupby(['Year','Match_Format','Name'], observed=True)['Batting_Rating'].sum().reset_index()
                # Best string per (Year,Format)
                tmp = batting_by_player.sort_values(['Year','Match_Format','Batting_Rating'], ascending=[True,True,False])
                idx = tmp.groupby(['Year','Match_Format'])['Batting_Rating'].idxmax()
                bb = tmp.loc[idx]
                best_batting = bb.assign(Best_Batting=bb['Name'] + ' - ' + bb['Batting_Rating'].round(0).astype(int).astype(str))[
                    ['Year','Match_Format','Best_Batting']
                ].reset_index(drop=True)

        # Best bowling by year
        if not filtered_bowl_df.empty:
            clean_bowl = filtered_bowl_df.dropna(subset=['Year', 'Match_Format', 'Name', 'Bowling_Rating'])
            clean_bowl = clean_bowl[clean_bowl['Name'].str.strip() != '']
            if not clean_bowl.empty:
                if pl is not None:
                    p = pl.from_pandas(_sanitize_for_polars(clean_bowl))
                    bowling_by_player = (
                        p.group_by(['Year','Match_Format','Name']).agg(pl.col('Bowling_Rating').sum().alias('Bowling_Rating'))
                    ).to_pandas()
                else:
                    bowling_by_player = clean_bowl.groupby(['Year','Match_Format','Name'], observed=True)['Bowling_Rating'].sum().reset_index()
                tmp = bowling_by_player.sort_values(['Year','Match_Format','Bowling_Rating'], ascending=[True,True,False])
                idx = tmp.groupby(['Year','Match_Format'])['Bowling_Rating'].idxmax()
                bw = tmp.loc[idx]
                best_bowling = bw.assign(Best_Bowling=bw['Name'] + ' - ' + bw['Bowling_Rating'].round(0).astype(int).astype(str))[
                    ['Year','Match_Format','Best_Bowling']
                ].reset_index(drop=True)

        # Calculate AR ratings with Match_Format
        if 'batting_by_player' in locals() and 'bowling_by_player' in locals() and not batting_by_player.empty and not bowling_by_player.empty:
            yearly_ar = pd.merge(
                batting_by_player,
                bowling_by_player,
                on=['Year','Match_Format','Name'],
                how='outer'
            ).fillna(0)
            yearly_ar['AR_Rating'] = pd.to_numeric(yearly_ar['Batting_Rating'], errors='coerce').fillna(0) + \
                                      pd.to_numeric(yearly_ar['Bowling_Rating'], errors='coerce').fillna(0)
            tmp = yearly_ar.sort_values(['Year','Match_Format','AR_Rating'], ascending=[True,True,False])
            idx = tmp.groupby(['Year','Match_Format'])['AR_Rating'].idxmax()
            arw = tmp.loc[idx]
            best_ar = arw.assign(Best_AllRounder=arw['Name'] + ' - ' + arw['AR_Rating'].round(0).astype(int).astype(str))[
                ['Year','Match_Format','Best_AllRounder']
            ].reset_index(drop=True)

        # Combine summaries safely
        if not best_batting.empty or not best_bowling.empty or not best_ar.empty:
            yearly_summary = pd.merge(best_batting, best_bowling, on=['Year', 'Match_Format'], how='outer')
            yearly_summary = pd.merge(yearly_summary, best_ar, on=['Year', 'Match_Format'], how='outer')
            yearly_summary = yearly_summary.fillna("-")
            yearly_summary = yearly_summary.sort_values(['Year', 'Match_Format'], ascending=[False, True])

            if not yearly_summary.empty:
                yearly_summary = _show_table_with_cap(
                    yearly_summary,
                    name='Yearly Best Performers',
                )
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

                peaks = _show_table_with_cap(peaks, name='Peak Rating Achievements')
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
                    all_no1 = _show_table_with_cap(all_no1, name='#1 Rankings')
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
        'goat': 15000,            # G.O.A.T status
        'hall_of_fame': 7500,      # Standard HOF threshold
        'elite': 10000,             # Elite status
        'legendary': 12500          # Legendary status
    }
    
    def get_hof_status(points):
        """Determine Hall of Fame status based on total rating points"""
        if points >= HOF_THRESHOLDS['goat']:
            return "🐐 G.O.A.T", "goat"
        elif points >= HOF_THRESHOLDS['legendary']:
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
                            <strong>🐐 G.O.A.T:</strong> 15,000+ points | 
                            <strong>🌟 Legendary:</strong> 12,500+ points | 
                            <strong>💎 Elite:</strong> 10,000+ points | 
                            <strong>�️ Hall of Fame:</strong> 7,500+ points
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                hof_tbl = _show_table_with_cap(
                    hof_players[['Name', 'Match_Format', 'Batting_Points', 'Bowling_Points', 
                               'AllRounder_Points', 'Max_Points', 'HOF_Status']],
                    name='Hall of Fame Players'
                )
                st.dataframe(
                    hof_tbl,
                    use_container_width=True,
                    hide_index=True,
                    height=min(500, (len(hof_tbl) + 1) * 35),
                    column_config={
                        "Name": st.column_config.TextColumn("🏆 Player", width="medium"),
                        "Match_Format": st.column_config.TextColumn("📊 Format", width="small"),
                        "Batting_Points": st.column_config.NumberColumn("🏏 Batting Points", format="%.0f"),
                        "Bowling_Points": st.column_config.NumberColumn("⚡ Bowling Points", format="%.0f"),
                        "AllRounder_Points": st.column_config.NumberColumn("🌟 AR Points", format="%.0f"),
                        "Max_Points": st.column_config.NumberColumn("🎯 Highest Points", format="%.0f"),
                        "HOF_Status": st.column_config.TextColumn("🏛️ HOF Status", width="medium")
                    }
                )
                
                # Show summary stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    goat_count = len(hof_players[hof_players['HOF_Level'] == 'goat'])
                    st.metric("🐐 G.O.A.T Players", goat_count)
                with col3:
                    legend_count = len(hof_players[hof_players['HOF_Level'] == 'legendary'])
                    st.metric("🌟 Legendary Players", legend_count)
                with col2:
                    elite_count = len(hof_players[hof_players['HOF_Level'] == 'elite'])
                    st.metric("💎 Elite Players", elite_count)
                with col4:
                    total_hof = len(hof_players)
                    st.metric("🏛️ Total HOF Players", total_hof)
                    
            else:
                st.info(f"No players have achieved Hall of Fame status (minimum {HOF_THRESHOLDS['hall_of_fame']:,} career rating points) for the selected filters.")
        else:
            st.info("No career rating data available to calculate Hall of Fame status.")
            
    except Exception as e:
        st.error(f"Error calculating Hall of Fame data: {str(e)}")
        st.info("Please try adjusting your filters or check the data.")

    # ...rest of the function remains unchanged...

@st.cache_data
@st.cache_data
def compute_batting_rankings(bat_df):
    if bat_df.empty:
        return pd.DataFrame()
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
    # Polars-first aggregation
    if pl is not None:
        pdf = _sanitize_for_polars(bat_df)
        p = pl.from_pandas(pdf)
        g = (
            p.group_by(['Year','Name','Team','Match_Format'])
             .agg([
                 pl.col('File Name').n_unique().alias('Matches'),
                 pl.col('Batting_Rating').sum().alias('Rating'),
                 pl.col('Runs').sum().alias('Total_Runs'),
                 pl.col('Runs').mean().alias('Average'),
                 (pl.col('Runs').sum() / pl.col('Balls').sum() * 100).alias('Strike_Rate'),
                 (pl.col('Runs')>=100).sum().alias('Centuries'),
                 (pl.col('Runs')>=200).sum().alias('Double_Centuries'),
                 pl.col('4s').sum().alias('4s'),
                 pl.col('6s').sum().alias('6s'),
             ])
        )
        batting_rankings = g.to_pandas()
    else:
        bat = bat_df.copy()
        bat['Centuries'] = (bat['Runs']>=100).astype(int)
        bat['Double_Centuries'] = (bat['Runs']>=200).astype(int)
        grp = bat.groupby(['Year','Name','Team','Match_Format'], observed=True)
        batting_rankings = grp.agg(
            Matches=('File Name','nunique'),
            Rating=('Batting_Rating','sum'),
            Total_Runs=('Runs','sum'),
            Average=('Runs','mean'),
            Balls=('Balls','sum'),
            Centuries=('Centuries','sum'),
            Double_Centuries=('Double_Centuries','sum'),
            **{'4s':('4s','sum'), '6s':('6s','sum')}
        ).reset_index()
        batting_rankings['Strike_Rate'] = np.where(
            batting_rankings['Balls']>0,
            batting_rankings['Total_Runs']/batting_rankings['Balls']*100,
            0
        )
        batting_rankings = batting_rankings.drop(columns=['Balls'])
    batting_rankings = batting_rankings.rename(columns={
        'Runs_mean': 'Average'
    })
    batting_rankings['RPG'] = batting_rankings['Rating'] / batting_rankings['Matches']
    batting_rankings['Rank'] = batting_rankings.groupby('Year')['Rating'].rank(method='dense', ascending=False).astype(int)
    batting_rankings = batting_rankings.sort_values(['Year', 'Rank'])
    numeric_cols = ['Rating', 'RPG', 'Total_Runs', 'Average', 'Strike_Rate']
    batting_rankings[numeric_cols] = batting_rankings[numeric_cols].round(2)
    batting_rankings['Year'] = batting_rankings['Year'].astype(int)
    return batting_rankings

def display_batting_rankings(bat_df):
    """Display Batting Rankings tab content"""
    # Apply global format filter if it exists
    selected_format = st.session_state.get('global_format_filter', [])
    if selected_format:
        bat_df = bat_df[bat_df['Match_Format'].isin(selected_format)]
    # Call the cached function to get the pre-computed rankings
    batting_rankings = compute_batting_rankings(bat_df)
    # Filters section with modern styling
    st.markdown("""
        <div style=\"text-align: center; margin: 30px 0;\">
            <div style=\"background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); \
                       padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);\">
                <h2 style=\"margin: 0; font-size: 1.8em; font-weight: bold;\">
                    🔧 Filters & Controls
                </h2>
                <p style=\"margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;\">
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
            max_value=int(batting_rankings['Matches'].max()) if not batting_rankings.empty else 1,
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
    # ...rest of the function remains unchanged...

    if filtered_batting.empty:
        st.info("No results match your filters. Try broadening your selection.")
    else:
        show_all_bat = st.checkbox("Show all rows (disable large-mode cap)", key="bat_show_all")
        display_batting = _show_table_with_cap(
            filtered_batting,
            name='Batting Rankings',
            force_full=show_all_bat
        )
        st.dataframe(
            display_batting,
            use_container_width=True,
            hide_index=True,
            height=min(850, (len(display_batting) + 1) * 35),
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
        # Download full filtered data
        st.download_button(
            "Download full batting results (CSV)",
            data=filtered_batting.to_csv(index=False).encode('utf-8'),
            file_name="batting_rankings_filtered.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_batting_full"
        )

    # Trend Graphs Section
    col1, col2 = st.columns(2)

    with col1:
        fig1 = go.Figure()
        # Always chart on the full filtered data (not the capped table)
        trend_data = filtered_batting.sort_values('Year')
        show_legend = len(selected_names) > 0
        for player in trend_data['Name'].unique():
            for fmt in trend_data.loc[trend_data['Name'] == player, 'Match_Format'].unique():
                player_data = trend_data[(trend_data['Name'] == player) & (trend_data['Match_Format'] == fmt)]
                if player_data.empty:
                    continue
                legend_name = f"{player} • {fmt}"
                customdata = np.stack([player_data['Team'].to_numpy(), player_data['Match_Format'].to_numpy()], axis=-1)
                fig1.add_trace(go.Scatter(
                    x=player_data['Year'],
                    y=player_data['Rating'],
                    name=legend_name,
                    mode='lines+markers',
                    line=dict(width=2),
                    hovertemplate="Year: %{x}<br>Rating: %{y:.2f}<br>Name • Format: " + legend_name + "<br>Team: %{customdata[0]}<extra></extra>",
                    customdata=customdata,
                    showlegend=show_legend
                ))
        fig1.update_layout(
            title="Rating Per Year Trends",
            xaxis_title="Year",
            yaxis_title="Rating Per Year",
            hovermode='closest',
            showlegend=show_legend,
            autosize=False,
            margin=dict(l=20, r=20, t=40, b=40, pad=0),
            height=500,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            plot_bgcolor='rgba(255,255,255,0)',
            paper_bgcolor='rgba(255,255,255,0)'
            # ... other layout options ...
        )
        fig1.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1, tickmode='linear', dtick=1)
        fig1.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1)
        fig1.update_xaxes(tickmode='linear', dtick=1)  # Ensure integer years only
    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False, "responsive": True})

    with col2:
        fig2 = go.Figure()
        for player in trend_data['Name'].unique():
            for fmt in trend_data.loc[trend_data['Name'] == player, 'Match_Format'].unique():
                player_data = trend_data[(trend_data['Name'] == player) & (trend_data['Match_Format'] == fmt)]
                if player_data.empty:
                    continue
                legend_name = f"{player} • {fmt}"
                customdata = np.stack([player_data['Team'].to_numpy(), player_data['Match_Format'].to_numpy()], axis=-1)
                fig2.add_trace(go.Scatter(
                    x=player_data['Year'],
                    y=player_data['Rank'],
                    name=legend_name,
                    mode='lines+markers',
                    line=dict(width=2),
                    hovertemplate="Year: %{x}<br>Rank: %{y}<br>Name • Format: " + legend_name + "<br>Team: %{customdata[0]}<extra></extra>",
                    customdata=customdata,
                    showlegend=show_legend
                ))
        fig2.update_layout(
            title="Ranking Trends",
            xaxis_title="Year",
            yaxis_title="Rank",
            hovermode='closest',
            showlegend=show_legend,
            autosize=False,
            margin=dict(l=20, r=20, t=40, b=40, pad=0),
            height=500,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            plot_bgcolor='rgba(255,255,255,0)',
            paper_bgcolor='rgba(255,255,255,0)'
            # ... other layout options ...
        )
        fig2.update_yaxes(autorange='reversed', showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1)
        fig2.update_xaxes(tickmode='linear', dtick=1, showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1)  # Ensure integer years only
    st.plotly_chart(fig2, use_container_width=True)

@st.cache_data
@st.cache_data
def compute_bowling_rankings(bowl_df):
    if bowl_df.empty:
        return pd.DataFrame()
    # Use the correct team column from bowling data
    if 'Team' not in bowl_df.columns:
        if 'Bowl_Team' in bowl_df.columns:
            bowl_df['Team'] = bowl_df['Bowl_Team']
        elif 'Bowling_Team' in bowl_df.columns:
            bowl_df['Team'] = bowl_df['Bowling_Team']
        elif 'Bowl_Team_y' in bowl_df.columns:
            bowl_df['Team'] = bowl_df['Bowl_Team_y']
        else:
            bowl_df['Team'] = 'Unknown'
    if pl is not None:
        pdf = _sanitize_for_polars(bowl_df)
        p = pl.from_pandas(pdf)
        # Ensure optional columns exist for aggregation
        if 'Bowler_Maidens' not in p.columns:
            p = p.with_columns(pl.lit(0).alias('Bowler_Maidens'))
        five = (pl.col('Bowler_Wkts')>=5).sum().alias('Five_Wickets')
        ten = (pl.col('Bowler_Wkts')>=10).sum().alias('Ten_Wickets')
        runs_sum = pl.col('Bowler_Runs').sum()
        balls_sum = pl.col('Bowler_Balls').sum()
        wkts_sum = pl.col('Bowler_Wkts').sum()
        maid_sum = pl.col('Bowler_Maidens').sum().alias('Maidens')
        g = (
            p.group_by(['Year','Name','Team','Match_Format'])
             .agg([
                 pl.col('File Name').n_unique().alias('Matches'),
                 pl.col('Bowling_Rating').sum().alias('Rating'),
                 wkts_sum.alias('Total_Wickets'),
                 (runs_sum / wkts_sum).alias('Average'),
                 (balls_sum / wkts_sum).alias('Strike_Rate'),
                 (runs_sum * 6.0 / balls_sum).alias('Economy'),
                 five,
                 ten,
                 maid_sum,
             ])
        )
        bowling_rankings = g.to_pandas()
    else:
        b = bowl_df.copy()
        b['Five_Wickets'] = (b['Bowler_Wkts']>=5).astype(int)
        b['Ten_Wickets'] = (b['Bowler_Wkts']>=10).astype(int)
        grp = b.groupby(['Year','Name','Team','Match_Format'], observed=True)
        agg_dict = {
            'File Name':'nunique',
            'Bowling_Rating':'sum',
            'Bowler_Wkts':'sum',
            'Bowler_Runs':'sum',
            'Bowler_Balls':'sum',
            'Five_Wickets':'sum',
            'Ten_Wickets':'sum'
        }
        if 'Bowler_Maidens' in b.columns:
            agg_dict['Bowler_Maidens'] = 'sum'
        agg = grp.agg(agg_dict).reset_index()
        agg = agg.rename(columns={'File Name':'Matches','Bowling_Rating':'Rating','Bowler_Wkts':'Total_Wickets'})
        if 'Bowler_Maidens' in agg.columns:
            agg = agg.rename(columns={'Bowler_Maidens':'Maidens'})
        else:
            agg['Maidens'] = 0
        agg['Average'] = np.where(agg['Total_Wickets']>0, agg['Bowler_Runs']/agg['Total_Wickets'], 0)
        agg['Strike_Rate'] = np.where(agg['Total_Wickets']>0, agg['Bowler_Balls']/agg['Total_Wickets'], 0)
        agg['Economy'] = np.where(agg['Bowler_Balls']>0, agg['Bowler_Runs']*6/agg['Bowler_Balls'], 0)
        bowling_rankings = agg.drop(columns=['Bowler_Runs','Bowler_Balls'])
    # columns already correctly named across branches
    bowling_rankings['RPG'] = bowling_rankings['Rating'] / bowling_rankings['Matches']
    bowling_rankings['Rank'] = bowling_rankings.groupby('Year')['Rating'].rank(method='dense', ascending=False).astype(int)
    bowling_rankings = bowling_rankings.sort_values(['Year', 'Rank'])
    numeric_cols = ['Rating', 'RPG', 'Total_Wickets', 'Average', 'Strike_Rate', 'Economy']
    bowling_rankings[numeric_cols] = bowling_rankings[numeric_cols].round(2)
    bowling_rankings['Year'] = bowling_rankings['Year'].astype(int)
    return bowling_rankings

def display_bowling_rankings(bowl_df):
    # Apply global format filter if it exists
    selected_format = st.session_state.get('global_format_filter', [])
    if selected_format:
        bowl_df = bowl_df[bowl_df['Match_Format'].isin(selected_format)]
    # Call the cached function to get the pre-computed rankings
    bowling_rankings = compute_bowling_rankings(bowl_df)
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
            show_all_bowl = st.checkbox("Show all rows (disable large-mode cap)", key="bowl_show_all")
            display_bowling = _show_table_with_cap(
                filtered_bowling,
                name='Bowling Rankings',
                force_full=show_all_bowl
            )
            st.dataframe(
                display_bowling,
                use_container_width=True,
                hide_index=True,
                height=min(850, (len(display_bowling) + 1) * 35),
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
            # Download full filtered data
            st.download_button(
                "Download full bowling results (CSV)",
                data=filtered_bowling.to_csv(index=False).encode('utf-8'),
                file_name="bowling_rankings_filtered.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_bowling_full"
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
        # Rating per year trend graph
        fig1 = go.Figure()
        # Always chart on the full filtered data (not the capped table)
        trend_data = filtered_bowling.sort_values('Year')
        show_legend = len(selected_names) > 0  # Only show legend if players are selected
        
        for player in trend_data['Name'].unique():
            for fmt in trend_data.loc[trend_data['Name'] == player, 'Match_Format'].unique():
                player_data = trend_data[(trend_data['Name'] == player) & (trend_data['Match_Format'] == fmt)]
                if player_data.empty:
                    continue
                legend_name = f"{player} • {fmt}"
                customdata = np.stack([player_data['Team'].to_numpy(), player_data['Match_Format'].to_numpy()], axis=-1)
                fig1.add_trace(go.Scatter(
                    x=player_data['Year'],
                    y=player_data['Rating'],
                    name=legend_name,
                    mode='lines+markers',
                    line=dict(width=2),
                    hovertemplate="Year: %{x}<br>Rating: %{y:.2f}<br>Name • Format: " + legend_name + "<br>Team: %{customdata[0]}<extra></extra>",
                    customdata=customdata,
                    showlegend=show_legend
                ))

        fig1.update_layout(
            title="Rating Per Year Trends",
            xaxis_title="Year",
            yaxis_title="Rating Per Year",
            hovermode='closest',
            showlegend=show_legend,
            autosize=False,
            margin=dict(l=20, r=20, t=40, b=40, pad=0),
            height=500,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            plot_bgcolor='rgba(255,255,255,0)',
            paper_bgcolor='rgba(255,255,255,0)'
        )
        fig1.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1, tickmode='linear', dtick=1)
        fig1.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1)

        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False, "responsive": True})

    with col2:
        # Ranking trend graph
        fig2 = go.Figure()

        for player in trend_data['Name'].unique():
            for fmt in trend_data.loc[trend_data['Name'] == player, 'Match_Format'].unique():
                player_data = trend_data[(trend_data['Name'] == player) & (trend_data['Match_Format'] == fmt)]
                if player_data.empty:
                    continue
                legend_name = f"{player} • {fmt}"
                customdata = np.stack([player_data['Team'].to_numpy(), player_data['Match_Format'].to_numpy()], axis=-1)
                fig2.add_trace(go.Scatter(
                    x=player_data['Year'],
                    y=player_data['Rank'],
                    name=legend_name,
                    mode='lines+markers',
                    line=dict(width=2),
                    hovertemplate="Year: %{x}<br>Rank: %{y}<br>Name • Format: " + legend_name + "<br>Team: %{customdata[0]}<extra></extra>",
                    customdata=customdata,
                    showlegend=show_legend
                ))

        fig2.update_layout(
            title="Ranking Trends",
            xaxis_title="Year",
            yaxis_title="Rank",
            hovermode='closest',
            showlegend=show_legend,
            autosize=False,
            margin=dict(l=20, r=20, t=40, b=40, pad=0),
            height=500,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            plot_bgcolor='rgba(255,255,255,0)',
            paper_bgcolor='rgba(255,255,255,0)'
        )
        fig2.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1, tickmode='linear', dtick=1)
        fig2.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1, autorange='reversed')

        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False, "responsive": True})

@st.cache_data
@st.cache_data
def compute_allrounder_rankings(bat_df, bowl_df):
    if bat_df.empty or bowl_df.empty:
        return pd.DataFrame()
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
    if pl is not None:
        pb = pl.from_pandas(_sanitize_for_polars(bat_df))
        pbo = pl.from_pandas(_sanitize_for_polars(bowl_df))
        batting_stats = (
            pb.group_by(['Year','Name','Team','Match_Format'])
              .agg([
                  pl.col('File Name').n_unique().alias('Matches'),
                  pl.col('Batting_Rating').sum().alias('Batting_Rating'),
                  pl.col('Runs').sum().alias('Runs')
              ])
        ).to_pandas()
        bowling_stats = (
            pbo.group_by(['Year','Name','Match_Format'])
               .agg([
                   pl.col('Bowling_Rating').sum().alias('Bowling_Rating'),
                   pl.col('Bowler_Wkts').sum().alias('Bowler_Wkts')
               ])
        ).to_pandas()
    else:
        batting_stats = bat_df.groupby(['Year','Name','Team','Match_Format'], observed=True).agg(
            Matches=('File Name','nunique'),
            Batting_Rating=('Batting_Rating','sum'),
            Runs=('Runs','sum')
        ).reset_index()
        bowling_stats = bowl_df.groupby(['Year','Name','Match_Format'], observed=True).agg(
            Bowling_Rating=('Bowling_Rating','sum'),
            Bowler_Wkts=('Bowler_Wkts','sum')
        ).reset_index()
    all_rounder_rankings = pd.merge(batting_stats, bowling_stats, on=['Year', 'Name', 'Match_Format'], how='outer').fillna(0)
    all_rounder_rankings['Matches'] = all_rounder_rankings['Matches'].replace(0, 1) # Avoid division by zero
    all_rounder_rankings['Batting_RPG'] = all_rounder_rankings['Batting_Rating'] / all_rounder_rankings['Matches']
    all_rounder_rankings['Bowling_RPG'] = all_rounder_rankings['Bowling_Rating'] / all_rounder_rankings['Matches']
    # Lowered qualification threshold for debugging
    qualified = (all_rounder_rankings['Batting_RPG'] >= 1) & (all_rounder_rankings['Bowling_RPG'] >= 1)
    all_rounder_rankings['AR_Rating'] = 0.0
    all_rounder_rankings.loc[qualified, 'AR_Rating'] = all_rounder_rankings.loc[qualified, 'Batting_Rating'] + all_rounder_rankings.loc[qualified, 'Bowling_Rating']
    all_rounder_rankings['AR_RPG'] = all_rounder_rankings['AR_Rating'] / all_rounder_rankings['Matches']
    qualified_rankings = all_rounder_rankings[all_rounder_rankings['AR_Rating'] > 0].copy()
    if not qualified_rankings.empty:
        qualified_rankings['Rank'] = qualified_rankings.groupby('Year')['AR_Rating'].rank(method='min', ascending=False)
    return qualified_rankings

def display_allrounder_rankings(bat_df, bowl_df):
    # Apply global format filter if it exists
    selected_format = st.session_state.get('global_format_filter', [])
    if selected_format:
        bat_df = bat_df[bat_df['Match_Format'].isin(selected_format)]
        bowl_df = bowl_df[bowl_df['Match_Format'].isin(selected_format)]
    # Call the cached function to get the pre-computed rankings
    qualified_rankings = compute_allrounder_rankings(bat_df, bowl_df)
    # --- Display Filters ---
    # ... (your filter widgets for selected_names, teams, etc.) ...
    # --- Apply Filters ---
    filtered_ar = qualified_rankings.copy()
    # ... (your logic to filter the 'filtered_ar' dataframe) ...
    # --- Display UI ---
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
    if filtered_ar.empty:
        st.info("No all-rounder data available for the selected filters.")
    else:
        show_all_ar = st.checkbox("Show all rows (disable large-mode cap)", key="ar_show_all")
        display_ar = _show_table_with_cap(filtered_ar, name='All-Rounder Rankings', force_full=show_all_ar)
        st.dataframe(
            display_ar,
            use_container_width=True,
            hide_index=True,
            height=min(850, (len(display_ar) + 1) * 35),
            column_config={
                "Year": st.column_config.NumberColumn("📅 Year", format="%d"),
                "Rank": st.column_config.NumberColumn("🏆 Rank", format="%d"),
                "Name": st.column_config.TextColumn("🌟 Player", width="medium"),
                "Team": st.column_config.TextColumn("🏛️ Team", width="small"),
                "Match_Format": st.column_config.TextColumn("📊 Format", width="small"),
                "Matches": st.column_config.NumberColumn("📝 Matches", format="%.0f"),
                "Batting_RPG": st.column_config.NumberColumn("🏏 Batting RPG", format="%.1f"),
                "Bowling_RPG": st.column_config.NumberColumn("⚡ Bowling RPG", format="%.1f"),
                "AR_RPG": st.column_config.NumberColumn("🌟 AR RPG", format="%.1f"),
                "Runs": st.column_config.NumberColumn("🏏 Runs", format="%.0f"),
                "Bowler_Wkts": st.column_config.NumberColumn("⚡ Total Wickets", format="%.0f"),
                "AR_Rating": st.column_config.NumberColumn("📈 AR Rating", format="%.1f")
            }
        )
        # Download full filtered data
        st.download_button(
            "Download full all-rounder results (CSV)",
            data=filtered_ar.to_csv(index=False).encode('utf-8'),
            file_name="allrounder_rankings_filtered.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_ar_full"
        )
        # All-Rounder Performance Trends (mirroring batting)
        st.markdown("""
            <div style="text-align: center; margin: 30px 0;">
                <div style="background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%); 
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

        ar_col1, ar_col2 = st.columns(2)
        # Always chart on the full filtered data (not the capped table)
        ar_trend = filtered_ar.sort_values('Year')
        ar_show_legend = True if 'Name' in ar_trend.columns else False

        with ar_col1:
            ar_fig1 = go.Figure()
            for player in ar_trend['Name'].unique():
                for fmt in ar_trend.loc[ar_trend['Name'] == player, 'Match_Format'].unique():
                    player_data = ar_trend[(ar_trend['Name'] == player) & (ar_trend['Match_Format'] == fmt)]
                    if player_data.empty:
                        continue
                    legend_name = f"{player} • {fmt}"
                    customdata = np.stack([player_data['Team'].to_numpy(), player_data['Match_Format'].to_numpy()], axis=-1)
                    ar_fig1.add_trace(go.Scatter(
                        x=player_data['Year'],
                        y=player_data['AR_Rating'],
                        name=legend_name,
                        mode='lines+markers',
                        line=dict(width=2),
                        hovertemplate="Year: %{x}<br>Rating: %{y:.2f}<br>Name • Format: " + legend_name + "<br>Team: %{customdata[0]}<extra></extra>",
                        customdata=customdata,
                        showlegend=ar_show_legend
                    ))
            ar_fig1.update_layout(
                title="Rating Per Year Trends",
                xaxis_title="Year",
                yaxis_title="Rating Per Year",
                hovermode='closest',
                showlegend=ar_show_legend,
                autosize=False,
                margin=dict(l=20, r=20, t=40, b=40, pad=0),
                height=500,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255,255,255,0.8)"
                ),
                plot_bgcolor='rgba(255,255,255,0)',
                paper_bgcolor='rgba(255,255,255,0)'
            )
            ar_fig1.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1, tickmode='linear', dtick=1)
            ar_fig1.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1)
            st.plotly_chart(ar_fig1, use_container_width=True, config={"displayModeBar": False, "responsive": True})

        with ar_col2:
            ar_fig2 = go.Figure()
            for player in ar_trend['Name'].unique():
                for fmt in ar_trend.loc[ar_trend['Name'] == player, 'Match_Format'].unique():
                    player_data = ar_trend[(ar_trend['Name'] == player) & (ar_trend['Match_Format'] == fmt)]
                    if player_data.empty:
                        continue
                    legend_name = f"{player} • {fmt}"
                    customdata = np.stack([player_data['Team'].to_numpy(), player_data['Match_Format'].to_numpy()], axis=-1)
                    ar_fig2.add_trace(go.Scatter(
                        x=player_data['Year'],
                        y=player_data['Rank'],
                        name=legend_name,
                        mode='lines+markers',
                        line=dict(width=2),
                        hovertemplate="Year: %{x}<br>Rank: %{y}<br>Name • Format: " + legend_name + "<br>Team: %{customdata[0]}<extra></extra>",
                        customdata=customdata,
                        showlegend=ar_show_legend
                    ))
            ar_fig2.update_layout(
                title="Ranking Trends",
                xaxis_title="Year",
                yaxis_title="Rank",
                hovermode='closest',
                showlegend=ar_show_legend,
                autosize=False,
                margin=dict(l=20, r=20, t=40, b=40, pad=0),
                height=500,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255,255,255,0.8)"
                ),
                plot_bgcolor='rgba(255,255,255,0)',
                paper_bgcolor='rgba(255,255,255,0)'
            )
            ar_fig2.update_yaxes(autorange='reversed', showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1)
            ar_fig2.update_xaxes(tickmode='linear', dtick=1, showgrid=True, gridcolor='rgba(128,128,128,0.2)', gridwidth=1)
            st.plotly_chart(ar_fig2, use_container_width=True, config={"displayModeBar": False, "responsive": True})

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

    def extract_date_series(series: pd.Series) -> pd.Series:
        """Vectorized parse: if string contains ' - ', take last segment; then to_datetime (dayfirst)."""
        s = series.copy()
        # if already datetime, return
        if pd.api.types.is_datetime64_any_dtype(s):
            return s
        s = s.astype('string')
        parts = s.str.rsplit(' - ', n=1).str[-1].str.strip()
        return pd.to_datetime(parts, errors='coerce', dayfirst=True)

    # Process dates and add Year column with robust date parsing
    if 'Date' in bat_df.columns:
        bat_df['Date'] = extract_date_series(bat_df['Date'])
        bat_df['Year'] = bat_df['Date'].dt.year
        # Remove any rows where Year is null or 0
        bat_df = bat_df[bat_df['Year'].notna() & (bat_df['Year'] != 0)]

    # Process bowling dataframe dates
    if 'Date' in bowl_df.columns:
        bowl_df['Date'] = extract_date_series(bowl_df['Date'])
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
