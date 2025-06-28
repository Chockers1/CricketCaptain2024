# Section 1: Imports and Setup
###############################################################################
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import gc
import sys
from datetime import datetime

# --- MODULE-LEVEL CONSTANTS ---------------------------------------
# common columns to drop for batting‐based tables
BAT_DROP_COMMON = [
    'Bat_Team_x','Bowl_Team_x','File Name','Total_Runs','Overs','Wickets',
    'Competition','Batted','Out','Not Out','50s','100s','200s','<25&Out',
    'Caught','Bowled','LBW','Stumped','Run Out','Boundary Runs','Team Balls',
    'Year','Player_of_the_Match'
]
# for carrying bat we keep Total_Runs and Wickets as Team stats
CARRY_BAT_DROP = [c for c in BAT_DROP_COMMON if c not in ('Total_Runs','Wickets')]

# bowling common drops (if needed)
BOWL_DROP_COMMON = [
    # e.g. repeated drop columns for bowling analyses...
]

# Section 2: Helper Functions and Page Setup
###############################################################################
def get_data_hash():
    """Generate a hash of the current data state"""
    hash_components = []
    for df_name in ['bat_df', 'bowl_df', 'match_df', 'game_df']:
        df = st.session_state.get(df_name)
        if df is not None and not df.empty:
            hash_components.append(str(df.shape))
            hash_components.append(str(df.index.values[-1] if not df.empty else ''))
    return hash(''.join(hash_components))

@st.cache_data(ttl=600, show_spinner=False)  # 10 minute cache
def parse_date(date_str):
    """Helper function to parse dates in multiple formats"""
    if pd.isna(date_str):
        return pd.NaT
    
    try:
        # Try common formats
        formats = [
            '%d/%m/%Y',
            '%Y-%m-%d',
            '%d-%m-%Y',
            '%m/%d/%Y',
            '%d %b %Y',
            '%Y%m%d'
        ]
        
        for fmt in formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except ValueError:
                continue
        
        # If no format works, try pandas default parser
        return pd.to_datetime(date_str)
    except Exception:
        # If all parsing attempts fail, return NaT (Not a Time)
        return pd.NaT

@st.cache_data(ttl=600, show_spinner=False)  # 10 minute cache
def process_dataframes():
    """Process all dataframes with improved memory management"""
    # Get current data hash
    current_hash = get_data_hash()
    
    # Clear cache if data hash changed
    if 'last_data_hash' not in st.session_state or st.session_state.last_data_hash != current_hash:
        st.cache_data.clear()
        st.session_state.last_data_hash = current_hash

    def safe_parse_dates(df):
        if df is None:
            return None
        
        try:
            # Make a copy to avoid modifying original
            df = df.copy()
            
            # Convert to smaller dtypes where possible
            for col in df.select_dtypes(include=['int64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
            
            # Parse dates
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date'])
            
            # Free memory
            gc.collect()
            
            return df
        except Exception as e:
            st.error(f"Error processing dataframe: {str(e)}")
            return None

    try:
        # Process each dataframe
        bat_df = safe_parse_dates(st.session_state.get('bat_df'))
        gc.collect()
        bowl_df = safe_parse_dates(st.session_state.get('bowl_df'))
        gc.collect()
        match_df = safe_parse_dates(st.session_state.get('match_df'))
        gc.collect()
        game_df = safe_parse_dates(st.session_state.get('game_df'))
        gc.collect()

        return bat_df, bowl_df, match_df, game_df
    except Exception as e:
        st.error(f"Error in process_dataframes: {str(e)}")
        return None, None, None, None

@st.cache_data(ttl=600, show_spinner=False)  # 10 minute cache
def filter_by_format(df, format_choice):
    """Filter dataframe by format with caching"""
    if df is None or df.empty:
        return df
        
    # Make a copy of the dataframe
    filtered_df = df.copy()
    
    # Only filter if format_choice doesn't include 'All' and isn't empty
    if 'All' not in format_choice and format_choice:
        filtered_df = filtered_df[filtered_df['Match_Format'].isin(format_choice)]
    
    return filtered_df

@st.cache_data(ttl=600, show_spinner=False)  # 10 minute cache
def get_formats():
    """Get unique formats from all dataframes with caching"""
    all_formats = set(['All'])
    
    for df_name in ['game_df', 'bat_df', 'bowl_df', 'match_df']:
        if df_name in st.session_state and st.session_state[df_name] is not None:
            df = st.session_state[df_name]
            if not df.empty and 'Match_Format' in df.columns:
                all_formats.update(df['Match_Format'].unique())
    
    return sorted(list(all_formats))

def init_page():
    """Initialize page styling with beautiful modern design"""
    # Beautiful page header
    st.markdown("""
        <div style="text-align: center; margin: 30px 0;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                       padding: 25px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                <h1 style="margin: 0; font-size: 2.2em; font-weight: bold;">
                    📊 Cricket Records
                </h1>
                <p style="margin: 10px 0 0 0; opacity: 0.9; font-size: 1.1em;">
                    Comprehensive cricket performance statistics and records
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Enhanced CSS styling
    st.markdown("""
    <style>
    /* Modern table styling */
    .stDataFrame > div {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: none;
    }
    
    /* Table headers */
    .stDataFrame thead tr th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 15px 10px !important;
        border: none !important;
        text-align: center !important;
    }
    
    /* Table rows */
    .stDataFrame tbody tr:nth-child(even) { 
        background-color: #f8f9fa !important; 
    }
    .stDataFrame tbody tr:nth-child(odd) { 
        background-color: white !important; 
    }
    .stDataFrame tbody tr:hover {
        background-color: #e3f2fd !important;
        transition: background-color 0.2s ease;
    }
    
    .stDataFrame tbody tr td {
        padding: 12px 10px !important;
        border: 1px solid #f0f0f0 !important;
        text-align: center !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        width: 100%;
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 10px;
        padding: 5px;
        margin-bottom: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        flex-grow: 1;
        text-align: center;
        background: transparent;
        color: #2c3e50;
        font-weight: 600;
        border-radius: 8px;
        margin: 2px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.3);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: white;
        color: #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Filter section styling */
    .stMultiSelect > div > div {
        background: white;
        border-radius: 8px;
        border: 2px solid #e1e5e9;
        transition: border-color 0.3s ease;
    }
    
    .stMultiSelect > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Info boxes */
    .stInfo {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border: none;
        border-radius: 10px;
        color: #2c3e50;
        font-weight: 500;
    }
    
    /* Success boxes */
    .stSuccess {
        background: linear-gradient(135deg, #d4fb79 0%, #97fb57 100%);
        border: none;
        border-radius: 10px;
        color: #2d5016;
        font-weight: 500;
    }
    
    /* Error boxes */
    .stError {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        border: none;
        border-radius: 10px;
        color: #8e2de2;
        font-weight: 500;
    }
    
    /* Plotly chart styling */
    .js-plotly-plot {
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_data():
    """Initialize data without reload button"""
    try:
        # Process dataframes
        bat_df, bowl_df, match_df, game_df = process_dataframes()
        
        # Get formats and filter choice
        formats = get_formats()
        format_choice = st.multiselect('Format:', formats, default=['All'])
        
        if not format_choice:
            format_choice = ['All']
        
        # Apply filters
        filtered_bat_df = filter_by_format(bat_df, format_choice)
        filtered_bowl_df = filter_by_format(bowl_df, format_choice)
        filtered_match_df = filter_by_format(match_df, format_choice)
        filtered_game_df = filter_by_format(game_df, format_choice)
        
        return filtered_bat_df, filtered_bowl_df, filtered_match_df, filtered_game_df
        
    except Exception as e:
        st.error(f"Error initializing data: {str(e)}")
        return None, None, None, None

@st.cache_data(ttl=600, show_spinner=False)  # 10 minute cache
def process_data_chunk(func, df, *args, **kwargs):
    """Process data in smaller chunks to manage memory"""
    if df is None or df.empty:
        return pd.DataFrame()
        
    try:
        # Process in chunks if dataframe is large
        if len(df) > 1000:
            chunk_size = 1000
            chunks = []
            for start in range(0, len(df), chunk_size):
                end = start + chunk_size
                chunk = func(df[start:end], *args, **kwargs)
                chunks.append(chunk)
                gc.collect()
            return pd.concat(chunks, ignore_index=True)
        else:
            return func(df, *args, **kwargs)
    except Exception as e:
        st.error(f"Error processing data chunk: {str(e)}")
        return pd.DataFrame()

# --- HELPERS ------------------------------------------------------
def format_date_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Format a datetime column to 'dd/mm/YYYY'."""
    df[col] = df[col].dt.strftime('%d/%m/%Y')
    return df

def safe_parse_dates(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Downcast numeric types and coerce 'Date' column to datetime.
    Returns a fresh copy or None on failure.
    """
    if df is None:
        return None
    try:
        df = df.copy()
        # downcast all integers/floats in one pass
        df = df.convert_dtypes()  
        # coerce Date
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
        return df
    except Exception as e:
        st.error(f"safe_parse_dates error: {e}")
        return None

# Section 3: Batting Records Functions
###############################################################################
def process_highest_scores(filtered_bat_df: pd.DataFrame) -> pd.DataFrame:
    """Return all centuries sorted by Runs desc, with minimal cols."""
    if filtered_bat_df is None or filtered_bat_df.empty:
        return pd.DataFrame()
    df = (
        filtered_bat_df[filtered_bat_df['Runs'] >= 100]
        .drop(columns=BAT_DROP_COMMON, errors='ignore')
        .sort_values('Runs', ascending=False)
        .rename(columns={'Bat_Team_y':'Bat Team','Bowl_Team_y':'Bowl Team'})
    )
    return format_date_column(df, 'Date')

def process_consecutive_scores(filtered_bat_df, threshold):
    """Process consecutive scores above threshold"""
    if filtered_bat_df is None or filtered_bat_df.empty:
        return pd.DataFrame()
    
    # Sort by name, format and date
    df = filtered_bat_df.sort_values(['Name', 'Match_Format', 'Date'])
    
    # Initialize variables for tracking streaks
    current_streaks = {}  # Will use (name, format) as key
    best_streaks = {}
    streak_dates = {}
    
    for _, row in df.iterrows():
        name = row['Name']
        match_format = row['Match_Format']
        key = (name, match_format)  # Composite key
        runs = row['Runs']
        date = row['Date']
        
        if key not in current_streaks:
            current_streaks[key] = 0
            best_streaks[key] = 0
            streak_dates[key] = {'start': None, 'end': None, 'current_start': None}
        
        if runs >= threshold:
            if current_streaks[key] == 0:
                streak_dates[key]['current_start'] = date
            
            current_streaks[key] += 1
            
            if current_streaks[key] > best_streaks[key]:
                best_streaks[key] = current_streaks[key]
                streak_dates[key]['start'] = streak_dates[key]['current_start']
                streak_dates[key]['end'] = date
        else:
            current_streaks[key] = 0
            streak_dates[key]['current_start'] = None
    
    # Create DataFrame from streak data
    streak_records = []
    for (name, match_format) in best_streaks:
        if best_streaks[(name, match_format)] >= 2:  # Only include streaks of 2 or more matches
            streak_info = {
                'Name': name,
                'Match_Format': match_format,
                'Consecutive Matches': best_streaks[(name, match_format)],
                'Start Date': streak_dates[(name, match_format)]['start'].strftime('%d/%m/%Y'),
                'End Date': streak_dates[(name, match_format)]['end'].strftime('%d/%m/%Y')
            }
            streak_records.append(streak_info)
    
    # If no streaks found, return empty DataFrame with correct columns
    if not streak_records:
        return pd.DataFrame(columns=['Name', 'Match_Format', 'Consecutive Matches', 'Start Date', 'End Date'])
    
    return pd.DataFrame(streak_records).sort_values('Consecutive Matches', ascending=False)

def process_not_out_99s(filtered_bat_df: pd.DataFrame) -> pd.DataFrame:
    """Return 99 not out entries."""
    if filtered_bat_df is None or filtered_bat_df.empty:
        return pd.DataFrame()
    df = (
        filtered_bat_df[(filtered_bat_df['Runs']==99)&(filtered_bat_df['Not Out']==1)]
        .drop(columns=BAT_DROP_COMMON, errors='ignore')
        .sort_values('Runs', ascending=False)
        .rename(columns={'Bat_Team_y':'Bat Team','Bowl_Team_y':'Bowl Team'})
    )
    return format_date_column(df, 'Date')

def process_carrying_bat(filtered_bat_df: pd.DataFrame) -> pd.DataFrame:
    """Return instances of carrying the bat."""
    if filtered_bat_df is None or filtered_bat_df.empty:
        return pd.DataFrame()
    df = (
        filtered_bat_df[
            filtered_bat_df['Position'].isin([1,2]) &
            (filtered_bat_df['Not Out']==1)&(filtered_bat_df['Wickets']==10)
        ]
        .drop(columns=CARRY_BAT_DROP, errors='ignore')
        .rename(columns={
            'Bat_Team_y':'Bat Team','Bowl_Team_y':'Bowl Team',
            'Total_Runs':'Team Total','Wickets':'Team Wickets'
        })
        .loc[:,['Name','Bat Team','Bowl Team','Runs','Team Total','Team Wickets',
                'Balls','4s','6s','Match_Format','Date']]
        .sort_values('Runs', ascending=False)
    )
    return format_date_column(df, 'Date')

def process_hundreds_both_innings(filtered_bat_df):
    """Process hundreds in both innings data"""
    if filtered_bat_df is None or filtered_bat_df.empty:
        return pd.DataFrame()
        
    bat_match_100_df = (filtered_bat_df.groupby(['Name', 'Date', 'Home Team'])
                .agg({
                    'Runs': lambda x: [
                        max([r for r, i in zip(x, filtered_bat_df.loc[x.index, 'Innings']) if i in [1, 2]], default=0),
                        max([r for r, i in zip(x, filtered_bat_df.loc[x.index, 'Innings']) if i in [3, 4]], default=0)
                    ],
                    'Bat_Team_y': 'first',
                    'Bowl_Team_y': 'first',
                    'Balls': 'sum',
                    '4s': 'sum',
                    '6s': 'sum'
                })
                .reset_index())

    bat_match_100_df['1st Innings'] = bat_match_100_df['Runs'].str[0]
    bat_match_100_df['2nd Innings'] = bat_match_100_df['Runs'].str[1]

    hundreds_both_df = (bat_match_100_df[
        (bat_match_100_df['1st Innings'] >= 100) & 
        (bat_match_100_df['2nd Innings'] >= 100)
    ]
    .drop(columns=['Runs'])
    .rename(columns={
        'Bat_Team_y': 'Bat Team',
        'Bowl_Team_y': 'Bowl Team',
        'Home Team': 'Country'
    }))

    hundreds_both_df = hundreds_both_df[
        ['Name', 'Bat Team', 'Bowl Team', 'Country', '1st Innings', '2nd Innings', 
         'Balls', '4s', '6s', 'Date']
    ]
    
    hundreds_both_df['Date'] = pd.to_datetime(hundreds_both_df['Date'])
    hundreds_both_df = hundreds_both_df.sort_values(by='Date', ascending=False)
    hundreds_both_df['Date'] = hundreds_both_df['Date'].dt.strftime('%d/%m/%Y')
    
    return hundreds_both_df

def process_position_scores(filtered_bat_df):
    """Process highest scores by position data"""
    if filtered_bat_df is None or filtered_bat_df.empty:
        return pd.DataFrame()
        
    return (filtered_bat_df.sort_values('Runs', ascending=False)
            .groupby('Position')
            .agg({
                'Name': 'first',
                'Runs': 'first',
                'Balls': 'first',
                'How Out': 'first',
                '4s': 'first',
                '6s': 'first'
            })
            .reset_index()
            .sort_values('Position')
            [['Position', 'Name', 'Runs', 'Balls', 'How Out', '4s', '6s']])

def create_centuries_plot(filtered_bat_df):
    """Create centuries scatter plot"""
    if filtered_bat_df is None or filtered_bat_df.empty:
        return None
        
    centuries_df = filtered_bat_df[filtered_bat_df['Runs'] >= 100].copy()
    fig = go.Figure()

    for name in centuries_df['Name'].unique():
        player_data = centuries_df[centuries_df['Name'] == name]
        fig.add_trace(go.Scatter(
            x=player_data['Balls'],
            y=player_data['Runs'],
            mode='markers+text',
            text=player_data['Name'],
            textposition='top center',
            marker=dict(size=10),
            name=name,
            hovertemplate=(
                f"<b>{name}</b><br><br>"
                "Runs: %{y}<br>"
                "Balls: %{x}<br>"
                f"4s: {player_data['4s'].iloc[0]}<br>"
                f"6s: {player_data['6s'].iloc[0]}<br>"
                "<extra></extra>"
            )
        ))

    fig.update_layout(
        xaxis_title="Balls Faced",
        yaxis_title="Runs Scored",
        height=700,
        font=dict(size=12),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )

    return fig

def process_team_scores(filtered_game_df, score_type='highest'):
    """Process team scores data"""
    if filtered_game_df is None or filtered_game_df.empty:
        return pd.DataFrame()
    
    df = filtered_game_df[['Bat_Team', 'Bowl_Team', 'Total_Runs', 'Wickets',
                          'Overs', 'Run_Rate', 'Competition', 'Match_Format',
                          'Date']].copy()
    
    if score_type == 'lowest':
        df = df[df['Wickets'] == 10]
        df = df.sort_values('Total_Runs', ascending=True)
    else:
        df = df.sort_values('Total_Runs', ascending=False)
    
    df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')
    df.columns = ['Bat Team', 'Bowl Team', 'Runs', 'Wickets', 'Overs',
                 'Run Rate', 'Competition', 'Format', 'Date']
    
    return df

# Initialize the application
init_page()
filtered_bat_df, filtered_bowl_df, filtered_match_df, filtered_game_df = initialize_data()

# Create tabs
tab_names = ["Batting Records", "Bowling Records", "Match Records", "Game Records", "Series Records"]
tabs = st.tabs(tab_names)

# Batting Records Tab
with tabs[0]:
    if filtered_bat_df is not None and not filtered_bat_df.empty:
        # Highest Scores
        st.markdown("""
            <div style="text-align: center; margin: 30px 0;">
                <div style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); 
                           padding: 20px; border-radius: 15px; color: #8e2de2; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                    <h3 style="margin: 0; font-size: 1.5em; font-weight: bold;">
                        🏏 Highest Scores
                    </h3>
                    <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 0.9em;">
                        Century makers and top batting performances
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        highest_scores_df = process_highest_scores(filtered_bat_df)
        if not highest_scores_df.empty:
            st.dataframe(highest_scores_df, use_container_width=True, hide_index=True)

        # Consecutive 50+ Scores
        st.markdown("""
            <div style="text-align: center; margin: 30px 0;">
                <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                           padding: 20px; border-radius: 15px; color: #2c3e50; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                    <h3 style="margin: 0; font-size: 1.5em; font-weight: bold;">
                        🔥 Consecutive Matches with 50+ Scores
                    </h3>
                    <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 0.9em;">
                        Consistency streaks with fifty-plus scores
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        fifties_streak_df = process_consecutive_scores(filtered_bat_df, 50)
        if not fifties_streak_df.empty:
            st.dataframe(fifties_streak_df, use_container_width=True, hide_index=True)

        # Consecutive Centuries
        st.markdown("""
            <div style="text-align: center; margin: 30px 0;">
                <div style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
                           padding: 20px; border-radius: 15px; color: #d4572a; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                    <h3 style="margin: 0; font-size: 1.5em; font-weight: bold;">
                        💯 Consecutive Matches with Centuries
                    </h3>
                    <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 0.9em;">
                        Rare feat of back-to-back hundreds
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        centuries_streak_df = process_consecutive_scores(filtered_bat_df, 100)
        if not centuries_streak_df.empty:
            st.dataframe(centuries_streak_df, use_container_width=True, hide_index=True)

        # 99 Not Out Club
        st.markdown("""
            <div style="text-align: center; margin: 30px 0;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                    <h3 style="margin: 0; font-size: 1.5em; font-weight: bold;">
                        😢 99 Not Out Club
                    </h3>
                    <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                        So close to a century, yet so far
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        not_out_99s_df = process_not_out_99s(filtered_bat_df)
        if not not_out_99s_df.empty:
            st.dataframe(not_out_99s_df, use_container_width=True, hide_index=True)

        # Carrying the Bat
        st.markdown("""
            <div style="text-align: center; margin: 30px 0;">
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                           padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                    <h3 style="margin: 0; font-size: 1.5em; font-weight: bold; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                        🛡️ Carrying the Bat
                    </h3>
                    <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                        Staying unbeaten through a completed innings
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        carrying_bat_df = process_carrying_bat(filtered_bat_df)
        if not carrying_bat_df.empty:
            st.dataframe(carrying_bat_df, use_container_width=True, hide_index=True)

        # Hundred in Each Innings
        st.markdown("""
            <div style="text-align: center; margin: 30px 0;">
                <div style="background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%); 
                           padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                    <h3 style="margin: 0; font-size: 1.5em; font-weight: bold;">
                        🔄 Hundred in Each Innings
                    </h3>
                    <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                        Rare achievement of centuries in both innings
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        hundreds_both_df = process_hundreds_both_innings(filtered_bat_df)
        if not hundreds_both_df.empty:
            st.dataframe(hundreds_both_df, use_container_width=True, hide_index=True)

        # Position Records
        st.markdown("""
            <div style="text-align: center; margin: 30px 0;">
                <div style="background: linear-gradient(135deg, #a8caba 0%, #5d4e75 100%); 
                           padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                    <h3 style="margin: 0; font-size: 1.5em; font-weight: bold;">
                        📍 Highest Score at Each Position
                    </h3>
                    <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                        Best batting performances by batting order
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        position_scores_df = process_position_scores(filtered_bat_df)
        if not position_scores_df.empty:
            st.dataframe(position_scores_df, use_container_width=True, hide_index=True, height=430)

        # Centuries Analysis Plot
        st.markdown("""
            <div style="text-align: center; margin: 30px 0;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                    <h3 style="margin: 0; font-size: 1.5em; font-weight: bold;">
                        📈 Centuries Analysis (Runs vs Balls)
                    </h3>
                    <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                        Visualization of century strike rates and scoring patterns
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        centuries_fig = create_centuries_plot(filtered_bat_df)
        if centuries_fig:
            st.plotly_chart(centuries_fig, use_container_width=True)
    else:
        st.info("No batting records available.")

# Bowling Records Functions
###############################################################################
def process_best_bowling(filtered_bowl_df):
    """Process best bowling figures data"""
    if filtered_bowl_df is None or filtered_bowl_df.empty:
        return pd.DataFrame()
    
    df = (
        filtered_bowl_df[filtered_bowl_df['Bowler_Wkts'] >= 5]
        .sort_values(by=['Bowler_Wkts', 'Bowler_Runs'], ascending=[False, True])
        [
            [
                'Innings', 'Position', 'Name', 'Bowler_Overs', 'Maidens', 'Bowler_Runs',
                'Bowler_Wkts', 'Bowler_Econ', 'Bat_Team', 'Bowl_Team', 'Home_Team',
                'Match_Format', 'Date', 'comp'
            ]
        ]
        .rename(
            columns={
                'Bowler_Overs': 'Overs',
                'Bowler_Runs': 'Runs',
                'Bowler_Wkts': 'Wickets',
                'Bowler_Econ': 'Econ',
                'Bat_Team': 'Bat Team',
                'Bowl_Team': 'Bowl Team',
                'Home_Team': 'Country',
                'Match_Format': 'Format',
                'comp': 'Competition'
            }
        )
    )
    
    df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')
    return df

def process_consecutive_five_wickets(filtered_bowl_df):
    """Process consecutive matches with 5+ wickets"""
    if filtered_bowl_df is None or filtered_bowl_df.empty:
        return pd.DataFrame(columns=['Name', 'Match_Format', 'Bowl_Team',
                                   'Consecutive Matches', 'Start Date', 'End Date'])
    
    # Sort by name, format and date
    df = filtered_bowl_df.sort_values(['Name', 'Match_Format', 'Date'])
    
    # Initialize tracking dicts
    current_streaks = {}  # Key: (name, format)
    best_streaks = {}
    streak_data = {}
    
    for _, row in df.iterrows():
        name = row['Name']
        match_format = row['Match_Format']
        key = (name, match_format)
        wickets = row['Bowler_Wkts']
        date = row['Date']
        bowl_team = row['Bowl_Team']
        
        if key not in current_streaks:
            current_streaks[key] = 0
            best_streaks[key] = 0
            streak_data[key] = {
                'start': None,
                'end': None,
                'current_start': None,
                'bowl_team': None,
                'dates': []
            }
        
        if wickets >= 5:
            if current_streaks[key] == 0:
                streak_data[key]['current_start'] = date
                streak_data[key]['bowl_team'] = bowl_team
            
            current_streaks[key] += 1
            streak_data[key]['dates'].append(date)
            
            if current_streaks[key] > best_streaks[key]:
                best_streaks[key] = current_streaks[key]
                streak_data[key]['start'] = streak_data[key]['current_start']
                streak_data[key]['end'] = date
        else:
            current_streaks[key] = 0
            streak_data[key]['current_start'] = None
            streak_data[key]['dates'] = []
    
    # Create DataFrame from streak data
    streak_records = []
    for (name, match_format) in best_streaks:
        if best_streaks[(name, match_format)] >= 2:  # Only include streaks of 2 or more matches
            # Get the Bowl_Team from the first match in the streak
            streak_start = streak_data[(name, match_format)]['start']
            streak_bowl_team = (df[(df['Name'] == name) & 
                                 (df['Match_Format'] == match_format) & 
                                 (df['Date'] == streak_start)]['Bowl_Team'].iloc[0])
            
            streak_info = {
                'Name': name,
                'Match_Format': match_format,
                'Bowl_Team': streak_bowl_team,
                'Consecutive Matches': best_streaks[(name, match_format)],
                'Start Date': streak_data[(name, match_format)]['start'].strftime('%d/%m/%Y'),
                'End Date': streak_data[(name, match_format)]['end'].strftime('%d/%m/%Y')
            }
            streak_records.append(streak_info)
    
    # If no streaks found, return empty DataFrame with correct columns
    if not streak_records:
        return pd.DataFrame(columns=['Name', 'Match_Format', 'Bowl_Team',
                                   'Consecutive Matches', 'Start Date', 'End Date'])
    
    return pd.DataFrame(streak_records).sort_values('Consecutive Matches', ascending=False)

def process_five_wickets_both(filtered_bowl_df):
    """Process 5+ wickets in both innings data"""
    if filtered_bowl_df is None or filtered_bowl_df.empty:
        return pd.DataFrame()
    
    bowl_5w = (filtered_bowl_df.groupby(['Name', 'Date', 'Home_Team'])
              .agg({
                  'Bowler_Wkts': lambda x: [
                      max([w for w, i in zip(x, filtered_bowl_df.loc[x.index, 'Innings']) 
                          if i in [1, 2]], default=0),
                      max([w for w, i in zip(x, filtered_bowl_df.loc[x.index, 'Innings']) 
                          if i in [3, 4]], default=0)
                  ],
                  'Bowler_Runs': lambda x: [
                      min([r for r, i, w in zip(x, filtered_bowl_df.loc[x.index, 'Innings'], 
                          filtered_bowl_df.loc[x.index, 'Bowler_Wkts']) 
                          if i in [1, 2] and w >= 5], default=0),
                      min([r for r, i, w in zip(x, filtered_bowl_df.loc[x.index, 'Innings'], 
                          filtered_bowl_df.loc[x.index, 'Bowler_Wkts']) 
                          if i in [3, 4] and w >= 5], default=0)
                  ],
                  'Bowler_Overs': lambda x: [
                      [o for o, i, w in zip(x, filtered_bowl_df.loc[x.index, 'Innings'], 
                       filtered_bowl_df.loc[x.index, 'Bowler_Wkts']) 
                       if i in [1, 2] and w >= 5][0] if any([w >= 5 for w, i in 
                       zip(filtered_bowl_df.loc[x.index, 'Bowler_Wkts'], 
                       filtered_bowl_df.loc[x.index, 'Innings']) if i in [1, 2]]) else 0,
                      [o for o, i, w in zip(x, filtered_bowl_df.loc[x.index, 'Innings'], 
                       filtered_bowl_df.loc[x.index, 'Bowler_Wkts']) 
                       if i in [3, 4] and w >= 5][0] if any([w >= 5 for w, i in 
                       zip(filtered_bowl_df.loc[x.index, 'Bowler_Wkts'], 
                       filtered_bowl_df.loc[x.index, 'Innings']) if i in [3, 4]]) else 0
                  ],
                  'Bat_Team': 'first',
                  'Bowl_Team': 'first',
                  'Match_Format': 'first'
              })
              .reset_index())
    
    bowl_5w['1st Innings Wkts'] = bowl_5w['Bowler_Wkts'].str[0]
    bowl_5w['2nd Innings Wkts'] = bowl_5w['Bowler_Wkts'].str[1]
    bowl_5w['1st Innings Runs'] = bowl_5w['Bowler_Runs'].str[0]
    bowl_5w['2nd Innings Runs'] = bowl_5w['Bowler_Runs'].str[1]
    bowl_5w['1st Innings Overs'] = bowl_5w['Bowler_Overs'].str[0]
    bowl_5w['2nd Innings Overs'] = bowl_5w['Bowler_Overs'].str[1]
    
    five_wickets_both_df = (bowl_5w[
        (bowl_5w['1st Innings Wkts'] >= 5) & 
        (bowl_5w['2nd Innings Wkts'] >= 5)
    ]
    .drop(columns=['Bowler_Wkts', 'Bowler_Runs', 'Bowler_Overs'])
    .rename(columns={
        'Home_Team': 'Country',
        'Bat_Team': 'Bat Team',
        'Bowl_Team': 'Bowl Team',
        'Match_Format': 'Format'
    })
    [['Name', 'Bat Team', 'Bowl Team', 'Country', 'Format',
      '1st Innings Wkts', '1st Innings Runs', '1st Innings Overs',
      '2nd Innings Wkts', '2nd Innings Runs', '2nd Innings Overs', 
      'Date']]
    .sort_values(by='Date', ascending=False))
    
    five_wickets_both_df['Date'] = five_wickets_both_df['Date'].dt.strftime('%d/%m/%Y')
    return five_wickets_both_df

def process_match_bowling(filtered_bowl_df):
    """Process match bowling figures data"""
    if filtered_bowl_df is None or filtered_bowl_df.empty:
        return pd.DataFrame()
    
    match_bowling_df = (filtered_bowl_df
        .groupby(['Name', 'Date', 'Home_Team', 'Bat_Team', 'Bowl_Team', 'Match_Format'])
        .agg({
            'Bowler_Wkts': lambda x: [
                max([w for w, i in zip(x, filtered_bowl_df.loc[x.index, 'Innings']) 
                    if i in [1, 2]] or [0]),
                max([w for w, i in zip(x, filtered_bowl_df.loc[x.index, 'Innings']) 
                    if i in [3, 4]] or [0])
            ],
            'Bowler_Runs': lambda x: [
                sum([r for r, i in zip(x, filtered_bowl_df.loc[x.index, 'Innings']) 
                    if i in [1, 2]] or [0]),
                sum([r for r, i in zip(x, filtered_bowl_df.loc[x.index, 'Innings']) 
                    if i in [3, 4]] or [0])
            ]
        })
        .reset_index())
    
    match_bowling_df['1st Innings Wkts'] = match_bowling_df['Bowler_Wkts'].str[0]
    match_bowling_df['2nd Innings Wkts'] = match_bowling_df['Bowler_Wkts'].str[1]
    match_bowling_df['1st Innings Runs'] = match_bowling_df['Bowler_Runs'].str[0]
    match_bowling_df['2nd Innings Runs'] = match_bowling_df['Bowler_Runs'].str[1]
    
    match_bowling_df['Match Wickets'] = match_bowling_df['1st Innings Wkts'] + match_bowling_df['2nd Innings Wkts']
    match_bowling_df['Match Runs'] = match_bowling_df['1st Innings Runs'] + match_bowling_df['2nd Innings Runs']
    
    df = (match_bowling_df
        .sort_values(by=['Match Wickets', 'Match Runs'], 
                    ascending=[False, True])
        .rename(columns={
            'Home_Team': 'Country',
            'Bat_Team': 'Bat Team',
            'Bowl_Team': 'Bowl Team',
            'Match_Format': 'Format'
        })
        [['Name', 'Bat Team', 'Bowl Team', 'Country', 'Format',
          '1st Innings Wkts', '1st Innings Runs',
          '2nd Innings Wkts', '2nd Innings Runs',
          'Match Wickets', 'Match Runs', 'Date']]
    )
    
    df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')
    return df

# Bowling Records Tab
with tabs[1]:
    if filtered_bowl_df is not None and not filtered_bowl_df.empty:
        # Best Bowling Figures
        st.markdown("""
            <div style="text-align: center; margin: 30px 0;">
                <div style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); 
                           padding: 20px; border-radius: 15px; color: #8e2de2; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                    <h3 style="margin: 0; font-size: 1.5em; font-weight: bold;">
                        🎯 Best Bowling Figures
                    </h3>
                    <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 0.9em;">
                        Outstanding bowling performances with 5+ wickets
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        best_bowling_df = process_best_bowling(filtered_bowl_df)
        if not best_bowling_df.empty:
            st.dataframe(best_bowling_df, use_container_width=True, hide_index=True)

        # Consecutive 5-Wicket Hauls
        st.markdown("""
            <div style="text-align: center; margin: 30px 0;">
                <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                           padding: 20px; border-radius: 15px; color: #2c3e50; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                    <h3 style="margin: 0; font-size: 1.5em; font-weight: bold;">
                        🔥 Consecutive Matches with 5+ Wickets
                    </h3>
                    <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 0.9em;">
                        Sustained bowling excellence across multiple matches
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        consecutive_five_wickets_df = process_consecutive_five_wickets(filtered_bowl_df)
        if not consecutive_five_wickets_df.empty:
            st.dataframe(consecutive_five_wickets_df, use_container_width=True, hide_index=True)

        # 5+ Wickets in Both Innings
        st.markdown("""
            <div style="text-align: center; margin: 30px 0;">
                <div style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
                           padding: 20px; border-radius: 15px; color: #d4572a; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                    <h3 style="margin: 0; font-size: 1.5em; font-weight: bold;">
                        🎳 5+ Wickets in Both Innings
                    </h3>
                    <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 0.9em;">
                        Exceptional bowling dominance throughout the match
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        five_wickets_both_df = process_five_wickets_both(filtered_bowl_df)
        if not five_wickets_both_df.empty:
            st.dataframe(five_wickets_both_df, use_container_width=True, hide_index=True)

        # Best Match Bowling Figures
        st.markdown("""
            <div style="text-align: center; margin: 30px 0;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                    <h3 style="margin: 0; font-size: 1.5em; font-weight: bold;">
                        🏆 Best Match Bowling Figures
                    </h3>
                    <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                        Combined bowling figures across both innings
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        best_match_bowling_df = process_match_bowling(filtered_bowl_df)
        if not best_match_bowling_df.empty:
            st.dataframe(best_match_bowling_df, use_container_width=True, hide_index=True)
    else:
        st.info("No bowling records available.")

# Match Records Functions
###############################################################################
def process_wins_data(filtered_match_df, win_type='runs', margin_type='big'):
    """Process wins data by type and margin"""
    if filtered_match_df is None or filtered_match_df is None or filtered_match_df.empty:
        return pd.DataFrame()
    
    # Rest of the function remains the same
    # Set conditions based on win type
    if win_type == 'runs':
        base_conditions = [
            (filtered_match_df['Margin_Runs'] > 0),
            (filtered_match_df['Innings_Win'] == 0),
            (filtered_match_df['Home_Drawn'] != 1)
        ]
        margin_column = 'Margin_Runs'
        margin_name = 'Runs'
    else:  # wickets
        base_conditions = [
            (filtered_match_df['Margin_Wickets'] > 0),
            (filtered_match_df['Innings_Win'] == 0),
            (filtered_match_df['Home_Drawn'] != 1)
        ]
        margin_column = 'Margin_Wickets'
        margin_name = 'Wickets'
    
    df = filtered_match_df[np.all(base_conditions, axis=0)].copy()
    
    df['Win Team'] = np.where(
        df['Home_Win'] == 1,
        df['Home_Team'],
        df['Away_Team']
    )
    
    df['Opponent'] = np.where(
        df['Home_Win'] == 1,
        df['Away_Team'],
        df['Home_Team']
    )
    
    ascending = True if margin_type == 'narrow' else False
    
    df = (df
        .sort_values(margin_column, ascending=ascending)
        [['Date', 'Win Team', 'Opponent', margin_column, 'Match_Result', 'Match_Format']]
        .rename(columns={
            margin_column: margin_name,
            'Match_Result': 'Match Result',
            'Match_Format': 'Format'
        }))
    
    df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')
    return df

def process_follow_on_victories(filtered_game_df, filtered_match_df):
    """Process victories after following on"""
    if filtered_game_df is None or filtered_match_df is None or filtered_game_df.empty:
        return pd.DataFrame()
    
    # Group by match to find consecutive batting innings
    match_groups = filtered_game_df.groupby(['Date', 'Bat_Team', 'Bowl_Team'])
    
    follow_on_victories = []
    for (date, bat_team, bowl_team), group in match_groups:
        # Check if team batted in innings 2 and 3 (follow-on)
        innings_2_3 = group[group['Innings'].isin([2, 3])]
        if len(innings_2_3) == 2 and all(innings_2_3['Bat_Team'] == bat_team):
            # Find corresponding match result
            match_result = filtered_match_df[
                (filtered_match_df['Date'] == date) & 
                ((filtered_match_df['Home_Team'] == bat_team) | 
                 (filtered_match_df['Away_Team'] == bat_team))
            ]
            
            if not match_result.empty:
                match = match_result.iloc[0]
                # Check if the following-on team won
                team_won = (
                    (match['Home_Team'] == bat_team and match['Home_Win'] == 1) or
                    (match['Away_Team'] == bat_team and match['Away_Won'] == 1)
                )
                
                if team_won:
                    innings_2 = group[group['Innings'] == 2].iloc[0]
                    innings_3 = group[group['Innings'] == 3].iloc[0]
                    
                    victory_info = {
                        'Winning Team': bat_team,
                        'Opponent': bowl_team,
                        'First Innings': innings_2['Total_Runs'],
                        'Second Innings': innings_3['Total_Runs'],
                        'Format': innings_2['Match_Format'],
                        'Competition': innings_2['Competition'],
                        'Date': date.strftime('%d/%m/%Y'),
                        'Margin': match['Match_Result']
                    }
                    follow_on_victories.append(victory_info)
    
    if follow_on_victories:
        df = pd.DataFrame(follow_on_victories)
        return df.sort_values('Date', ascending=False)
    
    return pd.DataFrame()


def process_innings_wins(filtered_match_df):
    """Process innings wins data"""
    if filtered_match_df is None or filtered_match_df.empty:
        return pd.DataFrame()
    
    df = filtered_match_df[
        (filtered_match_df['Innings_Win'] == 1) & 
        (filtered_match_df['Home_Drawn'] != 1)
    ].copy()
    
    df['Win Team'] = np.where(
        df['Home_Win'] == 1,
        df['Home_Team'],
        df['Away_Team']
    )
    
    df['Opponent'] = np.where(
        df['Home_Win'] == 1,
        df['Away_Team'],
        df['Home_Team']
    )
    
    df = (df
        .sort_values('Margin_Runs', ascending=False)
        [['Date', 'Win Team', 'Opponent', 'Margin_Runs', 'Innings_Win', 'Match_Result', 'Match_Format']]
        .rename(columns={
            'Margin_Runs': 'Runs',
            'Innings_Win': 'Innings',
            'Match_Result': 'Match Result',
            'Match_Format': 'Format'
        }))
    
    df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')
    return df

# Match Records Tab
with tabs[2]:
    if filtered_match_df is not None and not filtered_match_df.empty:
        # Biggest Wins (Runs)
        st.markdown("""
            <div style="text-align: center; margin: 30px 0;">
                <div style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); 
                           padding: 20px; border-radius: 15px; color: #8e2de2; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                    <h3 style="margin: 0; font-size: 1.5em; font-weight: bold;">
                        💥 Biggest Wins (Runs)
                    </h3>
                    <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 0.9em;">
                        Most dominant victories by run margin
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        bigwin_df = process_wins_data(filtered_match_df, 'runs', 'big')
        if not bigwin_df.empty:
            st.dataframe(bigwin_df, use_container_width=True, hide_index=True)

        # Biggest Wins (Wickets)
        st.markdown("""
            <div style="text-align: center; margin: 30px 0;">
                <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                           padding: 20px; border-radius: 15px; color: #2c3e50; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                    <h3 style="margin: 0; font-size: 1.5em; font-weight: bold;">
                        🎳 Biggest Wins (Wickets)
                    </h3>
                    <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 0.9em;">
                        Most commanding victories by wicket margin
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        bigwin_wickets_df = process_wins_data(filtered_match_df, 'wickets', 'big')
        if not bigwin_wickets_df.empty:
            st.dataframe(bigwin_wickets_df, use_container_width=True, hide_index=True)

        # Innings Wins
        st.markdown("""
            <div style="text-align: center; margin: 30px 0;">
                <div style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
                           padding: 20px; border-radius: 15px; color: #d4572a; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                    <h3 style="margin: 0; font-size: 1.5em; font-weight: bold;">
                        🏆 Biggest Wins (Innings)
                    </h3>
                    <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 0.9em;">
                        Ultimate dominance with innings victories
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        bigwin_innings_df = process_innings_wins(filtered_match_df)
        if not bigwin_innings_df.empty:
            st.dataframe(bigwin_innings_df, use_container_width=True, hide_index=True)

        # Narrowest Wins (Runs)
        st.markdown("""
            <div style="text-align: center; margin: 30px 0;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                    <h3 style="margin: 0; font-size: 1.5em; font-weight: bold;">
                        ⚡ Narrowest Wins (Runs)
                    </h3>
                    <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                        Heart-stopping close finishes by runs
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        narrow_wins_df = process_wins_data(filtered_match_df, 'runs', 'narrow')
        if not narrow_wins_df.empty:
            st.dataframe(narrow_wins_df, use_container_width=True, hide_index=True)

        # Narrowest Wins (Wickets)
        st.markdown("""
            <div style="text-align: center; margin: 30px 0;">
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                           padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                    <h3 style="margin: 0; font-size: 1.5em; font-weight: bold; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                        🎯 Narrowest Wins (Wickets)
                    </h3>
                    <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                        Thrilling last-wicket victories
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        narrowwin_wickets_df = process_wins_data(filtered_match_df, 'wickets', 'narrow')
        if not narrowwin_wickets_df.empty:
            st.dataframe(narrowwin_wickets_df, use_container_width=True, hide_index=True)
    else:
        st.info("No match records available.")

# Game Records Functions
###############################################################################
def get_match_details(filtered_game_df, filtered_match_df):
    """Process match details and add Result and Margin columns"""
    if filtered_game_df is None or filtered_match_df is None:
        return filtered_game_df
        
    def process_row(row):
        bat_team = row['Bat_Team']
        bowl_team = row['Bowl_Team']
        result = 'Unknown'
        margin_info = '-'
        innings_win = 0
        margin_runs = None
        margin_wickets = None
        
        # Check for matches where bat_team is home and bowl_team is away
        home_match = filtered_match_df[
            (filtered_match_df['Home_Team'] == bat_team) & 
            (filtered_match_df['Away_Team'] == bowl_team)
        ]
        
        # Check for matches where bat_team is away and bowl_team is home
        away_match = filtered_match_df[
            (filtered_match_df['Away_Team'] == bat_team) & 
            (filtered_match_df['Home_Team'] == bowl_team)
        ]
        
        if not home_match.empty:
            match = home_match.iloc[0]
            if match['Home_Win'] == 1:
                result = 'Win'
            elif match['Home_Lost'] == 1:
                result = 'Lost'
            elif match['Home_Drawn'] == 1:
                result = 'Draw'
            
            innings_win = match['Innings_Win']
            margin_runs = match['Margin_Runs'] if pd.notna(match['Margin_Runs']) else None
            margin_wickets = match['Margin_Wickets'] if pd.notna(match['Margin_Wickets']) else None
            
        elif not away_match.empty:
            match = away_match.iloc[0]
            if match['Away_Won'] == 1:
                result = 'Win'
            elif match['Away_Lost'] == 1:
                result = 'Lost'
            elif match['Away_Drawn'] == 1:
                result = 'Draw'
            
            innings_win = match['Innings_Win']
            margin_runs = match['Margin_Runs'] if pd.notna(match['Margin_Runs']) else None
            margin_wickets = match['Margin_Wickets'] if pd.notna(match['Margin_Wickets']) else None
        
        # Add margin details
        if innings_win == 1:
            margin_info = 'by an innings'
            if pd.notna(margin_runs) and margin_runs > 0:
                margin_info += f" and {int(margin_runs)} runs"
        elif pd.notna(margin_wickets) and margin_wickets > 0:
            margin_info = f"by {int(margin_wickets)} wickets"
        elif pd.notna(margin_runs) and margin_runs > 0:
            margin_info = f"by {int(margin_runs)} runs"
        
        return pd.Series([result, margin_info, innings_win, margin_runs, margin_wickets], 
                        index=['Result', 'Margin', 'Innings_Win', 'Margin_Runs', 'Margin_Wickets'])
    
    # Add the result columns
    result_df = filtered_game_df.copy()
    result_df[['Result', 'Margin', 'Innings_Win', 'Margin_Runs', 'Margin_Wickets']] = \
        filtered_game_df.apply(process_row, axis=1)
    
    return result_df

def get_highest_chases(processed_game_df, filtered_match_df):
    """Process highest successful run chases"""
    if processed_game_df is None or filtered_match_df is None or processed_game_df.empty:
        return pd.DataFrame()

    try:
        # Merge game and match dataframes
        merged_df = processed_game_df.merge(
            filtered_match_df,
            on=['File Name', 'Date'],
            how='left'
        )
        
        # Determine the correct column name for 'Competition'
        competition_col = 'Competition_x' if 'Competition_x' in merged_df.columns else 'Competition_y'
        
        # Filter for successful chases where:
        # - Test/First Class: 4th innings
        # - Other formats: 2nd innings 
        # - Won by wickets
        successful_chases = merged_df[(
            ((merged_df['Match_Format_x'].isin(['Test Match', 'First Class']) & (merged_df['Innings'] == 4)) |
            (~merged_df['Match_Format_x'].isin(['Test Match', 'First Class']) & (merged_df['Innings'] == 2))) &
            (merged_df['Margin_y'].str.contains('wickets|wicket', case=False, na=False)) &
            (merged_df.apply(lambda x: x['Bat_Team'].lower() in x['Margin_y'].lower() 
                       if pd.notnull(x['Margin_y']) else False, axis=1))
        )]

        # Sort by total runs descending
        chase_df = successful_chases.sort_values('Total_Runs', ascending=False)

        # Select and rename columns
        chase_df = chase_df[['Bat_Team', 'Bowl_Team', 'Innings', 'Total_Runs', 'Wickets', 'Overs', 
                            'Run_Rate', competition_col, 'Match_Format_x', 'Date', 'Margin_y']]
        chase_df.columns = ['Bat Team', 'Bowl Team', 'Innings', 'Runs', 'Wickets', 'Overs',
                           'Run Rate', 'Competition', 'Format', 'Date', 'Margin']

        # Format date
        chase_df['Date'] = chase_df['Date'].dt.strftime('%d/%m/%Y')
        
        return chase_df

    except Exception as e:
        st.error(f"Error processing chases: {str(e)}")
        return pd.DataFrame()

# Game Records Tab
with tabs[3]:
    if filtered_game_df is not None and filtered_match_df is not None and not filtered_game_df.empty:
        try:
            # Process match details
            processed_game_df = get_match_details(filtered_game_df, filtered_match_df)
            
            # Highest Team Scores
            st.markdown("""
                <div style="text-align: center; margin: 30px 0;">
                    <div style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); 
                               padding: 20px; border-radius: 15px; color: #8e2de2; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                        <h3 style="margin: 0; font-size: 1.5em; font-weight: bold;">
                            📊 Highest Team Scores
                        </h3>
                        <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 0.9em;">
                            Record-breaking team innings totals
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            highest_scores_df = process_team_scores(processed_game_df, 'highest')
            if not highest_scores_df.empty:
                st.dataframe(highest_scores_df, use_container_width=True, hide_index=True)
#######RAW DATAFRAME
            #st.markdown("<h3 style='color:#f04f53; text-align: center;'>filtered_game_df</h3>", )
            #st.dataframe(filtered_game_df, use_container_width=True, hide_index=True)

            #st.markdown("<h3 style='color:#f04f53; text-align: center;'>filtered_match_df</h3>",) 
            #st.dataframe(filtered_match_df, use_container_width=True, hide_index=True)

            # Processed Game DataFrame  
            #st.markdown("<h3 style='color:#f04f53; text-align: center;'>processed_game_df</h3>", )
            #st.dataframe(processed_game_df, use_container_width=True, hide_index=True)

            # Lowest Team Scores
            st.markdown("""
                <div style="text-align: center; margin: 30px 0;">
                    <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                               padding: 20px; border-radius: 15px; color: #2c3e50; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                        <h3 style="margin: 0; font-size: 1.5em; font-weight: bold;">
                            📉 Lowest Team Scores (All Out)
                        </h3>
                        <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 0.9em;">
                            Teams dismissed for their lowest totals
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            lowest_scores_df = process_team_scores(processed_game_df, 'lowest')
            if not lowest_scores_df.empty:
                st.dataframe(lowest_scores_df, use_container_width=True, hide_index=True)

 

            # Lowest First Innings Wins
            def process_lowest_first_innings_wins(processed_game_df, filtered_match_df):
                """Process lowest first innings winning scores"""
                if processed_game_df is None or filtered_match_df is None or processed_game_df.empty:
                    return pd.DataFrame()
                    
                # Merge game and match dataframes
                merged_df = processed_game_df.merge(
                    filtered_match_df,
                    on=['File Name', 'Date'], 
                    how='left'
                )
                
                # Select desired columns
                low_wins_df = merged_df[['Bat_Team', 'Bowl_Team', 'Innings', 'Total_Runs', 
                                       'Overs', 'Wickets', 'Run_Rate', 'Competition_x',
                                       'Match_Format_x', 'Player_of_the_Match_x', 'Date',
                                       'Result', 'Margin_y']].copy()

                # Filter for first innings wins and verify batting team appears in margin
                first_innings_wins = (
                    (low_wins_df['Innings'] == 1) & 
                    (low_wins_df['Result'] == 'Win') &
                    (low_wins_df.apply(lambda x: x['Bat_Team'].lower() in x['Margin_y'].lower() 
                                     if pd.notnull(x['Margin_y']) else False, axis=1))
                )
                low_wins_df = low_wins_df[first_innings_wins]
                
                # Format date and rename columns
                low_wins_df['Date'] = low_wins_df['Date'].dt.strftime('%d/%m/%Y')
                low_wins_df.columns = ['Bat Team', 'Bowl Team', 'Innings', 'Runs', 
                                     'Overs', 'Wickets', 'Run Rate', 'Competition', 
                                     'Format', 'Player of the Match', 'Date',
                                     'Result', 'Margin']
                
                # Remove Result column and sort by runs ascending
                low_wins_df = low_wins_df.drop('Result', axis=1)
                
                return low_wins_df.sort_values('Runs', ascending=True)

            st.markdown("""
                <div style="text-align: center; margin: 30px 0;">
                    <div style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
                               padding: 20px; border-radius: 15px; color: #d4572a; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                        <h3 style="margin: 0; font-size: 1.5em; font-weight: bold;">
                            🏃‍♂️ Lowest First Innings Winning Scores
                        </h3>
                        <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 0.9em;">
                            Winning matches with modest first innings totals
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            low_wins_df = process_lowest_first_innings_wins(processed_game_df, filtered_match_df)
            if not low_wins_df.empty:
                st.dataframe(low_wins_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error processing records: {str(e)}")

        # NEW WIN WITH BIGGEST FIRST INNINGS DEFICIT
        try:
            # Highest Successful Run Chases
            st.markdown("""
                <div style="text-align: center; margin: 30px 0;">
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                               padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                        <h3 style="margin: 0; font-size: 1.5em; font-weight: bold;">
                            🏃‍♂️ Highest Successful Run Chases
                        </h3>
                        <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                            Spectacular run chases completed successfully
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            chases_df = get_highest_chases(processed_game_df, filtered_match_df)
            if not chases_df.empty:
                st.dataframe(chases_df, use_container_width=True, hide_index=True)

            def process_first_innings_deficit_wins(proc_game_df, proc_match_df):
                """Process wins after first innings deficit"""
                if proc_game_df is None or proc_match_df is None or proc_game_df.empty:
                    return pd.DataFrame()
                    
                # Merge game and match dataframes
                merged_df = proc_game_df.merge(
                    proc_match_df,
                    on=['File Name', 'Date'], 
                    how='left'
                )

                # Create a pivot table to get first innings totals
                first_innings = merged_df[merged_df['Innings'] == 1][['File Name', 'Date', 'Total_Runs']]
                first_innings = first_innings.rename(columns={'Total_Runs': 'First_Innings_Total'})

                # Create a pivot table to get second innings totals 
                second_innings = merged_df[merged_df['Innings'] == 2][['File Name', 'Date', 'Total_Runs']]
                second_innings = second_innings.rename(columns={'Total_Runs': 'Second_Innings_Total'})

                # Merge both innings totals back
                merged_df = merged_df.merge(
                    first_innings,
                    on=['File Name', 'Date'],
                    how='left'
                ).merge(
                    second_innings, 
                    on=['File Name', 'Date'],
                    how='left'
                )

                # Calculate deficit properly for both innings
                merged_df['Diff'] = np.where(
                    merged_df['Innings'] == 1,
                    merged_df['First_Innings_Total'] - merged_df['Second_Innings_Total'],
                    merged_df['Second_Innings_Total'] - merged_df['First_Innings_Total']
                )

                # Filter out innings 3 and 4 and negative deficits
                merged_df = merged_df[merged_df['Innings'].isin([1,2])]
                merged_df = merged_df[merged_df['Diff'] < 0]

                # Filter rows where batting team appears in margin
                merged_df = merged_df[merged_df.apply(lambda x: str(x['Bat_Team']).lower() in str(x['Margin_y']).lower() 
                                                      if pd.notnull(x['Margin_y']) else False, axis=1)]

                # Sort by deficit ascending
                merged_df = merged_df.sort_values('Diff', ascending=True)

                # Format date
                merged_df['Date'] = merged_df['Date'].dt.strftime('%d/%m/%Y')

                # Select only specified columns and rename them nicely
                columns_to_show = [
                    'Bat_Team', 'Bowl_Team', 'Innings', 'Total_Runs', 'Overs',
                    'Wickets', 'Run_Rate', 'Competition_x', 'Match_Format_x', 
                    'Player_of_the_Match_x', 'Date', 'Margin_y', 'Diff'
                ]
                
                # Rename columns to nice headers
                column_renames = {
                    'Bat_Team': 'Batting Team',
                    'Bowl_Team': 'Bowling Team', 
                    'Total_Runs': 'Runs',
                    'Run_Rate': 'Run Rate',
                    'Competition_x': 'Competition',
                    'Match_Format_x': 'Format',
                    'Player_of_the_Match_x': 'Player of the Match',
                    'Margin_y': 'Margin',
                    'Diff': 'Deficit'
                }
                
                merged_df = merged_df.rename(columns=column_renames)
                columns_to_show = [column_renames.get(col, col) for col in columns_to_show]
                return merged_df[columns_to_show].copy()

            # First Innings Deficit Wins
            st.markdown("""
                <div style="text-align: center; margin: 30px 0;">
                    <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                               padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                        <h3 style="margin: 0; font-size: 1.5em; font-weight: bold; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                            💪 Winning after 1st Innings Deficit
                        </h3>
                        <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                            Amazing comebacks from first innings behind
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            deficit_df = process_first_innings_deficit_wins(processed_game_df, filtered_match_df)
            if deficit_df.empty:
                st.info("No deficit records available.")
            else:
                st.dataframe(deficit_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error processing records: {str(e)}")

        # NEW LEAD FROM 1st INNINGS AND LOST
        def process_first_innings_lead_losses(proc_game_df, proc_match_df):
            """Process losses after first innings lead"""
            if proc_game_df is None or proc_match_df is None or proc_game_df.empty:
                return pd.DataFrame()
                
            # Merge game and match dataframes
            merged_df = proc_game_df.merge(
                proc_match_df,
                on=['File Name', 'Date'], 
                how='left'
            )

            # Create a pivot table to get first innings totals
            first_innings = merged_df[merged_df['Innings'] == 1][['File Name', 'Date', 'Total_Runs']]
            first_innings = first_innings.rename(columns={'Total_Runs': 'First_Innings_Total'})

            # Create a pivot table to get second innings totals 
            second_innings = merged_df[merged_df['Innings'] == 2][['File Name', 'Date', 'Total_Runs']]
            second_innings = second_innings.rename(columns={'Total_Runs': 'Second_Innings_Total'})

            # Merge both innings totals back
            merged_df = merged_df.merge(
                first_innings,
                on=['File Name', 'Date'],
                how='left'
            ).merge(
                second_innings, 
                on=['File Name', 'Date'],
                how='left'
            )

            # Calculate lead properly for both innings
            merged_df['Lead'] = np.where(
                merged_df['Innings'] == 1,
                merged_df['First_Innings_Total'] - merged_df['Second_Innings_Total'],
                merged_df['Second_Innings_Total'] - merged_df['First_Innings_Total']
            )

            # Filter out innings 3 and 4
            merged_df = merged_df[merged_df['Innings'].isin([1,2])]
            
            # Filter for positive leads only
            merged_df = merged_df[merged_df['Lead'] > 0]

            # Filter rows where opposite batting team appears in margin
            merged_df = merged_df[merged_df.apply(lambda x: str(x['Bowl_Team']).lower() in str(x['Margin_y']).lower() 
                                                  if pd.notnull(x['Margin_y']) else False, axis=1)]

            # Sort by lead descending
            merged_df = merged_df.sort_values('Lead', ascending=False)

            # Format date
            merged_df['Date'] = merged_df['Date'].dt.strftime('%d/%m/%Y')

            # Select and rename columns
            columns_to_show = [
                'Bat_Team', 'Bowl_Team', 'Innings', 'Total_Runs', 'Overs',
                'Wickets', 'Run_Rate', 'Competition_x', 'Match_Format_x', 
                'Player_of_the_Match_x', 'Date', 'Margin_y', 'Lead'
            ]
            
            column_renames = {
                'Bat_Team': 'Batting Team',
                'Bowl_Team': 'Bowling Team', 
                'Total_Runs': 'Runs',
                'Run_Rate': 'Run Rate',
                'Competition_x': 'Competition',
                'Match_Format_x': 'Format',
                'Player_of_the_Match_x': 'Player of the Match',
                'Margin_y': 'Margin',
                'Lead': 'Lead'
            }
            
            merged_df = merged_df.rename(columns=column_renames)
            columns_to_show = [column_renames.get(col, col) for col in columns_to_show]
            return merged_df[columns_to_show].copy()

        try:
            # First Innings Lead Losses
            st.markdown("""
                <div style="text-align: center; margin: 30px 0;">
                    <div style="background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%); 
                               padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                        <h3 style="margin: 0; font-size: 1.5em; font-weight: bold;">
                            😔 Losing after 1st Innings Leads
                        </h3>
                        <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                            Disappointing defeats despite taking first innings lead
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            lead_losses_df = process_first_innings_lead_losses(processed_game_df, filtered_match_df)
            if not lead_losses_df.empty:
                st.dataframe(lead_losses_df, use_container_width=True, hide_index=True)
            else:
                st.info("No records available for losses after first innings leads.")
        except Exception as e:
            st.error(f"Error processing first innings lead losses: {str(e)}")
    else:
        st.info("No game or match records available.")
##############SERIES
with tabs[4]:
    # Beautiful Series Records Header
    st.markdown("""
        <div style="text-align: center; margin: 30px 0;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                       padding: 25px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                <h2 style="margin: 0; font-size: 2em; font-weight: bold;">
                    🏆 Series Records
                </h2>
                <p style="margin: 10px 0 0 0; opacity: 0.9; font-size: 1.1em;">
                    Comprehensive series analysis and tournament records
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Helper to check if competition is part of a numbered series
    def is_part_of_series(competition: str) -> bool:
        # Check if the competition starts with a numbered indicator
        if not isinstance(competition, str):
            return False
        parts = competition.lower().split()
        return parts[0] in ['1st', '2nd', '3rd', '4th', '5th']
    
    # Compute series information from match dataframe
    def compute_series_info(match_df):
        potential_series = [
            '1st Test Match','2nd Test Match','3rd Test Match','4th Test Match','5th Test Match',
            '1st One Day International','2nd One Day International','3rd One Day International',
            '4th One Day International','5th One Day International',
            '1st 20 Over International','2nd 20 Over International','3rd 20 Over International',
            '4th 20 Over International','5th 20 Over International',
            'Only One Day International','Only 20 Over International','Only Test Match',
            'Test Championship Final'
        ]
        s_df = match_df[match_df['Competition'].isin(potential_series)].copy()
        s_df = s_df.sort_values('Date', ascending=True)
        series_list = []
        for _, match in s_df.iterrows():
            comp = match['Competition']
            match_date = pd.to_datetime(match['Date'])
            if is_part_of_series(comp):
                subset = s_df[
                    (s_df['Home_Team'] == match['Home_Team']) &
                    (s_df['Away_Team'] == match['Away_Team']) &
                    (s_df['Match_Format'] == match['Match_Format']) &
                    (abs(pd.to_datetime(s_df['Date']) - match_date).dt.days <= 60)
                ]
                if not subset.empty:
                    info = {
                        'Start_Date': min(pd.to_datetime(subset['Date'])).date(),
                        'End_Date': max(pd.to_datetime(subset['Date'])).date(),
                        'Home_Team': match['Home_Team'],
                        'Away_Team': match['Away_Team'],
                        'Match_Format': match['Match_Format'],
                        'Games_Played': len(subset),
                        'Total_Home_Wins': subset['Home_Win'].sum(),
                        'Total_Away_Wins': subset['Home_Lost'].sum(),
                        'Total_Draws': subset['Home_Drawn'].sum(),
                    }
                    key = f"{info['Home_Team']}_{info['Away_Team']}_{info['Match_Format']}_{info['Start_Date']}"
                    if not any(x.get('key') == key for x in series_list):
                        info['key'] = key
                        series_list.append(info)
            elif comp in ['Only One Day International','Only 20 Over International','Only Test Match','Test Championship Final']:
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
                }
                info['key'] = f"{info['Home_Team']}_{info['Away_Team']}_{info['Match_Format']}_{info['Start_Date']}"
                series_list.append(info)
        if series_list:
            s_grouped = pd.DataFrame(series_list).drop_duplicates('key').drop('key', axis=1)
            s_grouped = s_grouped.sort_values('Start_Date')
            s_grouped['Series'] = range(1, len(s_grouped) + 1)
            def determine_series_winner(row):
                if row['Total_Home_Wins'] > row['Total_Away_Wins']:
                    return f"{row['Home_Team']} won {row['Total_Home_Wins']}-{row['Total_Away_Wins']}"
                elif row['Total_Away_Wins'] > row['Total_Home_Wins']:
                    return f"{row['Away_Team']} won {row['Total_Away_Wins']}-{row['Total_Home_Wins']}"
                else:
                    if row['Total_Draws'] > 0:
                        return f"Series Drawn {row['Total_Home_Wins']}-{row['Total_Away_Wins']}"
                    return f"Series Tied {row['Total_Home_Wins']}-{row['Total_Away_Wins']}"
            s_grouped['Series_Result'] = s_grouped.apply(determine_series_winner, axis=1)
            return s_grouped
        return pd.DataFrame(columns=['Start_Date', 'End_Date', 'Home_Team', 'Away_Team', 
                                    'Match_Format', 'Games_Played', 'Total_Home_Wins', 
                                    'Total_Away_Wins', 'Total_Draws', 'Series'])
    
    # Initialize empty series_info_df with proper columns to avoid KeyErrors
    series_info_df = pd.DataFrame(columns=['Series', 'Start_Date', 'End_Date', 'Home_Team', 
                                           'Away_Team', 'Match_Format', 'Series_Result'])
    
    # Series Info Section
    if filtered_match_df is not None and not filtered_match_df.empty:
        # Initialize with proper columns structure even if empty
        computed_info = compute_series_info(filtered_match_df)
        
        if not computed_info.empty:
            st.markdown("""
                <div style="text-align: center; margin: 30px 0;">
                    <div style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); 
                               padding: 20px; border-radius: 15px; color: #8e2de2; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                        <h3 style="margin: 0; font-size: 1.5em; font-weight: bold;">
                            📋 Series Overview
                        </h3>
                        <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 0.9em;">
                            Complete series results and match outcomes
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            series_info_df = computed_info
            # Format dates for display
            series_info_df['Start_Date'] = pd.to_datetime(series_info_df['Start_Date']).dt.strftime('%d/%m/%Y')
            series_info_df['End_Date'] = pd.to_datetime(series_info_df['End_Date']).dt.strftime('%d/%m/%Y')
            
            # Display the series information
            st.dataframe(series_info_df, use_container_width=True, hide_index=True)
        else:
            st.info("No series records available.")
    else:
        st.info("No match data available for series analysis.")

    # Create a copy of batting dataframe and show it
    if filtered_bat_df is not None and not filtered_bat_df.empty and not series_info_df.empty:
        st.markdown("""
            <div style="text-align: center; margin: 30px 0;">
                <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                           padding: 20px; border-radius: 15px; color: #2c3e50; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                    <h3 style="margin: 0; font-size: 1.5em; font-weight: bold;">
                        🏏 Series Batting Analysis
                    </h3>
                    <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 0.9em;">
                        Individual batting performances across series
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Create a copy of batting dataframe
        series_batting = filtered_bat_df.copy()
        
        # Remove unwanted columns
        series_batting = series_batting.drop(['Bat_Team_x', 'Bowl_Team_x'], axis=1, errors='ignore')
        
        # Add Series column by matching date ranges
        def find_series(row):
            try:
                match_date = pd.to_datetime(row['Date']).date()
                # Check if necessary columns exist in series_info_df
                if all(col in series_info_df.columns for col in ['Start_Date', 'End_Date', 'Home_Team', 'Away_Team', 'Series']):
                    # Convert date strings to datetime.date objects for comparison
                    start_dates = pd.to_datetime(series_info_df['Start_Date'], errors='coerce').dt.date
                    end_dates = pd.to_datetime(series_info_df['End_Date'], errors='coerce').dt.date
                    
                    # Filter rows where match_date is within range and teams match
                    mask = (
                        (start_dates <= match_date) & 
                        (end_dates >= match_date) & 
                        ((series_info_df['Home_Team'] == row['Bat_Team_y']) | 
                        (series_info_df['Home_Team'] == row['Bowl_Team_y']) |
                        (series_info_df['Away_Team'] == row['Bat_Team_y']) |
                        (series_info_df['Away_Team'] == row['Bowl_Team_y']))
                    )
                    matches = series_info_df[mask]
                    
                    if not matches.empty:
                        return matches['Series'].iloc[0]
                return None
            except Exception as e:
                # Log error but don't crash
                #st.write(f"Error in find_series: {e}")
                return None
        
        # Add Series column
        series_batting['Series'] = series_batting.apply(find_series, axis=1)
        
        # Filter out rows with no series match
        series_batting = series_batting[series_batting['Series'].notna()]
        
        if series_batting.empty:
            st.info("No batting data matches available series.")
        else:
            # Create series batting statistics
            series_stats = series_batting.groupby(['Series', 'Name', 'Bat_Team_y', 'Bowl_Team_y', 'Home Team']).agg({
                'File Name': 'nunique',
                'Batted': 'sum', 
                'Out': 'sum',
                'Not Out': 'sum',
                'Runs': 'sum',
                'Balls': 'sum',
                '4s': 'sum',
                '6s': 'sum',
                '50s': 'sum',
                '100s': 'sum'
            }).reset_index()
            
            # Rename columns for display
            series_stats.columns = ['Series', 'Name', 'Bat Team', 'Bowl Team', 'Country', 
                                  'Matches', 'Batted', 'Out', 'Not Out', 'Runs', 
                                  'Balls', '4s', '6s', '50s', '100s']
            
            # Display series batting statistics
            if not series_stats.empty:
                st.dataframe(series_stats, use_container_width=True, hide_index=True)
            else:
                st.info("No series batting statistics available.")
    else:
        st.info("No batting data available for series analysis or no series found")

    # Series Bowling Analysis
    if filtered_bowl_df is not None and not filtered_bowl_df.empty and not series_info_df.empty:
        st.markdown("""
            <div style="text-align: center; margin: 30px 0;">
                <div style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
                           padding: 20px; border-radius: 15px; color: #d4572a; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                    <h3 style="margin: 0; font-size: 1.5em; font-weight: bold;">
                        🎳 Series Bowling Analysis
                    </h3>
                    <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 0.9em;">
                        Individual bowling performances across series
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Create a copy of bowling dataframe
        series_bowling = filtered_bowl_df.copy()
        
        # Add Series column by matching date ranges
        def find_bowling_series(row):
            try:
                match_date = pd.to_datetime(row['Date']).date()
                # Check if necessary columns exist in series_info_df
                if all(col in series_info_df.columns for col in ['Start_Date', 'End_Date', 'Home_Team', 'Away_Team', 'Series']):
                    # Convert date strings to datetime.date objects for comparison
                    start_dates = pd.to_datetime(series_info_df['Start_Date'], errors='coerce').dt.date
                    end_dates = pd.to_datetime(series_info_df['End_Date'], errors='coerce').dt.date
                    
                    # Filter rows where match_date is within range and teams match
                    mask = (
                        (start_dates <= match_date) & 
                        (end_dates >= match_date) & 
                        ((series_info_df['Home_Team'] == row['Bat_Team']) | 
                        (series_info_df['Home_Team'] == row['Bowl_Team']) |
                        (series_info_df['Away_Team'] == row['Bat_Team']) |
                        (series_info_df['Away_Team'] == row['Bowl_Team']))
                    )
                    matches = series_info_df[mask]
                    
                    if not matches.empty:
                        return matches['Series'].iloc[0]
                return None
            except Exception as e:
                return None
        
        # Add Series column
        series_bowling['Series'] = series_bowling.apply(find_bowling_series, axis=1)
        
        # Filter out rows with no series match
        series_bowling = series_bowling[series_bowling['Series'].notna()]
        
        if series_bowling.empty:
            st.info("No bowling data matches available series.")
        else:
            # Create series bowling statistics
            series_bowl_stats = series_bowling.groupby(['Series', 'Name', 'Bat_Team', 'Bowl_Team', 'Home_Team']).agg({
                'File Name': 'nunique',
                'Bowler_Overs': 'sum',
                'Bowler_Wkts': 'sum',
                'Bowler_Runs': 'sum'
            }).reset_index()
            
            # Calculate 5-wicket and 10-wicket hauls from individual match data
            five_wicket_hauls = series_bowling[series_bowling['Bowler_Wkts'] >= 5].groupby(['Series', 'Name', 'Bat_Team', 'Bowl_Team', 'Home_Team']).size().reset_index(name='5w_hauls')
            ten_wicket_hauls = series_bowling[series_bowling['Bowler_Wkts'] >= 10].groupby(['Series', 'Name', 'Bat_Team', 'Bowl_Team', 'Home_Team']).size().reset_index(name='10w_hauls')
            
            # Merge the hauls data
            series_bowl_stats = series_bowl_stats.merge(five_wicket_hauls, on=['Series', 'Name', 'Bat_Team', 'Bowl_Team', 'Home_Team'], how='left')
            series_bowl_stats = series_bowl_stats.merge(ten_wicket_hauls, on=['Series', 'Name', 'Bat_Team', 'Bowl_Team', 'Home_Team'], how='left')
            
            # Fill NaN values with 0
            series_bowl_stats['5w_hauls'] = series_bowl_stats['5w_hauls'].fillna(0).astype(int)
            series_bowl_stats['10w_hauls'] = series_bowl_stats['10w_hauls'].fillna(0).astype(int)
            
            # Calculate bowling averages and economy rates
            series_bowl_stats['Bowling_Average'] = series_bowl_stats.apply(
                lambda row: round(row['Bowler_Runs'] / row['Bowler_Wkts'], 2) if row['Bowler_Wkts'] > 0 else 0, axis=1
            )
            series_bowl_stats['Economy_Rate'] = series_bowl_stats.apply(
                lambda row: round(row['Bowler_Runs'] / row['Bowler_Overs'], 2) if row['Bowler_Overs'] > 0 else 0, axis=1
            )
            
            # Rename columns for display
            series_bowl_stats.columns = ['Series', 'Name', 'Bat Team', 'Bowl Team', 'Country', 
                                       'Matches', 'Overs', 'Wickets', 'Runs', '5w', '10w', 
                                       'Average', 'Economy']
            
            # Sort by most wickets first
            series_bowl_stats = series_bowl_stats.sort_values(['Wickets', 'Average'], ascending=[False, True])
            
            # Display series bowling statistics
            if not series_bowl_stats.empty:
                st.dataframe(series_bowl_stats, use_container_width=True, hide_index=True)
            else:
                st.info("No series bowling statistics available.")
    else:
        st.info("No bowling data available for series analysis or no series found")