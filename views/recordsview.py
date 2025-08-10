# Section 1: Imports and Setup
###############################################################################
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import gc
import sys
from datetime import datetime
import polars as pl

# --- MODULE-LEVEL CONSTANTS ---------------------------------------
# Bump this when making notable changes to help verify live module
RECORDS_VIEW_VERSION = "recordsview v2025-08-10-02"
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

    # Tiny version banner to confirm live code
    st.caption(RECORDS_VIEW_VERSION)

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

# FastFrame helpers (Polars) --------------------------------------------------
def sanitize_df_for_polars(df: pd.DataFrame) -> pd.DataFrame:
    """Cast problematic dtypes to Arrow-friendly ones before Polars conversion."""
    if df is None or df.empty:
        return df
    df = df.copy()
    # Ensure datetime for Date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # Convert categoricals, periods, objects with nested types to strings
    for col in df.columns:
        if pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_period_dtype(df[col]):
            df[col] = df[col].astype(str)
        elif df[col].dtype == object:
            # Best-effort: leave primitives, stringify lists/dicts/sets
            sample = df[col].dropna().head(1)
            if not sample.empty and isinstance(sample.iloc[0], (list, dict, set, tuple)):
                df[col] = df[col].astype(str)
    return df

# Display helpers -------------------------------------------------------------
def _show_table_with_cap(df: pd.DataFrame, name: str | None = None, top_n: int = 500):
    """Render df; in large mode cap to top_n rows to keep UI fast."""
    if df is None or df.empty:
        return
    mode = st.session_state.get('upload_mode', 'small')
    if mode == 'large' and len(df) > top_n:
        if name:
            st.info(f"Large dataset mode: showing top {top_n} rows for {name}.")
        st.dataframe(df.head(top_n), use_container_width=True, hide_index=True)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

# Cached maps (vectorized) ----------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def build_winner_map(match_df: pd.DataFrame) -> pd.DataFrame:
    """Return mapping of ['File Name','Date'] -> Winner (team name) and Draw flag."""
    if match_df is None or match_df.empty:
        return pd.DataFrame(columns=['File Name', 'Date', 'Winner', 'Is_Draw'])
    df = match_df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    winner = np.where(df.get('Home_Win', 0) == 1, df['Home_Team'],
               np.where(df.get('Away_Won', 0) == 1, df['Away_Team'], None))
    is_draw = (df.get('Home_Drawn', 0) == 1) | (df.get('Away_Drawn', 0) == 1)
    out = df[['File Name', 'Date']].copy()
    out['Winner'] = winner
    out['Is_Draw'] = is_draw.astype(int)
    return out

@st.cache_data(ttl=600, show_spinner=False)
def build_series_map(match_df: pd.DataFrame) -> pd.DataFrame:
    """Build a vectorized series map: ['File Name','Date'] -> global Series id.
    Series segmentation is per (Home, Away, Format) with <=70-day gaps treated as one series.
    """
    if match_df is None or match_df.empty:
        return pd.DataFrame(columns=['File Name', 'Date', 'Series'])

    potential_series = [
        '1st Test Match','2nd Test Match','3rd Test Match','4th Test Match','5th Test Match',
        '1st One Day International','2nd One Day International','3rd One Day International',
        '4th One Day International','5th One Day International',
        '1st 20 Over International','2nd 20 Over International','3rd 20 Over International',
        '4th 20 Over International','5th 20 Over International',
        'Only One Day International','Only 20 Over International','Only Test Match',
        'Test Championship Final'
    ]
    df = match_df[match_df['Competition'].isin(potential_series)].copy()
    if df.empty:
        return pd.DataFrame(columns=['File Name', 'Date', 'Series'])

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date')

    # Group by pairing+format and start a new block if gap > 70 days
    df['_gap'] = (
        df.sort_values(['Home_Team', 'Away_Team', 'Match_Format', 'Date'])
          .groupby(['Home_Team', 'Away_Team', 'Match_Format'])['Date']
          .diff()
          .dt.days
    )
    df['_block'] = (
    (df['_gap'].fillna(0) > 70)
        .groupby([df['Home_Team'], df['Away_Team'], df['Match_Format']])
        .cumsum()
    )
    # Create a global Series id by ngroup over unique series blocks
    df['Series'] = (
        df.groupby(['Home_Team', 'Away_Team', 'Match_Format', '_block']).ngroup() + 1
    )
    series_map = df[['File Name', 'Date', 'Series', 'Home_Team', 'Away_Team', 'Match_Format']].copy()
    return series_map

# Cached per-game aggregations -----------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def build_batting_per_game(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (Name, Match_Format, File Name, Date) with max Runs."""
    if df is None or df.empty:
        return pd.DataFrame(columns=['Name','Match_Format','File Name','Date','Runs'])
    cols = ['Name','Match_Format','File Name','Date','Runs']
    present = [c for c in cols if c in df.columns]
    if not {'Name','Match_Format','File Name','Date','Runs'}.issubset(set(present)):
        return pd.DataFrame(columns=['Name','Match_Format','File Name','Date','Runs'])
    tmp = df[present].copy()
    tmp['Date'] = pd.to_datetime(tmp['Date'].astype('string'), errors='coerce')
    tmp = tmp.dropna(subset=['Date'])
    try:
        pdf = sanitize_df_for_polars(tmp)
        for c in ['Name','Match_Format','File Name']:
            if c in pdf.columns:
                pdf[c] = pdf[c].astype(str)
        if 'Runs' in pdf.columns:
            pdf['Runs'] = pd.to_numeric(pdf['Runs'], errors='coerce')
        pl_df = pl.from_pandas(pdf)
        out = (
            pl_df
            .group_by(['Name','Match_Format','File Name','Date'])
            .agg([pl.max('Runs').alias('Runs')])
            .sort_by(['Name','Match_Format','Date'])
            .to_pandas()
        )
        return out
    except Exception:
        for c in ['Name','Match_Format','File Name']:
            if c in tmp.columns:
                tmp[c] = tmp[c].astype(str)
        out = (
            tmp.groupby(['Name','Match_Format','File Name','Date'], observed=True, sort=False)
               .agg(Runs=('Runs','max'))
               .reset_index()
               .sort_values(['Name','Match_Format','Date'])
        )
        return out

@st.cache_data(ttl=600, show_spinner=False)
def build_bowling_per_game(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (Name, Match_Format, File Name, Date) with max Bowler_Wkts and a representative Bowl_Team."""
    if df is None or df.empty:
        return pd.DataFrame(columns=['Name','Match_Format','File Name','Date','Bowler_Wkts','Bowl_Team'])
    cols = ['Name','Match_Format','File Name','Date','Bowler_Wkts','Bowl_Team']
    present = [c for c in cols if c in df.columns]
    if not {'Name','Match_Format','File Name','Date','Bowler_Wkts'}.issubset(set(present)):
        return pd.DataFrame(columns=['Name','Match_Format','File Name','Date','Bowler_Wkts','Bowl_Team'])
    tmp = df[present].copy()
    tmp['Date'] = pd.to_datetime(tmp['Date'].astype('string'), errors='coerce')
    tmp = tmp.dropna(subset=['Date'])
    try:
        pdf = sanitize_df_for_polars(tmp)
        for c in ['Name','Match_Format','File Name']:
            if c in pdf.columns:
                pdf[c] = pdf[c].astype(str)
        if 'Bowler_Wkts' in pdf.columns:
            pdf['Bowler_Wkts'] = pd.to_numeric(pdf['Bowler_Wkts'], errors='coerce')
        pl_df = pl.from_pandas(pdf)
        out = (
            pl_df
            .group_by(['Name','Match_Format','File Name','Date'])
            .agg([
                pl.max('Bowler_Wkts').alias('Bowler_Wkts'),
                pl.first('Bowl_Team').alias('Bowl_Team'),
            ])
            .sort_by(['Name','Match_Format','Date'])
            .to_pandas()
        )
        return out
    except Exception:
        for c in ['Name','Match_Format','File Name']:
            if c in tmp.columns:
                tmp[c] = tmp[c].astype(str)
        out = (
            tmp.groupby(['Name','Match_Format','File Name','Date'], observed=True, sort=False)
               .agg(Bowler_Wkts=('Bowler_Wkts','max'), Bowl_Team=('Bowl_Team','first'))
               .reset_index()
               .sort_values(['Name','Match_Format','Date'])
        )
        return out

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
    """Vectorized consecutive 50+/100+ streaks per match (Name, Format),
    counting at most one hit per File Name (game)."""
    if filtered_bat_df is None or filtered_bat_df.empty:
        return pd.DataFrame(columns=['Name', 'Match_Format', 'Consecutive Matches', 'Start Date', 'End Date'])
    # Build a cached per-game view: one row per (Name, Match_Format, File Name, Date)
    per_game = build_batting_per_game(filtered_bat_df)
    if per_game is None or per_game.empty:
        return pd.DataFrame(columns=['Name', 'Match_Format', 'Consecutive Matches', 'Start Date', 'End Date'])
    per_game['hit'] = per_game['Runs'] >= threshold
    # Create a block id that increments on non-hit to segment consecutive hits
    per_game['block'] = (~per_game['hit']).groupby([per_game['Name'], per_game['Match_Format']]).cumsum()
    grp = per_game[per_game['hit']].groupby(['Name', 'Match_Format', 'block'])
    streaks = grp.agg(Start=('Date', 'min'), End=('Date', 'max'), Count=('hit', 'size')).reset_index()
    # Best streak per (Name, Format)
    best = streaks.sort_values(['Name', 'Match_Format', 'Count'], ascending=[True, True, False]) \
                 .groupby(['Name', 'Match_Format'], as_index=False).first()
    best = best[best['Count'] >= 2]
    if best.empty:
        return pd.DataFrame(columns=['Name', 'Match_Format', 'Consecutive Matches', 'Start Date', 'End Date'])
    out = best[['Name', 'Match_Format', 'Count', 'Start', 'End']].rename(columns={'Count': 'Consecutive Matches'})
    out['Start Date'] = pd.to_datetime(out['Start']).dt.strftime('%d/%m/%Y')
    out['End Date'] = pd.to_datetime(out['End']).dt.strftime('%d/%m/%Y')
    out = out[['Name', 'Match_Format', 'Consecutive Matches', 'Start Date', 'End Date']]
    return out.sort_values('Consecutive Matches', ascending=False)

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

    try:
        pdf = sanitize_df_for_polars(filtered_bat_df)
        use_cols = ['Name', 'Date', 'Home Team', 'Bat_Team_y', 'Bowl_Team_y',
                    'Runs', 'Innings', 'Balls', '4s', '6s']
        pdf = pdf[[c for c in use_cols if c in pdf.columns]].copy()
        # Ensure numeric types where needed
        for c in ['Runs', 'Innings', 'Balls', '4s', '6s']:
            if c in pdf.columns:
                pdf[c] = pd.to_numeric(pdf[c], errors='coerce')
        pl_df = pl.from_pandas(pdf)

        cond_12 = pl.col('Innings').is_in([1, 2])
        cond_34 = pl.col('Innings').is_in([3, 4])

        agg = (
            pl_df
            .group_by(['Name', 'Date', 'Home Team'])
            .agg([
                pl.first('Bat_Team_y'),
                pl.first('Bowl_Team_y'),
                pl.sum('Balls'),
                pl.sum('4s'),
                pl.sum('6s'),
                pl.max(pl.when(cond_12).then(pl.col('Runs')).otherwise(None)).alias('1st Innings'),
                pl.max(pl.when(cond_34).then(pl.col('Runs')).otherwise(None)).alias('2nd Innings'),
            ])
        )

        out = (
            agg
            .filter((pl.col('1st Innings').fill_null(0) >= 100) & (pl.col('2nd Innings').fill_null(0) >= 100))
            .rename({
                'Bat_Team_y': 'Bat Team',
                'Bowl_Team_y': 'Bowl Team',
                'Home Team': 'Country'
            })
            .select(['Name', 'Bat Team', 'Bowl Team', 'Country', '1st Innings', '2nd Innings',
                     'Balls', '4s', '6s', 'Date'])
            .sort(pl.col('Date').cast(pl.Datetime), descending=True)
            .with_columns(pl.col('Date').cast(pl.Datetime).dt.strftime('%d/%m/%Y'))
        )
        return out.to_pandas()
    except Exception:
        # Fallback to original implementation if anything goes wrong
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
    """Create a single-trace centuries scatter plot, capped in large mode."""
    if filtered_bat_df is None or filtered_bat_df.empty:
        return None

    df = filtered_bat_df[filtered_bat_df['Runs'] >= 100].copy()
    if df.empty:
        return None

    mode = st.session_state.get('upload_mode', 'small')
    if mode == 'large' and len(df) > 2000:
        df = df.nlargest(2000, 'Runs')

    hovertext = (
        df['Name'].astype(str) +
        '<br>Runs: ' + df['Runs'].astype(str) +
        '<br>Balls: ' + df['Balls'].astype(str) +
        '<br>4s: ' + df['4s'].astype(str) +
        '<br>6s: ' + df['6s'].astype(str)
    )

    fig = go.Figure(
        data=[go.Scatter(
            x=df['Balls'],
            y=df['Runs'],
            mode='markers',
            text=hovertext,
            hovertemplate='%{text}<extra></extra>',
            marker=dict(size=8, opacity=0.8)
        )]
    )
    fig.update_layout(
        xaxis_title='Balls Faced',
        yaxis_title='Runs Scored',
        height=600,
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
# Quick refresh to clear caches and reload computations
with st.expander("⚙️ Records options", expanded=False):
    if st.button("Reload Records", use_container_width=True):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.rerun()

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
        _show_table_with_cap(highest_scores_df, name='Highest Scores')

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
        _show_table_with_cap(fifties_streak_df, name='50+ Streaks')

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
        _show_table_with_cap(centuries_streak_df, name='Consecutive Centuries')

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
        _show_table_with_cap(not_out_99s_df, name='99* Club')

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
        _show_table_with_cap(carrying_bat_df, name='Carrying the Bat')

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
        _show_table_with_cap(hundreds_both_df, name='Hundreds in Both Innings')

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
        _show_table_with_cap(position_scores_df, name='Position Records')

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
    """Vectorized consecutive 5+ wicket streaks per match (Name, Format),
    counting at most one hit per File Name (game)."""
    if filtered_bowl_df is None or filtered_bowl_df.empty:
        return pd.DataFrame(columns=['Name', 'Match_Format', 'Bowl_Team', 'Consecutive Matches', 'Start Date', 'End Date'])
    per_game = build_bowling_per_game(filtered_bowl_df)
    if per_game is None or per_game.empty:
        return pd.DataFrame(columns=['Name', 'Match_Format', 'Bowl_Team', 'Consecutive Matches', 'Start Date', 'End Date'])
    # Streak segmentation on per-game hits
    per_game['hit'] = per_game['Bowler_Wkts'] >= 5
    per_game['block'] = (~per_game['hit']).groupby([per_game['Name'], per_game['Match_Format']]).cumsum()
    grp = per_game[per_game['hit']].groupby(['Name','Match_Format','block'])
    agg = grp.agg(Start=('Date','min'), End=('Date','max'), Count=('hit','size'), Bowl_Team=('Bowl_Team','first')).reset_index()
    best = agg.sort_values(['Name','Match_Format','Count'], ascending=[True, True, False]) \
             .groupby(['Name','Match_Format'], as_index=False).first()
    best = best[best['Count'] >= 2]
    if best.empty:
        return pd.DataFrame(columns=['Name', 'Match_Format', 'Bowl_Team', 'Consecutive Matches', 'Start Date', 'End Date'])
    out = best[['Name','Match_Format','Bowl_Team','Count','Start','End']].rename(columns={'Count':'Consecutive Matches'})
    out['Start Date'] = pd.to_datetime(out['Start']).dt.strftime('%d/%m/%Y')
    out['End Date'] = pd.to_datetime(out['End']).dt.strftime('%d/%m/%Y')
    out = out[['Name','Match_Format','Bowl_Team','Consecutive Matches','Start Date','End Date']]
    return out.sort_values('Consecutive Matches', ascending=False)

def process_five_wickets_both(filtered_bowl_df):
    """Process 5+ wickets in both innings data"""
    if filtered_bowl_df is None or filtered_bowl_df.empty:
        return pd.DataFrame()

    try:
        pdf = sanitize_df_for_polars(filtered_bowl_df)
        use_cols = ['Name', 'Date', 'Home_Team', 'Bat_Team', 'Bowl_Team', 'Match_Format',
                    'Innings', 'Bowler_Wkts', 'Bowler_Runs', 'Bowler_Overs']
        pdf = pdf[[c for c in use_cols if c in pdf.columns]].copy()
        # Cast to numeric robustly
        for c in ['Innings', 'Bowler_Wkts', 'Bowler_Runs', 'Bowler_Overs']:
            if c in pdf.columns:
                pdf[c] = pd.to_numeric(pdf[c], errors='coerce')
        pl_df = pl.from_pandas(pdf)

        cond_12 = pl.col('Innings').is_in([1, 2])
        cond_34 = pl.col('Innings').is_in([3, 4])

        agg = (
            pl_df
            .group_by(['Name', 'Date', 'Home_Team'])
            .agg([
                pl.first('Bat_Team'),
                pl.first('Bowl_Team'),
                pl.first('Match_Format'),
                pl.max(pl.when(cond_12).then(pl.col('Bowler_Wkts')).otherwise(None)).alias('1st Innings Wkts'),
                pl.max(pl.when(cond_34).then(pl.col('Bowler_Wkts')).otherwise(None)).alias('2nd Innings Wkts'),
                pl.min(pl.when(cond_12 & (pl.col('Bowler_Wkts') >= 5)).then(pl.col('Bowler_Runs')).otherwise(None)).alias('1st Innings Runs'),
                pl.min(pl.when(cond_34 & (pl.col('Bowler_Wkts') >= 5)).then(pl.col('Bowler_Runs')).otherwise(None)).alias('2nd Innings Runs'),
                pl.col('Bowler_Overs').filter(cond_12 & (pl.col('Bowler_Wkts') >= 5)).alias('overs12_list'),
                pl.col('Bowler_Overs').filter(cond_34 & (pl.col('Bowler_Wkts') >= 5)).alias('overs34_list'),
            ])
            .with_columns([
                pl.when(pl.col('overs12_list').list.len() > 0).then(pl.col('overs12_list').list.first()).otherwise(0).alias('1st Innings Overs'),
                pl.when(pl.col('overs34_list').list.len() > 0).then(pl.col('overs34_list').list.first()).otherwise(0).alias('2nd Innings Overs'),
            ])
        )

        out = (
            agg
            .filter((pl.col('1st Innings Wkts').fill_null(0) >= 5) & (pl.col('2nd Innings Wkts').fill_null(0) >= 5))
            .rename({
                'Home_Team': 'Country',
                'Bat_Team': 'Bat Team',
                'Bowl_Team': 'Bowl Team',
                'Match_Format': 'Format'
            })
            .select(['Name', 'Bat Team', 'Bowl Team', 'Country', 'Format',
                     '1st Innings Wkts', '1st Innings Runs', '1st Innings Overs',
                     '2nd Innings Wkts', '2nd Innings Runs', '2nd Innings Overs',
                     'Date'])
            .sort(pl.col('Date').cast(pl.Datetime), descending=True)
            .with_columns(pl.col('Date').cast(pl.Datetime).dt.strftime('%d/%m/%Y'))
        )
        return out.to_pandas()
    except Exception:
        # Memory-safe pandas fallback using part mapping and pivots
        df = filtered_bowl_df.copy()
        for c in ['Innings', 'Bowler_Wkts', 'Bowler_Runs', 'Bowler_Overs']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df['part'] = np.where(df['Innings'].isin([1, 2]), 'first', np.where(df['Innings'].isin([3, 4]), 'second', None))
        df = df[df['part'].notna()]

        idx = ['Name', 'Date', 'Home_Team']
        # Ensure non-categorical groupers to avoid cartesian explosion
        for g in ['Name', 'Home_Team', 'Bat_Team', 'Bowl_Team', 'Match_Format']:
            if g in df.columns and g != 'Date':
                df[g] = df[g].astype(str)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        keep_first = df.groupby(idx, observed=True, sort=False).agg({'Bat_Team': 'first', 'Bowl_Team': 'first', 'Match_Format': 'first'})

        wkts = df.groupby(idx + ['part'], observed=True, sort=False)['Bowler_Wkts'].max().unstack('part').fillna(0)
        runs_first = df[(df['Bowler_Wkts'] >= 5)].groupby(idx + ['part'], observed=True, sort=False)['Bowler_Runs'].min().unstack('part')
        overs_first = df[(df['Bowler_Wkts'] >= 5)].groupby(idx + ['part'], observed=True, sort=False)['Bowler_Overs'].first().unstack('part')

        out = keep_first.join(wkts, how='left').join(runs_first, rsuffix='_runs', how='left').join(overs_first, rsuffix='_overs', how='left')
        out = out.reset_index()
        out = out.rename(columns={
            'first': '1st Innings Wkts', 'second': '2nd Innings Wkts',
            'first_runs': '1st Innings Runs', 'second_runs': '2nd Innings Runs',
            'first_overs': '1st Innings Overs', 'second_overs': '2nd Innings Overs'
        })
        out = out[(out['1st Innings Wkts'] >= 5) & (out['2nd Innings Wkts'] >= 5)]
        out = out.rename(columns={'Home_Team': 'Country', 'Bat_Team': 'Bat Team', 'Bowl_Team': 'Bowl Team', 'Match_Format': 'Format'})
        out['Date'] = pd.to_datetime(out['Date'], errors='coerce').dt.strftime('%d/%m/%Y')
        out = out[['Name', 'Bat Team', 'Bowl Team', 'Country', 'Format', '1st Innings Wkts', '1st Innings Runs', '1st Innings Overs', '2nd Innings Wkts', '2nd Innings Runs', '2nd Innings Overs', 'Date']]
        return out.sort_values('Date', ascending=False)

def process_match_bowling(filtered_bowl_df):
    """Process match bowling figures data"""
    if filtered_bowl_df is None or filtered_bowl_df.empty:
        return pd.DataFrame()

    try:
        pdf = sanitize_df_for_polars(filtered_bowl_df)
        use_cols = ['Name', 'Date', 'Home_Team', 'Bat_Team', 'Bowl_Team', 'Match_Format',
                    'Innings', 'Bowler_Wkts', 'Bowler_Runs']
        pdf = pdf[[c for c in use_cols if c in pdf.columns]].copy()
        for c in ['Innings', 'Bowler_Wkts', 'Bowler_Runs']:
            if c in pdf.columns:
                pdf[c] = pd.to_numeric(pdf[c], errors='coerce')
        pl_df = pl.from_pandas(pdf)

        cond_12 = pl.col('Innings').is_in([1, 2])
        cond_34 = pl.col('Innings').is_in([3, 4])

        agg = (
            pl_df
            .group_by(['Name', 'Date', 'Home_Team', 'Bat_Team', 'Bowl_Team', 'Match_Format'])
            .agg([
                pl.max(pl.when(cond_12).then(pl.col('Bowler_Wkts')).otherwise(0)).alias('1st Innings Wkts'),
                pl.max(pl.when(cond_34).then(pl.col('Bowler_Wkts')).otherwise(0)).alias('2nd Innings Wkts'),
                pl.sum(pl.when(cond_12).then(pl.col('Bowler_Runs')).otherwise(0)).alias('1st Innings Runs'),
                pl.sum(pl.when(cond_34).then(pl.col('Bowler_Runs')).otherwise(0)).alias('2nd Innings Runs'),
            ])
            .with_columns([
                (pl.col('1st Innings Wkts') + pl.col('2nd Innings Wkts')).alias('Match Wickets'),
                (pl.col('1st Innings Runs') + pl.col('2nd Innings Runs')).alias('Match Runs'),
            ])
        )

        out = (
            agg
            .rename({
                'Home_Team': 'Country',
                'Bat_Team': 'Bat Team',
                'Bowl_Team': 'Bowl Team',
                'Match_Format': 'Format'
            })
            .select(['Name', 'Bat Team', 'Bowl Team', 'Country', 'Format',
                     '1st Innings Wkts', '1st Innings Runs', '2nd Innings Wkts', '2nd Innings Runs',
                     'Match Wickets', 'Match Runs', 'Date'])
            .with_columns(pl.col('Date').cast(pl.Datetime))
            .filter(pl.col('Match Wickets') >= 10)
            .sort(['Match Wickets', 'Match Runs'], descending=[True, False])
            .with_columns(pl.col('Date').dt.strftime('%d/%m/%Y'))
        )
        return out.to_pandas()
    except Exception:
        # Memory-safe pandas fallback using part mapping and pivot
        df = filtered_bowl_df.copy()
        for c in ['Innings', 'Bowler_Wkts', 'Bowler_Runs']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df['part'] = np.where(df['Innings'].isin([1, 2]), 'first', np.where(df['Innings'].isin([3, 4]), 'second', None))
        df = df[df['part'].notna()]
        # Ensure non-categorical groupers
        for g in ['Name', 'Home_Team', 'Bat_Team', 'Bowl_Team', 'Match_Format']:
            if g in df.columns and g != 'Date':
                df[g] = df[g].astype(str)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        idx = ['Name', 'Date', 'Home_Team', 'Bat_Team', 'Bowl_Team', 'Match_Format']
        # Process per-player to keep intermediate groupby small
        chunks = []
        for name, sub in df.groupby('Name', sort=False):
            agg = sub.groupby(['Date', 'Home_Team', 'Bat_Team', 'Bowl_Team', 'Match_Format', 'part'], observed=True, sort=False) \
                    .agg({'Bowler_Wkts': 'max', 'Bowler_Runs': 'sum'}) \
                    .reset_index()
            wkts = agg.pivot_table(index=['Date', 'Home_Team', 'Bat_Team', 'Bowl_Team', 'Match_Format'], columns='part', values='Bowler_Wkts', aggfunc='max', observed=True).fillna(0)
            runs = agg.pivot_table(index=['Date', 'Home_Team', 'Bat_Team', 'Bowl_Team', 'Match_Format'], columns='part', values='Bowler_Runs', aggfunc='sum', observed=True).fillna(0)
            out_sub = wkts.join(runs, lsuffix='_wkts', rsuffix='_runs').reset_index()
            out_sub.insert(0, 'Name', name)
            chunks.append(out_sub)
        if not chunks:
            return pd.DataFrame(columns=['Name', 'Bat Team', 'Bowl Team', 'Country', 'Format', '1st Innings Wkts', '1st Innings Runs', '2nd Innings Wkts', '2nd Innings Runs', 'Match Wickets', 'Match Runs', 'Date'])
        out = pd.concat(chunks, ignore_index=True)
        out = out.rename(columns={
            'first_wkts': '1st Innings Wkts', 'second_wkts': '2nd Innings Wkts',
            'first_runs': '1st Innings Runs', 'second_runs': '2nd Innings Runs'
        })
        out['Match Wickets'] = out['1st Innings Wkts'] + out['2nd Innings Wkts']
        out['Match Runs'] = out['1st Innings Runs'] + out['2nd Innings Runs']
        out = out.rename(columns={'Home_Team': 'Country', 'Bat_Team': 'Bat Team', 'Bowl_Team': 'Bowl Team', 'Match_Format': 'Format'})
        out['Date'] = pd.to_datetime(out['Date'], errors='coerce').dt.strftime('%d/%m/%Y')
        out = out[['Name', 'Bat Team', 'Bowl Team', 'Country', 'Format', '1st Innings Wkts', '1st Innings Runs', '2nd Innings Wkts', '2nd Innings Runs', 'Match Wickets', 'Match Runs', 'Date']]
        out = out[out['Match Wickets'] >= 10]
        return out.sort_values(['Match Wickets', 'Match Runs'], ascending=[False, True])

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
        _show_table_with_cap(best_bowling_df, name='Best Bowling')

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
        _show_table_with_cap(consecutive_five_wickets_df, name='5w Streaks')

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
        _show_table_with_cap(five_wickets_both_df, name='5+ in Both Innings')

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
        _show_table_with_cap(best_match_bowling_df, name='Best Match Bowling', top_n=500)
    else:
        st.info("No bowling records available.")

# Match Records Functions
###############################################################################
def process_wins_data(filtered_match_df, win_type='runs', margin_type='big'):
    """Process wins data by type and margin"""
    if filtered_match_df is None or filtered_match_df is None or filtered_match_df.empty:
        return pd.DataFrame()
    # Work on a copy and coerce comparison columns to numeric to avoid categorical comparison errors
    df = filtered_match_df.copy()
    for col in ['Margin_Runs', 'Margin_Wickets', 'Innings_Win', 'Home_Drawn', 'Home_Win', 'Away_Won']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Rest of the function remains the same
    # Set conditions based on win type
    if win_type == 'runs':
        base_conditions = [
            (df['Margin_Runs'] > 0),
            (df['Innings_Win'] == 0),
            (df['Home_Drawn'] != 1)
        ]
        margin_column = 'Margin_Runs'
        margin_name = 'Runs'
    else:  # wickets
        base_conditions = [
            (df['Margin_Wickets'] > 0),
            (df['Innings_Win'] == 0),
            (df['Home_Drawn'] != 1)
        ]
        margin_column = 'Margin_Wickets'
        margin_name = 'Wickets'
    
    df = df[np.all(base_conditions, axis=0)].copy()
    
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
    # Coerce to numeric to avoid comparison errors on categoricals
    df = filtered_match_df.copy()
    for col in ['Innings_Win', 'Home_Drawn', 'Home_Win', 'Margin_Runs']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df = df[(df['Innings_Win'] == 1) & (df['Home_Drawn'] != 1)].copy()
    
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
            _show_table_with_cap(bigwin_df, name='Biggest Wins (Runs)')

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
            _show_table_with_cap(bigwin_wickets_df, name='Biggest Wins (Wickets)')

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
            _show_table_with_cap(bigwin_innings_df, name='Biggest Wins (Innings)')

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
            _show_table_with_cap(narrow_wins_df, name='Narrowest Wins (Runs)')

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
            _show_table_with_cap(narrowwin_wickets_df, name='Narrowest Wins (Wickets)')
    else:
        st.info("No match records available.")

# Game Records Functions
###############################################################################
def get_match_details(filtered_game_df, filtered_match_df):
    """Process match details and add Result and Margin columns"""
    if filtered_game_df is None or filtered_match_df is None:
        return filtered_game_df

    try:
        # Align Date types for safe merges
        if 'Date' in filtered_game_df.columns:
            gdf = filtered_game_df.copy()
            gdf['Date'] = gdf['Date'].astype(str)
        else:
            gdf = filtered_game_df.copy()
        mdf_base = filtered_match_df.copy()
        if 'Date' in mdf_base.columns:
            mdf_base['Date'] = mdf_base['Date'].astype(str)
        # Prepare oriented match mapping (two rows per match: home perspective and away perspective)
        mcols_needed = ['Date', 'Home_Team', 'Away_Team', 'Home_Win', 'Home_Lost', 'Home_Drawn',
                        'Away_Won', 'Away_Lost', 'Away_Drawn', 'Innings_Win', 'Margin_Runs', 'Margin_Wickets']
        mdf = mdf_base[[c for c in mcols_needed if c in mdf_base.columns]].copy()

        home_map = mdf.copy()
        home_map['Bat_Team'] = home_map['Home_Team']
        home_map['Bowl_Team'] = home_map['Away_Team']

        away_map = mdf.copy()
        away_map['Bat_Team'] = away_map['Away_Team']
        away_map['Bowl_Team'] = away_map['Home_Team']

        orient = pd.concat([home_map, away_map], ignore_index=True)

        # Compute Result from batting team's perspective
        win = (
            ((orient['Bat_Team'] == orient['Home_Team']) & (orient.get('Home_Win', 0) == 1)) |
            ((orient['Bat_Team'] == orient['Away_Team']) & (orient.get('Away_Won', 0) == 1))
        )
        lost = (
            ((orient['Bat_Team'] == orient['Home_Team']) & (orient.get('Home_Lost', 0) == 1)) |
            ((orient['Bat_Team'] == orient['Away_Team']) & (orient.get('Away_Lost', 0) == 1))
        )
        drawn = (orient.get('Home_Drawn', 0) == 1) | (orient.get('Away_Drawn', 0) == 1)

        orient['Result'] = np.select([win, lost, drawn], ['Win', 'Lost', 'Draw'], default='Unknown')

        # Bring over numeric margins
        orient['Innings_Win'] = orient['Innings_Win']
        orient['Margin_Runs'] = orient['Margin_Runs']
        orient['Margin_Wickets'] = orient['Margin_Wickets']

        # Build margin text like the original logic
        innings_mask = orient['Innings_Win'] == 1
        wickets_mask = orient['Margin_Wickets'].fillna(0) > 0
        runs_mask = orient['Margin_Runs'].fillna(0) > 0

        margin = np.full(len(orient), '-', dtype=object)
        # innings first (can include runs)
        margin[innings_mask] = 'by an innings'
        # add runs for innings where present
        with_runs = innings_mask & runs_mask
        margin[with_runs] = [f"by an innings and {int(r)} runs" for r in orient.loc[with_runs, 'Margin_Runs']]
        # non-innings wickets/runs
        normal_mask = ~innings_mask
        wickets_only = normal_mask & wickets_mask
        runs_only = normal_mask & ~wickets_mask & runs_mask
        margin[wickets_only] = [f"by {int(w)} wickets" for w in orient.loc[wickets_only, 'Margin_Wickets']]
        margin[runs_only] = [f"by {int(r)} runs" for r in orient.loc[runs_only, 'Margin_Runs']]

        orient['Margin'] = margin

        # Join to game_df on Date + Bat_Team + Bowl_Team
        result_df = gdf.merge(
            orient[['Date', 'Bat_Team', 'Bowl_Team', 'Result', 'Margin', 'Innings_Win', 'Margin_Runs', 'Margin_Wickets']],
            on=['Date', 'Bat_Team', 'Bowl_Team'], how='left'
        )
        return result_df
    except Exception:
        # Fallback to original per-row apply
        def process_row(row):
            bat_team = row['Bat_Team']
            bowl_team = row['Bowl_Team']
            result = 'Unknown'
            margin_info = '-'
            innings_win = 0
            margin_runs = None
            margin_wickets = None
            home_match = filtered_match_df[(filtered_match_df['Home_Team'] == bat_team) & (filtered_match_df['Away_Team'] == bowl_team)]
            away_match = filtered_match_df[(filtered_match_df['Away_Team'] == bat_team) & (filtered_match_df['Home_Team'] == bowl_team)]
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

        result_df = filtered_game_df.copy()
        result_df[['Result', 'Margin', 'Innings_Win', 'Margin_Runs', 'Margin_Wickets']] = \
            filtered_game_df.apply(process_row, axis=1)
        return result_df

def get_highest_chases(processed_game_df, filtered_match_df):
    """Process highest successful run chases (vectorized)."""
    if processed_game_df is None or filtered_match_df is None or processed_game_df.empty:
        return pd.DataFrame()

    try:
        df = processed_game_df.copy()
        # Determine chase innings by format
        is_redball = df['Match_Format'].isin(['Test Match', 'First Class'])
        chase_innings = (is_redball & (df['Innings'] == 4)) | (~is_redball & (df['Innings'] == 2))
        # Successful when batting team's perspective result is Win
        success = df['Result'] == 'Win'
        chases = df[chase_innings & success]
        # Sort by runs desc and select/rename
        chase_df = chases.sort_values('Total_Runs', ascending=False)[
            ['Bat_Team', 'Bowl_Team', 'Innings', 'Total_Runs', 'Wickets', 'Overs',
             'Run_Rate', 'Competition', 'Match_Format', 'Date', 'Margin']
        ].rename(columns={
            'Bat_Team': 'Bat Team', 'Bowl_Team': 'Bowl Team', 'Total_Runs': 'Runs',
            'Run_Rate': 'Run Rate', 'Match_Format': 'Format'
        })
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
                _show_table_with_cap(highest_scores_df, name='Highest Team Scores')
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
                _show_table_with_cap(lowest_scores_df, name='Lowest Team Scores')

 

            # Lowest First Innings Wins
            def process_lowest_first_innings_wins(processed_game_df, filtered_match_df):
                """Lowest first-innings winning scores using winner_map; type-safe Date merge."""
                if processed_game_df is None or filtered_match_df is None or processed_game_df.empty:
                    return pd.DataFrame()
                g = processed_game_df.copy()
                m = filtered_match_df.copy()
                # Align join keys
                if 'Date' in g.columns:
                    g['Date'] = g['Date'].astype(str)
                if 'Date' in m.columns:
                    m['Date'] = m['Date'].astype(str)
                merged_df = g.merge(m, on=['File Name','Date'], how='left')
                # Winner map
                wmap = build_winner_map(filtered_match_df)
                if not wmap.empty:
                    wmap2 = wmap.copy()
                    wmap2['Date'] = wmap2['Date'].astype(str)
                    merged_df = merged_df.merge(wmap2[['File Name','Date','Winner','Is_Draw']], on=['File Name','Date'], how='left')
                # Filter: innings 1, not draw, winner equals batting team
                mask = (
                    (merged_df['Innings'] == 1) &
                    (merged_df.get('Is_Draw', 0) == 0) &
                    (merged_df.get('Winner').notna()) &
                    (merged_df['Winner'] == merged_df['Bat_Team'])
                )
                out = merged_df.loc[mask, ['Bat_Team','Bowl_Team','Innings','Total_Runs','Overs','Wickets','Run_Rate','Competition_x','Match_Format_x','Player_of_the_Match_x','Date','Margin_y']].copy()
                # Format and rename
                out['Date'] = pd.to_datetime(out['Date'], errors='coerce').dt.strftime('%d/%m/%Y')
                out.columns = ['Bat Team','Bowl Team','Innings','Runs','Overs','Wickets','Run Rate','Competition','Format','Player of the Match','Date','Margin']
                return out.sort_values('Runs', ascending=True)

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
                _show_table_with_cap(low_wins_df, name='Lowest 1st Innings Winning Scores')

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
                _show_table_with_cap(chases_df, name='Highest Successful Run Chases')

            def process_first_innings_deficit_wins(proc_game_df, proc_match_df):
                """Wins after first innings deficit using winner_map; type-safe Date merge."""
                if proc_game_df is None or proc_match_df is None or proc_game_df.empty:
                    return pd.DataFrame()
                g = proc_game_df.copy(); m = proc_match_df.copy()
                if 'Date' in g.columns: g['Date'] = g['Date'].astype(str)
                if 'Date' in m.columns: m['Date'] = m['Date'].astype(str)
                df = g.merge(m, on=['File Name','Date'], how='left')
                # Get first/second innings totals per match
                fi = df[df['Innings']==1][['File Name','Date','Bat_Team','Total_Runs']].rename(columns={'Bat_Team':'Team1','Total_Runs':'First_Total'})
                si = df[df['Innings']==2][['File Name','Date','Bat_Team','Total_Runs']].rename(columns={'Bat_Team':'Team2','Total_Runs':'Second_Total'})
                match_i = fi.merge(si, on=['File Name','Date'], how='inner')
                # Winner
                wmap = build_winner_map(proc_match_df)
                if not wmap.empty:
                    w2 = wmap.copy(); w2['Date'] = w2['Date'].astype(str)
                    match_i = match_i.merge(w2[['File Name','Date','Winner','Is_Draw']], on=['File Name','Date'], how='left')
                # Determine leader after first innings and deficit for winner's first-innings team
                match_i['LeaderAfter1st'] = np.where(match_i['First_Total'] >= match_i['Second_Total'], match_i['Team1'], match_i['Team2'])
                match_i['Deficit'] = np.where(match_i['Winner'] == match_i['Team1'], match_i['First_Total'] - match_i['Second_Total'], match_i['Second_Total'] - match_i['First_Total'])
                # Keep only matches where winner trailed after 1st inns (i.e., Winner != LeaderAfter1st) and not draws
                match_i = match_i[(match_i.get('Is_Draw',0)==0) & match_i['Winner'].notna() & (match_i['Winner'] != match_i['LeaderAfter1st'])]
                # Determine the innings number of the winner's first-innings (1 if Team1 else 2)
                match_i['Desired_Innings'] = np.where(match_i['Winner'] == match_i['Team1'], 1, 2)
                # Join back the exact row from game data
                base = df[df['Innings'].isin([1,2])][['File Name','Date','Innings','Bat_Team','Bowl_Team','Total_Runs','Overs','Wickets','Run_Rate','Competition_x','Match_Format_x','Player_of_the_Match_x','Margin_y']]
                disp = base.merge(match_i[['File Name','Date','Desired_Innings','Winner','Deficit']], left_on=['File Name','Date','Innings'], right_on=['File Name','Date','Desired_Innings'], how='inner')
                # Ensure the row matches the winner team
                disp = disp[disp['Bat_Team'] == disp['Winner']]
                disp = disp.drop(columns=['Desired_Innings','Winner'])
                disp['Date'] = pd.to_datetime(disp['Date'], errors='coerce').dt.strftime('%d/%m/%Y')
                disp = disp.rename(columns={'Bat_Team':'Batting Team','Bowl_Team':'Bowling Team','Total_Runs':'Runs','Run_Rate':'Run Rate','Competition_x':'Competition','Match_Format_x':'Format','Player_of_the_Match_x':'Player of the Match','Margin_y':'Margin','Deficit':'Deficit'})
                # Only negative deficits (behind)
                disp = disp[disp['Deficit'] < 0].sort_values('Deficit', ascending=True)
                return disp

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
                _show_table_with_cap(deficit_df, name='Wins after 1st Innings Deficit')

        except Exception as e:
            st.error(f"Error processing records: {str(e)}")

        # NEW LEAD FROM 1st INNINGS AND LOST
        def process_first_innings_lead_losses(proc_game_df, proc_match_df):
            """Losses after first innings lead using winner_map; type-safe Date merge."""
            if proc_game_df is None or proc_match_df is None or proc_game_df.empty:
                return pd.DataFrame()
            g = proc_game_df.copy(); m = proc_match_df.copy()
            if 'Date' in g.columns: g['Date'] = g['Date'].astype(str)
            if 'Date' in m.columns: m['Date'] = m['Date'].astype(str)
            df = g.merge(m, on=['File Name','Date'], how='left')
            # First/Second totals and teams per match
            fi = df[df['Innings']==1][['File Name','Date','Bat_Team','Total_Runs']].rename(columns={'Bat_Team':'Team1','Total_Runs':'First_Total'})
            si = df[df['Innings']==2][['File Name','Date','Bat_Team','Total_Runs']].rename(columns={'Bat_Team':'Team2','Total_Runs':'Second_Total'})
            match_i = fi.merge(si, on=['File Name','Date'], how='inner')
            wmap = build_winner_map(proc_match_df)
            if not wmap.empty:
                w2 = wmap.copy(); w2['Date'] = w2['Date'].astype(str)
                match_i = match_i.merge(w2[['File Name','Date','Winner','Is_Draw']], on=['File Name','Date'], how='left')
            # Leader after first innings and lead size
            match_i['LeaderAfter1st'] = np.where(match_i['First_Total'] >= match_i['Second_Total'], match_i['Team1'], match_i['Team2'])
            match_i['Lead'] = np.abs(match_i['First_Total'] - match_i['Second_Total'])
            # Keep matches where leader lost and not draws
            match_i = match_i[(match_i.get('Is_Draw',0)==0) & match_i['Winner'].notna() & (match_i['Winner'] != match_i['LeaderAfter1st'])]
            # Determine the innings number corresponding to the leader's first-innings
            match_i['Desired_Innings'] = np.where(match_i['LeaderAfter1st'] == match_i['Team1'], 1, 2)
            base = df[df['Innings'].isin([1,2])][['File Name','Date','Innings','Bat_Team','Bowl_Team','Total_Runs','Overs','Wickets','Run_Rate','Competition_x','Match_Format_x','Player_of_the_Match_x','Margin_y']]
            disp = base.merge(match_i[['File Name','Date','Desired_Innings','LeaderAfter1st','Lead']], left_on=['File Name','Date','Innings'], right_on=['File Name','Date','Desired_Innings'], how='inner')
            disp = disp[disp['Bat_Team'] == disp['LeaderAfter1st']]
            disp = disp.drop(columns=['Desired_Innings','LeaderAfter1st'])
            disp['Date'] = pd.to_datetime(disp['Date'], errors='coerce').dt.strftime('%d/%m/%Y')
            disp = disp.rename(columns={'Bat_Team':'Batting Team','Bowl_Team':'Bowling Team','Total_Runs':'Runs','Run_Rate':'Run Rate','Competition_x':'Competition','Match_Format_x':'Format','Player_of_the_Match_x':'Player of the Match','Margin_y':'Margin','Lead':'Lead'})
            disp = disp.sort_values('Lead', ascending=False)
            return disp

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
                _show_table_with_cap(lead_losses_df, name='Losses after 1st Innings Leads')
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

    # Build vectorized maps
    series_info_df = pd.DataFrame()
    smap = pd.DataFrame()
    wmap = pd.DataFrame()
    if filtered_match_df is not None and not filtered_match_df.empty:
        smap = build_series_map(filtered_match_df)
        wmap = build_winner_map(filtered_match_df)
        # Align key types for safe merges
        if 'Date' in smap.columns:
            smap['Date'] = smap['Date'].astype(str)
        if 'Date' in wmap.columns:
            wmap['Date'] = wmap['Date'].astype(str)
        if not smap.empty:
            # Series overview: join maps
            over = smap.merge(wmap, on=['File Name', 'Date'], how='left')
            over['Date'] = pd.to_datetime(over['Date'], errors='coerce')
            grp = over.groupby(['Series', 'Home_Team', 'Away_Team', 'Match_Format'], sort=False)
            series_info_df = grp.agg(
                Start_Date=('Date', 'min'),
                End_Date=('Date', 'max'),
                Games_Played=('Date', 'size'),
                Total_Home_Wins=('Winner', lambda s: (s == over.loc[s.index, 'Home_Team']).sum()),
                Total_Away_Wins=('Winner', lambda s: (s == over.loc[s.index, 'Away_Team']).sum()),
                Total_Draws=('Is_Draw', 'sum'),
            ).reset_index()
            def _series_result(row):
                if row['Total_Home_Wins'] > row['Total_Away_Wins']:
                    return f"{row['Home_Team']} won {row['Total_Home_Wins']}-{row['Total_Away_Wins']}"
                if row['Total_Away_Wins'] > row['Total_Home_Wins']:
                    return f"{row['Away_Team']} won {row['Total_Away_Wins']}-{row['Total_Home_Wins']}"
                return f"Series Drawn {row['Total_Home_Wins']}-{row['Total_Away_Wins']}" if row['Total_Draws'] > 0 else f"Series Tied {row['Total_Home_Wins']}-{row['Total_Away_Wins']}"
            series_info_df['Series_Result'] = series_info_df.apply(_series_result, axis=1)
            series_info_df = series_info_df.sort_values('Start_Date')
            # Format dates for display
            series_info_df['Start_Date'] = pd.to_datetime(series_info_df['Start_Date']).dt.strftime('%d/%m/%Y')
            series_info_df['End_Date'] = pd.to_datetime(series_info_df['End_Date']).dt.strftime('%d/%m/%Y')

    # Series Info Section
    if series_info_df is not None and not series_info_df.empty:
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
        _show_table_with_cap(series_info_df, name='Series Overview')
    else:
        st.info("No series records available.")

    # Series Batting Analysis
    if filtered_bat_df is not None and not filtered_bat_df.empty and series_info_df is not None and not series_info_df.empty and not smap.empty:
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
        # Align Date types
        sb = filtered_bat_df.copy()
        if 'Date' in sb.columns:
            sb['Date'] = sb['Date'].astype(str)
        sb = sb.merge(smap[['File Name', 'Date', 'Series']], on=['File Name', 'Date'], how='left')
        sb = sb[sb['Series'].notna()].copy()
        try:
            # Polars fast path
            pdf = sanitize_df_for_polars(sb)
            for c in ['Batted','Out','Not Out','Runs','Balls','4s','6s','50s','100s']:
                if c in pdf.columns:
                    pdf[c] = pd.to_numeric(pdf[c], errors='coerce').fillna(0)
            pl_df = pl.from_pandas(pdf)
            agg = (
                pl_df
                .group_by(['Series','Match_Format','Name','Bat_Team_y','Bowl_Team_y','Home Team'])
                .agg([
                    pl.n_unique('File Name').alias('Matches'),
                    pl.sum('Batted'), pl.sum('Out'), pl.sum('Not Out'),
                    pl.sum('Runs'), pl.sum('Balls'), pl.sum('4s'), pl.sum('6s'),
                    pl.sum('50s'), pl.sum('100s'),
                ])
                .rename({
                    'Bat_Team_y':'Bat Team','Bowl_Team_y':'Bowl Team','Home Team':'Country'
                })
            )
            series_stats = agg.to_pandas()
        except Exception:
            # pandas fallback without categorical blowup
            sb2 = sb.copy()
            for g in ['Series','Match_Format','Name','Bat_Team_y','Bowl_Team_y','Home Team']:
                if g in sb2.columns:
                    sb2[g] = sb2[g].astype(str)
            sums = sb2.groupby(['Series','Match_Format','Name','Bat_Team_y','Bowl_Team_y','Home Team'], sort=False, observed=True)[['Batted','Out','Not Out','Runs','Balls','4s','6s','50s','100s']].sum()
            matches = sb2.groupby(['Series','Match_Format','Name','Bat_Team_y','Bowl_Team_y','Home Team'], sort=False, observed=True)['File Name'].nunique()
            series_stats = sums.join(matches.rename('Matches')).reset_index()
            series_stats = series_stats.rename(columns={'Bat_Team_y':'Bat Team','Bowl_Team_y':'Bowl Team','Home Team':'Country'})
        # Order columns
        pref = ['Series','Match_Format','Name','Bat Team','Bowl Team','Country','Matches','Batted','Out','Not Out','Runs','Balls','4s','6s','50s','100s']
        series_stats = series_stats[[c for c in pref if c in series_stats.columns] + [c for c in series_stats.columns if c not in pref]]
        _show_table_with_cap(series_stats, name='Series Batting')
    else:
        st.info("No batting data available for series analysis or no series found")

    # Series Bowling Analysis
    if filtered_bowl_df is not None and not filtered_bowl_df.empty and series_info_df is not None and not series_info_df.empty and not smap.empty:
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
        sbo = filtered_bowl_df.copy()
        if 'Date' in sbo.columns:
            sbo['Date'] = sbo['Date'].astype(str)
        sbo = sbo.merge(smap[['File Name', 'Date', 'Series']], on=['File Name', 'Date'], how='left')
        sbo = sbo[sbo['Series'].notna()].copy()
        try:
            pdf = sanitize_df_for_polars(sbo)
            for c in ['Bowler_Overs','Bowler_Wkts','Bowler_Runs']:
                if c in pdf.columns:
                    pdf[c] = pd.to_numeric(pdf[c], errors='coerce').fillna(0)
            pl_df = pl.from_pandas(pdf)
            agg = (
                pl_df
                .group_by(['Series','Match_Format','Name','Bat_Team','Bowl_Team','Home_Team'])
                .agg([
                    pl.n_unique('File Name').alias('Matches'),
                    pl.sum('Bowler_Overs').alias('Overs'),
                    pl.sum('Bowler_Wkts').alias('Wickets'),
                    pl.sum('Bowler_Runs').alias('Runs'),
                    (pl.col('Bowler_Wkts') >= 5).sum().alias('5w'),
                    (pl.col('Bowler_Wkts') >= 10).sum().alias('10w'),
                ])
                .rename({'Bat_Team':'Bat Team','Bowl_Team':'Bowl Team','Home_Team':'Country'})
            )
            series_bowl_stats = agg.to_pandas()
        except Exception:
            sbo2 = sbo.copy()
            for g in ['Series','Match_Format','Name','Bat_Team','Bowl_Team','Home_Team']:
                if g in sbo2.columns:
                    sbo2[g] = sbo2[g].astype(str)
            sums = sbo2.groupby(['Series','Match_Format','Name','Bat_Team','Bowl_Team','Home_Team'], sort=False, observed=True)[['Bowler_Overs','Bowler_Wkts','Bowler_Runs']].sum()
            matches = sbo2.groupby(['Series','Match_Format','Name','Bat_Team','Bowl_Team','Home_Team'], sort=False, observed=True)['File Name'].nunique()
            fivew = (sbo2['Bowler_Wkts'] >= 5).groupby([sbo2['Series'], sbo2['Match_Format'], sbo2['Name'], sbo2['Bat_Team'], sbo2['Bowl_Team'], sbo2['Home_Team']]).sum()
            tenw = (sbo2['Bowler_Wkts'] >= 10).groupby([sbo2['Series'], sbo2['Match_Format'], sbo2['Name'], sbo2['Bat_Team'], sbo2['Bowl_Team'], sbo2['Home_Team']]).sum()
            series_bowl_stats = sums.join(matches.rename('Matches')).join(fivew.rename('5w')).join(tenw.rename('10w')).reset_index()
            series_bowl_stats = series_bowl_stats.rename(columns={'Bat_Team':'Bat Team','Bowl_Team':'Bowl Team','Home_Team':'Country','Bowler_Overs':'Overs','Bowler_Wkts':'Wickets','Bowler_Runs':'Runs'})
        # Derived metrics
        series_bowl_stats['Average'] = series_bowl_stats.apply(lambda r: round(r['Runs']/r['Wickets'], 2) if r['Wickets']>0 else 0, axis=1)
        series_bowl_stats['Economy'] = series_bowl_stats.apply(lambda r: round(r['Runs']/r['Overs'], 2) if r['Overs']>0 else 0, axis=1)
        # Order columns
        prefb = ['Series','Match_Format','Name','Bat Team','Bowl Team','Country','Matches','Overs','Wickets','Runs','5w','10w','Average','Economy']
        series_bowl_stats = series_bowl_stats[[c for c in prefb if c in series_bowl_stats.columns] + [c for c in series_bowl_stats.columns if c not in prefb]]
        series_bowl_stats = series_bowl_stats.sort_values(['Wickets','Average'], ascending=[False, True])
        _show_table_with_cap(series_bowl_stats, name='Series Bowling')
    else:
        st.info("No bowling data available for series analysis or no series found")  
