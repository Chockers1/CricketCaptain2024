# Section 1: Imports and Setup
###############################################################################
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import gc
import sys
import time
from datetime import datetime
from typing import Optional, Iterable
from pandas.api.types import CategoricalDtype, is_scalar

try:
    import polars as pl
except Exception:  # pragma: no cover - optional dependency
    pl = None

try:
    from .logging_utils import FastViewLogger
except ImportError:  # pragma: no cover - support direct execution
    from views.logging_utils import FastViewLogger

try:
    from performance_utils import memory_efficient_cache, get_tracked_dataframe
except ImportError:  # pragma: no cover - support package-relative imports
    from ..performance_utils import memory_efficient_cache, get_tracked_dataframe  # type: ignore

logger = FastViewLogger(st, "Records")


def _to_polars_frame(df: pd.DataFrame):
    """Return a sanitized Polars frame when Polars is available."""
    if pl is None or df is None or df.empty:
        return None
    sanitized = df.copy()
    if 'Date' in sanitized.columns:
        sanitized['Date'] = pd.to_datetime(sanitized['Date'], errors='coerce')
    for col in sanitized.columns:
        col_dtype = sanitized[col].dtype
        if isinstance(col_dtype, pd.CategoricalDtype):
            sanitized[col] = sanitized[col].astype(str)
        elif col_dtype == object:
            sample = sanitized[col].dropna().head(1)
            if not sample.empty and isinstance(sample.iloc[0], (dict, list, set, tuple)):
                sanitized[col] = sanitized[col].astype(str)
    try:
        return pl.from_pandas(sanitized, include_index=False)
    except Exception:
        return None


def _polars_not_out_condition(polars_df):
    if pl is None or polars_df is None:
        return None
    conditions = []
    if 'Not Out' in polars_df.columns:
        numeric_check = (
            pl.col('Not Out').cast(pl.Float64, strict=False).fill_null(0) == 1
        )
        string_check = (
            pl.col('Not Out')
            .cast(pl.Utf8, strict=False)
            .str.to_lowercase()
            .str.strip_chars()
            .is_in(['1', 'true', 'yes', 'y'])
        )
        conditions.append(numeric_check | string_check)
    if 'How Out' in polars_df.columns:
        conditions.append(
            pl.col('How Out')
            .cast(pl.Utf8, strict=False)
            .str.to_lowercase()
            .str.strip_chars()
            .is_in(['not out'])
        )
    if not conditions:
        return None
    combined = conditions[0]
    for extra in conditions[1:]:
        combined = combined | extra
    return combined


def _pandas_not_out_mask(frame: pd.DataFrame):
    if frame is None or frame.empty:
        return None
    mask = None
    if 'Not Out' in frame.columns:
        numeric_mask = pd.to_numeric(frame['Not Out'], errors='coerce') == 1
        string_mask = frame['Not Out'].astype(str).str.lower().str.strip().isin(['1', 'true', 'yes', 'y'])
        mask = numeric_mask | string_mask
    if 'How Out' in frame.columns:
        how_mask = frame['How Out'].astype(str).str.lower().str.strip().isin(['not out'])
        mask = how_mask if mask is None else (mask | how_mask)
    return mask
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

def _safe_parse_dates(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Coerce date columns and downcast numerics without mutating the source frame."""
    if df is None or df.empty:
        return None

    try:
        working = df.copy()

        int_cols = working.select_dtypes(include=['int64']).columns
        if len(int_cols):
            working[int_cols] = working[int_cols].apply(pd.to_numeric, downcast='integer')

        float_cols = working.select_dtypes(include=['float64']).columns
        if len(float_cols):
            working[float_cols] = working[float_cols].apply(pd.to_numeric, downcast='float')

        if 'Date' in working.columns:
            working['Date'] = pd.to_datetime(working['Date'], errors='coerce')
            working = working.dropna(subset=['Date'])

        gc.collect()
        return working
    except Exception as exc:  # pragma: no cover - defensive guard
        st.error(f"Error processing dataframe: {exc}")
        return None


@memory_efficient_cache(ttl=600, data_types=['batting', 'bowling', 'match', 'game'])
def process_dataframes():
    """Process all session DataFrames with improved memory management."""
    try:
        bat_df = _safe_parse_dates(get_tracked_dataframe('bat_df'))
        bowl_df = _safe_parse_dates(get_tracked_dataframe('bowl_df'))
        match_df = _safe_parse_dates(get_tracked_dataframe('match_df'))
        game_df = _safe_parse_dates(get_tracked_dataframe('game_df'))
        gc.collect()
        return bat_df, bowl_df, match_df, game_df
    except Exception as exc:  # pragma: no cover - defensive guard
        st.error(f"Error in process_dataframes: {exc}")
        return None, None, None, None


@memory_efficient_cache(ttl=600, data_types=['batting', 'bowling', 'match', 'game'])
def filter_by_format(df, format_choice):
    """Filter dataframe by format with caching"""
    if df is None or df.empty:
        return df
    if 'All' in format_choice or not format_choice:
        return df
    return df[df['Match_Format'].isin(format_choice)]

def _unique_sorted_strings(series: Optional[pd.Series]) -> list[str]:
    """Return sorted unique string values while dropping empty/null placeholders."""
    if series is None or series.empty:
        return []
    cleaned = series.dropna().astype(str).str.strip()
    if cleaned.empty:
        return []
    cleaned = cleaned[cleaned != ""]
    cleaned = cleaned[cleaned.str.lower() != 'nan']
    unique_values = cleaned.unique().tolist()
    return sorted(unique_values)

def _filter_by_files(df: Optional[pd.DataFrame], files: set[str]) -> Optional[pd.DataFrame]:
    """Limit dataframe rows to the provided file name set."""
    if df is None or df.empty or 'File Name' not in df.columns:
        return df
    return df[df['File Name'].isin(files)]

def _filter_by_team_columns(
    df: Optional[pd.DataFrame],
    candidate_columns: Iterable[str],
    team: str,
) -> Optional[pd.DataFrame]:
    """Filter dataframe rows where any of the candidate team columns matches team."""
    if df is None or df.empty or not team:
        return df

    mask = None
    for column in candidate_columns:
        if column not in df.columns:
            continue
        series = df[column]
        # Normalize to string while preserving missing values
        normalized = series.astype('string').fillna('').str.strip()
        current = normalized.eq(team)
        mask = current if mask is None else (mask | current)

    if mask is None:
        return df

    filtered = df[mask]
    if filtered.empty:
        return filtered
    return filtered.copy()

def _first_existing_column(
    df: Optional[pd.DataFrame],
    candidates: Iterable[str],
) -> Optional[str]:
    """Return the first column name from candidates that exists in df."""
    if df is None:
        return None
    for column in candidates:
        if column in df.columns:
            return column
    return None

def _apply_user_filters(
    bat_df: Optional[pd.DataFrame],
    bowl_df: Optional[pd.DataFrame],
    match_df: Optional[pd.DataFrame],
    game_df: Optional[pd.DataFrame],
    *,
    player: str,
    team: str,
    match_date: str,
    player_of_match: str,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Apply player/team/date/POM selections across the cached dataframes."""

    filtered_bat = bat_df
    filtered_bowl = bowl_df
    filtered_match = match_df
    filtered_game = game_df

    player_active = bool(player) and player != 'All'
    team_active = bool(team) and team != 'All'
    date_active = bool(match_date) and match_date != 'All'
    pom_active = bool(player_of_match) and player_of_match != 'All'

    if player_active and filtered_bat is not None and 'Name' in filtered_bat.columns:
        filtered_bat = filtered_bat[filtered_bat['Name'] == player]
    if player_active and filtered_bowl is not None and 'Name' in filtered_bowl.columns:
        filtered_bowl = filtered_bowl[filtered_bowl['Name'] == player]

    if pom_active:
        if filtered_match is not None and 'Player_of_the_Match' in filtered_match.columns:
            filtered_match = filtered_match[filtered_match['Player_of_the_Match'] == player_of_match]
        if filtered_bat is not None and 'Player_of_the_Match' in filtered_bat.columns:
            filtered_bat = filtered_bat[filtered_bat['Player_of_the_Match'] == player_of_match]
        if filtered_bowl is not None and 'Player_of_the_Match' in filtered_bowl.columns:
            filtered_bowl = filtered_bowl[filtered_bowl['Player_of_the_Match'] == player_of_match]
        if filtered_game is not None and 'Player_of_the_Match' in filtered_game.columns:
            filtered_game = filtered_game[filtered_game['Player_of_the_Match'] == player_of_match]

    if date_active:
        if filtered_match is not None and 'Date' in filtered_match.columns:
            match_dates = pd.to_datetime(filtered_match['Date'], errors='coerce')
            filtered_match = filtered_match[match_dates.dt.strftime('%d/%m/%Y') == match_date]
        if filtered_bat is not None and 'Date' in filtered_bat.columns:
            bat_dates = pd.to_datetime(filtered_bat['Date'], errors='coerce')
            filtered_bat = filtered_bat[bat_dates.dt.strftime('%d/%m/%Y') == match_date]
        if filtered_bowl is not None and 'Date' in filtered_bowl.columns:
            bowl_dates = pd.to_datetime(filtered_bowl['Date'], errors='coerce')
            filtered_bowl = filtered_bowl[bowl_dates.dt.strftime('%d/%m/%Y') == match_date]
        if filtered_game is not None and 'Date' in filtered_game.columns:
            game_dates = pd.to_datetime(filtered_game['Date'], errors='coerce')
            filtered_game = filtered_game[game_dates.dt.strftime('%d/%m/%Y') == match_date]

    if team_active:
        filtered_bat = _filter_by_team_columns(
            filtered_bat,
            ['Bat_Team_y', 'Bat_Team', 'Bat Team', 'Bat_Team_x', 'Team'],
            team,
        )
        filtered_bowl = _filter_by_team_columns(
            filtered_bowl,
            ['Bowl_Team_y', 'Bowl_Team', 'Bowl Team', 'Bowl_Team_x'],
            team,
        )
        filtered_match = _filter_by_team_columns(
            filtered_match,
            ['Home_Team', 'Away_Team', 'Home Team', 'Away Team'],
            team,
        )
        filtered_game = _filter_by_team_columns(
            filtered_game,
            ['Bat_Team', 'Bowl_Team', 'Bat Team', 'Bowl Team'],
            team,
        )

    def _intersect(existing: Optional[set[str]], new_values: set[str]) -> Optional[set[str]]:
        if new_values is None:
            return existing
        if existing is None:
            return new_values
        return existing & new_values

    eligible_files: Optional[set[str]] = None

    if player_active:
        player_files: set[str] = set()
        if filtered_bat is not None and 'File Name' in filtered_bat.columns:
            player_files.update(filtered_bat['File Name'])
        if filtered_bowl is not None and 'File Name' in filtered_bowl.columns:
            player_files.update(filtered_bowl['File Name'])
        eligible_files = _intersect(eligible_files, player_files)

    if (pom_active or date_active) and filtered_match is not None and 'File Name' in filtered_match.columns:
        match_files = set(filtered_match['File Name'])
        eligible_files = _intersect(eligible_files, match_files)

    if team_active:
        team_files: set[str] = set()
        for subset in (filtered_bat, filtered_bowl, filtered_match, filtered_game):
            if subset is not None and not subset.empty and 'File Name' in subset.columns:
                team_files.update(subset['File Name'].dropna())
        eligible_files = _intersect(eligible_files, team_files)

    if eligible_files is not None:
        filtered_bat = _filter_by_files(filtered_bat, eligible_files)
        filtered_bowl = _filter_by_files(filtered_bowl, eligible_files)
        filtered_match = _filter_by_files(filtered_match, eligible_files)
        filtered_game = _filter_by_files(filtered_game, eligible_files)

    return filtered_bat, filtered_bowl, filtered_match, filtered_game

def _prepare_display_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Return a defensive copy with a clean RangeIndex for safe UI rendering."""
    if df is None or df.empty:
        return df
    return df.reset_index(drop=True).copy()

def _table_key(prefix: str, columns: Optional[pd.Index], rows: Optional[int] = None) -> str:
    """Create a Streamlit widget key that shifts when layout or row count changes."""
    if columns is None:
        return f"{prefix}__{rows}" if rows is not None else prefix
    sanitized = [str(col).strip().replace(' ', '_') or 'column' for col in columns]
    joined = "__".join(sanitized)
    if rows is not None:
        return f"{prefix}__{rows}__{joined}" if joined else f"{prefix}__{rows}"
    return f"{prefix}__{joined}" if joined else prefix

@memory_efficient_cache(ttl=600, data_types=['batting', 'bowling', 'match', 'game'])
def get_formats():
    """Return sorted list of format names with 'All' pinned to the top."""
    collected: set[str] = {'All'}

    for df_name in ['game_df', 'bat_df', 'bowl_df', 'match_df']:
        df = get_tracked_dataframe(df_name)
        if df.empty or 'Match_Format' not in df.columns:
            continue
        raw_values = df['Match_Format'].dropna().unique().tolist()
        if not raw_values:
            continue
        stringified = []
        for value in raw_values:
            if not is_scalar(value):
                logger.log(
                    "Non-scalar format option skipped",
                    source=df_name,
                    value_type=type(value).__name__,
                )
                continue
            stringified.append(str(value))
        collected.update(stringified)

    tail = sorted(value for value in collected if value != 'All')
    options = ['All', *tail]
    logger.log(
        "Format options prepared",
        total=len(options),
        sample=options[:5],
    )
    return options

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
        raw_formats = get_formats() or []
        formats: list[str] = []
        seen = set()

        def _normalize_format(value) -> Optional[str]:
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        for option in raw_formats:
            if not is_scalar(option):
                continue
            normalized = _normalize_format(option)
            if normalized and normalized not in seen:
                seen.add(normalized)
                formats.append(normalized)

        candidate_frames = [bat_df, bowl_df, match_df, game_df]
        if (not formats or formats == ['All']) and any(frame is not None for frame in candidate_frames):
            derived = set()
            for frame in candidate_frames:
                if frame is None or frame.empty or 'Match_Format' not in frame.columns:
                    continue
                column = frame['Match_Format']
                if isinstance(column.dtype, CategoricalDtype):
                    column = column.astype(str)
                values = column.dropna().unique().tolist()
                for value in values:
                    if not is_scalar(value):
                        continue
                    normalized = _normalize_format(value)
                    if normalized:
                        derived.add(normalized)
            if derived:
                formats = ['All', *sorted(derived)]

        if 'All' in formats:
            ordered = ['All']
            ordered.extend(sorted(option for option in formats if option != 'All'))
            formats = ordered
        else:
            formats.insert(0, 'All')

        logger.log(
            "Format widget options",
            count=len(formats),
            sample=formats[:5],
        )

        format_choice = st.multiselect('Format:', formats, default=['All'])
        
        if not format_choice:
            format_choice = ['All']
        
        # Apply primary format filter
        filtered_bat_df = filter_by_format(bat_df, format_choice)
        filtered_bowl_df = filter_by_format(bowl_df, format_choice)
        filtered_match_df = filter_by_format(match_df, format_choice)
        filtered_game_df = filter_by_format(game_df, format_choice)

        player_candidates: set[str] = set()
        if filtered_bat_df is not None and 'Name' in filtered_bat_df.columns:
            player_candidates.update(_unique_sorted_strings(filtered_bat_df['Name']))
        if filtered_bowl_df is not None and 'Name' in filtered_bowl_df.columns:
            player_candidates.update(_unique_sorted_strings(filtered_bowl_df['Name']))
        player_options = ['All']
        if player_candidates:
            player_options.extend(sorted(player_candidates))

        date_options = ['All']
        if filtered_match_df is not None and 'Date' in filtered_match_df.columns:
            match_dates = pd.to_datetime(filtered_match_df['Date'], errors='coerce')
            formatted_dates = match_dates.dropna().dt.strftime('%d/%m/%Y')
            if not formatted_dates.empty:
                unique_dates = sorted(formatted_dates.unique().tolist(), reverse=True)
                date_options.extend(unique_dates)

        pom_candidates = []
        if filtered_match_df is not None and 'Player_of_the_Match' in filtered_match_df.columns:
            pom_candidates = _unique_sorted_strings(filtered_match_df['Player_of_the_Match'])
        pom_options = ['All', *pom_candidates]

        team_candidates: set[str] = set()

        def _collect_team_values(frame: Optional[pd.DataFrame], columns: Iterable[str]) -> None:
            if frame is None or frame.empty:
                return
            for column in columns:
                if column in frame.columns:
                    team_candidates.update(_unique_sorted_strings(frame[column]))

        _collect_team_values(filtered_bat_df, ['Bat_Team_y', 'Bat_Team', 'Bat Team', 'Bat_Team_x', 'Team'])
        _collect_team_values(filtered_bowl_df, ['Bowl_Team_y', 'Bowl_Team', 'Bowl Team', 'Bowl_Team_x'])
        _collect_team_values(filtered_match_df, ['Home_Team', 'Away_Team', 'Home Team', 'Away Team'])
        _collect_team_values(filtered_game_df, ['Bat_Team', 'Bowl_Team', 'Bat Team', 'Bowl Team'])

        team_options = ['All']
        if team_candidates:
            team_options.extend(sorted(team_candidates))

        filter_col_player, filter_col_team, filter_col_date, filter_col_pom = st.columns(4)
        selected_player = filter_col_player.selectbox(
            'Player Name', player_options, index=0, key='records_player_filter'
        )
        selected_team = filter_col_team.selectbox(
            'Team', team_options, index=0, key='records_team_filter'
        )
        selected_date = filter_col_date.selectbox(
            'Match Date', date_options, index=0, key='records_date_filter'
        )
        selected_pom = filter_col_pom.selectbox(
            'Player of the Match', pom_options, index=0, key='records_pom_filter'
        )

        (
            filtered_bat_df,
            filtered_bowl_df,
            filtered_match_df,
            filtered_game_df,
        ) = _apply_user_filters(
            filtered_bat_df,
            filtered_bowl_df,
            filtered_match_df,
            filtered_game_df,
            player=selected_player,
            team=selected_team,
            match_date=selected_date,
            player_of_match=selected_pom,
        )
        
        return filtered_bat_df, filtered_bowl_df, filtered_match_df, filtered_game_df
        
    except Exception as e:
        st.error(f"Error initializing data: {str(e)}")
        return None, None, None, None

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
        logger.log("process_in_chunks error", error=str(e), chunk_size=len(df) if df is not None else 0)
        st.error(f"Error processing data chunk: {str(e)}")
        return pd.DataFrame()

# --- HELPERS ------------------------------------------------------
def format_date_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Format a datetime column to 'dd/mm/YYYY', working on a safe copy."""
    if df is None or col not in df.columns:
        return df
    formatted_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(formatted_df[col]):
        formatted_df[col] = pd.to_datetime(formatted_df[col], errors='coerce')
    formatted_df[col] = formatted_df[col].dt.strftime('%d/%m/%Y')
    return formatted_df

# Section 3: Batting Records Functions
###############################################################################
def process_highest_scores(filtered_bat_df: pd.DataFrame) -> pd.DataFrame:
    """Return all centuries sorted by Runs desc, with minimal cols."""
    if filtered_bat_df is None or filtered_bat_df.empty:
        return pd.DataFrame()
    logger.log("process_highest_scores start", rows=len(filtered_bat_df))
    df = pd.DataFrame()
    if pl is not None:
        polars_df = _to_polars_frame(filtered_bat_df)
        if polars_df is not None:
            drop_cols = [c for c in BAT_DROP_COMMON if c in polars_df.columns]
            if drop_cols:
                polars_df = polars_df.drop(drop_cols)
            polars_df = polars_df.filter(pl.col('Runs') >= 100)
            rename_map = {
                old: new
                for old, new in (('Bat_Team_y', 'Bat Team'), ('Bowl_Team_y', 'Bowl Team'))
                if old in polars_df.columns
            }
            if rename_map:
                polars_df = polars_df.rename(rename_map)
            polars_df = polars_df.sort('Runs', descending=True)
            df = polars_df.to_pandas()
            logger.log_dataframe("process_highest_scores polars", df)
        else:
            logger.log("process_highest_scores fallback", reason="polars_frame_none")
    if df.empty:
        df = (
            filtered_bat_df[filtered_bat_df['Runs'] >= 100]
            .drop(columns=BAT_DROP_COMMON, errors='ignore')
            .sort_values('Runs', ascending=False)
            .rename(columns={'Bat_Team_y': 'Bat Team', 'Bowl_Team_y': 'Bowl Team'})
        )
        logger.log_dataframe("process_highest_scores pandas", df)
    logger.log_dataframe("process_highest_scores result", df)
    return format_date_column(df, 'Date')

def process_consecutive_scores(filtered_bat_df, threshold):
    """Return longest streaks of scores above threshold per player/format using vectorised operations."""
    columns = ['Name', 'Match_Format', 'Consecutive Matches', 'Start Date', 'End Date']
    if filtered_bat_df is None or filtered_bat_df.empty:
        return pd.DataFrame(columns=columns)

    df = (
        filtered_bat_df
        .loc[:, ['Name', 'Match_Format', 'Date', 'Runs']]
        .dropna(subset=['Name', 'Match_Format', 'Date'])
        .sort_values(['Name', 'Match_Format', 'Date'])
    )

    if df.empty:
        return pd.DataFrame(columns=columns)

    # Categoricals can break rolling aggregates; coerce relevant columns early.
    categorical_cols = df.select_dtypes(include=['category']).columns.tolist()
    for col in categorical_cols:
        if col == 'Date':
            df[col] = pd.to_datetime(df[col].astype(str), errors='coerce')
        else:
            df[col] = df[col].astype('string')

    for col in ['Name', 'Match_Format']:
        if col in df.columns:
            df[col] = df[col].astype('string')

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    df['hit_flag'] = (df['Runs'] >= threshold).astype(int)
    if not df['hit_flag'].any():
        return pd.DataFrame(columns=columns)

    group_keys = ['Name', 'Match_Format']
    grouped = df.groupby(group_keys, sort=False)
    df['segment'] = grouped['hit_flag'].transform(lambda s: s.eq(0).cumsum())
    df['streak_length'] = (
        df.groupby(group_keys + ['segment'])['hit_flag']
        .cumsum()
        * df['hit_flag']
    )

    streaks = (
        df[df['hit_flag'] == 1]
        .groupby(group_keys + ['segment'], as_index=False)
        .agg(
            streak_length=('streak_length', 'max'),
            start_date=('Date', 'min'),
            end_date=('Date', 'max'),
        )
    )

    streaks = streaks[streaks['streak_length'] >= 2]
    if streaks.empty:
        return pd.DataFrame(columns=columns)

    streaks = streaks.sort_values(['streak_length', 'end_date'], ascending=[False, False])
    best_streaks = streaks.drop_duplicates(subset=group_keys, keep='first')
    best_streaks = best_streaks.sort_values('streak_length', ascending=False)

    best_streaks['Start Date'] = best_streaks['start_date'].dt.strftime('%d/%m/%Y')
    best_streaks['End Date'] = best_streaks['end_date'].dt.strftime('%d/%m/%Y')

    result = best_streaks.rename(columns={'streak_length': 'Consecutive Matches'})[
        ['Name', 'Match_Format', 'Consecutive Matches', 'Start Date', 'End Date']
    ]
    return result.reset_index(drop=True)


def _ensure_run_rate(
    df: pd.DataFrame,
    *,
    runs_col: str = 'Total_Runs',
    overs_col: str = 'Overs',
    result_col: str = 'Run_Rate',
) -> pd.DataFrame:
    """Guarantee a run-rate column exists by computing it when missing.

    Handles overs recorded in cricket notation (e.g. 47.3 = 47 overs, 3 balls).
    """

    if df is None or df.empty or result_col in df.columns:
        return df
    if runs_col not in df.columns or overs_col not in df.columns:
        return df

    overs_values = pd.to_numeric(df[overs_col], errors='coerce').to_numpy()
    runs_values = pd.to_numeric(df[runs_col], errors='coerce').to_numpy()

    safe_overs = np.nan_to_num(overs_values, nan=0.0)
    whole_overs = np.floor(safe_overs)
    partial_balls = safe_overs - whole_overs
    overs_float = whole_overs + (partial_balls * 10) / 6

    run_rate = np.zeros_like(overs_float, dtype=float)
    valid = overs_float > 0
    run_rate[valid] = np.nan_to_num(runs_values[valid], nan=0.0) / overs_float[valid]

    df[result_col] = np.nan_to_num(run_rate, nan=0.0).round(2)
    return df

def process_not_out_99s(filtered_bat_df: pd.DataFrame) -> pd.DataFrame:
    """Return 99 not out entries."""
    if filtered_bat_df is None or filtered_bat_df.empty:
        return pd.DataFrame()
    logger.log("process_not_out_99s start", rows=len(filtered_bat_df))
    df = pd.DataFrame()
    if pl is not None:
        polars_df = _to_polars_frame(filtered_bat_df)
        if polars_df is not None:
            polars_df = polars_df.filter(pl.col('Runs') == 99)
            not_out_condition = _polars_not_out_condition(polars_df)
            if not_out_condition is not None:
                polars_df = polars_df.filter(not_out_condition)
            else:
                logger.log("process_not_out_99s missing_not_out_indicators")
            rename_map = {
                old: new
                for old, new in (('Bat_Team_y', 'Bat Team'), ('Bowl_Team_y', 'Bowl Team'))
                if old in polars_df.columns
            }
            if rename_map:
                polars_df = polars_df.rename(rename_map)
            drop_cols = [c for c in BAT_DROP_COMMON if c in polars_df.columns]
            if drop_cols:
                polars_df = polars_df.drop(drop_cols)
            polars_df = polars_df.sort('Date', descending=True)
            df = polars_df.to_pandas()
            logger.log_dataframe("process_not_out_99s polars", df)
        else:
            logger.log("process_not_out_99s fallback", reason="polars_frame_none")
    if df.empty:
        mask = filtered_bat_df['Runs'] == 99
        not_out_mask = _pandas_not_out_mask(filtered_bat_df)
        if not_out_mask is not None:
            mask &= not_out_mask
        else:
            logger.log("process_not_out_99s pandas_missing_not_out_indicators")
        df = (
            filtered_bat_df[mask]
            .drop(columns=BAT_DROP_COMMON, errors='ignore')
            .sort_values('Runs', ascending=False)
            .rename(columns={'Bat_Team_y': 'Bat Team', 'Bowl_Team_y': 'Bowl Team'})
        )
        logger.log_dataframe("process_not_out_99s pandas", df)
    logger.log_dataframe("process_not_out_99s result", df)
    return format_date_column(df, 'Date')

def process_carrying_bat(filtered_bat_df: pd.DataFrame) -> pd.DataFrame:
    """Return instances of carrying the bat."""
    if filtered_bat_df is None or filtered_bat_df.empty:
        return pd.DataFrame()
    logger.log("process_carrying_bat start", rows=len(filtered_bat_df))
    df = pd.DataFrame()
    if pl is not None:
        polars_df = _to_polars_frame(filtered_bat_df)
        if polars_df is not None:
            polars_df = polars_df.filter(
                pl.col('Position').is_in([1, 2])
                & (pl.col('Wickets') == 10)
            )
            not_out_condition = _polars_not_out_condition(polars_df)
            if not_out_condition is not None:
                polars_df = polars_df.filter(not_out_condition)
            else:
                logger.log("process_carrying_bat missing_not_out_indicators")
            rename_map = {
                'Bat_Team_y': 'Bat Team',
                'Bowl_Team_y': 'Bowl Team',
                'Total_Runs': 'Team Total',
                'Wickets': 'Team Wickets',
            }
            rename_map = {k: v for k, v in rename_map.items() if k in polars_df.columns}
            if rename_map:
                polars_df = polars_df.rename(rename_map)
            drop_cols = [c for c in CARRY_BAT_DROP if c in polars_df.columns]
            if drop_cols:
                polars_df = polars_df.drop(drop_cols)
            polars_df = polars_df.sort('Runs', descending=True)
            df = polars_df.to_pandas()
            logger.log_dataframe("process_carrying_bat polars", df)
        else:
            logger.log("process_carrying_bat fallback", reason="polars_frame_none")
    if df.empty:
        mask = filtered_bat_df['Position'].isin([1, 2]) & (filtered_bat_df['Wickets'] == 10)
        not_out_mask = _pandas_not_out_mask(filtered_bat_df)
        if not_out_mask is not None:
            mask &= not_out_mask
        else:
            logger.log("process_carrying_bat pandas_missing_not_out_indicators")
        df = (
            filtered_bat_df[mask]
            .drop(columns=CARRY_BAT_DROP, errors='ignore')
            .rename(columns={
                'Bat_Team_y': 'Bat Team', 'Bowl_Team_y': 'Bowl Team',
                'Total_Runs': 'Team Total', 'Wickets': 'Team Wickets'
            })
            .sort_values('Runs', ascending=False)
        )
        logger.log_dataframe("process_carrying_bat pandas", df)
    desired_order = ['Name', 'Bat Team', 'Bowl Team', 'Runs', 'Team Total', 'Team Wickets',
                     'Balls', '4s', '6s', 'Match_Format', 'Date']
    existing_cols = [col for col in desired_order if col in df.columns]
    df = df.loc[:, existing_cols]
    logger.log_dataframe("process_carrying_bat result", df)
    return format_date_column(df, 'Date')

def process_hundreds_both_innings(filtered_bat_df):
    """Process hundreds in both innings data."""
    if filtered_bat_df is None or filtered_bat_df.empty:
        return pd.DataFrame()
    logger.log("process_hundreds_both_innings start", rows=len(filtered_bat_df))

    required_columns = {'Name', 'Date', 'Innings', 'Runs'}
    if not required_columns.issubset(filtered_bat_df.columns):
        return pd.DataFrame()

    format_mask = (
        filtered_bat_df['Match_Format'].isin(['First Class', 'Test Match'])
        if 'Match_Format' in filtered_bat_df.columns
        else True
    )
    innings_mask = filtered_bat_df['Innings'].isin([1, 2, 3, 4])
    runs_mask = filtered_bat_df['Runs'] >= 100

    df = filtered_bat_df[format_mask & innings_mask & runs_mask].copy()
    if df.empty:
        return pd.DataFrame()

    polars_result = None
    if pl is not None:
        polars_df = _to_polars_frame(df)
        if polars_df is not None:
            polars_df = polars_df.with_columns(
                pl.when(pl.col('Innings') <= 2).then(1).otherwise(2).alias('Team_Innings')
            )
            group_cols = ['Name', 'Date']
            for optional_col in ['Home Team', 'Bat_Team_y', 'Bowl_Team_y', 'Match_Format']:
                if optional_col in polars_df.columns and optional_col not in group_cols:
                    group_cols.append(optional_col)
            agg_exprs = [
                (
                    pl.when(pl.col('Team_Innings') == 1)
                    .then(pl.col('Runs'))
                    .otherwise(None)
                    .max()
                    .alias('1st Innings')
                ),
                (
                    pl.when(pl.col('Team_Innings') == 2)
                    .then(pl.col('Runs'))
                    .otherwise(None)
                    .max()
                    .alias('2nd Innings')
                ),
            ]
            if 'Balls' in polars_df.columns:
                agg_exprs.append(pl.col('Balls').sum().alias('Balls'))
            if '4s' in polars_df.columns:
                agg_exprs.append(pl.col('4s').sum().alias('4s'))
            if '6s' in polars_df.columns:
                agg_exprs.append(pl.col('6s').sum().alias('6s'))

            polars_result = (
                polars_df.group_by(group_cols)
                .agg(agg_exprs)
                .filter((pl.col('1st Innings') >= 100) & (pl.col('2nd Innings') >= 100))
            )
            if polars_result.height == 0:
                polars_result = None

    if polars_result is not None:
        combined = polars_result.to_pandas()
        logger.log_dataframe("process_hundreds_both_innings polars", combined)
    else:
        logger.log("process_hundreds_both_innings fallback", reason="polars_frame_none" if pl is not None else "polars_unavailable")
        df['Team_Innings'] = np.where(df['Innings'] <= 2, 1, 2)
        group_cols = ['Name', 'Date']
        for optional_col in ['Home Team', 'Bat_Team_y', 'Bowl_Team_y', 'Match_Format']:
            if optional_col in df.columns:
                group_cols.append(optional_col)

        runs_pivot = (
            df.groupby(group_cols + ['Team_Innings'], observed=True)['Runs']
            .max()
            .unstack('Team_Innings')
            .rename(columns={1: '1st Innings', 2: '2nd Innings'})
        )

        if {'1st Innings', '2nd Innings'} - set(runs_pivot.columns):
            return pd.DataFrame()

        runs_pivot = runs_pivot[(runs_pivot['1st Innings'] >= 100) & (runs_pivot['2nd Innings'] >= 100)]
        if runs_pivot.empty:
            return pd.DataFrame()

        stats_cols = [col for col in ['Balls', '4s', '6s'] if col in df.columns]
        stats_df = (
            df.groupby(group_cols, observed=True)[stats_cols].sum()
            if stats_cols
            else pd.DataFrame(index=runs_pivot.index)
        )

        combined = runs_pivot.join(stats_df, how='left')
        combined = combined.reset_index()
        logger.log_dataframe("process_hundreds_both_innings pandas", combined)

    rename_map = {
        'Bat_Team_y': 'Bat Team',
        'Bowl_Team_y': 'Bowl Team',
        'Home Team': 'Country',
        'Match_Format': 'Format',
    }
    combined = combined.rename(columns=rename_map)

    display_cols = [
        'Name', 'Bat Team', 'Bowl Team', 'Country', 'Format',
        '1st Innings', '2nd Innings', 'Balls', '4s', '6s', 'Date'
    ]
    existing_cols = [col for col in display_cols if col in combined.columns]
    if 'Date' in combined.columns:
        combined['Date'] = pd.to_datetime(combined['Date'], errors='coerce', dayfirst=True)
        combined = combined.sort_values('Date', ascending=False, na_position='last')
        combined = format_date_column(combined, 'Date')
    logger.log_dataframe("process_hundreds_both_innings result", combined)
    return combined[existing_cols]

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

    working_df = _ensure_run_rate(filtered_game_df.copy())

    df = working_df[['Bat_Team', 'Bowl_Team', 'Total_Runs', 'Wickets',
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
tab_names = ["Batting Records", "Bowling Records", "Match Records", "Game Records", "Series Records", "Best Records", "Win/Loss Records"]
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
        hundreds_both_df = _prepare_display_df(hundreds_both_df)
        if hundreds_both_df is not None and not hundreds_both_df.empty:
            st.dataframe(
                hundreds_both_df,
                use_container_width=True,
                hide_index=True,
                key=_table_key("hundreds_both_df", hundreds_both_df.columns, len(hundreds_both_df)),
            )

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
        position_scores_df = _prepare_display_df(position_scores_df)
        if position_scores_df is not None and not position_scores_df.empty:
            st.dataframe(
                position_scores_df,
                use_container_width=True,
                hide_index=True,
                height=430,
                key=_table_key("position_scores_df", position_scores_df.columns, len(position_scores_df)),
            )

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
    
    desired_columns = [
        'Innings', 'Position', 'Name', 'Bowler_Overs', 'Maidens', 'Bowler_Runs',
        'Bowler_Wkts', 'Bowler_Econ', 'Bat_Team', 'Bowl_Team', 'Home_Team',
        'Match_Format', 'Date', 'comp', 'Competition'
    ]
    available_columns = [col for col in desired_columns if col in filtered_bowl_df.columns]

    df = (
        filtered_bowl_df[filtered_bowl_df['Bowler_Wkts'] >= 5]
        .sort_values(by=['Bowler_Wkts', 'Bowler_Runs'], ascending=[False, True])
        [available_columns]
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
            }
        )
    )

    # Normalize competition column naming
    if 'comp' in df.columns and 'Competition' not in df.columns:
        df = df.rename(columns={'comp': 'Competition'})
    elif 'comp' in df.columns and 'Competition' in df.columns:
        df = df.drop(columns=['comp'])

    display_order = [
        'Innings', 'Position', 'Name', 'Overs', 'Maidens', 'Runs', 'Wickets',
        'Econ', 'Bat Team', 'Bowl Team', 'Country', 'Format', 'Competition', 'Date'
    ]
    df = df[[col for col in display_order if col in df.columns] + [
        col for col in df.columns if col not in display_order
    ]]

    df = format_date_column(df, 'Date') if 'Date' in df.columns else df
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
    """Process 5+ wickets in both innings data."""
    if filtered_bowl_df is None or filtered_bowl_df.empty:
        return pd.DataFrame()

    required_columns = {'Name', 'Date', 'Innings', 'Bowler_Wkts'}
    if not required_columns.issubset(filtered_bowl_df.columns):
        return pd.DataFrame()

    format_mask = (
        filtered_bowl_df['Match_Format'].isin(['First Class', 'Test Match'])
        if 'Match_Format' in filtered_bowl_df.columns
        else True
    )
    innings_mask = filtered_bowl_df['Innings'].isin([1, 2, 3, 4])
    wickets_mask = filtered_bowl_df['Bowler_Wkts'] >= 5

    df = filtered_bowl_df[format_mask & innings_mask & wickets_mask].copy()
    if df.empty:
        return pd.DataFrame()

    df['Team_Innings'] = np.where(df['Innings'] <= 2, 1, 2)

    group_cols = ['Name', 'Date']
    for optional_col in ['Home_Team', 'Bat_Team', 'Bowl_Team', 'Match_Format', 'Competition', 'comp']:
        if optional_col in df.columns and optional_col not in group_cols:
            group_cols.append(optional_col)

    wkts_pivot = (
        df.groupby(group_cols + ['Team_Innings'], observed=True)['Bowler_Wkts']
        .max()
        .unstack('Team_Innings')
        .rename(columns={1: '1st Innings Wkts', 2: '2nd Innings Wkts'})
    )

    if {'1st Innings Wkts', '2nd Innings Wkts'} - set(wkts_pivot.columns):
        return pd.DataFrame()

    wkts_pivot = wkts_pivot[
        (wkts_pivot['1st Innings Wkts'] >= 5) & (wkts_pivot['2nd Innings Wkts'] >= 5)
    ]
    if wkts_pivot.empty:
        return pd.DataFrame()

    runs_pivot = None
    if 'Bowler_Runs' in df.columns:
        runs_pivot = (
            df.groupby(group_cols + ['Team_Innings'], observed=True)['Bowler_Runs']
            .sum()
            .unstack('Team_Innings')
            .rename(columns={1: '1st Innings Runs', 2: '2nd Innings Runs'})
        )

    overs_pivot = None
    if 'Bowler_Overs' in df.columns:
        overs_pivot = (
            df.groupby(group_cols + ['Team_Innings'], observed=True)['Bowler_Overs']
            .max()
            .unstack('Team_Innings')
            .rename(columns={1: '1st Innings Overs', 2: '2nd Innings Overs'})
        )

    combined = wkts_pivot
    for extra in [runs_pivot, overs_pivot]:
        if extra is not None:
            combined = combined.join(extra, how='left')

    combined = combined.reset_index()

    rename_map = {
        'Bat_Team': 'Bat Team',
        'Bowl_Team': 'Bowl Team',
        'Home_Team': 'Country',
        'Match_Format': 'Format',
        'comp': 'Competition',
    }
    combined = combined.rename(columns=rename_map)

    if 'Date' in combined.columns:
        combined['Date'] = pd.to_datetime(combined['Date'], errors='coerce', dayfirst=True)
        combined = combined.sort_values('Date', ascending=False, na_position='last')
        combined = format_date_column(combined, 'Date')

    display_cols = [
        'Name', 'Bat Team', 'Bowl Team', 'Country', 'Format', 'Competition',
        '1st Innings Wkts', '1st Innings Runs', '1st Innings Overs',
        '2nd Innings Wkts', '2nd Innings Runs', '2nd Innings Overs', 'Date'
    ]
    existing_cols = [col for col in display_cols if col in combined.columns]
    return combined[existing_cols]

def process_match_bowling(filtered_bowl_df):
    """Process match bowling figures data"""
    if filtered_bowl_df is None or filtered_bowl_df.empty:
        return pd.DataFrame()
    
    # Prepare a consolidated view to avoid repeated global lookups inside lambdas
    grouped = filtered_bowl_df.groupby(
        ['Name', 'Date', 'Home_Team', 'Bat_Team', 'Bowl_Team', 'Match_Format', 'Innings'],
        observed=True
    ).agg({
        'Bowler_Wkts': 'sum',
        'Bowler_Runs': 'sum'
    }).reset_index()

    def _innings_metrics(df: pd.DataFrame, innings: tuple[int, ...]) -> tuple[int, int]:
        subset = df[df['Innings'].isin(innings)]
        if subset.empty:
            return 0, 0
        wickets = subset['Bowler_Wkts'].max()
        runs = subset['Bowler_Runs'].sum()
        return wickets, runs

    def _collect_innings_metrics(df: pd.DataFrame) -> dict[str, int]:
        first_wkts, first_runs = _innings_metrics(df, (1, 2))
        second_wkts, second_runs = _innings_metrics(df, (3, 4))
        return {
            '1st Innings Wkts': first_wkts,
            '2nd Innings Wkts': second_wkts,
            '1st Innings Runs': first_runs,
            '2nd Innings Runs': second_runs,
        }

    # Aggregate once per player/match, then split figures into innings bands
    match_bowling_df = (
        grouped.groupby(
            ['Name', 'Date', 'Home_Team', 'Bat_Team', 'Bowl_Team', 'Match_Format'],
            observed=True
        )
        .apply(
            lambda g: pd.Series(_collect_innings_metrics(g)),
            include_groups=False
        )
        .reset_index()
    )

    match_bowling_df['Match Wickets'] = (
        match_bowling_df['1st Innings Wkts'] + match_bowling_df['2nd Innings Wkts']
    )
    match_bowling_df['Match Runs'] = (
        match_bowling_df['1st Innings Runs'] + match_bowling_df['2nd Innings Runs']
    )

    # Exclude match entries where fewer than 10 wickets were taken
    match_bowling_df = match_bowling_df[match_bowling_df['Match Wickets'] >= 10]

    df = (
        match_bowling_df
        .sort_values(by=['Match Wickets', 'Match Runs'], ascending=[False, True])
        .rename(
            columns={
                'Home_Team': 'Country',
                'Bat_Team': 'Bat Team',
                'Bowl_Team': 'Bowl Team',
                'Match_Format': 'Format',
            }
        )
    )

    display_cols = [
        'Name', 'Bat Team', 'Bowl Team', 'Country', 'Format',
        '1st Innings Wkts', '1st Innings Runs',
        '2nd Innings Wkts', '2nd Innings Runs',
        'Match Wickets', 'Match Runs', 'Date'
    ]
    df = df[[col for col in display_cols if col in df.columns]]
    df = format_date_column(df, 'Date') if 'Date' in df.columns else df
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

    df = filtered_match_df.copy()

    # Ensure comparison columns are numeric to avoid categorical ordering errors
    if 'Margin_Runs' in df.columns:
        df['Margin_Runs'] = pd.to_numeric(df['Margin_Runs'], errors='coerce')
    if 'Margin_Wickets' in df.columns:
        df['Margin_Wickets'] = pd.to_numeric(df['Margin_Wickets'], errors='coerce')

    # Set conditions based on win type
    if win_type == 'runs':
        base_conditions = [
            df['Margin_Runs'].notna(),
            df['Margin_Runs'] > 0,
            df['Innings_Win'] == 0,
            df['Home_Drawn'] != 1
        ]
        margin_column = 'Margin_Runs'
        margin_name = 'Runs'
    else:  # wickets
        base_conditions = [
            df['Margin_Wickets'].notna(),
            df['Margin_Wickets'] > 0,
            df['Innings_Win'] == 0,
            df['Home_Drawn'] != 1
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

        merged_df = _ensure_run_rate(merged_df)
        
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
        keep_columns = ['Bat_Team', 'Bowl_Team', 'Innings', 'Total_Runs', 'Wickets', 'Overs',
                         'Run_Rate', competition_col, 'Match_Format_x', 'Date', 'Margin_y']
        chase_df = chase_df[[col for col in keep_columns if col in chase_df.columns]].copy()
        rename_map = {
            'Bat_Team': 'Bat Team',
            'Bowl_Team': 'Bowl Team',
            'Total_Runs': 'Runs',
            'Run_Rate': 'Run Rate',
            competition_col: 'Competition',
            'Match_Format_x': 'Format',
            'Margin_y': 'Margin',
        }
        chase_df = chase_df.rename(columns=rename_map)

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

                merged_df = _ensure_run_rate(merged_df)
                
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
                rename_map = {
                    'Bat_Team': 'Bat Team',
                    'Bowl_Team': 'Bowl Team',
                    'Total_Runs': 'Runs',
                    'Run_Rate': 'Run Rate',
                    'Competition_x': 'Competition',
                    'Match_Format_x': 'Format',
                    'Player_of_the_Match_x': 'Player of the Match',
                    'Margin_y': 'Margin',
                }
                low_wins_df = low_wins_df.rename(columns=rename_map)

                if 'Result' in low_wins_df.columns:
                    low_wins_df = low_wins_df.drop('Result', axis=1)

                keep_order = ['Bat Team', 'Bowl Team', 'Innings', 'Runs', 'Overs', 'Wickets',
                              'Run Rate', 'Competition', 'Format', 'Player of the Match', 'Date', 'Margin']
                low_wins_df = low_wins_df[[col for col in keep_order if col in low_wins_df.columns]]
                
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

                merged_df = _ensure_run_rate(merged_df)

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
                existing_cols = [col for col in columns_to_show if col in merged_df.columns]
                return merged_df[existing_cols].copy()

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

            merged_df = _ensure_run_rate(merged_df)

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
            existing_cols = [col for col in columns_to_show if col in merged_df.columns]
            return merged_df[existing_cols].copy()

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
    series_batting_summary = pd.DataFrame()
    series_bowling_summary = pd.DataFrame()
    player_series_summary_display = pd.DataFrame()
    player_series_detail_display = pd.DataFrame()
    
    # Series Info Section
    if filtered_match_df is not None and not filtered_match_df.empty:
        # Initialize with proper columns structure even if empty
        computed_info = compute_series_info(filtered_match_df)
        
        if not computed_info.empty:
            logger.log_dataframe("series_info_df", computed_info, include_dtypes=True)
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
            
            series_info_df = computed_info.copy()
            series_info_df['Start_Date'] = pd.to_datetime(
                series_info_df['Start_Date'], errors='coerce', dayfirst=True
            )
            series_info_df['End_Date'] = pd.to_datetime(
                series_info_df['End_Date'], errors='coerce', dayfirst=True
            )
            series_info_df['Home_Team_norm'] = (
                series_info_df['Home_Team'].astype(str).str.strip().str.lower()
            )
            series_info_df['Away_Team_norm'] = (
                series_info_df['Away_Team'].astype(str).str.strip().str.lower()
            )
            series_info_df['match_format_norm'] = (
                series_info_df['Match_Format'].astype(str).str.strip().str.lower()
            )

            def _series_side_results(row: pd.Series) -> pd.Series:
                home_wins = row.get('Total_Home_Wins', 0) or 0
                away_wins = row.get('Total_Away_Wins', 0) or 0
                draws = row.get('Total_Draws', 0) or 0

                if home_wins > away_wins:
                    return pd.Series({'Home_Result': 'Won', 'Away_Result': 'Lost'})
                if away_wins > home_wins:
                    return pd.Series({'Home_Result': 'Lost', 'Away_Result': 'Won'})
                if draws > 0:
                    return pd.Series({'Home_Result': 'Drawn', 'Away_Result': 'Drawn'})
                return pd.Series({'Home_Result': 'Tied', 'Away_Result': 'Tied'})

            side_results = series_info_df.apply(_series_side_results, axis=1)
            series_info_df = pd.concat([series_info_df, side_results], axis=1)
            series_info_df['Series_Label'] = (
                series_info_df['Home_Team'].astype('string').fillna('').str.strip()
                + ' vs '
                + series_info_df['Away_Team'].astype('string').fillna('').str.strip()
                + ' ('
                + series_info_df['Match_Format'].astype('string').fillna('').str.strip()
                + ')'
            ).str.replace(' ()', '', regex=False)

            # Precompute comparison arrays to avoid repeated conversions inside row lookups
            series_start_dates = series_info_df['Start_Date'].dt.date
            series_end_dates = series_info_df['End_Date'].dt.date
            series_home_norm = series_info_df['Home_Team_norm']
            series_away_norm = series_info_df['Away_Team_norm']
            series_format_norm = series_info_df['match_format_norm']

            series_display_df = series_info_df.copy()
            series_display_df = format_date_column(series_display_df, 'Start_Date')
            series_display_df = format_date_column(series_display_df, 'End_Date')

            # Display the series information
            st.dataframe(series_display_df, use_container_width=True, hide_index=True)
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
        logger.log_dataframe("series_batting initial", series_batting)
        logger.log(
            "series_batting team columns",
            bat_columns=[c for c in series_batting.columns if 'Bat_Team' in c],
            bowl_columns=[c for c in series_batting.columns if 'Bowl_Team' in c],
        )
        series_batting['Date_dt'] = pd.to_datetime(
            series_batting['Date'], errors='coerce', dayfirst=True
        )

        # Remove unwanted columns
        series_batting = series_batting.drop(['Bat_Team_x', 'Bowl_Team_x'], axis=1, errors='ignore')

        def _normalize_team(value):
            if value is None or pd.isna(value):
                return None
            return str(value).strip().lower()

        team_column_candidates = [
            'Bat_Team_y', 'Bat_Team', 'Bat Team',
            'Bowl_Team_y', 'Bowl_Team', 'Bowl Team',
            'Home Team', 'Away Team'
        ]
        team_columns_present = [col for col in team_column_candidates if col in series_batting.columns]

        if 'Match_Format' not in series_batting.columns or not team_columns_present:
            logger.log(
                "series_batting missing required columns",
                has_match_format='Match_Format' in series_batting.columns,
                team_columns=team_columns_present,
            )
            series_batting['Series'] = np.nan
        else:
            series_batting['match_format_norm'] = (
                series_batting['Match_Format'].astype(str).str.strip().str.lower()
            )
            series_batting['row_id'] = np.arange(len(series_batting))

            team_long = (
                series_batting[['row_id'] + team_columns_present]
                .melt(id_vars='row_id', value_vars=team_columns_present, value_name='team_raw')
                .dropna(subset=['team_raw'])
            )
            if not team_long.empty:
                team_long['team_norm'] = [
                    _normalize_team(value) for value in team_long['team_raw']
                ]
                team_long = team_long[team_long['team_norm'].notna()]

                meta_cols = ['row_id', 'Date_dt', 'match_format_norm']
                team_long = team_long.merge(series_batting[meta_cols], on='row_id', how='left')

                series_expanded = pd.concat([
                    series_info_df[['Series', 'Start_Date', 'End_Date', 'match_format_norm']]
                    .assign(team_norm=series_info_df['Home_Team_norm']),
                    series_info_df[['Series', 'Start_Date', 'End_Date', 'match_format_norm']]
                    .assign(team_norm=series_info_df['Away_Team_norm']),
                ], ignore_index=True)
                series_expanded = series_expanded.dropna(subset=['team_norm'])

                candidate_matches = team_long.merge(
                    series_expanded,
                    on=['team_norm', 'match_format_norm'],
                    how='left',
                )
                candidate_matches = candidate_matches[
                    candidate_matches['Series'].notna()
                ]
                candidate_matches = candidate_matches[
                    candidate_matches['Date_dt'].notna()
                ]
                if not candidate_matches.empty:
                    for date_col in ('Date_dt', 'Start_Date', 'End_Date'):
                        if date_col in candidate_matches.columns:
                            candidate_matches[date_col] = pd.to_datetime(
                                candidate_matches[date_col], errors='coerce'
                            )
                    candidate_matches = candidate_matches.dropna(
                        subset=['Date_dt', 'Start_Date', 'End_Date']
                    )
                    candidate_matches = candidate_matches[
                        (candidate_matches['Date_dt'] >= candidate_matches['Start_Date']) &
                        (candidate_matches['Date_dt'] <= candidate_matches['End_Date'])
                    ]
                    candidate_matches = candidate_matches.sort_values(
                        ['row_id', 'Start_Date', 'Series']
                    )
                    row_to_series = (
                        candidate_matches
                        .drop_duplicates('row_id')[['row_id', 'Series']]
                    )
                    series_batting = series_batting.merge(
                        row_to_series,
                        on='row_id',
                        how='left',
                    )
                else:
                    series_batting['Series'] = np.nan
            else:
                series_batting['Series'] = np.nan

        matched_count = int(series_batting['Series'].notna().sum())
        logger.log("series_batting matches", matched=matched_count, total=len(series_batting))

        # Filter out rows with no series match
        series_batting = series_batting[series_batting['Series'].notna()]
        series_batting = series_batting.drop(columns=['Date_dt', 'match_format_norm', 'row_id'], errors='ignore')
        
        if series_batting.empty:
            logger.log("series_batting empty_after_filter")
            st.info("No batting data matches available series.")
        else:
            logger.log_dataframe("series_batting matched", series_batting)
            bat_team_src = next((col for col in ['Bat_Team_y', 'Bat_Team', 'Bat Team'] if col in series_batting.columns), None)
            bowl_team_src = next((col for col in ['Bowl_Team_y', 'Bowl_Team', 'Bowl Team'] if col in series_batting.columns), None)
            country_src = next((col for col in ['Home Team', 'Country'] if col in series_batting.columns), None)
            if bat_team_src:
                series_batting['Bat Team'] = series_batting[bat_team_src]
            else:
                series_batting['Bat Team'] = np.nan
            if bowl_team_src:
                series_batting['Bowl Team'] = series_batting[bowl_team_src]
            else:
                series_batting['Bowl Team'] = np.nan
            if country_src:
                series_batting['Country'] = series_batting[country_src]
            else:
                series_batting['Country'] = np.nan
            logger.log(
                "series_batting group sources",
                bat_team_source=bat_team_src,
                bowl_team_source=bowl_team_src,
                country_source=country_src,
            )
            # Create series batting statistics
            series_stats = series_batting.groupby(
                ['Series', 'Name', 'Bat Team', 'Bowl Team', 'Country'],
                observed=True
            ).agg({
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

            if 'Runs' in series_stats.columns:
                series_stats = series_stats.sort_values('Runs', ascending=False)

            if not series_stats.empty:
                with np.errstate(divide='ignore', invalid='ignore'):
                    series_stats['Average'] = np.where(
                        series_stats['Out'] > 0,
                        (series_stats['Runs'] / series_stats['Out']).round(2),
                        np.nan,
                    )
                    series_stats['Strike Rate'] = np.where(
                        series_stats['Balls'] > 0,
                        ((series_stats['Runs'] / series_stats['Balls']) * 100).round(2),
                        np.nan,
                    )
                    series_stats['Balls per Out'] = np.where(
                        series_stats['Out'] > 0,
                        (series_stats['Balls'] / series_stats['Out']).round(2),
                        np.nan,
                    )

                preferred_order = [
                    'Series', 'Name', 'Bat Team', 'Bowl Team', 'Country',
                    'Matches', 'Batted', 'Out', 'Not Out', 'Runs', 'Balls',
                    'Average', 'Strike Rate', 'Balls per Out', '4s', '6s', '50s', '100s'
                ]
                existing_cols = [col for col in preferred_order if col in series_stats.columns]
                trailing_cols = [
                    col for col in series_stats.columns
                    if col not in existing_cols
                ]
                series_stats = series_stats[existing_cols + trailing_cols]
            
            # Display series batting statistics
            if not series_stats.empty:
                series_batting_summary = series_stats.copy()
                logger.log_dataframe("series_batting stats", series_stats)
                series_stats_display = _prepare_display_df(series_stats)
                st.dataframe(
                    series_stats_display,
                    use_container_width=True,
                    hide_index=True,
                    key="series_batting_stats",
                )
            else:
                logger.log("series_stats empty")
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
        logger.log_dataframe("series_bowling initial", series_bowling)
        logger.log(
            "series_bowling team columns",
            bat_columns=[c for c in series_bowling.columns if 'Bat_Team' in c or 'Bat Team' in c],
            bowl_columns=[c for c in series_bowling.columns if 'Bowl_Team' in c or 'Bowl Team' in c],
        )

        series_bowling['Date_dt'] = pd.to_datetime(
            series_bowling['Date'], errors='coerce', dayfirst=True
        )

        numeric_cols = ['Bowler_Overs', 'Bowler_Wkts', 'Bowler_Runs']
        for col in numeric_cols:
            if col in series_bowling.columns:
                series_bowling[col] = pd.to_numeric(series_bowling[col], errors='coerce').fillna(0)

        def _normalize_team_bowling(value):
            if value is None or pd.isna(value):
                return None
            return str(value).strip().lower()

        team_column_candidates = [
            'Bat_Team', 'Bat Team',
            'Bowl_Team', 'Bowl Team',
            'Home_Team', 'Home Team',
            'Away_Team', 'Away Team'
        ]
        team_columns_present = [col for col in team_column_candidates if col in series_bowling.columns]

        if 'Match_Format' not in series_bowling.columns or not team_columns_present:
            logger.log(
                "series_bowling missing required columns",
                has_match_format='Match_Format' in series_bowling.columns,
                team_columns=team_columns_present,
            )
            series_bowling['Series'] = np.nan
        else:
            series_bowling['match_format_norm'] = (
                series_bowling['Match_Format'].astype(str).str.strip().str.lower()
            )
            series_bowling['row_id'] = np.arange(len(series_bowling))

            team_long = (
                series_bowling[['row_id'] + team_columns_present]
                .melt(id_vars='row_id', value_vars=team_columns_present, value_name='team_raw')
                .dropna(subset=['team_raw'])
            )
            if not team_long.empty:
                team_long['team_norm'] = [
                    _normalize_team_bowling(value) for value in team_long['team_raw']
                ]
                team_long = team_long[team_long['team_norm'].notna()]

                meta_cols = ['row_id', 'Date_dt', 'match_format_norm']
                team_long = team_long.merge(series_bowling[meta_cols], on='row_id', how='left')

                series_expanded = pd.concat([
                    series_info_df[['Series', 'Start_Date', 'End_Date', 'match_format_norm']]
                    .assign(team_norm=series_info_df['Home_Team_norm']),
                    series_info_df[['Series', 'Start_Date', 'End_Date', 'match_format_norm']]
                    .assign(team_norm=series_info_df['Away_Team_norm']),
                ], ignore_index=True)
                series_expanded = series_expanded.dropna(subset=['team_norm'])

                candidate_matches = team_long.merge(
                    series_expanded,
                    on=['team_norm', 'match_format_norm'],
                    how='left',
                )
                candidate_matches = candidate_matches[
                    candidate_matches['Series'].notna()
                ]
                candidate_matches = candidate_matches[
                    candidate_matches['Date_dt'].notna()
                ]
                if not candidate_matches.empty:
                    for date_col in ('Date_dt', 'Start_Date', 'End_Date'):
                        if date_col in candidate_matches.columns:
                            candidate_matches[date_col] = pd.to_datetime(
                                candidate_matches[date_col], errors='coerce'
                            )
                    candidate_matches = candidate_matches.dropna(
                        subset=['Date_dt', 'Start_Date', 'End_Date']
                    )
                    candidate_matches = candidate_matches[
                        (candidate_matches['Date_dt'] >= candidate_matches['Start_Date']) &
                        (candidate_matches['Date_dt'] <= candidate_matches['End_Date'])
                    ]
                    candidate_matches = candidate_matches.sort_values(
                        ['row_id', 'Start_Date', 'Series']
                    )
                    row_to_series = (
                        candidate_matches
                        .drop_duplicates('row_id')[['row_id', 'Series']]
                    )
                    series_bowling = series_bowling.merge(
                        row_to_series,
                        on='row_id',
                        how='left',
                    )
                else:
                    series_bowling['Series'] = np.nan
            else:
                series_bowling['Series'] = np.nan

        matched_bowl = int(series_bowling['Series'].notna().sum())
        logger.log("series_bowling matches", matched=matched_bowl, total=len(series_bowling))

        # Filter out rows with no series match
        series_bowling = series_bowling[series_bowling['Series'].notna()]
        series_bowling = series_bowling.drop(columns=['Date_dt', 'match_format_norm', 'row_id'], errors='ignore')
        
        if series_bowling.empty:
            st.info("No bowling data matches available series. ")
        else:
            bat_team_src = next((col for col in ['Bat_Team', 'Bat Team'] if col in series_bowling.columns), None)
            bowl_team_src = next((col for col in ['Bowl_Team', 'Bowl Team'] if col in series_bowling.columns), None)
            country_src = next((col for col in ['Home_Team', 'Home Team', 'Country'] if col in series_bowling.columns), None)

            if bat_team_src:
                series_bowling['Bat Team'] = series_bowling[bat_team_src]
            else:
                series_bowling['Bat Team'] = np.nan
            if bowl_team_src:
                series_bowling['Bowl Team'] = series_bowling[bowl_team_src]
            else:
                series_bowling['Bowl Team'] = np.nan
            if country_src:
                series_bowling['Country'] = series_bowling[country_src]
            else:
                series_bowling['Country'] = np.nan

            # Create series bowling statistics
            series_bowl_stats = series_bowling.groupby(
                ['Series', 'Name', 'Bat Team', 'Bowl Team', 'Country'],
                observed=True
            ).agg({
                'File Name': 'nunique',
                'Bowler_Overs': 'sum',
                'Bowler_Wkts': 'sum',
                'Bowler_Runs': 'sum'
            }).reset_index()
            
            # Calculate 5-wicket and 10-wicket hauls from individual match data
            five_wicket_hauls = (
                series_bowling[series_bowling['Bowler_Wkts'] >= 5]
                .groupby(['Series', 'Name', 'Bat Team', 'Bowl Team', 'Country'], observed=True)
                .size()
                .reset_index(name='5w_hauls')
            )
            ten_wicket_hauls = (
                series_bowling[series_bowling['Bowler_Wkts'] >= 10]
                .groupby(['Series', 'Name', 'Bat Team', 'Bowl Team', 'Country'], observed=True)
                .size()
                .reset_index(name='10w_hauls')
            )
            
            # Merge the hauls data
            series_bowl_stats = series_bowl_stats.merge(five_wicket_hauls, on=['Series', 'Name', 'Bat Team', 'Bowl Team', 'Country'], how='left')
            series_bowl_stats = series_bowl_stats.merge(ten_wicket_hauls, on=['Series', 'Name', 'Bat Team', 'Bowl Team', 'Country'], how='left')
            
            # Fill NaN values with 0
            series_bowl_stats['5w_hauls'] = series_bowl_stats['5w_hauls'].fillna(0).astype(int)
            series_bowl_stats['10w_hauls'] = series_bowl_stats['10w_hauls'].fillna(0).astype(int)
            
            # Calculate bowling averages and economy rates
            with np.errstate(divide='ignore', invalid='ignore'):
                series_bowl_stats['Bowling_Average'] = np.where(
                    series_bowl_stats['Bowler_Wkts'] > 0,
                    (series_bowl_stats['Bowler_Runs'] / series_bowl_stats['Bowler_Wkts']).round(2),
                    0
                )
                series_bowl_stats['Economy_Rate'] = np.where(
                    series_bowl_stats['Bowler_Overs'] > 0,
                    (series_bowl_stats['Bowler_Runs'] / series_bowl_stats['Bowler_Overs']).round(2),
                    0
                )
            
            # Rename columns for display 
            series_bowl_stats.columns = ['Series', 'Name', 'Bat Team', 'Bowl Team', 'Country', 
                                       'Matches', 'Overs', 'Wickets', 'Runs', '5w', '10w', 
                                       'Average', 'Economy']
            
            # Sort by most wickets first
            series_bowl_stats = series_bowl_stats.sort_values(['Wickets', 'Average'], ascending=[False, True])
            
            # Display series bowling statistics
            if not series_bowl_stats.empty: 
                series_bowling_summary = series_bowl_stats.copy()
                logger.log_dataframe("series_bowling stats", series_bowl_stats)
                balls_col = None
                for candidate in ['Bowler_Balls', 'Balls']:
                    if candidate in series_bowling.columns:
                        balls_col = candidate
                        break

                series_bowl_stats['Strike Rate'] = np.nan

                if balls_col is not None:
                    player_series_index = ['Series', 'Name', 'Bat Team', 'Bowl Team', 'Country']
                    aggregated_balls = (
                        series_bowling
                        .groupby(player_series_index, observed=True)[balls_col]
                        .sum()
                        .rename('Balls Delivered')
                    )
                    series_bowl_stats = series_bowl_stats.merge(
                        aggregated_balls,
                        on=player_series_index,
                        how='left',
                    )
                    with np.errstate(divide='ignore', invalid='ignore'):
                        series_bowl_stats['Strike Rate'] = np.where(
                            series_bowl_stats['Wickets'] > 0,
                            (series_bowl_stats['Balls Delivered'] / series_bowl_stats['Wickets']).round(2),
                            np.nan,
                        )

                preferred_order = [
                    'Series', 'Name', 'Bat Team', 'Bowl Team', 'Country',
                    'Matches', 'Overs', 'Balls Delivered', 'Wickets', 'Runs',
                    'Average', 'Economy', 'Strike Rate', '5w', '10w'
                ]
                existing_cols = [col for col in preferred_order if col in series_bowl_stats.columns]
                trailing_cols = [
                    col for col in series_bowl_stats.columns
                    if col not in existing_cols
                ]
                series_bowl_stats = series_bowl_stats[existing_cols + trailing_cols]

                series_bowl_stats_display = _prepare_display_df(series_bowl_stats)
                st.dataframe(
                    series_bowl_stats_display,
                    use_container_width=True,
                    hide_index=True,
                    key="series_bowling_stats",
                )
            else:
                st.info("No series bowling statistics available.")
    else:
        st.info("No bowling data available for series analysis or no series found")  

    if not series_info_df.empty:
        player_series_sources = []

        if not series_batting_summary.empty:
            bat_players = series_batting_summary[['Series', 'Name', 'Bat Team', 'Matches']].copy()
            bat_players['Player_Team'] = bat_players['Bat Team'].astype('string').fillna('').str.strip()
            bat_players = bat_players.drop(columns=['Bat Team'])
            player_series_sources.append(bat_players)

        if not series_bowling_summary.empty:
            team_source = 'Bowl Team' if 'Bowl Team' in series_bowling_summary.columns else None
            if team_source is None:
                team_source = 'Bat Team' if 'Bat Team' in series_bowling_summary.columns else None
            if team_source is not None:
                bowl_players = series_bowling_summary[['Series', 'Name', team_source, 'Matches']].copy()
                bowl_players['Player_Team'] = bowl_players[team_source].astype('string').fillna('').str.strip()
                bowl_players = bowl_players.drop(columns=[team_source])
                player_series_sources.append(bowl_players)

        if player_series_sources:
            combined_players = pd.concat(player_series_sources, ignore_index=True)
            combined_players['Player_Team'] = combined_players['Player_Team'].astype('string').fillna('').str.strip()
            combined_players['Matches'] = pd.to_numeric(combined_players['Matches'], errors='coerce').fillna(0)
            combined_players = combined_players[combined_players['Player_Team'] != '']

            def _first_non_empty(series: pd.Series) -> str:
                for value in series:
                    if value is None or (isinstance(value, float) and np.isnan(value)):
                        continue
                    text = str(value).strip()
                    if text:
                        return text
                return ''

            player_series = (
                combined_players
                .groupby(['Series', 'Name'], as_index=False)
                .agg({
                    'Player_Team': _first_non_empty,
                    'Matches': 'max'
                })
            )

            player_series = player_series[player_series['Player_Team'] != '']
            if not player_series.empty:
                join_columns = [
                    'Series', 'Series_Label', 'Start_Date', 'End_Date', 'Series_Result',
                    'Home_Team', 'Away_Team', 'Home_Team_norm', 'Away_Team_norm',
                    'Home_Result', 'Away_Result'
                ]
                join_data = series_info_df[join_columns].copy()

                def _normalize_identifier(value: str) -> str:
                    if value is None or pd.isna(value):
                        return ''
                    return ''.join(ch for ch in str(value).lower().strip() if ch.isalnum())

                player_series['Series'] = player_series['Series'].astype('Int64')
                player_series = player_series.merge(join_data, on='Series', how='left')
                player_series['Matches'] = (
                    pd.to_numeric(player_series['Matches'], errors='coerce')
                    .fillna(0)
                    .round(0)
                    .astype('Int64')
                )
                player_series['Team_Norm'] = player_series['Player_Team'].apply(_normalize_identifier)
                player_series['Series Outcome'] = np.where(
                    player_series['Team_Norm'] == player_series['Home_Team_norm'],
                    player_series['Home_Result'],
                    np.where(
                        player_series['Team_Norm'] == player_series['Away_Team_norm'],
                        player_series['Away_Result'],
                        'Unknown'
                    )
                )
                player_series['Series Outcome'] = player_series['Series Outcome'].fillna('Unknown')

                summary_base = (
                    player_series
                    .groupby(['Name', 'Player_Team'], observed=True)
                    .agg({
                        'Series': 'nunique',
                        'Matches': 'sum'
                    })
                    .rename(columns={
                        'Series': 'Series Played',
                        'Matches': 'Matches Played'
                    })
                    .reset_index()
                )

                result_counts = (
                    player_series
                    .groupby(['Name', 'Player_Team', 'Series Outcome'], observed=True)
                    .size()
                    .unstack(fill_value=0)
                )
                summary = summary_base.merge(
                    result_counts,
                    on=['Name', 'Player_Team'],
                    how='left'
                ).fillna(0)

                for column in ['Won', 'Lost', 'Drawn', 'Tied']:
                    if column not in summary.columns:
                        summary[column] = 0

                summary = summary.drop(columns=['Unknown'], errors='ignore')

                int_columns = [
                    'Series Played', 'Matches Played', 'Won', 'Lost', 'Drawn', 'Tied'
                ]
                for column in int_columns:
                    summary[column] = (
                        pd.to_numeric(summary[column], errors='coerce')
                        .fillna(0)
                        .round(0)
                        .astype('Int64')
                    )

                summary['Win %'] = np.where(
                    summary['Series Played'] > 0,
                    (summary['Won'] / summary['Series Played']) * 100,
                    0
                ).round(1)

                summary = summary.sort_values(
                    ['Won', 'Win %', 'Series Played'],
                    ascending=[False, False, False]
                ).reset_index(drop=True)

                summary_display = summary.rename(columns={'Player_Team': 'Team'})
                summary_order = [
                    'Name', 'Team', 'Series Played', 'Matches Played',
                    'Won', 'Lost', 'Drawn', 'Tied', 'Win %'
                ]
                summary_display = summary_display[[
                    col for col in summary_order if col in summary_display.columns
                ]]

                detail_columns = [
                    'Name', 'Player_Team', 'Series_Label', 'Series Outcome',
                    'Matches', 'Series_Result', 'Start_Date', 'End_Date'
                ]
                detail_display = player_series[detail_columns].copy()
                detail_display = detail_display.rename(columns={
                    'Player_Team': 'Team',
                    'Series_Label': 'Series',
                    'Series Outcome': 'Outcome',
                    'Series_Result': 'Series Result'
                })
                detail_display = format_date_column(detail_display, 'Start_Date')
                detail_display = format_date_column(detail_display, 'End_Date')
                player_series_summary_display = summary_display
                player_series_detail_display = detail_display

        if not player_series_summary_display.empty:
            st.divider()
            st.markdown("""
                <div style="text-align: center; margin: 30px 0;">
                    <div style="background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%); 
                               padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                        <h3 style="margin: 0; font-size: 1.5em; font-weight: bold;">
                            🙌 Player Series Records
                        </h3>
                        <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                            Series wins, losses, and draws for every player
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.dataframe(
                player_series_summary_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Name": st.column_config.Column("Name", pinned=True),
                    "Team": st.column_config.Column("Team"),
                    "Win %": st.column_config.NumberColumn("Win %", format="%.1f%%")
                }
            )

            if not player_series_detail_display.empty:
                with st.expander("View player-by-series breakdown"):
                    st.dataframe(
                        player_series_detail_display,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Name": st.column_config.Column("Name", pinned=True),
                            "Team": st.column_config.Column("Team")
                        }
                    )

# Records Tab (auto-loaded)
with tabs[5]:
    best_tabs = st.tabs(["Batting Bests", "Bowling Bests"])

    with best_tabs[0]:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">🏅 Single Innings Bests</h3>
            </div>
            """, unsafe_allow_html=True)

        if filtered_bat_df is not None and not filtered_bat_df.empty:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">🎯 Top 10 Highest Scores</h3>
                </div>
                """, unsafe_allow_html=True)

                opponent_column = _first_existing_column(
                    filtered_bat_df,
                    ['Bowl_Team_y', 'Bowl_Team', 'Bowl Team', 'Opponent']
                )
                base_columns = [col for col in ['Name', 'Runs', 'Balls', 'Year'] if col in filtered_bat_df.columns]
                columns_to_use = base_columns.copy()
                if opponent_column:
                    columns_to_use.append(opponent_column)
                required_columns = {'Runs'}
                if not required_columns.issubset(filtered_bat_df.columns) or not columns_to_use:
                    best_inns_df = pd.DataFrame()
                else:
                    best_inns_df = filtered_bat_df[columns_to_use].copy()
                    best_inns_df = best_inns_df.sort_values(by='Runs', ascending=False).head(10)
                    best_inns_df.insert(0, 'Rank', range(1, 1 + len(best_inns_df)))
                    if opponent_column:
                        best_inns_df = best_inns_df.rename(columns={opponent_column: 'Opponent'})
                    else:
                        best_inns_df['Opponent'] = '-'
                    if 'Year' not in best_inns_df.columns:
                        best_inns_df['Year'] = np.nan
                    best_inns_df = best_inns_df[['Rank', 'Name', 'Runs', 'Balls', 'Opponent', 'Year']]

                if not best_inns_df.empty:
                    st.dataframe(
                        best_inns_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={"Name": st.column_config.Column("Name", pinned=True)}
                    )
                else:
                    st.info("No innings data available for highest scores with the current filters.")

            with col2:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">⚡ Top 10 Best SR (Inns, Min 50 Runs)</h3>
                </div>
                """, unsafe_allow_html=True)

                inns_min_50 = filtered_bat_df[filtered_bat_df['Runs'] >= 50].copy()
                inns_min_50['SR'] = ((inns_min_50['Runs'] / inns_min_50['Balls'].replace(0, np.nan)) * 100).round(2)
                opponent_column_sr = _first_existing_column(
                    inns_min_50,
                    ['Bowl_Team_y', 'Bowl_Team', 'Bowl Team', 'Opponent']
                )
                base_sr_columns = [col for col in ['Name', 'Runs', 'Balls', 'SR', 'Year'] if col in inns_min_50.columns]
                columns_to_use_sr = base_sr_columns.copy()
                if opponent_column_sr:
                    columns_to_use_sr.append(opponent_column_sr)
                required_sr_columns = {'SR'}
                if not required_sr_columns.issubset(inns_min_50.columns) or not columns_to_use_sr:
                    best_inns_sr_df = pd.DataFrame()
                else:
                    best_inns_sr_df = inns_min_50[columns_to_use_sr].copy()
                    best_inns_sr_df = best_inns_sr_df.sort_values(by='SR', ascending=False, na_position='last').head(10)
                    best_inns_sr_df.insert(0, 'Rank', range(1, 1 + len(best_inns_sr_df)))
                    if opponent_column_sr:
                        best_inns_sr_df = best_inns_sr_df.rename(columns={opponent_column_sr: 'Opponent'})
                    else:
                        best_inns_sr_df['Opponent'] = '-'
                    if 'Year' not in best_inns_sr_df.columns:
                        best_inns_sr_df['Year'] = np.nan
                    best_inns_sr_df = best_inns_sr_df[['Rank', 'Name', 'SR', 'Runs', 'Balls', 'Opponent', 'Year']]

                if not best_inns_sr_df.empty:
                    st.dataframe(
                        best_inns_sr_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={"Name": st.column_config.Column("Name", pinned=True)}
                    )
                else:
                    st.info("No innings meet the strike-rate criteria under the current filters.")

            st.divider()
            st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">📊 Seasonal Bests (by Format)</h3>
            </div>
            """, unsafe_allow_html=True)
            col3, col4 = st.columns(2)

            with col3:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">🏆 Top 10 Most Runs (Season)</h3>
                </div>
                """, unsafe_allow_html=True)

                season_runs_df = filtered_bat_df.groupby(['Name', 'Year', 'Match_Format']).agg(
                    Runs=('Runs', 'sum')
                ).reset_index()
                best_season_runs_df = season_runs_df.sort_values(by='Runs', ascending=False).head(10)
                best_season_runs_df.insert(0, 'Rank', range(1, 1 + len(best_season_runs_df)))
                best_season_runs_df = best_season_runs_df[['Rank', 'Name', 'Year', 'Match_Format', 'Runs']]

                st.dataframe(best_season_runs_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            with col4:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">📈 Top 10 Best Average (Season)</h3>
                </div>
                """, unsafe_allow_html=True)

                season_avg_df = filtered_bat_df.groupby(['Name', 'Year', 'Match_Format']).agg(
                    Runs=('Runs', 'sum'),
                    Out=('Out', 'sum')
                ).reset_index()
                season_avg_df['Average'] = (season_avg_df['Runs'] / season_avg_df['Out'].replace(0, np.nan)).round(2)
                best_season_avg_df = season_avg_df.sort_values(by='Average', ascending=False, na_position='last').head(10)
                best_season_avg_df.insert(0, 'Rank', range(1, 1 + len(best_season_avg_df)))
                best_season_avg_df = best_season_avg_df[['Rank', 'Name', 'Year', 'Match_Format', 'Average', 'Runs', 'Out']]

                st.dataframe(best_season_avg_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            st.divider()
            col5, col6 = st.columns(2)

            with col5:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">🏆 Top 10 Most POM (Season)</h3>
                </div>
                """, unsafe_allow_html=True)

                pom_df_seasonal = filtered_bat_df[
                    filtered_bat_df['Player_of_the_Match'].astype(str)
                    == filtered_bat_df['Name'].astype(str)
                ]
                season_pom_df = pom_df_seasonal.groupby(['Name', 'Year', 'Match_Format'])['File Name'].nunique().reset_index(name='POM')
                best_season_pom_df = season_pom_df.sort_values(by='POM', ascending=False).head(10)
                best_season_pom_df.insert(0, 'Rank', range(1, 1 + len(best_season_pom_df)))
                best_season_pom_df = best_season_pom_df[['Rank', 'Name', 'Year', 'Match_Format', 'POM']]

                st.dataframe(best_season_pom_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            with col6:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">🎯 Top 10 Boundary % (Season)</h3>
                </div>
                """, unsafe_allow_html=True)

                season_boundary_df = filtered_bat_df.groupby(['Name', 'Year', 'Match_Format']).agg(
                    Runs=('Runs', 'sum'),
                    Fours=('4s', 'sum'),
                    Sixes=('6s', 'sum')
                ).reset_index()
                season_boundary_df['Boundary Runs'] = (season_boundary_df['Fours'] * 4) + (season_boundary_df['Sixes'] * 6)
                season_boundary_df['Boundary %'] = ((season_boundary_df['Boundary Runs'] / season_boundary_df['Runs'].replace(0, np.nan)) * 100).round(2)
                best_season_boundary_pct_df = season_boundary_df.sort_values(by='Boundary %', ascending=False, na_position='last').head(10)
                best_season_boundary_pct_df.insert(0, 'Rank', range(1, 1 + len(best_season_boundary_pct_df)))
                best_season_boundary_pct_df = best_season_boundary_pct_df[['Rank', 'Name', 'Year', 'Match_Format', 'Boundary %', 'Boundary Runs', 'Runs']]

                st.dataframe(best_season_boundary_pct_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            st.divider()
            col7, col8 = st.columns(2)

            file_stats_seasonal = filtered_bat_df.groupby(['File Name', 'Year', 'Match_Format']).agg({
                'Total_Runs': 'first',
                'Team Balls': 'first',
                'Wickets': 'first'
            }).reset_index()
            file_stats_seasonal['Match_Avg'] = file_stats_seasonal['Total_Runs'] / file_stats_seasonal['Wickets'].replace(0, np.nan)
            file_stats_seasonal['Match_SR'] = (file_stats_seasonal['Total_Runs'] / file_stats_seasonal['Team Balls'].replace(0, np.nan)) * 100
            seasonal_match_metrics = file_stats_seasonal.groupby(['Year', 'Match_Format']).agg(
                Avg_Match_Avg=('Match_Avg', 'mean'),
                Avg_Match_SR=('Match_SR', 'mean')
            ).reset_index()
            seasonal_match_metrics['Avg_Match_Avg'] = seasonal_match_metrics['Avg_Match_Avg'].round(2)
            seasonal_match_metrics['Avg_Match_SR'] = seasonal_match_metrics['Avg_Match_SR'].round(2)

            with col7:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">📊 Top 10 Best Match+ Avg (Season)</h3>
                </div>
                """, unsafe_allow_html=True)

                player_seasonal_stats = filtered_bat_df.groupby(['Name', 'Year', 'Match_Format']).agg(
                    Runs=('Runs', 'sum'),
                    Out=('Out', 'sum')
                ).reset_index()
                player_seasonal_stats['Player_Avg'] = (player_seasonal_stats['Runs'] / player_seasonal_stats['Out'].replace(0, np.nan)).round(2)
                merged_stats = pd.merge(player_seasonal_stats, seasonal_match_metrics, on=['Year', 'Match_Format'], how='left')
                merged_stats['Match+ Avg'] = ((merged_stats['Player_Avg'] / merged_stats['Avg_Match_Avg'].replace(0, np.nan)) * 100).round(2)
                best_season_match_plus_avg_df = merged_stats.sort_values(by='Match+ Avg', ascending=False, na_position='last').head(10)
                best_season_match_plus_avg_df.insert(0, 'Rank', range(1, 1 + len(best_season_match_plus_avg_df)))
                best_season_match_plus_avg_df = best_season_match_plus_avg_df[['Rank', 'Name', 'Year', 'Match_Format', 'Match+ Avg', 'Player_Avg', 'Avg_Match_Avg']]

                st.dataframe(best_season_match_plus_avg_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            with col8:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">⚡ Top 10 Best Match+ SR (Season)</h3>
                </div>
                """, unsafe_allow_html=True)

                player_seasonal_stats_sr = filtered_bat_df.groupby(['Name', 'Year', 'Match_Format']).agg(
                    Runs=('Runs', 'sum'),
                    Balls=('Balls', 'sum')
                ).reset_index()
                player_seasonal_stats_sr['Player_SR'] = ((player_seasonal_stats_sr['Runs'] / player_seasonal_stats_sr['Balls'].replace(0, np.nan)) * 100).round(2)
                merged_stats_sr = pd.merge(player_seasonal_stats_sr, seasonal_match_metrics, on=['Year', 'Match_Format'], how='left')
                merged_stats_sr['Match+ SR'] = ((merged_stats_sr['Player_SR'] / merged_stats_sr['Avg_Match_SR'].replace(0, np.nan)) * 100).round(2)
                best_season_match_plus_sr_df = merged_stats_sr.sort_values(by='Match+ SR', ascending=False, na_position='last').head(10)
                best_season_match_plus_sr_df.insert(0, 'Rank', range(1, 1 + len(best_season_match_plus_sr_df)))
                best_season_match_plus_sr_df = best_season_match_plus_sr_df[['Rank', 'Name', 'Year', 'Match_Format', 'Match+ SR', 'Player_SR', 'Avg_Match_SR']]

                st.dataframe(best_season_match_plus_sr_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            st.divider()
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">🏆 Career Bests (by Format)</h3>
            </div>
            """, unsafe_allow_html=True)
            col9, col10 = st.columns(2)

            with col9:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">🏆 Top 10 Most Runs</h3>
                </div>
                """, unsafe_allow_html=True)

                career_runs_df = filtered_bat_df.groupby(['Name', 'Match_Format']).agg({
                    'Runs': 'sum'
                }).reset_index()
                best_career_runs_df = career_runs_df.sort_values(by='Runs', ascending=False).head(10)
                best_career_runs_df.insert(0, 'Rank', range(1, 1 + len(best_career_runs_df)))
                best_career_runs_df = best_career_runs_df[['Rank', 'Name', 'Match_Format', 'Runs']]

                st.dataframe(best_career_runs_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            with col10:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">📈 Top 10 Best Average (Min 10 Inns)</h3>
                </div>
                """, unsafe_allow_html=True)

                career_avg_df = filtered_bat_df.groupby(['Name', 'Match_Format']).agg({
                    'Runs': 'sum',
                    'Out': 'sum',
                    'Batted': 'sum'
                }).reset_index()
                career_avg_df = career_avg_df[career_avg_df['Batted'] >= 10]
                career_avg_df['Average'] = (career_avg_df['Runs'] / career_avg_df['Out'].replace(0, np.nan)).round(2)
                best_career_avg_df = career_avg_df.sort_values(by='Average', ascending=False, na_position='last').head(10)
                best_career_avg_df.insert(0, 'Rank', range(1, 1 + len(best_career_avg_df)))
                best_career_avg_df = best_career_avg_df[['Rank', 'Name', 'Match_Format', 'Average', 'Runs', 'Out', 'Batted']]
                best_career_avg_df = best_career_avg_df.rename(columns={'Batted': 'Inns'})

                st.dataframe(best_career_avg_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            st.divider()
            col11, col12 = st.columns(2)

            with col11:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">⚡ Top 10 Best SR (Min 500 Balls)</h3>
                </div>
                """, unsafe_allow_html=True)

                career_sr_df = filtered_bat_df.groupby(['Name', 'Match_Format']).agg(
                    Runs=('Runs', 'sum'),
                    Balls=('Balls', 'sum')
                ).reset_index()
                career_sr_df = career_sr_df[career_sr_df['Balls'] >= 500]
                career_sr_df['SR'] = ((career_sr_df['Runs'] / career_sr_df['Balls'].replace(0, np.nan)) * 100).round(2)
                best_career_sr_df = career_sr_df.sort_values(by='SR', ascending=False, na_position='last').head(10)
                best_career_sr_df.insert(0, 'Rank', range(1, 1 + len(best_career_sr_df)))
                best_career_sr_df = best_career_sr_df[['Rank', 'Name', 'Match_Format', 'SR', 'Runs', 'Balls']]

                st.dataframe(best_career_sr_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            with col12:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">🎯 Top 10 Best BPO (Min 10 Inns)</h3>
                </div>
                """, unsafe_allow_html=True)

                career_bpo_df = filtered_bat_df.groupby(['Name', 'Match_Format']).agg(
                    Balls=('Balls', 'sum'),
                    Out=('Out', 'sum'),
                    Inns=('Batted', 'sum')
                ).reset_index()
                career_bpo_df = career_bpo_df[career_bpo_df['Inns'] >= 10]
                career_bpo_df['BPO'] = (career_bpo_df['Balls'] / career_bpo_df['Out'].replace(0, np.nan)).round(2)
                best_career_bpo_df = career_bpo_df.sort_values(by='BPO', ascending=False, na_position='last').head(10)
                best_career_bpo_df.insert(0, 'Rank', range(1, 1 + len(best_career_bpo_df)))
                best_career_bpo_df = best_career_bpo_df[['Rank', 'Name', 'Match_Format', 'BPO', 'Balls', 'Out', 'Inns']]

                st.dataframe(best_career_bpo_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            st.divider()
            col13, col14 = st.columns(2)

            with col13:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">💰 Top 10 Runs Per Match (Min 10 Matches)</h3>
                </div>
                """, unsafe_allow_html=True)

                career_rpm_df = filtered_bat_df.groupby(['Name', 'Match_Format']).agg(
                    Runs=('Runs', 'sum'),
                    Matches=('File Name', 'nunique')
                ).reset_index()
                career_rpm_df = career_rpm_df[career_rpm_df['Matches'] >= 10]
                career_rpm_df['Runs Per Match'] = (career_rpm_df['Runs'] / career_rpm_df['Matches'].replace(0, np.nan)).round(2)
                best_career_rpm_df = career_rpm_df.sort_values(by='Runs Per Match', ascending=False, na_position='last').head(10)
                best_career_rpm_df.insert(0, 'Rank', range(1, 1 + len(best_career_rpm_df)))
                best_career_rpm_df = best_career_rpm_df[['Rank', 'Name', 'Match_Format', 'Runs Per Match', 'Runs', 'Matches']]

                st.dataframe(best_career_rpm_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            with col14:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">🥇 Top 10 Most 50s</h3>
                </div>
                """, unsafe_allow_html=True)

                career_50s_df = filtered_bat_df.groupby(['Name', 'Match_Format']).agg(
                    Fifties=('50s', 'sum')
                ).reset_index()
                best_career_50s_df = career_50s_df.sort_values(by='Fifties', ascending=False).head(10)
                best_career_50s_df.insert(0, 'Rank', range(1, 1 + len(best_career_50s_df)))
                best_career_50s_df = best_career_50s_df[['Rank', 'Name', 'Match_Format', 'Fifties']]

                st.dataframe(best_career_50s_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            st.divider()
            col15, col16 = st.columns(2)

            with col15:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">💯 Top 10 Most 100s</h3>
                </div>
                """, unsafe_allow_html=True)

                career_100s_df = filtered_bat_df.groupby(['Name', 'Match_Format']).agg(
                    Hundreds=('100s', 'sum')
                ).reset_index()
                best_career_100s_df = career_100s_df.sort_values(by='Hundreds', ascending=False).head(10)
                best_career_100s_df.insert(0, 'Rank', range(1, 1 + len(best_career_100s_df)))
                best_career_100s_df = best_career_100s_df[['Rank', 'Name', 'Match_Format', 'Hundreds']]

                st.dataframe(best_career_100s_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            with col16:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">🏆 Top 10 Most POM</h3>
                </div>
                """, unsafe_allow_html=True)

                pom_mask_career = (
                    filtered_bat_df['Player_of_the_Match'].astype(str).str.strip()
                    == filtered_bat_df['Name'].astype(str).str.strip()
                )
                pom_df_career = filtered_bat_df[pom_mask_career]
                career_pom_df = pom_df_career.groupby(['Name', 'Match_Format'])['File Name'].nunique().reset_index(name='POM')
                best_career_pom_df = career_pom_df.sort_values(by='POM', ascending=False).head(10)
                best_career_pom_df.insert(0, 'Rank', range(1, 1 + len(best_career_pom_df)))
                best_career_pom_df = best_career_pom_df[['Rank', 'Name', 'Match_Format', 'POM']]

                st.dataframe(best_career_pom_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            st.divider()
            col17, col18 = st.columns(2)

            with col17:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">🎯 Top 10 Boundary % (Min 500 Runs)</h3>
                </div>
                """, unsafe_allow_html=True)

                career_boundary_df = filtered_bat_df.groupby(['Name', 'Match_Format']).agg(
                    Runs=('Runs', 'sum'),
                    Fours=('4s', 'sum'),
                    Sixes=('6s', 'sum')
                ).reset_index()
                career_boundary_df = career_boundary_df[career_boundary_df['Runs'] >= 500]
                career_boundary_df['Boundary Runs'] = (career_boundary_df['Fours'] * 4) + (career_boundary_df['Sixes'] * 6)
                career_boundary_df['Boundary %'] = ((career_boundary_df['Boundary Runs'] / career_boundary_df['Runs'].replace(0, np.nan)) * 100).round(2)
                best_career_boundary_pct_df = career_boundary_df.sort_values(by='Boundary %', ascending=False, na_position='last').head(10)
                best_career_boundary_pct_df.insert(0, 'Rank', range(1, 1 + len(best_career_boundary_pct_df)))
                best_career_boundary_pct_df = best_career_boundary_pct_df[['Rank', 'Name', 'Match_Format', 'Boundary %', 'Boundary Runs', 'Runs']]

                st.dataframe(best_career_boundary_pct_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            file_stats_overall = filtered_bat_df.groupby(['File Name', 'Match_Format']).agg({
                'Total_Runs': 'first',
                'Team Balls': 'first',
                'Wickets': 'first'
            }).reset_index()
            file_stats_overall['Match_Avg'] = file_stats_overall['Total_Runs'] / file_stats_overall['Wickets'].replace(0, np.nan)
            file_stats_overall['Match_SR'] = (file_stats_overall['Total_Runs'] / file_stats_overall['Team Balls'].replace(0, np.nan)) * 100
            overall_match_metrics = file_stats_overall.groupby(['Match_Format']).agg(
                Overall_Avg_Match_Avg=('Match_Avg', 'mean'),
                Overall_Avg_Match_SR=('Match_SR', 'mean')
            ).reset_index()
            overall_match_metrics['Overall_Avg_Match_Avg'] = overall_match_metrics['Overall_Avg_Match_Avg'].round(2)
            overall_match_metrics['Overall_Avg_Match_SR'] = overall_match_metrics['Overall_Avg_Match_SR'].round(2)

            with col18:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">📊 Top 10 Best Match+ Avg (Min 10 Inns)</h3>
                </div>
                """, unsafe_allow_html=True)

                player_career_stats_avg = filtered_bat_df.groupby(['Name', 'Match_Format']).agg(
                    Runs=('Runs', 'sum'),
                    Out=('Out', 'sum'),
                    Inns=('Batted', 'sum')
                ).reset_index()
                player_career_stats_avg = player_career_stats_avg[player_career_stats_avg['Inns'] >= 10]
                player_career_stats_avg['Player_Avg'] = (player_career_stats_avg['Runs'] / player_career_stats_avg['Out'].replace(0, np.nan)).round(2)
                merged_career_avg = pd.merge(player_career_stats_avg, overall_match_metrics, on='Match_Format', how='left')
                merged_career_avg['Match+ Avg'] = ((merged_career_avg['Player_Avg'] / merged_career_avg['Overall_Avg_Match_Avg'].replace(0, np.nan)) * 100).round(2)
                best_career_match_plus_avg_df = merged_career_avg.sort_values(by='Match+ Avg', ascending=False, na_position='last').head(10)
                best_career_match_plus_avg_df.insert(0, 'Rank', range(1, 1 + len(best_career_match_plus_avg_df)))
                best_career_match_plus_avg_df = best_career_match_plus_avg_df[['Rank', 'Name', 'Match_Format', 'Match+ Avg', 'Player_Avg', 'Overall_Avg_Match_Avg', 'Inns']]

                st.dataframe(best_career_match_plus_avg_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})

            st.divider()
            col19, col20 = st.columns(2)

            with col19:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                            box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">⚡ Top 10 Best Match+ SR (Min 500 Balls)</h3>
                </div>
                """, unsafe_allow_html=True)

                player_career_stats_sr = filtered_bat_df.groupby(['Name', 'Match_Format']).agg(
                    Runs=('Runs', 'sum'),
                    Balls=('Balls', 'sum')
                ).reset_index()
                player_career_stats_sr = player_career_stats_sr[player_career_stats_sr['Balls'] >= 500]
                player_career_stats_sr['Player_SR'] = ((player_career_stats_sr['Runs'] / player_career_stats_sr['Balls'].replace(0, np.nan)) * 100).round(2)
                merged_career_sr = pd.merge(player_career_stats_sr, overall_match_metrics, on='Match_Format', how='left')
                merged_career_sr['Match+ SR'] = ((merged_career_sr['Player_SR'] / merged_career_sr['Overall_Avg_Match_SR'].replace(0, np.nan)) * 100).round(2)
                best_career_match_plus_sr_df = merged_career_sr.sort_values(by='Match+ SR', ascending=False, na_position='last').head(10)
                best_career_match_plus_sr_df.insert(0, 'Rank', range(1, 1 + len(best_career_match_plus_sr_df)))
                best_career_match_plus_sr_df = best_career_match_plus_sr_df[['Rank', 'Name', 'Match_Format', 'Match+ SR', 'Player_SR', 'Overall_Avg_Match_SR', 'Balls']]

                st.dataframe(best_career_match_plus_sr_df, use_container_width=True, hide_index=True, column_config={"Name": st.column_config.Column("Name", pinned=True)})
        else:
            st.warning("No batting data available for records analysis")

    with best_tabs[1]:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">🎳 Bowling Bests</h3>
            </div>
            """, unsafe_allow_html=True)

        if filtered_bowl_df is not None and not filtered_bowl_df.empty:
            working_bowl = filtered_bowl_df.copy()

            def _ensure_numeric(df: pd.DataFrame, column: str) -> None:
                if column in df.columns:
                    df[column] = pd.to_numeric(df[column], errors='coerce')

            for numeric_col in ['Bowler_Wkts', 'Bowler_Runs', 'Bowler_Balls', 'Balls', 'Bowler_Overs', 'Overs']:
                _ensure_numeric(working_bowl, numeric_col)

            match_count_col = 'File Name' if 'File Name' in working_bowl.columns else None
            runs_available = 'Bowler_Runs' in working_bowl.columns
            wickets_available = 'Bowler_Wkts' in working_bowl.columns

            def _overs_to_balls(value) -> float:
                if pd.isna(value):
                    return np.nan
                try:
                    s = str(value).strip()
                    if not s:
                        return np.nan
                    if ':' in s:
                        over_str, ball_str = s.split(':', 1)
                    elif '.' in s:
                        over_str, ball_str = s.split('.', 1)
                    else:
                        over_str, ball_str = s, '0'
                    overs_val = int(float(over_str))
                    ball_digits = ''.join(ch for ch in ball_str if ch.isdigit())
                    balls_val = int(ball_digits) if ball_digits else 0
                    if balls_val >= 6:
                        overs_val += balls_val // 6
                        balls_val = balls_val % 6
                    return float(overs_val * 6 + balls_val)
                except Exception:
                    return np.nan

            balls_available = False
            balls_source = next((col for col in ['Bowler_Balls', 'Balls'] if col in working_bowl.columns), None)
            if balls_source:
                working_bowl['Balls_Used'] = pd.to_numeric(working_bowl[balls_source], errors='coerce')
                balls_available = working_bowl['Balls_Used'].notna().any()
            else:
                overs_source = next((col for col in ['Bowler_Overs', 'Overs'] if col in working_bowl.columns), None)
                if overs_source:
                    working_bowl['Balls_Used'] = working_bowl[overs_source].apply(_overs_to_balls)
                    balls_available = working_bowl['Balls_Used'].notna().any()
                else:
                    working_bowl['Balls_Used'] = np.nan

            if balls_available:
                working_bowl['Balls_Used'] = working_bowl['Balls_Used'].astype(float)
                working_bowl['Overs_Used'] = working_bowl['Balls_Used'] / 6
            else:
                working_bowl['Overs_Used'] = np.nan

            st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                            box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">🎯 Single Innings Bests</h3>
                </div>
                """, unsafe_allow_html=True)

            best_single_df = process_best_bowling(filtered_bowl_df)
            if not best_single_df.empty:
                top_single_df = best_single_df.head(10).reset_index(drop=True)
                st.dataframe(
                    top_single_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={"Name": st.column_config.Column("Name", pinned=True)}
                )
            else:
                st.info("No bowling innings satisfy the criteria for single-innings bests with the current filters.")

            col_si1, col_si2 = st.columns(2)

            with col_si1:
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                                padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                                box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                                border: 1px solid rgba(255, 255, 255, 0.2);">
                        <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">💨 Best Economy (Min 20 Overs)</h3>
                    </div>
                    """, unsafe_allow_html=True)

                if runs_available and working_bowl['Overs_Used'].notna().any():
                    economy_candidates = working_bowl[
                        working_bowl['Overs_Used'].notna()
                        & (working_bowl['Overs_Used'] >= 20)
                        & working_bowl['Bowler_Runs'].notna()
                    ].copy()

                    if not economy_candidates.empty:
                        economy_candidates['Economy'] = (
                            economy_candidates['Bowler_Runs'] / economy_candidates['Overs_Used']
                        ).round(2)
                        economy_candidates['Overs'] = economy_candidates['Overs_Used'].round(1)
                        economy_candidates['Runs'] = economy_candidates['Bowler_Runs'].round(0).astype('Int64')
                        economy_candidates['Wickets'] = economy_candidates['Bowler_Wkts'].astype('Int64')
                        if 'Date' in economy_candidates.columns:
                            economy_candidates = format_date_column(economy_candidates, 'Date')

                        economy_best = economy_candidates.sort_values(
                            by=['Economy', 'Wickets', 'Runs'], ascending=[True, False, True]
                        ).head(10)
                        economy_best.insert(0, 'Rank', range(1, 1 + len(economy_best)))

                        econ_display = ['Rank', 'Name']
                        if 'Match_Format' in economy_best.columns:
                            econ_display.append('Match_Format')
                        econ_display.extend(['Economy', 'Overs', 'Runs', 'Wickets'])
                        if 'Date' in economy_best.columns:
                            econ_display.append('Date')
                        if 'Bat_Team' in economy_best.columns:
                            econ_display.append('Bat_Team')

                        st.dataframe(
                            economy_best[econ_display],
                            use_container_width=True,
                            hide_index=True,
                            column_config={"Name": st.column_config.Column("Name", pinned=True)}
                        )
                    else:
                        st.info("No innings meet the 20-over minimum for economy analysis.")
                else:
                    st.info("Economy analysis requires overs and run data.")

            with col_si2:
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                                padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                                box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                                border: 1px solid rgba(255, 255, 255, 0.2);">
                        <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">⚡ Best Strike Rate (Min 5 Wkts)</h3>
                    </div>
                    """, unsafe_allow_html=True)

                if balls_available and working_bowl['Balls_Used'].notna().any():
                    strike_candidates = working_bowl[
                        working_bowl['Balls_Used'].notna()
                        & (working_bowl['Balls_Used'] > 0)
                        & (working_bowl['Bowler_Wkts'] >= 5)
                    ].copy()

                    if not strike_candidates.empty:
                        strike_candidates['Strike Rate'] = (
                            strike_candidates['Balls_Used'] / strike_candidates['Bowler_Wkts']
                        ).round(2)
                        strike_candidates['Balls'] = strike_candidates['Balls_Used'].round(0).astype('Int64')
                        strike_candidates['Wickets'] = strike_candidates['Bowler_Wkts'].astype('Int64')
                        if 'Date' in strike_candidates.columns:
                            strike_candidates = format_date_column(strike_candidates, 'Date')

                        strike_best = strike_candidates.sort_values(
                            by=['Strike Rate', 'Wickets', 'Balls'], ascending=[True, False, True]
                        ).head(10)
                        strike_best.insert(0, 'Rank', range(1, 1 + len(strike_best)))

                        sr_display = ['Rank', 'Name']
                        if 'Match_Format' in strike_best.columns:
                            sr_display.append('Match_Format')
                        sr_display.extend(['Strike Rate', 'Balls', 'Wickets'])
                        if 'Date' in strike_best.columns:
                            sr_display.append('Date')
                        if 'Bat_Team' in strike_best.columns:
                            sr_display.append('Bat_Team')

                        st.dataframe(
                            strike_best[sr_display],
                            use_container_width=True,
                            hide_index=True,
                            column_config={"Name": st.column_config.Column("Name", pinned=True)}
                        )
                    else:
                        st.info("No innings meet the 5-wicket minimum for strike rate analysis.")
                else:
                    st.info("Strike rate analysis requires balls bowled and wicket data.")

            st.divider()
            st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                            box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">📊 Seasonal Bests (by Format)</h3>
                </div>
                """, unsafe_allow_html=True)

            if wickets_available and 'Year' in working_bowl.columns and 'Match_Format' in working_bowl.columns:
                season_agg: dict[str, tuple[str, str]] = {
                    'Wickets': ('Bowler_Wkts', 'sum')
                }
                if runs_available:
                    season_agg['Runs'] = ('Bowler_Runs', 'sum')
                if balls_available:
                    season_agg['Balls'] = ('Balls_Used', 'sum')
                    season_agg['Overs'] = ('Overs_Used', 'sum')
                if match_count_col:
                    season_agg['Matches'] = (match_count_col, 'nunique')
                else:
                    season_agg['Matches'] = ('Bowler_Wkts', 'count')

                season_group = (
                    working_bowl
                    .groupby(['Name', 'Year', 'Match_Format'], observed=True)
                    .agg(**season_agg)
                    .reset_index()
                )
                season_group = season_group[season_group['Wickets'] > 0]

                if 'Runs' in season_group.columns:
                    season_group['Average'] = np.where(
                        season_group['Wickets'] > 0,
                        (season_group['Runs'] / season_group['Wickets']).round(2),
                        np.nan
                    )
                else:
                    season_group['Average'] = np.nan

                if 'Balls' in season_group.columns and 'Runs' in season_group.columns:
                    season_group['Strike Rate'] = np.where(
                        (season_group['Wickets'] > 0) & (season_group['Balls'] > 0),
                        (season_group['Balls'] / season_group['Wickets']).round(2),
                        np.nan
                    )
                    season_group['Economy'] = np.where(
                        (season_group['Balls'] > 0),
                        (season_group['Runs'] / (season_group['Balls'] / 6)).round(2),
                        np.nan
                    )
                else:
                    season_group['Strike Rate'] = np.nan
                    season_group['Economy'] = np.nan

                col_s1, col_s2 = st.columns(2)

                with col_s1:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                                padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                                box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                                border: 1px solid rgba(255, 255, 255, 0.2);">
                        <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">🎯 Top 10 Most Wickets (Season)</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    top_season_wkts = season_group.sort_values(by='Wickets', ascending=False).head(10).copy()
                    if top_season_wkts.empty:
                        st.info("No seasonal wicket data available for the current filters.")
                    else:
                        top_season_wkts.insert(0, 'Rank', range(1, 1 + len(top_season_wkts)))
                        display_cols = ['Rank', 'Name', 'Year', 'Match_Format', 'Wickets']
                        if 'Matches' in top_season_wkts.columns:
                            display_cols.append('Matches')
                        st.dataframe(
                            top_season_wkts[display_cols],
                            use_container_width=True,
                            hide_index=True,
                            column_config={"Name": st.column_config.Column("Name", pinned=True)}
                        )

                with col_s2:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                                padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                                box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                                border: 1px solid rgba(255, 255, 255, 0.2);">
                        <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">📈 Top 10 Best Bowling Avg (Season, Min 10 Wkts)</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    if 'Average' in season_group.columns:
                        season_avg = season_group[(season_group['Wickets'] >= 10) & season_group['Average'].notna()].sort_values(by='Average', ascending=True).head(10).copy()
                    else:
                        season_avg = pd.DataFrame()

                    if season_avg.empty:
                        st.info("Not enough data to compute seasonal bowling averages (requires runs and at least 10 wickets).")
                    else:
                        season_avg = season_avg.copy()
                        if 'Overs' in season_avg.columns:
                            season_avg['Overs'] = season_avg['Overs'].round(1)
                        season_avg.insert(0, 'Rank', range(1, 1 + len(season_avg)))
                        display_cols = ['Rank', 'Name', 'Year', 'Match_Format', 'Average', 'Wickets']
                        if 'Overs' in season_avg.columns:
                            display_cols.append('Overs')
                        st.dataframe(
                            season_avg[display_cols],
                            use_container_width=True,
                            hide_index=True,
                            column_config={"Name": st.column_config.Column("Name", pinned=True)}
                        )

                col_s3, col_s4 = st.columns(2)

                with col_s3:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                                padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                                box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                                border: 1px solid rgba(255, 255, 255, 0.2);">
                        <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">⚡ Top 10 Best Strike Rate (Season, Min 10 Wkts)</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    if 'Strike Rate' in season_group.columns:
                        season_sr = season_group[(season_group['Wickets'] >= 10) & season_group['Strike Rate'].notna()].sort_values(by='Strike Rate', ascending=True).head(10).copy()
                    else:
                        season_sr = pd.DataFrame()

                    if season_sr.empty:
                        st.info("Not enough ball data to compute seasonal strike rates (requires balls and at least 10 wickets).")
                    else:
                        season_sr.insert(0, 'Rank', range(1, 1 + len(season_sr)))
                        sr_cols = ['Rank', 'Name', 'Year', 'Match_Format', 'Strike Rate', 'Wickets']
                        if 'Balls' in season_sr.columns:
                            season_sr['Balls'] = season_sr['Balls'].round(0).astype(int)
                            sr_cols.append('Balls')
                        st.dataframe(
                            season_sr[sr_cols],
                            use_container_width=True,
                            hide_index=True,
                            column_config={"Name": st.column_config.Column("Name", pinned=True)}
                        )

                with col_s4:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                                padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                                box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                                border: 1px solid rgba(255, 255, 255, 0.2);">
                        <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">💨 Top 10 Best Economy (Season, Min 10 Wkts)</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    if 'Economy' in season_group.columns:
                        season_econ = season_group[(season_group['Wickets'] >= 10) & season_group['Economy'].notna()].sort_values(by='Economy', ascending=True).head(10).copy()
                    else:
                        season_econ = pd.DataFrame()

                    if season_econ.empty:
                        st.info("Not enough data to compute seasonal economy rates (requires runs/balls and at least 10 wickets).")
                    else:
                        season_econ.insert(0, 'Rank', range(1, 1 + len(season_econ)))
                        econ_cols = ['Rank', 'Name', 'Year', 'Match_Format', 'Economy', 'Wickets']
                        if 'Overs' in season_econ.columns:
                            season_econ['Overs'] = (season_econ['Overs'].round(2))
                            econ_cols.append('Overs')
                        st.dataframe(
                            season_econ[econ_cols],
                            use_container_width=True,
                            hide_index=True,
                            column_config={"Name": st.column_config.Column("Name", pinned=True)}
                        )
            else:
                st.info("Seasonal bowling metrics require Year, Match_Format, and wicket data.")

            st.divider()
            st.markdown("""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                            box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">🏆 Career Bests (by Format)</h3>
                </div>
                """, unsafe_allow_html=True)

            if wickets_available and 'Match_Format' in working_bowl.columns:
                career_agg: dict[str, tuple[str, str]] = {
                    'Wickets': ('Bowler_Wkts', 'sum')
                }
                if runs_available:
                    career_agg['Runs'] = ('Bowler_Runs', 'sum')
                if balls_available:
                    career_agg['Balls'] = ('Balls_Used', 'sum')
                    career_agg['Overs'] = ('Overs_Used', 'sum')
                if match_count_col:
                    career_agg['Matches'] = (match_count_col, 'nunique')
                else:
                    career_agg['Matches'] = ('Bowler_Wkts', 'count')

                career_group = (
                    working_bowl
                    .groupby(['Name', 'Match_Format'], observed=True)
                    .agg(**career_agg)
                    .reset_index()
                )
                career_group = career_group[career_group['Wickets'] > 0]

                if 'Runs' in career_group.columns:
                    career_group['Average'] = np.where(
                        career_group['Wickets'] > 0,
                        (career_group['Runs'] / career_group['Wickets']).round(2),
                        np.nan
                    )
                else:
                    career_group['Average'] = np.nan

                if 'Balls' in career_group.columns and 'Runs' in career_group.columns:
                    career_group['Strike Rate'] = np.where(
                        (career_group['Wickets'] > 0) & (career_group['Balls'] > 0),
                        (career_group['Balls'] / career_group['Wickets']).round(2),
                        np.nan
                    )
                    career_group['Economy'] = np.where(
                        (career_group['Balls'] > 0),
                        (career_group['Runs'] / (career_group['Balls'] / 6)).round(2),
                        np.nan
                    )
                else:
                    career_group['Strike Rate'] = np.nan
                    career_group['Economy'] = np.nan

                col_c1, col_c2 = st.columns(2)

                with col_c1:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                                padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                                box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                                border: 1px solid rgba(255, 255, 255, 0.2);">
                        <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">🎯 Top 10 Most Wickets (Career)</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    top_career_wkts = career_group.sort_values(by='Wickets', ascending=False).head(10).copy()
                    if top_career_wkts.empty:
                        st.info("No career wicket data available for the current filters.")
                    else:
                        top_career_wkts.insert(0, 'Rank', range(1, 1 + len(top_career_wkts)))
                        display_cols = ['Rank', 'Name', 'Match_Format', 'Wickets']
                        if 'Matches' in top_career_wkts.columns:
                            display_cols.append('Matches')
                        st.dataframe(
                            top_career_wkts[display_cols],
                            use_container_width=True,
                            hide_index=True,
                            column_config={"Name": st.column_config.Column("Name", pinned=True)}
                        )

                with col_c2:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                                padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                                box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                                border: 1px solid rgba(255, 255, 255, 0.2);">
                        <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">📈 Top 10 Best Bowling Avg (Career, Min 50 Wkts)</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    if 'Average' in career_group.columns:
                        career_avg = career_group[(career_group['Wickets'] >= 50) & career_group['Average'].notna()].sort_values(by='Average', ascending=True).head(10).copy()
                    else:
                        career_avg = pd.DataFrame()

                    if career_avg.empty:
                        st.info("Not enough data to compute career bowling averages (requires runs and at least 50 wickets).")
                    else:
                        career_avg.insert(0, 'Rank', range(1, 1 + len(career_avg)))
                        st.dataframe(
                            career_avg[['Rank', 'Name', 'Match_Format', 'Average', 'Wickets', 'Runs']],
                            use_container_width=True,
                            hide_index=True,
                            column_config={"Name": st.column_config.Column("Name", pinned=True)}
                        )

                col_c3, col_c4 = st.columns(2)

                with col_c3:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                                padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                                box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                                border: 1px solid rgba(255, 255, 255, 0.2);">
                        <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">⚡ Top 10 Best Strike Rate (Career, Min 50 Wkts)</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    if 'Strike Rate' in career_group.columns:
                        career_sr = career_group[(career_group['Wickets'] >= 50) & career_group['Strike Rate'].notna()].sort_values(by='Strike Rate', ascending=True).head(10).copy()
                    else:
                        career_sr = pd.DataFrame()

                    if career_sr.empty:
                        st.info("Not enough ball data to compute career strike rates (requires balls and at least 50 wickets).")
                    else:
                        career_sr.insert(0, 'Rank', range(1, 1 + len(career_sr)))
                        sr_cols = ['Rank', 'Name', 'Match_Format', 'Strike Rate', 'Wickets']
                        if 'Balls' in career_sr.columns:
                            career_sr['Balls'] = career_sr['Balls'].round(0).astype(int)
                            sr_cols.append('Balls')
                        st.dataframe(
                            career_sr[sr_cols],
                            use_container_width=True,
                            hide_index=True,
                            column_config={"Name": st.column_config.Column("Name", pinned=True)}
                        )

                with col_c4:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                                padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                                box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                                border: 1px solid rgba(255, 255, 255, 0.2);">
                        <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">💨 Top 10 Best Economy (Career, Min 50 Wkts)</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    if 'Economy' in career_group.columns:
                        career_econ = career_group[(career_group['Wickets'] >= 50) & career_group['Economy'].notna()].sort_values(by='Economy', ascending=True).head(10).copy()
                    else:
                        career_econ = pd.DataFrame()

                    if career_econ.empty:
                        st.info("Not enough data to compute career economy rates (requires runs/balls and at least 50 wickets).")
                    else:
                        career_econ.insert(0, 'Rank', range(1, 1 + len(career_econ)))
                        econ_cols = ['Rank', 'Name', 'Match_Format', 'Economy', 'Wickets']
                        if 'Overs' in career_econ.columns:
                            career_econ['Overs'] = career_econ['Overs'].round(2)
                            econ_cols.append('Overs')
                        st.dataframe(
                            career_econ[econ_cols],
                            use_container_width=True,
                            hide_index=True,
                            column_config={"Name": st.column_config.Column("Name", pinned=True)}
                        )

                col_c5, col_c6 = st.columns(2)

                with col_c5:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                                padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                                box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                                border: 1px solid rgba(255, 255, 255, 0.2);">
                        <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">🎳 Top 10 Most 5W Hauls (Career)</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    if wickets_available:
                        fivew_source = filtered_bowl_df[filtered_bowl_df.get('Bowler_Wkts', 0) >= 5]
                        if not fivew_source.empty:
                            if match_count_col:
                                career_fivew = fivew_source.groupby(['Name', 'Match_Format'])[match_count_col].nunique().reset_index(name='5W Hauls')
                            else:
                                career_fivew = fivew_source.groupby(['Name', 'Match_Format']).size().reset_index(name='5W Hauls')
                            career_fivew = career_fivew.sort_values(by='5W Hauls', ascending=False).head(10)
                            career_fivew.insert(0, 'Rank', range(1, 1 + len(career_fivew)))
                            st.dataframe(
                                career_fivew[['Rank', 'Name', 'Match_Format', '5W Hauls']],
                                use_container_width=True,
                                hide_index=True,
                                column_config={"Name": st.column_config.Column("Name", pinned=True)}
                            )
                        else:
                            st.info("No five-wicket hauls found for career analysis.")
                    else:
                        st.info("Wicket data is required to compute five-wicket hauls.")

                with col_c6:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                                padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                                box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                                border: 1px solid rgba(255, 255, 255, 0.2);">
                        <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">💥 Top 10 Most 10W Hauls (Match)</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    if wickets_available and 'Match_Format' in filtered_bowl_df.columns:
                        match_keys = ['Name', 'Match_Format']
                        match_identifier = None

                        if match_count_col:
                            match_keys.append(match_count_col)
                            match_identifier = match_count_col
                        elif 'Date' in filtered_bowl_df.columns:
                            match_keys.append('Date')
                            match_identifier = 'Date'

                        if match_identifier is None:
                            st.info("Match identifiers (e.g. File Name or Date) are required to calculate 10-wicket hauls.")
                        else:
                            match_totals = (
                                filtered_bowl_df
                                .groupby(match_keys, observed=True)['Bowler_Wkts']
                                .sum()
                                .reset_index(name='Match Wickets')
                            )
                            match_totals['Match Wickets'] = pd.to_numeric(match_totals['Match Wickets'], errors='coerce')
                            match_totals = match_totals.dropna(subset=['Match Wickets'])

                            tenw_matches = match_totals[match_totals['Match Wickets'] >= 10].copy()

                            if tenw_matches.empty:
                                st.info("No ten-wicket match hauls found for career analysis.")
                            else:
                                tenw_summary = (
                                    tenw_matches
                                    .groupby(['Name', 'Match_Format'], observed=True)['Match Wickets']
                                    .agg(['count', 'max'])
                                    .reset_index()
                                    .rename(columns={'count': '10W Hauls', 'max': 'Best Match Wickets'})
                                )

                                if match_identifier in tenw_matches.columns:
                                    if isinstance(tenw_matches[match_identifier].dtype, CategoricalDtype):
                                        tenw_matches = tenw_matches.copy()
                                        tenw_matches[match_identifier] = tenw_matches[match_identifier].astype(str)
                                    elif match_identifier != 'Date' and tenw_matches[match_identifier].dtype == object:
                                        tenw_matches[match_identifier] = tenw_matches[match_identifier].astype(str)
                                    recent_match = (
                                        tenw_matches
                                        .groupby(['Name', 'Match_Format'], observed=True)[match_identifier]
                                        .max()
                                        .reset_index(name='Most Recent Match')
                                    )
                                    tenw_summary = tenw_summary.merge(recent_match, on=['Name', 'Match_Format'], how='left')
                                    if match_identifier == 'Date':
                                        tenw_summary['Most Recent Match'] = pd.to_datetime(
                                            tenw_summary['Most Recent Match'], errors='coerce'
                                        ).dt.strftime('%d/%m/%Y')

                                tenw_summary = tenw_summary.sort_values(
                                    by=['10W Hauls', 'Best Match Wickets'], ascending=[False, False]
                                ).head(10)
                                tenw_summary['Best Match Wickets'] = (
                                    pd.to_numeric(tenw_summary['Best Match Wickets'], errors='coerce')
                                    .round(0)
                                    .astype('Int64')
                                )
                                tenw_summary.insert(0, 'Rank', range(1, 1 + len(tenw_summary)))

                                display_cols = ['Rank', 'Name', 'Match_Format', '10W Hauls', 'Best Match Wickets']
                                if 'Most Recent Match' in tenw_summary.columns:
                                    display_cols.append('Most Recent Match')

                                st.dataframe(
                                    tenw_summary[display_cols],
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config={"Name": st.column_config.Column("Name", pinned=True)}
                                )
                    else:
                        st.info("Wicket data and match formats are required to compute ten-wicket hauls.")

                col_c7, _ = st.columns(2)

                with col_c7:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                                padding: 0.8rem; margin: 1rem 0; border-radius: 12px; 
                                box-shadow: 0 6px 24px rgba(250, 112, 154, 0.3);
                                border: 1px solid rgba(255, 255, 255, 0.2);">
                        <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.2rem; text-align: center;">🏆 Top 10 Most POM (Career)</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    if {'Player_of_the_Match', 'Name', 'Match_Format'}.issubset(filtered_bowl_df.columns):
                        pom_mask_bowl = (
                            filtered_bowl_df['Player_of_the_Match'].astype(str).str.strip()
                            == filtered_bowl_df['Name'].astype(str).str.strip()
                        )
                        pom_df_bowl_career = filtered_bowl_df[pom_mask_bowl]

                        if not pom_df_bowl_career.empty and match_count_col:
                            career_bowl_pom_df = (
                                pom_df_bowl_career
                                .groupby(['Name', 'Match_Format'])[match_count_col]
                                .nunique()
                                .reset_index(name='POM')
                            )
                        elif not pom_df_bowl_career.empty:
                            career_bowl_pom_df = (
                                pom_df_bowl_career
                                .groupby(['Name', 'Match_Format'])
                                .size()
                                .reset_index(name='POM')
                            )
                        else:
                            career_bowl_pom_df = pd.DataFrame()

                        if career_bowl_pom_df.empty:
                            st.info("No Player of the Match awards found for the current bowling filters.")
                        else:
                            best_career_bowl_pom_df = career_bowl_pom_df.sort_values(by='POM', ascending=False).head(10)
                            best_career_bowl_pom_df.insert(0, 'Rank', range(1, 1 + len(best_career_bowl_pom_df)))
                            st.dataframe(
                                best_career_bowl_pom_df[['Rank', 'Name', 'Match_Format', 'POM']],
                                use_container_width=True,
                                hide_index=True,
                                column_config={"Name": st.column_config.Column("Name", pinned=True)}
                            )
                    else:
                        st.info("Insufficient bowling data to calculate Player of the Match totals.")
            else:
                st.info("Career bowling metrics require Match_Format and wicket data.")
        else:
            st.warning("No bowling data available for best-of analysis")

# Win/Loss record Tab
with tabs[6]:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.2);">
        <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">🏆 Player Win/Loss Record</h3>
    </div>
    """, unsafe_allow_html=True)

    if filtered_bat_df is not None and not filtered_bat_df.empty and filtered_match_df is not None and not filtered_match_df.empty:
        team_column = _first_existing_column(
            filtered_bat_df,
            ['Bat_Team', 'Bat Team', 'Bat_Team_x', 'Bat_Team_y', 'Team']
        )

        if team_column is None:
            st.warning("Unable to determine player teams for win/loss analysis with the current data.")
        else:
            format_column = _first_existing_column(
                filtered_bat_df,
                ['Match_Format', 'Match Format', 'Format']
            )

            base_columns = ['Name', 'File Name', team_column]
            if format_column:
                base_columns.append(format_column)

            player_matches = filtered_bat_df[base_columns].copy()
            player_matches['Name'] = player_matches['Name'].astype('string').str.strip()
            player_matches['File Name'] = player_matches['File Name'].astype('string').str.strip()
            player_matches['Player_Team'] = (
                player_matches[team_column]
                .astype('string')
                .fillna('')
                .str.strip()
            )
            if format_column:
                player_matches['Player_Format'] = (
                    player_matches[format_column]
                    .astype('string')
                    .fillna('')
                    .str.strip()
                )

            player_matches['_team_sort'] = player_matches['Player_Team'].eq('').astype(int)
            sort_fields = ['Name', 'File Name', '_team_sort']
            if format_column:
                player_matches['_format_sort'] = player_matches['Player_Format'].eq('').astype(int)
                sort_fields.append('_format_sort')

            player_matches = player_matches.sort_values(sort_fields)
            player_matches = player_matches.drop_duplicates(subset=['Name', 'File Name'], keep='first')
            drop_cols = ['_team_sort']
            if format_column:
                drop_cols.append('_format_sort')
            player_matches = player_matches.drop(columns=drop_cols)
            player_matches = player_matches[player_matches['Player_Team'] != '']

            if player_matches.empty:
                st.warning("No player team information available for the selected filters.")
            elif 'File Name' not in filtered_match_df.columns:
                st.warning("Match data is missing file identifiers required for win/loss analysis.")
            else:
                match_results_df = filtered_match_df.copy()
                match_results_df['File Name'] = match_results_df['File Name'].astype('string').str.strip()
                match_results_df = match_results_df.drop_duplicates(subset=['File Name'], keep='last')

                wl_df = player_matches.merge(
                    match_results_df,
                    on='File Name',
                    how='left',
                    suffixes=('', '_match')
                )

                if wl_df.empty:
                    st.warning("No player matches align with match results for win/loss analysis.")
                elif {'Home_Team', 'Away_Team'}.issubset(wl_df.columns) is False:
                    st.warning("Match data is missing home/away team information; cannot build win/loss summary.")
                else:
                    result_columns = [
                        'Home_Win', 'Away_Won', 'Home_Drawn', 'Away_Drawn',
                        'Tie', 'Innings_Win', 'Home_Lost', 'Away_Lost'
                    ]

                    for column in result_columns:
                        if column not in wl_df.columns:
                            wl_df[column] = 0
                        else:
                            wl_df[column] = pd.to_numeric(wl_df[column], errors='coerce').fillna(0)

                    def _normalize_label(series: pd.Series) -> pd.Series:
                        normalized = series.astype('string').fillna('').str.strip().str.lower()
                        return normalized.str.replace(r'[^a-z0-9]+', '', regex=True)

                    wl_df['Player_Team_Norm'] = _normalize_label(wl_df['Player_Team'])
                    wl_df['Home_Team_Norm'] = _normalize_label(wl_df['Home_Team'])
                    wl_df['Away_Team_Norm'] = _normalize_label(wl_df['Away_Team'])

                    wl_df['is_home'] = (
                        wl_df['Player_Team_Norm'].ne('') &
                        wl_df['Player_Team_Norm'].eq(wl_df['Home_Team_Norm'])
                    )
                    wl_df['is_away'] = (
                        wl_df['Player_Team_Norm'].ne('') &
                        wl_df['Player_Team_Norm'].eq(wl_df['Away_Team_Norm'])
                    )

                    wl_df = wl_df[wl_df['is_home'] | wl_df['is_away']]

                    if wl_df.empty:
                        st.warning("Unable to align player teams with match records for win/loss analysis.")
                    else:
                        wl_df['Home_Match_Flag'] = wl_df['is_home'].astype(int)
                        wl_df['Away_Match_Flag'] = wl_df['is_away'].astype(int)
                        wl_df['Total_Match_Flag'] = wl_df['Home_Match_Flag'] + wl_df['Away_Match_Flag']

                        wl_df['Home_Win_Flag'] = np.where(wl_df['is_home'], wl_df['Home_Win'], 0)
                        wl_df['Away_Win_Flag'] = np.where(wl_df['is_away'], wl_df['Away_Won'], 0)
                        wl_df['Home_Loss_Flag'] = np.where(wl_df['is_home'], wl_df['Home_Lost'], 0)
                        wl_df['Away_Loss_Flag'] = np.where(wl_df['is_away'], wl_df['Away_Lost'], 0)
                        wl_df['Home_Draw_Flag'] = np.where(wl_df['is_home'], wl_df['Home_Drawn'], 0)
                        wl_df['Away_Draw_Flag'] = np.where(wl_df['is_away'], wl_df['Away_Drawn'], 0)
                        wl_df['Tie_Flag'] = np.where(
                            (wl_df['is_home'] | wl_df['is_away']) & (wl_df['Tie'] > 0),
                            1,
                            0
                        )
                        wl_df['Innings_Win_Flag'] = np.where(
                            (
                                (wl_df['is_home'] & (wl_df['Home_Win'] > 0)) |
                                (wl_df['is_away'] & (wl_df['Away_Won'] > 0))
                            ) & (wl_df['Innings_Win'] > 0),
                            1,
                            0
                        )

                        wl_df = wl_df[wl_df['Total_Match_Flag'] > 0]

                        if wl_df.empty:
                            st.warning("No qualifying matches remain after aligning players with teams")
                        else:
                            def _primary_team(series: pd.Series) -> str:
                                non_empty = (
                                    series.astype('string')
                                    .fillna('')
                                    .str.strip()
                                )
                                non_empty = non_empty[non_empty != '']
                                if non_empty.empty:
                                    return ''
                                mode = non_empty.mode()
                                return mode.iloc[0] if not mode.empty else non_empty.iloc[0]

                            team_mode = (
                                player_matches[['Name', 'Player_Team']]
                                .groupby('Name')['Player_Team']
                                .agg(_primary_team)
                                .reset_index(name='Primary Team')
                            )

                            aggregation_map = {
                                'Total_Match_Flag': 'sum',
                                'Home_Match_Flag': 'sum',
                                'Away_Match_Flag': 'sum',
                                'Home_Win_Flag': 'sum',
                                'Away_Win_Flag': 'sum',
                                'Home_Loss_Flag': 'sum',
                                'Away_Loss_Flag': 'sum',
                                'Home_Draw_Flag': 'sum',
                                'Away_Draw_Flag': 'sum',
                                'Tie_Flag': 'sum',
                                'Innings_Win_Flag': 'sum'
                            }

                            winloss_summary = (
                                wl_df
                                .groupby('Name', as_index=False)
                                .agg(aggregation_map)
                                .rename(columns={
                                    'Total_Match_Flag': 'Total Matches',
                                    'Home_Match_Flag': 'Home Matches',
                                    'Away_Match_Flag': 'Away Matches',
                                    'Home_Win_Flag': 'Home Wins',
                                    'Away_Win_Flag': 'Away Wins',
                                    'Home_Loss_Flag': 'Home Losses',
                                    'Away_Loss_Flag': 'Away Losses',
                                    'Home_Draw_Flag': 'Home Draws',
                                    'Away_Draw_Flag': 'Away Draws',
                                    'Tie_Flag': 'Ties',
                                    'Innings_Win_Flag': 'Innings Wins'
                                })
                            )

                            winloss_summary = winloss_summary.merge(team_mode, on='Name', how='left')

                            winloss_summary['Total Wins'] = (
                                winloss_summary['Home Wins'] + winloss_summary['Away Wins']
                            )
                            winloss_summary['Total Losses'] = (
                                winloss_summary['Home Losses'] + winloss_summary['Away Losses']
                            )
                            winloss_summary['Total Draws'] = (
                                winloss_summary['Home Draws'] + winloss_summary['Away Draws']
                            )
                            winloss_summary['Win %'] = np.where(
                                winloss_summary['Total Matches'] > 0,
                                (winloss_summary['Total Wins'] / winloss_summary['Total Matches']) * 100,
                                0
                            )

                            int_columns = [
                                'Total Matches', 'Home Matches', 'Away Matches', 'Home Wins', 'Away Wins',
                                'Home Losses', 'Away Losses', 'Home Draws', 'Away Draws', 'Ties',
                                'Innings Wins', 'Total Wins', 'Total Losses', 'Total Draws'
                            ]

                            for col in int_columns:
                                if col in winloss_summary.columns:
                                    winloss_summary[col] = (
                                        pd.to_numeric(winloss_summary[col], errors='coerce')
                                        .fillna(0)
                                        .round(0)
                                        .astype('Int64')
                                    )

                            winloss_summary['Win %'] = winloss_summary['Win %'].round(1)

                            display_order = ['Name']
                            if 'Primary Team' in winloss_summary.columns:
                                display_order.append('Primary Team')
                            display_order.extend([
                                'Home Matches', 'Home Wins', 'Home Losses', 'Home Draws',
                                'Away Matches', 'Away Wins', 'Away Losses', 'Away Draws',
                                'Total Matches', 'Total Wins', 'Total Losses', 'Total Draws', 'Ties',
                                'Innings Wins', 'Win %'
                            ])

                            winloss_summary = winloss_summary[[
                                col for col in display_order if col in winloss_summary.columns
                            ]]

                            winloss_summary = winloss_summary.sort_values(
                                ['Total Wins', 'Win %', 'Total Matches'],
                                ascending=[False, False, False]
                            ).reset_index(drop=True)

                            st.dataframe(
                                winloss_summary,
                                use_container_width=True,
                                hide_index=True,
                                height=600,
                                column_config={
                                    "Name": st.column_config.Column("Name", pinned=True),
                                    "Win %": st.column_config.NumberColumn("Win %", format="%.1f%%"),
                                    "Primary Team": st.column_config.Column("Primary Team")
                                }
                            )

                            st.divider()
    else:
        st.warning("No batting or match data available for Win/Loss analysis.")
