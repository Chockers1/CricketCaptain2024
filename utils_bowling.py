# Utility functions for overs, economy, strike rate, and average

import numpy as np
import pandas as pd

def convert_overs_to_balls(overs, match_format=None):
    """Convert overs to balls, handling different formats"""
    if pd.isna(overs): 
        return 0
    if match_format in ['The Hundred', '100 Ball Trophy']: 
        return overs
    whole = int(overs)
    part = int(round((overs - whole) * 10))
    return (whole * 6) + part

def convert_balls_to_overs(balls, match_format=None):
    """Convert balls to overs, handling different formats"""
    if match_format in ['The Hundred', '100 Ball Trophy']:
        return balls / 5
    whole = balls // 6
    part = balls % 6
    return float(f"{whole}.{part}")

def calculate_economy(runs, balls, match_format=None):
    """Calculate economy rate based on format"""
    if balls == 0: 
        return 0
    return (runs / balls) * (5 if match_format in ['The Hundred', '100 Ball Trophy'] else 6)

def calculate_strike_rate(balls, wickets):
    """Calculate bowling strike rate (balls per wicket)"""
    return balls / wickets if wickets else np.nan

def calculate_average(runs, wickets):
    """Calculate bowling average (runs per wicket)"""
    return runs / wickets if wickets else np.nan

def calculate_bowling_metrics_vectorized(df):
    """
    Vectorized calculation of all bowling metrics for better performance.
    Uses pandas operations instead of apply() for speed.
    
    Args:
        df: DataFrame with bowling data
    
    Returns:
        DataFrame with calculated metrics
    """
    result_df = df.copy()
    
    # Vectorized overs calculation
    is_hundred = df['Match_Format'].isin(['The Hundred', '100 Ball Trophy'])
    
    # For regular formats (6 balls per over)
    regular_mask = ~is_hundred
    if regular_mask.any():
        regular_balls = df.loc[regular_mask, 'Bowler_Balls']
        result_df.loc[regular_mask, 'Overs'] = (regular_balls // 6) + (regular_balls % 6) / 10
    
    # For The Hundred (5 balls per over)
    if is_hundred.any():
        hundred_balls = df.loc[is_hundred, 'Bowler_Balls']
        result_df.loc[is_hundred, 'Overs'] = hundred_balls / 5
    
    # Vectorized metrics calculation
    result_df['Average'] = np.where(
        df['Bowler_Wkts'] > 0, 
        (df['Bowler_Runs'] / df['Bowler_Wkts']).round(2), 
        np.nan
    )
    
    result_df['Strike_Rate'] = np.where(
        df['Bowler_Wkts'] > 0, 
        (df['Bowler_Balls'] / df['Bowler_Wkts']).round(2), 
        np.nan
    )
    
    # Vectorized economy calculation
    result_df['Economy'] = np.where(
        result_df['Overs'] > 0,
        (df['Bowler_Runs'] / result_df['Overs']).round(2),
        0
    )
    
    # Clean up infinities
    result_df = result_df.replace([np.inf, -np.inf], np.nan)
    
    return result_df

def get_bowling_summary_stats(df, group_cols):
    """
    Generate bowling summary statistics for grouped data.
    Optimized for performance with vectorized operations.
    
    Args:
        df: Filtered bowling dataframe
        group_cols: List of columns to group by
        
    Returns:
        DataFrame with summary statistics
    """
    # Group and aggregate
    summary = df.groupby(group_cols).agg({
        'File Name': 'nunique',
        'Bowler_Balls': 'sum',
        'Maidens': 'sum',
        'Bowler_Runs': 'sum',
        'Bowler_Wkts': 'sum'
    }).reset_index()
    
    # Calculate metrics using vectorized operations
    summary['Overs'] = (summary['Bowler_Balls'] // 6) + (summary['Bowler_Balls'] % 6) / 10
    summary['Overs'] = summary['Overs'].round(2)
    
    summary['Average'] = np.where(
        summary['Bowler_Wkts'] > 0,
        (summary['Bowler_Runs'] / summary['Bowler_Wkts']).round(2),
        np.nan
    )
    
    summary['Strike_Rate'] = np.where(
        summary['Bowler_Wkts'] > 0,
        (summary['Bowler_Balls'] / summary['Bowler_Wkts']).round(2),
        np.nan
    )
    
    summary['Economy'] = np.where(
        summary['Overs'] > 0,
        (summary['Bowler_Runs'] / summary['Overs']).round(2),
        0
    )
    
    summary['WPM'] = np.where(
        summary['File Name'] > 0,
        (summary['Bowler_Wkts'] / summary['File Name']).round(2),
        0
    )
    
    # Clean up infinities
    summary = summary.replace([np.inf, -np.inf], np.nan)
    
    return summary

def calculate_vectorized_bowling_stats(df, runs_col='Bowler_Runs', wickets_col='Bowler_Wkts', balls_col='Bowler_Balls'):
    """
    Calculate bowling statistics using vectorized operations for better performance.
    Replaces slow .apply() loops with fast NumPy operations.
    
    Args:
        df: DataFrame with bowling data
        runs_col: Column name for runs conceded
        wickets_col: Column name for wickets taken
        balls_col: Column name for balls bowled
        
    Returns:
        DataFrame with calculated Average, Strike_Rate, and Overs columns
    """
    result_df = df.copy()
    
    # Vectorized average calculation
    result_df['Average'] = np.where(
        df[wickets_col] > 0,
        (df[runs_col] / df[wickets_col]).round(2),
        np.nan
    )
    
    # Vectorized strike rate calculation
    result_df['Strike_Rate'] = np.where(
        df[wickets_col] > 0,
        (df[balls_col] / df[wickets_col]).round(2),
        np.nan
    )
    
    # Vectorized overs calculation
    result_df['Overs'] = (df[balls_col] // 6) + (df[balls_col] % 6) / 10
    result_df['Overs'] = result_df['Overs'].round(2)
    
    # Vectorized economy calculation
    result_df['Economy_Rate'] = np.where(
        result_df['Overs'] > 0,
        (df[runs_col] / result_df['Overs']).round(2),
        0
    )
    
    return result_df

def get_latest_innings_summary(df, num_innings=20):
    """
    Get the latest bowling innings with optimized calculations.
    
    Args:
        df: Filtered bowling dataframe
        num_innings: Number of latest innings to return
        
    Returns:
        DataFrame with latest innings and calculated metrics
    """
    # Group by player and innings, then sort by date
    latest_inns = df.groupby(['Name', 'Match_Format', 'Date', 'Innings']).agg({
        'Bowl_Team': 'first',
        'Bat_Team': 'first',
        'Overs': 'sum',
        'Maidens': 'sum',
        'Bowler_Runs': 'sum',
        'Bowler_Wkts': 'sum',
        'File Name': 'first'
    }).reset_index()
    
    # Sort by date and take latest innings
    latest_inns['Date'] = pd.to_datetime(latest_inns['Date'])
    latest_inns = latest_inns.sort_values(by='Date', ascending=False).head(num_innings)
    
    return latest_inns

def calculate_opponent_stats(df):
    """
    Calculate bowling statistics by opponent with vectorized operations.
    
    Args:
        df: Filtered bowling dataframe
        
    Returns:
        DataFrame with opponent statistics
    """
    opponent_summary = df.groupby(['Name', 'Bat_Team']).agg({
        'File Name': 'nunique',
        'Bowler_Balls': 'sum',
        'Maidens': 'sum',
        'Bowler_Runs': 'sum',
        'Bowler_Wkts': 'sum'
    }).reset_index()
    
    opponent_summary.columns = ['Name', 'Opposition', 'Matches', 'Balls', 'M/D', 'Runs', 'Wickets']
    
    # Use vectorized calculations
    opponent_summary = calculate_vectorized_bowling_stats(
        opponent_summary, 'Runs', 'Wickets', 'Balls'
    )
    
    # Calculate wickets per match
    opponent_summary['WPM'] = np.where(
        opponent_summary['Matches'] > 0,
        (opponent_summary['Wickets'] / opponent_summary['Matches']).round(2),
        0
    )
    
    return opponent_summary

def calculate_innings_stats(df):
    """
    Calculate bowling statistics by innings with vectorized operations.
    
    Args:
        df: Filtered bowling dataframe
        
    Returns:
        DataFrame with innings statistics
    """
    innings_summary = df.groupby(['Name', 'Innings']).agg({
        'File Name': 'nunique',
        'Bowler_Balls': 'sum',
        'Maidens': 'sum',
        'Bowler_Runs': 'sum',
        'Bowler_Wkts': 'sum'
    }).reset_index()
    
    innings_summary.columns = ['Name', 'Innings', 'Matches', 'Balls', 'M/D', 'Runs', 'Wickets']
    
    # Use vectorized calculations
    innings_summary = calculate_vectorized_bowling_stats(
        innings_summary, 'Runs', 'Wickets', 'Balls'
    )
    
    # Calculate wickets per match
    innings_summary['WPM'] = np.where(
        innings_summary['Matches'] > 0,
        (innings_summary['Wickets'] / innings_summary['Matches']).round(2),
        0
    )
    
    return innings_summary
