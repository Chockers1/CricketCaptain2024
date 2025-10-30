"""
Utility functions to handle categorical column comparisons safely
"""

import pandas as pd
import numpy as np

def safe_categorical_compare(series1, series2, operation='eq'):
    """
    Safely compare two pandas Series that may contain categorical data
    
    Args:
        series1, series2: pandas Series to compare
        operation: 'eq' for equality, 'ne' for not equal, etc.
    
    Returns:
        Boolean Series with comparison results
    """
    try:
        # Try direct comparison first
        if operation == 'eq':
            return series1 == series2
        elif operation == 'ne':
            return series1 != series2
        else:
            return getattr(series1, f'__{operation}__')(series2)
    except (TypeError, ValueError):
        # If categorical comparison fails, convert to string and compare
        str1 = series1.astype(str)
        str2 = series2.astype(str)
        
        if operation == 'eq':
            return str1 == str2
        elif operation == 'ne':
            return str1 != str2
        else:
            return getattr(str1, f'__{operation}__')(str2)

def safe_team_comparison(df, team_col1, team_col2):
    """
    Safely compare two team columns that may be categorical
    
    Args:
        df: DataFrame containing the columns
        team_col1, team_col2: column names to compare
    
    Returns:
        Boolean Series indicating where teams match
    """
    return safe_categorical_compare(df[team_col1], df[team_col2], 'eq')

def add_home_away_column_safe(df, home_team_col, away_team_col, bat_team_col, target_col='HomeOrAway'):
    """
    Safely add HomeOrAway column handling categorical comparisons
    
    Args:
        df: DataFrame to modify
        home_team_col: name of home team column
        away_team_col: name of away team column  
        bat_team_col: name of batting team column
        target_col: name of column to create (default 'HomeOrAway')
    """
    # Initialize with neutral
    df[target_col] = 'Neutral'
    
    # Safe comparisons
    home_mask = safe_team_comparison(df, home_team_col, bat_team_col)
    away_mask = safe_team_comparison(df, away_team_col, bat_team_col)
    
    # Set values
    df.loc[home_mask, target_col] = 'Home'
    df.loc[away_mask, target_col] = 'Away'
    
    return df

def convert_categorical_safely(series):
    """
    Convert a series to string safely, handling both categorical and regular series
    """
    if pd.api.types.is_categorical_dtype(series):
        return series.astype(str)
    return series.astype(str)

# Example usage functions for common patterns
def safe_filter_by_team(df, team_col, team_name):
    """Safely filter DataFrame by team name"""
    return df[safe_categorical_compare(df[team_col], pd.Series([team_name] * len(df)), 'eq')]

def get_team_matches_safe(df, home_col, away_col, team_name):
    """Get all matches involving a specific team"""
    home_matches = safe_categorical_compare(df[home_col], pd.Series([team_name] * len(df)), 'eq')
    away_matches = safe_categorical_compare(df[away_col], pd.Series([team_name] * len(df)), 'eq')
    return df[home_matches | away_matches]