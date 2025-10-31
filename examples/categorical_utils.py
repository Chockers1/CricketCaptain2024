"""
Helper routines for working with categorical columns when migrating legacy
scorecard data. These utilities remain in the examples folder until a production
use-case emerges.
"""

import numpy as np
import pandas as pd


def safe_categorical_compare(series1, series2, operation="eq"):
    """Safely compare two pandas Series that may contain categorical values."""
    try:
        if operation == "eq":
            return series1 == series2
        if operation == "ne":
            return series1 != series2
        return getattr(series1, f"__{operation}__")(series2)
    except (TypeError, ValueError):
        str1 = series1.astype(str)
        str2 = series2.astype(str)
        if operation == "eq":
            return str1 == str2
        if operation == "ne":
            return str1 != str2
        return getattr(str1, f"__{operation}__")(str2)


def safe_team_comparison(df, team_col1, team_col2):
    """Return rows where the two team columns match."""
    return safe_categorical_compare(df[team_col1], df[team_col2], "eq")


def add_home_away_column_safe(df, home_team_col, away_team_col, bat_team_col, target_col="HomeOrAway"):
    """Add a derived home/away column without breaking on categorical data."""
    df[target_col] = "Neutral"
    home_mask = safe_team_comparison(df, home_team_col, bat_team_col)
    away_mask = safe_team_comparison(df, away_team_col, bat_team_col)
    df.loc[home_mask, target_col] = "Home"
    df.loc[away_mask, target_col] = "Away"
    return df


def convert_categorical_safely(series):
    """Cast a categorical Series to string without raising warnings."""
    if pd.api.types.is_categorical_dtype(series):
        return series.astype(str)
    return series


def safe_replace(series, to_replace, value):
    """Replace values while preserving categorical dtype when possible."""
    if pd.api.types.is_categorical_dtype(series):
        categories = series.cat.categories
        if value not in categories:
            categories = categories.append(pd.Index([value]))
            series = series.cat.set_categories(categories)
        return series.replace(to_replace, value)
    return series.replace(to_replace, value)


def ensure_categorical(series, categories=None):
    """Force a Series to categorical with optional ordering."""
    return pd.Categorical(series, categories=categories, ordered=True)


def safe_team_filter(df, team_col, team_name):
    """Filter a DataFrame by team name using safe comparisons."""
    return df[safe_categorical_compare(df[team_col], pd.Series([team_name] * len(df)), "eq")]


def add_home_away_indicator(df, home_col, away_col, team_name):
    """Add an indicator showing where the specified team played."""
    home_matches = safe_categorical_compare(df[home_col], pd.Series([team_name] * len(df)), "eq")
    away_matches = safe_categorical_compare(df[away_col], pd.Series([team_name] * len(df)), "eq")
    indicator = np.where(home_matches, "Home", np.where(away_matches, "Away", "Neutral"))
    return pd.Series(indicator, index=df.index)
