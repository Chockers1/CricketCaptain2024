#!/usr/bin/env python3

import sys
import pandas as pd
sys.path.append('.')

# Import the data loading functions
from cricketcaptain import load_batting_data
from match import load_match_data

print("Loading batting data...")
bat_df = load_batting_data()
print(f"Batting data loaded: {bat_df.shape} rows")
print(f"Batting columns: {list(bat_df.columns)}")
print()

print("Loading match data...")
match_df = load_match_data()
print(f"Match data loaded: {match_df.shape} rows")
print(f"Match columns: {list(match_df.columns)}")
print()

if not match_df.empty:
    print("Sample match data:")
    print(match_df.head(3))
    print()
    
    # Check for unique file names in both datasets
    bat_files = set(bat_df['File Name'].unique()) if not bat_df.empty else set()
    match_files = set(match_df['File Name'].unique())
    
    print(f"Number of unique files in batting data: {len(bat_files)}")
    print(f"Number of unique files in match data: {len(match_files)}")
    print(f"Common files: {len(bat_files.intersection(match_files))}")
    
    if bat_files and match_files:
        print(f"Example batting files: {list(bat_files)[:3]}")
        print(f"Example match files: {list(match_files)[:3]}")
else:
    print("No match data available!")
