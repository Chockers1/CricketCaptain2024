#!/usr/bin/env python3

import sys
import pandas as pd
sys.path.append('.')

# Import the data loading modules directly
import bat
import match

print("Testing bat.py data loading...")
try:
    bat_df = bat.load_data()
    print(f"Batting data loaded: {bat_df.shape} rows")
    print(f"Batting columns: {list(bat_df.columns)}")
    if not bat_df.empty:
        print(f"Sample File Names: {bat_df['File Name'].head(3).tolist()}")
except Exception as e:
    print(f"Error loading batting data: {e}")

print("\nTesting match.py data loading...")
try:
    match_df = match.load_data()
    print(f"Match data loaded: {match_df.shape} rows")
    print(f"Match columns: {list(match_df.columns)}")
    if not match_df.empty:
        print(f"Sample File Names: {match_df['File Name'].head(3).tolist()}")
        
        # Check specific required columns
        required_columns = ['File Name', 'Home_Win', 'Away_Won', 'Home_Lost', 'Away_Lost', 'Home_Drawn', 'Away_Drawn', 'Tie']
        missing_columns = [col for col in required_columns if col not in match_df.columns]
        print(f"Missing required columns: {missing_columns}")
        
        # Check if any match results exist
        if not missing_columns:
            print("Match result summary:")
            for col in ['Home_Win', 'Away_Won', 'Home_Lost', 'Away_Lost', 'Home_Drawn', 'Away_Drawn', 'Tie']:
                if col in match_df.columns:
                    count = match_df[col].sum()
                    print(f"  {col}: {count}")
except Exception as e:
    print(f"Error loading match data: {e}")
    import traceback
    traceback.print_exc()
