#!/usr/bin/env python3
"""
Debug script to check column names in bat_df and bowl_df.
This script assumes the data is loaded in session state.
"""

import sys
import os

# Add the project directory to Python path
project_dir = r"c:\Users\rtayl\OneDrive\Rob Documents\Python Scripts\CricketCaptain2024"
sys.path.insert(0, project_dir)

def debug_columns():
    """Print available columns in the dataframes"""
    try:
        import streamlit as st
        import pandas as pd
        
        print("=== DEBUGGING COLUMN NAMES ===")
        
        # Check if session state has the required dataframes
        if hasattr(st, 'session_state'):
            if 'bat_df' in st.session_state:
                bat_df = st.session_state['bat_df']
                print(f"\nBAT_DF COLUMNS ({len(bat_df.columns)} total):")
                for i, col in enumerate(sorted(bat_df.columns)):
                    print(f"  {i+1:2d}. '{col}'")
                    
                print(f"\nBAT_DF SHAPE: {bat_df.shape}")
                
                # Check for team-related columns specifically
                team_cols = [col for col in bat_df.columns if 'team' in col.lower() or 'bat' in col.lower() or 'bowl' in col.lower()]
                print(f"\nTEAM-RELATED COLUMNS IN BAT_DF:")
                for col in team_cols:
                    print(f"  - '{col}'")
            else:
                print("❌ bat_df not found in session_state")
                
            if 'bowl_df' in st.session_state:
                bowl_df = st.session_state['bowl_df']
                print(f"\nBOWL_DF COLUMNS ({len(bowl_df.columns)} total):")
                for i, col in enumerate(sorted(bowl_df.columns)):
                    print(f"  {i+1:2d}. '{col}'")
                    
                print(f"\nBOWL_DF SHAPE: {bowl_df.shape}")
                
                # Check for team-related columns specifically
                team_cols = [col for col in bowl_df.columns if 'team' in col.lower() or 'bat' in col.lower() or 'bowl' in col.lower()]
                print(f"\nTEAM-RELATED COLUMNS IN BOWL_DF:")
                for col in team_cols:
                    print(f"  - '{col}'")
            else:
                print("❌ bowl_df not found in session_state")
        else:
            print("❌ Streamlit session_state not available")
    
    except Exception as e:
        print(f"❌ Error in debug_columns: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_columns()
