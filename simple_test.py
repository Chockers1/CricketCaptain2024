#!/usr/bin/env python3

import pandas as pd
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Import the data processing modules
    import bat
    import bowl
    
    print("Testing CricketCaptain2024 data structure...")
    
    # Try to load the data similar to how the app does it
    try:
        bat_data = bat.process_batting_data()
        if bat_data is not None and not bat_data.empty:
            print(f"Batting data loaded successfully. Shape: {bat_data.shape}")
            print("Batting data columns:")
            for col in sorted(bat_data.columns):
                print(f"  - {col}")
            
            # Check for the specific columns we're interested in
            if 'Bat_Team' in bat_data.columns:
                print("✓ 'Bat_Team' column found")
            else:
                print("✗ 'Bat_Team' column NOT found")
                
            if 'Bat_Team_y' in bat_data.columns:
                print("✓ 'Bat_Team_y' column found")
            else:
                print("✗ 'Bat_Team_y' column NOT found")
        else:
            print("No batting data returned")
    except Exception as e:
        print(f"Error loading batting data: {e}")
    
    print("\n" + "="*50 + "\n")
    
    try:
        bowl_data = bowl.process_bowling_data()
        if bowl_data is not None and not bowl_data.empty:
            print(f"Bowling data loaded successfully. Shape: {bowl_data.shape}")
            print("Bowling data columns:")
            for col in sorted(bowl_data.columns):
                print(f"  - {col}")
            
            # Check for the specific columns we're interested in
            if 'Bowl_Team' in bowl_data.columns:
                print("✓ 'Bowl_Team' column found")
            else:
                print("✗ 'Bowl_Team' column NOT found")
                
            if 'Bowl_Team_y' in bowl_data.columns:
                print("✓ 'Bowl_Team_y' column found")
            else:
                print("✗ 'Bowl_Team_y' column NOT found")
        else:
            print("No bowling data returned")
    except Exception as e:
        print(f"Error loading bowling data: {e}")

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure bat.py and bowl.py exist and are properly formatted")
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()
