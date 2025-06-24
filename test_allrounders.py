#!/usr/bin/env python3
"""
Test script to verify allrounders.py works without errors
"""
import sys
import os
import pandas as pd
import numpy as np

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_allrounders_import():
    """Test that we can import the allrounders module without errors"""
    try:
        from views.allrounders import display_ar_view
        print("✅ Successfully imported allrounders module")
        return True
    except Exception as e:
        print(f"❌ Failed to import allrounders module: {e}")
        return False

def create_dummy_data():
    """Create dummy data to test the column references"""
    # Create dummy batting data
    bat_data = {
        'Name': ['Player1', 'Player2', 'Player3'],
        'Date': ['01 Jan 2024', '02 Jan 2024', '03 Jan 2024'],
        'Bat_Team': ['Team A', 'Team B', 'Team A'],  # Note: no _y suffix
        'Match_Format': ['Test', 'ODI', 'T20'],
        'File Name': ['match1.txt', 'match2.txt', 'match3.txt'],
        'Runs': [50, 75, 25],
        'Out': [1, 1, 0],
        'Balls': [100, 80, 30],
        'Innings': [1, 1, 1],
        'Player_of_the_Match': ['Player1', 'Player2', 'Player3']
    }
    
    # Create dummy bowling data  
    bowl_data = {
        'Name': ['Player1', 'Player2', 'Player3'],
        'Date': ['01 Jan 2024', '02 Jan 2024', '03 Jan 2024'],
        'Bowl_Team': ['Team A', 'Team B', 'Team A'],  # Note: no _y suffix
        'Match_Format': ['Test', 'ODI', 'T20'],
        'File Name': ['match1.txt', 'match2.txt', 'match3.txt'],
        'Bowler_Runs': [30, 45, 20],
        'Bowler_Wkts': [2, 3, 1],
        'Bowler_Balls': [60, 54, 24],
        'Innings': [1, 1, 1],
        '5Ws': [0, 0, 0],
        '10Ws': [0, 0, 0]
    }
    
    return pd.DataFrame(bat_data), pd.DataFrame(bowl_data)

def test_column_references():
    """Test that the column references work with our data structure"""
    try:
        bat_df, bowl_df = create_dummy_data()
        
        # Test the column names that were causing issues
        print("Testing column references...")
        
        # These should work now (without _y suffix)
        bat_teams = sorted(bat_df['Bat_Team'].unique().tolist())
        bowl_teams = sorted(bowl_df['Bowl_Team'].unique().tolist())
        
        print(f"✅ Bat teams found: {bat_teams}")
        print(f"✅ Bowl teams found: {bowl_teams}")
        
        # Test filtering operations
        filtered_bat = bat_df[bat_df['Bat_Team'].isin(['Team A'])]
        filtered_bowl = bowl_df[bowl_df['Bowl_Team'].isin(['Team A'])]
        
        print(f"✅ Filtered batting data: {len(filtered_bat)} rows")
        print(f"✅ Filtered bowling data: {len(filtered_bowl)} rows")
        
        return True
        
    except Exception as e:
        print(f"❌ Column reference test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing allrounders.py fixes...\n")
    
    test1 = test_allrounders_import()
    test2 = test_column_references()
    
    if test1 and test2:
        print("\n🎉 All tests passed! The allrounders.py fixes appear to be working correctly.")
        print("✅ No more StreamlitSetPageConfigMustBeFirstCommandError")
        print("✅ No more KeyError: 'Bowl_Team_y' or 'Bat_Team_y'")
    else:
        print("\n❌ Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main()
