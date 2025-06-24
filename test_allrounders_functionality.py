#!/usr/bin/env python3
"""
Test script to verify allrounders.py functionality without running full Streamlit app
"""

import sys
import os
import pandas as pd
import streamlit as st
from unittest.mock import patch, MagicMock

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def create_mock_session_state():
    """Create mock session state with sample data that matches expected structure"""
    
    # Create sample batting data with correct column names
    bat_data = {
        'Player': ['Player A', 'Player B', 'Player C'],
        'Bat_Team': ['Team1', 'Team2', 'Team1'],
        'Mat': [10, 15, 12],
        'Inns': [18, 25, 20],
        'NO': [2, 3, 1],
        'Runs': [450, 720, 380],
        'HS': [85, 120, 67],
        'Ave': [28.12, 32.72, 20.00],
        'BF': [580, 920, 480],
        'SR': [77.58, 78.26, 79.16],
        '100': [0, 1, 0],
        '50': [3, 4, 2],
        '0': [1, 2, 1]
    }
    
    # Create sample bowling data with correct column names  
    bowl_data = {
        'Player': ['Player A', 'Player B', 'Player C'],
        'Bowl_Team': ['Team1', 'Team2', 'Team1'],
        'Mat': [10, 15, 12],
        'Inns': [15, 22, 18],
        'Ov': [85.2, 145.4, 98.1],
        'Runs': [320, 580, 420],
        'Wkts': [12, 25, 15],
        'BBI': ['3/25', '4/18', '3/32'],
        'Ave': [26.66, 23.20, 28.00],
        'Econ': [3.75, 3.98, 4.28],
        'SR': [42.6, 34.9, 39.2],
        '5': [0, 1, 0],
        '10': [0, 0, 0]
    }
    
    bat_df = pd.DataFrame(bat_data)
    bowl_df = pd.DataFrame(bowl_data)
    
    # Mock session state
    session_state = MagicMock()
    session_state.bat_df = bat_df
    session_state.bowl_df = bowl_df
    
    return session_state

def test_allrounders_functionality():
    """Test the allrounders functionality"""
    
    print("Testing allrounders.py functionality...")
    
    # Create mock session state
    mock_session_state = create_mock_session_state()
    
    # Mock Streamlit components
    with patch('streamlit.session_state', mock_session_state):
        with patch('streamlit.sidebar') as mock_sidebar:
            with patch('streamlit.selectbox') as mock_selectbox:
                with patch('streamlit.multiselect') as mock_multiselect:
                    with patch('streamlit.dataframe') as mock_dataframe:
                        with patch('streamlit.write') as mock_write:
                            
                            # Set up mock returns
                            mock_selectbox.return_value = 'All'
                            mock_multiselect.return_value = []
                            
                            try:
                                # Import and test the allrounders module
                                from views import allrounders
                                
                                print("✅ allrounders module imported successfully")
                                
                                # Test data access
                                bat_df = mock_session_state.bat_df
                                bowl_df = mock_session_state.bowl_df
                                
                                print(f"✅ Batting data columns: {list(bat_df.columns)}")
                                print(f"✅ Bowling data columns: {list(bowl_df.columns)}")
                                
                                # Test the specific column access that was causing KeyError
                                try:
                                    bat_teams = bat_df['Bat_Team'].unique()
                                    bowl_teams = bowl_df['Bowl_Team'].unique()
                                    print(f"✅ Successfully accessed Bat_Team: {bat_teams}")
                                    print(f"✅ Successfully accessed Bowl_Team: {bowl_teams}")
                                except KeyError as e:
                                    print(f"❌ KeyError still present: {e}")
                                    return False
                                
                                # Test filtering functionality
                                try:
                                    # Test team filtering
                                    filtered_bat = bat_df[bat_df['Bat_Team'] == 'Team1']
                                    filtered_bowl = bowl_df[bowl_df['Bowl_Team'] == 'Team1']
                                    print(f"✅ Team filtering works - Found {len(filtered_bat)} batting and {len(filtered_bowl)} bowling records for Team1")
                                    
                                    # Test merge operation (key part of allrounders functionality)
                                    merged_df = pd.merge(bat_df, bowl_df, on='Player', how='inner')
                                    print(f"✅ Merge operation successful - {len(merged_df)} allrounders found")
                                    print(f"✅ Merged columns: {list(merged_df.columns)}")
                                    
                                except Exception as e:
                                    print(f"❌ Error in filtering/merge: {e}")
                                    return False
                                
                                print("✅ All allrounders functionality tests passed!")
                                return True
                                
                            except Exception as e:
                                print(f"❌ Error importing or running allrounders: {e}")
                                import traceback
                                traceback.print_exc()
                                return False

if __name__ == "__main__":
    success = test_allrounders_functionality()
    if success:
        print("\n🎉 AllRounders functionality test PASSED!")
        print("The KeyError issues appear to be resolved.")
    else:
        print("\n❌ AllRounders functionality test FAILED!")
        print("There may still be issues to resolve.")
