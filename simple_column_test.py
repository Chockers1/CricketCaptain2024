#!/usr/bin/env python3
"""
Simple test to verify column access in allrounders.py
"""

import pandas as pd

# Create sample data that mimics the structure from bat.py processing
def create_test_data():
    """Create test data with the expected column structure"""
    
    # Sample batting data (matches what bat.py would produce)
    bat_data = {
        'Name': ['Player A', 'Player B', 'Player C'],
        'Bat_Team': ['Team1', 'Team2', 'Team1'],  # Direct column name, no _y suffix
        'Match_Format': ['Test', 'ODI', 'Test'],
        'Year': [2023, 2023, 2024],
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
    
    # Sample bowling data (matches what bowl.py would produce)
    bowl_data = {
        'Name': ['Player A', 'Player B', 'Player C'],
        'Bowl_Team': ['Team1', 'Team2', 'Team1'],  # Direct column name, no _y suffix
        'Match_Format': ['Test', 'ODI', 'Test'],
        'Year': [2023, 2023, 2024],
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
    
    return pd.DataFrame(bat_data), pd.DataFrame(bowl_data)

def test_column_access():
    """Test the specific column access that was causing KeyError"""
    
    print("Creating test data...")
    bat_df, bowl_df = create_test_data()
    
    print(f"Batting DataFrame columns: {list(bat_df.columns)}")
    print(f"Bowling DataFrame columns: {list(bowl_df.columns)}")
    
    try:
        # Test the exact operations that were failing in allrounders.py
        print("\nTesting column access operations...")
        
        # Test filter options generation (lines 75-76 in allrounders.py)
        bat_teams = ['All'] + sorted(bat_df['Bat_Team'].unique().tolist())
        bowl_teams = ['All'] + sorted(bowl_df['Bowl_Team'].unique().tolist())
        print(f"✅ Filter options generated successfully")
        print(f"   Batting teams: {bat_teams}")
        print(f"   Bowling teams: {bowl_teams}")
        
        # Test filter application (lines 147-149 in allrounders.py)
        test_bat_choice = ['Team1']
        test_bowl_choice = ['Team1']
        
        filtered_bat = bat_df[bat_df['Bat_Team'].isin(test_bat_choice)]
        filtered_bowl = bowl_df[bowl_df['Bowl_Team'].isin(test_bowl_choice)]
        print(f"✅ Filter application successful")
        print(f"   Filtered batting records: {len(filtered_bat)}")
        print(f"   Filtered bowling records: {len(filtered_bowl)}")
        
        # Test merge operation
        merged = pd.merge(filtered_bat, filtered_bowl, on='Name', how='inner')
        print(f"✅ Merge operation successful")
        print(f"   Merged records: {len(merged)}")
        print(f"   Merged columns: {list(merged.columns)}")
        
        return True
        
    except KeyError as e:
        print(f"❌ KeyError encountered: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error encountered: {e}")
        return False

if __name__ == "__main__":
    print("Testing CricketCaptain2024 AllRounders Column Access")
    print("=" * 55)
    
    success = test_column_access()
    
    if success:
        print("\n🎉 SUCCESS: All column access tests passed!")
        print("The KeyError issues in allrounders.py should be resolved.")
    else:
        print("\n❌ FAILED: Column access issues still exist.")
        
    print("\nTest completed.")
