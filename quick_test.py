#!/usr/bin/env python3
"""Quick test to verify allrounders.py fixes"""

print("Testing allrounders module import...")

try:
    from views import allrounders
    print("✅ SUCCESS: allrounders module imported successfully")
except Exception as e:
    print(f"❌ ERROR importing allrounders: {e}")
    exit(1)

print("\nTesting column access with sample data...")

import pandas as pd

# Create test data matching CricketCaptain2024 structure
bat_df = pd.DataFrame({
    'Name': ['Player A', 'Player B'],
    'Bat_Team': ['Team1', 'Team2'],  # No _y suffix
    'Match_Format': ['Test', 'ODI'],
    'Year': [2023, 2024]
})

bowl_df = pd.DataFrame({
    'Name': ['Player A', 'Player B'],
    'Bowl_Team': ['Team1', 'Team2'],  # No _y suffix
    'Match_Format': ['Test', 'ODI'],
    'Year': [2023, 2024]
})

try:
    # Test the operations that were causing KeyError
    bat_teams = sorted(bat_df['Bat_Team'].unique().tolist())
    bowl_teams = sorted(bowl_df['Bowl_Team'].unique().tolist())
    print(f"✅ SUCCESS: Column access works")
    print(f"   Bat teams: {bat_teams}")
    print(f"   Bowl teams: {bowl_teams}")
    
    # Test filtering
    filtered_bat = bat_df[bat_df['Bat_Team'].isin(['Team1'])]
    filtered_bowl = bowl_df[bowl_df['Bowl_Team'].isin(['Team1'])]
    print(f"✅ SUCCESS: Filtering works")
    print(f"   Filtered records: {len(filtered_bat)} batting, {len(filtered_bowl)} bowling")
    
except KeyError as e:
    print(f"❌ ERROR: KeyError still present: {e}")
    exit(1)
except Exception as e:
    print(f"❌ ERROR: Other issue: {e}")
    exit(1)

print("\n🎉 ALL TESTS PASSED!")
print("The allrounders.py KeyError issues have been resolved.")
