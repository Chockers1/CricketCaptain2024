import pandas as pd
import os
import re
import traceback

def process_bat_stats(directory_path, game_df, match_df):
    """
    Process batting statistics from text files and merge with game data.
    
    Args:
        directory_path: Path to directory containing batting statistics text files
        game_df: DataFrame containing game-level information
        match_df: DataFrame containing match-level information
    
    Returns:
        DataFrame containing processed batting statistics
    """
    try:
        # Log the start of processing and input data dimensions
        print("Starting process_bat_stats")
        print(f"Directory path: {directory_path}")
        print(f"Game DataFrame shape: {game_df.shape}")
        print(f"Game DataFrame columns: {game_df.columns}")
        print(f"Match DataFrame shape: {match_df.shape}")
        print(f"Match DataFrame columns: {match_df.columns}")

        # Initialize an empty list to hold DataFrames for each file
        # Each file will become a DataFrame, and they'll all be combined at the end
        dataframes = []

        # Loop through all files in the specified directory
        for filename in os.listdir(directory_path):
            # Process only text files - these contain the match batting data
            if filename.endswith('.txt'):
                file_path = os.path.join(directory_path, filename)

                # Open and read the file with UTF-8 encoding to handle special characters
                with open(file_path, 'r', encoding='utf-8') as file:
                    file_content = file.read()

                # Split file content into separate lines for processing
                file_lines = file_content.splitlines()

                # Initialize variables for tracking position in the file
                Row_No = 0           # Current row number being processed
                Inns = 0             # Current innings (1-4 for Test matches, 1-2 for limited overs)
                Line_no = 0          # Track separator lines to determine innings
                Bat_Team = ""        # Team currently batting
                Bowl_Team = ""       # Team currently bowling
                innings_data = []    # List to collect player data rows
                
                # Process the file line by line
                for i, line in enumerate(file_lines):
                    Row_No += 1

                    # Extract Home_Team and Away_Team from Row 2 (format: "TeamA v TeamB")
                    if Row_No == 2:
                        teams = line.split(" v ")
                        if len(teams) == 2:
                            Home_Team = teams[0].strip()
                            Away_Team = teams[1].strip()

                    # Check for innings separator lines (format: "TeamName -----------")
                    if "-------------" in line:
                        Line_no += 1
                        # Lines 1, 5, 9, 13 indicate the start of innings 1, 2, 3, 4 respectively
                        if Line_no in [1, 5, 9, 13]:
                            Inns += 1  # Increment innings counter

                            # Extract batting team name from left side of separator
                            Bat_Team = line.split('-')[0].strip()
                            # Determine bowling team (if not batting, must be bowling)
                            Bowl_Team = Away_Team if Bat_Team == Home_Team else Home_Team

                            # Initialize batting position counter for this innings
                            position = 1

                            # Process up to 11 players (maximum in a cricket team)
                            for j in range(11):
                                # Avoid index errors at end of file
                                if i + j + 1 >= len(file_lines):
                                    break

                                # Get the line containing player data
                                player_line = file_lines[i + j + 1]
                                if player_line.strip() == "":  # Skip empty lines
                                    continue

                                # Use regex to extract player stats from the line
                                # Format: Name How_Out Runs Balls 4s 6s
                                player_match = re.search(
                                    r"^(?P<Name>.+?)(?:\s+(?P<How_Out>(lbw|c|b|not out|run out|st|retired).+?))?\s+(?P<Runs>\d+)\s+(?P<Balls>\d+)\s+(?P<Fours>\d+|-)\s+(?P<Sixes>\d+|-)$",
                                    player_line
                                )
                                
                                if player_match:
                                    # Extract and clean player data from regex match
                                    Name = player_match.group('Name').replace('rtrd ht', '').strip()
                                    How_Out = player_match.group('How_Out').strip() if player_match.group('How_Out') else "Did not bat"
                                    Runs = int(player_match.group('Runs'))
                                    Balls = int(player_match.group('Balls'))
                                    # Handle '-' notation for no boundaries
                                    Fours = int(player_match.group('Fours')) if player_match.group('Fours') != '-' else 0
                                    Sixes = int(player_match.group('Sixes')) if player_match.group('Sixes') != '-' else 0

                                    # Add player data to innings_data list
                                    innings_data.append([
                                        filename, Inns, Bat_Team, Bowl_Team, position, 
                                        Name, How_Out, Runs, Balls, Fours, Sixes, 
                                        Home_Team, Away_Team
                                    ])
                                    position += 1  # Increment batting position
                                else:
                                    # If regex doesn't match, treat as "Did not bat"
                                    innings_data.append([
                                        filename, Inns, Bat_Team, Bowl_Team, position,
                                        player_line.strip(), "Did not bat", 0, 0, 0, 0,
                                        Home_Team, Away_Team
                                    ])
                                    position += 1

                            # Fill remaining positions up to 11 with empty "Did not bat" entries
                            # This ensures consistent team size across all innings
                            while position <= 11:
                                innings_data.append([
                                    filename, Inns, Bat_Team, Bowl_Team, position,
                                    '', 'Did not bat', 0, 0, 0, 0,
                                    Home_Team, Away_Team
                                ])
                                position += 1

                # Convert innings data to DataFrame with appropriate column names
                df_innings = pd.DataFrame(
                    innings_data, 
                    columns=[
                        'File Name', 'Innings', 'Bat Team', 'Bowl Team', 'Position', 
                        'Name', 'How Out', 'Runs', 'Balls', '4s', '6s', 
                        'Home Team', 'Away Team'
                    ]
                )

                # Add this file's DataFrame to our list
                dataframes.append(df_innings)

        # Combine all individual file DataFrames into one master DataFrame
        final_innings_df = pd.concat(dataframes, ignore_index=True)

        # Create batting DataFrame from the final innings data
        bat_df = final_innings_df

        # Rename columns to match game_df naming convention (standardize column names)
        bat_df = bat_df.rename(columns={
            'Bat Team': 'Bat_Team',
            'Bowl Team': 'Bowl_Team'
        })

        # Merge batting data with game data to get additional match information
        # This adds match details like competition, format, player of the match, etc.
        merged_df = bat_df.merge(
            game_df[['File Name', 'Innings', 'Bat_Team', 'Bowl_Team', 'Total_Runs', 'Overs', 'Wickets', 
                    'Competition', 'Match_Format', 'Player_of_the_Match', 'Date']],
            on=['File Name', 'Innings'],
            how='left'  # Left join to keep all batting records
        )

        bat_df = merged_df

        # Extract year from date for easier year-based filtering
        bat_df['Year'] = bat_df['Date'].str[-4:]  # Extract last 4 characters (year)
        bat_df['Year'] = bat_df['Year'].astype(int)  # Convert to integer
        bat_df['Year'] = bat_df['Year'].apply(lambda x: f"{x:d}")  # Format without commas
        
        # Add derived boolean columns for different batting statistics
        # These will be useful for aggregations and filtering
        bat_df['Batted'] = bat_df['How Out'].apply(lambda x: 0 if x == 'Did not bat' else 1)
        bat_df['Out'] = bat_df['How Out'].apply(lambda x: 0 if x == 'Did not bat' or x == 'not out' else 1)
        bat_df['Not Out'] = bat_df['How Out'].apply(lambda x: 1 if x == 'not out' else 0)
        bat_df['DNB'] = bat_df['How Out'].apply(lambda x: 1 if x == 'Did not bat' else 0)
        
        # Add milestone innings indicators
        bat_df['50s'] = bat_df['Runs'].apply(lambda x: 1 if 50 <= x < 100 else 0)
        bat_df['100s'] = bat_df['Runs'].apply(lambda x: 1 if 100 <= x < 200 else 0)
        bat_df['200s'] = bat_df['Runs'].apply(lambda x: 1 if x >= 200 else 0)
        bat_df['<25&Out'] = bat_df.apply(lambda row: 1 if row['Runs'] <= 25 and row['Out'] == 1 else 0, axis=1)

        # Add dismissal type indicators
        bat_df['Caught'] = bat_df['How Out'].apply(lambda x: 1 if x.startswith('c ') else 0)
        bat_df['Bowled'] = bat_df['How Out'].apply(lambda x: 1 if x.startswith('b ') else 0)
        bat_df['LBW'] = bat_df['How Out'].apply(lambda x: 1 if x.startswith('lbw ') else 0)
        bat_df['Run Out'] = bat_df['How Out'].apply(lambda x: 1 if x.startswith('run') else 0)
        bat_df['Stumped'] = bat_df['How Out'].apply(lambda x: 1 if x.startswith('st') else 0)

        # Calculate boundary runs (4s and 6s)
        bat_df['Boundary Runs'] = (bat_df['4s'] * 4) + (bat_df['6s'] * 6)

        # Calculate strike rate (runs per 100 balls)
        bat_df['Strike Rate'] = (bat_df['Runs'] / bat_df['Balls'] * 100).round(2)

        # Function to calculate total team balls based on format and innings state
        def calculate_team_balls(row):
            # If team is all out, use actual balls faced (sum of batsmen's balls)
            if row['Wickets'] == 10:
                return bat_df[
                    (bat_df['File Name'] == row['File Name']) & 
                    (bat_df['Innings'] == row['Innings'])
                ]['Balls'].sum()
            else:
                # For incomplete innings, use format-specific ball counts
                if row['Match_Format'] in ['The Hundred', '100 Ball Trophy']:
                    return 100  # 100-ball format
                elif row['Match_Format'] == 'T20':
                    return 120  # 20 overs = 120 balls
                elif row['Match_Format'] == 'One Day':
                    return 300  # 50 overs = 300 balls
                else:  # Test Match or First Class
                    # Handle decimal overs (e.g., 90.3 overs = 90*6 + 3 = 543 balls)
                    if pd.isna(row['Overs']):
                        return 0
                    try:
                        # Split overs into whole and partial
                        whole_overs = int(float(row['Overs']))
                        partial_balls = int((float(row['Overs']) % 1) * 10)
                        return (whole_overs * 6) + partial_balls
                    except (ValueError, TypeError):
                        # Fallback: calculate from actual balls faced
                        return bat_df[
                            (bat_df['File Name'] == row['File Name']) & 
                            (bat_df['Innings'] == row['Innings'])
                        ]['Balls'].sum()

        # Apply team balls calculation to each row
        bat_df['Team Balls'] = bat_df.apply(calculate_team_balls, axis=1)

        # Handle division by zero in strike rate calculation
        bat_df['Strike Rate'] = bat_df.apply(lambda x: x['Strike Rate'] if x['Balls'] > 0 else 0, axis=1)

        # Transform competition names to standardized format
        def transform_competition(row):
            # Define mappings for Test trophy names based on participating teams
            test_trophies = {
                ('Australia', 'England'): 'The Ashes',
                ('England', 'Australia'): 'The Ashes',
                ('India', 'Australia'): 'Border-Gavaskar Trophy',
                ('Australia', 'India'): 'Border-Gavaskar Trophy',
                ('West Indies', 'Australia'): 'Frank Worrell Trophy',
                ('Australia', 'West Indies'): 'Frank Worrell Trophy',
                ('South Africa', 'England'): "Basil D'Oliveira Trophy",
                ('England', 'South Africa'): "Basil D'Oliveira Trophy",
                ('England', 'India'): 'Pataudi Trophy',
                ('India', 'England'): 'Anthony de Mello Trophy',
                ('West Indies', 'Sri Lanka'): 'Sobers-Tissera Trophy',
                ('Sri Lanka', 'West Indies'): 'Sobers-Tissera Trophy',
                ('England', 'West Indies'): 'Wisden Trophy',
                ('West Indies', 'England'): 'Wisden Trophy',
                ('Australia', 'New Zealand'): 'Trans-Tasman Trophy',
                ('New Zealand', 'Australia'): 'Trans-Tasman Trophy',
                ('Australia', 'Sri Lanka'): 'Warne-Muralitharan Trophy',
                ('Sri Lanka', 'Australia'): 'Warne-Muralitharan Trophy',
                ('India', 'South Africa'): 'Freedom Trophy',
                ('South Africa', 'India'): 'Freedom Trophy'
            }

            comp = row['Competition']
            # Apply different transformations based on competition name patterns
            if 'Test Match' in comp:
                # Use specific trophy names for Test matches between certain teams
                team_pair = (row['Home Team'], row['Away Team'])
                if team_pair in test_trophies:
                    return test_trophies[team_pair]
                return 'Test Match'
            elif comp.startswith('World Cup 20'):
                return 'T20 World Cup'
            elif comp.startswith('World Cup'):
                return 'ODI World Cup'
            elif comp.startswith('Champions Cup'):
                return 'Champions Cup'
            elif comp.startswith('Asia Trophy ODI'):
                return 'ODI Asia Cup'
            elif comp.startswith('Asia Trophy 20'):
                return 'T20 Asia Cup'
            elif 'One Day International' in comp:
                return 'ODI'
            elif '20 Over International' in comp:
                return 'T20I'
            elif 'Australian League' in comp:
                return 'Sheffield Shield'
            elif 'English FC League - D2' in comp:
                return 'County Championship Division 2'
            elif 'English FC League - D1' in comp:
                return 'County Championship Division 1'    
            elif 'Challenge Trophy' in comp:
                return 'Royal London Cup'         
            else:
                # Default: use original competition name
                return comp

        # Apply competition name transformation
        try:
            bat_df['comp'] = bat_df.apply(transform_competition, axis=1)
        except Exception as e:
            # Log error and fall back to original competition name
            print(f"Error creating comp column: {e}")
            bat_df['comp'] = bat_df['Competition']

        # Verify comp column exists before returning
        if 'comp' not in bat_df.columns:
            print("Warning: comp column not created, using Competition instead")
            bat_df['comp'] = bat_df['Competition']

        # Return the fully processed batting statistics DataFrame
        return bat_df
        
    except Exception as e:
        # Catch and log any errors that occur during processing
        print("An error occurred:", e)
        print(traceback.format_exc())  # Print full traceback for debugging


# Code that runs when this script is executed directly (not imported)
if __name__ == "__main__":
    # Define directory path for batting statistics files
    directory_path = r"C:\Users\rtayl\AppData\Roaming\Childish Gambino\Desktop\My Files\Cricket Stats\Batting Stats"
    
    # In standalone mode, we'd need to provide game_df and match_df
    # Here they're empty placeholders - in actual use they would be loaded from files
    game_df = pd.DataFrame()  # Replace with actual DataFrame loading code
    match_df = pd.DataFrame()  # Replace with actual DataFrame loading code

    # Process batting stats
    bat_df = process_bat_stats(directory_path, game_df, match_df)

    # Display the first few rows of the result if processing was successful
    if bat_df is not None:
        print(bat_df.head())