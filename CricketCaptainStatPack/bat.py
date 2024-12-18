import pandas as pd
import os
import re
import traceback

def process_bat_stats(directory_path, game_df, match_df):
    try:
        print("Starting process_bat_stats")
        print(f"Directory path: {directory_path}")
        print(f"Game DataFrame shape: {game_df.shape}")
        print(f"Game DataFrame columns: {game_df.columns}")
        print(f"Match DataFrame shape: {match_df.shape}")
        print(f"Match DataFrame columns: {match_df.columns}")

        # Initialize an empty list to hold DataFrames for each file
        dataframes = []

        # Loop through all files in the specified directory
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):  # Process only text files
                file_path = os.path.join(directory_path, filename)

                # Open and read the file
                with open(file_path, 'r', encoding='utf-8') as file:
                    file_content = file.read()

                # Split file content into lines
                file_lines = file_content.splitlines()

                # Initialize variables
                Row_No = 0
                Inns = 0
                Line_no = 0
                Bat_Team = ""
                Bowl_Team = ""
                innings_data = []  # A list to collect the rows of data for players
                
                # Initialize match data for innings
                for i, line in enumerate(file_lines):
                    Row_No += 1

                    # Extract Home_Team and Away_Team from Row 2 (assuming it's in "TeamA v TeamB" format)
                    if Row_No == 2:
                        teams = line.split(" v ")
                        if len(teams) == 2:
                            Home_Team = teams[0].strip()
                            Away_Team = teams[1].strip()

                    # Check if the line contains a separator "-------------"
                    if "-------------" in line:
                        Line_no += 1
                        if Line_no in [1, 5, 9, 13]:  # Increment innings based on separators
                            Inns += 1

                            # Determine Bat_Team and Bowl_Team
                            Bat_Team = line.split('-')[0].strip()  # Assuming the batting team is on the left of the separator
                            Bowl_Team = Away_Team if Bat_Team == Home_Team else Home_Team

                            # Initialize position and batsman data
                            position = 1

                            # Extract player statistics
                            for j in range(11):  # Loop through the maximum 11 players
                                if i + j + 1 >= len(file_lines):
                                    break  # Avoid index out of range

                                player_line = file_lines[i + j + 1]  # Getting lines for each player
                                if player_line.strip() == "":  # Check for empty lines
                                    continue  # Skip empty lines

                                # Adjust regex pattern to accurately capture the required fields
                                player_match = re.search(
                                    r"^(?P<Name>.+?)(?:\s+(?P<How_Out>(lbw|c|b|not out|run out|st|retired).+?))?\s+(?P<Runs>\d+)\s+(?P<Balls>\d+)\s+(?P<Fours>\d+|-)\s+(?P<Sixes>\d+|-)$",
                                    player_line
                                )
                                if player_match:
                                    Name = player_match.group('Name').replace('rtrd ht', '').strip()  # Remove 'rtrd ht' and trim spaces
                                    How_Out = player_match.group('How_Out').strip() if player_match.group('How_Out') else "Did not bat"  # How Out
                                    Runs = int(player_match.group('Runs'))  # Runs
                                    Balls = int(player_match.group('Balls'))  # Balls
                                    Fours = int(player_match.group('Fours')) if player_match.group('Fours') != '-' else 0  # 4s
                                    Sixes = int(player_match.group('Sixes')) if player_match.group('Sixes') != '-' else 0  # 6s

                                    # Append the player data to innings_data
                                    innings_data.append([filename, Inns, Bat_Team, Bowl_Team, position, Name, How_Out, Runs, Balls, Fours, Sixes, Home_Team, Away_Team])
                                    position += 1
                                else:
                                    # If the regex does not match, treat the player as "Did not bat"
                                    innings_data.append([filename, Inns, Bat_Team, Bowl_Team, position, player_line.strip(), "Did not bat", 0, 0, 0, 0, Home_Team, Away_Team])
                                    position += 1

                            # If fewer than 11 players, fill in empty rows for remaining players
                            for k in range(len(innings_data), 11):
                                innings_data.append([filename, Inns, Bat_Team, Bowl_Team, position, '', 'Did not bat', 0, 0, 0, 0, Home_Team, Away_Team])
                                position += 1

                # Convert to a DataFrame for easier manipulation
                df_innings = pd.DataFrame(innings_data, columns=['File Name', 'Innings', 'Bat Team', 'Bowl Team', 'Position', 'Name', 'How Out', 'Runs', 'Balls', '4s', '6s', 'Home Team', 'Away Team'])

                # Add the DataFrame to the list
                dataframes.append(df_innings)

        # Combine all DataFrames from the text files into a single DataFrame
        final_innings_df = pd.concat(dataframes, ignore_index=True)

        # Create bat_df from final_innings_df
        bat_df = final_innings_df

        # Step 1: No longer drop the 'Bat Team' and 'Bowl Team' columns from bat_df
        # Keep both columns in bat_df

        # Rename bat_df columns to match game_df
        bat_df = bat_df.rename(columns={
            'Bat Team': 'Bat_Team',
            'Bowl Team': 'Bowl_Team'
        })

        # Then merge as you had it
        merged_df = bat_df.merge(
            game_df[['File Name', 'Innings', 'Bat_Team', 'Bowl_Team', 'Total_Runs', 'Overs', 'Wickets', 
                    'Competition', 'Match_Format', 'Player_of_the_Match', 'Date']],
            on=['File Name', 'Innings'],
            how='left'
        )

        bat_df = merged_df

        # Add the specified columns
        bat_df['Year'] = bat_df['Date'].str[-4:]  # Extract the last four characters, which are the year
        # Convert to integer type
        bat_df['Year'] = bat_df['Year'].astype(int)

        # Optionally, if you want to display without commas when printing
        bat_df['Year'] = bat_df['Year'].apply(lambda x: f"{x:d}")        
        bat_df['Batted'] = bat_df['How Out'].apply(lambda x: 0 if x == 'Did not bat' else 1)
        bat_df['Out'] = bat_df['How Out'].apply(lambda x: 0 if x == 'Did not bat' or x == 'not out' else 1)
        bat_df['Not Out'] = bat_df['How Out'].apply(lambda x: 1 if x == 'not out' else 0)
        bat_df['DNB'] = bat_df['How Out'].apply(lambda x: 1 if x == 'Did not bat' else 0)
        

        bat_df['50s'] = bat_df['Runs'].apply(lambda x: 1 if 50 <= x < 100 else 0)
        bat_df['100s'] = bat_df['Runs'].apply(lambda x: 1 if 100 <= x < 200 else 0)
        bat_df['200s'] = bat_df['Runs'].apply(lambda x: 1 if x >= 200 else 0)
        bat_df['<25&Out'] = bat_df.apply(lambda row: 1 if row['Runs'] <= 25 and row['Out'] == 1 else 0, axis=1)

        bat_df['Caught'] = bat_df['How Out'].apply(lambda x: 1 if x.startswith('c ') else 0)
        bat_df['Bowled'] = bat_df['How Out'].apply(lambda x: 1 if x.startswith('b ') else 0)
        bat_df['LBW'] = bat_df['How Out'].apply(lambda x: 1 if x.startswith('lbw ') else 0)
        bat_df['Run Out'] = bat_df['How Out'].apply(lambda x: 1 if x.startswith('run') else 0)
        bat_df['Stumped'] = bat_df['How Out'].apply(lambda x: 1 if x.startswith('st') else 0)

        bat_df['Boundary Runs'] = (bat_df['4s'] * 4) + (bat_df['6s'] * 6)

        # Calculate Strike Rate
        bat_df['Strike Rate'] = (bat_df['Runs'] / bat_df['Balls'] * 100).round(2)

        # Calculate team balls based on format and innings state
        def calculate_team_balls(row):
            # If team is all out in any format, use sum of actual balls faced
            if row['Wickets'] == 10:
                # Group by File Name and Innings to sum actual balls faced in this innings
                return bat_df[
                    (bat_df['File Name'] == row['File Name']) & 
                    (bat_df['Innings'] == row['Innings'])
                ]['Balls'].sum()
            else:
                # Format-specific ball calculations for incomplete innings
                if row['Match_Format'] in ['The Hundred', '100 Ball Trophy']:
                    return 100  # Standard 100 balls
                elif row['Match_Format'] == 'T20':
                    return 120  # Standard 20 overs = 120 balls
                elif row['Match_Format'] == 'One Day':
                    return 300  # Standard 50 overs = 300 balls
                else:  # Test Match or First Class
                    # Handle decimal overs correctly
                    if pd.isna(row['Overs']):
                        return 0
                    try:
                        # Split overs into whole and partial
                        whole_overs = int(float(row['Overs']))
                        partial_balls = int((float(row['Overs']) % 1) * 10)  # Convert decimal part to balls
                        return (whole_overs * 6) + partial_balls
                    except (ValueError, TypeError):
                        # If conversion fails, calculate from actual balls faced
                        return bat_df[
                            (bat_df['File Name'] == row['File Name']) & 
                            (bat_df['Innings'] == row['Innings'])
                        ]['Balls'].sum()

        # Apply team balls calculation
        bat_df['Team Balls'] = bat_df.apply(calculate_team_balls, axis=1)

        # Handle divisions by zero
        bat_df['Strike Rate'] = bat_df.apply(lambda x: x['Strike Rate'] if x['Balls'] > 0 else 0, axis=1)

        return bat_df  # Return the modified DataFrame
    except Exception as e:
        print("An error occurred:", e)
        print(traceback.format_exc())


if __name__ == "__main__":
    directory_path = r"C:\Users\rtayl\AppData\Roaming\Childish Gambino\Desktop\My Files\Cricket Stats\Batting Stats"
    # game_df and match_df should be defined or loaded before calling this function
    game_df = pd.DataFrame()  # Replace with actual DataFrame loading code
    match_df = pd.DataFrame()  # Replace with actual DataFrame loading code

    bat_df = process_bat_stats(directory_path, game_df, match_df)

    if bat_df is not None:
        print(bat_df.head())  # Display the first few rows of the resulting DataFrame
