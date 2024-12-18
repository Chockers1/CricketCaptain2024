import pandas as pd
import os
import re
import traceback

def process_bowl_stats(directory_path, game_df, match_df):
    try:
        print("Starting bowl stats processing")
        print(f"Directory path: {directory_path}")
        print(f"game_df shape: {game_df.shape}")
        print(f"game_df columns: {game_df.columns}")
        print(f"match_df shape: {match_df.shape}")
        print(f"match_df columns: {match_df.columns}")

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
                        if Line_no in [3, 7, 11, 15]:  # Increment innings based on separators
                            Inns += 1

                            # Determine Bat_Team and Bowl_Team
                            Bat_Team = line.split('-')[0].strip()  # Assuming the batting team is on the left of the separator
                            Bowl_Team = Away_Team if Bat_Team == Home_Team else Home_Team

                            # Initialize position and bowler data
                            position = 1

                            # Extract bowler statistics
                            for j in range(11):  # Loop through the maximum 11 players
                                if i + j + 1 >= len(file_lines):
                                    break  # Avoid index out of range

                                player_line = file_lines[i + j + 1]  # Getting lines for each player
                                if player_line.strip() == "":  # Check for empty lines
                                    continue  # Skip empty lines

                                # Adjust regex pattern to accurately capture the bowler statistics
                                bowler_match = re.search(
                                    r"^(?P<Name>.+?)\s+(?P<Overs>\d+(\.\d+)?)\s+(?P<Maidens>\d+)\s+(?P<Runs>\d+)\s+(?P<Wickets>\d+)\s+(?P<Econ>\d+(\.\d+)?)$",
                                    player_line
                                )
                                if bowler_match:
                                    Name = bowler_match.group('Name').strip()  # Bowler's Name
                                    Overs = float(bowler_match.group('Overs'))  # Overs
                                    Maidens = int(bowler_match.group('Maidens'))  # Maidens
                                    Runs = int(bowler_match.group('Runs'))  # Runs
                                    Wickets = int(bowler_match.group('Wickets'))  # Wickets
                                    Econ = float(bowler_match.group('Econ'))  # Economy rate

                                    # Append the bowler data to innings_data
                                    innings_data.append([filename, Inns, Bat_Team, Bowl_Team, position, Name, Overs, Maidens, Runs, Wickets, Econ])
                                    position += 1
                                else:
                                    # Handle cases where the regex doesn't match
                                    continue

                            # If fewer than 11 players, fill in empty rows for remaining players
                            for k in range(len(innings_data), 11):
                                innings_data.append([filename, Inns, Bat_Team, Bowl_Team, position, '', 0, 0, 0, 0, 0])
                                position += 1

                # Convert to a DataFrame for easier manipulation
                df_innings = pd.DataFrame(innings_data, columns=['File Name', 'Innings', 'Bat Team', 'Bowl Team', 'Position', 'Name', 'Bowler_Overs', 'Maidens', 'Bowler_Runs', 'Bowler_Wkts', 'Bowler_Econ'])

                # Add the DataFrame to the list
                dataframes.append(df_innings)

        # Combine all DataFrames from the text files into a single DataFrame
        final_innings_df = pd.concat(dataframes, ignore_index=True)

        # Create bowl_df from final_innings_df
        bowl_df = final_innings_df

        # Step 1: Merge with match_df to get Home_Team and Away_Team
        bowl_df = bowl_df.merge(
            match_df[['File Name', 'Home_Team', 'Away_Team']],
            on='File Name',
            how='left'
        )

        # Step 2: Drop the 'Bat Team' and 'Bowl Team' columns from bowl_df
        bowl_df_dropped = bowl_df.drop(columns=['Bat Team', 'Bowl Team'])

        # Step 3: Merge bowl_df_dropped with game_df based on 'File Name' and 'Innings'
        merged_df = bowl_df_dropped.merge(
            game_df[['File Name', 'Innings', 'Bat_Team', 'Bowl_Team', 'Total_Runs', 'Overs', 'Wickets', 'Competition', 'Match_Format', 'Player_of_the_Match', 'Date']],
            on=['File Name', 'Innings'],
            how='left'  # Use 'left' to keep all entries from bowl_df_dropped
        )

        bowl_df = merged_df

        # Add the specified columns
        bowl_df['Bowled'] = 1
        bowl_df['5Ws'] = bowl_df['Bowler_Wkts'].apply(lambda x: 1 if 5 <= x < 10 else 0)
        bowl_df['10Ws'] = bowl_df['Bowler_Wkts'].apply(lambda x: 1 if x >= 10 else 0)

        # Calculate Balls from Overs
        bowl_df['Bowler_Balls'] = bowl_df['Bowler_Overs'].apply(lambda x: int(x) * 6 + round((x - int(x)) * 10))

        # Remove rows where the 'Name' column is blank
        bowl_df = bowl_df[bowl_df['Name'].str.strip() != '']

        # Calculate additional statistics
        bowl_df['Runs_Per_Over'] = bowl_df['Bowler_Runs'] / bowl_df['Bowler_Overs']
        bowl_df['Balls_Per_Wicket'] = bowl_df['Bowler_Balls'] / bowl_df['Bowler_Wkts'].replace(0, 1)  # Avoid division by zero
        bowl_df['Dot_Ball_Percentage'] = (bowl_df['Maidens'] * 6) / bowl_df['Bowler_Balls'] * 100

        # Calculate Strike Rate (balls per wicket)
        bowl_df['Strike_Rate'] = bowl_df['Bowler_Balls'] / bowl_df['Bowler_Wkts'].replace(0, 1)  # Avoid division by zero

        # Calculate Average (runs per wicket)
        bowl_df['Average'] = bowl_df['Bowler_Runs'] / bowl_df['Bowler_Wkts'].replace(0, 1)  # Avoid division by zero

        print("Bowl stats processing completed successfully")
        print(f"Final bowl_df shape: {bowl_df.shape}")
        print(f"Final bowl_df columns: {bowl_df.columns}")

        return bowl_df

    except Exception as e:
        print(f"Error in bowl stats processing: {str(e)}")
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    directory_path = r"C:\Users\rtayl\AppData\Roaming\Childish Things\Cricket Captain 2024\Saves\Scorecards"
    
    # Load game_df and match_df from CSV files
    game_csv_path = os.path.join(directory_path, "game_data.csv")
    match_csv_path = os.path.join(directory_path, "match_data.csv")
    
    if os.path.exists(game_csv_path) and os.path.exists(match_csv_path):
        game_df = pd.read_csv(game_csv_path)
        match_df = pd.read_csv(match_csv_path)
        print("Loaded game and match data from CSV files")
    else:
        print("game_data.csv or match_data.csv not found. Please run match.py and game.py first.")
        exit()

    bowl_df = process_bowl_stats(directory_path, game_df, match_df)
    if bowl_df is not None:
        print(bowl_df.head(50))
        
        # Save bowl_df to CSV
        bowl_csv_path = os.path.join(directory_path, "bowl_data.csv")
        bowl_df.to_csv(bowl_csv_path, index=False)
        print(f"Bowl data saved to {bowl_csv_path}")
    else:
        print("Failed to process bowl stats")

# Add these lines at the end of the file
directory_path = r"C:\Users\rtayl\AppData\Roaming\Childish Things\Cricket Captain 2024\Saves\Scorecards"
game_csv_path = os.path.join(directory_path, "game_data.csv")
match_csv_path = os.path.join(directory_path, "match_data.csv")

if os.path.exists(game_csv_path) and os.path.exists(match_csv_path):
    game_df = pd.read_csv(game_csv_path)
    match_df = pd.read_csv(match_csv_path)
    bowl_df = process_bowl_stats(directory_path, game_df, match_df)
else:
    print("game_data.csv or match_data.csv not found. Please run match.py and game.py first.")
    bowl_df = None
    
