import pandas as pd
import os
import re
import numpy as np
import traceback

def process_game_stats(directory_path, match_df):
    try:
        print("Starting process_game_stats")
        print(f"Directory path: {directory_path}")
        print(f"Match DataFrame shape: {match_df.shape}")
        print(f"Match DataFrame columns: {match_df.columns}")

        # Initialize an empty list to hold DataFrames for each file
        dataframes = []

        # Loop through all files in the specified directory
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):  # Process only text files
                file_path = os.path.join(directory_path, filename)
                print(f"Processing file: {filename}")

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
                Home_Team = ""
                Away_Team = ""
                M_Arr = []  # A list to collect the rows of data

                # Initialize match data
                for i, line in enumerate(file_lines):
                    Row_No += 1

                    # Extract Home_Team and Away_Team from Row 2 (assuming it's in "TeamA v TeamB" format)
                    if Row_No == 2:
                        teams = line.split(" v ")
                        if len(teams) == 2:
                            Home_Team = teams[0].strip()
                            Away_Team = teams[1].strip()
                        print(f"Home Team: {Home_Team}, Away Team: {Away_Team}")

                    # Extract Bat_Team if the row contains "Innings"
                    if "Innings" in line:
                        Bat_Team = line.split(" - ")[0]
                        print(f"Batting Team: {Bat_Team}")

                    # Check if the line contains a separator "-------------"
                    if "-------------" in line:
                        Line_no += 1
                        if Line_no in [1, 5, 9, 13]:  # Increment innings based on separators
                            Inns += 1
                            print(f"Processing Innings {Inns}")

                            # Determine Bowl_Team (the opposite of Bat_Team)
                            Bowl_Team = Away_Team if Bat_Team == Home_Team else Home_Team

                            # Initialize match data
                            match_data = [filename, Home_Team, Away_Team, Bat_Team, Bowl_Team, Inns]

                            # Initialize total runs, overs, and wickets
                            Runs = 0
                            Overs = None
                            Wickets = None
                            Bowled_Balls = 0  # Initialize Bowled_Balls

                            # Extract total runs, overs, and wickets from the following lines
                            for j, total_line in enumerate(file_lines[i + 1:]):
                                print(f"Analyzing line: {total_line}")
                                # Check for the all-out scenario
                                all_out_match = re.search(r"TOTAL: \(all out, (\d+\.\d+|\d+) overs\)\s+(\d+)", total_line)
                                if all_out_match:
                                    Runs = int(all_out_match.group(2))  # Total runs
                                    Overs = float(all_out_match.group(1))  # Overs
                                    Wickets = 10  # Set to 10 for all out
                                    print(f"All out: Runs={Runs}, Overs={Overs}, Wickets={Wickets}")
                                    break
                                    
                                # Check for wickets scenario with overs
                                wickets_match = re.search(r"TOTAL: \((\d+) wkts, (\d+\.\d+|\d+) overs\)", total_line)
                                if wickets_match:
                                    Runs = int(total_line.split()[-1])  # Extract runs from the end of the line
                                    Wickets = int(wickets_match.group(1))  # Number of wickets
                                    Overs = float(wickets_match.group(2))  # Overs
                                    print(f"Wickets with overs: Runs={Runs}, Overs={Overs}, Wickets={Wickets}")
                                    break
                                    
                                # Check for wickets scenario with balls
                                balls_match = re.search(r"TOTAL: \((\d+) wkts, (\d+) balls\)", total_line)
                                if balls_match:
                                    Runs = int(total_line.split()[-1])  # Extract runs from the end of the line
                                    Wickets = int(balls_match.group(1))  # Number of wickets
                                    Bowled_Balls = int(balls_match.group(2))  # Balls
                                    Overs = Bowled_Balls / 5  # Convert balls to overs using 5-ball overs
                                    print(f"Wickets with balls: Runs={Runs}, Overs={Overs}, Wickets={Wickets}, Bowled_Balls={Bowled_Balls}")
                                    break

                            # If Wickets is still None, it means no wickets found, set to 0
                            if Wickets is None:
                                Wickets = 0
                                print("No wickets data found, setting Wickets to 0")

                            # Append extracted data
                            match_data += [Runs, Overs, Wickets, Bowled_Balls]
                            M_Arr.append(match_data)
                            print(f"Appended match data: {match_data}")

                # Convert to a DataFrame for easier manipulation
                df = pd.DataFrame(M_Arr, columns=['File Name', 'Home_Team', 'Away_Team', 'Bat_Team', 'Bowl_Team', 'Innings', 'Total_Runs', 'Overs', 'Wickets', 'Bowled_Balls'])
                print(f"Created DataFrame for file {filename}:")
                print(df)

                # Add the DataFrame to the list
                dataframes.append(df)

        # Combine all DataFrames from the text files into a single DataFrame
        final_df = pd.concat(dataframes, ignore_index=True)
        print("Combined all DataFrames:")
        print(final_df)

        # Calculate Run_Rate as Total_Runs / Overs
        final_df['Run_Rate'] = np.where(final_df['Overs'] > 0, final_df['Total_Runs'] / final_df['Overs'], 0)

        # Format Run_Rate to two decimal places
        final_df['Run_Rate'] = final_df['Run_Rate'].round(2)

        # Calculate Balls_Per_Wicket as Bowled_Balls / Wickets
        final_df['Balls_Per_Wicket'] = np.where(final_df['Wickets'] > 0, final_df['Bowled_Balls'] / final_df['Wickets'], 0)

        # Format Balls_Per_Wicket to two decimal places
        final_df['Balls_Per_Wicket'] = final_df['Balls_Per_Wicket'].round(2)

        # Merge final_df with match_df to bring in Competition, Format, Level, Year
        final_df = final_df.merge(match_df[['File Name', 'Competition', 'Match_Format', 'Player_of_the_Match', 'Date']], on='File Name', how='left')

        print("Game stats processing completed successfully")
        print(f"Final game DataFrame shape: {final_df.shape}")
        print(f"Final game DataFrame columns: {final_df.columns}")

        return final_df

    except Exception as e:
        print(f"Error in game stats processing: {str(e)}")
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    directory_path = r"C:\Users\rtayl\AppData\Roaming\Childish Things\Cricket Captain 2024\Saves\Scorecards"
    
    # Load match_df from CSV
    csv_path = os.path.join(directory_path, "match_data.csv")
    if os.path.exists(csv_path):
        match_df = pd.read_csv(csv_path)
        print("Loaded match data from CSV")
    else:
        print("match_data.csv not found. Please run match.py first.")
        exit()

    game_df = process_game_stats(directory_path, match_df)
    if game_df is not None:
        print(game_df.head(50))
        
        # Optionally, save game_df to CSV
        game_csv_path = os.path.join(directory_path, "game_data.csv")
        game_df.to_csv(game_csv_path, index=False)
        print(f"Game data saved to {game_csv_path}")
    else:
        print("Failed to process game stats")