import pandas as pd
import os
import re
import numpy as np
import traceback
from pathlib import Path

def process_game_stats(directory_path, match_df):
    """
    Process innings-level statistics from match text files.
    
    Args:
        directory_path: Path to directory containing match scorecard text files
        match_df: DataFrame containing match-level information
    
    Returns:
        DataFrame containing innings-level statistics
    """
    try:
        # Log the start of processing and input data dimensions
        print("Starting process_game_stats")
        print(f"Directory path: {directory_path}")
        print(f"Match DataFrame shape: {match_df.shape}")
        print(f"Match DataFrame columns: {match_df.columns}")

        # Pre-compile regex patterns for better performance
        all_out_pattern = re.compile(r"TOTAL: \(all out, (\d+\.\d+|\d+) overs\)\s+(\d+)")
        wickets_pattern = re.compile(r"TOTAL: \((\d+) wkts, (\d+\.\d+|\d+) overs\)")
        balls_pattern = re.compile(r"TOTAL: \((\d+) wkts, (\d+) balls\)")
        
        # Use pathlib for faster file operations
        directory = Path(directory_path)
        txt_files = list(directory.glob("*.txt"))
        
        # Pre-allocate list for all match data
        all_match_data = []
        
        # Standardize format names early to avoid repeated operations
        match_df = match_df.copy()
        match_df['Match_Format'] = match_df['Match_Format'].replace({
            '100 Trophy': 'The Hundred',
            '100 Ball Trophy': 'The Hundred',
            '100T': 'The Hundred'
        })
        
        # Create lookup dictionary for faster merging
        match_lookup = match_df.set_index('File Name')[['Competition', 'Match_Format', 'Player_of_the_Match', 'Date']].to_dict('index')

        # Loop through all text files
        for file_path in txt_files:
            filename = file_path.name
            print(f"Processing file: {filename}")

            # Read file content more efficiently
            file_content = file_path.read_text(encoding='utf-8')
            file_lines = file_content.splitlines()

            # Initialize variables for tracking position in the file
            Row_No = 0          # Current row number being processed
            Inns = 0            # Current innings (1-4 for Test matches, 1-2 for limited overs)
            Line_no = 0         # Track separator lines to determine innings
            Bat_Team = ""       # Team currently batting
            Home_Team = ""      # Home team
            Away_Team = ""      # Away team
            
            # Process the file line by line
            for i, line in enumerate(file_lines):
                Row_No += 1

                # Extract Home_Team and Away_Team from Row 2 (format: "TeamA v TeamB")
                if Row_No == 2:
                    teams = line.split(" v ", 1)  # Limit split for efficiency
                    if len(teams) == 2:
                        Home_Team = teams[0].strip()
                        Away_Team = teams[1].strip()
                    print(f"Home Team: {Home_Team}, Away Team: {Away_Team}")

                # Extract Bat_Team if the row contains "Innings"
                if "Innings" in line:
                    Bat_Team = line.split(" - ", 1)[0]
                    print(f"Batting Team: {Bat_Team}")

                # Check for batting innings separator lines (format: "TeamName -----------")
                if "-------------" in line:
                    Line_no += 1
                    # Lines 1, 5, 9, 13 indicate the start of innings 1, 2, 3, 4 respectively
                    if Line_no in [1, 5, 9, 13]:
                        Inns += 1  # Increment innings counter
                        print(f"Processing Innings {Inns}")

                        # Determine bowling team (if not batting, must be bowling)
                        Bowl_Team = Away_Team if Bat_Team == Home_Team else Home_Team

                        # Initialize innings total statistics
                        Runs = 0
                        Overs = None
                        Wickets = 0
                        Bowled_Balls = 0

                        # Extract total runs, overs, and wickets from the TOTAL line
                        # Limit search to next 50 lines for efficiency
                        search_limit = min(len(file_lines), i + 51)
                        for j in range(i + 1, search_limit):
                            total_line = file_lines[j]
                            print(f"Analyzing line: {total_line}")
                            
                            # Use pre-compiled regex patterns
                            all_out_match = all_out_pattern.search(total_line)
                            if all_out_match:
                                Runs = int(all_out_match.group(2))  # Total runs
                                Overs = float(all_out_match.group(1))  # Overs
                                Wickets = 10  # All out = 10 wickets
                                print(f"All out: Runs={Runs}, Overs={Overs}, Wickets={Wickets}")
                                break
                                
                            # Check for wickets scenario with overs (format: "TOTAL: (5 wkts, 90.2 overs) 301")
                            wickets_match = wickets_pattern.search(total_line)
                            if wickets_match:
                                Runs = int(total_line.split()[-1])  # Extract runs from the end of the line
                                Wickets = int(wickets_match.group(1))  # Number of wickets
                                Overs = float(wickets_match.group(2))  # Overs
                                print(f"Wickets with overs: Runs={Runs}, Overs={Overs}, Wickets={Wickets}")
                                break
                                
                            # Check for wickets scenario with balls (format: "TOTAL: (5 wkts, 100 balls) 301")
                            # Used in The Hundred format
                            balls_match = balls_pattern.search(total_line)
                            if balls_match:
                                Runs = int(total_line.split()[-1])  # Extract runs from the end of the line
                                Wickets = int(balls_match.group(1))  # Number of wickets
                                Bowled_Balls = int(balls_match.group(2))  # Balls
                                Overs = Bowled_Balls / 5  # Convert balls to overs using 5-ball overs for The Hundred
                                print(f"Wickets with balls: Runs={Runs}, Overs={Overs}, Wickets={Wickets}, Bowled_Balls={Bowled_Balls}")
                                break

                        # Default wickets to 0 if not found
                        if Wickets is None:
                            Wickets = 0
                            print("No wickets data found, setting Wickets to 0")

                        # Add all extracted innings data directly to main list
                        match_data = [filename, Home_Team, Away_Team, Bat_Team, Bowl_Team, Inns, Runs, Overs, Wickets, Bowled_Balls]
                        all_match_data.append(match_data)
                        print(f"Appended match data: {match_data}")

        # Create single DataFrame at the end instead of concatenating multiple
        if not all_match_data:
            return pd.DataFrame(columns=[
                'File Name', 'Home_Team', 'Away_Team', 'Bat_Team', 
                'Bowl_Team', 'Innings', 'Total_Runs', 'Overs', 
                'Wickets', 'Bowled_Balls', 'Run_Rate', 'Balls_Per_Wicket',
                'Competition', 'Match_Format', 'Player_of_the_Match', 'Date'
            ])
            
        final_df = pd.DataFrame(all_match_data, columns=[
            'File Name', 'Home_Team', 'Away_Team', 'Bat_Team', 
            'Bowl_Team', 'Innings', 'Total_Runs', 'Overs', 
            'Wickets', 'Bowled_Balls'
        ])
        print("Created final DataFrame:")
        print(final_df)

        # Calculate run rate (runs per over)
        # Only calculate when overs > 0 to avoid division by zero
        final_df['Run_Rate'] = np.where(final_df['Overs'] > 0, final_df['Total_Runs'] / final_df['Overs'], 0)

        # Format run rate to two decimal places
        final_df['Run_Rate'] = final_df['Run_Rate'].round(2)

        # Calculate balls per wicket
        # Only calculate when wickets > 0 to avoid division by zero
        final_df['Balls_Per_Wicket'] = np.where(final_df['Wickets'] > 0, final_df['Bowled_Balls'] / final_df['Wickets'], 0)

        # Format balls per wicket to two decimal places
        final_df['Balls_Per_Wicket'] = final_df['Balls_Per_Wicket'].round(2)

        # Use dictionary lookup for faster merging
        for idx, row in final_df.iterrows():
            filename = row['File Name']
            if filename in match_lookup:
                match_info = match_lookup[filename]
                final_df.at[idx, 'Competition'] = match_info['Competition']
                final_df.at[idx, 'Match_Format'] = match_info['Match_Format']
                final_df.at[idx, 'Player_of_the_Match'] = match_info['Player_of_the_Match']
                final_df.at[idx, 'Date'] = match_info['Date']

        # Log successful completion
        print("Game stats processing completed successfully")
        print(f"Final game DataFrame shape: {final_df.shape}")
        print(f"Final game DataFrame columns: {final_df.columns}")

        return final_df

    except Exception as e:
        # Catch and log any errors that occur during processing
        print(f"Error in game stats processing: {str(e)}")
        print(traceback.format_exc())  # Print full traceback for debugging
        return None

# Code that runs when this script is executed directly (not imported)
if __name__ == "__main__":
    # Define directory path for match scorecard files
    directory_path = r"C:\Users\rtayl\AppData\Roaming\Childish Things\Cricket Captain 2024\Saves\Scorecards"
    
    # Load match data from CSV (requires match.py to be run first)
    csv_path = os.path.join(directory_path, "match_data.csv")
    if os.path.exists(csv_path):
        match_df = pd.read_csv(csv_path)
        print("Loaded match data from CSV")
    else:
        print("match_data.csv not found. Please run match.py first.")
        exit()

    # Process game statistics
    game_df = process_game_stats(directory_path, match_df)
    
    # If processing was successful, display results and save to CSV
    if game_df is not None:
        print(game_df.head(50))
        
        # Save game data to CSV for use by other scripts
        game_csv_path = os.path.join(directory_path, "game_data.csv")
        game_df.to_csv(game_csv_path, index=False)
        print(f"Game data saved to {game_csv_path}")
    else:
        print("Failed to process game stats")
