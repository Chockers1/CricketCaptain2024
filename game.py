import os
import re
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from performance_utils import perf_manager
except Exception:  # pragma: no cover - optional when running standalone
    perf_manager = None

ALL_OUT_PATTERN = re.compile(r"TOTAL: \(all out, (\d+\.\d+|\d+) overs\)\s+(\d+)")
WICKETS_PATTERN = re.compile(r"TOTAL: \((\d+) wkts, (\d+\.\d+|\d+) overs\)")
BALLS_PATTERN = re.compile(r"TOTAL: \((\d+) wkts, (\d+) balls\)")

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
        def _log_dataframe_summary(label: str, df: Optional[pd.DataFrame]) -> None:
            prefix = "[GAME]"
            if df is None:
                print(f"{prefix} {label}: DataFrame is None")
                return
            rows, cols = df.shape
            try:
                memory_bytes = int(df.memory_usage(deep=True).sum())
            except Exception:
                memory_bytes = -1
            columns_preview = ", ".join(str(col) for col in list(df.columns)[:10])
            memory_part = f" memory_bytes={memory_bytes}" if memory_bytes >= 0 else ""
            columns_part = f" columns=[{columns_preview}]" if columns_preview else ""
            print(f"{prefix} {label}: rows={rows} cols={cols}{memory_part}{columns_part}")

        # Log the start of processing and input data dimensions
        print("[GAME] Starting game stats processing")
        print(f"[GAME] Directory path: {directory_path}")
        _log_dataframe_summary("match_df_input", match_df)

        # Use pathlib for faster file operations
        directory = Path(directory_path)
        txt_files = list(directory.glob("*.txt"))
        print(f"[GAME] Scorecard files detected: {len(txt_files)}")
        
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
                            all_out_match = ALL_OUT_PATTERN.search(total_line)
                            if all_out_match:
                                Runs = int(all_out_match.group(2))  # Total runs
                                Overs = float(all_out_match.group(1))  # Overs
                                Wickets = 10  # All out = 10 wickets
                                print(f"All out: Runs={Runs}, Overs={Overs}, Wickets={Wickets}")
                                break
                                
                            # Check for wickets scenario with overs (format: "TOTAL: (5 wkts, 90.2 overs) 301")
                            wickets_match = WICKETS_PATTERN.search(total_line)
                            if wickets_match:
                                Runs = int(total_line.split()[-1])  # Extract runs from the end of the line
                                Wickets = int(wickets_match.group(1))  # Number of wickets
                                Overs = float(wickets_match.group(2))  # Overs
                                print(f"Wickets with overs: Runs={Runs}, Overs={Overs}, Wickets={Wickets}")
                                break
                                
                            # Check for wickets scenario with balls (format: "TOTAL: (5 wkts, 100 balls) 301")
                            # Used in The Hundred format
                            balls_match = BALLS_PATTERN.search(total_line)
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

        def enrich_game_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
            chunk = chunk.copy()

            runs = pd.to_numeric(chunk['Total_Runs'], errors='coerce')
            overs = pd.to_numeric(chunk['Overs'], errors='coerce')
            wickets = pd.to_numeric(chunk['Wickets'], errors='coerce')
            balls = pd.to_numeric(chunk['Bowled_Balls'], errors='coerce')

            chunk['Run_Rate'] = np.where(overs > 0, runs / overs, 0).round(2)
            chunk['Run_Rate'] = chunk['Run_Rate'].fillna(0)
            chunk['Balls_Per_Wicket'] = np.where(wickets > 0, balls / wickets, 0).round(2)
            chunk['Balls_Per_Wicket'] = chunk['Balls_Per_Wicket'].fillna(0)

            metadata = chunk['File Name'].map(match_lookup)
            competition = []
            match_format = []
            player_of_match = []
            date_vals = []
            for info in metadata:
                if isinstance(info, dict):
                    competition.append(info.get('Competition'))
                    match_format.append(info.get('Match_Format'))
                    player_of_match.append(info.get('Player_of_the_Match'))
                    date_vals.append(info.get('Date'))
                else:
                    competition.append(None)
                    match_format.append(None)
                    player_of_match.append(None)
                    date_vals.append(None)

            chunk['Competition'] = competition
            chunk['Match_Format'] = match_format
            chunk['Player_of_the_Match'] = player_of_match
            chunk['Date'] = date_vals
            return chunk

        if perf_manager:
            chunk_size = getattr(perf_manager, 'chunk_size', 10_000)
            print(f"[GAME] Enriching game data in chunks (chunk_size={chunk_size})")
            enriched_chunks = perf_manager.process_in_chunks(final_df, enrich_game_chunk, chunk_size)
            final_df = pd.concat(enriched_chunks, ignore_index=True)
        else:
            print("[GAME] Enriching game data without chunk manager")
            final_df = enrich_game_chunk(final_df)

        _log_dataframe_summary("final_enriched", final_df)

        # Log successful completion
        print("[GAME] Game stats processing completed successfully")

        _log_dataframe_summary("final_saved", final_df)

        return final_df

    except Exception as e:
        # Catch and log any errors that occur during processing
        print(f"[GAME] Error in game stats processing: {str(e)}")
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
