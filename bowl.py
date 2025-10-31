import pandas as pd
import numpy as np
import os
import re
import traceback

try:
    from performance_utils import perf_manager
except Exception:  # pragma: no cover - optional dependency when running standalone
    perf_manager = None

def process_bowl_stats(directory_path, game_df, match_df):
    """
    Process bowling statistics from text files and merge with game data.
    
    Args:
        directory_path: Path to directory containing bowling statistics text files
        game_df: DataFrame containing game-level information
        match_df: DataFrame containing match-level information
    
    Returns:
        DataFrame containing processed bowling statistics
    """
    try:
        # Log the start of processing and input data dimensions
        print("Starting bowl stats processing")
        print(f"Directory path: {directory_path}")
        print(f"game_df shape: {game_df.shape}")
        print(f"game_df columns: {game_df.columns}")
        print(f"match_df shape: {match_df.shape}")
        print(f"match_df columns: {match_df.columns}")

        # Initialize an empty list to hold DataFrames for each file
        # Each file will become a DataFrame, and they'll all be combined at the end
        dataframes = []

        # Loop through all files in the specified directory
        for filename in os.listdir(directory_path):
            # Process only text files - these contain the match bowling data
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
                innings_data = []    # List to collect bowler data rows
                
                # Process the file line by line
                for i, line in enumerate(file_lines):
                    Row_No += 1

                    # Extract Home_Team and Away_Team from Row 2 (format: "TeamA v TeamB")
                    if Row_No == 2:
                        teams = line.split(" v ")
                        if len(teams) == 2:
                            Home_Team = teams[0].strip()
                            Away_Team = teams[1].strip()

                    # Check for bowling separator lines 
                    # Note: Bowling data appears after lines 3, 7, 11, 15
                    if "-------------" in line:
                        Line_no += 1
                        if Line_no in [3, 7, 11, 15]:  # Increment innings based on bowling data separators
                            Inns += 1  # Increment innings counter

                            # Extract batting team name from left side of separator
                            Bat_Team = line.split('-')[0].strip()
                            # Determine bowling team (if not batting, must be bowling)
                            Bowl_Team = Away_Team if Bat_Team == Home_Team else Home_Team

                            # Initialize bowling position counter for this innings
                            position = 1

                            # Process up to 11 bowlers (maximum in a cricket team)
                            for j in range(11):
                                # Avoid index errors at end of file
                                if i + j + 1 >= len(file_lines):
                                    break

                                # Get the line containing bowler data
                                player_line = file_lines[i + j + 1]
                                if player_line.strip() == "":  # Skip empty lines
                                    continue

                                # Use regex to extract bowler stats from the line
                                # Format: Name Overs Maidens Runs Wickets Economy
                                bowler_match = re.search(
                                    r"^(?P<Name>.+?)\s+(?P<Overs>\d+(\.\d+)?)\s+(?P<Maidens>\d+)\s+(?P<Runs>\d+)\s+(?P<Wickets>\d+)\s+(?P<Econ>\d+(\.\d+)?)$",
                                    player_line
                                )
                                
                                if bowler_match:
                                    # Extract bowler data from regex match
                                    Name = bowler_match.group('Name').strip()
                                    Overs = float(bowler_match.group('Overs'))
                                    Maidens = int(bowler_match.group('Maidens'))
                                    Runs = int(bowler_match.group('Runs'))
                                    Wickets = int(bowler_match.group('Wickets'))
                                    Econ = float(bowler_match.group('Econ'))

                                    # Add bowler data to innings_data list
                                    innings_data.append([
                                        filename, Inns, Bat_Team, Bowl_Team, position, 
                                        Name, Overs, Maidens, Runs, Wickets, Econ
                                    ])
                                    position += 1  # Increment bowling position
                                else:
                                    # If regex doesn't match, skip this line
                                    continue

                            # If fewer than 11 bowlers, fill in empty rows for remaining positions
                            # This ensures consistent team size across all innings
                            for k in range(position, 12):  # Fill up to position 11 (index starts at 1)
                                innings_data.append([
                                    filename, Inns, Bat_Team, Bowl_Team, position, 
                                    '', 0, 0, 0, 0, 0
                                ])
                                position += 1

                # Convert innings data to DataFrame with appropriate column names
                df_innings = pd.DataFrame(
                    innings_data, 
                    columns=[
                        'File Name', 'Innings', 'Bat Team', 'Bowl Team', 'Position', 
                        'Name', 'Bowler_Overs', 'Maidens', 'Bowler_Runs', 'Bowler_Wkts', 'Bowler_Econ'
                    ]
                )

                # Add this file's DataFrame to our list
                dataframes.append(df_innings)

        # Combine all individual file DataFrames into one master DataFrame
        final_innings_df = pd.concat(dataframes, ignore_index=True)

        # Create bowling DataFrame from the final innings data
        bowl_df = final_innings_df

        # Step 1: Merge with match_df to get Home_Team and Away_Team
        # This adds match-level team information
        bowl_df = bowl_df.merge(
            match_df[['File Name', 'Home_Team', 'Away_Team']],
            on='File Name',
            how='left'  # Left join to keep all bowling records
        )

        # Step 2: Drop the 'Bat Team' and 'Bowl Team' columns from bowl_df
        # These will be replaced with the standardized columns from game_df
        bowl_df_dropped = bowl_df.drop(columns=['Bat Team', 'Bowl Team'])

        # Step 3: Merge with game_df to get additional match information
        # This adds innings-specific data like total runs, overs, wickets, etc.
        merged_df = bowl_df_dropped.merge(
            game_df[['File Name', 'Innings', 'Bat_Team', 'Bowl_Team', 'Total_Runs', 'Overs', 'Wickets', 
                    'Competition', 'Match_Format', 'Player_of_the_Match', 'Date']],
            on=['File Name', 'Innings'],
            how='left'  # Left join to keep all bowling records
        )

        bowl_df = merged_df

        def transform_competition(row):
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
            if not isinstance(comp, str):
                return '' if pd.isna(comp) else comp

            if 'Test Match' in comp:
                team_pair = (row['Home_Team'], row['Away_Team'])
                if team_pair in test_trophies:
                    return test_trophies[team_pair]
                return 'Test Match'
            if comp.startswith('World Cup 20'):
                return 'T20 World Cup'
            if comp.startswith('World Cup'):
                return 'ODI World Cup'
            if comp.startswith('Champions Cup'):
                return 'Champions Cup'
            if comp.startswith('Asia Trophy ODI'):
                return 'ODI Asia Cup'
            if comp.startswith('Asia Trophy 20'):
                return 'T20 Asia Cup'
            if 'One Day International' in comp:
                return 'ODI'
            if '20 Over International' in comp:
                return 'T20I'
            if comp.startswith('FC League'):
                return 'FC League'
            if comp.startswith('Super Cup'):
                return 'Super Cup'
            if comp.startswith('20 Over Trophy'):
                return '20 Over Trophy'
            if comp.startswith('One Day Cup'):
                return 'One Day Cup'
            return comp

        bowl_df = bowl_df[bowl_df['Name'].astype(str).str.strip() != '']

        if not bowl_df.empty:
            for col in ['Bowler_Overs', 'Bowler_Econ', 'Bowler_Runs', 'Bowler_Wkts', 'Maidens']:
                if col in bowl_df.columns:
                    bowl_df[col] = pd.to_numeric(bowl_df[col], errors='coerce')

            def enrich_bowling_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
                chunk = chunk.copy()

                wickets = pd.to_numeric(chunk.get('Bowler_Wkts'), errors='coerce').fillna(0)
                runs = pd.to_numeric(chunk.get('Bowler_Runs'), errors='coerce').fillna(0.0)
                overs_original = pd.to_numeric(chunk.get('Bowler_Overs'), errors='coerce').fillna(0.0)
                econ_original = pd.to_numeric(chunk.get('Bowler_Econ'), errors='coerce').fillna(0.0)
                maidens = pd.to_numeric(chunk.get('Maidens'), errors='coerce').fillna(0.0)

                chunk['Bowled'] = 1
                chunk['5Ws'] = ((wickets >= 5) & (wickets < 10)).astype(int)
                chunk['10Ws'] = (wickets >= 10).astype(int)

                match_format_series = (
                    chunk['Match_Format'] if 'Match_Format' in chunk.columns else pd.Series([''] * len(chunk))
                )
                hundred_mask = match_format_series.isin(['The Hundred', '100 Ball Trophy']).to_numpy()

                whole = np.floor(overs_original).astype(int)
                partial = np.round((overs_original - whole) * 10).astype(int)
                balls_regular = (whole * 6 + partial).astype(float)
                runs_array = runs.to_numpy(dtype=float)
                overs_array_original = overs_original.to_numpy(dtype=float)
                balls = np.where(hundred_mask, overs_array_original, balls_regular)
                balls = np.where(balls < 0, 0, balls)

                adjusted_overs = np.where(hundred_mask, np.where(balls > 0, balls / 5, 0), overs_array_original)
                with np.errstate(divide='ignore', invalid='ignore'):
                    adjusted_econ = np.where(
                        hundred_mask,
                        np.where(balls > 0, (runs_array / balls) * 5, 0),
                        econ_original.to_numpy(dtype=float),
                    )

                chunk['Bowler_Balls'] = balls
                chunk['Bowler_Overs'] = adjusted_overs
                chunk['Bowler_Econ'] = adjusted_econ

                overs_array = np.asarray(adjusted_overs, dtype=float)
                wickets_array = wickets.to_numpy(dtype=float)
                maidens_array = maidens.to_numpy(dtype=float)

                runs_per_over = np.divide(
                    runs_array,
                    overs_array,
                    out=np.zeros_like(runs_array, dtype=float),
                    where=overs_array > 0,
                )
                chunk['Runs_Per_Over'] = runs_per_over

                balls_per_wicket = np.divide(
                    balls,
                    wickets_array,
                    out=balls.astype(float),
                    where=wickets_array > 0,
                )
                chunk['Balls_Per_Wicket'] = balls_per_wicket

                ball_factor = np.where(hundred_mask, 5, 6)
                dot_percentage = np.divide(
                    maidens_array * ball_factor * 100,
                    balls,
                    out=np.zeros_like(maidens_array, dtype=float),
                    where=balls > 0,
                )
                chunk['Dot_Ball_Percentage'] = dot_percentage

                strike_rate = np.divide(
                    balls,
                    wickets_array,
                    out=balls.astype(float),
                    where=wickets_array > 0,
                )
                chunk['Strike_Rate'] = strike_rate

                averages = np.divide(
                    runs_array,
                    wickets_array,
                    out=runs_array,
                    where=wickets_array > 0,
                )
                chunk['Average'] = averages

                try:
                    chunk['comp'] = chunk.apply(transform_competition, axis=1)
                except Exception:
                    chunk['comp'] = chunk.get('Competition')

                chunk['comp'] = chunk['comp'].fillna(chunk.get('Competition'))
                return chunk

            if perf_manager:
                chunk_size = getattr(perf_manager, 'chunk_size', 10_000)
                enriched_chunks = perf_manager.process_in_chunks(bowl_df, enrich_bowling_chunk, chunk_size)
                bowl_df = pd.concat(enriched_chunks, ignore_index=True)
            else:
                bowl_df = enrich_bowling_chunk(bowl_df)

        if 'comp' not in bowl_df.columns:
            bowl_df['comp'] = bowl_df.get('Competition')

        print("Bowl stats processing completed successfully")
        print(f"Final bowl_df shape: {bowl_df.shape}")
        print(f"Final bowl_df columns: {bowl_df.columns}")

        return bowl_df

    except Exception as e:
        # Catch and log any errors that occur during processing
        print(f"Error in bowl stats processing: {str(e)}")
        print(traceback.format_exc())  # Print full traceback for debugging
        return None

# Code that runs when this script is executed directly (not imported)
if __name__ == "__main__":
    # Define directory path for bowling statistics files
    directory_path = r"C:\Users\rtayl\AppData\Roaming\Childish Things\Cricket Captain 2024\Saves\Scorecards"
    
    # Load game_df and match_df from CSV files
    game_csv_path = os.path.join(directory_path, "game_data.csv")
    match_csv_path = os.path.join(directory_path, "match_data.csv")
    
    # Check if required data files exist before proceeding
    if os.path.exists(game_csv_path) and os.path.exists(match_csv_path):
        # Load the data files into DataFrames
        game_df = pd.read_csv(game_csv_path)
        match_df = pd.read_csv(match_csv_path)
        print("Loaded game and match data from CSV files")
    else:
        print("game_data.csv or match_data.csv not found. Please run match.py and game.py first.")
        exit()

    # Process bowling stats
    bowl_df = process_bowl_stats(directory_path, game_df, match_df)
    
    # If processing was successful, display results and save to CSV
    if bowl_df is not None:
        print(bowl_df.head(50))
        
        # Save bowl_df to CSV
        bowl_csv_path = os.path.join(directory_path, "bowl_data.csv")
        bowl_df.to_csv(bowl_csv_path, index=False)
        print(f"Bowl data saved to {bowl_csv_path}")
    else:
        print("Failed to process bowl stats")

# Code for direct file execution and debugging
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
