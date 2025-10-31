import pandas as pd
import os
import re
import traceback
import numpy as np
from pathlib import Path

try:
    from performance_utils import perf_manager
except Exception:  # pragma: no cover - optional dependency when running standalone
    perf_manager = None

PLAYER_LINE_PATTERN = re.compile(
    r"^(?P<Name>.+?)(?:\s+(?P<How_Out>(lbw|c|b|not out|run out|st|retired).+?))?\s+"
    r"(?P<Runs>\d+)\s+(?P<Balls>\d+)\s+(?P<Fours>\d+|-)\s+(?P<Sixes>\d+|-)$"
)

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

        # Use pathlib for faster file operations
        directory = Path(directory_path)
        txt_files = list(directory.glob("*.txt"))
        
        # Pre-allocate list for all batting data
        all_batting_data = []

        # Loop through all text files
        for file_path in txt_files:
            filename = file_path.name

            # Read file content more efficiently
            file_content = file_path.read_text(encoding='utf-8')
            file_lines = file_content.splitlines()

            # Initialize variables for tracking position in the file
            Row_No = 0           # Current row number being processed
            Inns = 0             # Current innings (1-4 for Test matches, 1-2 for limited overs)
            Line_no = 0          # Track separator lines to determine innings
            Bat_Team = ""        # Team currently batting
            Bowl_Team = ""       # Team currently bowling
            Home_Team = ""       # Home team
            Away_Team = ""       # Away team
            
            # Process the file line by line
            for i, line in enumerate(file_lines):
                Row_No += 1

                # Extract Home_Team and Away_Team from Row 2 (format: "TeamA v TeamB")
                if Row_No == 2:
                    teams = line.split(" v ", 1)  # Limit split for efficiency
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
                        Bat_Team = line.split('-', 1)[0].strip()  # Limit split for efficiency
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

                            # Use pre-compiled regex to extract player stats
                            player_match = PLAYER_LINE_PATTERN.search(player_line)
                            
                            if player_match:
                                # Extract and clean player data from regex match
                                Name = player_match.group('Name').replace('rtrd ht', '').strip()
                                How_Out = player_match.group('How_Out').strip() if player_match.group('How_Out') else "Did not bat"
                                Runs = int(player_match.group('Runs'))
                                Balls = int(player_match.group('Balls'))
                                # Handle '-' notation for no boundaries
                                Fours = int(player_match.group('Fours')) if player_match.group('Fours') != '-' else 0
                                Sixes = int(player_match.group('Sixes')) if player_match.group('Sixes') != '-' else 0

                                # Add player data directly to main list
                                all_batting_data.append([
                                    filename, Inns, Bat_Team, Bowl_Team, position, 
                                    Name, How_Out, Runs, Balls, Fours, Sixes, 
                                    Home_Team, Away_Team
                                ])
                                position += 1  # Increment batting position
                            else:
                                # If regex doesn't match, treat as "Did not bat"
                                all_batting_data.append([
                                    filename, Inns, Bat_Team, Bowl_Team, position,
                                    player_line.strip(), "Did not bat", 0, 0, 0, 0,
                                    Home_Team, Away_Team
                                ])
                                position += 1

                        # Fill remaining positions up to 11 with empty "Did not bat" entries
                        # This ensures consistent team size across all innings
                        while position <= 11:
                            all_batting_data.append([
                                filename, Inns, Bat_Team, Bowl_Team, position,
                                '', 'Did not bat', 0, 0, 0, 0,
                                Home_Team, Away_Team
                            ])
                            position += 1

        # Create single DataFrame at the end instead of concatenating multiple
        if not all_batting_data:
            return pd.DataFrame(columns=[
                'File Name', 'Innings', 'Bat Team', 'Bowl Team', 'Position', 
                'Name', 'How Out', 'Runs', 'Balls', '4s', '6s', 
                'Home Team', 'Away Team'
            ])

        # Convert all batting data to DataFrame with appropriate column names
        bat_df = pd.DataFrame(
            all_batting_data, 
            columns=[
                'File Name', 'Innings', 'Bat Team', 'Bowl Team', 'Position', 
                'Name', 'How Out', 'Runs', 'Balls', '4s', '6s', 
                'Home Team', 'Away Team'
            ]
        )

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
                return comp

            if 'Test Match' in comp:
                team_pair = (row['Home Team'], row['Away Team'])
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
            if 'Australian League' in comp:
                return 'Sheffield Shield'
            if 'English FC League - D2' in comp:
                return 'County Championship Division 2'
            if 'English FC League - D1' in comp:
                return 'County Championship Division 1'
            if 'Challenge Trophy' in comp:
                return 'Royal London Cup'
            if comp.startswith('FC League'):
                return 'FC League'
            if comp.startswith('Super Cup'):
                return 'Super Cup'
            if comp.startswith('20 Over Trophy'):
                return '20 Over Trophy'
            if comp.startswith('One Day Cup'):
                return 'One Day Cup'
            return comp

        if not bat_df.empty:
            for numeric_col in ['Runs', 'Balls', '4s', '6s', 'Wickets', 'Overs', 'Bowled_Balls']:
                if numeric_col in bat_df.columns:
                    bat_df[numeric_col] = pd.to_numeric(bat_df[numeric_col], errors='coerce')

            balls_by_innings = (
                bat_df.groupby(['File Name', 'Innings'])['Balls'].sum().to_dict()
                if {'File Name', 'Innings', 'Balls'}.issubset(bat_df.columns)
                else {}
            )

            format_ball_map = {
                'The Hundred': 100,
                '100 Ball Trophy': 100,
                'T20': 120,
                'One Day': 300,
            }

            def overs_to_balls(series: pd.Series) -> np.ndarray:
                overs = pd.to_numeric(series, errors='coerce').fillna(0).astype(float)
                whole = np.floor(overs).astype(int)
                partial = np.round((overs - whole) * 10).astype(int)
                return (whole * 6 + partial).astype(int)

            def enrich_batting_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
                chunk = chunk.copy()

                date_series = chunk.get('Date', '').fillna('').astype(str)
                chunk['Year'] = pd.to_numeric(date_series.str[-4:], errors='coerce').fillna(0).astype(int).astype(str)

                how_out = chunk['How Out'].fillna('')
                chunk['Batted'] = (how_out != 'Did not bat').astype(int)
                chunk['Out'] = ((how_out != 'Did not bat') & (how_out != 'not out')).astype(int)
                chunk['Not Out'] = (how_out == 'not out').astype(int)
                chunk['DNB'] = (how_out == 'Did not bat').astype(int)

                runs = pd.to_numeric(chunk['Runs'], errors='coerce').fillna(0)
                balls = pd.to_numeric(chunk['Balls'], errors='coerce').fillna(0)
                fours = pd.to_numeric(chunk.get('4s'), errors='coerce').fillna(0)
                sixes = pd.to_numeric(chunk.get('6s'), errors='coerce').fillna(0)
                wickets = pd.to_numeric(chunk.get('Wickets'), errors='coerce').fillna(0)

                chunk['50s'] = ((runs >= 50) & (runs < 100)).astype(int)
                chunk['100s'] = ((runs >= 100) & (runs < 200)).astype(int)
                chunk['200s'] = (runs >= 200).astype(int)
                chunk['<25&Out'] = ((runs <= 25) & (chunk['Out'] == 1)).astype(int)

                lower_how_out = how_out.str.lower()
                chunk['Caught'] = lower_how_out.str.startswith('c ').astype(int)
                chunk['Bowled'] = lower_how_out.str.startswith('b ').astype(int)
                chunk['LBW'] = lower_how_out.str.startswith('lbw ').astype(int)
                chunk['Run Out'] = lower_how_out.str.startswith('run').astype(int)
                chunk['Stumped'] = lower_how_out.str.startswith('st').astype(int)

                chunk['Boundary Runs'] = (fours * 4 + sixes * 6).astype(int)

                strike_rate = np.where(balls > 0, (runs / balls) * 100, 0)
                chunk['Strike Rate'] = np.round(strike_rate, 2)

                keys = list(zip(
                    chunk['File Name'] if 'File Name' in chunk.columns else [''] * len(chunk),
                    chunk['Innings'] if 'Innings' in chunk.columns else [0] * len(chunk),
                ))
                all_out_mask = wickets == 10
                all_out_values = np.array([balls_by_innings.get(key, 0) for key in keys], dtype=float)

                match_format_series = (
                    chunk['Match_Format'] if 'Match_Format' in chunk.columns else pd.Series([''] * len(chunk))
                )
                format_values = match_format_series.map(format_ball_map).fillna(0).to_numpy(dtype=float)
                overs_based = overs_to_balls(chunk['Overs']) if 'Overs' in chunk.columns else np.zeros(len(chunk))
                bowled_balls_series = (
                    chunk['Bowled_Balls'] if 'Bowled_Balls' in chunk.columns else pd.Series([0] * len(chunk))
                )
                bowled_balls = pd.to_numeric(bowled_balls_series, errors='coerce').fillna(0).to_numpy(dtype=float)

                team_balls = np.where(
                    all_out_mask,
                    all_out_values,
                    np.where(
                        format_values > 0,
                        format_values,
                        np.where(bowled_balls > 0, bowled_balls, overs_based)
                    ),
                )
                chunk['Team Balls'] = np.nan_to_num(team_balls, nan=0).astype(int)

                try:
                    chunk['comp'] = chunk.apply(transform_competition, axis=1)
                except Exception:
                    chunk['comp'] = chunk.get('Competition')

                chunk['comp'] = chunk['comp'].fillna(chunk.get('Competition'))
                return chunk

            if perf_manager:
                chunk_size = getattr(perf_manager, 'chunk_size', 10_000)
                enriched_chunks = perf_manager.process_in_chunks(bat_df, enrich_batting_chunk, chunk_size)
                bat_df = pd.concat(enriched_chunks, ignore_index=True)
            else:
                bat_df = enrich_batting_chunk(bat_df)

        if 'comp' not in bat_df.columns:
            bat_df['comp'] = bat_df.get('Competition')

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
