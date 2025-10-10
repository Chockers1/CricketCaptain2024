import pandas as pd
import re
import os

def process_match_data(directory_path):
    """
    Process match data from text files to extract metadata about cricket matches.
    
    Args:
        directory_path: Path to directory containing match scorecard text files
    
    Returns:
        DataFrame containing match metadata (teams, results, dates, etc.)
    """
    # Initialize an empty list to hold DataFrames for each file
    # Each file will become a row in the final DataFrame
    dataframes = []

    # Loop through all files in the specified directory
    for filename in os.listdir(directory_path):
        # Process only text files - these contain the match data
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)

            # Open and read the file with UTF-8 encoding to handle special characters
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()

            # Extract match header from line 2 (format: "TeamA v TeamB")
            match_header = file_content.splitlines()[1].strip()
            home_team, away_team = match_header.split(' v ')

            # Extract competition and date from line 3
            # Format: "Competition Name - Date"
            line_3 = file_content.splitlines()[2].strip()

            try:
                # Split on the last occurrence of " - " to handle competition names that contain " - "
                competition, date = line_3.rsplit(' - ', 1)
            except ValueError:
                # Log error and skip file if format is incorrect
                print(f"Error processing file: {filename}")
                print(f"Line 3 content: {line_3}")
                continue

            # Remove any format details in parentheses
            competition = competition.split('(')[0].strip()

            # Determine match format based on competition name
            if 'Test' in line_3:
                match_format = 'Test Match'
            elif '20 Over International' in line_3 or 'World Cup 20' in line_3 or '20 Over Tournament' in line_3 or 'Asia Trophy 20' in line_3 or 'Int20 Tournament' in line_3:
                match_format = 'T20I'
            elif ('One Day International' in line_3 or 'World Cup -' in line_3 or 'Champions Cup' in line_3 or 
                  'ODI Tournament' in line_3 or 'Asia Trophy ODI' in line_3 or 
                  ('Asia Trophy' in line_3 and 'Asia Trophy 20' not in line_3)):
                match_format = 'ODI'
            elif '20 Over Trophy' in line_3 or '20 Over League' in line_3 or 'Domestic 20 Over' in line_3 or 'Provincial 20 Over' in line_3 or 'Super Trophy' in line_3 or 'Vitality Blast' in line_3 or 'Big Bash League' in line_3 or 'T20 Blast' in line_3 or 'Super Cup' in line_3:    
                match_format = 'T20'
            elif 'Test Championship Final' in line_3:
                match_format = 'Test Match'
            elif '100 Ball Trophy' in line_3 or 'The Hundred' in line_3:
                match_format = 'The Hundred'
            elif 'English FC League' in line_3 or 'Australian League' in line_3 or 'FC League' in line_3 or 'FC Plate' in line_3 or '4 Day Competition' in line_3 or 'Vitality County Championship' in line_3 or 'University Match' in line_3 or 'Sheffield Shield' in line_3 or 'County Championship' in line_3 or 'Three Day Friendly' in line_3:
                match_format = 'First Class'
            elif 'Challenge Trophy' in line_3 or 'One Day Cup' in line_3 or 'One Day Friendly' in line_3 or 'Dean Jones Trophy' in line_3:
                match_format = 'One Day'
            else:
                # Default if no recognized format
                match_format = 'Unknown'

            # Extract "Player of the Match" and match result from end of file
            lines = file_content.splitlines()
            pom = None              # Player of the Match
            match_result = None     # Match result text
            inns_win = 0            # Flag for innings victory (default: no)

            # Scan backwards through the file for results info
            for line in reversed(lines):
                if "Man of the match:" in line:
                    # Extract Player of the Match name
                    pom = line.split(":")[1].strip()
                elif "Match tied" in line:
                    match_result = "Match tied"
                elif "Match drawn" in line or "won" in line or "lost" in line:
                    # Store the full result text
                    match_result = line.strip()
                    # Check if this was an innings victory
                    if "innings" in match_result.lower():
                        inns_win = 1

            # Initialize result flag variables (all default to 0/False)
            home_win = home_lost = home_drawn = away_won = away_lost = away_drawn = Tie = 0

            # Set appropriate flags based on match result
            if match_result:
                if "Match tied" in match_result:
                    Tie = 1  # Tie flag
                elif "Match drawn" in match_result:
                    home_drawn = 1  # Both teams get a draw
                    away_drawn = 1
                elif home_team in match_result and "won" in match_result:
                    home_win = 1    # Home team won
                    away_lost = 1   # Away team lost
                elif away_team in match_result and "won" in match_result:
                    away_won = 1    # Away team won
                    home_lost = 1   # Home team lost

            # Extract margin of victory from the third-to-last line
            line_bottom_3 = file_content.splitlines()[-3].strip()

            # Check for abandoned matches (marked with *)
            if '*' in line_bottom_3:
                margin = "Match abandoned"
                home_drawn = 1
                away_drawn = 1
            else:
                margin = line_bottom_3

            # Initialize margin_runs and margin_wickets
            margin_runs = None
            margin_wickets = None

            # Extract numeric margin values using regex
            if "run" in margin:
                # Extract number of runs from margins like "123 runs"
                runs_match = re.search(r'(\d+)\s*run[s]?', margin)
                if runs_match:
                    margin_runs = int(runs_match.group(1))

            if "wicket" in margin:
                # Extract number of wickets from margins like "5 wickets"
                wickets_match = re.search(r'(\d+)\s*wicket[s]?', margin)
                if wickets_match:
                    margin_wickets = int(wickets_match.group(1))

            # Store all extracted data in a dictionary
            match_data = {
                "File Name": filename,           # Filename to link with other tables
                "Home_Team": home_team,          # Home team
                "Away_Team": away_team,          # Away team
                "Date": date,                    # Match date
                "Competition": competition,      # Competition name
                "Match_Format": match_format,    # Format (Test, ODI, T20, etc.)
                "Player_of_the_Match": pom,      # Player of the Match
                "Match_Result": match_result,    # Full result text
                "Margin": margin,                # Margin of victory text
                "Innings_Win": inns_win,         # Flag for innings victory
                "Home_Win": home_win,            # Home team won flag
                "Away_Won": away_won,            # Away team won flag
                "Home_Lost": home_lost,          # Home team lost flag
                "Away_Lost": away_lost,          # Away team lost flag
                "Home_Drawn": home_drawn,        # Home team drawn flag
                "Away_Drawn": away_drawn,        # Away team drawn flag
                "Tie": Tie,                      # Match tied flag
                "Margin_Runs": margin_runs,      # Run margin (if applicable)
                "Margin_Wickets": margin_wickets # Wicket margin (if applicable)
            }

            # Append the match data to our list of DataFrames
            dataframes.append(pd.DataFrame([match_data]))

    # Combine all individual file DataFrames into one master DataFrame
    if dataframes:
        final_df = pd.concat(dataframes, ignore_index=True)
        
        # Define a function to standardize competition names
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
                team_pair = (row['Home_Team'], row['Away_Team'])
                if team_pair in test_trophies:
                    return test_trophies[team_pair]
                return 'Test Match'
            elif comp.startswith('World Cup 20'):
                return 'T20 World Cup'
            elif comp.startswith('World Cup'):
                return 'ODI World Cup'
            elif comp.startswith('Champions Cup'):
                return 'Champions Cup'
            elif comp.startswith('Asia Trophy 20'):
                return 'T20 Asia Cup'
            elif comp.startswith('Asia Trophy'):
                return 'ODI Asia Trophy'
            elif comp.startswith('Asia Trophy ODI'):
                return 'ODI Asia Cup'
            elif comp.startswith('FC League'):
                return 'FC League'
            elif comp.startswith('Super Cup'):
                return 'Super Cup'
            elif comp.startswith('20 Over Trophy'):
                return '20 Over Trophy'
            elif comp.startswith('One Day Cup'):
                return 'One Day Cup'
            elif 'One Day International' in comp:
                return 'ODI'
            elif '20 Over International' in comp:
                return 'T20I'
            else:
                return comp

        # Apply competition name transformation
        try:
            final_df['comp'] = final_df.apply(transform_competition, axis=1)
        except Exception as e:
            # Log error and fall back to original competition name
            print(f"Error creating comp column: {e}")
            final_df['comp'] = final_df['Competition']

        # Verify comp column exists before proceeding
        if 'comp' not in final_df.columns:
            print("Warning: comp column not created, using Competition instead")
            final_df['comp'] = final_df['Competition']

        # Save the DataFrame to a CSV file for use by other scripts
        csv_path = os.path.join(directory_path, "match_data.csv")
        final_df.to_csv(csv_path, index=False)
        print(f"Match data saved to {csv_path}")
        
        return final_df  # Return the processed DataFrame
    else:
        print("No match data found.")
        return pd.DataFrame()  # Return an empty DataFrame if no data found

# Code that runs when this script is executed directly (not imported)
if __name__ == "__main__":
    # Define directory path for match scorecard files
    directory_path = r"C:\Users\rtayl\AppData\Roaming\Childish Things\Cricket Captain 2024\Saves\Scorecards"
    
    # Process match data
    match_df = process_match_data(directory_path)
    
    # Display the processed data
    print(match_df)
 
