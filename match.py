import pandas as pd
import re
import os

def process_match_data(directory_path):
    # Initialize an empty list to hold DataFrames for each file
    dataframes = []

    # Loop through all files in the specified directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)

            # Open and read the file
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()

            match_header = file_content.splitlines()[1].strip()
            home_team, away_team = match_header.split(' v ')

            line_3 = file_content.splitlines()[2].strip()

            try:
                competition, date = line_3.rsplit(' - ', 1)
            except ValueError:
                print(f"Error processing file: {filename}")
                print(f"Line 3 content: {line_3}")
                continue

            competition = competition.split('(')[0].strip()

            # Determine match format
            if 'Test' in line_3:
                match_format = 'Test Match'
            elif 'One Day International' in line_3 or 'World Cup -' in line_3 or 'Champions Cup' in line_3 or 'ODI Tournament' in line_3 or 'Asia Trophy ODI' in line_3:
                match_format = 'ODI'
            elif '20 Over International' in line_3 or 'World Cup 20' in line_3 or '20 Over Tournament' in line_3 or 'Asia Trophy 20' in line_3:
                match_format = 'T20I'
            elif '20 Over Trophy' in line_3 or '20 Over League' in line_3 or 'Domestic 20 Over' in line_3 or 'Provincial 20 Over' in line_3 or 'Super Trophy' in line_3 or 'Vitality Blast' in line_3 or 'Big Bash League' in line_3 or 'T20 Blast' in line_3:   
                match_format = 'T20'
            elif 'Test Championship Final' in line_3:
                match_format = 'Test Match'
            elif '100 Ball Trophy' in line_3 or 'The Hundred' in line_3:
                match_format = 'The Hundred'
            elif 'English FC League' in line_3 or 'Australian League' in line_3 or 'FC League' in line_3 or 'FC Plate' in line_3 or '4 Day Competition' in line_3 or 'Vitality County Championship' in line_3 or 'University Match' in line_3 or 'Sheffield Shield' in line_3 or 'County Championship' in line_3:
                match_format = 'First Class'
            elif 'Challenge Trophy' in line_3 or 'One Day Cup' in line_3 or 'One Day Friendly' in line_3 or 'Dean Jones Trophy' in line_3:
                match_format = 'One Day'
            else:
                match_format = 'Unknown'  # Default value if no match found

            # Extract "Player of the Match" and match result
            lines = file_content.splitlines()
            pom = None
            match_result = None
            inns_win = 0  # Default value is 0 unless "innings" is found in the result line

            for line in reversed(lines):
                if "Man of the match:" in line:
                    pom = line.split(":")[1].strip()
                elif "Match tied" in line:
                    match_result = "Match tied"
                elif "Match drawn" in line or "won" in line or "lost" in line:
                    match_result = line.strip()
                    if "innings" in match_result.lower():
                        inns_win = 1  # Set `inns_win` to 1 if "innings" is found

            # Initialize new columns
            home_win = home_lost = home_drawn = away_won = away_lost = away_drawn = Tie = 0

            # Determine match result conditions
            if match_result:
                if "Match tied" in match_result:
                    Tie = 1
                elif "Match drawn" in match_result:
                    home_drawn = 1
                    away_drawn = 1
                elif home_team in match_result and "won" in match_result:
                    home_win = 1
                    away_lost = 1
                elif away_team in match_result and "won" in match_result:
                    away_won = 1
                    home_lost = 1

            # Extract margin from the third-to-last line
            line_bottom_3 = file_content.splitlines()[-3].strip()

            if '*' in line_bottom_3:
                margin = "Match abandoned"
                home_drawn = 1
                away_drawn = 1
            else:
                margin = line_bottom_3

            # Initialize margin_runs and margin_wickets
            margin_runs = None
            margin_wickets = None

            # Extract numeric values for margin_runs and margin_wickets
            if "run" in margin:
                runs_match = re.search(r'(\d+)\s*run[s]?', margin)
                if runs_match:
                    margin_runs = int(runs_match.group(1))

            if "wicket" in margin:
                wickets_match = re.search(r'(\d+)\s*wicket[s]?', margin)
                if wickets_match:
                    margin_wickets = int(wickets_match.group(1))

            # Store extracted data in a dictionary
            match_data = {
                "File Name": filename,
                "Home_Team": home_team,
                "Away_Team": away_team,
                "Date": date,
                "Competition": competition,
                "Match_Format": match_format,
                "Player_of_the_Match": pom,
                "Match_Result": match_result,
                "Margin": margin,
                "Innings_Win": inns_win,
                "Home_Win": home_win,
                "Away_Won": away_won,
                "Home_Lost": home_lost,
                "Away_Lost": away_lost,
                "Home_Drawn": home_drawn,
                "Away_Drawn": away_drawn,
                "Tie": Tie,
                "Margin_Runs": margin_runs,
                "Margin_Wickets": margin_wickets
            }

            # Append the match data to the DataFrame
            dataframes.append(pd.DataFrame([match_data]))

    # Concatenate all DataFrames and reset the index
    if dataframes:
        final_df = pd.concat(dataframes, ignore_index=True)
        
        # Ensure that only one competition is selected when processing matches
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
            elif comp.startswith('Asia Trophy ODI'):
                return 'ODI Asia Cup'
            elif comp.startswith('Asia Trophy 20'):
                return 'T20 Asia Cup'
            elif 'One Day International' in comp:
                return 'ODI'
            elif '20 Over International' in comp:
                return 'T20I'
            else:
                return comp

        # Apply the competition transformation
        try:
            final_df['comp'] = final_df.apply(transform_competition, axis=1)
        except Exception as e:
            print(f"Error creating comp column: {e}")
            final_df['comp'] = final_df['Competition']

        if 'comp' not in final_df.columns:
            print("Warning: comp column not created, using Competition instead")
            final_df['comp'] = final_df['Competition']

        # Save the DataFrame to a CSV file
        csv_path = os.path.join(directory_path, "match_data.csv")
        final_df.to_csv(csv_path, index=False)
        print(f"Match data saved to {csv_path}")
        
        return final_df  # Return the final DataFrame
    else:
        print("No match data found.")
        return pd.DataFrame()  # Return an empty DataFrame if no data found

if __name__ == "__main__":
    directory_path = r"C:\Users\rtayl\AppData\Roaming\Childish Things\Cricket Captain 2024\Saves\Scorecards"
    match_df = process_match_data(directory_path)
    print(match_df)