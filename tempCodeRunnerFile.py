from shiny import App, render, ui
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
import numpy as np
from pathlib import Path

# Data processing function
def process_data():
    # Read the CSV file
    df = pd.read_csv(r"C:\Users\rtayl\Downloads\team_elo_by_month_ranked_labeled.csv")
    
    # Convert 'Elo' and 'Rank' columns to numeric, forcing errors to NaN
    df['Elo'] = pd.to_numeric(df['Elo'], errors='coerce')
    df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
    
    # Sort the DataFrame
    df.sort_values(by=['Team', 'Month_ID'], inplace=True)
    
    # Create the Season column by extracting the first 4 digits from 'Year/Month'
    df['Season'] = df['Year/Month'].str[:4].astype(int)
    
    # Create the Previous_Elo column
    df['Previous_Elo'] = df.groupby('Team')['Elo'].shift(1)
    df['Previous_Elo'] = df['Previous_Elo'].fillna(1000).astype(int)
    
    # Create the Previous_Rank column
    df['Previous_Rank'] = df.groupby('Team')['Rank'].shift(1)
    df['Previous_Rank'] = df['Previous_Rank'].fillna(0).astype(int)
    
    # Define a function to return the appropriate HTML for the arrow
    def rank_arrow(row):
        if pd.isna(row['Previous_Rank']):
            return ''
        if row['Rank'] < row['Previous_Rank']:
            return '▲'  # Up arrow
        elif row['Rank'] > row['Previous_Rank']:
            return '▼'  # Down arrow
        else:
            return '►'  # Right arrow
    
    # Apply the function to the DataFrame and create a new column for arrows
    df['Rank_Arrow'] = df.apply(rank_arrow, axis=1)
    
    # Create a new DataFrame with the last Elo of each season for each team
    season_end_elo = df.sort_values('Month_ID').groupby(['Team', 'Season'])['Elo'].last().reset_index()
    season_end_elo['Next_Season'] = season_end_elo['Season'] + 1
    
    # Merge the season_end_elo with the original DataFrame
    df = pd.merge(df, season_end_elo[['Team', 'Next_Season', 'Elo']], 
                  left_on=['Team', 'Season'], 
                  right_on=['Team', 'Next_Season'], 
                  how='left', 
                  suffixes=('', '_season_start'))
    
    # Rename the column and drop the unnecessary 'Next_Season' column
    df.rename(columns={'Elo_season_start': 'season_start_elo'}, inplace=True)
    df.drop('Next_Season', axis=1, inplace=True)
    
    # Fill NaN values in season_start_elo with a default value (e.g., 1000)
    df['season_start_elo'] = df['season_start_elo'].fillna(1000).astype(int)
    
    return df

app_ui = ui.page_fluid(
    ui.h2("Latest Rankings"),
    ui.output_data_frame("rankings_table")
)

def server(input, output, session):
    @output
    @render.data_frame
    def rankings_table():
        df = process_data()
        return df.sort_values('Rank').head(20)

app = App(app_ui, server)