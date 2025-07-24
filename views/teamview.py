import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random
import plotly.express as px

# ===================== CACHED TEAM BATTING FUNCTIONS =====================
@st.cache_data
def compute_team_batting_career(df):
    df = df.copy()
    df['50s'] = ((df['Runs'] >= 50) & (df['Runs'] < 100)).astype(int)
    df['100s'] = ((df['Runs'] >= 100) & (df['Runs'] < 150)).astype(int)
    df['150s'] = ((df['Runs'] >= 150) & (df['Runs'] < 200)).astype(int)
    df['200s'] = (df['Runs'] >= 200).astype(int)
    bat_team_career_df = df.groupby('Bat_Team_y').agg({
        'File Name': 'nunique',
        'Batted': 'sum',
        'Out': 'sum',
        'Not Out': 'sum',    
        'Balls': 'sum',
        'Runs': ['sum', 'max'],
        '4s': 'sum',
        '6s': 'sum',
        '50s': 'sum',
        '100s': 'sum',
        '150s': 'sum',
        '200s': 'sum',
        '<25&Out': 'sum',
        'Caught': 'sum',
        'Bowled': 'sum',
        'LBW': 'sum',
        'Run Out': 'sum',
        'Stumped': 'sum',
        'Total_Runs': 'sum',
        'Overs': 'sum',
        'Wickets': 'sum',
        'Team Balls': 'sum'
    }).reset_index()
    bat_team_career_df.columns = ['Team', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 
        'Runs', 'HS', '4s', '6s', '50s', '100s', '150s', '200s', 
        '<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 
        'Team Runs', 'Overs', 'Wickets', 'Team Balls']
    bat_team_career_df['Avg'] = (bat_team_career_df['Runs'] / bat_team_career_df['Out'].replace(0, np.nan)).round(2).fillna(0)
    bat_team_career_df['SR'] = ((bat_team_career_df['Runs'] / bat_team_career_df['Balls'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_career_df['BPO'] = (bat_team_career_df['Balls'] / bat_team_career_df['Out'].replace(0, np.nan)).round(2).fillna(0)
    bat_team_career_df['Team Avg'] = (bat_team_career_df['Team Runs'] / bat_team_career_df['Wickets'].replace(0, np.nan)).round(2).fillna(0)
    bat_team_career_df['Team SR'] = (bat_team_career_df['Team Runs'] / bat_team_career_df['Team Balls'].replace(0, np.nan) * 100).round(2).fillna(0)
    bat_team_career_df['P+ Avg'] = (bat_team_career_df['Avg'] / bat_team_career_df['Team Avg'].replace(0, np.nan) * 100).round(2).fillna(0)
    bat_team_career_df['P+ SR'] = (bat_team_career_df['SR'] / bat_team_career_df['Team SR'].replace(0, np.nan) * 100).round(2).fillna(0)
    bat_team_career_df['BPB'] = (bat_team_career_df['Balls'] / (bat_team_career_df['4s'] + bat_team_career_df['6s']).replace(0, np.nan)).round(2).fillna(0)
    bat_team_career_df['50+PI'] = (((bat_team_career_df['50s'] + bat_team_career_df['100s'] + bat_team_career_df['150s'] + bat_team_career_df['200s']) / bat_team_career_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_career_df['100PI'] = (((bat_team_career_df['100s'] + bat_team_career_df['150s'] + bat_team_career_df['200s']) / bat_team_career_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_career_df['150PI'] = (((bat_team_career_df['150s'] + bat_team_career_df['200s']) / bat_team_career_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_career_df['200PI'] = ((bat_team_career_df['200s'] / bat_team_career_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_career_df['<25&OutPI'] = ((bat_team_career_df['<25&Out'] / bat_team_career_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_career_df['Caught%'] = ((bat_team_career_df['Caught'] / bat_team_career_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_career_df['Bowled%'] = ((bat_team_career_df['Bowled'] / bat_team_career_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_career_df['LBW%'] = ((bat_team_career_df['LBW'] / bat_team_career_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_career_df['Run Out%'] = ((bat_team_career_df['Run Out'] / bat_team_career_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_career_df['Stumped%'] = ((bat_team_career_df['Stumped'] / bat_team_career_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_career_df['Not Out%'] = ((bat_team_career_df['Not Out'] / bat_team_career_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    return bat_team_career_df

@st.cache_data
def compute_team_batting_season(df):
    df = df.copy()
    bat_team_season_df = df.groupby(['Bat_Team_y', 'Year']).agg({
        'File Name': 'nunique',
        'Batted': 'sum',
        'Out': 'sum',
        'Not Out': 'sum',    
        'Balls': 'sum',
        'Runs': ['sum', 'max'],
        '4s': 'sum',
        '6s': 'sum',
        '50s': 'sum',
        '100s': 'sum',
        '200s': 'sum',
        '<25&Out': 'sum',
        'Caught': 'sum',
        'Bowled': 'sum',
        'LBW': 'sum',
        'Run Out': 'sum',
        'Stumped': 'sum',
        'Total_Runs': 'sum',
        'Overs': 'sum',
        'Wickets': 'sum',
        'Team Balls': 'sum'
    }).reset_index()
    bat_team_season_df.columns = ['Team', 'Year', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 
        'Runs', 'HS', '4s', '6s', '50s', '100s', '200s', 
        '<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 
        'Team Runs', 'Overs', 'Wickets', 'Team Balls']
    bat_team_season_df['Avg'] = (bat_team_season_df['Runs'] / bat_team_season_df['Out'].replace(0, np.nan)).round(2).fillna(0)
    bat_team_season_df['SR'] = ((bat_team_season_df['Runs'] / bat_team_season_df['Balls'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_season_df['BPO'] = (bat_team_season_df['Balls'] / bat_team_season_df['Out'].replace(0, np.nan)).round(2).fillna(0)
    bat_team_season_df['Team Avg'] = (bat_team_season_df['Team Runs'] / bat_team_season_df['Wickets'].replace(0, np.nan)).round(2).fillna(0)
    bat_team_season_df['Team SR'] = (bat_team_season_df['Team Runs'] / bat_team_season_df['Team Balls'].replace(0, np.nan) * 100).round(2).fillna(0)
    bat_team_season_df['P+ Avg'] = (bat_team_season_df['Avg'] / bat_team_season_df['Team Avg'].replace(0, np.nan) * 100).round(2).fillna(0)
    bat_team_season_df['P+ SR'] = (bat_team_season_df['SR'] / bat_team_season_df['Team SR'].replace(0, np.nan) * 100).round(2).fillna(0)
    bat_team_season_df['BPB'] = (bat_team_season_df['Balls'] / (bat_team_season_df['4s'] + bat_team_season_df['6s']).replace(0, np.nan)).round(2).fillna(0)
    bat_team_season_df['50+PI'] = (((bat_team_season_df['50s'] + bat_team_season_df['100s']) / bat_team_season_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_season_df['100PI'] = ((bat_team_season_df['100s'] / bat_team_season_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_season_df['<25&OutPI'] = ((bat_team_season_df['<25&Out'] / bat_team_season_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_season_df['Caught%'] = ((bat_team_season_df['Caught'] / bat_team_season_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_season_df['Bowled%'] = ((bat_team_season_df['Bowled'] / bat_team_season_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_season_df['LBW%'] = ((bat_team_season_df['LBW'] / bat_team_season_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_season_df['Run Out%'] = ((bat_team_season_df['Run Out'] / bat_team_season_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_season_df['Stumped%'] = ((bat_team_season_df['Stumped'] / bat_team_season_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_season_df['Not Out%'] = ((bat_team_season_df['Not Out'] / bat_team_season_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    return bat_team_season_df

@st.cache_data
def compute_team_batting_opponent(df):
    df = df.copy()
    bat_team_opponent_df = df.groupby(['Bat_Team_y', 'Bowl_Team_y']).agg({
        'File Name': 'nunique',
        'Batted': 'sum',
        'Out': 'sum',
        'Not Out': 'sum',    
        'Balls': 'sum',
        'Runs': ['sum', 'max'],
        '4s': 'sum',
        '6s': 'sum',
        '50s': 'sum',
        '100s': 'sum',
        '200s': 'sum',
        '<25&Out': 'sum',
        'Caught': 'sum',
        'Bowled': 'sum',
        'LBW': 'sum',
        'Run Out': 'sum',
        'Stumped': 'sum',
        'Total_Runs': 'sum',
        'Overs': 'sum',
        'Wickets': 'sum',
        'Team Balls': 'sum'
    }).reset_index()
    bat_team_opponent_df.columns = ['Team', 'Opponent', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 
        'Runs', 'HS', '4s', '6s', '50s', '100s', '200s', 
        '<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 
        'Team Runs', 'Overs', 'Wickets', 'Team Balls']
    bat_team_opponent_df['Avg'] = (bat_team_opponent_df['Runs'] / bat_team_opponent_df['Out'].replace(0, np.nan)).round(2).fillna(0)
    bat_team_opponent_df['SR'] = ((bat_team_opponent_df['Runs'] / bat_team_opponent_df['Balls'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_opponent_df['BPO'] = (bat_team_opponent_df['Balls'] / bat_team_opponent_df['Out'].replace(0, np.nan)).round(2).fillna(0)
    bat_team_opponent_df['Team Avg'] = (bat_team_opponent_df['Team Runs'] / bat_team_opponent_df['Wickets'].replace(0, np.nan)).round(2).fillna(0)
    bat_team_opponent_df['Team SR'] = (bat_team_opponent_df['Team Runs'] / bat_team_opponent_df['Team Balls'].replace(0, np.nan) * 100).round(2).fillna(0)
    bat_team_opponent_df['P+ Avg'] = (bat_team_opponent_df['Avg'] / bat_team_opponent_df['Team Avg'].replace(0, np.nan) * 100).round(2).fillna(0)
    bat_team_opponent_df['P+ SR'] = (bat_team_opponent_df['SR'] / bat_team_opponent_df['Team SR'].replace(0, np.nan) * 100).round(2).fillna(0)
    bat_team_opponent_df['BPB'] = (bat_team_opponent_df['Balls'] / (bat_team_opponent_df['4s'] + bat_team_opponent_df['6s']).replace(0, np.nan)).round(2).fillna(0)
    bat_team_opponent_df['50+PI'] = (((bat_team_opponent_df['50s'] + bat_team_opponent_df['100s']) / bat_team_opponent_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_opponent_df['100PI'] = ((bat_team_opponent_df['100s'] / bat_team_opponent_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_opponent_df['<25&OutPI'] = ((bat_team_opponent_df['<25&Out'] / bat_team_opponent_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_opponent_df['Caught%'] = ((bat_team_opponent_df['Caught'] / bat_team_opponent_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_opponent_df['Bowled%'] = ((bat_team_opponent_df['Bowled'] / bat_team_opponent_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_opponent_df['LBW%'] = ((bat_team_opponent_df['LBW'] / bat_team_opponent_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_opponent_df['Run Out%'] = ((bat_team_opponent_df['Run Out'] / bat_team_opponent_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_opponent_df['Stumped%'] = ((bat_team_opponent_df['Stumped'] / bat_team_opponent_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_opponent_df['Not Out%'] = ((bat_team_opponent_df['Not Out'] / bat_team_opponent_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    return bat_team_opponent_df

@st.cache_data
def compute_team_batting_location(df):
    df = df.copy()
    bat_team_location_df = df.groupby(['Bat_Team_y', 'Home Team']).agg({
        'File Name': 'nunique',
        'Batted': 'sum',
        'Out': 'sum',
        'Not Out': 'sum',    
        'Balls': 'sum',
        'Runs': ['sum', 'max'],
        '4s': 'sum',
        '6s': 'sum',
        '50s': 'sum',
        '100s': 'sum',
        '200s': 'sum',
        '<25&Out': 'sum',
        'Caught': 'sum',
        'Bowled': 'sum',
        'LBW': 'sum',
        'Run Out': 'sum',
        'Stumped': 'sum',
        'Total_Runs': 'sum',
        'Overs': 'sum',
        'Wickets': 'sum',
        'Team Balls': 'sum'
    }).reset_index()
    bat_team_location_df.columns = ['Team', 'Location', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 
        'Runs', 'HS', '4s', '6s', '50s', '100s', '200s', 
        '<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 
        'Team Runs', 'Overs', 'Wickets', 'Team Balls']
    bat_team_location_df['Avg'] = (bat_team_location_df['Runs'] / bat_team_location_df['Out'].replace(0, np.nan)).round(2).fillna(0)
    bat_team_location_df['SR'] = ((bat_team_location_df['Runs'] / bat_team_location_df['Balls'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_location_df['BPO'] = (bat_team_location_df['Balls'] / bat_team_location_df['Out'].replace(0, np.nan)).round(2).fillna(0)
    bat_team_location_df['Team Avg'] = (bat_team_location_df['Team Runs'] / bat_team_location_df['Wickets'].replace(0, np.nan)).round(2).fillna(0)
    bat_team_location_df['Team SR'] = (bat_team_location_df['Team Runs'] / bat_team_location_df['Team Balls'].replace(0, np.nan) * 100).round(2).fillna(0)
    bat_team_location_df['P+ Avg'] = (bat_team_location_df['Avg'] / bat_team_location_df['Team Avg'].replace(0, np.nan) * 100).round(2).fillna(0)
    bat_team_location_df['P+ SR'] = (bat_team_location_df['SR'] / bat_team_location_df['Team SR'].replace(0, np.nan) * 100).round(2).fillna(0)
    bat_team_location_df['BPB'] = (bat_team_location_df['Balls'] / (bat_team_location_df['4s'] + bat_team_location_df['6s']).replace(0, np.nan)).round(2).fillna(0)
    bat_team_location_df['50+PI'] = (((bat_team_location_df['50s'] + bat_team_location_df['100s']) / bat_team_location_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_location_df['100PI'] = ((bat_team_location_df['100s'] / bat_team_location_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_location_df['<25&OutPI'] = ((bat_team_location_df['<25&Out'] / bat_team_location_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_location_df['Caught%'] = ((bat_team_location_df['Caught'] / bat_team_location_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_location_df['Bowled%'] = ((bat_team_location_df['Bowled'] / bat_team_location_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_location_df['LBW%'] = ((bat_team_location_df['LBW'] / bat_team_location_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_location_df['Run Out%'] = ((bat_team_location_df['Run Out'] / bat_team_location_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_location_df['Stumped%'] = ((bat_team_location_df['Stumped'] / bat_team_location_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_location_df['Not Out%'] = ((bat_team_location_df['Not Out'] / bat_team_location_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    return bat_team_location_df

@st.cache_data
def compute_team_batting_position(df):
    df = df.copy()
    bat_team_position_df = df.groupby(['Bat_Team_y', 'Position']).agg({
        'File Name': 'nunique',
        'Batted': 'sum',
        'Out': 'sum',
        'Not Out': 'sum',    
        'Balls': 'sum',
        'Runs': ['sum', 'max'],
        '4s': 'sum',
        '6s': 'sum',
        '50s': 'sum',
        '100s': 'sum',
        '200s': 'sum',
        '<25&Out': 'sum',
        'Caught': 'sum',
        'Bowled': 'sum',
        'LBW': 'sum',
        'Run Out': 'sum',
        'Stumped': 'sum',
        'Total_Runs': 'sum',
        'Overs': 'sum',
        'Wickets': 'sum',
        'Team Balls': 'sum'
    }).reset_index()
    bat_team_position_df.columns = ['Team', 'Position', 'Matches', 'Inns', 'Out', 'Not Out', 'Balls', 
        'Runs', 'HS', '4s', '6s', '50s', '100s', '200s', 
        '<25&Out', 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 
        'Team Runs', 'Overs', 'Wickets', 'Team Balls']
    bat_team_position_df['Avg'] = (bat_team_position_df['Runs'] / bat_team_position_df['Out'].replace(0, np.nan)).round(2).fillna(0)
    bat_team_position_df['SR'] = ((bat_team_position_df['Runs'] / bat_team_position_df['Balls'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_position_df['BPO'] = (bat_team_position_df['Balls'] / bat_team_position_df['Out'].replace(0, np.nan)).round(2).fillna(0)
    bat_team_position_df['Team Avg'] = (bat_team_position_df['Team Runs'] / bat_team_position_df['Wickets'].replace(0, np.nan)).round(2).fillna(0)
    bat_team_position_df['Team SR'] = (bat_team_position_df['Team Runs'] / bat_team_position_df['Team Balls'].replace(0, np.nan) * 100).round(2).fillna(0)
    bat_team_position_df['P+ Avg'] = (bat_team_position_df['Avg'] / bat_team_position_df['Team Avg'].replace(0, np.nan) * 100).round(2).fillna(0)
    bat_team_position_df['P+ SR'] = (bat_team_position_df['SR'] / bat_team_position_df['Team SR'].replace(0, np.nan) * 100).round(2).fillna(0)
    bat_team_position_df['BPB'] = (bat_team_position_df['Balls'] / (bat_team_position_df['4s'] + bat_team_position_df['6s']).replace(0, np.nan)).round(2).fillna(0)
    bat_team_position_df['50+PI'] = (((bat_team_position_df['50s'] + bat_team_position_df['100s']) / bat_team_position_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_position_df['100PI'] = ((bat_team_position_df['100s'] / bat_team_position_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_position_df['<25&OutPI'] = ((bat_team_position_df['<25&Out'] / bat_team_position_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_position_df['Caught%'] = ((bat_team_position_df['Caught'] / bat_team_position_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_position_df['Bowled%'] = ((bat_team_position_df['Bowled'] / bat_team_position_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_position_df['LBW%'] = ((bat_team_position_df['LBW'] / bat_team_position_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_position_df['Run Out%'] = ((bat_team_position_df['Run Out'] / bat_team_position_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_position_df['Stumped%'] = ((bat_team_position_df['Stumped'] / bat_team_position_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    bat_team_position_df['Not Out%'] = ((bat_team_position_df['Not Out'] / bat_team_position_df['Inns'].replace(0, np.nan)) * 100).round(2).fillna(0)
    return bat_team_position_df

# ===================== CACHED TEAM BOWLING FUNCTIONS =====================
@st.cache_data
def compute_team_bowling_career(df):
    if df.empty:
        return pd.DataFrame()
    bowl_team_df = df.groupby('Bowl_Team').agg({
        'File Name': 'nunique', 'Bowler_Balls': 'sum', 'Maidens': 'sum',
        'Bowler_Runs': 'sum', 'Bowler_Wkts': 'sum'
    }).reset_index()
    bowl_team_df.columns = ['Bowl_Team', 'Matches', 'Balls', 'M/D', 'Runs', 'Wickets']
    bowl_team_df['Overs'] = ((bowl_team_df['Balls'] // 6) + (bowl_team_df['Balls'] % 6) / 10).round(1)
    wickets_safe = bowl_team_df['Wickets'].replace(0, np.nan)
    overs_safe = bowl_team_df['Overs'].replace(0, np.nan)
    bowl_team_df['Strike Rate'] = (bowl_team_df['Balls'] / wickets_safe).round(2).fillna(0)
    bowl_team_df['Economy Rate'] = (bowl_team_df['Runs'] / overs_safe).round(2).fillna(0)
    bowl_team_df['Avg'] = (bowl_team_df['Runs'] / wickets_safe).round(2).fillna(0)
    five_wickets = df[df['Bowler_Wkts'] >= 5].groupby('Bowl_Team').size().reset_index(name='5W')
    bowl_team_df = bowl_team_df.merge(five_wickets, on='Bowl_Team', how='left').fillna({'5W': 0})
    match_wickets = df.groupby(['Bowl_Team', 'File Name'])['Bowler_Wkts'].sum().reset_index()
    ten_wickets = match_wickets[match_wickets['Bowler_Wkts'] >= 10].groupby('Bowl_Team').size().reset_index(name='10W')
    bowl_team_df = bowl_team_df.merge(ten_wickets, on='Bowl_Team', how='left').fillna({'10W': 0})
    bowl_team_df['WPM'] = (bowl_team_df['Wickets'] / bowl_team_df['Matches'].replace(0, np.nan)).round(2).fillna(0)
    return bowl_team_df.sort_values(by='Avg')

@st.cache_data
def compute_team_bowling_season(df):
    if df.empty:
        return pd.DataFrame()
    bowl_team_season_df = df.groupby(['Bowl_Team', 'Year']).agg({
        'File Name': 'nunique', 'Bowler_Balls': 'sum', 'Maidens': 'sum',
        'Bowler_Runs': 'sum', 'Bowler_Wkts': 'sum'
    }).reset_index()
    bowl_team_season_df.columns = ['Bowl_Team', 'Year', 'Matches', 'Balls', 'M/D', 'Runs', 'Wickets']
    bowl_team_season_df['Overs'] = ((bowl_team_season_df['Balls'] // 6) + (bowl_team_season_df['Balls'] % 6) / 10).round(1)
    wickets_safe = bowl_team_season_df['Wickets'].replace(0, np.nan)
    overs_safe = bowl_team_season_df['Overs'].replace(0, np.nan)
    bowl_team_season_df['Strike Rate'] = (bowl_team_season_df['Balls'] / wickets_safe).round(2).fillna(0)
    bowl_team_season_df['Economy Rate'] = (bowl_team_season_df['Runs'] / overs_safe).round(2).fillna(0)
    bowl_team_season_df['Avg'] = (bowl_team_season_df['Runs'] / wickets_safe).round(2).fillna(0)
    return bowl_team_season_df.sort_values(by=['Year', 'Avg'], ascending=[False, True])

@st.cache_data
def compute_team_bowling_opponent(df):
    if df.empty:
        return pd.DataFrame()
    opponent_summary = df.groupby(['Bowl_Team', 'Bat_Team']).agg({
        'File Name': 'nunique', 'Bowler_Balls': 'sum', 'Maidens': 'sum',
        'Bowler_Runs': 'sum', 'Bowler_Wkts': 'sum'
    }).reset_index()
    opponent_summary.columns = ['Bowl_Team', 'Opposition', 'Matches', 'Balls', 'M/D', 'Runs', 'Wickets']
    opponent_summary['Overs'] = ((opponent_summary['Balls'] // 6) + (opponent_summary['Balls'] % 6) / 10).round(1)
    wickets_safe = opponent_summary['Wickets'].replace(0, np.nan)
    overs_safe = opponent_summary['Overs'].replace(0, np.nan)
    opponent_summary['Strike Rate'] = (opponent_summary['Balls'] / wickets_safe).round(2).fillna(0)
    opponent_summary['Economy Rate'] = (opponent_summary['Runs'] / overs_safe).round(2).fillna(0)
    opponent_summary['Avg'] = (opponent_summary['Runs'] / wickets_safe).round(2).fillna(0)
    return opponent_summary.sort_values(by=['Bowl_Team', 'Avg'])

@st.cache_data
def compute_team_bowling_location(df):
    if df.empty:
        return pd.DataFrame()
    location_summary = df.groupby(['Bowl_Team', 'Home_Team']).agg({
        'File Name': 'nunique', 'Bowler_Balls': 'sum', 'Maidens': 'sum',
        'Bowler_Runs': 'sum', 'Bowler_Wkts': 'sum'
    }).reset_index()
    location_summary.columns = ['Team', 'Location', 'Matches', 'Balls', 'M/D', 'Runs', 'Wickets']
    location_summary['Overs'] = ((location_summary['Balls'] // 6) + (location_summary['Balls'] % 6) / 10).round(1)
    wickets_safe = location_summary['Wickets'].replace(0, np.nan)
    overs_safe = location_summary['Overs'].replace(0, np.nan)
    location_summary['Strike Rate'] = (location_summary['Balls'] / wickets_safe).round(2).fillna(0)
    location_summary['Economy Rate'] = (location_summary['Runs'] / overs_safe).round(2).fillna(0)
    location_summary['Avg'] = (location_summary['Runs'] / wickets_safe).round(2).fillna(0)
    return location_summary.sort_values(by=['Team', 'Avg'])

@st.cache_data
def compute_team_bowling_position(df):
    if df.empty:
        return pd.DataFrame()
    position_summary = df.groupby(['Bowl_Team', 'Position']).agg({
        'File Name': 'nunique', 'Bowler_Balls': 'sum', 'Maidens': 'sum',
        'Bowler_Runs': 'sum', 'Bowler_Wkts': 'sum'
    }).reset_index()
    position_summary.columns = ['Bowl_Team', 'Position', 'Matches', 'Balls', 'M/D', 'Runs', 'Wickets']
    position_summary['Overs'] = ((position_summary['Balls'] // 6) + (position_summary['Balls'] % 6) / 10).round(1)
    wickets_safe = position_summary['Wickets'].replace(0, np.nan)
    overs_safe = position_summary['Overs'].replace(0, np.nan)
    position_summary['Strike Rate'] = (position_summary['Balls'] / wickets_safe).round(2).fillna(0)
    position_summary['Economy Rate'] = (position_summary['Runs'] / overs_safe).round(2).fillna(0)
    position_summary['Avg'] = (position_summary['Runs'] / wickets_safe).round(2).fillna(0)
    return position_summary.sort_values(by=['Bowl_Team', 'Position'])

def parse_date(date_str):
    """Helper function to parse dates in multiple formats"""
    try:
        for fmt in ['%d/%m/%Y', '%d %b %Y', '%Y-%m-%d']:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except ValueError:
                continue
        return pd.to_datetime(date_str)
    except Exception:
        return pd.NaT

def display_team_view():
    # Beautiful main title with purple gradient background banner
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px 20px; border-radius: 20px; text-align: center; 
                margin-bottom: 2rem; box-shadow: 0 8px 25px rgba(0,0,0,0.15);'>
        <h1 style='color: white; font-size: 2.2rem; font-weight: 700; margin: 0 0 8px 0;'>
            🏏 Team Statistics
        </h1>
        <p style='color: white; font-size: 1rem; margin: 0; opacity: 0.9;'>
            Cricket team analysis system based on comprehensive performance metrics
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Modern CSS styling for beautiful UI
    st.markdown("""
    <style>
    /* Global styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Filter controls */
    .stMultiSelect label {
        color: #f04f53 !important;
        font-weight: 500;
        font-size: 1rem;
    }
    
    .stSlider label {
        color: #f04f53 !important;
        font-weight: 500;
    }
    
    /* Slider track styling - red color */
    .stSlider [data-baseweb="slider-track"] {
        background: linear-gradient(90deg, #f04f53 0%, #f04f53 100%) !important;
    }
    
    /* Table styling with modern look */
    table { 
        color: black; 
        width: 100%; 
        border-collapse: collapse;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-radius: 8px;
        overflow: hidden;
    }
    
    thead tr th {
        background: linear-gradient(135deg, #f04f53 0%, #e03a3e 100%) !important;
        color: white !important;
        font-weight: 600;
        text-align: center;
        padding: 12px 8px;
        border: none;
    }
    
    tbody tr:nth-child(even) { 
        background-color: #f8f9fa; 
    }
    
    tbody tr:nth-child(odd) { 
        background-color: white; 
    }
    
    tbody tr:hover {
        background-color: #e8f4fd !important;
        transition: background-color 0.3s ease;
    }
    
    tbody td {
        padding: 10px 8px;
        border-bottom: 1px solid #dee2e6;
        text-align: center;
    }
    
    /* Tab styling with beautiful gradients - matching playerrankings.py */
    .stTabs [data-baseweb="tab-list"] {
        width: 100%;
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 15px;
        padding: 12px;
        box-shadow: 0 8px 32px rgba(168, 237, 234, 0.3);
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        flex-grow: 1;
        text-align: center;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        margin: 0 6px;
        transition: all 0.4s ease;
        color: #2c3e50 !important;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 12px 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(255, 255, 255, 0.2);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #f04f53 0%, #f5576c 100%);
        color: white !important;
        box-shadow: 0 6px 20px rgba(240, 79, 83, 0.4);
        transform: translateY(-3px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"]:hover {
        background: linear-gradient(135deg, #e03a3e 0%, #f04f53 100%);
    }
    
    /* Card-style containers */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    
    /* Section headers with gradients */
    .section-header {
        background: linear-gradient(135deg, #f04f53 0%, #e03a3e 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(240, 79, 83, 0.3);
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    .career-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    .season-header {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    .opposition-header {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    .location-header {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #2c3e50;
        padding: 1rem 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(168, 237, 234, 0.3);
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    .position-header {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #2c3e50;
        padding: 1rem 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(255, 154, 158, 0.3);
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    .bowling-header {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #2c3e50;
        padding: 1rem 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(255, 236, 210, 0.3);
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    .chart-header {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #2c3e50;
        padding: 1rem 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(132, 250, 176, 0.3);
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    /* Dataframe container styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Custom multiselect styling */
    .stMultiSelect > div > div > div {
        border-radius: 8px;
        border: 2px solid #e9ecef;
        transition: border-color 0.3s ease;
    }
    
    .stMultiSelect > div > div > div:focus-within {
        border-color: #f04f53;
        box-shadow: 0 0 0 3px rgba(240, 79, 83, 0.1);
    }
    
    /* Filter container styling - transparent background */
    .filter-container {
        background: transparent;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Check if required DataFrames exist in session state
    if 'bat_df' in st.session_state and 'bowl_df' in st.session_state:
        bat_df = st.session_state['bat_df'].copy()
        bowl_df = st.session_state['bowl_df'].copy()

        # Convert dates to datetime with safer parsing
        try:
            # Try to parse dates with the correct format
            bat_df['Date'] = pd.to_datetime(bat_df['Date'], format='%d %b %Y', errors='coerce')
            bowl_df['Date'] = pd.to_datetime(bowl_df['Date'], format='%d %b %Y', errors='coerce')
        except:
            # If that fails, try with dayfirst=True
            bat_df['Date'] = pd.to_datetime(bat_df['Date'], dayfirst=True, errors='coerce')
            bowl_df['Date'] = pd.to_datetime(bowl_df['Date'], dayfirst=True, errors='coerce')

        # Extract years from the parsed dates
        bat_df['Year'] = bat_df['Date'].dt.year
        bowl_df['Year'] = bowl_df['Date'].dt.year

        # Convert Year columns to integers and handle any NaN values
        bat_df['Year'] = pd.to_numeric(bat_df['Year'], errors='coerce').fillna(0).astype(int)
        bowl_df['Year'] = pd.to_numeric(bowl_df['Year'], errors='coerce').fillna(0).astype(int)

        # Get filter options (exclude year 0 from the years list)
        match_formats = ['All'] + sorted(bat_df['Match_Format'].unique().tolist())
        years = sorted(list(set(bat_df['Year'].unique()) | set(bowl_df['Year'].unique())))
        years = [year for year in years if year != 0]  # Remove year 0 if present
        
        if not years:
            years = [pd.Timestamp.now().year]

        # Get unique teams
        bat_teams = ['All'] + sorted(set(list(bat_df['Bat_Team_y'].unique()) + list(bowl_df['Bat_Team'].unique())))
        bowl_teams = ['All'] + sorted(set(list(bat_df['Bowl_Team_y'].unique()) + list(bowl_df['Bowl_Team'].unique())))

        # Modern Filter Controls with styling
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        
        # Create the filters row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("🏏 **Bat Team**", unsafe_allow_html=True)
            bat_team_choice = st.multiselect('', bat_teams, default='All', key='bat_team_filter', label_visibility='collapsed')

        with col2:
            st.markdown("🏏 **Bowl Team**", unsafe_allow_html=True)
            bowl_team_choice = st.multiselect('', bowl_teams, default='All', key='bowl_team_filter', label_visibility='collapsed')

        with col3:
            st.markdown("🏆 **Format**", unsafe_allow_html=True)
            match_format_choice = st.multiselect('', match_formats, default='All', key='match_format_filter', label_visibility='collapsed')

        with col4:
            st.markdown("📅 **Year**", unsafe_allow_html=True)
            if len(years) == 1:
                st.markdown(f"<p style='text-align: center; color: #f04f53; font-weight: 500;'>{years[0]}</p>", unsafe_allow_html=True)
                year_choice = (years[0], years[0])
            else:
                year_choice = st.slider('',
                                    min_value=min(years),
                                    max_value=max(years),
                                    value=(min(years), max(years)),
                                    key='year_slider',
                                    label_visibility='collapsed')
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Create filtered DataFrames based on selections
        filtered_bat_df = bat_df.copy()
        filtered_bowl_df = bowl_df.copy()

        # Apply filters
        if 'All' not in bat_team_choice:
            filtered_bat_df = filtered_bat_df[filtered_bat_df['Bat_Team_y'].isin(bat_team_choice)]
            filtered_bowl_df = filtered_bowl_df[filtered_bowl_df['Bat_Team'].isin(bat_team_choice)]

        if 'All' not in bowl_team_choice:
            filtered_bat_df = filtered_bat_df[filtered_bat_df['Bowl_Team_y'].isin(bowl_team_choice)]
            filtered_bowl_df = filtered_bowl_df[filtered_bowl_df['Bowl_Team'].isin(bowl_team_choice)]

        if 'All' not in match_format_choice:
            filtered_bat_df = filtered_bat_df[filtered_bat_df['Match_Format'].isin(match_format_choice)]
            filtered_bowl_df = filtered_bowl_df[filtered_bowl_df['Match_Format'].isin(match_format_choice)]

        filtered_bat_df = filtered_bat_df[filtered_bat_df['Year'].between(year_choice[0], year_choice[1])]
        filtered_bowl_df = filtered_bowl_df[filtered_bowl_df['Year'].between(year_choice[0], year_choice[1])]


        # Create filtered DataFrames based on selections
        filtered_bat_df = bat_df.copy()
        filtered_bowl_df = bowl_df.copy()

        # Apply filters
        if 'All' not in bat_team_choice:
            filtered_bat_df = filtered_bat_df[filtered_bat_df['Bat_Team_y'].isin(bat_team_choice)]
            filtered_bowl_df = filtered_bowl_df[filtered_bowl_df['Bat_Team'].isin(bat_team_choice)]

        if 'All' not in bowl_team_choice:
            filtered_bat_df = filtered_bat_df[filtered_bat_df['Bowl_Team_y'].isin(bowl_team_choice)]
            filtered_bowl_df = filtered_bowl_df[filtered_bowl_df['Bowl_Team'].isin(bowl_team_choice)]

        if 'All' not in match_format_choice:
            filtered_bat_df = filtered_bat_df[filtered_bat_df['Match_Format'].isin(match_format_choice)]
            filtered_bowl_df = filtered_bowl_df[filtered_bowl_df['Match_Format'].isin(match_format_choice)]

        filtered_bat_df = filtered_bat_df[filtered_bat_df['Year'].between(year_choice[0], year_choice[1])]
        filtered_bowl_df = filtered_bowl_df[filtered_bowl_df['Year'].between(year_choice[0], year_choice[1])]


        # Create tabs
        tab1, tab2, tab3 = st.tabs(["🏏 Batting Statistics", "🎳 Bowling Statistics", "🏆 Rank"])

    # Batting  Statistics Tab
    with tab1:
        # Create subtabs for different batting statistics views
        batting_subtabs = st.tabs(["🏏 Career", "📅 Season", "⚔️ Opposition", "🌍 Location", "📍 Position"])
        
        # Career Statistics Subtab
        with batting_subtabs[0]:
            bat_team_career_df = compute_team_batting_career(filtered_bat_df)
            if bat_team_career_df.empty:
                st.info("No team career batting statistics to display for the current selection.")
            else:
                st.markdown('<div class="career-header">🏏 Team Career Statistics</div>', unsafe_allow_html=True)
                st.markdown(
                    """
                    <style>
                    /* Make the first column of any table sticky */
                    .stDataFrame table tbody tr :first-child, 
                    .stDataFrame table thead tr :first-child {
                        position: sticky;
                        left: 0;
                        background: white;
                        z-index: 1;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                st.dataframe(
                    bat_team_career_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Team": st.column_config.Column("Team", pinned=True)
                    }
                )
                # Create a new figure for the scatter plot
                scatter_fig = go.Figure()
                for team in bat_team_career_df['Team'].unique():
                    team_stats = bat_team_career_df[bat_team_career_df['Team'] == team]
                    batting_avg = team_stats['Avg'].iloc[0]
                    strike_rate = team_stats['SR'].iloc[0]
                    runs = team_stats['Runs'].iloc[0]
                    scatter_fig.add_trace(go.Scatter(
                        x=[batting_avg],
                        y=[strike_rate],
                        mode='markers+text',
                        text=[team],
                        textposition='top center',
                        marker=dict(size=10),
                        name=team,
                        hovertemplate=(
                            f"{team}<br>"
                            f"Batting Average: {batting_avg:.2f}<br>"
                            f"Strike Rate: {strike_rate:.2f}<br>"
                            f"Runs: {runs}<br>"
                        )
                    ))
                st.markdown('<div class="chart-header">📊 Team Batting Average vs Strike Rate</div>', unsafe_allow_html=True)
                scatter_fig.update_layout(
                    xaxis_title="Batting Average",
                    yaxis_title="Strike Rate",
                    height=500,
                    font=dict(size=12),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False
                )
                st.plotly_chart(scatter_fig, use_container_width=True)

        # Season Stats Subtab
        with batting_subtabs[1]:
            bat_team_season_df = compute_team_batting_season(filtered_bat_df)
            if bat_team_season_df.empty:
                st.info("No team season batting statistics to display for the current selection.")
            else:
                st.markdown('<div class="season-header">📅 Season Statistics</div>', unsafe_allow_html=True)
                st.dataframe(
                    bat_team_season_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Team": st.column_config.Column("Team", pinned=True)
                    }
                )
                scatter_fig = go.Figure()
                for _, row in bat_team_season_df.iterrows():
                    scatter_fig.add_trace(go.Scatter(
                        x=[row['Avg']],
                        y=[row['SR']],
                        mode='markers+text',
                        text=[f"{row['Team']} ({row['Year']})"],
                        textposition='top center', 
                        marker=dict(size=10),
                        name=f"{row['Team']} {row['Year']}",
                        hovertemplate=(
                            f"{row['Team']} ({row['Year']})<br>"
                            f"Batting Average: {row['Avg']:.3f}<br>"
                            f"Strike Rate: {row['SR']:.2f}<br>"
                        )
                    ))
                st.markdown('<div class="chart-header">📈 Team Batting Average vs Strike Rate Season</div>', unsafe_allow_html=True)
                scatter_fig.update_layout(
                    xaxis_title="Batting Average",
                    yaxis_title="Strike Rate",
                    height=500,
                    font=dict(size=12),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False
                )
                st.plotly_chart(scatter_fig, key="team_season_scatter", use_container_width=True)

        # Opposition Stats Subtab
        with batting_subtabs[2]:
            bat_team_opponent_df = compute_team_batting_opponent(filtered_bat_df)
            if bat_team_opponent_df.empty:
                st.info("No team opposition batting statistics to display for the current selection.")
            else:
                st.markdown('<div class="opposition-header">⚔️ Opposition Statistics</div>', unsafe_allow_html=True)
                st.dataframe(
                    bat_team_opponent_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Team": st.column_config.Column("Team", pinned=True)
                    }
                )
                opponent_stats_df = filtered_bat_df.groupby('Bowl_Team_y').agg({
                    'Runs': 'sum',
                    'Out': 'sum',
                    'Balls': 'sum'
                }).reset_index()
                opponent_stats_df['Avg'] = (opponent_stats_df['Runs'] / opponent_stats_df['Out'].replace(0, np.nan)).round(2)
                opponent_stats_df['SR'] = ((opponent_stats_df['Runs'] / opponent_stats_df['Balls'].replace(0, np.nan)) * 100).round(2)
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=opponent_stats_df['Bowl_Team_y'], 
                        y=opponent_stats_df['Avg'], 
                        name='Average', 
                        marker_color='#f84e4e',
                        text=opponent_stats_df['Avg'].round(2),
                        textposition='auto'
                    )
                )
                fig.update_layout(
                    showlegend=True,
                    height=500,
                    xaxis_title="Opposition Team",
                    yaxis_title="Average",
                    font=dict(size=12),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis={'categoryorder': 'total ascending'}
                )
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                st.plotly_chart(fig, use_container_width=True)

        # Location Stats Subtab
        with batting_subtabs[3]:
            bat_team_location_df = compute_team_batting_location(filtered_bat_df)
            if bat_team_location_df.empty:
                st.info("No team location batting statistics to display for the current selection.")
            else:
                st.markdown('<div class="location-header">🌍 Location Statistics</div>', unsafe_allow_html=True)
                st.dataframe(
                    bat_team_location_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Team": st.column_config.Column("Team", pinned=True)
                    }
                )
                location_stats_df = bat_team_location_df.groupby('Location').agg({
                    'Runs': 'sum',
                    'Out': 'sum',
                    'Balls': 'sum'
                }).reset_index()
                location_stats_df['Avg'] = (location_stats_df['Runs'] / location_stats_df['Out'].replace(0, np.nan)).round(2)
                location_stats_df['SR'] = ((location_stats_df['Runs'] / location_stats_df['Balls'].replace(0, np.nan)) * 100).round(2)
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=location_stats_df['Location'], 
                        y=location_stats_df['Avg'], 
                        name='Average', 
                        marker_color='#f84e4e',
                        text=location_stats_df['Avg'].round(2),
                        textposition='auto'
                    )
                )
                fig.update_layout(
                    showlegend=False,
                    height=500,
                    xaxis_title='Location',
                    yaxis_title='Average',
                    font=dict(size=12),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis={'categoryorder': 'total ascending'}
                )
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                st.plotly_chart(fig, use_container_width=True)

        # Position Stats Subtab
        with batting_subtabs[4]:
            bat_team_position_df = compute_team_batting_position(filtered_bat_df)
            if bat_team_position_df.empty:
                st.info("No team position batting statistics to display for the current selection.")
            else:
                st.markdown('<div class="position-header">📍 Position Statistics</div>', unsafe_allow_html=True)
                st.dataframe(
                    bat_team_position_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Team": st.column_config.Column("Team", pinned=True)
                    }
                )
                fig = go.Figure()
                position_stats_df = bat_team_position_df.groupby('Position').agg({
                    'Runs': 'sum',
                    'Out': 'sum',
                    'Inns': 'sum'
                }).reset_index()
                position_stats_df['Avg'] = (position_stats_df['Runs'] / position_stats_df['Out'].replace(0, np.nan)).round(2)
                position_stats_df['Position'] = position_stats_df['Position'].astype(int)
                position_stats_df = position_stats_df.sort_values('Position')
                fig.add_trace(
                    go.Bar(
                        y=position_stats_df['Position'],
                        x=position_stats_df['Avg'],
                        orientation='h',
                        name='Average', 
                        marker_color='#f84e4e',
                        text=position_stats_df['Avg'].round(2),
                        textposition='auto'
                    )
                )
                fig.update_layout(
                    showlegend=False,
                    height=500,
                    yaxis_title='Position',
                    xaxis_title='Average',
                    font=dict(size=12),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis={
                        'categoryorder':'array',
                        'categoryarray': sorted(position_stats_df['Position'].unique()),
                        'dtick': 1
                    }
                )
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                st.plotly_chart(fig, use_container_width=True)

###-------------------------------------BOWLING STATS-------------------------------------###

    # Bowling Statistics Tab
    with tab2:
        bowling_subtabs = st.tabs(["🎳 Career", "📅 Season", "⚔️ Opposition", "🌍 Location", "👥 Team Position"])
        # Career Statistics Subtab
        with bowling_subtabs[0]:
            bowl_team_df = compute_team_bowling_career(filtered_bowl_df)
            st.markdown('<div class="bowling-header">🎳 Team Bowling Career Statistics</div>', unsafe_allow_html=True)
            if bowl_team_df.empty:
                st.info("No career bowling stats to display.")
            else:
                st.dataframe(bowl_team_df, use_container_width=True, hide_index=True)
        # Season Stats Subtab
        with bowling_subtabs[1]:
            bowl_team_season_df = compute_team_bowling_season(filtered_bowl_df)
            st.markdown('<div class="season-header">📅 Team Bowling Season Stats</div>', unsafe_allow_html=True)
            if bowl_team_season_df.empty:
                st.info("No season bowling stats to display.")
            else:
                st.dataframe(bowl_team_season_df, use_container_width=True, hide_index=True)
        # Opposition Stats Subtab
        with bowling_subtabs[2]:
            bowl_team_opponent_df = compute_team_bowling_opponent(filtered_bowl_df)
            st.markdown('<div class="opposition-header">⚔️ Opposition Statistics</div>', unsafe_allow_html=True)
            if bowl_team_opponent_df.empty:
                st.info("No opposition bowling stats to display.")
            else:
                st.dataframe(bowl_team_opponent_df, use_container_width=True, hide_index=True)
        # Location Stats Subtab
        with bowling_subtabs[3]:
            location_summary = compute_team_bowling_location(filtered_bowl_df)
            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Location Statistics</h3>", unsafe_allow_html=True)
            if location_summary.empty:
                st.info("No location bowling stats to display.")
            else:
                st.dataframe(location_summary, use_container_width=True, hide_index=True)
        # Team Position Stats Subtab
        with bowling_subtabs[4]:
            team_position_summary = compute_team_bowling_position(filtered_bowl_df)
            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Team Position Statistics</h3>", unsafe_allow_html=True)
            if team_position_summary.empty:
                st.info("No position bowling stats to display.")
            else:
                st.dataframe(team_position_summary, use_container_width=True, hide_index=True)

    # Rank Tab - Combined Season Stats
    with tab3:
        st.markdown('<div class="section-header">🏆 Team Season Rankings</div>', unsafe_allow_html=True)
        
        with st.expander("Understanding Performance Metrics", expanded=False):
            st.markdown("""
            ### Performance Metrics Explained
            
            #### Performance Index
            The Performance Index is a balanced metric that compares a team's performance to format averages:
            - **Formula**: 100 × (Batting Avg ÷ Format Batting Mean) + 100 × (Format Bowling Mean ÷ Bowling Avg)
            - **Higher is better**: Teams with higher performance index have better overall performance
            - **Interpretation**: 
              - 200 represents a perfectly balanced team at format average
              - > 220 indicates title-contending teams
              - 190-220 indicates solid, competitive teams
              - < 190 indicates teams lagging in one or both disciplines
            
            #### Avg Difference
            The Average Difference is simply: Batting Average - Bowling Average
            - **Higher is better**: A positive value means a team's batters score more runs per wicket than their bowlers concede
            - **Interpretation**:
              - Positive values indicate teams that generally win matches
              - Negative values suggest teams that struggle to win consistently
            """)
        
        # --- START OF OPTIMIZED CODE ---
        # Call the cached functions directly instead of recalculating
        bat_team_season_df = compute_team_batting_season(filtered_bat_df)
        bowl_team_season_df = compute_team_bowling_season(filtered_bowl_df)
        
        # Check if the necessary dataframes are available
        if bat_team_season_df.empty or bowl_team_season_df.empty:
            st.warning("Not enough seasonal data for both batting and bowling to generate rankings.")
        else:
            # Call the cached ranking function
            combined_df = compute_team_rankings(bat_team_season_df, bowl_team_season_df)
            
            # Display the styled dataframe
            st.dataframe(
                combined_df,  # or whatever your rankings DataFrame variable is
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Batting Avg": st.column_config.NumberColumn("Batting Avg", format="%.2f"),
                    "Bowling Avg": st.column_config.NumberColumn("Bowling Avg", format="%.2f"),
                    "Batting SR": st.column_config.NumberColumn("Batting SR", format="%.2f"),
                    "Bowling SR": st.column_config.NumberColumn("Bowling SR", format="%.2f"),
                    "Performance Index": st.column_config.NumberColumn("Performance Index", format="%.2f"),
                    # ...other columns as needed...
                }
            )
            
            # Display the visualizations
            st.markdown('<div class="chart-header">📊 Performance Visualizations</div>', unsafe_allow_html=True)
            viz_tabs = st.tabs(["📈 Performance Index", "⚖️ Average Difference"])

            with viz_tabs[0]:
                # ... your performance index plot logic ...
                available_years = sorted(combined_df['Year'].unique(), reverse=True)
                if available_years:
                    selected_year = st.selectbox(
                        "Select Year", 
                        available_years,
                        index=0,
                        key="perf_index_year"
                    )
                    year_data = combined_df[combined_df['Year'] == selected_year].sort_values('Performance Index', ascending=False)
                    if not year_data.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=year_data['Team'],
                            y=year_data['Performance Index'],
                            marker_color='#f84e4e',
                            text=year_data['Performance Index'].round(2),
                            textposition='auto',
                            hovertemplate=(
                                "Team: %{x}<br>"
                                "Performance Index: %{y:.2f}<br>"
                                "<extra></extra>"
                            )
                        ))
                        fig.update_layout(
                            title=f"Team Performance Index for {selected_year}",
                            showlegend=False,
                            height=500,
                            xaxis_title="Team",
                            yaxis_title="Performance Index",
                            font=dict(size=12),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                        )
                        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                        st.plotly_chart(fig, use_container_width=True)

            with viz_tabs[1]:
                # ... your average difference plot logic ...
                if available_years:
                    selected_year_diff = st.selectbox(
                        "Select Year", 
                        available_years,
                        index=0,
                        key="avg_diff_year"
                    )
                    year_data_diff = combined_df[combined_df['Year'] == selected_year_diff].sort_values('Avg Difference', ascending=False)
                    if not year_data_diff.empty:
                        fig_diff = go.Figure()
                        colors = ['#f84e4e' if x >= 0 else '#4e8ff8' for x in year_data_diff['Avg Difference']]
                        fig_diff.add_trace(go.Bar(
                            x=year_data_diff['Team'],
                            y=year_data_diff['Avg Difference'],
                            marker_color=colors,
                            text=year_data_diff['Avg Difference'].round(2),
                            textposition='auto',
                            hovertemplate=(
                                "Team: %{x}<br>"
                                "Average Difference: %{y:.2f}<br>"
                                "Batting Average: %{customdata[0]:.2f}<br>"
                                "Bowling Average: %{customdata[1]:.2f}<br>"
                                "<extra></extra>"
                            ),
                            customdata=np.stack((year_data_diff['Batting Avg'], year_data_diff['Bowling Avg']), axis=-1)
                        ))
                        fig_diff.update_layout(
                            title=f"Average Difference (Batting Avg - Bowling Avg) for {selected_year_diff}",
                            showlegend=False,
                            height=500,
                            xaxis_title="Team",
                            yaxis_title="Average Difference",
                            font=dict(size=12),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                        )
                        fig_diff.add_shape(
                            type="line",
                            x0=-0.5,
                            x1=len(year_data_diff) - 0.5,
                            y0=0,
                            y1=0,
                            line=dict(color="gray", width=1, dash="dash")
                        )
                        fig_diff.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                        fig_diff.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                        st.plotly_chart(fig_diff, use_container_width=True)
                        st.markdown("""
                        **Note on Average Difference:**
                        - **Positive values** (red bars): Team's batting average exceeds bowling average - typically successful teams
                        - **Negative values** (blue bars): Team's bowling average exceeds batting average - typically struggling teams
                        - The larger the positive difference, the more dominant the team typically is
                        """)

            # --- Restore Batting vs Bowling Average Comparison ---
            st.markdown('<div class="chart-header">🎯 Batting vs Bowling Average Comparison</div>', unsafe_allow_html=True)
            # Allow user to select year for the scatter plot
            selected_year_scatter = st.selectbox(
                "Select Year for Analysis", 
                available_years,
                index=0,
                key="scatter_year"
            )
            scatter_data = combined_df[combined_df['Year'] == selected_year_scatter]
            if not scatter_data.empty:
                scatter_fig = go.Figure()
                for _, row in scatter_data.iterrows():
                    team = row['Team']
                    bat_avg = row['Batting Avg']
                    bowl_avg = row['Bowling Avg']
                    perf_index = row['Performance Index']
                    avg_diff = row['Avg Difference']
                    if bat_avg > 0 and bowl_avg > 0:
                        scatter_fig.add_trace(go.Scatter(
                            x=[bat_avg],
                            y=[bowl_avg],
                            mode='markers+text',
                            text=[team],
                            textposition='top center',
                            marker=dict(
                                size=15,
                                color='#f84e4e',
                                opacity=0.8
                            ),
                            name=team,
                            hovertemplate=(
                                f"<b>{team}</b><br><br>"
                                f"Batting Avg: {bat_avg:.2f}<br>"
                                f"Bowling Avg: {bowl_avg:.2f}<br>"
                                f"Avg Difference: {avg_diff:.2f}<br>"
                                f"Performance Index: {perf_index:.2f}<br>"
                                "<extra></extra>"
                            )
                        ))
                # Add a diagonal reference line (where batting avg = bowling avg)
                max_val = max(scatter_data['Batting Avg'].max(), scatter_data['Bowling Avg'].max()) + 5
                min_val = min(scatter_data['Batting Avg'].min(), scatter_data['Bowling Avg'].min(), 0)
                scatter_fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='gray', dash='dash'),
                    showlegend=False,
                    hoverinfo='none'
                ))
                scatter_fig.update_layout(
                    title=f"Batting Average vs Bowling Average for {selected_year_scatter}",
                    xaxis_title="Batting Average",
                    yaxis_title="Bowling Average",
                    height=600,
                    font=dict(size=12),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False,
                    xaxis=dict(
                        range=[min_val, max_val],
                    ),
                    yaxis=dict(
                        range=[min_val, max_val],
                    ),
                )
                st.plotly_chart(scatter_fig, use_container_width=True)
                st.markdown("""
                **Interpreting the Scatter Plot:**
                - **Above the diagonal line**: Bowling average is higher than batting average (challenging position)
                - **Below the diagonal line**: Batting average is higher than bowling average (advantageous position)
                - **Top-right**: Teams with high batting and high bowling averages (batsman-friendly conditions)
                - **Bottom-left**: Teams with low batting and low bowling averages (bowler-friendly conditions)
                - **Top-left**: Strong batting teams with lower bowling averages (ideal position)
                - **Bottom-right**: Weaker batting teams with higher bowling averages (struggling position)
                """)

                # --- Restore Performance Index vs Average Difference ---
                st.markdown('<div class="chart-header">📊 Performance Index vs Average Difference</div>', unsafe_allow_html=True)
                perf_index_fig = go.Figure()
                for _, row in scatter_data.iterrows():
                    team = row['Team']
                    perf_index = row['Performance Index']
                    avg_diff = row['Avg Difference']
                    bat_avg = row['Batting Avg']
                    bowl_avg = row['Bowling Avg']
                    # Determine marker color based on performance index
                    if perf_index > 200:
                        marker_color = '#32CD32'  # Bright green for good performance
                    elif perf_index > 180:
                        marker_color = '#FFA500'  # Orange for average performance
                    else:
                        marker_color = '#DC143C'  # Crimson for below average performance
                    perf_index_fig.add_trace(go.Scatter(
                        x=[perf_index],
                        y=[avg_diff],
                        mode='markers+text',
                        text=[team],
                        textposition='top center',
                        marker=dict(
                            size=15,
                            color=marker_color,
                            opacity=0.8
                        ),
                        name=team,
                        hovertemplate=(
                            f"<b>{team}</b><br><br>"
                            f"Performance Index: {perf_index:.2f}<br>"
                            f"Avg Difference: {avg_diff:.2f}<br>"
                            f"Batting Avg: {bat_avg:.2f}<br>"
                            f"Bowling Avg: {bowl_avg:.2f}<br>"
                            "<extra></extra>"
                        )
                    ))
                # Add a horizontal reference line at y=0
                perf_index_fig.add_shape(
                    type="line",
                    x0=min(scatter_data['Performance Index'].min() - 10, 160),
                    x1=max(scatter_data['Performance Index'].max() + 10, 240),
                    y0=0,
                    y1=0,
                    line=dict(color="gray", dash="dash")
                )
                # Add a vertical reference line at x=200 (balanced performance)
                perf_index_fig.add_shape(
                    type="line",
                    x0=200,
                    x1=200,
                    y0=min(scatter_data['Avg Difference'].min() - 5, -10),
                    y1=max(scatter_data['Avg Difference'].max() + 5, 10),
                    line=dict(color="gray", dash="dash")
                )
                perf_index_fig.update_layout(
                    title=f"Team Performance Analysis for {selected_year_scatter}",
                    xaxis_title="Performance Index",
                    yaxis_title="Average Difference",
                    height=600,
                    font=dict(size=12),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False,
                    xaxis=dict(
                        range=[min(scatter_data['Performance Index'].min() - 10, 160), 
                              max(scatter_data['Performance Index'].max() + 10, 240)]
                    ),
                    yaxis=dict(
                        range=[min(scatter_data['Avg Difference'].min() - 5, -10),
                              max(scatter_data['Avg Difference'].max() + 5, 10)]
                    ),
                    xaxis_gridcolor='rgba(200, 200, 200, 0.2)',
                    yaxis_gridcolor='rgba(200, 200, 200, 0.2)',
                )
                st.plotly_chart(perf_index_fig, use_container_width=True)
                st.markdown("""
                **Interpreting the Performance Index Chart:**
                - **Top-Right**: Elite teams (high performance index, positive avg difference)
                - **Bottom-Right**: Good bowling teams, weaker batting (high performance index, negative avg difference)
                - **Top-Left**: Good batting teams, weaker bowling (lower performance index, positive avg difference)
                - **Bottom-Left**: Struggling teams (lower performance index, negative avg difference)
                - **Vertical line at 200**: Balanced team at format average
                """)
        # --- END OF OPTIMIZED CODE ---

@st.cache_data
def compute_team_rankings(bat_season_df, bowl_season_df):
    if bat_season_df.empty or bowl_season_df.empty:
        return pd.DataFrame()

    # --- Start: Logic copied directly from your original tab3 ---
    format_bat_mean = bat_season_df.groupby('Year')['Avg'].mean().to_dict()
    format_bowl_mean = bowl_season_df.groupby('Year')['Avg'].mean().to_dict()
    default_bat_mean = bat_season_df['Avg'].mean() if not bat_season_df.empty else 30
    default_bowl_mean = bowl_season_df['Avg'].mean() if not bowl_season_df.empty else 25

    combined_df = pd.merge(
        bat_season_df,
        bowl_season_df,
        left_on=['Team', 'Year'],
        right_on=['Bowl_Team', 'Year'],
        how='outer'
    )
    if 'Bowl_Team' in combined_df.columns:
        combined_df = combined_df.drop('Bowl_Team', axis=1)
    combined_df = combined_df.fillna(0)

    combined_df = combined_df.rename(columns={
        'Avg_x': 'Batting Avg',
        'SR_x': 'Batting SR',
        'Avg_y': 'Bowling Avg',
        'SR_y': 'Bowling SR',
        'Matches_x': 'Bat_Matches',
        'Matches_y': 'Bowl_Matches',
        'Runs_x': 'Runs',
        'Runs_y': 'Bowl_Runs',
        'Balls_x': 'Balls/Out',
        'Balls_y': 'Bowl_Balls',
        'Economy Rate': 'Economy',
        '5W': '5W',
        '10W': '10W',
        '50s': '50s',
        '100s': '100s',
        '150s': '150s',
        '200s': '200s',
        'Overs': 'Overs',
        'Wickets': 'Wickets'
    })

    bat_avg_col = 'Batting Avg' if 'Batting Avg' in combined_df.columns else 0
    bowl_avg_col = 'Bowling Avg' if 'Bowling Avg' in combined_df.columns else 0

    combined_df['Performance Index'] = combined_df.apply(
        lambda row: round(
            100 * (row.get(bat_avg_col, 0) / format_bat_mean.get(row['Year'], default_bat_mean)) +
            100 * (format_bowl_mean.get(row['Year'], default_bowl_mean) / (row.get(bowl_avg_col, 0) + 0.01)),
            2
        ),
        axis=1
    )
    combined_df['Avg Difference'] = (combined_df.get(bat_avg_col, 0) - combined_df.get(bowl_avg_col, 0)).round(2)
    combined_df = combined_df.sort_values(by=['Year', 'Performance Index'], ascending=[False, False])

    # After renaming columns and before rounding/column selection, ensure Batting SR and Bowling SR exist and are populated
    if 'Batting SR' not in combined_df.columns or combined_df['Batting SR'].isnull().all():
        if 'SR_x' in combined_df.columns:
            combined_df['Batting SR'] = combined_df['SR_x']
        elif 'SR' in combined_df.columns:
            combined_df['Batting SR'] = combined_df['SR']
        else:
            combined_df['Batting SR'] = 0.0
    if 'Bowling SR' not in combined_df.columns or combined_df['Bowling SR'].isnull().all():
        if 'SR_y' in combined_df.columns:
            combined_df['Bowling SR'] = combined_df['SR_y']
        elif 'Bowl_SR' in combined_df.columns:
            combined_df['Bowling SR'] = combined_df['Bowl_SR']
        else:
            combined_df['Bowling SR'] = 0.0

    # After sorting and before returning, round Batting Avg, Bowling Avg, Batting SR, Bowling SR, and Performance Index to 2 decimals if they exist
    for col in ['Batting Avg', 'Bowling Avg', 'Batting SR', 'Bowling SR', 'Performance Index']:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].round(2)

    columns_to_show = [
        'Team', 'Year', 'Runs', 'Batting Avg', 'Balls/Out', 'Batting SR', '50s', '100s', '150s', '200s',
        'Overs', 'Wickets', 'Bowling Avg', 'Bowling SR', 'Economy', '5W', '10W', 'Performance Index', 'Avg Difference'
    ]
    columns_to_show = [col for col in columns_to_show if col in combined_df.columns]
    result_df = combined_df[columns_to_show]
    return result_df

# --- Add these at module scope (top of the file, after imports) ---
def color_avg_difference(val):
    if val > 0:
        return f'background-color: rgba(144, 238, 144, {min(0.3 + val/20, 0.7)})' # Light green with intensity based on value
    elif val < 0:
        return f'background-color: rgba(255, 182, 193, {min(0.3 + abs(val)/20, 0.7)})' # Light red with intensity based on value
    else:
        return ''

def color_bat_avg(s):
    # Calculate means by year for batting average
    year_means = s.groupby(s.index.get_level_values('Year') if hasattr(s.index, 'get_level_values') and 'Year' in s.index.names else None).mean().to_dict() if hasattr(s.index, 'get_level_values') and 'Year' in s.index.names else {}
    result = pd.Series(index=s.index, data='')
    for idx, val in s.items():
        year = idx[1] if isinstance(idx, tuple) and len(idx) > 1 else None
        year_mean = year_means.get(year, val) if year is not None else val
        if val > year_mean:
            result.loc[idx] = f'background-color: rgba(144, 238, 144, {min(0.3 + (val-year_mean)/10, 0.7)})'
        elif val < year_mean:
            result.loc[idx] = f'background-color: rgba(255, 182, 193, {min(0.3 + (year_mean-val)/10, 0.7)})'
    return result

def color_bowl_avg(s):
    year_means = s.groupby(s.index.get_level_values('Year') if hasattr(s.index, 'get_level_values') and 'Year' in s.index.names else None).mean().to_dict() if hasattr(s.index, 'get_level_values') and 'Year' in s.index.names else {}
    result = pd.Series(index=s.index, data='')
    for idx, val in s.items():
        year = idx[1] if isinstance(idx, tuple) and len(idx) > 1 else None
        year_mean = year_means.get(year, val) if year is not None else val
        if val < year_mean:
            result.loc[idx] = f'background-color: rgba(144, 238, 144, {min(0.3 + (year_mean-val)/10, 0.7)})'
        elif val > year_mean:
            result.loc[idx] = f'background-color: rgba(255, 182, 193, {min(0.3 + (val-year_mean)/10, 0.7)})'
    return result

# No need for the if __name__ == "__main__" part
display_team_view()
