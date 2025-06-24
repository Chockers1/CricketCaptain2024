#!/usr/bin/env python3
"""
Debug script to check actual column names in session state data
when the allrounders view is accessed.
"""

import streamlit as st
import pandas as pd

def debug_columns():
    """Debug function to print actual column names"""
    
    st.title("🔍 Column Debug Information")
    
    # Check if session state data exists
    if 'bat_df' not in st.session_state:
        st.error("❌ bat_df not found in session state")
        return
    
    if 'bowl_df' not in st.session_state:
        st.error("❌ bowl_df not found in session state") 
        return
    
    # Get the dataframes
    bat_df = st.session_state['bat_df']
    bowl_df = st.session_state['bowl_df']
    
    # Display basic info
    st.success("✅ Both dataframes found in session state!")
    
    # Display batting dataframe info
    st.subheader("🏏 Batting DataFrame Info")
    st.write(f"Shape: {bat_df.shape}")
    st.write("Columns:")
    for i, col in enumerate(bat_df.columns):
        st.write(f"{i+1}. '{col}'")
    
    st.write("Sample data:")
    st.dataframe(bat_df.head())
    
    # Display bowling dataframe info  
    st.subheader("⚾ Bowling DataFrame Info")
    st.write(f"Shape: {bowl_df.shape}")
    st.write("Columns:")
    for i, col in enumerate(bowl_df.columns):
        st.write(f"{i+1}. '{col}'")
        
    st.write("Sample data:")
    st.dataframe(bowl_df.head())
    
    # Check for team-related columns
    st.subheader("🔍 Team Column Analysis")
    
    bat_team_cols = [col for col in bat_df.columns if 'team' in col.lower()]
    bowl_team_cols = [col for col in bowl_df.columns if 'team' in col.lower()]
    
    st.write("Batting team-related columns:", bat_team_cols)
    st.write("Bowling team-related columns:", bowl_team_cols)
    
    # Check for columns with _y suffix
    bat_y_cols = [col for col in bat_df.columns if col.endswith('_y')]
    bowl_y_cols = [col for col in bowl_df.columns if col.endswith('_y')]
    
    st.write("Batting columns with _y suffix:", bat_y_cols)
    st.write("Bowling columns with _y suffix:", bowl_y_cols)

if __name__ == "__main__":
    debug_columns()
