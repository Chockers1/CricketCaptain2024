import streamlit as st
import os
import sys
import traceback
import pandas as pd
import tempfile
import time

# Import the processing functions from each script
from match import process_match_data
from game import process_game_stats
from bat import process_bat_stats
from bowl import process_bowl_stats

def load_data(uploaded_files):
    # Total number of files
    total_files = len(uploaded_files)
    
    # Create a progress bar
    progress_bar = st.progress(0, text=f"Processing 0 out of {total_files} matches...")
    
    try:
        # Create a temporary directory to store uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded files to temporary directory with progress tracking
            processed_files = 0
            processed_files_list = []
            
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                processed_files_list.append(file_path)
                
                # Update progress
                processed_files += 1
                progress_percentage = int((processed_files / total_files) * 100)
                progress_bar.progress(progress_percentage, 
                    text=f"Processing {processed_files} out of {total_files} matches...")
            
            # Display initial message about number of matches
            st.info(f"Uploaded {total_files} match scorecards to process")
            
            # Process match data
            st.write("Processing match data...")
            match_df = process_match_data(temp_dir)
            if match_df is not None and not match_df.empty:
                st.success("Match data processed successfully.")
                st.session_state['match_df'] = match_df

                # Process game stats
                st.write("Processing game stats...")
                game_df = process_game_stats(temp_dir, match_df)
                if game_df is not None and not game_df.empty:
                    st.success("Game stats processed successfully.")
                    st.session_state['game_df'] = game_df

                    # Process bowling stats
                    st.write("Processing bowling stats...")
                    bowl_df = process_bowl_stats(temp_dir, game_df, match_df)
                    if bowl_df is not None and not bowl_df.empty:
                        st.success("Bowling stats processed successfully.")
                        st.session_state['bowl_df'] = bowl_df

                        # Process batting stats
                        st.write("Processing batting stats...")
                        bat_df = process_bat_stats(temp_dir, game_df, match_df)
                        if bat_df is not None and not bat_df.empty:
                            st.success("Batting stats processed successfully.")
                            st.session_state['bat_df'] = bat_df

                            # Final success message
                            st.success(f"All {total_files} scorecards processed successfully. Data is now available across all pages.")
                            st.session_state['data_loaded'] = True
                            
                            # Complete the progress bar
                            progress_bar.progress(100, text=f"Processed {total_files} out of {total_files} matches complete!")
                        else:
                            st.error("Batting stats processing failed or returned empty DataFrame.")
                            progress_bar.progress(0, text="Processing failed")
                    else:
                        st.error("Bowling stats processing failed or returned empty DataFrame.")
                        progress_bar.progress(0, text="Processing failed")
                else:
                    st.error("Game stats processing failed or returned empty DataFrame.")
                    progress_bar.progress(0, text="Processing failed")
            else:
                st.error("Match data processing failed or returned empty DataFrame.")
                progress_bar.progress(0, text="Processing failed")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Traceback:")
        st.text(traceback.format_exc())
        progress_bar.progress(0, text="Processing failed")

# --- HEADER MARKDOWN ---
st.markdown(
    """
    <h1 style="text-align: center; color: #f04f53; font-size: 2em; white-space: nowrap;">
    Welcome to Ultimate Cricket Captain 2024 Dashboard</h1>
    """,
    unsafe_allow_html=True
)

# --- UPDATE BANNER ---
st.markdown(
    """
    <div style="background-color: #f04f53; padding: 10px; border-radius: 5px; text-align: center; margin: 10px 0;">
        <span style="color: white; font-weight: bold;">🎉 NEW UPDATE v1.21:</span>
        <span style="color: white;"> Updated layout for Batting & Bowling Tabs </span>
    </div>
    """,
    unsafe_allow_html=True
)
# --- WELCOME MESSAGE SECTION ---
st.markdown(
    """
    <div style="text-align: center; max-width: 1200px; margin: auto;">
        <p>Thank you for subscribing to The Ultimate Cricket Captain 2024 Dashboard. 
        Explore comprehensive statistics and performance metrics from your saves and take your game to the next level:</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- HOW TO USE THE DASHBOARD SECTION ---
st.markdown(
    """
    <div style="text-align: center; max-width: 800px; margin: auto; font-family: Arial, sans-serif;">
        <h2 style="color: #f04f53;">How To Use The Dashboard</h2>
        <ol style="text-align: left; margin: auto;">
            <li><strong>In your Cricket Captain game,</strong> go into any scorecard you want the data from and click save.</li>
            <li><strong>Locate the .txt files</strong> which contain the scorecard data:
                <ul>
                    <li><em>Windows:</em> <code>C:\\Users\\[USER NAME]\\AppData\\Roaming\\Childish Things\\Cricket Captain 2021</code></li>
                    <li><em>MAC:</em> <code>~/Library/Containers/com.childishthings.cricketcaptain2021mac/Data/Library/Application Support/Cricket Captain 2021/childish things/cricket captain 2021/saves</code></li>
                </ul>
            </li>
            <li><strong>Select the browse files button</strong> </li>
            <li><strong>To select all files:</strong> Press <code>Ctrl + A</code> on Windows or <code>Command + A</code> on Mac, and then click open (you will need to select all everytime you load them.).</li>
            <li><strong>Click Process Scorecards</strong> to analyze your data.</li>
            <li><strong>Click on any of the tabs</strong> to see your saved data.</li>
        </ol>
    </div>
    """,
    unsafe_allow_html=True

)

# Channel link at the bottom
st.markdown("""
<div style='text-align: center; margin-top: 20px;'>
    <a href='https://www.youtube.com/@RobTaylor1985' target='_blank' 
    style='background-color: #f04f53; color: white; padding: 10px 20px; 
    text-decoration: none; border-radius: 5px; font-weight: bold;'>
    Watch My Cricket Captain 2024 Saves</a>
</div>
""", unsafe_allow_html=True)

# File uploader with a custom label
uploaded_files = st.file_uploader("Upload your scorecard files", 
                                   type=['txt'], 
                                   accept_multiple_files=True, 
                                   help="Select multiple .txt scorecard files from your Cricket Captain game")

# Process Scorecards button
if st.button("Process Scorecards"):
    if uploaded_files:
        load_data(uploaded_files)
    else:
        st.warning("Please upload your scorecard files.")
