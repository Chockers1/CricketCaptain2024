import streamlit as st
import os
import sys
import traceback
import pandas as pd

# Add the directory containing your scripts to the Python path
script_dir = r"C:\Users\rtayl\OneDrive\Rob Documents\Python Scripts\Cricket Captain"
sys.path.append(script_dir)

# Import the processing functions from each script
from match import process_match_data
from game import process_game_stats
from bat import process_bat_stats
from bowl import process_bowl_stats

# Create the temp directory if it doesn't exist
temp_dir = "tempDir"
os.makedirs(temp_dir, exist_ok=True)

def load_data(userfilelocation):
    with st.spinner("Processing scorecards..."):
        try:
            # Process match data
            st.write("Processing match data...")
            match_df = process_match_data(userfilelocation)
            if match_df is not None and not match_df.empty:
                st.success("Match data processed successfully.")
                st.session_state['match_df'] = match_df

                # Process game stats
                st.write("Processing game stats...")
                game_df = process_game_stats(userfilelocation, match_df)
                if game_df is not None and not game_df.empty:
                    st.success("Game stats processed successfully.")
                    st.session_state['game_df'] = game_df

                    # Process bowling stats
                    st.write("Processing bowling stats...")
                    bowl_df = process_bowl_stats(userfilelocation, game_df, match_df)
                    if bowl_df is not None and not bowl_df.empty:
                        st.success("Bowling stats processed successfully.")
                        st.session_state['bowl_df'] = bowl_df

                        # Process batting stats
                        st.write("Processing batting stats...")
                        bat_df = process_bat_stats(userfilelocation, game_df, match_df)
                        if bat_df is not None and not bat_df.empty:
                            st.success("Batting stats processed successfully.")
                            st.session_state['bat_df'] = bat_df

                            st.success("All scorecards processed successfully. Data is now available across all pages.")
                            st.session_state['data_loaded'] = True

                        else:
                            st.error("Batting stats processing failed or returned empty DataFrame.")
                    else:
                        st.error("Bowling stats processing failed or returned empty DataFrame.")
                else:
                    st.error("Game stats processing failed or returned empty DataFrame.")
            else:
                st.error("Match data processing failed or returned empty DataFrame.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Traceback:")
            st.text(traceback.format_exc())

# --- HEADER MARKDOWN ---
st.markdown(
    """
    <h1 style="text-align: center; color: #f04f53; font-size: 2em; white-space: nowrap;">
    Welcome to Ultimate Cricket Captain 2024 Dashboard</h1>
    """,
    unsafe_allow_html=True
)

# --- UPLOAD SCORECARDS SECTION ---
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
                Just change the <code>[USER NAME]</code> and game you are playing.
            </li>
            <li><strong>In "Saves,"</strong> create a new folder called <code>Scorecards</code> and move all your .txt files you want in there.</li>
            <li><strong>Put the path in the text box below</strong> and click <strong>Process Scorecards</strong>.</li>
            <li><strong>Click on any of the tabs</strong> to see your saved data.</li>
        </ol>
    </div>
    """,
    unsafe_allow_html=True
)

# Input field for folder selection with default value
default_directory = r"C:\Users\rtayl\AppData\Roaming\Childish Things\Cricket Captain 2024\Saves\Scorecards"
userfilelocation = st.text_input("Enter the folder save location:", value=default_directory)

# Process Scorecards button
if st.button("Process Scorecards"):
    if userfilelocation:
        load_data(userfilelocation)
    else:
        st.warning("Please specify a folder location.")

