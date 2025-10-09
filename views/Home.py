# ==============================================================================
# FINAL, CLOUD-OPTIMIZED SCRIPT
# ==============================================================================
import streamlit as st
import os
import traceback
import pandas as pd
import tempfile
import time

# Import your original, working processing functions (NO CHANGES NEEDED HERE)
# Make sure these files (match.py, etc.) are in your GitHub repo.
from match import process_match_data
from game import process_game_stats
from bat import process_bat_stats
from bowl import process_bowl_stats

# --- CONFIGURATION (NEW) ---
# Calibrated based on user feedback (1664 files took ~112s).
# This value represents the average processing time per scorecard in seconds.
# You can adjust this value if you find the estimate is consistently off
# on your machine. A lower value (e.g., 0.06) means a faster estimate.
PROCESSING_TIME_PER_FILE = 0.065


# --- HELPER FUNCTIONS (FROM YOUR ORIGINAL FILE) ---
def process_duplicates(bat_df):
    """Your original duplicate processing logic."""
    # (Your original, working code for this function goes here. No changes needed.)
    try:
        duplicates_base = bat_df[['Name', 'Bat_Team_y', 'Year', 'Competition']].copy()
        team_counts = duplicates_base.groupby(['Name', 'Year', 'Competition'])['Bat_Team_y'].nunique()
        multi_team_players = team_counts[team_counts > 1].reset_index()
        multi_team_players.columns = ['Name', 'Year', 'Competition', 'Team_Count']
        
        duplicates_list = []
        if not multi_team_players.empty:
            for _, row in multi_team_players.iterrows():
                player_teams = duplicates_base[
                    (duplicates_base['Name'] == row['Name']) & 
                    (duplicates_base['Year'] == row['Year']) & 
                    (duplicates_base['Competition'] == row['Competition'])
                ]['Bat_Team_y'].unique()
                duplicates_list.append({
                    'Name': row['Name'], 'Year': row['Year'], 'Competition': row['Competition'],
                    'Teams': ', '.join(player_teams), 'Team_Count': len(player_teams)
                })
        
        duplicates = pd.DataFrame(duplicates_list) if duplicates_list else pd.DataFrame(columns=['Name', 'Year', 'Competition', 'Teams', 'Team_Count'])
        
        innings_counts = bat_df.groupby(['Name', 'File Name', 'Innings']).size()
        multi_innings_players = innings_counts[innings_counts > 1].reset_index()
        multi_innings_players.columns = ['Name', 'File Name', 'Innings', 'Count']
        
        teamduplicates_list = []
        if not multi_innings_players.empty:
            for _, row in multi_innings_players.iterrows():
                player_details = bat_df[
                    (bat_df['Name'] == row['Name']) & 
                    (bat_df['File Name'] == row['File Name']) & 
                    (bat_df['Innings'] == row['Innings'])
                ][['Name', 'File Name', 'Innings', 'Year', 'Competition', 'Bat_Team_y']].iloc[0]
                teamduplicates_list.append({
                    'Name': row['Name'], 'File Name': row['File Name'], 'Year': player_details['Year'],
                    'Competition': player_details['Competition'], 'Team': player_details['Bat_Team_y'], 'Count': row['Count']
                })
        
        teamduplicates = pd.DataFrame(teamduplicates_list).drop_duplicates() if teamduplicates_list else pd.DataFrame(columns=['Name', 'File Name', 'Year', 'Competition', 'Team', 'Count'])
        
        return {
            'multi_team': duplicates,
            'team_duplicates': teamduplicates,
            'has_duplicates': not duplicates.empty or not teamduplicates.empty
        }
    except Exception as e:
        print(f"Error processing duplicates: {str(e)}") # Log to console
        return {'multi_team': pd.DataFrame(), 'team_duplicates': pd.DataFrame(), 'has_duplicates': False}


# --- CACHED DATA PROCESSING ENGINE ---

@st.cache_data(ttl=3600, show_spinner=False) # Cache for 1 hour
def process_zip_data(_uploaded_file_bytes):
    """
    This is the "heavy lifting" function. It takes file bytes, processes them,
    and returns the final data. It has NO Streamlit UI calls (like st.write).
    This function is what gets cached to prevent re-running on every interaction.
    """
    try:
        total_files = 0
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(io.BytesIO(_uploaded_file_bytes)) as z:
                scorecard_files = [f for f in z.namelist() if f.endswith('.txt') and not f.startswith('__MACOSX/')]
                # Only extract the .txt files, not everything
                for f in scorecard_files:
                    z.extract(f, temp_dir)
                total_files = len(scorecard_files)
                if total_files == 0:
                    return {"error": "The uploaded ZIP file does not contain any .txt scorecards."}
                # Debug: show which files are being processed
                print(f"Debug: Files extracted for processing: {scorecard_files}")
            
            # --- Process data ---
            match_df = process_match_data(temp_dir)
            if match_df is None or match_df.empty: raise ValueError("match.py failed.")

            game_df = process_game_stats(temp_dir, match_df)
            if game_df is None or game_df.empty: raise ValueError("game.py failed.")
            
            bowl_df = process_bowl_stats(temp_dir, game_df, match_df)
            
            bat_df = process_bat_stats(temp_dir, game_df, match_df)
            if bat_df is None or bat_df.empty: raise ValueError("bat.py failed.")

            duplicates_result = process_duplicates(bat_df)

            # --- CRITICAL: Optimize all dataframes before returning ---
            match_df = optimize_df_memory(match_df)
            game_df = optimize_df_memory(game_df)
            bowl_df = optimize_df_memory(bowl_df)
            bat_df = optimize_df_memory(bat_df)
            
            # Explicitly collect garbage to free up memory from intermediate steps
            gc.collect()

            return {
                "match_df": match_df,
                "game_df": game_df,
                "bowl_df": bowl_df,
                "bat_df": bat_df,
                "duplicates_result": duplicates_result,
                "total_files": total_files,
                "error": None,
                "traceback": None
            }
            
    except zipfile.BadZipFile:
        return {"error": "The uploaded file is not a valid ZIP archive."}
    except Exception as e:
        return {"error": f"A critical error occurred: {e}", "traceback": traceback.format_exc()}


# --- UI FUNCTIONS (Your originals are mostly fine) ---

def show_processing_results(total_files, duplicates_result, processing_time=None):
    # Your original function here...
    time_text = f"{processing_time:.1f} seconds" if processing_time and processing_time < 60 else f"{int(processing_time // 60)}m {processing_time % 60:.1f}s"

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 15px; text-align: center; margin: 20px 0; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
            <h2 style="color: white; margin: 0;">🎉 Processing Complete!</h2>
            <p style="color: white; margin: 10px 0; font-size: 18px;">Successfully processed {total_files} scorecards in {time_text}</p>
            <p style="color: white; margin: 5px 0; font-size: 14px; opacity: 0.9;">Your cricket data is ready to explore!</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not duplicates_result['has_duplicates']:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); padding: 20px; border-radius: 15px; text-align: center; margin: 20px 0; box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
                <h3 style="color: white; margin: 0;">✅ Data Quality Check</h3>
                <p style="color: white; margin: 10px 0;">Excellent! No duplicate players detected. Your data is clean and ready for analysis.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 15px; text-align: center; margin: 20px 0; box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
                <h3 style="color: white; margin: 0;">⚠️ Duplicate Players Detected</h3>
                <p style="color: white; margin: 10px 0;"><strong>Quick Fix:</strong> In Cricket Captain, edit player profiles and add initials to distinguish players with similar names.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not duplicates_result['multi_team'].empty:
            st.markdown("### 🔄 Multi-Team Players")
            st.dataframe(duplicates_result['multi_team'], use_container_width=True)
        if not duplicates_result['team_duplicates'].empty:
            st.markdown("### 👥 Team Duplicates")
            st.dataframe(duplicates_result['team_duplicates'], use_container_width=True)


def show_error(message, traceback_info=None):
    # Your original function here...
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); padding: 20px; border-radius: 15px; text-align: center; margin: 20px 0; box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
            <h3 style="color: white; margin: 0;">❌ Processing Error</h3>
            <p style="color: white; margin: 10px 0;">{message}</p>
        </div>
    """, unsafe_allow_html=True)
    if traceback_info:
        with st.expander("🔧 Technical Details"): st.code(traceback_info)

def load_data(uploaded_files):
    """The reliable data loading engine that uses your original scripts and UI."""
    total_files = len(uploaded_files)
    start_time = time.time()
    
    progress_container = st.container()
    with progress_container:
        st.markdown("""
            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; margin: 20px 0;">
                <h3 style="color: white; margin: 0; text-align: center;">🏏 Processing Cricket Data</h3>
            </div>
        """, unsafe_allow_html=True)
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            status_text.text(f"📁 Preparing {total_files} files...")
            for i, uploaded_file in enumerate(uploaded_files):
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, 'wb') as f: f.write(uploaded_file.getbuffer())
                progress_bar.progress((i + 1) / total_files * 0.2)
            
            status_text.text("🔄 Processing match data..."); progress_bar.progress(0.3)
            match_df = process_match_data(temp_dir)
            if match_df is None or match_df.empty: raise ValueError("match.py failed to produce data.")

            status_text.text("📊 Processing game statistics..."); progress_bar.progress(0.5)
            game_df = process_game_stats(temp_dir, match_df)
            if game_df is None or game_df.empty: raise ValueError("game.py failed to produce data.")
            
            status_text.text("🎯 Processing bowling statistics..."); progress_bar.progress(0.7)
            bowl_df = process_bowl_stats(temp_dir, game_df, match_df)
            
            status_text.text("🏏 Processing batting statistics..."); progress_bar.progress(0.9)
            bat_df = process_bat_stats(temp_dir, game_df, match_df)
            if bat_df is None or bat_df.empty: raise ValueError("bat.py failed to produce data.")

            status_text.text("🔍 Checking for duplicates..."); progress_bar.progress(0.95)
            duplicates_result = process_duplicates(bat_df)

            st.session_state['match_df'] = match_df
            st.session_state['game_df'] = game_df
            st.session_state['bowl_df'] = bowl_df
            st.session_state['bat_df'] = bat_df
            st.session_state['data_loaded'] = True
            
            progress_bar.progress(1.0)
            end_time = time.time()
            
            progress_container.empty() 
            show_processing_results(total_files, duplicates_result, end_time - start_time)
            st.info("Navigate to other views from the sidebar to see your stats.")

    except Exception as e:
        progress_container.empty()
        show_error(f"A critical error occurred during processing: {e}", traceback.format_exc())

# --- UI & PAGE LOGIC (YOUR FULL, ORIGINAL UI) ---

st.markdown("""
<style>
    .main > div { padding-top: 2rem; }
    .stButton > button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 10px; padding: 12px 24px; font-weight: bold; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.3); }
    .uploadedFile { border-radius: 10px; border: 2px dashed #667eea; padding: 20px; }
    h1, h2, h3 { font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; }
    .selection-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid #e6e6e6;
        transition: all 0.3s ease;
    }
    .selection-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    .selection-card h3 {
        margin-top: 0;
        margin-bottom: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .selection-card p {
        color: #555;
        font-size: 14px;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)
# ... all your other st.markdown calls for the UI ...
st.markdown("""
    <div style="text-align: center; margin-bottom: 40px;">
        <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3.5em; font-weight: bold; margin: 0;">
            🏏 Cricket Captain 2025 Dashboard
        </h1>
        <p style="font-size: 1.2em; color: #666; margin-top: 10px;">
            Transform your cricket data into powerful insights
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%); padding: 15px; border-radius: 12px; text-align: center; margin: 20px 0; box-shadow: 0 6px 20px rgba(0,0,0,0.1);">
        <span style="color: white; font-weight: bold; font-size: 18px;">⚡ NEW UPDATE v1.23</span><br>
        <span style="color: white; font-size: 16px;">25% faster scorecard loading and up to 75% speed boost for data-heavy tabs with smart caching for batting and bowling tabs</span>
    </div>
""", unsafe_allow_html=True)


st.markdown("""
    <div style="text-align: center; max-width: 1000px; margin: 40px auto;">
        <p style="font-size: 1.1em; line-height: 1.6; color: #444;">
            Welcome to the ultimate Cricket Captain 2025 analytics platform. 
            Upload your match scorecards and unlock comprehensive statistics and performance insights.
        </p>
    </div>
""", unsafe_allow_html=True)

with st.expander("📋 How to Use This Dashboard", expanded=False):
    st.markdown("### Quick Start Guide")
    st.info("📋 **Choose your upload method below and follow the steps:**")
    st.markdown("#### For Small Batches (< 1500 scorecards)")
    st.markdown("- **Step 1:** Click the `Small Batch` option.")
    st.markdown("- **Step 2:** Click 'Browse files' and select all your individual `.txt` scorecards from your saves folder.")
    st.markdown("#### For Large Batches (> 1500 scorecards)")
    st.markdown("- **Step 1:** Click the `Large Batch (ZIP)` option.")
    st.markdown("- **Step 2:** In your saves folder, select all `.txt` files, right-click, and choose **'Send to' -> 'Compressed (zipped) folder'** (Windows) or **'Compress'** (Mac).")
    st.markdown("- **Step 3:** Upload that single `.zip` file.")
    st.markdown("---")
    st.markdown("##### 📂 Save Folder Locations:")
    st.markdown("- **Windows:** `C:\\Users\\[USERNAME]\\AppData\\Roaming\\Childish Things\\Cricket Captain 2025`")
    st.markdown("- **Mac:** `~/Library/Containers/com.childishthings.cricketcaptain2025mac/Data/Library/Application Support/Cricket Captain 2025/childish things/cricket captain 2025/saves`")
    st.markdown("**Step 4:** Use the file browser below to select your .txt scorecard files")
    st.caption("💡 Tip: Select all files with Ctrl+A (Windows) or Cmd+A (Mac)")
    st.markdown("**Step 5:** Click 'Process Scorecards' and explore your data in the various tabs")

st.markdown("### 📁 Upload Your Scorecard Files")
uploaded_files = st.file_uploader(
    "Select your Cricket Captain 2025 scorecard files (.txt)",
    type=['txt'], accept_multiple_files=True,
    help="Browse and select multiple .txt files from your Cricket Captain saves folder"
)

if uploaded_files:
    # --- UPDATED CALCULATION ---
    # Use the more accurate constant defined at the top of the file
    estimated_time = len(uploaded_files) * PROCESSING_TIME_PER_FILE
    
    # Improved display logic for the time estimate
    if estimated_time < 60:
        time_estimate_str = f"~{max(1, round(estimated_time))} seconds" # Show at least 1 second
    else:
        minutes = int(estimated_time // 60)
        seconds = int(estimated_time % 60)
        time_estimate_str = f"~{minutes}m {seconds}s" if minutes > 0 else f"~{seconds}s"

    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 15px; border-radius: 10px; margin: 15px 0;">
            <strong>📊 Files Selected:</strong> {len(uploaded_files)} scorecard files ready for processing<br>
            <strong>⏱️ Estimated Time:</strong> {time_estimate_str}
        </div>
    """, unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Process Scorecards", use_container_width=True):
        if uploaded_files:
            load_data(uploaded_files)
        else:
            st.warning("⚠️ Please select your scorecard files first")

st.markdown("---")
# ... copy-paste the rest of your UI code here ...
st.markdown("### 🎥 Helpful Resources")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
        <a href='https://www.youtube.com/@Robscricket' target='_blank' 
           style='display: block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  color: white; padding: 15px; text-decoration: none; border-radius: 10px; 
                  font-weight: bold; text-align: center; margin: 10px 0;
                  box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
            📺 Watch Cricket Captain Saves
        </a>
    """, unsafe_allow_html=True)
    st.markdown("""
        <a href='https://youtu.be/ykn5jal7ZdY' target='_blank' 
           style='display: block; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                  color: white; padding: 15px; text-decoration: none; border-radius: 10px; 
                  font-weight: bold; text-align: center; margin: 10px 0;
                  box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
            🖥️ Real Name Fix - PC/Mac
        </a>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <a href='https://youtu.be/MenffAx4KoQ' target='_blank' 
           style='display: block; background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); 
                  color: white; padding: 15px; text-decoration: none; border-radius: 10px; 
                  font-weight: bold; text-align: center; margin: 10px 0;
                  box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
            📱 Real Name Fix - Mobile
        </a>
    """, unsafe_allow_html=True)  
    st.markdown("""
        <a href='https://youtu.be/lcAozvTeezg' target='_blank' 
           style='display: block; background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%); 
                  color: white; padding: 15px; text-decoration: none; border-radius: 10px; 
                  font-weight: bold; text-align: center; margin: 10px 0;
                  box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
            ⚾ Player Editor Tutorial
        </a>
    """, unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    st.markdown("""
        <a href='https://youtu.be/PqdVAuRwx0g' target='_blank' 
           style='display: block; background: linear-gradient(135deg, #ff7b7b 0%, #667eea 100%); 
                  color: white; padding: 15px; text-decoration: none; border-radius: 10px; 
                  font-weight: bold; text-align: center; margin: 10px 0;
                  box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
            🏏 T20 Tips and Tricks
        </a>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
        <a href='https://youtu.be/N-u7zwACAPk' target='_blank' 
           style='display: block; background: linear-gradient(135deg, #a8e6cf 0%, #dcedc8 100%); 
                  color: white; padding: 15px; text-decoration: none; border-radius: 10px; 
                  font-weight: bold; text-align: center; margin: 10px 0;
                  box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
            💼 Master Player Contracts
        </a>
    """, unsafe_allow_html=True)

col5, col6 = st.columns(2)
with col5:
    st.markdown("""
        <a href='https://youtu.be/OggYwlM_mv4' target='_blank' 
           style='display: block; background: linear-gradient(135deg, #fd79a8 0%, #fdcb6e 100%); 
                  color: white; padding: 15px; text-decoration: none; border-radius: 10px; 
                  font-weight: bold; text-align: center; margin: 10px 0;
                  box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
            🎯 Coaching Tutorial
        </a>
    """, unsafe_allow_html=True)
with col6:
    st.markdown("""
        <a href='https://youtu.be/GU75BvgRax0' target='_blank' 
           style='display: block; background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); 
                  color: white; padding: 15px; text-decoration: none; border-radius: 10px; 
                  font-weight: bold; text-align: center; margin: 10px 0;
                  box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
            🌦️ Weather & Pitch Tutorial
        </a>
    """, unsafe_allow_html=True)