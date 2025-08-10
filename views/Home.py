# ==============================================================================
# FINAL, CLOUD-OPTIMIZED SCRIPT
# ==============================================================================
import streamlit as st
import os
import traceback
import pandas as pd
import tempfile
import time
import zipfile
import io
import gc # Import garbage collector

# Import your original, working processing functions (NO CHANGES NEEDED HERE)
# Make sure these files (match.py, etc.) are in your GitHub repo.
from match import process_match_data
from game import process_game_stats
from bat import process_bat_stats
from bowl import process_bowl_stats

# --- HELPER FUNCTIONS ---

def optimize_df_memory(df, verbose=False):
    """
    Iterates through all columns of a DataFrame and modifies data types
    to reduce memory usage. This is CRITICAL for large datasets.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    if verbose: print(f'Memory usage before optimization: {start_mem:.2f} MB')

    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if pd.isna(c_min) or pd.isna(c_max): continue
                if c_min > 0 and c_max < 255:
                    df[col] = df[col].astype('uint8')
                elif c_min > -128 and c_max < 128:
                    df[col] = df[col].astype('int8')
                elif c_min > -32768 and c_max < 32768:
                    df[col] = df[col].astype('int16')
                elif c_min > -2147483648 and c_max < 2147483648:
                    df[col] = df[col].astype('int32')
                else:
                    df[col] = df[col].astype('int64')
            else:
                if pd.isna(c_min) or pd.isna(c_max): continue
                if c_min > -3.4e38 and c_max < 3.4e38:
                    df[col] = df[col].astype('float32')
                else:
                    df[col] = df[col].astype('float64')
        elif col_type == 'object':
            # Convert to category if the number of unique values is less than 50%
            if len(df[col].unique()) / len(df[col]) < 0.5:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print(f'Memory usage after optimization: {end_mem:.2f} MB')
    if verbose: print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
    return df

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
    
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 15px; text-align: center; margin: 20px 0; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
            <h2 style="color: white; margin: 0;">🎉 Processing Complete!</h2>
            <p style="color: white; margin: 10px 0; font-size: 18px;">Successfully processed {total_files} scorecards in {time_text}</p>
            <p style="color: white; margin: 5px 0; font-size: 14px; opacity: 0.9;">Your cricket data is ready to explore!</p>
        </div>
    """, unsafe_allow_html=True)
    
    if not duplicates_result['has_duplicates']:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); padding: 20px; border-radius: 15px; text-align: center; margin: 20px 0; box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
                <h3 style="color: white; margin: 0;">✅ Data Quality Check</h3>
                <p style="color: white; margin: 10px 0;">Excellent! No duplicate players detected. Your data is clean and ready for analysis.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 15px; text-align: center; margin: 20px 0; box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
                <h3 style="color: white; margin: 0;">⚠️ Duplicate Players Detected</h3>
                <p style="color: white; margin: 10px 0;"><strong>Quick Fix:</strong> In Cricket Captain, edit player profiles and add initials to distinguish players with similar names.</p>
            </div>
        """, unsafe_allow_html=True)
        if not duplicates_result['multi_team'].empty:
            st.markdown("### 🔄 Multi-Team Players"); st.dataframe(duplicates_result['multi_team'], use_container_width=True)
        if not duplicates_result['team_duplicates'].empty:
            st.markdown("### 👥 Team Duplicates"); st.dataframe(duplicates_result['team_duplicates'], use_container_width=True)


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


# --- MAIN APP UI & LOGIC ---

# Your entire page styling and markdown sections go here...
# ... (copy-paste your st.markdown blocks for the title, header, instructions, etc.)
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
        <span style="color: white; font-weight: bold; font-size: 18px;">⚡ NEW UPDATE v1.24</span><br>
        <span style="color: white; font-size: 16px;">New Team Rankings scatter charts + Match Impact breakdown by win/loss/draw for batting performance insights</span>
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


st.markdown("### 📁 Upload Your Scorecards")

# Initialize session state to manage the UI flow
if 'upload_choice' not in st.session_state:
    st.session_state.upload_choice = None

# --- Step 1: Show Selection Cards ---
if st.session_state.upload_choice is None:
    st.markdown("##### **Step 1:** Choose your upload method", help="Select the option that matches the number of files you have.")
    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            st.markdown(
                """
                <div class="selection-card">
                    <h3>📂 Small Batch</h3>
                    <p>Best for a few seasons or <strong>fewer than 1500</strong> individual scorecard files.</p>
                </div>
                """, unsafe_allow_html=True
            )
            if st.button("Select Small Batch", use_container_width=True, key="small_batch_btn"):
                st.session_state.upload_choice = 'small'
                st.session_state.upload_mode = 'small' # <-- ADD THIS LINE
                st.rerun()
    with col2:
        with st.container():
            st.markdown(
                """
                <div class="selection-card">
                    <h3>📦 Large Batch (ZIP)</h3>
                    <p><strong>Recommended.</strong> Required for long-term saves or <strong>more than 1500</strong> files.</p>
                </div>
                """, unsafe_allow_html=True
            )
            if st.button("Select Large Batch (ZIP)", use_container_width=True, key="large_batch_btn"):
                st.session_state.upload_choice = 'large'
                st.session_state.upload_mode = 'large' # <-- ADD THIS LINE
                st.rerun()

# --- Step 2: Show the Correct File Uploader ---
else:
    uploaded_files = None
    if st.session_state.upload_choice == 'small':
        st.markdown("##### **Step 2:** Upload your .txt files")
        uploaded_files = st.file_uploader(
            "Select all your .txt scorecard files",
            type=['txt'],
            accept_multiple_files=True
        )
    elif st.session_state.upload_choice == 'large':
        st.markdown("##### **Step 2:** Upload your .zip file")
        uploaded_files = st.file_uploader(
            "Select a single ZIP file containing your scorecards",
            type=['zip'],
            accept_multiple_files=False
        )

    # Button to go back and change the method
    if st.button("‹ Change Upload Method"):
        st.session_state.upload_choice = None
        st.rerun()

    # --- Step 3: Processing Logic ---
    if uploaded_files:
        if isinstance(uploaded_files, list) and len(uploaded_files) > 0:
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 15px; border-radius: 10px; margin: 15px 0;">
                    <strong>📂 Files Selected:</strong> {len(uploaded_files)} .txt files<br>
                    <strong>Ready to process!</strong>
                </div>
            """, unsafe_allow_html=True)
        elif not isinstance(uploaded_files, list):
             st.markdown(f"""
                <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 15px; border-radius: 10px; margin: 15px 0;">
                    <strong>📦 ZIP File Selected:</strong> {uploaded_files.name}<br>
                    <strong>Ready to process!</strong>
                </div>
            """, unsafe_allow_html=True)

        col1_proc, col2_proc, col3_proc = st.columns([1, 2, 1])
        with col2_proc:
            if st.button("🚀 Process Scorecards", use_container_width=True):
                # ADD THIS CHECK to be safe
                if st.session_state.upload_choice == 'small':
                    st.session_state.upload_mode = 'small'
                elif st.session_state.upload_choice == 'large':
                    st.session_state.upload_mode = 'large'
                
                zip_bytes_to_process = None

                # Prepare the zip bytes based on the upload method
                if st.session_state.upload_choice == 'small':
                    if len(uploaded_files) > 1500:
                        st.error("❌ Too many files selected. Please use the 'Large Batch (ZIP)' method for more than 1500 files.")
                    else:
                        zip_buffer = io.BytesIO()
                        uploaded_names = []
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                            for file in uploaded_files:
                                zf.writestr(file.name, file.getvalue())
                                uploaded_names.append(file.name)
                        st.info(f"Debug: Files zipped for processing: {uploaded_names}")
                        zip_bytes_to_process = zip_buffer.getvalue()
                
                elif st.session_state.upload_choice == 'large':
                    zip_bytes_to_process = uploaded_files.getvalue()

                # Process the data
                if zip_bytes_to_process:
                    start_time = time.time()
                    with st.spinner("Analyzing thousands of scorecards... this might take a minute for the first run."):
                        results = process_zip_data(zip_bytes_to_process)
                    end_time = time.time()

                    if results.get("error"):
                        show_error(results["error"], results.get("traceback"))
                    else:
                        st.session_state['match_df'] = results['match_df']
                        st.session_state['game_df'] = results['game_df']
                        st.session_state['bowl_df'] = results['bowl_df']
                        st.session_state['bat_df'] = results['bat_df']
                        st.session_state['processed_match_df'] = results['match_df']
                        st.session_state['processed_game_df'] = results['game_df']
                        st.session_state['processed_bowl_df'] = results['bowl_df']
                        st.session_state['processed_bat_df'] = results['bat_df']
                        st.session_state['data_loaded'] = True
                        
                        show_processing_results(results['total_files'], results['duplicates_result'], end_time - start_time)
                        st.info("Navigate to other pages from the sidebar to explore your stats.")
                        st.balloons()
    elif st.session_state.upload_choice:
        st.info("Please select your files using the uploader above to continue.")

# ... (The rest of your UI, like "Helpful Resources", goes here unchanged)
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