import streamlit as st
import os
import traceback
import pandas as pd
import tempfile
import time
import zipfile

# Import your original, working processing functions
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
    """Process duplicate player detection."""
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
        st.session_state['duplicates'] = duplicates
        
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
        st.session_state['teamduplicates'] = teamduplicates
        
        return {
            'multi_team': duplicates,
            'team_duplicates': teamduplicates,
            'has_duplicates': not duplicates.empty or not teamduplicates.empty
        }
    except Exception as e:
        st.error(f"Error processing duplicates: {str(e)}")
        return {'multi_team': pd.DataFrame(), 'team_duplicates': pd.DataFrame(), 'has_duplicates': False}

def show_processing_results(total_files, duplicates_result, processing_time=None):
    """Show modern processing results with enhanced styling"""
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
    """Show modern error messages"""
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); padding: 20px; border-radius: 15px; text-align: center; margin: 20px 0; box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
            <h3 style="color: white; margin: 0;">❌ Processing Error</h3>
            <p style="color: white; margin: 10px 0;">{message}</p>
        </div>
    """, unsafe_allow_html=True)
    if traceback_info:
        with st.expander("🔧 Technical Details"): st.code(traceback_info)

def load_data(uploaded_files=None, uploaded_zip=None):
    """The reliable data loading engine that uses your original scripts and UI.

    Accepts either:
    - uploaded_files: list[UploadedFile] of .txt files, or
    - uploaded_zip: a single UploadedFile that's a .zip containing .txt files
    """

    def _safe_extract_all(zip_path: str, extract_to: str):
        """Extract zip contents safely to avoid path traversal (zip slip)."""
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.infolist():
                member_path = os.path.normpath(os.path.join(extract_to, member.filename))
                if not member_path.startswith(os.path.abspath(extract_to)):
                    # Skip entries trying to escape the extraction dir
                    continue
                # Create parent dirs as needed
                if member.is_dir():
                    os.makedirs(member_path, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(member_path), exist_ok=True)
                    with zf.open(member) as src, open(member_path, 'wb') as dst:
                        dst.write(src.read())

    def _flatten_txt_files(root_dir: str):
        """Move all .txt files from subdirectories into root_dir to ensure processing functions find them.

        If a filename collision occurs, append a numeric suffix before the extension.
        Returns the number of .txt files placed in root_dir.
        """
        count = 0
        existing = set(name.lower() for name in os.listdir(root_dir) if name.lower().endswith('.txt'))
        for dirpath, _, filenames in os.walk(root_dir):
            # Skip root_dir itself for moving
            if os.path.abspath(dirpath) == os.path.abspath(root_dir):
                continue
            for fname in filenames:
                if not fname.lower().endswith('.txt'):
                    continue
                src = os.path.join(dirpath, fname)
                base, ext = os.path.splitext(os.path.basename(fname))
                dest_name = f"{base}{ext}"
                # Resolve collisions case-insensitively
                i = 1
                while dest_name.lower() in existing:
                    dest_name = f"{base} ({i}){ext}"
                    i += 1
                dest = os.path.join(root_dir, dest_name)
                try:
                    # Move file into root
                    os.replace(src, dest)
                    existing.add(dest_name.lower())
                    count += 1
                except Exception:
                    # If move fails for any reason, skip
                    continue
        # Count also includes any .txt files that were already at root
        return len([n for n in os.listdir(root_dir) if n.lower().endswith('.txt')])

    # Determine total files for UX
    total_files = 0
    if uploaded_files:
        total_files = len(uploaded_files)
    elif uploaded_zip is not None:
        try:
            with zipfile.ZipFile(uploaded_zip, 'r') as zf:
                total_files = sum(1 for n in zf.namelist() if n.lower().endswith('.txt'))
        except zipfile.BadZipFile:
            show_error("The uploaded ZIP file is invalid or corrupted.")
            return

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
            if uploaded_files:
                status_text.text(f"📁 Preparing {total_files} files...")
                for i, uploaded_file in enumerate(uploaded_files):
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    # Cap the initial prep progress to 20%
                    progress_bar.progress(min(0.2, (i + 1) / max(1, total_files) * 0.2))
            elif uploaded_zip is not None:
                status_text.text("🗜️ Extracting ZIP contents...")
                # Save ZIP to disk first
                zip_temp_path = os.path.join(temp_dir, "upload.zip")
                with open(zip_temp_path, 'wb') as f:
                    f.write(uploaded_zip.getbuffer())
                # Extract safely
                _safe_extract_all(zip_temp_path, temp_dir)
                # Flatten nested .txt files so processors that scan only the top-level can find them
                total_files = _flatten_txt_files(temp_dir)
                if total_files == 0:
                    show_error("No .txt scorecards found inside the ZIP. Please ensure it contains your saved scorecard .txt files.")
                    return
                progress_bar.progress(0.2)
            else:
                show_error("No files provided. Please upload .txt files or a .zip archive.")
                return

            status_text.text("� Processing match data..."); progress_bar.progress(0.35)
            match_df = process_match_data(temp_dir)
            if match_df is None or match_df.empty:
                raise ValueError("match.py failed to produce data.")

            status_text.text("📊 Processing game statistics..."); progress_bar.progress(0.55)
            game_df = process_game_stats(temp_dir, match_df)
            if game_df is None or game_df.empty:
                raise ValueError("game.py failed to produce data.")

            status_text.text("🎯 Processing bowling statistics..."); progress_bar.progress(0.75)
            bowl_df = process_bowl_stats(temp_dir, game_df, match_df)

            status_text.text("🏏 Processing batting statistics..."); progress_bar.progress(0.9)
            bat_df = process_bat_stats(temp_dir, game_df, match_df)
            if bat_df is None or bat_df.empty:
                raise ValueError("bat.py failed to produce data.")

            status_text.text("🔍 Checking for duplicates..."); progress_bar.progress(0.95)
            duplicates_result = process_duplicates(bat_df)

            # Keep state keys identical so other tabs work seamlessly
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
</style>
""", unsafe_allow_html=True)

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
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 15px; border-radius: 12px; text-align: center; margin: 20px 0; box-shadow: 0 6px 20px rgba(0,0,0,0.1);">
        <span style="color: white; font-weight: bold; font-size: 18px;">⚡ NEW UPDATE v1.24</span><br>
        <span style="color: white; font-size: 16px;">Added in option to load ZIP files for more than 1200 scorecards, works seamlessly with all other data tabs, and many other improvements </span>
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
    st.info("📋 **Choose Your Method:**")
    col1, col2 = st.columns(2)
    with col1:
        st.success("🔄 **Option 1: Auto-Save (Recommended)**")
        st.write("**Step 1:** In Cricket Captain 2025, go to Options and enable 'Auto Save Scorecards'")
        st.write("**Step 2:** Scorecards will now automatically save after each match to the locations below")
    with col2:
        st.warning("📋 **Option 2: Manual Save**")
        st.write("**Step 1:** After each match, open the scorecard")
        st.write("**Step 2:** Click 'Save' to save the scorecard file")
    st.markdown("---")
    st.markdown("**Step 3:** Locate your scorecard files:")
    st.markdown("- **Windows:** `C:\\Users\\[USERNAME]\\AppData\\Roaming\\Childish Things\\Cricket Captain 2025`")
    st.markdown("- **Mac:** `~/Library/Containers/com.childishthings.cricketcaptain2025mac/Data/Library/Application Support/Cricket Captain 2025/childish things/cricket captain 2025/saves`")
    st.markdown("**Step 4:** Choose an upload method below:")
    st.markdown("- Select multiple `.txt` scorecard files, or")
    st.markdown("- Upload a single `.zip` containing all your `.txt` files")
    st.caption("💡 Tips: Select all with Ctrl+A (Windows) / Cmd+A (Mac). ZIPs can contain folders—this app flattens them and handles duplicate names safely.")
    st.markdown("**Step 5:** Click 'Process Scorecards' and explore your data in the various tabs")

st.markdown("### 📁 Upload Your Scorecard Files")

uploaded_files_txt = None
uploaded_zip_file = None

col_txt, col_zip = st.columns(2)

with col_txt:
    st.markdown("#### TXT files (< 1200)")
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 16px; border-radius: 15px; margin-bottom: 10px; box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
            <strong>When to use:</strong> Less than 1200 scorecards.<br>
            <strong>Use the file browser below to select your .txt scorecard files.<br>
            <strong>💡 Tip:</strong> Click <em>Browse files</em>, then press <code>Ctrl+A</code> (Windows) or <code>Cmd+A</code> (Mac) to select all scorecard .txt files.<br>
            <strongStep 5:</strong> Click 'Process Scorecards' and explore your data in the various tabs.
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_files_txt = st.file_uploader(
        "Select your Cricket Captain 2025 scorecard files (.txt)",
        type=["txt"],
        accept_multiple_files=True,
        key="txt_uploader",
        help="Browse and select multiple .txt files from your Cricket Captain saves folder",
    )
    if uploaded_files_txt:
        estimated_time = len(uploaded_files_txt) * PROCESSING_TIME_PER_FILE
        if estimated_time < 60:
            time_estimate_str = f"~{max(1, round(estimated_time))} seconds"
        else:
            minutes = int(estimated_time // 60)
            seconds = int(estimated_time % 60)
            time_estimate_str = f"~{minutes}m {seconds}s" if minutes > 0 else f"~{seconds}s"
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 15px; border-radius: 10px; margin: 15px 0;">
                <strong>📊 Files Selected:</strong> {len(uploaded_files_txt)} scorecard files ready for processing<br>
                <strong>⏱️ Estimated Time:</strong> {time_estimate_str}
            </div>
            """,
            unsafe_allow_html=True,
        )

with col_zip:
    st.markdown("#### ZIP (> 1200)")
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 16px; border-radius: 15px; margin-bottom: 10px; box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
            <strong>When to use:</strong> Greater than 1200 scorecards.<br>
            <strong>How:</strong> Go to your scorecards folder, press Ctrl+A to select all, then right-click and choose Compress/Zip.<br>
            <strong>Next:</strong> Click <em>Browse files</em> below and select the ZIP you just created.<br>
            <strong>Good to know:</strong> Folders inside the ZIP are fine—this app flattens them and handles duplicate names safely.
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_zip_file = st.file_uploader(
        "Upload a ZIP containing your .txt scorecard files",
        type=["zip"],
        accept_multiple_files=False,
        key="zip_uploader",
        help="Compress your scorecards folder to a single .zip for faster upload of large sets.",
    )
    if uploaded_zip_file is not None:
        try:
            with zipfile.ZipFile(uploaded_zip_file, "r") as zf:
                total = sum(1 for n in zf.namelist() if n.lower().endswith(".txt"))
        except zipfile.BadZipFile:
            total = 0
            st.error("The uploaded ZIP looks invalid. Please try re-zipping your files.")

        if total > 0:
            estimated_time = total * PROCESSING_TIME_PER_FILE
            if estimated_time < 60:
                time_estimate_str = f"~{max(1, round(estimated_time))} seconds"
            else:
                minutes = int(estimated_time // 60)
                seconds = int(estimated_time % 60)
                time_estimate_str = f"~{minutes}m {seconds}s" if minutes > 0 else f"~{seconds}s"
            st.markdown(
                f"""
                <div style=\"background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 15px; border-radius: 10px; margin: 15px 0;\"> 
                    <strong>🗜️ Files Detected in ZIP:</strong> {total} .txt files<br>
                    <strong>⏱️ Estimated Time:</strong> {time_estimate_str}
                </div>
                """,
                unsafe_allow_html=True,
            )

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Process Scorecards", use_container_width=True):
        # Prevent confusion if both are provided
        if uploaded_files_txt and uploaded_zip_file is not None:
            st.warning("Please upload either TXT files or a ZIP, not both at the same time.")
        elif uploaded_files_txt:
            load_data(uploaded_files=uploaded_files_txt, uploaded_zip=None)
        elif uploaded_zip_file is not None:
            load_data(uploaded_files=None, uploaded_zip=uploaded_zip_file)
        else:
            st.warning("⚠️ Please select your TXT files or upload a ZIP first")

st.markdown("---")
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
