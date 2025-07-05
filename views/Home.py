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
    """Process uploaded scorecard files with modern progress tracking"""
    total_files = len(uploaded_files)
    start_time = time.time()  # Start timing
    
    # Modern progress container
    progress_container = st.container()
    with progress_container:
        st.markdown("""
            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                        padding: 20px; border-radius: 15px; margin: 20px 0;">
                <h3 style="color: white; margin: 0; text-align: center;">
                    🏏 Processing Cricket Data
                </h3>
            </div>
        """, unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # File upload phase
            processed_files = 0
            processed_files_list = []
            
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                processed_files_list.append(file_path)
                
                processed_files += 1
                progress = processed_files / total_files * 0.2  # 20% for file processing
                progress_bar.progress(progress)
                status_text.text(f"📁 Uploading files... {processed_files}/{total_files}")
            
            # Success message for upload
            st.success(f"✅ Successfully uploaded {total_files} match scorecards")
            
            # Process match data (20-40%)
            status_text.text("🔄 Processing match data...")
            progress_bar.progress(0.3)
            match_df = process_match_data(temp_dir)
            
            if match_df is not None and not match_df.empty:
                st.session_state['match_df'] = match_df
                st.success("✅ Match data processed successfully")
                
                # Process game stats (40-60%)
                status_text.text("📊 Processing game statistics...")
                progress_bar.progress(0.5)
                game_df = process_game_stats(temp_dir, match_df)
                
                if game_df is not None and not game_df.empty:
                    st.session_state['game_df'] = game_df
                    st.success("✅ Game statistics processed successfully")
                    
                    # Process bowling stats (60-80%)
                    status_text.text("🎯 Processing bowling statistics...")
                    progress_bar.progress(0.7)
                    bowl_df = process_bowl_stats(temp_dir, game_df, match_df)
                    
                    if bowl_df is not None and not bowl_df.empty:
                        st.session_state['bowl_df'] = bowl_df
                        st.success("✅ Bowling statistics processed successfully")
                        
                        # Process batting stats (80-100%)
                        status_text.text("🏏 Processing batting statistics...")
                        progress_bar.progress(0.9)
                        bat_df = process_bat_stats(temp_dir, game_df, match_df)
                        
                        if bat_df is not None and not bat_df.empty:
                            st.session_state['bat_df'] = bat_df
                            st.success("✅ Batting statistics processed successfully")
                            
                            # Process duplicates
                            status_text.text("🔍 Checking for duplicate players...")
                            duplicates_result = process_duplicates(bat_df)
                            
                            # Complete processing
                            progress_bar.progress(1.0)
                            end_time = time.time()
                            processing_time = end_time - start_time
                            
                            status_text.text(f"✅ Processing complete! ({processing_time:.1f} seconds)")
                            st.session_state['data_loaded'] = True
                            
                            # Show final results with timing
                            show_processing_results(total_files, duplicates_result, processing_time)
                            
                        else:
                            show_error("Batting statistics processing failed")
                    else:
                        show_error("Bowling statistics processing failed")
                else:
                    show_error("Game statistics processing failed")
            else:
                show_error("Match data processing failed")
                
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        show_error(f"Processing error after {processing_time:.1f} seconds: {str(e)}", traceback.format_exc())

def process_duplicates(bat_df):
    """Process duplicate player detection with modern error handling"""
    try:
        # Multi-team players
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
                    'Name': row['Name'],
                    'Year': row['Year'], 
                    'Competition': row['Competition'],
                    'Teams': ', '.join(player_teams),
                    'Team_Count': len(player_teams)
                })
        
        duplicates = pd.DataFrame(duplicates_list) if duplicates_list else pd.DataFrame(columns=['Name', 'Year', 'Competition', 'Teams', 'Team_Count'])
        st.session_state['duplicates'] = duplicates
        
        # Team duplicates (same player multiple times in same innings)
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
                    'Name': row['Name'],
                    'File Name': row['File Name'],
                    'Year': player_details['Year'],
                    'Competition': player_details['Competition'],
                    'Team': player_details['Bat_Team_y'],
                    'Count': row['Count']
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
    # Format time display
    if processing_time:
        if processing_time < 60:
            time_text = f"{processing_time:.1f} seconds"
        else:
            minutes = int(processing_time // 60)
            seconds = processing_time % 60
            time_text = f"{minutes}m {seconds:.1f}s"
    else:
        time_text = ""
    
    # Success message with timing
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 25px; border-radius: 15px; text-align: center; margin: 20px 0;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
            <h2 style="color: white; margin: 0;">🎉 Processing Complete!</h2>
            <p style="color: white; margin: 10px 0; font-size: 18px;">
                Successfully processed {total_files} scorecards in {time_text}
            </p>
            <p style="color: white; margin: 5px 0; font-size: 14px; opacity: 0.9;">
                Your cricket data is ready to explore!
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Duplicate check results
    if not duplicates_result['has_duplicates']:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); 
                        padding: 20px; border-radius: 15px; text-align: center; margin: 20px 0;
                        box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
                <h3 style="color: white; margin: 0;">✅ Data Quality Check</h3>
                <p style="color: white; margin: 10px 0;">
                    Excellent! No duplicate players detected. Your data is clean and ready for analysis.
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 20px; border-radius: 15px; text-align: center; margin: 20px 0;
                        box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
                <h3 style="color: white; margin: 0;">⚠️ Duplicate Players Detected</h3>
                <p style="color: white; margin: 10px 0;">
                    <strong>Quick Fix:</strong> In Cricket Captain 2025, edit player profiles and add initials to distinguish players with similar names.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        if not duplicates_result['multi_team'].empty:
            st.markdown("### 🔄 Multi-Team Players")
            st.markdown("*Players who appeared for multiple teams in the same year/competition*")
            st.dataframe(duplicates_result['multi_team'], use_container_width=True)
        
        if not duplicates_result['team_duplicates'].empty:
            st.markdown("### 👥 Team Duplicates")
            st.markdown("*Players appearing multiple times in the same match*")
            st.dataframe(duplicates_result['team_duplicates'], use_container_width=True)

def show_error(message, traceback_info=None):
    """Show modern error messages"""
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); 
                    padding: 20px; border-radius: 15px; text-align: center; margin: 20px 0;
                    box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
            <h3 style="color: white; margin: 0;">❌ Processing Error</h3>
            <p style="color: white; margin: 10px 0;">{message}</p>
        </div>
    """, unsafe_allow_html=True)
    
    if traceback_info:
        with st.expander("🔧 Technical Details"):
            st.code(traceback_info)


# Custom CSS for modern styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    .uploadedFile {
        border-radius: 10px;
        border: 2px dashed #667eea;
        padding: 20px;
    }
    
    h1, h2, h3 {
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Modern header
st.markdown("""
    <div style="text-align: center; margin-bottom: 40px;">
        <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-size: 3.5em; font-weight: bold; margin: 0;">
            🏏 Cricket Captain 2025 Dashboard
        </h1>
        <p style="font-size: 1.2em; color: #666; margin-top: 10px;">
            Transform your cricket data into powerful insights
        </p>
    </div>
""", unsafe_allow_html=True)

# Update banner
st.markdown("""
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                padding: 15px; border-radius: 12px; text-align: center; margin: 20px 0;
                box-shadow: 0 6px 20px rgba(0,0,0,0.1);">
        <span style="color: white; font-weight: bold; font-size: 18px;">🎉 NEW UPDATE v1.23</span>
        <br>
        <span style="color: white; font-size: 16px;">Enhanced duplicate player detection & modern UI</span>
    </div>
""", unsafe_allow_html=True)

# Welcome section
st.markdown("""
    <div style="text-align: center; max-width: 1000px; margin: 40px auto;">
        <p style="font-size: 1.1em; line-height: 1.6; color: #444;">
            Welcome to the ultimate Cricket Captain 2025 analytics platform. 
            Upload your match scorecards and unlock comprehensive statistics and performance insights.
        </p>
    </div>
""", unsafe_allow_html=True)

# Modern how-to section
with st.expander("📋 How to Use This Dashboard", expanded=False):
    st.markdown("### Quick Start Guide")
    
    # Choose your method header
    st.info("📋 **Choose Your Method:**")
    
    # Create two columns for the options
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
    
    # Common steps
    st.markdown("**Step 3:** Locate your scorecard files:")
    st.markdown("""
    - **Windows:** `C:\\Users\\[USERNAME]\\AppData\\Roaming\\Childish Things\\Cricket Captain 2025`
    - **Mac:** `~/Library/Containers/com.childishthings.cricketcaptain2025mac/Data/Library/Application Support/Cricket Captain 2025/childish things/cricket captain 2025/saves`
    """)
    
    st.markdown("**Step 4:** Use the file browser below to select your .txt scorecard files")
    st.caption("💡 Tip: Select all files with Ctrl+A (Windows) or Cmd+A (Mac)")
    
    st.markdown("**Step 5:** Click 'Process Scorecards' and explore your data in the various tabs")

# Modern file uploader
st.markdown("### 📁 Upload Your Scorecard Files")
uploaded_files = st.file_uploader(
    "Select your Cricket Captain 2025 scorecard files (.txt)",
    type=['txt'],
    accept_multiple_files=True,
    help="Browse and select multiple .txt files from your Cricket Captain saves folder"
)

# File info display
if uploaded_files:
    estimated_time = len(uploaded_files) * 0.5  # Rough estimate of 0.5 seconds per file
    if estimated_time < 60:
        time_estimate = f"~{estimated_time:.0f} seconds"
    else:
        time_estimate = f"~{estimated_time/60:.1f} minutes"
    
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                    padding: 15px; border-radius: 10px; margin: 15px 0;">
            <strong>📊 Files Selected:</strong> {len(uploaded_files)} scorecard files ready for processing
            <br>
            <strong>⏱️ Estimated Time:</strong> {time_estimate}
        </div>
    """, unsafe_allow_html=True)

# Modern process button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Process Scorecards", use_container_width=True):
        if uploaded_files:
            load_data(uploaded_files)
        else:
            st.warning("⚠️ Please select your scorecard files first")

# Resource links
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

# Second row of buttons
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

# Third row of buttons
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
