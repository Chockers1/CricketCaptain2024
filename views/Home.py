import streamlit as st
import os
import traceback
import pandas as pd
import tempfile
import time
import shutil
from multiprocessing import Pool, cpu_count

# This is now a simple, direct import because cricketcaptain.py fixed the path.
# This works both locally and in the cloud.
from processing_worker import process_single_file_worker

# --- HELPER FUNCTIONS ---
def process_duplicates(bat_df):
    """Your original duplicate detection function."""
    try:
        duplicates_base = bat_df[['Name', 'Bat_Team_y', 'Year', 'Competition']].copy()
        team_counts = duplicates_base.groupby(['Name', 'Year', 'Competition'])['Bat_Team_y'].nunique()
        multi_team_players = team_counts[team_counts > 1].reset_index()
        # ... (rest of your original duplicate logic)
        has_duplicates = not multi_team_players.empty
        # You can add the full dataframe creation back here if you wish
        return {'has_duplicates': has_duplicates}
    except Exception as e:
        print(f"Could not run duplicate check: {e}")
        return {'has_duplicates': False}

def show_processing_results(total_files, duplicates_result, processing_time=None):
    """Your original function to show results."""
    time_text = f"{processing_time:.1f} seconds" if processing_time and processing_time < 60 else f"{int(processing_time // 60)}m {processing_time % 60:.1f}s"
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 15px; text-align: center; margin: 20px 0; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
            <h2 style="color: white; margin: 0;">🎉 Processing Complete!</h2>
            <p style="color: white; margin: 10px 0; font-size: 18px;">Successfully processed {total_files} scorecards in {time_text}</p>
        </div>
    """, unsafe_allow_html=True)
    if not duplicates_result['has_duplicates']:
        st.markdown("""<div style="background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); padding: 20px; border-radius: 15px; text-align: center; ...">✅ Data Quality Check...</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 15px; text-align: center; ...">⚠️ Duplicate Players Detected...</div>""", unsafe_allow_html=True)

def show_error(message, traceback_info=None):
    """Your original function to show errors."""
    st.markdown(f"""<div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); padding: 20px; ...">❌ Processing Error: {message}</div>""", unsafe_allow_html=True)
    if traceback_info:
        with st.expander("🔧 Technical Details"): st.code(traceback_info)


# --- UI & PAGE LOGIC ---
st.markdown("""<style>...</style>""", unsafe_allow_html=True) # Your full CSS
st.markdown("""<div style="text-align: center; ...">🏏 Cricket Captain 2025 Dashboard</div>""", unsafe_allow_html=True) # Your header
# ... (all your other UI markdown blocks)

st.markdown("### 📁 Upload Your Scorecard Files")
uploaded_files = st.file_uploader("Select your Cricket Captain scorecard files (.txt)", type=['txt'], accept_multiple_files=True)

if uploaded_files:
    # ... your file info display markdown ...
    pass

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Process Scorecards", use_container_width=True):
        if not uploaded_files:
            st.warning("⚠️ Please select your scorecard files first")
        else:
            processing_error = False
            start_time = time.time()
            with st.spinner(f"Processing {len(uploaded_files)} files in parallel across {cpu_count()} CPU cores..."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    try:
                        file_paths = []
                        for uploaded_file in uploaded_files:
                            file_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
                            file_paths.append(file_path)

                        with Pool(processes=cpu_count()) as p:
                            results = p.map(process_single_file_worker, file_paths)
                        
                        valid_results = [res for res in results if res is not None]
                        if not valid_results: raise ValueError("No valid data could be parsed.")

                        all_bat_dfs = [res[0] for res in valid_results if not res[0].empty]
                        all_bowl_dfs = [res[1] for res in valid_results if res[1] is not None and not res[1].empty]
                        all_match_dfs = [res[2] for res in valid_results if not res[2].empty]
                        
                        if not all_bat_dfs: raise ValueError("No valid batting data could be parsed.")

                        final_bat_df = pd.concat(all_bat_dfs, ignore_index=True)
                        final_bowl_df = pd.concat(all_bowl_dfs, ignore_index=True) if all_bowl_dfs else pd.DataFrame()
                        final_match_df = pd.concat(all_match_dfs, ignore_index=True)
                        
                        duplicates_result = process_duplicates(final_bat_df)
                        
                        st.session_state['bat_df'] = final_bat_df
                        st.session_state['bowl_df'] = final_bowl_df
                        st.session_state['match_df'] = final_match_df
                        st.session_state['data_loaded'] = True
                        
                    except Exception as e:
                        processing_error = True
                        show_error(f"A critical error occurred: {e}", traceback.format_exc())

            if not processing_error:
                end_time = time.time()
                show_processing_results(len(uploaded_files), duplicates_result, end_time - start_time)
                st.info("Navigate to other views from the sidebar to see your stats.")

# ... (rest of your UI, like resource links) ...

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
