import streamlit as st
import pandas as pd
from db_connection import get_db_connection

# --- HEADER MARKDOWN ---
st.markdown(
    """
    <h1 style="text-align: center; color: #006542; font-size: 2em; white-space: nowrap;">Player Stats</h1>
    """,
    unsafe_allow_html=True
)

# --- DATABASE QUERY ---
connection = get_db_connection()

# Fetch unique player IDs
player_id_query = "SELECT DISTINCT player_id FROM bat"
player_ids = pd.read_sql(player_id_query, connection)

# Close the connection
connection.close()

# --- SEARCH BAR FOR PLAYER ID ---
# Create a text input for search
search_query = st.text_input("Search Player ID")

# Filter player IDs based on search query
if search_query:
    filtered_player_ids = player_ids[player_ids['player_id'].astype(str).str.contains(search_query, case=False)]
else:
    filtered_player_ids = player_ids

# Sort the filtered player IDs
sorted_player_ids = filtered_player_ids.sort_values(by='player_id')

# Create a select box for unique player IDs
selected_player_id = st.selectbox("Select Player ID", options=sorted_player_ids['player_id'].unique())

# Optional: Display the selected player ID for confirmation
st.write(f"Selected Player ID: {selected_player_id}")

# You can continue to add your additional functionality below this point
