import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
from db_connection import get_db_connection  # Ensure this imports your connection function

# --- HEADER MARKDOWN ---
st.markdown(
    """
    <h1 style="text-align: center; color: #006542; font-size: 2em; white-space: nowrap;">Batting Stats</h1>
    """,
    unsafe_allow_html=True
)

# --- DATABASE QUERY ---
# Establish a connection to the database
connection = get_db_connection()

# Fetch unique values for filters
player_id_query = "SELECT DISTINCT player_id FROM bat"
batting_team_query = "SELECT DISTINCT batting_team FROM bat"
batting_position_query = "SELECT DISTINCT batting_position FROM bat"
captain_query = "SELECT DISTINCT captain FROM bat"  
wkt_query = "SELECT DISTINCT wkt FROM bat" 
level_query = "SELECT DISTINCT level FROM matches"
competition_query = "SELECT DISTINCT competition FROM matches"
country_query = "SELECT DISTINCT country FROM matches"
format_query = "SELECT DISTINCT format FROM matches"
competition_year_query = "SELECT DISTINCT competition_year FROM matches"
year_query = "SELECT DISTINCT year FROM matches"

# Load unique values into DataFrames
player_ids = pd.read_sql(player_id_query, connection)
batting_teams = pd.read_sql(batting_team_query, connection)
batting_positions = pd.read_sql(batting_position_query, connection)
captains = pd.read_sql(captain_query, connection)
wkts = pd.read_sql(wkt_query, connection)
levels = pd.read_sql(level_query, connection)
competitions = pd.read_sql(competition_query, connection)
countries = pd.read_sql(country_query, connection)
formats = pd.read_sql(format_query, connection)
competition_years = pd.read_sql(competition_year_query, connection)
years = pd.read_sql(year_query, connection)

# Close the connection after fetching filter values
connection.close()

# --- FILTERS ---
# Add "Select All" option
player_id_options = ['Select All'] + player_ids['player_id'].unique().tolist()
batting_team_options = ['Select All'] + batting_teams['batting_team'].unique().tolist()
batting_position_options = ['Select All'] + batting_positions['batting_position'].unique().tolist()

# Map the values for Wicket Keeper and Captain
wkts['wkt'] = wkts['wkt'].map({1: 'Yes', 0: 'No'})  # Change 1 to Yes and 0 to No
captains['captain'] = captains['captain'].map({1: 'Yes', 0: 'No'})  # Change 1 to Yes and 0 to No

captain_options = ['Select All'] + captains['captain'].unique().tolist()
wkt_options = ['Select All'] + wkts['wkt'].unique().tolist()

level_options = ['Select All'] + levels['level'].unique().tolist()
competition_options = ['Select All'] + competitions['competition'].unique().tolist()
country_options = ['Select All'] + countries['country'].unique().tolist()
format_options = ['Select All'] + formats['format'].unique().tolist()
competition_year_options = ['Select All'] + competition_years['competition_year'].unique().tolist()
year_options = ['Select All'] + years['year'].unique().tolist()

# Use columns to arrange filters in one row
col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 = st.columns(11)

with col1:
    selected_player_id = st.selectbox("Player", options=player_id_options)

with col2:
    selected_batting_team = st.selectbox("Team", options=batting_team_options)

with col3:
    selected_batting_position = st.selectbox("Position", options=batting_position_options)

with col4:
    selected_captain = st.selectbox("Captain", options=captain_options)

with col5:
    selected_wkt = st.selectbox("Wicket Keeper", options=wkt_options)

with col6:
    selected_level = st.selectbox("Level", options=level_options)

with col7:
    selected_competition = st.selectbox("Competition", options=competition_options)

with col8:
    selected_country = st.selectbox("Country", options=country_options)

with col9:
    selected_format = st.selectbox("Format", options=format_options)

with col10:
    selected_competition_year = st.selectbox("Competition Year", options=competition_year_options)

with col11:
    selected_year = st.selectbox("Year", options=year_options)

# --- DATABASE QUERY FOR STATS ---
# Re-establish a connection to the database to fetch batting stats
connection = get_db_connection()

# SQL query with filters,
query = f"""
SELECT 
    b.player_id, 
    COUNT(b.match_id) AS Matches, 
    SUM(b.bat_inns) AS Innings, 
    SUM(b.Dismissed) AS Outs, 
    SUM(b.not_out) AS `Not Out`, 
    SUM(b.bat_runs) AS Runs, 
    SUM(CASE 
        WHEN b.bat_balls >= 1 THEN b.bat_balls 
        ELSE 0 
    END) AS Balls, 
    SUM(CASE 
        WHEN b.bat_balls >= 1 THEN b.bat_runs 
        ELSE 0 
    END) AS runs_balls,     
    SUM(b.bat_4s) AS `4s`, 
    SUM(b.bat_6s) AS `6s`, 
    SUM(b.fiftty) AS `50s`, 
    SUM(b.hundred) AS `100s`, 
    SUM(b.c) AS Caught, 
    SUM(b.b) AS Bowled, 
    SUM(b.lbw) AS LBW, 
    SUM(b.run_out) AS `Run Out`, 
    SUM(b.St) AS Stumped, 
    SUM(b.hw) AS `Hit Wicket`
FROM 
    bat b
JOIN 
    matches m ON b.match_id = m.match_id
WHERE 
    (b.player_id = '{selected_player_id}' OR '{selected_player_id}' = 'Select All') AND
    (b.batting_team = '{selected_batting_team}' OR '{selected_batting_team}' = 'Select All') AND
    (b.batting_position = '{selected_batting_position}' OR '{selected_batting_position}' = 'Select All') AND
    (b.captain = '{selected_captain}' OR '{selected_captain}' = 'Select All') AND
    (b.wkt = '{selected_wkt}' OR '{selected_wkt}' = 'Select All') AND
    (m.level = '{selected_level}' OR '{selected_level}' = 'Select All') AND
    (m.competition = '{selected_competition}' OR '{selected_competition}' = 'Select All') AND
    (m.country = '{selected_country}' OR '{selected_country}' = 'Select All') AND
    (m.format = '{selected_format}' OR '{selected_format}' = 'Select All') AND
    (m.competition_year = '{selected_competition_year}' OR '{selected_competition_year}' = 'Select All') AND
    (m.year = '{selected_year}' OR '{selected_year}' = 'Select All')
GROUP BY 
    b.player_id;
"""

# Execute the query
batting_stats = pd.read_sql(query, connection)

# Close the connection
connection.close()

# --- ADD NEW COLUMNS ---
if not batting_stats.empty:
    # Calculate Average Runs/Balls
    batting_stats['Average'] = batting_stats.apply(
        lambda row: round(row['Runs'] / row['Outs'], 2) if row['Outs'] > 0 else 0, axis=1
    )
    
    # Calculate Strike Rate
    batting_stats['Strike Rate'] = batting_stats.apply(
        lambda row: round((row['runs_balls'] / row['Balls']) * 100, 2) if row['Balls'] > 0 else 0, axis=1
    )

    # Handle Outs for average calculation
    batting_stats['Outs'] = batting_stats['Outs'].replace(0, pd.NA)

    # Sort by Strike Rate in descending order
    batting_stats.sort_values(by='Strike Rate', ascending=False, inplace=True)

    # Drop the runs_balls column
    batting_stats.drop(columns=['runs_balls'], inplace=True)

    # --- DISPLAYING THE DATA ---
    # Set grid options for pagination and visible rows
    grid_options = GridOptionsBuilder.from_dataframe(batting_stats)
    grid_options.configure_grid_options(domLayout='normal')  # Adjust for normal layout
    grid_options.configure_pagination(paginationPageSize=20)  # Set page size to 20
    grid_options.configure_default_column(
        editable=False, 
        filterable=True, 
        sortable=True,
        resizable=True
    )

    # Create the grid with specified height
    AgGrid(
        batting_stats, 
        gridOptions=grid_options.build(),
        editable=False, 
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,  # Enable JS code if needed
        height=600  # Adjust this height as necessary
    )
else:
    st.warning("No data available for the selected filters.")