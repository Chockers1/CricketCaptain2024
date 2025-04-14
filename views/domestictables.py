from operator import index
import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import io
import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px


# Title
st.markdown("<h1 style='color:#f04f53; text-align: center;'>Domestic Tables</h1>", unsafe_allow_html=True)
# Add file upload widget (only once)
st.markdown("### Upload Previous Tables (Optional) this is a tables.csv file that saves into Scorecard file")
uploaded_file = st.file_uploader("Upload your tables CSV file", type=['csv'])

# Initialize session states
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = pd.DataFrame()

if 'manual_points' not in st.session_state:
    st.session_state.manual_points = {}

# Load data from uploaded file if it exists
if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        st.session_state.historical_data = uploaded_df
        st.success("Tables loaded successfully!")
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

# Dropdown options
COUNTRIES = [
    "Australia", 
    "Bangladesh", 
    "England", 
    "India", 
    "New Zealand", 
    "Pakistan", 
    "South Africa", 
    "West Indies", 

]

# England Configurations
ENGLAND_COMPETITIONS = [
    "County Championship",
    "Royal London One Day Cup",
    "T20 Blast",
    "The Hundred"
]

ENGLAND_TEAMS = [
    "Derbyshire",
    "Durham", 
    "Essex",
    "Glamorgan",
    "Gloucestershire",
    "Hampshire",
    "Kent",
    "Lancashire",
    "Leicestershire",
    "Middlesex",
    "Northamptonshire",
    "Nottinghamshire",
    "Somerset",
    "Surrey",
    "Sussex",
    "Warwickshire",
    "Worcestershire",
    "Yorkshire"
]

HUNDRED_TEAMS = [
    "Northern Superchargers", "London Spirit", "Birmingham Phoenix",
    "Welsh Fire", "Trent Rockets", "Manchester Originals",
    "Southern Brave", "Oval Invincibles"
]

# Australia Configurations
AUSTRALIA_COMPETITIONS = ["Sheffield Shield", "One-Day Cup", "Big Bash League"]

AUSTRALIA_SHIELD_TEAMS = [
    "New South Wales", "Queensland", "South Australia",
    "Tasmania", "Victoria", "Western Australia"
]

BBL_TEAMS = [
    "Adelaide Strikers", "Brisbane Heat", "Hobart Hurricanes",
    "Melbourne Renegades", "Melbourne Stars", "Perth Scorchers",
    "Sydney Sixers", "Sydney Thunder"
]

# New Zealand Configurations
NZ_COMPETITIONS = ["Plunket Shield", "Ford Trophy", "Super Smash"]

NZ_DOMESTIC_TEAMS = [
    "Auckland", "Canterbury", "Central Districts",
    "Northern Districts", "Otago", "Wellington"
]

# West Indies Configurations
WINDIES_COMPETITIONS = ["Regional 4 Day Competition", "Super 50 Cup", "Caribbean Premier League"]

WINDIES_DOMESTIC_TEAMS = [
    "Barbados CC", "Guyana CC", "Jamaica CC", "Leeward Islands CC",
    "Trinidad and Tobago CC", "Windward Islands CC",
    "Combined Campuses and Colleges CC", "West Indies Academy"
]

CPL_TEAMS = [
    "Barbados Royals", "Guyana Amazon Warriors",
    "St Kitts & Nevis Patriots", "Saint Lucia Kings",
    "Trinbago Knight Riders", "Antigua & Barbuda Falcons"
]

# Pakistan Configurations
PAKISTAN_COMPETITIONS = [
    "Quaid-e-Azam Trophy", "Pakistan One Day Cup",
    "National T20 Cup", "Pakistan Super League"
]

PAKISTAN_DOMESTIC_TEAMS = [
    "Peshawar CC", "Karachi Whites", "Multan CC", "Tribal Areas",
    "Rawalpindi CC", "Faisalabad", "Lahore Blues", "Lahore Whites"
]

PSL_TEAMS = [
    "Islamabad United", "Karachi Kings", "Lahore Qalandars",
    "Multan Sultans", "Peshawar Zalmi", "Quetta Gladiators"
]

# Bangladesh Configurations
BANGLADESH_COMPETITIONS = ["Bangladesh Premier League"]

BPL_TEAMS = [
    "Chittagong Kings", "Dhaka Capitals", "Fortune Barishal",
    "Khulna Tigers", "Rangpur Riders", "Sylhet Strikers",
    "Durbar Rajshahi"
]

# South Africa Configurations
SA_COMPETITIONS = [
    "Sunfoil Series", "CSA One-Day Cup", "CSA T20 Challenge",
    "SA20", "Provincial T20"
]

SA_DOMESTIC_TEAMS = [
    "Boland", "Central Gauteng", "Eastern Province",
    "KwaZulu-Natal Coastal", "KwaZulu-Natal Inland", "North West",
    "Northerns", "Western Province", "Free State", "Border",
    "Easterns", "Limpopo", "Mpumalanga", "Northern Cape",
    "South Western Districts"
]

SA_PROVINCIAL_T20_TEAMS = [
    "Free State", "Border", "Easterns", "Limpopo",
    "Mpumalanga", "Northern Cape", "South Western Districts"
]

SA20_TEAMS = [
    "Durban's Super Giants", "Joburg Super Kings", "MI Cape Town",
    "Paarl Royals", "Pretoria Capitals", "Sunrisers Eastern Cape"
]

# India Configurations
INDIA_COMPETITIONS = [
    "Ranji Trophy",
    "FC Plate",
    "Vijay Hazare One Day Trophy",
    "Syed Mushtaq Ali T20 Trophy",
    "IPL"
]

IPL_TEAMS = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad"
]

INDIA_DOMESTIC_TEAMS = [
    "Arunachal Pradesh",
    "Assam",
    "Baroda",
    "Bengal",
    "Bihar",
    "Chandigarh",
    "Chhattisgarh",
    "Delhi",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Hyderabad",
    "Jammu and Kashmir",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Mumbai",
    "Nagaland",
    "Odisha",
    "Pondicherry",
    "Punjab",
    "Railways",
    "Rajasthan",
    "Saurashtra",
    "Services",
    "Sikkim",
    "Tamil Nadu",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "Vidarbha"
]

FC_PLATE_TEAMS = [
    "Meghalaya",
    "Sikkim",
    "Nagaland",
    "Hyderabad CC",
    "Mizoram",
    "Arunachal Pradesh"
]

# Initialize session states
if 'historical_data' not in st.session_state:
    if os.path.exists(SAVE_PATH):
        st.session_state.historical_data = pd.read_csv(SAVE_PATH)
    else:
        st.session_state.historical_data = pd.DataFrame()

if 'manual_points' not in st.session_state:
    st.session_state.manual_points = {}

# Title
st.markdown("<h1 style='color:#f04f53; text-align: center;'>Domestic Tables  (Add new table) </h1>", unsafe_allow_html=True)

# Year, Country, and Competition selection (filters first)
col1, col2, col3 = st.columns(3)
with col1:
    selected_year = st.number_input("Year", min_value=1900, max_value=2100, value=datetime.now().year, step=1)
with col2:
    selected_country = st.selectbox("Country", COUNTRIES)
with col3:
# When England is selected and County Championship is chosen
    if selected_country == "England":
        selected_competition = st.selectbox("Competition", ENGLAND_COMPETITIONS)
        if selected_competition == "County Championship":
            default_played = 14
            teams = ENGLAND_TEAMS
            num_positions = 18
            final_position_options = [
                "Group Stage",
                "D1 Winner",
                "D1 Runner Up",
                "Relegated",
                "D2 Winner",
                "D2 Runner Up",
                "Wooden Spoon"
            ]
            division_options = ["Division 1", "Division 2"]
        elif selected_competition == "Royal London One Day Cup":
            default_played = 8
            num_positions = 18
            teams = ENGLAND_TEAMS
            final_position_options = ["Group Stage", "Quater Final", "Semi Final", "Runner Up", "Winner"]
            division_options = ["A", "B"]
        elif selected_competition == "T20 Blast":
            default_played = 14
            num_positions = 18
            teams = ENGLAND_TEAMS
            final_position_options = ["Group Stage", "Quater Final", "Semi Final", "Runner Up", "Winner"]
            division_options = ["North", "South"]
        else:  # The Hundred
            default_played = 8
            num_positions = 8
            teams = HUNDRED_TEAMS
            final_position_options = ["Group Stage", "Preliminary Final", "Runner Up", "Winner"]

    elif selected_country == "Australia":
            selected_competition = st.selectbox("Competition", AUSTRALIA_COMPETITIONS)
            if selected_competition == "Sheffield Shield":
                default_played = 10
                teams = AUSTRALIA_SHIELD_TEAMS
                num_positions = 6
                final_position_options = ["Group Stage", "Runner Up", "Winner"]
                division = ""
            elif selected_competition == "One-Day Cup":
                default_played = 7
                teams = AUSTRALIA_SHIELD_TEAMS
                num_positions = 6
                final_position_options = ["Group Stage", "Runner Up", "Winner"]
                division = ""
                auto_calculate_points = True  # Flag for automatic point calculation
            else:  # Big Bash League
                default_played = 10
                teams = BBL_TEAMS
                num_positions = 8
                final_position_options = ["Group Stage", "Qualifier", "Runner Up", "Winner"]
                division = ""
                auto_calculate_points = True  # Flag for automatic point calculation

    elif selected_country == "New Zealand":
        selected_competition = st.selectbox("Competition", NZ_COMPETITIONS)
        if selected_competition == "Plunket Shield":
            default_played = 10
            teams = NZ_DOMESTIC_TEAMS
            num_positions = 6
            final_position_options = ["Group Stage", "Runner Up", "Winner"]
            division = ""
        elif selected_competition == "Ford Trophy":
            default_played = 10
            teams = NZ_DOMESTIC_TEAMS
            num_positions = 6
            final_position_options = ["Group Stage", "Preliminary Final", "Runner Up", "Winner"]
            division = ""
        else:  # Super Smash
            default_played = 10
            teams = NZ_DOMESTIC_TEAMS
            num_positions = 6
            final_position_options = ["Group Stage", "Elimination Final", "Grand Final", "Winner"]
            division = ""

    elif selected_country == "West Indies":
        selected_competition = st.selectbox("Competition", WINDIES_COMPETITIONS)
        if selected_competition == "Regional 4 Day Competition":
            default_played = 7
            teams = WINDIES_DOMESTIC_TEAMS
            num_positions = 8
            final_position_options = None
            division = ""
        elif selected_competition == "Super 50 Cup":
            default_played = 7
            teams = WINDIES_DOMESTIC_TEAMS
            num_positions = 8
            final_position_options = ["Group Stage", "Semi Final", "Runner Up", "Winner"]
            division = ""
        else:  # Caribbean Premier League
            default_played = 10
            teams = CPL_TEAMS
            num_positions = 6
            final_position_options = ["Group Stage", "Eliminator", "Qualifier", "Runner Up", "Winner"]
            division = ""

    elif selected_country == "Pakistan":
        selected_competition = st.selectbox("Competition", PAKISTAN_COMPETITIONS)
        if selected_competition == "Quaid-e-Azam Trophy":
            default_played = 10
            teams = PAKISTAN_DOMESTIC_TEAMS
            num_positions = 8
            final_position_options = ["Group Stage", "Runner Up", "Winner"]
            division = ""
        elif selected_competition == "Pakistan One Day Cup":
            default_played = 7
            teams = PAKISTAN_DOMESTIC_TEAMS
            num_positions = 8
            final_position_options = ["Group Stage", "Semi Final", "Runner Up", "Winner"]
            division = ""
        elif selected_competition == "National T20 Cup":
            default_played = 10
            teams = PAKISTAN_DOMESTIC_TEAMS
            num_positions = 8
            final_position_options = ["Group Stage", "Semi Final", "Runner Up", "Winner"]
            division = ""
        else:  # Pakistan Super League
            default_played = 10
            teams = PSL_TEAMS
            num_positions = 6
            final_position_options = ["Group Stage", "Eliminator 1", "Eliminator 2", "Qualifier", "Runner Up", "Winner"]
            division = ""

    elif selected_country == "Bangladesh":
        selected_competition = st.selectbox("Competition", BANGLADESH_COMPETITIONS)
        default_played = 12
        teams = BPL_TEAMS
        num_positions = 7
        final_position_options = ["Group Stage", "Eliminator", "Qualifier 1", "Qualifier 2", "Runner Up", "Winner"]
        division = ""

    elif selected_country == "South Africa":
        selected_competition = st.selectbox("Competition", SA_COMPETITIONS)
        if selected_competition == "Sunfoil Series":
            default_played = 14
            teams = SA_DOMESTIC_TEAMS
            num_positions = 15
            final_position_options = ["Group Stage", "Div 1 Winner", "Div 1 Runner Up", "D2 Playoff Winner", "D2 Playoff Runner Up"]
            division_options = ["Division 1", "Division 2"]
        elif selected_competition == "CSA One-Day Cup":
            default_played = 10
            teams = SA_DOMESTIC_TEAMS
            num_positions = 15
            final_position_options = ["Group Stage", "Div 1 Winner", "Div 1 Runner Up", "D2 Playoff Winner", "D2 Playoff Runner Up"]
            division_options = ["Division 1", "Division 2"]
        elif selected_competition == "CSA T20 Challenge":
            default_played = 10
            teams = SA_DOMESTIC_TEAMS
            num_positions = 15
            final_position_options = ["Group Stage", "Semi Final", "Runner Up", "Winner"]
            division = ""
        elif selected_competition == "Provincial T20":
            default_played = 6
            teams = SA_PROVINCIAL_T20_TEAMS
            num_positions = 7
            final_position_options = ["Group Stage", "Semi Final", "Runner Up", "Winner"]
            division = ""
        else:  # SA20
            default_played = 10
            teams = SA20_TEAMS
            num_positions = 6
            final_position_options = ["Group Stage", "Qualifier 1", "Qualifier 2", "Eliminator", "Runner Up", "Winner"]
            division = ""

    # Add this to your country selection logic:
    elif selected_country == "India":
        selected_competition = st.selectbox("Competition", INDIA_COMPETITIONS)
        if selected_competition == "Ranji Trophy":
            default_played = 7
            teams = INDIA_DOMESTIC_TEAMS
            num_positions = 38
            final_position_options = ["Group Stage", "Quarter Final", "Semi Final", "Runner Up", "Winner"]
            division_options = ["Elite Group A", "Elite Group B", "Elite Group C", "Elite Group D"]
        elif selected_competition == "FC Plate":
            default_played = 5
            teams = FC_PLATE_TEAMS
            num_positions = 6
            final_position_options = ["Group Stage", "Semi Final", "Runner Up", "Winner"]
            division = ""
        elif selected_competition == "Vijay Hazare One Day Trophy":
            default_played = 7
            teams = INDIA_DOMESTIC_TEAMS
            num_positions = 38
            final_position_options = ["Group Stage", "Preliminary Quarter Final", "Quarter Final", "Semi Final", "Runner Up", "Winner"]
            division_options = ["Elite Group A", "Elite Group B", "Elite Group C", "Elite Group D", "Elite Group E"]
        elif selected_competition == "Syed Mushtaq Ali T20 Trophy":
            default_played = 7
            teams = INDIA_DOMESTIC_TEAMS
            num_positions = 38
            final_position_options = ["Group Stage", "Preliminary Quarter Final", "Quarter Final", "Semi Final", "Runner Up", "Winner"]
            division_options = ["Elite Group A", "Elite Group B", "Elite Group C", "Elite Group D", "Elite Group E"]
        else:  # IPL
            default_played = 14
            teams = IPL_TEAMS
            num_positions = 10
            final_position_options = ["Group Stage", "Eliminator", "Qualifier 1", "Qualifier 2", "Runner Up", "Winner"]

    else:
        selected_competition = st.selectbox("Competition", OTHER_COMPETITIONS)
        default_played = None
        teams = COUNTRIES
        num_positions = 18
        final_position_options = None
        division_options = None

# Initialize empty data for input table
table_data = {
    'Position': [], 'Team': [], 'P': [], 'W': [], 'L': [], 'D': [],
    'Bat BP': [], 'Bowl BP': [], 'Points': [], 'Year': [],
    'Country': [], 'Competition': [], 'Division': [], 'Final Position': []
}

# Now add column headers below the filters
header_cols = st.columns([1, 3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1])
header_cols[0].markdown("**Position**")
header_cols[1].markdown("**Team**")
header_cols[2].markdown("**Played**")
header_cols[3].markdown("**Wins**")
header_cols[4].markdown("**Losses**")
header_cols[5].markdown("**Draws**")
header_cols[6].markdown("**Bat BP**")
header_cols[7].markdown("**Bowl BP**")
header_cols[8].markdown("**Points**")
header_cols[9].markdown("**Division**")
header_cols[10].markdown("**Final Position**")

# Add a small space between headers and data rows
st.markdown("<br>", unsafe_allow_html=True)

# Create rows for each position
# Create rows for each position
for pos in range(1, num_positions + 1):
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 = st.columns([1, 3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1])
    
    with col1:
        st.markdown(f"**{pos}**")
    with col2:
        team = st.selectbox(f"Team {pos}", teams, key=f"team_{pos}")
    with col3:
        p = st.number_input("", min_value=0, step=1, value=default_played if default_played else 0, key=f"played_{pos}")
    with col4:
        w = st.number_input("", min_value=0, step=1, key=f"wins_{pos}")
    with col5:
        l = st.number_input("", min_value=0, step=1, key=f"losses_{pos}")
    with col6:
        d = st.number_input("", min_value=0, step=1, key=f"draws_{pos}")
    with col7:
        bat_bp = st.number_input("", min_value=0.0, step=0.1, key=f"bat_bp_{pos}")
    with col8:
        bowl_bp = st.number_input("", min_value=0.0, step=0.1, key=f"bowl_bp_{pos}")
    with col9:
        points_key = f"points_{pos}"
        # Check for Australian limited-overs competitions specifically
        is_aus_limited_overs = (
            selected_country == "Australia" and 
            selected_competition in ["One-Day Cup", "Big Bash League"]
        )
        
        # Check for other limited-overs competitions
        is_other_limited_overs = selected_competition in [
            "Royal London One Day Cup", "T20 Blast", "The Hundred",
            "Super 50 Cup", "CPL", "PSL", "BPL", "SA20"
        ]
        
        # If it's any limited-overs competition, calculate points automatically
        if is_aus_limited_overs or is_other_limited_overs:
            base_points = (w * 2) + d  # 2 points for win, 1 for draw
            
            # Initialize manual points adjustment if not exists
            if points_key not in st.session_state.manual_points:
                st.session_state.manual_points[points_key] = 0
            
            # Create number input with calculated base value
            points = st.number_input(
                "",
                value=float(base_points + st.session_state.manual_points[points_key]),
                step=0.1,
                key=points_key,
                format="%.1f"
            )
            
            # Store any manual adjustments
            st.session_state.manual_points[points_key] = points - base_points
        
        else:
            # For non-limited-overs competitions, just show regular input
            points = st.number_input(
                "",
                min_value=0.0,
                step=0.1,
                key=points_key,
                format="%.1f"
            )
    
    with col10:
        if 'division_options' in locals():
            division = st.selectbox(
                "",
                options=division_options,
                key=f"division_{pos}"
            )
        elif 'division' in locals():
            st.write(division)
        else:
            division = ""
            st.write("")

    # The rest of your column handling remains the same
    with col11:
        if final_position_options:
            final_position = st.selectbox(
                f"Final Position",
                options=final_position_options,
                key=f"final_position_{pos}"
            )
        else:
            final_position = ""
            st.write("")

    # Append data to table_data
    table_data['Position'].append(pos)
    table_data['Team'].append(team)
    table_data['P'].append(p)
    table_data['W'].append(w)
    table_data['L'].append(l)
    table_data['D'].append(d)
    table_data['Bat BP'].append(bat_bp)
    table_data['Bowl BP'].append(bowl_bp)
    table_data['Points'].append(points)
    table_data['Year'].append(selected_year)
    table_data['Country'].append(selected_country)
    table_data['Competition'].append(selected_competition)
    table_data['Division'].append(division if division else "")
    table_data['Final Position'].append(final_position if final_position_options else "")

# Create columns for buttons
col1, col2, col3 = st.columns(3)

# Save the data
with col1:
    if st.button("Save Table"):
        try:
            df = pd.DataFrame(table_data)
            if not st.session_state.historical_data.empty:
                df = pd.concat([st.session_state.historical_data, df], ignore_index=True)
            
            # Update session state
            st.session_state.historical_data = df
            
            # Create download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Tables CSV",
                data=csv,
                file_name=f"tables_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv'
            )
            
            st.success("Table saved successfully!")
            
        except Exception as e:
            st.error(f"Error preparing data: {str(e)}")



# Enter Another Year button
with col2:
    if st.button("Enter Another Year"):
        try:
            # Save current data first
            df = pd.DataFrame(table_data)
            if not st.session_state.historical_data.empty:
                df = pd.concat([st.session_state.historical_data, df], ignore_index=True)
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            df.to_csv(SAVE_PATH, index=False)
            st.session_state.historical_data = df
            
            # Clear manual points adjustments
            st.session_state.manual_points = {}
            
            # Clear all the input fields by forcing a rerun
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")

# Clear Historical Tables button
with col3:
    if st.button("Clear Historical Tables"):
        try:
            # Clear the session state
            st.session_state.historical_data = pd.DataFrame()
            st.session_state.manual_points = {}
            
            st.success("Historical tables cleared successfully!")
            
            # Force a rerun to clear the interface
            st.rerun()
        except Exception as e:
            st.error(f"Error clearing data: {str(e)}")

# Display historical data if it exists
if not st.session_state.historical_data.empty:
    st.markdown("<h3 style='color:#f04f53; text-align: center;'>Historical Data</h3>", unsafe_allow_html=True)
    
    # Create lists for filters
    teams = ['All'] + sorted(st.session_state.historical_data['Team'].unique().tolist())
    competitions = ['All'] + sorted(st.session_state.historical_data['Competition'].unique().tolist())
    
    # Create filters at the top of the page using two columns
    col1, col2 = st.columns(2)
    with col1:
        team_choice = st.multiselect('Team:', teams, default='All', key="team_filter_main")
    with col2:
        competition_choice = st.multiselect('Competition:', competitions, default='All', key="competition_filter_main")
        
    # Apply filters to the dataframe
    filtered_df = st.session_state.historical_data.copy()
    
    if 'All' not in team_choice:
        filtered_df = filtered_df[filtered_df['Team'].isin(team_choice)]
        
    if 'All' not in competition_choice:
        filtered_df = filtered_df[filtered_df['Competition'].isin(competition_choice)]
    
    # Display the filtered dataframe
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    
    # Add the missing summaries
    st.markdown("<h3 style='color:#f04f53; text-align: center;'>Historical Summary</h3>", unsafe_allow_html=True)
    
    # Create two columns for the summaries
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
        st.markdown("#### Summary By Format")
        
        # Group by Team first, then Competition as requested
        format_summary = filtered_df.groupby(['Team', 'Competition']).agg({
            'Year': 'count',  # Count of entries for this team-competition
            'P': 'sum',
            'W': 'sum',
            'L': 'sum',
            'D': 'sum',
            'Bat BP': 'sum',
            'Bowl BP': 'sum',
            'Points': 'sum'
        }).reset_index()
        
        # Calculate percentages
        format_summary['Win %'] = (format_summary['W'] / format_summary['P'] * 100).round(1)
        format_summary['Loss %'] = (format_summary['L'] / format_summary['P'] * 100).round(1)
        format_summary['Draw %'] = (format_summary['D'] / format_summary['P'] * 100).round(1)
        
        # Count titles, runner ups, and wooden spoons by team and competition
        titles_by_team_comp = filtered_df[filtered_df['Final Position'].str.contains('Winner', na=False)].groupby(['Team', 'Competition']).size().reset_index(name='Titles')
        runner_ups_by_team_comp = filtered_df[filtered_df['Final Position'].str.contains('Runner Up', na=False)].groupby(['Team', 'Competition']).size().reset_index(name='Runner Ups')
        wooden_spoons_by_team_comp = filtered_df[filtered_df['Final Position'].str.contains('Wooden Spoon', na=False)].groupby(['Team', 'Competition']).size().reset_index(name='Wooden Spoons')
        
        # Merge these into format_summary
        format_summary = pd.merge(format_summary, titles_by_team_comp, on=['Team', 'Competition'], how='left')
        format_summary = pd.merge(format_summary, runner_ups_by_team_comp, on=['Team', 'Competition'], how='left')
        format_summary = pd.merge(format_summary, wooden_spoons_by_team_comp, on=['Team', 'Competition'], how='left')
        
        # Fill NA values with 0
        format_summary['Titles'] = format_summary['Titles'].fillna(0).astype(int)
        format_summary['Runner Ups'] = format_summary['Runner Ups'].fillna(0).astype(int)
        format_summary['Wooden Spoons'] = format_summary['Wooden Spoons'].fillna(0).astype(int)
        
        # Rename columns for clarity
        format_summary = format_summary.rename(columns={
            'Year': 'Entries'
        })
        
        # Sort by Team and then Competition
        format_summary = format_summary.sort_values(['Team', 'Competition'])
        
        st.dataframe(format_summary, use_container_width=True, hide_index=True)
    
    with summary_col2:
        st.markdown("#### Summary By Team (Competition)")
        
        # Group by team AND competition - change from the previous version
        team_summary = filtered_df.groupby(['Team', 'Competition']).agg({
            'Year': 'nunique',
            'Position': ['mean', 'min'],
            'Points': ['mean', 'sum'],
            'W': ['sum'],
            'P': ['sum']
        }).reset_index()
        
        # Flatten the column names
        team_summary.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in team_summary.columns]
        
        # Rename columns for clarity
        team_summary = team_summary.rename(columns={
            'Year_nunique': 'Years',
            'Position_mean': 'Avg Pos',
            'Position_min': 'Best Pos',
            'Points_mean': 'Avg Pts',
            'Points_sum': 'Total Pts',
            'W_sum': 'Wins',
            'P_sum': 'Played'
        })
        
        # Calculate win percentage
        team_summary['Win %'] = (team_summary['Wins'] / team_summary['Played'] * 100).round(1)
        
        # Add Titles, Runner Ups, and Wooden Spoons columns for each team-competition combo
        titles_by_team_comp = filtered_df[filtered_df['Final Position'].str.contains('Winner', na=False)].groupby(['Team', 'Competition']).size().reset_index(name='Titles')
        runner_ups_by_team_comp = filtered_df[filtered_df['Final Position'].str.contains('Runner Up', na=False)].groupby(['Team', 'Competition']).size().reset_index(name='Runner Ups')
        wooden_spoons_by_team_comp = filtered_df[filtered_df['Final Position'].str.contains('Wooden Spoon', na=False)].groupby(['Team', 'Competition']).size().reset_index(name='Wooden Spoons')
        
        # Merge these into team_summary
        team_summary = pd.merge(team_summary, titles_by_team_comp, on=['Team', 'Competition'], how='left')
        team_summary = pd.merge(team_summary, runner_ups_by_team_comp, on=['Team', 'Competition'], how='left')
        team_summary = pd.merge(team_summary, wooden_spoons_by_team_comp, on=['Team', 'Competition'], how='left')
        
        # Fill NA values with 0
        team_summary['Titles'] = team_summary['Titles'].fillna(0).astype(int)
        team_summary['Runner Ups'] = team_summary['Runner Ups'].fillna(0).astype(int)
        team_summary['Wooden Spoons'] = team_summary['Wooden Spoons'].fillna(0).astype(int)
        
        # Round numeric columns
        team_summary[['Avg Pos', 'Avg Pts', 'Total Pts', 'Win %']] = team_summary[['Avg Pos', 'Avg Pts', 'Total Pts', 'Win %']].round(2)
        
        # Sort by team name and then average position
        team_summary = team_summary.sort_values(['Team', 'Avg Pos'])
        
        st.dataframe(team_summary, use_container_width=True, hide_index=True)
    
    # Add styled header for the graph
    st.markdown("<h3 style='color:#f04f53; text-align: center;'>Position History</h3>", unsafe_allow_html=True)
    
    # Create new filters for position history section with unique keys
    position_col1, position_col2 = st.columns(2)
    with position_col1:
        position_team_choice = st.multiselect('Team:', teams, default='All', key="team_filter_position")
    with position_col2:
        position_competition_choice = st.multiselect('Competition:', competitions, default='All', key="competition_filter_position")
    
    # Apply position history filters
    position_filtered_df = st.session_state.historical_data.copy()
    
    if 'All' not in position_team_choice:
        position_filtered_df = position_filtered_df[position_filtered_df['Team'].isin(position_team_choice)]
        
    if 'All' not in position_competition_choice:
        position_filtered_df = position_filtered_df[position_filtered_df['Competition'].isin(position_competition_choice)]
    
    # Prepare data for position history plotting
    if 'All' not in position_team_choice:
        plot_teams = position_team_choice
    else:
        # If 'All' is selected, show all teams
        plot_teams = position_filtered_df['Team'].unique().tolist()
    
    # Filter data for selected teams
    plot_df = position_filtered_df[position_filtered_df['Team'].isin(plot_teams)]
    
    # Create the position history line plot
    if not plot_df.empty:
        fig = px.line(plot_df, 
                     x='Year', 
                     y='Position',
                     color='Team',
                     markers=True,
                     hover_data={
                         'Year': True,
                         'Team': True,
                         'Position': True,
                         'Competition': True
                     })
        
        # Customize the layout and hover
        fig.update_layout(
            yaxis_title='Position',
            yaxis_autorange='reversed',  # Reverse Y-axis so position 1 is at the top
            xaxis_title='Year',
            height=600,
            showlegend=True,
            hovermode='closest',
            hoverlabel=dict(
                bgcolor="rgba(0,0,0,0)",
                font_size=14
            )
        )
        
        # Update hover template
        fig.update_traces(
            hovertemplate="<b>%{customdata[1]}</b><br>" +
                         "Year: %{x}<br>" +
                         "Position: %{y}<br>" +
                         "<extra></extra>"
        )
        
        # Set x-axis to show only full years
        fig.update_xaxes(
            dtick=1,  # Show every year
            type='category'  # Use categorical axis to display all years properly
        )
        
        # Set y-axis to show only whole numbers
        fig.update_yaxes(
            dtick=1,
            tickmode='linear'
        )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True, key="position_history_main")
    else:
        st.info("No data available for the selected filters.")
        
    # Remove the duplicate dataframe display that was appearing twice
    # (The second st.dataframe call for the same filtered_df was removed)

# Custom CSS for styling
st.markdown("""
    <style>
    /* Table styling */
    table { color: black; width: 100%; }
    thead tr th {
        background-color: #f04f53 !important;
        color: white !important;
    }
    tbody tr:nth-child(even) { background-color: #f0f2f6; }
    tbody tr:nth-child(odd) { background-color: white; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        width: 100%;
        display: flex;
        justify-content: space-between; /* This will space out the tabs evenly */
    }
    .stTabs [data-baseweb="tab"] {
        flex-grow: 1;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# After the existing position history graph, add new visualizations
if not st.session_state.historical_data.empty:
    # Create all tabs at once including the new heatmap tab
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Performance Metrics", "Season Points Comparison", "Points Distribution", "Current Rankings", "Heatmap"])

    with tab1:
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Team Performance Metrics</h3>", unsafe_allow_html=True)
        performance_df = filtered_df.copy()
        performance_df['Win_Rate'] = (performance_df['W'] / performance_df['P'] * 100).round(2)
        fig_metrics = px.line(performance_df, x='Year', y='Win_Rate', color='Team', markers=True)
        # Update x-axis settings to show full years only
        fig_metrics.update_xaxes(
            dtick=1,  # Show every year
            type='category'  # Use categorical axis
        )
        fig_metrics.update_layout(height=500)
        st.plotly_chart(fig_metrics, use_container_width=True, key="performance_metrics_tab")

    with tab2:
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Season Points Comparison</h3>", unsafe_allow_html=True)
        season_df = filtered_df.pivot_table(values='Points', index='Team', columns='Year', aggfunc='sum').fillna(0)
        fig_season = px.imshow(season_df, labels=dict(x="Year", y="Team", color="Points"), aspect="auto")
        
        # Update layout and ensure x-axis shows only whole numbers
        fig_season.update_layout(height=600)
        fig_season.update_xaxes(
            dtick=1,  # Show every year
            tickmode='linear',  # Use linear tick mode
            type='category'  # Use categorical axis to ensure whole numbers
        )
        
        st.plotly_chart(fig_season, use_container_width=True, key="season_points_tab")

    with tab3:
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Points Distribution</h3>", unsafe_allow_html=True)
        fig_box = px.box(filtered_df, x='Team', y='Points')
        fig_box.update_layout(height=500, xaxis_title="Team", yaxis_title="Points", showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True, key="points_dist_tab")

    with tab4:
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Current Rankings</h3>", unsafe_allow_html=True)
        
        # Calculate rankings for all teams
        rankings_df = filtered_df.copy()
        
        # Calculate win percentage
        rankings_df['Win_Percentage'] = (rankings_df['W'] / rankings_df['P'] * 100).round(2)

        # Initialize points columns
        rankings_df['Position_Points'] = 0
        rankings_df['Win_Percentage_Bonus'] = 0
        
        # Assign position points
        position_points = {
            'Winner': 50,
            'Runner Up': 40,
            'Semi Final': 30,
            'Qualifier': 30,
            'Quarter Final': 20,
            'Eliminator': 20,
            'Group Stage': 10,
            'D1 Winner': 50,
            'D2 Winner': 45,
            'D1 Runner Up': 40,
            'D2 Runner Up': 35,
            'Preliminary Final': 25,
            'Eliminator 1': 20,
            'Eliminator 2': 25,
            'Qualifier 1': 30,
            'Qualifier 2': 25
        }

        # Calculate position points
        for position, points in position_points.items():
            rankings_df.loc[rankings_df['Final Position'] == position, 'Position_Points'] = points

        # Subtract points for lower positions
        rankings_df['Position_Points'] = rankings_df['Position_Points'] - ((rankings_df['Position'] - 1) * 2)

        # Calculate win percentage bonus
        win_percentage_bonus = [
            (100, 50), (90, 40), (80, 30), (70, 25),
            (60, 20), (50, 15), (40, 10), (30, 7),
            (20, 5), (10, 3)
        ]

        for threshold, bonus in win_percentage_bonus:
            rankings_df.loc[rankings_df['Win_Percentage'] >= threshold, 'Win_Percentage_Bonus'] = bonus

        # Calculate total ranking points for each entry
        rankings_df['Total_Ranking_Points'] = rankings_df['Position_Points'] + rankings_df['Win_Percentage_Bonus']

        # Get unique years in descending order
        years = sorted(rankings_df['Year'].unique(), reverse=True)
        
        if len(years) >= 1:
            # Current year points (100%)
            current_year = rankings_df[rankings_df['Year'] == years[0]]
            current_points = current_year.groupby('Team')['Total_Ranking_Points'].sum()
            
            # Previous year points (50%)
            prev_year_points = pd.Series(0, index=current_points.index)
            if len(years) >= 2:
                prev_year = rankings_df[rankings_df['Year'] == years[1]]
                prev_year_points = prev_year.groupby('Team')['Total_Ranking_Points'].sum() * 0.5
            
            # Two years ago points (33.3%)
            two_years_points = pd.Series(0, index=current_points.index)
            if len(years) >= 3:
                two_years = rankings_df[rankings_df['Year'] == years[2]]
                two_years_points = two_years.groupby('Team')['Total_Ranking_Points'].sum() * 0.333

            # Combine all points
            team_rankings = pd.DataFrame({
                'Current Season Points': current_points,
                'Previous Season Points': prev_year_points,
                'Two Seasons Ago Points': two_years_points
            }).fillna(0)

            # Calculate weighted total
            team_rankings['Total Weighted Points'] = team_rankings.sum(axis=1)

            # Add additional statistics
            team_stats = rankings_df.groupby('Team').agg({
                'Win_Percentage': 'mean',
                'Position': 'mean',
                'W': 'sum',
                'P': 'sum',
                'Final Position': lambda x: x.value_counts().iloc[0] if not x.empty else 'N/A'
            }).round(2)

            # Calculate overall win percentage
            team_stats['Overall Win %'] = (team_stats['W'] / team_stats['P'] * 100).round(2)

            # Merge with team_rankings and sort
            final_rankings = pd.concat([team_rankings, team_stats], axis=1)
            final_rankings = final_rankings.sort_values('Total Weighted Points', ascending=False)
            
            # Add rank column first
            final_rankings = final_rankings.reset_index()
            final_rankings.insert(0, 'Rank', range(1, len(final_rankings) + 1))
            
            # Rename 'index' to 'Team' and reorder columns
            final_rankings = final_rankings.rename(columns={'index': 'Team'})
            
            # Select and order columns - Update column order to show Total Weighted Points right after Team and remove Position
            columns_order = [
                'Rank', 'Team', 'Total Weighted Points', 'Current Season Points', 'Previous Season Points', 
                'Two Seasons Ago Points', 'Overall Win %'
            ]
            final_rankings = final_rankings[columns_order]

            # Display rankings table without index
            st.dataframe(final_rankings, use_container_width=True, hide_index=True)

            # Update rankings visualization
            fig_rankings = px.bar(
                final_rankings,
                x='Team',
                y='Total Weighted Points',
                color='Overall Win %',
                text=final_rankings['Rank'].astype(str) + '. ' + final_rankings['Total Weighted Points'].round(1).astype(str)
            )

            fig_rankings.update_layout(
                height=500,
                xaxis_title="Team",
                yaxis_title="Ranking Points",
                showlegend=True,
                title={
                    'text': 'Current Rankings (Weighted by Season Recency)',
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'color': '#f04f53'}
                }
            )

            st.plotly_chart(fig_rankings, use_container_width=True, key="current_rankings_viz")

            # Add explanation of points system
            with st.expander("How are ranking points calculated?"):
                st.markdown(f"""
                ### Points Weighting by Season
                - Current Season ({years[0]}): 100% of points
                - Previous Season ({years[1] if len(years) > 1 else 'N/A'}): 50% of points
                - Two Seasons Ago ({years[2] if len(years) > 2 else 'N/A'}): 33.3% of points
                
                ### Position Points
                - Winner/D1 Winner: 50 points
                - D2 Winner: 45 points
                - Runner Up/D1 Runner Up: 40 points
                - D2 Runner Up: 35 points
                - Semi Final/Qualifier: 30 points
                - Quarter Final/Eliminator: 20 points
                - Group Stage: 10 points
                - Each position below 1st: -2 points
                
                ### Win Percentage Bonus Points
                - 100%: 50 bonus points
                - 90%+: 40 bonus points
                - 80%+: 30 bonus points
                - 70%+: 25 bonus points
                - 60%+: 20 bonus points
                - 50%+: 15 bonus points
                - 40%+: 10 bonus points
                - 30%+: 7 bonus points
                - 20%+: 5 bonus points
                - 10%+: 3 bonus points
                """)

            # After the existing rankings visualization, add new insights
            st.markdown("<h4 style='color:#f04f53; text-align: center;'>Ranking Insights</h4>", unsafe_allow_html=True)
            
            # Create three columns for different visualizations
            insight_col1, insight_col2 = st.columns(2)
            
            with insight_col1:
                # Points Breakdown Chart
                points_breakdown = pd.DataFrame({
                    'Team': final_rankings['Team'],
                    'Current Season': final_rankings['Current Season Points'],
                    'Previous Season': final_rankings['Previous Season Points'],
                    'Two Seasons Ago': final_rankings['Two Seasons Ago Points']
                })
                
                fig_breakdown = px.bar(
                    points_breakdown.melt(id_vars=['Team'], var_name='Season', value_name='Points'),
                    x='Team',
                    y='Points',
                    color='Season',
                    barmode='stack'
                )
                fig_breakdown.update_layout(
                    height=400,
                    title={
                        'text': 'Points Breakdown by Season',
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': {'color': '#f04f53'}
                    }
                )
                st.plotly_chart(fig_breakdown, use_container_width=True, key="points_breakdown_viz")

            with insight_col2:
                # Ranking Movement (if more than one year available)
                if len(years) >= 2:
                    current_positions = rankings_df[rankings_df['Year'] == years[0]].groupby('Team')['Position'].mean()
                    previous_positions = rankings_df[rankings_df['Year'] == years[1]].groupby('Team')['Position'].mean()
                    
                    movement_df = pd.DataFrame({
                        'Current': current_positions,
                        'Previous': previous_positions
                    }).fillna(0)
                    
                    movement_df['Change'] = movement_df['Previous'] - movement_df['Current']
                    movement_df = movement_df.sort_values('Change', ascending=False)
                    
                    fig_movement = px.bar(
                        movement_df.reset_index(),
                        x='Team',
                        y='Change',
                        color='Change',
                        color_continuous_scale=['red', 'lightgray', 'green'],
                        text=movement_df['Change'].round(1)
                    )
                    fig_movement.update_layout(
                        height=400,
                        title={
                            'text': 'Position Change from Previous Season',
                            'x': 0.5,
                            'xanchor': 'center',
                            'yanchor': 'top',
                            'font': {'color': '#f04f53'}
                        }
                    )
                    st.plotly_chart(fig_movement, use_container_width=True, key="position_change_viz")
                else:
                    # Provide a fallback to avoid 'NoneType' issues
                    movement_df = pd.DataFrame()
                    most_improved = "N/A"

            # Add a summary metrics section
            st.markdown("<h4 style='color:#f04f53; text-align: center;'>Key Statistics</h4>", unsafe_allow_html=True)
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                top_team = final_rankings.iloc[0]
                st.metric("Current #1", top_team['Team'], 
                         f"{top_team['Total Weighted Points']:.1f} pts")
            
            with metric_col2:
                if 'Change' in movement_df.columns:
                    most_improved = movement_df[movement_df['Change'] > 0].index[0] if len(movement_df[movement_df['Change'] > 0]) > 0 else "N/A"
                else:
                    most_improved = "N/A"
                max_improvement = movement_df['Change'].max() if len(movement_df) > 0 else 0
                st.metric("Most Improved", most_improved, 
                         f"↑ {max_improvement:.1f} positions" if max_improvement > 0 else "N/A")
            
            with metric_col3:
                highest_win_rate = final_rankings.loc[final_rankings['Overall Win %'].idxmax()]
                st.metric("Highest Win Rate", highest_win_rate['Team'],
                         f"{highest_win_rate['Overall Win %']:.1f}%")
            
            with metric_col4:
                avg_points = final_rankings['Total Weighted Points'].mean()
                st.metric("Average Points", f"{avg_points:.1f}",
                         f"Over {len(final_rankings)} teams")

            # Add existing expander with points calculation explanation
            # ...rest of existing code...

# ...rest of existing code...

            # Add historical rankings comparison
            st.markdown("<h4 style='color:#f04f53; text-align: center;'>Historical Rankings Comparison</h4>", unsafe_allow_html=True)
            
            # Add competition format filter above the historical rankings
            format_col1, format_col2 = st.columns([3, 1])
            
            with format_col1:
                format_filter = st.multiselect(
                    'Filter by Competition:',
                    options=filtered_df['Competition'].unique().tolist(),
                    default=filtered_df['Competition'].unique().tolist()[0] if len(filtered_df['Competition'].unique()) > 0 else None,
                    key="format_filter_rankings"
                )
            
            with format_col2:
                # Add a reset button
                if st.button("Reset Filters", key="reset_format_filter"):
                    format_filter = filtered_df['Competition'].unique().tolist()
            
            # Apply competition filter
            if format_filter:
                format_filtered_df = rankings_df[rankings_df['Competition'].isin(format_filter)]
            else:
                format_filtered_df = rankings_df.copy()
            
            # Create columns for the two charts
            hist_col1, hist_col2 = st.columns(2)
            
            # Prepare historical rankings data with competition filter applied
            historical_rankings = []
            for year in years:
                year_data = format_filtered_df[format_filtered_df['Year'] == year].copy()
                
                # Skip if no data for this year after filtering
                if year_data.empty:
                    continue
                    
                # Calculate total ranking points by team for the filtered competitions
                year_points = year_data.groupby('Team')['Total_Ranking_Points'].sum().reset_index()
                year_points['Year'] = year
                year_points = year_points.sort_values('Total_Ranking_Points', ascending=False)
                year_points['Rank'] = range(1, len(year_points) + 1)
                historical_rankings.append(year_points)
            
            if not historical_rankings:
                st.warning("No data available for the selected competitions.")
            else:
                historical_df = pd.concat(historical_rankings)
                
                with hist_col1:
                    # Create rank by year visualization
                    fig_rank_history = px.line(
                        historical_df,
                        x='Year',
                        y='Rank',
                        color='Team',
                        markers=True,
                        title='Team Rankings by Year'
                    )
                    
                    # Customize the rank chart
                    fig_rank_history.update_layout(
                        height=500,
                        yaxis_title="Rank",
                        yaxis_autorange='reversed',  # Higher rank (1) at the top
                        xaxis_title="Year",
                        showlegend=True,
                        title={
                            'x': 0.5,
                            'xanchor': 'center',
                            'font': {'color': '#f04f53'}
                        }
                    )
                    
                    # Set x-axis to show only full years
                    fig_rank_history.update_xaxes(
                        dtick=1,  # Show every year
                        type='category'  # Use categorical axis
                    )
                    
                    st.plotly_chart(fig_rank_history, use_container_width=True, key="rank_history_viz")
                
                with hist_col2:
                    # Create points by year visualization - Update title
                    fig_points_history = px.line(
                        historical_df,
                        x='Year',
                        y='Total_Ranking_Points',
                        color='Team',
                        markers=True,
                        title='Team Total Ranking Points by Year'
                    )
                    
                    # Customize the points chart
                    fig_points_history.update_layout(
                        height=500,
                        yaxis_title="Total Ranking Points",
                        xaxis_title="Year",
                        showlegend=True,
                        title={
                            'x': 0.5,
                            'xanchor': 'center',
                            'font': {'color': '#f04f53'}
                        }
                    )
                    
                    # Set x-axis to show only full years
                    fig_points_history.update_xaxes(
                        dtick=1,  # Show every year
                        type='category'  # Use categorical axis
                    )
                    
                    st.plotly_chart(fig_points_history, use_container_width=True, key="points_history_viz")
                
                # Update the Historical Statistics Summary with filtered data
                st.markdown("<h4 style='color:#f04f53; text-align: center;'>Historical Statistics Summary</h4>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>Showing data for: {', '.join(format_filter)}</p>", unsafe_allow_html=True)

                # Calculate historical statistics based on filtered data
                # Number 1 teams by year from filtered data
                top_teams_by_year = historical_df.loc[historical_df.groupby('Year')['Rank'].idxmin()][['Year', 'Team', 'Total_Ranking_Points']]
                
                # Most frequent #1 team
                most_frequent_top = top_teams_by_year['Team'].value_counts().index[0] if not top_teams_by_year.empty else "N/A"
                most_frequent_top_count = top_teams_by_year['Team'].value_counts().iloc[0] if not top_teams_by_year.empty else 0

                # Highest total points ever
                if not historical_df.empty:
                    max_idx = historical_df['Total_Ranking_Points'].idxmax()
                    highest_pts = historical_df.loc[max_idx, 'Total_Ranking_Points']
                    highest_pts_team = historical_df.loc[max_idx, 'Team']
                    highest_pts_year = historical_df.loc[max_idx, 'Year']
                    
                    # Ensure highest_pts is a simple float before formatting
                    try:
                        highest_pts = float(highest_pts)
                        highest_pts_string = f"{highest_pts:.1f} pts in {highest_pts_year}"
                    except (ValueError, TypeError):
                        highest_pts_string = f"{highest_pts} pts in {highest_pts_year}"
                else:
                    highest_pts_team = "N/A"
                    highest_pts_string = "N/A"

                # Most years as last place
                last_place_counts = historical_df[historical_df['Rank'] == historical_df.groupby('Year')['Rank'].transform('max')]['Team'].value_counts()
                most_last_place = last_place_counts.index[0] if not last_place_counts.empty else "N/A"
                most_last_place_count = last_place_counts.iloc[0] if not last_place_counts.empty else 0

                # Lowest total points ever
                if not historical_df.empty:
                    min_idx = historical_df['Total_Ranking_Points'].idxmin()
                    lowest_pts = historical_df.loc[min_idx, 'Total_Ranking_Points']
                    lowest_pts_team = historical_df.loc[min_idx, 'Team']
                    lowest_pts_year = historical_df.loc[min_idx, 'Year']
                    
                    # Ensure lowest_pts is a simple float before formatting
                    try:
                        lowest_pts = float(lowest_pts)
                        lowest_pts_string = f"{lowest_pts:.1f} pts in {lowest_pts_year}"
                    except (ValueError, TypeError):
                        lowest_pts_string = f"{lowest_pts} pts in {lowest_pts_year}"
                else:
                    lowest_pts_team = "N/A"
                    lowest_pts_string = "N/A"

                # Average rank by team
                avg_ranks = historical_df.groupby('Team')['Rank'].mean().sort_values()
                most_consistent = avg_ranks.index[0] if not avg_ranks.empty else "N/A"
                most_consistent_avg = avg_ranks.iloc[0] if not avg_ranks.empty else 0

                # Create a DataFrame for number 1 teams by year
                top_teams_df = top_teams_by_year.set_index('Year')

                # Display statistics in expandable sections
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 🏆 Number 1 Teams By Year")
                    st.dataframe(top_teams_df, use_container_width=True)

                with col2:
                    st.markdown("#### 📊 All-Time Records")
                    
                    # Create records DataFrame
                    records_df = pd.DataFrame({
                        'Record': [
                            'Most Frequent #1',
                            'Highest Points Ever',
                            'Most Last Place Finishes',
                            'Lowest Points Ever',
                            'Most Consistent Team'
                        ],
                        'Team': [
                            most_frequent_top,
                            highest_pts_team,
                            most_last_place,
                            lowest_pts_team,
                            most_consistent
                        ],
                        'Details': [
                            f"{most_frequent_top_count} times",
                            highest_pts_string,
                            f"{most_last_place_count} times",
                            lowest_pts_string,
                            f"Avg. Rank: {most_consistent_avg:.1f}" if isinstance(most_consistent_avg, (int, float)) else "N/A"
                        ]
                    })
                    
                    # Display the records table
                    st.dataframe(
                        records_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Record": st.column_config.TextColumn(
                                "Record Type",
                                width="medium"
                            ),
                            "Team": st.column_config.TextColumn(
                                "Team",
                                width="medium"
                            ),
                            "Details": st.column_config.TextColumn(
                                "Achievement Details",
                                width="medium"
                            )
                        }
                    )

                # Additional Statistics section - need to fix the Total_Points column reference
                st.markdown("#### 📈 Detailed Statistics")
                stat_col1, stat_col2, stat_col3 = st.columns(3)

                with stat_col1:
                    # Point Distribution Stats - Fixed to use Total_Ranking_Points instead of Total_Points
                    if not historical_df.empty:
                        points_stats = historical_df.groupby('Team')['Total_Ranking_Points'].agg(['mean', 'max', 'min']).round(1)
                        points_stats.columns = ['Avg Points', 'Max Points', 'Min Points']
                        points_stats = points_stats.sort_values('Avg Points', ascending=False)
                        st.markdown("##### Average Points by Team")
                        st.dataframe(points_stats, use_container_width=True, hide_index=True)
                    else:
                        st.write("No data available")

                with stat_col2:
                    # Rank Volatility (standard deviation of ranks)
                    rank_volatility = historical_df.groupby('Team')['Rank'].agg(['mean', 'std']).round(2)
                    rank_volatility.columns = ['Avg Rank', 'Rank Volatility']
                    rank_volatility = rank_volatility.sort_values('Avg Rank')
                    st.markdown("##### Rank Consistency")
                    st.dataframe(rank_volatility, use_container_width=True, hide_index=True)

                with stat_col3:
                    # Top 3 Finishes
                    top_3_mask = historical_df['Rank'] <= 3
                    top_3_finishes = historical_df[top_3_mask]['Team'].value_counts()
                    top_3_df = pd.DataFrame({
                        'Team': top_3_finishes.index,
                        'Top 3 Finishes': top_3_finishes.values
                    })
                    st.markdown("##### Top 3 Finish Counts")
                    st.dataframe(top_3_df, use_container_width=True, hide_index=True)

            # Add interesting insights
            st.markdown("#### 🎯 Key Insights")
            insights_col1, insights_col2 = st.columns(2)

            with insights_col1:
                # Year-over-year improvement
                year_improvements = []
                for team in historical_df['Team'].unique():
                    team_data = historical_df[historical_df['Team'] == team].sort_values('Year')
                    if len(team_data) >= 2:
                        rank_diff = team_data['Rank'].iloc[-1] - team_data['Rank'].iloc[-2]
                        year_improvements.append((team, -rank_diff))  # Negative because improvement means rank number decreases
                
                if year_improvements:
                    year_improvements.sort(key=lambda x: x[1], reverse=True)
                    st.markdown("##### Biggest Recent Improvers")
                    for team, improvement in year_improvements[:5]:
                        if improvement > 0:
                            st.markdown(f"- {team}: ⬆️ {improvement} positions")

            with insights_col2:
                # Longest streaks in top 3
                st.markdown("##### Notable Streaks")
                for team in historical_df['Team'].unique():
                    team_data = historical_df[historical_df['Team'] == team].sort_values('Year')
                    current_streak = 0
                    max_streak = 0
                    for rank in team_data['Rank']:
                        if rank <= 3:
                            current_streak += 1
                            max_streak = max(max_streak, current_streak)
                        else:
                            current_streak = 0
                    if max_streak >= 2:  # Only show streaks of 2 or more years
                        st.markdown(f"- {team}: {max_streak} consecutive years in top 3")

    # Add the new Heatmap tab functionality
    with tab5:
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Tournament Achievements Heatmap</h3>", unsafe_allow_html=True)
        
        # Create a copy of the filtered data
        heatmap_df = filtered_df.copy()
        
        # Create a simplified mapping function for colors based on Final Position
        def map_final_position_to_value(position):
            if pd.isna(position) or position == "":
                return 0  # No data
                
            position_str = str(position).strip()
            
            # Gold Category - Winners (excluding D2)
            if "Winner" in position_str and "D2 Winner" not in position_str:
                return 5  # Gold for main winners only
                
            # Silver Category - Runner-ups (excluding D2)
            if "Runner Up" in position_str and "D2 Runner Up" not in position_str:
                return 4  # Silver for main runner-ups only
                
            # Bronze Category - Semi Finals, D2 Winners, D2 Runner Ups, Quarter Finals
            if any(bronze in position_str for bronze in [
                "Semi Final", "D2 Winner", "D2 Runner Up", "Quarter Final", "Quater Final",
                "Qualifier", "Qualifier 1", "Qualifier 2", "Preliminary Final",
                "Eliminator", "Eliminator 1", "Eliminator 2"
            ]):
                return 3  # Bronze for semi-finals and similar stages
                
            # Light Blue Category - Group Stage
            if "Group Stage" in position_str:
                return 2  # Light Blue for group stage
                
            # Red Category - Relegated, Wooden Spoon
            if "Relegated" in position_str or "Wooden Spoon" in position_str:
                return 1  # Red for relegated or wooden spoon
                
            # Default to Light Blue for any other positions
            return 2
        
        # Apply the mapping to create a numerical column for the heatmap
        heatmap_df['Position_Value'] = heatmap_df['Final Position'].apply(map_final_position_to_value)
        
        # Create a custom color scale with proper spacing and light blue for group stage
        custom_color_scale = [
            [0.0, 'white'],          # Empty/no data
            [0.2, '#FF4C4C'],        # Red for Wooden Spoon/Relegated
            [0.4, '#ADD8E6'],        # Light blue for Group Stage
            [0.6, '#CD7F32'],        # Bronze for Quarter Finals, Semi Finals, etc.
            [0.8, '#C0C0C0'],        # Silver for Runner Up
            [1.0, '#FFD700']         # Gold for Winners
        ]
        
        # Competition selector - directly select from available competitions
        competitions = sorted(heatmap_df['Competition'].unique().tolist())
        
        if len(competitions) > 0:
            selected_competition = st.selectbox(
                "Select Competition for Heatmap", 
                options=competitions
            )
            
            # Filter data based on selection
            display_df = heatmap_df[heatmap_df['Competition'] == selected_competition].copy()
        else:
            display_df = heatmap_df.copy()
            selected_competition = competitions[0] if competitions else "None"
        
        # Group the data by Team and Year, taking the maximum value if there are multiple entries
        pivot_data = display_df.pivot_table(
            index='Team', 
            columns='Year', 
            values='Position_Value',
            aggfunc='max'
        ).fillna(0)
        
        # Create the heatmap
        if not pivot_data.empty:
            fig = px.imshow(
                pivot_data,
                labels=dict(x="Year", y="Team", color="Achievement"),
                aspect="auto",
                color_continuous_scale=custom_color_scale,
                zmin=0,  # Set minimum value to ensure proper color scaling
                zmax=5   # Set maximum value to ensure proper color scaling
            )
            
            # Update layout
            fig.update_layout(
                height=max(300, len(pivot_data) * 30),  # Dynamic height based on team count
                title={
                    'text': f"{selected_competition} Achievements",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'color': '#f04f53'}
                }
            )
            
            # Ensure x-axis shows only whole numbers
            fig.update_xaxes(
                dtick=1,  # Show every year
                tickmode='linear',  # Use linear tick mode
                type='category'  # Use categorical axis to ensure whole numbers
            )
            
            # Display the heatmap
            st.plotly_chart(fig, use_container_width=True, key="achievements_heatmap")
            
            # Add a consistent legend
            legend_html = """
            <div style="background-color:#f9f9f9; padding:10px; border-radius:5px; margin-top:10px;">
                <h5 style="color:#f04f53">Legend:</h5>
                <div style="display:flex; flex-wrap:wrap; gap:10px;">
                    <div style="display:flex; align-items:center; margin-right:15px;">
                        <div style="background-color:#FFD700; width:20px; height:20px; margin-right:5px;"></div>
                        <span>Winner/D1 Winner</span>
                    </div>
                    <div style="display:flex; align-items:center; margin-right:15px;">
                        <div style="background-color:#C0C0C0; width:20px; height:20px; margin-right:5px;"></div>
                        <span>Runner Up/D1 Runner Up</span>
                    </div>
                    <div style="display:flex; align-items:center; margin-right:15px;">
                        <div style="background-color:#CD7F32; width:20px; height:20px; margin-right:5px;"></div>
                        <span>Semi Final/Quarter Final/D2 Winner/D2 Runner Up</span>
                    </div>
                    <div style="display:flex; align-items:center; margin-right:15px;">
                        <div style="background-color:#ADD8E6; width:20px; height:20px; margin-right:5px;"></div>
                        <span>Group Stage</span>
                    </div>
                    <div style="display:flex; align-items:center;">
                        <div style="background-color:#FF4C4C; width:20px; height:20px; margin-right:5px;"></div>
                        <span>Wooden Spoon/Relegated</span>
                    </div>
                </div>
            </div>
            """
            st.markdown(legend_html, unsafe_allow_html=True)
        else:
            st.info("No data available to display in the heatmap.")
        
        # Add explanatory text
        st.markdown("""
        This heatmap visualizes team achievements over time. The color represents the highest achievement 
        in each year for each team:
        
        - **Gold**: Tournament winners (Winner, D1 Winner)
        - **Silver**: Runners-up (Runner Up, D1 Runner Up)
        - **Bronze**: Semi-finalists, Quarter-finalists, D2 Winners & Runner-ups
        - **Light Blue**: Group Stage participants
        - **Red**: Teams that finished last (Wooden Spoon) or were relegated
        """)
