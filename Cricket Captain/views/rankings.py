import streamlit as st
import pandas as pd
import os
from datetime import datetime
import plotly.graph_objects as go


# Define the path where the CSV will be saved
SAVE_PATH = r"C:\Users\rtayl\AppData\Roaming\Childish Things\Cricket Captain 2024\Saves\Scorecards\rankings.csv"

# Define teams and positions
TEAMS = [
    "Australia", "Bangladesh", "England", "India", "Ireland", "New Zealand",
    "Pakistan", "South Africa", "Sri Lanka", "West Indies", "Zimbabwe", "Afghanistan"
]
POSITIONS = list(range(1, 13))  # 1 to 12

def save_data(df, append=True):
    """Save the DataFrame to CSV"""
    try:
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        if append and os.path.exists(SAVE_PATH):
            existing_df = pd.read_csv(SAVE_PATH)
            df = pd.concat([existing_df, df], ignore_index=True)
        df.to_csv(SAVE_PATH, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")
        return False

def clear_rankings():
    """Clear the rankings CSV file"""
    try:
        if os.path.exists(SAVE_PATH):
            os.remove(SAVE_PATH)
            return True
    except Exception as e:
        st.error(f"Error clearing rankings: {str(e)}")
        return False

def load_existing_data():
    """Load existing rankings data if it exists"""
    try:
        if os.path.exists(SAVE_PATH):
            df = pd.read_csv(SAVE_PATH)
            if 'Year' not in df.columns:
                df['Year'] = datetime.now().year
            return df
        else:
            return pd.DataFrame(columns=['Position', 'Team', 'Rating', 'Year', 'Last Updated'])
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(columns=['Position', 'Team', 'Rating', 'Year', 'Last Updated'])

# Title
st.markdown("<h1 style='color:#f04f53; text-align: center;'>Rankings</h1>", unsafe_allow_html=True)

# Year input at top
st.markdown("### Insert Year")
selected_year = st.number_input(
    "Year",
    min_value=1900,
    max_value=2100,
    value=datetime.now().year,
    step=1,
    key="global_year"
)

# Initialize session state
if 'current_rankings' not in st.session_state:
    st.session_state.current_rankings = {}
    existing_data = load_existing_data()
    if not existing_data.empty:
        for _, row in existing_data.iterrows():
            st.session_state.current_rankings[row['Position']] = {
                'team': row['Team'],
                'rating': row['Rating'],
                'year': selected_year  # Use the global year
            }
    else:
        for pos in POSITIONS:
            st.session_state.current_rankings[pos] = {
                'team': '',
                'rating': 0,
                'year': selected_year  # Use the global year
            }

# Column headers
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.markdown("**Position**")
with col2:
    st.markdown("**Team**")
with col3:
    st.markdown("**Rating**")

# Track used teams
used_teams = [st.session_state.current_rankings[pos]['team'] for pos in POSITIONS 
              if st.session_state.current_rankings[pos]['team'] and 
              st.session_state.current_rankings[pos]['team'] != '']

# Create rows for each position
for pos in POSITIONS:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.write(f"{pos}")
    
    with col2:
        current_team = st.session_state.current_rankings[pos]['team']
        available_teams = [team for team in TEAMS if team not in used_teams or team == current_team]
        available_teams.sort()
        display_options = [''] + available_teams
        
        current_index = 0
        if current_team and current_team in available_teams:
            current_index = available_teams.index(current_team) + 1
            
        selected_team = st.selectbox(
            f"Select team for position {pos}",
            options=display_options,
            index=current_index,
            key=f"team_{pos}",
            label_visibility='collapsed'
        )
        
        if selected_team != st.session_state.current_rankings[pos]['team']:
            st.session_state.current_rankings[pos]['team'] = selected_team
            used_teams = [st.session_state.current_rankings[p]['team'] for p in POSITIONS 
                         if st.session_state.current_rankings[p]['team'] and 
                         st.session_state.current_rankings[p]['team'] != '']
    
    with col3:
        current_rating = st.session_state.current_rankings[pos]['rating']
        rating = st.number_input(
            f"Rating for position {pos}",
            min_value=0,
            max_value=1000,
            value=int(current_rating),
            step=1,
            key=f"rating_{pos}",
            label_visibility='collapsed'
        )
        st.session_state.current_rankings[pos]['rating'] = rating
        st.session_state.current_rankings[pos]['year'] = selected_year

def reset_form():
    """Reset the form to default values"""
    for pos in POSITIONS:
        st.session_state.current_rankings[pos] = {
            'team': '',
            'rating': 0,
            'year': st.session_state.global_year
        }

# Create a row for the buttons - modify to have three equal columns
col1, col2, col3 = st.columns(3)

# Save button
with col1:
    if st.button("Save Rankings"):
        # Check for existing years first
        if not any(st.session_state.current_rankings[pos]['team'] for pos in POSITIONS):
            st.warning("No rankings to save. Please select teams and add ratings.")
        else:
            # Check for year duplicates
            existing_data = load_existing_data()
            if not existing_data.empty and selected_year in existing_data['Year'].unique():
                st.warning(f"Data already exists for year {selected_year}. Please choose a different year or clear historical rankings.")
            else:
                # Create the rankings data
                rankings_data = {
                    'Position': [],
                    'Team': [],
                    'Rating': [],
                    'Year': [],
                    'Last Updated': []
                }
                
                for pos in POSITIONS:
                    if st.session_state.current_rankings[pos]['team'] and st.session_state.current_rankings[pos]['team'] != '':
                        rankings_data['Position'].append(pos)
                        rankings_data['Team'].append(st.session_state.current_rankings[pos]['team'])
                        rankings_data['Rating'].append(int(st.session_state.current_rankings[pos]['rating']))
                        rankings_data['Year'].append(selected_year)
                        rankings_data['Last Updated'].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                
                rankings_df = pd.DataFrame(rankings_data)
                save_data(rankings_df, append=True)
                st.success("Rankings saved successfully!")

# Reset Form button
with col2:
    if st.button("Enter Another Year"):
        reset_form()
        st.rerun()

# Clear Historical Rankings button
with col3:
    if st.button("Clear Historical Rankings"):
        if clear_rankings():
            st.success("Historical rankings cleared successfully!")
            st.rerun()


# Load the existing data without modifying it directly
existing_data = load_existing_data()
if not existing_data.empty:
    st.markdown("<h3 style='color:#f04f53; text-align: center;'>Current Rankings</h3>", unsafe_allow_html=True)
    
    # Create a new DataFrame to manipulate, keeping existing_data unaltered
    current_rating_df = (existing_data
                         .sort_values(by=['Year', 'Position'], ascending=[False, True])  # Sort by Year descending, Position ascending
                         .drop(columns=['Last Updated'])
                         .head(12))  # Display only the top 12 rows

    # Display the manipulated DataFrame with adjusted height
    st.dataframe(current_rating_df, use_container_width=True, hide_index=True, height=455)

ranking_per_year = existing_data

# Define team colors at the top of your file after TEAMS list
TEAM_COLORS = {
    "Australia": "#fdcd3c",       # Yellow
    "Bangladesh": "#006a4e",      # Dark Green
    "England": "#1e22aa",         # Royal Blue
    "India": "#0033FF",          # Blue
    "Ireland": "#169b62",        # Green
    "New Zealand": "#000000",    # Black
    "Pakistan": "#00894c",       # Pakistan Green
    "South Africa": "#007b2e",   # Green
    "Sri Lanka": "#003478",      # Deep Blue
    "West Indies": "#7b0041",    # Maroon
    "Zimbabwe": "#d40000",       # Red
    "Afghanistan": "#0066FF"     # Light Blue
}

# After your existing rankings display code:
ranking_per_year = existing_data

# Create line graph
st.markdown("<h3 style='color:#f04f53; text-align: center;'>Rankings History</h3>", unsafe_allow_html=True)
fig = go.Figure()

# Add traces for each team
for team in TEAMS:
    team_data = ranking_per_year[ranking_per_year['Team'] == team]
    if not team_data.empty:
        fig.add_trace(go.Scatter(
            x=team_data['Year'],
            y=team_data['Position'],
            name=team,
            line=dict(color=TEAM_COLORS[team], width=2),
            mode='lines+markers',
            marker=dict(size=8),
            opacity=1,
            hovertemplate=f"{team}<br>Year: %{{x}}<br>Position: %{{y}}<extra></extra>"
        ))

# Update layout
fig.update_layout(
    height=600,
    xaxis_title="Year",
    yaxis_title="Position",
    yaxis=dict(
        autorange="reversed",
        tickmode='linear',
        tick0=1,
        dtick=1,
        range=[12.5, 0.5]
    ),
    xaxis=dict(
        tickmode='linear',
        dtick=1
    ),
    hovermode='x unified',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=1.02,
        itemsizing='constant',
        itemwidth=30,
        bgcolor='rgba(0, 0, 0, 0)'
    ),
    showlegend=True,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(r=150)
)

# Add gridlines
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

# Update trace defaults
for trace in fig.data:
    trace.update(
        selected=dict(
            marker=dict(size=12, opacity=1),
        ),
        unselected=dict(
            marker=dict(size=8, opacity=0.3),
        )
    )

# Display the plot
st.plotly_chart(fig, use_container_width=True)

###############RATING PER YEAR

# Display the first plot with a unique key
#st.plotly_chart(fig, use_container_width=True, key="rankings_history")

# Add a new line graph with Year as the x-axis and Rating as the y-axis, with a different key
st.markdown("<h3 style='color:#f04f53; text-align: center;'>Ratings History</h3>", unsafe_allow_html=True)
fig_rating = go.Figure()

# Add traces for each team
for team in TEAMS:
    team_data = ranking_per_year[ranking_per_year['Team'] == team]
    if not team_data.empty:
        fig_rating.add_trace(go.Scatter(
            x=team_data['Year'],             # Year as the x-axis
            y=team_data['Rating'],           # Rating as the y-axis
            name=team,
            line=dict(color=TEAM_COLORS[team], width=2),
            mode='lines+markers',
            marker=dict(size=8),
            opacity=1,
            hovertemplate=f"{team}<br>Year: %{{x}}<br>Rating: %{{y}}<extra></extra>"
        ))

# Update layout for the new chart with gridlines and labels every 10 units on the y-axis
fig_rating.update_layout(
    height=600,
    xaxis_title="Year",
    yaxis_title="Rating",
    yaxis=dict(
        tickmode='linear',
        dtick=10,                       # Sets labels every 10 units on the y-axis
        range=[0, max(ranking_per_year['Rating'].max(), 100)],  # Adjust max range as needed
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGrey'
    ),
    xaxis=dict(
        tickmode='linear',
        dtick=1
    ),
    hovermode='x unified',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=1.02,
        itemsizing='constant',
        itemwidth=30,
        bgcolor='rgba(0, 0, 0, 0)'
    ),
    showlegend=True,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(r=150)
)

# Display the second plot with a unique key
st.plotly_chart(fig_rating, use_container_width=True, key="ratings_history")

