import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import io

# Add file upload option
st.markdown("### Upload Previous Rankings (Optional) this is the rankings.csv from the Scorecard folder")
uploaded_file = st.file_uploader("Upload your rankings CSV file", type=['csv'])

# Load data from uploaded file if it exists
if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        st.session_state.rankings_data = uploaded_df
        st.success("Rankings loaded successfully!")
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

# Initialize rankings_data in session state if it doesn't exist
if 'rankings_data' not in st.session_state:
    st.session_state.rankings_data = pd.DataFrame(columns=['Position', 'Team', 'Rating', 'Year', 'Last Updated'])

# Remove local save path and define teams and positions
TEAMS = [
    "Australia", "Bangladesh", "England", "India", "Ireland", "New Zealand",
    "Pakistan", "South Africa", "Sri Lanka", "West Indies", "Zimbabwe", "Afghanistan"
]
POSITIONS = list(range(1, 13))  # 1 to 12

def save_data(df, append=True):
    """Convert DataFrame to CSV and provide download link"""
    try:
        # Convert DataFrame to CSV
        csv = df.to_csv(index=False)
        
        # Create download button
        st.download_button(
            label="Download Rankings CSV",
            data=csv,
            file_name=f"rankings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )
        return True
    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        return False

def clear_rankings():
    """Clear the rankings from session state"""
    try:
        if 'rankings_data' in st.session_state:
            st.session_state.rankings_data = pd.DataFrame(columns=['Position', 'Team', 'Rating', 'Year', 'Last Updated'])
            return True
    except Exception as e:
        st.error(f"Error clearing rankings: {str(e)}")
        return False

def load_existing_data():
    """Load existing rankings data from session state"""
    try:
        if 'rankings_data' not in st.session_state:
            st.session_state.rankings_data = pd.DataFrame(columns=['Position', 'Team', 'Rating', 'Year', 'Last Updated'])
        return st.session_state.rankings_data
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

with col1:
    if st.button("Save Rankings"):
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
                
                # Update session state with new data
                if 'rankings_data' in st.session_state:
                    st.session_state.rankings_data = pd.concat([st.session_state.rankings_data, rankings_df], ignore_index=True)
                else:
                    st.session_state.rankings_data = rankings_df
                
                # Save to CSV for download
                save_data(st.session_state.rankings_data)
                st.success("Rankings saved successfully!")
                
                # Force refresh to show new data
                st.rerun()

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

#

#######################

# Add Rating Distribution and Gap Analysis
st.markdown("<h3 style='color:#f04f53; text-align: center;'>Rating Distribution and Gap Analysis</h3>", unsafe_allow_html=True)

# Get the most recent year's data and sort by rating
latest_year = ranking_per_year['Year'].max()
latest_data = ranking_per_year[ranking_per_year['Year'] == latest_year].sort_values('Rating', ascending=False)

# Calculate rating gaps and percentage differences
latest_data['Rating_Gap'] = latest_data['Rating'].diff().abs()
latest_data['Percentage_Diff'] = (latest_data['Rating_Gap'] / latest_data['Rating'].shift()) * 100

# Create visualization with just the bars and annotations
fig_dist = go.Figure()

# Add bars for ratings
fig_dist.add_trace(go.Bar(
    x=latest_data['Team'],
    y=latest_data['Rating'],
    name='Rating',
    marker_color=[TEAM_COLORS[team] for team in latest_data['Team']],
    text=latest_data['Rating'].round(1),
    textposition='auto',
))

# Add annotations for gaps between bars
for idx in range(1, len(latest_data)):
    fig_dist.add_annotation(
        x=latest_data['Team'].iloc[idx],
        y=latest_data['Rating'].iloc[idx-1],
        text=f'↓ {latest_data["Rating_Gap"].iloc[idx]:.1f}<br>({latest_data["Percentage_Diff"].iloc[idx]:.1f}%)',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='rgba(255, 0, 0, 0.5)',
        ax=0,
        ay=30
    )

fig_dist.update_layout(
    height=500,
    xaxis_title="Team",
    yaxis_title="Rating",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=False,
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGrey',
        range=[0, max(latest_data['Rating']) * 1.2]
    ),
    bargap=0.2,
    margin=dict(t=50, b=50, l=50, r=50)
)

st.plotly_chart(fig_dist, use_container_width=True)

##########

# Year-over-Year Analysis
st.markdown(f"<h3 style='color:#f04f53; text-align: center;'>Year-over-Year Rating Changes</h3>", unsafe_allow_html=True)
#st.markdown(f"<p style='text-align: center; font-size: 1.1em;'>Comparing {latest_year} vs {latest_year-1}</p>", unsafe_allow_html=True)

# Get previous year's data
prev_year = latest_year - 1
prev_data = ranking_per_year[ranking_per_year['Year'] == prev_year]

if not prev_data.empty:
    # Merge current and previous year data
    yoy_data = latest_data.merge(
        prev_data[['Team', 'Rating']], 
        on='Team', 
        suffixes=('_current', '_prev')
    )
    
    yoy_data['Rating_Change'] = yoy_data['Rating_current'] - yoy_data['Rating_prev']
    
    # Create YoY change visualization
    fig_yoy = go.Figure()
    
    # Add bars with custom hover text
    fig_yoy.add_trace(go.Bar(
        x=yoy_data['Team'],
        y=yoy_data['Rating_Change'],
        marker_color=[TEAM_COLORS[team] for team in yoy_data['Team']],
        text=yoy_data['Rating_Change'].apply(lambda x: f"+{x:.1f}" if x > 0 else f"{x:.1f}"),
        textposition='auto',
        hovertemplate=(
            "<b>%{x}</b><br>" +
            f"{latest_year} Rating: %{{customdata[0]:.1f}}<br>" +
            f"{prev_year} Rating: %{{customdata[1]:.1f}}<br>" +
            "Change: %{text}<br>" +
            "<extra></extra>"
        ),
        customdata=yoy_data[['Rating_current', 'Rating_prev']].values
    ))
    
    fig_yoy.update_layout(
        height=400,
        xaxis_title="Team",
        yaxis_title="Rating Change",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGrey',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black'
        ),
        # Add a more detailed title with years
        title=dict(
            text=f"Rating Changes from {prev_year} to {latest_year}",
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top'
        )
    )
    
    st.plotly_chart(fig_yoy, use_container_width=True)
    
##########

# Best Ratings Ever Section
st.markdown("<h3 style='color:#f04f53; text-align: center;'>Best Ratings Ever</h3>", unsafe_allow_html=True)

# Get all ratings sorted by value
best_ratings = ranking_per_year.sort_values('Rating', ascending=False)

# Keep top 20 ratings overall
best_ratings_table = best_ratings[['Team', 'Rating', 'Position', 'Year']].head(20)
best_ratings_table.columns = ['Team', 'Rating', 'Rank', 'Year']

# Display the table
st.dataframe(best_ratings_table.round(1), use_container_width=True, hide_index=True)

# Create enhanced scatter plot visualization
st.markdown("<h3 style='color:#f04f53; text-align: center;'>All-Time Ratings & Rankings Distribution</h3>", unsafe_allow_html=True)

fig_scatter = go.Figure()

# Add scatter points for each team
for team in TEAMS:
    team_data = ranking_per_year[ranking_per_year['Team'] == team]
    
    fig_scatter.add_trace(go.Scatter(
        x=team_data['Year'],
        y=team_data['Rating'],
        name=team,
        mode='markers',
        marker=dict(
            color=TEAM_COLORS[team],
            size=10,
            symbol='circle',
        ),
        hovertemplate=(
            f"<b>{team}</b><br>" +
            "Year: %{x}<br>" +
            "Rating: %{y:.1f}<br>" +
            "Rank: %{customdata}<br>" +
            "<extra></extra>"
        ),
        customdata=team_data['Position']
    ))

    # Add connecting lines between points
    fig_scatter.add_trace(go.Scatter(
        x=team_data['Year'],
        y=team_data['Rating'],
        name=team + " (line)",
        mode='lines',
        line=dict(
            color=TEAM_COLORS[team],
            width=1,
            dash='dot'
        ),
        showlegend=False,
        hoverinfo='skip'
    ))

# Update layout
fig_scatter.update_layout(
    height=600,
    xaxis_title="Year",
    yaxis_title="Rating",
    showlegend=True,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGrey',
        zeroline=False
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGrey',
        zeroline=False
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    margin=dict(l=10, r=10, t=50, b=10)
)

# Add a horizontal line for the overall mean rating
mean_rating = ranking_per_year['Rating'].mean()
fig_scatter.add_hline(
    y=mean_rating, 
    line_dash="dash", 
    line_color="grey",
    annotation_text=f"Overall Mean Rating ({mean_rating:.1f})",
    annotation_position="bottom right"
)

# Display the plot
st.plotly_chart(fig_scatter, use_container_width=True)

# Add a second visualization for rankings distribution
st.markdown("<h3 style='color:#f04f53; text-align: center;'>Rankings Movement Over Time</h3>", unsafe_allow_html=True)

fig_rankings = go.Figure()

# Add traces for each team's ranking
for team in TEAMS:
    team_data = ranking_per_year[ranking_per_year['Team'] == team]
    
    fig_rankings.add_trace(go.Scatter(
        x=team_data['Year'],
        y=team_data['Position'],
        name=team,
        mode='lines+markers',
        line=dict(
            color=TEAM_COLORS[team],
            width=2
        ),
        marker=dict(
            size=8,
            symbol='circle'
        ),
        hovertemplate=(
            f"<b>{team}</b><br>" +
            "Year: %{x}<br>" +
            "Rank: %{y}<br>" +
            "Rating: %{customdata:.1f}<br>" +
            "<extra></extra>"
        ),
        customdata=team_data['Rating']
    ))

# Update layout for rankings
fig_rankings.update_layout(
    height=600,
    xaxis_title="Year",
    yaxis_title="Ranking",
    showlegend=True,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGrey',
        zeroline=False
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGrey',
        zeroline=False,
        autorange="reversed",  # Reverse y-axis so rank 1 is at the top
        dtick=1  # Show every rank
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    margin=dict(l=10, r=10, t=50, b=10)
)

# Display the rankings plot
st.plotly_chart(fig_rankings, use_container_width=True)
