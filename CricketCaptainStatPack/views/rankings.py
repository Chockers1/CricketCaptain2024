import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import io

# Add file upload option
st.markdown("### Upload Previous Rankings (Optional) this is the rankings.csv from the Scorecard folder")
uploaded_file = st.file_uploader("Upload your rankings CSV file", type=['csv'], key="rankings_file_uploader")

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
    key="year_input"  # Changed from global_year to year_input
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
                'year': selected_year
            }
    else:
        for pos in POSITIONS:
            st.session_state.current_rankings[pos] = {
                'team': '',
                'rating': 0,
                'year': selected_year
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
            'year': selected_year
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
                         .sort_values(by=['Year', 'Position'], ascending=[False, True])
                         .drop(columns=['Last Updated'])
                         .head(12))

    # Display the manipulated DataFrame with adjusted height
    st.dataframe(current_rating_df, use_container_width=True, hide_index=True, height=455)

ranking_per_year = existing_data

# Define team colors
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

# Year-over-Year Analysis
st.markdown(f"<h3 style='color:#f04f53; text-align: center;'>Year-over-Year Rating Changes</h3>", unsafe_allow_html=True)

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
        title=dict(
            text=f"Rating Changes from {prev_year} to {latest_year}",
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top'
        )
    )
    
    st.plotly_chart(fig_yoy, use_container_width=True)

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

###########################################

# Box Plot Analysis
st.markdown("<h3 style='color:#f04f53; text-align: center;'>Team Performance Distribution Analysis</h3>", unsafe_allow_html=True)

# Create two columns for the box plots
col1, col2 = st.columns(2)


# Ratings Box Plot
with col1:
    fig_box_ratings = go.Figure()
    
    for team in TEAMS:
        team_data = ranking_per_year[ranking_per_year['Team'] == team]
        
        fig_box_ratings.add_trace(go.Box(
            y=team_data['Rating'],
            name=team,
            marker_color=TEAM_COLORS[team],
            boxpoints=False,  # No points will be shown
            jitter=0.3,
            pointpos=-1.8,
            hovertemplate=(
                f"<b>{team}</b><br>" +
                "Min: %{customdata[0]:.1f}<br>" +
                "Q1: %{customdata[1]:.1f}<br>" +
                "Median: %{customdata[2]:.1f}<br>" +
                "Q3: %{customdata[3]:.1f}<br>" +
                "Max: %{customdata[4]:.1f}<br>" +
                "<extra></extra>"
            ),
            customdata=[[ 
                team_data['Rating'].min(),
                team_data['Rating'].quantile(0.25),
                team_data['Rating'].median(),
                team_data['Rating'].quantile(0.75),
                team_data['Rating'].max()
            ]]
        ))
    
    fig_box_ratings.update_layout(
        title={
            'text': "Rating Distribution by Team",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'color': '#f04f53'}
        },
        height=600,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_title="Rating",
        xaxis_title="Team",
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGrey'
        )
    )

    st.plotly_chart(fig_box_ratings, use_container_width=True)

# Rankings Box Plot
with col2:
    fig_box_rankings = go.Figure()
    
    for team in TEAMS:
        team_data = ranking_per_year[ranking_per_year['Team'] == team]
        
        fig_box_rankings.add_trace(go.Box(
            y=team_data['Position'],
            name=team,
            marker_color=TEAM_COLORS[team],
            boxpoints=False,  # No points will be shown
            jitter=0.3,
            pointpos=-1.8,
            hovertemplate=(
                f"<b>{team}</b><br>" +
                "Best Rank: %{customdata[0]}<br>" +
                "Q1: %{customdata[1]:.1f}<br>" +
                "Median: %{customdata[2]:.1f}<br>" +
                "Q3: %{customdata[3]:.1f}<br>" +
                "Worst Rank: %{customdata[4]}<br>" +
                "<extra></extra>"
            ),
            customdata=[[ 
                int(team_data['Position'].min()),
                team_data['Position'].quantile(0.25),
                team_data['Position'].median(),
                team_data['Position'].quantile(0.75),
                int(team_data['Position'].max())
            ]]
        ))
    
    fig_box_rankings.update_layout(
        title={
            'text': "Ranking Distribution by Team",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'color': '#f04f53'}
        },
        height=600,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_title="Ranking",
        xaxis_title="Team",
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGrey',
            autorange="reversed"  # Reverse y-axis so rank 1 is at the top
        )
    )
    
    st.plotly_chart(fig_box_rankings, use_container_width=True)

# Add new analytics sections
st.markdown("<h3 style='color:#f04f53; text-align: center;'>Advanced Analytics</h3>", unsafe_allow_html=True)

# Create tabs for different analytics views
tab1, tab2, tab3 = st.tabs(["Head-to-Head Analysis", "Dominance Periods", "Milestones & Records"])

# First, add this CSS to improve tab spacing
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px !important;
        width: 100%;
    }
    
    .stTabs [data-baseweb="tab"] {
        width: calc(100%/3) !important;
        margin: 0px !important;
        white-space: normal !important;
    }
    </style>
""", unsafe_allow_html=True)

with tab1:
    st.markdown("<h4 style='color:#f04f53; text-align: center;'>Head-to-Head Rating Comparison</h4>", unsafe_allow_html=True)
    
    # Create two columns for team selection
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Select First Team", TEAMS, key="team1_select")
    with col2:
        team2 = st.selectbox("Select Second Team", [t for t in TEAMS if t != team1], key="team2_select")
    
    # Filter data for selected teams
    team1_data = ranking_per_year[ranking_per_year['Team'] == team1]
    team2_data = ranking_per_year[ranking_per_year['Team'] == team2]
    
    # Create head-to-head comparison plot
    fig_h2h = go.Figure()
    
    # Add traces for both teams
    fig_h2h.add_trace(go.Scatter(
        x=team1_data['Year'],
        y=team1_data['Rating'],
        name=team1,
        line=dict(color=TEAM_COLORS[team1], width=2),
        fill='tonexty'
    ))
    
    fig_h2h.add_trace(go.Scatter(
        x=team2_data['Year'],
        y=team2_data['Rating'],
        name=team2,
        line=dict(color=TEAM_COLORS[team2], width=2),
        fill='tonexty'
    ))
    
    fig_h2h.update_layout(
        height=400,
        title={
            'text': f"Rating Comparison: {team1} vs {team2}",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'color': '#f04f53'}
        }
    )
    st.plotly_chart(fig_h2h, use_container_width=True)
    
    # Add head-to-head statistics
    st.markdown("#### Key Statistics")
    col1, col2, col3 = st.columns(3)
    
    # Create a DataFrame with aligned years
    comparison_df = pd.merge(
        team1_data[['Year', 'Rating']], 
        team2_data[['Year', 'Rating']], 
        on='Year', 
        suffixes=('_1', '_2')
    )
    
    with col1:
        team1_wins = len(comparison_df[comparison_df['Rating_1'] > comparison_df['Rating_2']])
        team2_wins = len(comparison_df[comparison_df['Rating_2'] > comparison_df['Rating_1']])
        st.metric(
            "Higher Rating Count",
            f"{team1}: {team1_wins}",
            f"{team2}: {team2_wins}"
        )
    
    with col2:
        avg_diff = (comparison_df['Rating_1'] - comparison_df['Rating_2']).mean()
        st.metric(
            "Average Rating Difference",
            f"{avg_diff:.1f}"
        )
    
    with col3:
        if not comparison_df.empty:
            current_gap = comparison_df.iloc[-1]['Rating_1'] - comparison_df.iloc[-1]['Rating_2']
            st.metric(
                "Current Rating Gap",
                f"{current_gap:.1f}"
            )
        else:
            st.metric(
                "Current Rating Gap",
                "N/A"
            )

with tab2:
    st.markdown("<h4 style='color:#f04f53; text-align: center;'>Team Dominance Periods</h4>", unsafe_allow_html=True)
    
    # Calculate dominance periods (when teams were ranked #1)
    top_ranked_periods = []
    for year in sorted(ranking_per_year['Year'].unique()):
        year_data = ranking_per_year[ranking_per_year['Year'] == year]
        top_team = year_data[year_data['Position'] == 1].iloc[0]
        top_ranked_periods.append({
            'Year': year,
            'Team': top_team['Team'],
            'Rating': top_team['Rating']
        })
    
    dominance_df = pd.DataFrame(top_ranked_periods)
    
    # Create dominance visualization
    fig_dom = go.Figure()
    
    for team in TEAMS:
        team_periods = dominance_df[dominance_df['Team'] == team]
        if not team_periods.empty:
            fig_dom.add_trace(go.Scatter(
                x=team_periods['Year'],
                y=[team] * len(team_periods),
                mode='markers',
                name=team,
                marker=dict(
                    size=20,
                    color=TEAM_COLORS[team],
                    symbol='square'
                ),
                hovertemplate=(
                    f"<b>{team}</b><br>" +
                    "Year: %{x}<br>" +
                    "Rating: %{customdata:.1f}<br>" +
                    "<extra></extra>"
                ),
                customdata=team_periods['Rating']
            ))
    
    fig_dom.update_layout(
        height=400,
        title={
            'text': "Periods of #1 Ranking",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'color': '#f04f53'}
        }
    )
    st.plotly_chart(fig_dom, use_container_width=True)
    
    # Add dominance statistics
    st.markdown("#### Dominance Statistics")
    dom_stats = dominance_df['Team'].value_counts().reset_index()
    dom_stats.columns = ['Team', 'Years at #1']
    st.dataframe(dom_stats, use_container_width=True, hide_index=True)

with tab3:
    st.markdown("<h4 style='color:#f04f53; text-align: center;'>Notable Milestones & Records</h4>", unsafe_allow_html=True)
    
    # Calculate various records
    records = []
    
    # Highest ever rating
    highest_rating = ranking_per_year.nlargest(1, 'Rating').iloc[0]
    records.append({
        'Record': 'Highest Rating Ever',
        'Team': highest_rating['Team'],
        'Value': f"{highest_rating['Rating']:.1f}",
        'Year': highest_rating['Year']
    })
    
    # Highest average rating
    avg_ratings = ranking_per_year.groupby('Team')['Rating'].mean().reset_index()
    highest_avg = avg_ratings.nlargest(1, 'Rating').iloc[0]
    records.append({
        'Record': 'Highest Average Rating',
        'Team': highest_avg['Team'],
        'Value': f"{highest_avg['Rating']:.1f}",
        'Year': 'All time'
    })
    
    # Biggest yearly improvement
    ranking_per_year['Rating_Change'] = ranking_per_year.groupby('Team')['Rating'].diff()
    biggest_improvement = ranking_per_year.nlargest(1, 'Rating_Change').iloc[0]
    records.append({
        'Record': 'Biggest Year-on-Year Gain',
        'Team': biggest_improvement['Team'],
        'Value': f"+{biggest_improvement['Rating_Change']:.1f}",
        'Year': biggest_improvement['Year']
    })
    
    # Biggest yearly decline
    biggest_decline = ranking_per_year.nsmallest(1, 'Rating_Change').iloc[0]
    records.append({
        'Record': 'Biggest Year-on-Year Drop',
        'Team': biggest_decline['Team'],
        'Value': f"{biggest_decline['Rating_Change']:.1f}",
        'Year': biggest_decline['Year']
    })
    
    # Most years at #1
    years_at_1 = ranking_per_year[ranking_per_year['Position'] == 1]['Team'].value_counts().reset_index()
    top_team = years_at_1.iloc[0]
    records.append({
        'Record': 'Most Years at #1',
        'Team': top_team['Team'],
        'Value': f"{top_team['count']} years",
        'Year': 'All time'
    })
    
    # Most years at #2
    years_at_2 = ranking_per_year[ranking_per_year['Position'] == 2]['Team'].value_counts().reset_index()
    second_most = years_at_2.iloc[0]
    records.append({
        'Record': 'Most Years at #2',
        'Team': second_most['Team'],
        'Value': f"{second_most['count']} years",
        'Year': 'All time'
    })
    
    # Most wooden spoons (#12)
    wooden_spoons = ranking_per_year[ranking_per_year['Position'] == 12]['Team'].value_counts().reset_index()
    most_spoons = wooden_spoons.iloc[0]
    records.append({
        'Record': 'Most Wooden Spoons (#12)',
        'Team': most_spoons['Team'],
        'Value': f"{most_spoons['count']} years",
        'Year': 'All time'
    })
    
    # Longest streak at #1 (fixed calculation)
    consecutive_years = ranking_per_year.sort_values(['Year'])
    streaks = []
    for team in TEAMS:
        team_data = consecutive_years[consecutive_years['Team'] == team]
        if team_data.empty:
            continue
            
        current_streak = 0
        max_streak = 0
        streak_start = None
        max_streak_start = None
        
        for year, position in zip(team_data['Year'], team_data['Position']):
            if position == 1:
                if current_streak == 0:
                    streak_start = year
                current_streak += 1
                if current_streak > max_streak:
                    max_streak = current_streak
                    max_streak_start = streak_start
            else:
                current_streak = 0
                
        if max_streak > 0:
            streaks.append({
                'team': team,
                'streak': max_streak,
                'start': max_streak_start,
                'end': max_streak_start + max_streak - 1
            })
    
    # Find the longest streak
    longest_streak = max(streaks, key=lambda x: x['streak'])
    records.append({
        'Record': 'Longest Consecutive #1 Streak',
        'Team': longest_streak['team'],
        'Value': f"{longest_streak['streak']} years",
        'Year': f"{longest_streak['start']}-{longest_streak['end']}"
    })

    # Biggest rating gap between 1st and 2nd
    yearly_gaps = ranking_per_year.groupby('Year').apply(
        lambda x: x[x['Position'] == 1]['Rating'].iloc[0] - x[x['Position'] == 2]['Rating'].iloc[0]
    ).reset_index()
    biggest_gap_year = yearly_gaps.nlargest(1, 0).iloc[0]
    gap_data = ranking_per_year[ranking_per_year['Year'] == biggest_gap_year['Year']]
    first_team = gap_data[gap_data['Position'] == 1].iloc[0]
    
    records.append({
        'Record': 'Biggest #1 vs #2 Gap',
        'Team': first_team['Team'],
        'Value': f"{biggest_gap_year[0]:.1f} points",
        'Year': biggest_gap_year['Year']
    })
    
    # Highest position improvement in one year
    ranking_per_year['Position_Change'] = ranking_per_year.groupby('Team')['Position'].diff() * -1  # Multiply by -1 so positive is improvement
    best_climb = ranking_per_year.nlargest(1, 'Position_Change').iloc[0]
    records.append({
        'Record': 'Biggest Position Improvement',
        'Team': best_climb['Team'],
        'Value': f"+{int(best_climb['Position_Change'])} places",
        'Year': best_climb['Year']
    })
    
    # Display records in a styled dataframe
    records_df = pd.DataFrame(records)
    st.dataframe(records_df, use_container_width=True, hide_index=True)
    
    # Add a summary visualization of the records
    col1, col2 = st.columns(2)
    
    # Fix the position counts visualization
    with col1:
        # Create a bar chart of years at different positions
        position_counts = pd.DataFrame()
        for pos in [1, 2, 12]:  # Top 2 and wooden spoon
            counts = ranking_per_year[ranking_per_year['Position'] == pos]['Team'].value_counts()
            position_counts[f'#{pos}'] = counts
        
        # Reset index and rename the team column
        position_counts = position_counts.reset_index()
        position_counts = position_counts.rename(columns={'index': 'Team'})
        
        fig_positions = px.bar(
            position_counts,
            x='Team',
            y=[f'#{pos}' for pos in [1, 2, 12]],
            title='Years at Key Positions',
            labels={'value': 'Years', 'variable': 'Position'},
            barmode='group'
        )
        
        fig_positions.update_layout(
            height=400,
            title={
                'text': 'Years at Key Positions',
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'color': '#f04f53'}
            }
        )
        st.plotly_chart(fig_positions, use_container_width=True)
    
    with col2:
        # Create a scatter plot of highest ratings vs average ratings with corrected calculations
        team_stats = pd.DataFrame({
            'Team': TEAMS,  # Use TEAMS list to ensure all teams are included
            'Highest Rating': [ranking_per_year[ranking_per_year['Team'] == team]['Rating'].max() for team in TEAMS],
            'Average Rating': [ranking_per_year[ranking_per_year['Team'] == team]['Rating'].mean() for team in TEAMS]
        }).fillna(0)  # Fill any NaN values with 0
        
        fig_ratings = px.scatter(
            team_stats,
            x='Average Rating',
            y='Highest Rating',
            text='Team',
            color='Team',
            color_discrete_map=TEAM_COLORS,
            labels={
                'Average Rating': 'Career Average Rating',
                'Highest Rating': 'Peak Rating'
            }
        )
        
        fig_ratings.update_layout(
            height=400,
            title={
                'text': 'Peak vs Average Performance',
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'color': '#f04f53'}
            },
            xaxis=dict(
                title='Career Average Rating',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='LightGrey',
                range=[0, max(team_stats['Average Rating']) * 1.1]
            ),
            yaxis=dict(
                title='Peak Rating',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='LightGrey',
                range=[0, max(team_stats['Highest Rating']) * 1.1]
            )
        )
        
        # Add hover template with more details
        fig_ratings.update_traces(
            hovertemplate="<b>%{text}</b><br>" +
                         "Peak Rating: %{y:.1f}<br>" +
                         "Average Rating: %{x:.1f}<br>" +
                         "<extra></extra>"
        )
        
        st.plotly_chart(fig_ratings, use_container_width=True)

# ...rest of existing code...
