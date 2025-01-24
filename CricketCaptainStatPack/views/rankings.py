import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

# Add CSS styling
st.markdown("""
    <style>
    /* Table styling */
    table { 
        color: black; 
        width: 100%; 
    }
    thead tr th {
        background-color: #f04f53 !important;
        color: white !important;
    }
    tbody tr:nth-child(even) { background-color: #f0f2f6; }
    tbody tr:nth-child(odd) { background-color: white; }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        width: 100%;
        gap: 0px;
    }
    .stTabs [data-baseweb="tab"] {
        flex-grow: 1;
        text-align: center;
        padding: 1rem 0.5rem;
        background-color: #f0f2f6;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e0e2e6;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #f04f53 !important;
        color: white !important;
    }
    
    /* DataFrame styling */
    .dataframe {
        width: 100% !important;
        max-width: 100% !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize all session state variables at the top
if 'rankings_data' not in st.session_state:
    st.session_state.rankings_data = pd.DataFrame(columns=['Position', 'Team', 'Rating', 'Year', 'Format', 'Last Updated'])

if 'update_df' not in st.session_state:
    st.session_state.update_df = pd.DataFrame(columns=['Year', 'Format', 'Position', 'Team', 'Rating'])

# Remove local save path and define teams and positions
TEAMS = [
    "Australia", "Bangladesh", "England", "India", "Ireland", "New Zealand",
    "Pakistan", "South Africa", "Sri Lanka", "West Indies", "Zimbabwe", "Afghanistan"
]
POSITIONS = list(range(1, 13))  # 1 to 12

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

def save_data(df):
    """Save DataFrame to CSV using Streamlit download button"""
    try:
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"rankings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )
        return True
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")
        return False

def save_session_updates(df):
    """Save session updates to CSV"""
    try:
        save_type = st.radio(
            "Select save option:",
            ["Create new file", "Append to existing file"],
            key="save_type_radio"
        )
        
        if save_type == "Create new file":
            # Use Streamlit download button for new file
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"session_updates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv',
                key="download_new_btn"
            )
        else:  # Append to existing file
            # Use Streamlit file uploader for selecting file to append to
            uploaded_file = st.file_uploader("Select CSV file to append to", type=['csv'], key="append_file_uploader")
            
            if uploaded_file is not None:
                try:
                    # Read existing CSV
                    existing_df = pd.read_csv(uploaded_file)
                    # Append new data
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    # Remove duplicates and sort
                    combined_df = combined_df.drop_duplicates(
                        subset=['Year', 'Format', 'Position', 'Team'], 
                        keep='last'
                    ).sort_values(by=['Year', 'Position'])
                    # Save back to the same file
                    csv = combined_df.to_csv(index=False)
                    st.download_button(
                        label="Download Updated CSV",
                        data=csv,
                        file_name=f"updated_{uploaded_file.name}",
                        mime='text/csv',
                        key="download_updated_btn"
                    )
                    st.success(f"Successfully appended data to {uploaded_file.name}")
                    return True
                except Exception as e:
                    st.error(f"Error appending to file: {str(e)}")
                    return False
        return True
    except Exception as e:
        st.error(f"Error saving session updates: {str(e)}")
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

# Create tabs using list approach
tab_names = ["📊 Update Rankings", "📋 Current Rankings"]
tabs = st.tabs(tab_names)

# Use tabs[0] for Update Rankings content
with tabs[0]:
    # Add file upload option
    st.markdown("### Step 1- Upload Previous Rankings or Insert Year, Format, Teams and Rating below")
    uploaded_file = st.file_uploader("Upload your rankings CSV file", type=['csv'], key="rankings_file_uploader")
    
    # Load data from uploaded file if it exists
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            # Ensure 'Format' column exists
            if 'Format' not in uploaded_df.columns:
                uploaded_df['Format'] = 'Test Match'
            # Update rankings_data
            st.session_state.rankings_data = uploaded_df
            
            # Create update_df data from uploaded file
            update_data = {
                'Year': [], 'Format': [], 'Position': [], 'Team': [], 'Rating': []
            }
            
            # Extract data from uploaded file
            for _, row in uploaded_df.iterrows():
                update_data['Year'].append(row['Year'])
                update_data['Format'].append(row['Format'])
                update_data['Position'].append(row['Position'])
                update_data['Team'].append(row['Team'])
                update_data['Rating'].append(row['Rating'])
            
            # Create DataFrame and update session state
            temp_df = pd.DataFrame(update_data)
            if 'update_df' not in st.session_state:
                st.session_state.update_df = temp_df
            else:
                st.session_state.update_df = pd.concat([st.session_state.update_df, temp_df], ignore_index=True)
                st.session_state.update_df = st.session_state.update_df.drop_duplicates(
                    subset=['Year', 'Format', 'Position', 'Team'], keep='last'
                ).sort_values(by=['Year', 'Position'])
                
            st.success("Rankings loaded successfully and added to Current Session Updates (duplicates removed)!")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

    # Add the year and format inputs
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Insert Year")
        selected_year = st.number_input(
            "Year", min_value=1900, max_value=2100, value=datetime.now().year, step=1, key="year_input"
        )
    
    with col2:
        st.markdown("### Format")
        selected_format = st.selectbox(
            "Select Format", ["Test Match", "ODI", "T20I"], index=0, key="format_select"
        )
    
    # Initialize session state
    if 'current_rankings' not in st.session_state:
        st.session_state.current_rankings = {}
        existing_data = load_existing_data()
        if not existing_data.empty:
            for _, row in existing_data.iterrows():
                st.session_state.current_rankings[row['Position']] = {
                    'team': row['Team'], 'rating': row['Rating'], 'year': selected_year
                }
        else:
            for pos in POSITIONS:
                st.session_state.current_rankings[pos] = {
                    'team': '', 'rating': 0, 'year': selected_year
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
                'team': '', 'rating': 0, 'year': selected_year
            }

    # Create a row for the buttons - modify to have three equal columns
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Step 2 - Click here to Save Rankings to Current State"):
            if not any(st.session_state.current_rankings[pos]['team'] for pos in POSITIONS):
                st.warning("No rankings to save. Please select teams and add ratings.")
            else:
                # Create the rankings data for update_df
                update_data = {
                    'Year': [], 'Format': [], 'Position': [], 'Team': [], 'Rating': []
                }
                
                for pos in POSITIONS:
                    if st.session_state.current_rankings[pos]['team'] and st.session_state.current_rankings[pos]['team'] != '':
                        update_data['Year'].append(selected_year)
                        update_data['Format'].append(selected_format)
                        update_data['Position'].append(pos)
                        update_data['Team'].append(st.session_state.current_rankings[pos]['team'])
                        update_data['Rating'].append(int(st.session_state.current_rankings[pos]['rating']))
                
                # Create temporary DataFrame and concatenate with existing update_df
                temp_df = pd.DataFrame(update_data)
                if 'update_df' not in st.session_state:
                    st.session_state.update_df = temp_df
                else:
                    st.session_state.update_df = pd.concat([st.session_state.update_df, temp_df], ignore_index=True)
                    st.session_state.update_df = st.session_state.update_df.drop_duplicates(
                        subset=['Year', 'Format', 'Position', 'Team'], keep='last'
                    ).sort_values(by=['Year', 'Position'])
                
                # Create the rankings data for file saving
                rankings_data = {
                    'Position': [], 'Team': [], 'Rating': [], 'Year': [], 'Format': [], 'Last Updated': []
                }
                
                for pos in POSITIONS:
                    if st.session_state.current_rankings[pos]['team'] and st.session_state.current_rankings[pos]['team'] != '':
                        rankings_data['Position'].append(pos)
                        rankings_data['Team'].append(st.session_state.current_rankings[pos]['team'])
                        rankings_data['Rating'].append(int(st.session_state.current_rankings[pos]['rating']))
                        rankings_data['Year'].append(selected_year)
                        rankings_data['Format'].append(selected_format)
                        rankings_data['Last Updated'].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                
                rankings_df = pd.DataFrame(rankings_data)
                
                # Update session state with new data
                if 'rankings_data' in st.session_state:
                    st.session_state.rankings_data = pd.concat([st.session_state.rankings_data, rankings_df], ignore_index=True)
                else:
                    st.session_state.rankings_data = rankings_df
                
                # Save the data using Streamlit download button
                save_data(st.session_state.rankings_data)
                
                # Clear the input tables
                for pos in POSITIONS:
                    st.session_state.current_rankings[pos] = {
                        'team': '', 'rating': 0, 'year': selected_year
                    }
                
                # Force refresh to show new data
                st.rerun()

    # Add Current Session Updates display before the Clear button
    st.markdown("### Current Session Updates")
    if not st.session_state.update_df.empty:
        st.dataframe(
            st.session_state.update_df.sort_values(by=['Year', 'Position']), 
            use_container_width=True,
            hide_index=True
        )
        
        # Add Save button
        col1, col2 = st.columns(2)
        with col1:
            # Create new file with download button
            csv = st.session_state.update_df.to_csv(index=False)
            current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.download_button(
            label="Step 3 - Download Latest Rankings with your updates to use next time",
            data=csv,
            file_name=f"rankings_{current_date}.csv",
            mime='text/csv',
            key="download_new_btn"
            )
        
        with col2:
            if st.button("Clear Session Updates", key="clear_session_btn"):
                st.session_state.update_df = pd.DataFrame(columns=['Year', 'Format', 'Position', 'Team', 'Rating'])
                st.rerun()
    else:
        st.info("No rankings have been saved in this session yet.")

# Use tabs[1] for Current Rankings content
with tabs[1]:
    st.markdown("### Current Rankings")
    
    if not st.session_state.rankings_data.empty:
        # Add format filter
        format_filter = st.selectbox(
            "Select Format",
            ["All"] + st.session_state.rankings_data['Format'].unique().tolist(),
            key="rankings_format_filter"
        )
        
        # Filter data based on format
        filtered_df = st.session_state.rankings_data
        if format_filter != "All":
            filtered_df = filtered_df[filtered_df['Format'] == format_filter]
        
        # Ensure year is displayed as a full year without commas
        filtered_df['Year'] = filtered_df['Year'].astype(str)
            
        # Check if 'Last Updated' column exists before attempting to drop it
        if 'Last Updated' in filtered_df.columns:
            filtered_df = filtered_df.drop('Last Updated', axis=1)
        
        # Display the rankings table with sorted by Year (desc) and Position (asc)
        display_df = filtered_df.sort_values(by=['Year', 'Format', 'Position'], ascending=[True, True, False])
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # Best Ratings Ever Section
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Best Ratings Ever</h3>", unsafe_allow_html=True)

        # Get all ratings sorted by value 
        best_ratings = filtered_df.sort_values('Rating', ascending=False)

        # Keep top 20 ratings overall
        best_ratings_table = best_ratings[['Team', 'Rating', 'Position', 'Year', 'Format']].head(20)

        # Ensure year is displayed as a full year without commas
        best_ratings_table['Year'] = best_ratings_table['Year'].astype(str)

        # Display the table
        st.dataframe(best_ratings_table.round(1), use_container_width=True, hide_index=True)

        # Define team colors
        TEAM_COLORS = {
            "Australia": "#FFD700",
            "Bangladesh": "#006A4E",
            "England": "#00247D",
            "India": "#FF9933",
            "Ireland": "#169B62",
            "New Zealand": "#000000",
            "Pakistan": "#01411C",
            "South Africa": "#007A4D",
            "Sri Lanka": "#000080",
            "West Indies": "#7B0000",
            "Zimbabwe": "#C60C30",
            "Afghanistan": "#002395"
        }

        # Create a DataFrame with all rankings per year
        ranking_per_year = filtered_df.copy()

        ranking_per_year['Rating'] = pd.to_numeric(ranking_per_year['Rating'], errors='coerce')
        ranking_per_year.dropna(subset=['Rating'], inplace=True)

        # Ensure year is displayed as a full year in all relevant DataFrames
        ranking_per_year['Year'] = ranking_per_year['Year'].astype(int)

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
                "Year: %{x:.0f}<br>" +  # Ensure full year is displayed
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
                zeroline=False,
                dtick=1  # Show whole years
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

        st.plotly_chart(fig_scatter, use_container_width=True)

        # Rankings movement visualization
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Rankings Movement Over Time</h3>", unsafe_allow_html=True)

        fig_rankings = go.Figure()

        # Add data points for each team
        for team in TEAMS:
            team_data = ranking_per_year[ranking_per_year['Team'] == team]
            
            # Add scatter points
            fig_rankings.add_trace(go.Scatter(
            x=team_data['Year'],
            y=team_data['Position'],
            name=team,
            mode='markers',
            marker=dict(
                color=TEAM_COLORS[team],
                size=10,
                symbol='circle',
            ),
            hovertemplate=(
                f"<b>{team}</b><br>" +
                "Year: %{x:.0f}<br>" +  # Ensure full year is displayed
                "Rank: %{y}<br>" +
                "Rating: %{customdata:.1f}<br>" +
                "<extra></extra>"
            ),
            customdata=team_data['Rating']
            ))

            # Add connecting lines
            fig_rankings.add_trace(go.Scatter(
            x=team_data['Year'],
            y=team_data['Position'],
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
            zeroline=False,
            dtick=1  # Show whole years
            ),
            yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGrey',
            zeroline=False,
            autorange="reversed"
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

        st.plotly_chart(fig_rankings, use_container_width=True)

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
                        int(team_data['Position'].min()) if not pd.isna(team_data['Position'].min()) else 0,
                        team_data['Position'].quantile(0.25),
                        team_data['Position'].median(),
                        team_data['Position'].quantile(0.75),
                        int(team_data['Position'].max()) if not pd.isna(team_data['Position'].max()) else 0
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
        tab1, tab2 = st.tabs(["Head-to-Head Analysis", "Dominance Periods"])

        # First, add this CSS to improve tab spacing
        st.markdown("""
            <style>
            .stTabs [data-baseweb="tab-list"] {
                gap: 0px !important;
                width: 100%;
            }
            
            .stTabs [data-baseweb="tab"] {
                width: calc(100%/2) !important;
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
            


        with tab2:
            st.markdown("<h4 style='color:#f04f53; text-align: center;'>Team Dominance Periods</h4>", unsafe_allow_html=True)
            
            # Calculate dominance periods (when teams were ranked #1)
            top_ranked_periods = []
            for year in sorted(ranking_per_year['Year'].unique()):
                year_data = ranking_per_year[ranking_per_year['Year'] == year]
                if not year_data[year_data['Position'] == 1].empty:
                    top_team = year_data[year_data['Position'] == 1].iloc[0]
                    top_ranked_periods.append({
                        'Year': year,
                        'Team': top_team['Team'],
                        'Rating': top_team['Rating']
                    })
            
            dominance_df = pd.DataFrame(top_ranked_periods)
            
            if not dominance_df.empty:
                # Create dominance visualization
                fig_dom = go.Figure()
                
                # Add scatter points for each #1 ranking
                fig_dom.add_trace(go.Scatter(
                    x=dominance_df['Year'],
                    y=dominance_df['Team'],
                    mode='markers',
                    marker=dict(
                        size=20,
                        color=[TEAM_COLORS[team] for team in dominance_df['Team']],
                        symbol='square'
                    ),
                    hovertemplate=(
                        "<b>%{y}</b><br>" +
                        "Year: %{x}<br>" +
                        "Rating: %{customdata:.1f}<br>" +
                        "<extra></extra>"
                    ),
                    customdata=dominance_df['Rating'],
                    showlegend=False
                ))
                
                fig_dom.update_layout(
                    height=400,
                    title={
                        'text': "Periods of #1 Ranking",
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': {'color': '#f04f53'}
                    },
                    xaxis_title="Year",
                    yaxis_title="Team",
                    xaxis=dict(
                        dtick=1,
                        tickmode='linear',
                        tickformat='d'
                    ),
                    yaxis=dict(
                        categoryorder='category ascending'
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_dom, use_container_width=True)
                
                # Add dominance statistics
                st.markdown("#### Dominance Statistics")
                dom_stats = dominance_df['Team'].value_counts().reset_index()
                dom_stats.columns = ['Team', 'Years at #1']
                dom_stats = dom_stats.sort_values('Years at #1', ascending=False)
                st.dataframe(dom_stats, use_container_width=True, hide_index=True)
            else:
                st.info("No #1 ranking data available for the selected format and time period.")


        ##### TOTAL RANKINGS####

        # Total Rankings Section
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Total Rankings Across All 3 Formats</h3>", unsafe_allow_html=True)

        if not st.session_state.rankings_data.empty:
            # Group by Year and Team, sum the Ratings
            total_ratings = st.session_state.rankings_data.groupby(['Year', 'Team'])['Rating'].sum().reset_index()
            
            # For each year, assign positions based on total rating (highest = 1)
            total_rankings = []
            for year in total_ratings['Year'].unique():
                year_data = total_ratings[total_ratings['Year'] == year]
                sorted_teams = year_data.sort_values('Rating', ascending=False)
                for idx, row in enumerate(sorted_teams.itertuples(), 1):
                    total_rankings.append({
                        'Position': idx,
                        'Team': row.Team,
                        'Rating': row.Rating,
                        'Year': row.Year
                    })
            
            # Convert to DataFrame and sort
            total_rankings_df = pd.DataFrame(total_rankings)
            total_rankings_df = total_rankings_df.sort_values(['Year', 'Rating'], ascending=[False, False])
            
            # Display the total rankings table
            st.dataframe(
                total_rankings_df, 
                use_container_width=True,
                hide_index=True
            )

            # All-Time Ratings & Rankings Distribution for Total Rankings
            st.markdown("<h3 style='color:#f04f53; text-align: center;'>All-Time Total Ratings & Rankings Distribution</h3>", unsafe_allow_html=True)

            if not st.session_state.rankings_data.empty:
                fig_total_scatter = go.Figure()

                # Add scatter points for each team
                for team in TEAMS:
                    team_data = total_rankings_df[total_rankings_df['Team'] == team]
                    
                    fig_total_scatter.add_trace(go.Scatter(
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
                            "Year: %{x:.0f}<br>" +
                            "Total Rating: %{y:.1f}<br>" +
                            "Total Rank: %{customdata}<br>" +
                            "<extra></extra>"
                        ),
                        customdata=team_data['Position']
                    ))

                    # Add connecting lines between points
                    fig_total_scatter.add_trace(go.Scatter(
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
                fig_total_scatter.update_layout(
                    height=600,
                    xaxis_title="Year",
                    yaxis_title="Total Rating",
                    showlegend=True,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='LightGrey',
                        zeroline=False,
                        dtick=1
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

                st.plotly_chart(fig_total_scatter, use_container_width=True)

                # Rankings Movement visualization for Total Rankings
                st.markdown("<h3 style='color:#f04f53; text-align: center;'>Total Rankings Movement Over Time</h3>", unsafe_allow_html=True)

                fig_total_rankings = go.Figure()

                # Add data points for each team
                for team in TEAMS:
                    team_data = total_rankings_df[total_rankings_df['Team'] == team]
                    
                    # Add scatter points
                    fig_total_rankings.add_trace(go.Scatter(
                        x=team_data['Year'],
                        y=team_data['Position'],
                        name=team,
                        mode='markers',
                        marker=dict(
                            color=TEAM_COLORS[team],
                            size=10,
                            symbol='circle',
                        ),
                        hovertemplate=(
                            f"<b>{team}</b><br>" +
                            "Year: %{x:.0f}<br>" +
                            "Total Rank: %{y}<br>" +
                            "Total Rating: %{customdata:.1f}<br>" +
                            "<extra></extra>"
                        ),
                        customdata=team_data['Rating']
                    ))

                    # Add connecting lines
                    fig_total_rankings.add_trace(go.Scatter(
                        x=team_data['Year'],
                        y=team_data['Position'],
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

                fig_total_rankings.update_layout(
                    height=600,
                    xaxis_title="Year",
                    yaxis_title="Total Ranking",
                    showlegend=True,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='LightGrey',
                        zeroline=False,
                        dtick=1
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='LightGrey',
                        zeroline=False,
                        autorange="reversed"
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

                st.plotly_chart(fig_total_rankings, use_container_width=True)

                # Total Rankings Dominance Periods Analysis
                st.markdown("<h3 style='color:#f04f53; text-align: center;'>Total Rankings Dominance Periods</h3>", unsafe_allow_html=True)
                
                # Calculate dominance periods (when teams were ranked #1 in total rankings)
                top_ranked_total_periods = []
                for year in sorted(total_rankings_df['Year'].unique()):
                    year_data = total_rankings_df[total_rankings_df['Year'] == year]
                    if not year_data[year_data['Position'] == 1].empty:
                        top_team = year_data[year_data['Position'] == 1].iloc[0]
                    top_ranked_total_periods.append({
                        'Year': year,
                        'Team': top_team['Team'],
                        'Rating': top_team['Rating']
                    })
                
                total_dominance_df = pd.DataFrame(top_ranked_total_periods)
                
                if not total_dominance_df.empty:
                    # Create dominance visualization
                    fig_total_dom = go.Figure()
                    
                    # Add scatter points for each #1 ranking
                    fig_total_dom.add_trace(go.Scatter(
                    x=total_dominance_df['Year'],
                    y=total_dominance_df['Team'],
                    mode='markers',
                    marker=dict(
                        size=20,
                        color=[TEAM_COLORS[team] for team in total_dominance_df['Team']],
                        symbol='square'
                    ),
                    hovertemplate=(
                        "<b>%{y}</b><br>" +
                        "Year: %{x}<br>" +
                        "Total Rating: %{customdata:.1f}<br>" +
                        "<extra></extra>"
                    ),
                    customdata=total_dominance_df['Rating'],
                    showlegend=False
                    ))
                    
                    fig_total_dom.update_layout(
                    height=400,
                    title={
                        'text': "Periods of #1 Total Ranking",
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': {'color': '#f04f53'}
                    },
                    xaxis_title="Year",
                    yaxis_title="Team",
                    xaxis=dict(
                        dtick=1,
                        tickmode='linear',
                        tickformat='d'
                    ),
                    yaxis=dict(
                        categoryorder='category ascending'
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_total_dom, use_container_width=True)
                    
                    # Add dominance statistics
                    st.markdown("#### Total Rankings Dominance Statistics")
                    total_dom_stats = total_dominance_df['Team'].value_counts().reset_index()
                    total_dom_stats.columns = ['Team', 'Years at #1 (Total Rankings)']
                    total_dom_stats = total_dom_stats.sort_values('Years at #1 (Total Rankings)', ascending=False)
                    st.dataframe(total_dom_stats, use_container_width=True, hide_index=True)
                else:
                    st.info("No #1 total ranking data available.")


    if st.session_state.rankings_data.empty:
        st.info("No rankings data available. Please update rankings in the 'Update Rankings' tab.")
