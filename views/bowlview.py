import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

# Add this CSS styling after imports
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

/* Tab styling for better fit and scrolling */
.stTabs [data-baseweb="tab-list"] {
    width: 100%;
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    border-radius: 15px;
    padding: 12px;
    box-shadow: 0 8px 32px rgba(168, 237, 234, 0.3);
    margin-bottom: 2rem;
    overflow-x: auto; /* Make it scrollable horizontally */
    white-space: nowrap; /* Prevent tabs from wrapping */
    scrollbar-width: thin; /* For Firefox */
    scrollbar-color: rgba(255, 255, 255, 0.5) rgba(255, 255, 255, 0.2);
}

/* Custom Scrollbar for Tabs - Webkit browsers */
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
    height: 8px;
}

.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 10px;
}

.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.5);
    border-radius: 10px;
    transition: background 0.3s ease;
}

.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.7);
}

.stTabs [data-baseweb="tab"] {
    flex-grow: 1;
    flex-shrink: 1;
    text-align: center;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 8px;
    margin: 0 3px;
    transition: all 0.4s ease;
    color: #2c3e50 !important;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 8px 12px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255, 255, 255, 0.4);
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(255, 255, 255, 0.2);
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #f04f53 0%, #f5576c 100%);
    color: white !important;
    box-shadow: 0 6px 20px rgba(240, 79, 83, 0.4);
    transform: translateY(-3px);
    border: 1px solid rgba(255, 255, 255, 0.3);
}

.stTabs [data-baseweb="tab"][aria-selected="true"]:hover {
    background: linear-gradient(135deg, #e03a3e 0%, #f04f53 100%);
}
</style>
""", unsafe_allow_html=True)

def get_filtered_options(df, column, selected_filters=None):
    """
    Get available options for a column based on current filter selections.
    
    Args:
        df: The DataFrame to filter
        column: The column to get unique values from
        selected_filters: Dictionary of current filter selections
    """
    if selected_filters is None:
        return ['All'] + sorted(df[column].unique().tolist())
    
    filtered_df = df.copy()
    
    # Apply each active filter
    for filter_col, filter_val in selected_filters.items():
        if filter_val and 'All' not in filter_val and filter_col != column:
            filtered_df = filtered_df[filtered_df[filter_col].isin(filter_val)]
    
    return ['All'] + sorted(filtered_df[column].unique().tolist())

def display_bowl_view():
    if 'bowl_df' in st.session_state:
        # Get the bowling dataframe with safer date parsing
        try:
            bowl_df = st.session_state['bowl_df'].copy()
            bowl_df['Date'] = pd.to_datetime(bowl_df['Date'], format='%d %b %Y', errors='coerce')
            bowl_df['Year'] = bowl_df['Date'].dt.year
            # Add HomeOrAway column
            bowl_df['HomeOrAway'] = np.where(bowl_df['Bowl_Team'] == bowl_df['Home_Team'], 'Home', 'Away')

        except Exception as e:
            st.error(f"Error processing dates or adding HomeOrAway column. Using original dates.")
            bowl_df = st.session_state['bowl_df'].copy()
            bowl_df['Date'] = pd.to_datetime(bowl_df['Date'], errors='coerce')
            bowl_df['Year'] = bowl_df['Date'].dt.year
            # Add HomeOrAway column even if date parsing fails partially
            if 'Bowl_Team' in bowl_df.columns and 'Home_Team' in bowl_df.columns:
                 bowl_df['HomeOrAway'] = np.where(bowl_df['Bowl_Team'] == bowl_df['Home_Team'], 'Home', 'Away')
            else:
                 st.warning("Could not determine Home/Away status due to missing columns.")
                 bowl_df['HomeOrAway'] = 'Unknown' # Default value

        # Add data validation check and reset filters if needed
        if 'prev_bowl_teams' not in st.session_state:
            st.session_state.prev_bowl_teams = set()
            
        current_bowl_teams = set(bowl_df['Bowl_Team'].unique())
        
        # Reset filters if the available teams have changed
        if current_bowl_teams != st.session_state.prev_bowl_teams:
            st.session_state.bowl_filter_state = {
                'name': ['All'],
                'bowl_team': ['All'],
                'bat_team': ['All'],
                'match_format': ['All'],
                'comp': ['All']  # Initialize 'comp' filter
            }
            st.session_state.prev_bowl_teams = current_bowl_teams        ###-------------------------------------HEADER AND FILTERS-------------------------------------###
        # Modern Main Header
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 20px; margin: 1rem 0 2rem 0; text-align: center; 
                    box-shadow: 0 12px 48px rgba(102, 126, 234, 0.3); 
                    border: 1px solid rgba(255, 255, 255, 0.2);">
            <h1 style="color: white !important; margin: 0 !important; font-weight: bold; 
                       font-size: 2.5rem; text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);">
                🏏 Bowling Statistics & Analysis
            </h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if only one scorecard is loaded
        unique_matches = bowl_df['File Name'].nunique()
        if unique_matches <= 1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
                        border-left: 4px solid #ffc107; padding: 1rem 1.5rem; border-radius: 10px;
                        margin: 1rem 0; box-shadow: 0 4px 16px rgba(255, 193, 7, 0.2);">
                <p style="margin: 0; font-weight: 600; color: #856404;">
                    ⚠️ Please upload more than 1 scorecard to use the bowling statistics view effectively. 
                    With only one match loaded, statistical analysis and comparisons are limited.
                </p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Initialize session state for filters if not exists
        if 'bowl_filter_state' not in st.session_state:
            st.session_state.bowl_filter_state = {
                'name': ['All'],
                'bowl_team': ['All'],
                'bat_team': ['All'],
                'match_format': ['All'],
                'comp': ['All']  # Initialize 'comp' filter
            }
        
        # Create filters at the top of the page
        selected_filters = {
            'Name': st.session_state.bowl_filter_state['name'],
            'Bowl_Team': st.session_state.bowl_filter_state['bowl_team'],
            'Bat_Team': st.session_state.bowl_filter_state['bat_team'],
            'Match_Format': st.session_state.bowl_filter_state['match_format'],
            'comp': st.session_state.bowl_filter_state['comp']
        }

        # Create filter lists
        names = get_filtered_options(bowl_df, 'Name', 
            {k: v for k, v in selected_filters.items() if k != 'Name' and 'All' not in v})
        bowl_teams = get_filtered_options(bowl_df, 'Bowl_Team', 
            {k: v for k, v in selected_filters.items() if k != 'Bowl_Team' and 'All' not in v})
        bat_teams = get_filtered_options(bowl_df, 'Bat_Team', 
            {k: v for k, v in selected_filters.items() if k != 'Bat_Team' and 'All' not in v})
        match_formats = get_filtered_options(bowl_df, 'Match_Format', 
            {k: v for k, v in selected_filters.items() if k != 'Match_Format' and 'All' not in v})

        # Get list of years before creating the slider
        years = sorted(bowl_df['Year'].unique().tolist())

        # Create five columns for filters
        col1, col2, col3, col4, col5 = st.columns(5)  # Add fifth column for comp
        
        with col1:
            name_choice = st.multiselect('Name:', 
                                       names,
                                       default=st.session_state.bowl_filter_state['name'])
            if name_choice != st.session_state.bowl_filter_state['name']:
                st.session_state.bowl_filter_state['name'] = name_choice
                st.rerun()

        with col2:
            bowl_team_choice = st.multiselect('Bowl Team:', 
                                            bowl_teams,
                                            default=st.session_state.bowl_filter_state['bowl_team'])
            if bowl_team_choice != st.session_state.bowl_filter_state['bowl_team']:
                st.session_state.bowl_filter_state['bowl_team'] = bowl_team_choice
                st.rerun()

        with col3:
            bat_team_choice = st.multiselect('Bat Team:', 
                                           bat_teams,
                                           default=st.session_state.bowl_filter_state['bat_team'])
            if bat_team_choice != st.session_state.bowl_filter_state['bat_team']:
                st.session_state.bowl_filter_state['bat_team'] = bat_team_choice
                st.rerun()

        with col4:
            match_format_choice = st.multiselect('Format:', 
                                               match_formats,
                                               default=st.session_state.bowl_filter_state['match_format'])
            if match_format_choice != st.session_state.bowl_filter_state['match_format']:
                st.session_state.bowl_filter_state['match_format'] = match_format_choice
                st.rerun()

        with col5:
            try:
                available_comp = get_filtered_options(bowl_df, 'comp',
                    {k: v for k, v in selected_filters.items() if k != 'comp' and 'All' not in v})
            except KeyError:
                print("Error accessing comp column, using Competition instead")
                available_comp = get_filtered_options(bowl_df, 'Competition',
                    {k: v for k, v in selected_filters.items() if k != 'comp' and 'All' not in v})
            
            comp_choice = st.multiselect('Competition:',
                                       available_comp,
                                       default=[c for c in st.session_state.bowl_filter_state['comp'] if c in available_comp])
            if comp_choice != st.session_state.bowl_filter_state['comp']:
                st.session_state.bowl_filter_state['comp'] = comp_choice
                st.rerun()

        # Get individual players and create color mapping
        individual_players = [name for name in name_choice if name != 'All']
        
        # Create color dictionary for selected players
        player_colors = {}
        if individual_players:
            player_colors[individual_players[0]] = '#f84e4e'
            for name in individual_players[1:]:
                player_colors[name] = f'#{random.randint(0, 0xFFFFFF):06x}'
        all_color = '#f84e4e' if not individual_players else 'black'
        player_colors['All'] = all_color

        # Calculate range filter statistics
        career_stats = bowl_df.groupby('Name').agg({
            'File Name': 'nunique',
            'Bowler_Wkts': 'sum',
            'Bowler_Runs': 'sum',
            'Bowler_Balls': 'sum'
        }).reset_index()

        career_stats['Avg'] = career_stats['Bowler_Runs'] / career_stats['Bowler_Wkts'].replace(0, np.inf)
        career_stats['SR'] = career_stats['Bowler_Balls'] / career_stats['Bowler_Wkts'].replace(0, np.inf)
        career_stats['Avg'] = career_stats['Avg'].replace([np.inf, -np.inf], np.nan)
        career_stats['SR'] = career_stats['SR'].replace([np.inf, -np.inf], np.nan)

        # Calculate max values
        max_wickets = int(career_stats['Bowler_Wkts'].max())
        max_matches = int(career_stats['File Name'].max())
        max_avg = float(career_stats['Avg'].max())
        max_sr = float(career_stats['SR'].max())

        # Add range filters
        col5, col6, col7, col8, col9, col10 = st.columns(6)

        # Replace the year slider section with this:
        with col5:
            st.markdown("<p style='text-align: center;'>Choose Year:</p>", unsafe_allow_html=True)
            if len(years) == 1:
                st.markdown(f"<p style='text-align: center;'>{years[0]}</p>", unsafe_allow_html=True)
                year_choice = (years[0], years[0])  # Set the year range to the single year
            else:
                year_choice = st.slider('', 
                        min_value=min(years),
                        max_value=max(years),
                        value=(min(years), max(years)),
                        label_visibility='collapsed',
                        key='year_slider')

        # The rest of the sliders remain the same
        with col6:
            st.markdown("<p style='text-align: center;'>Choose Position:</p>", unsafe_allow_html=True)
            position_choice = st.slider('', 
                        min_value=1, 
                        max_value=11, 
                        value=(1, 11),
                        label_visibility='collapsed',
                        key='position_slider')

        with col7:
            st.markdown("<p style='text-align: center;'>Wickets Range</p>", unsafe_allow_html=True)
            wickets_range = st.slider('', 
                                    min_value=0, 
                                    max_value=max_wickets, 
                                    value=(0, max_wickets),
                                    label_visibility='collapsed',
                                    key='wickets_slider')

        with col8:
            st.markdown("<p style='text-align: center;'>Matches Range</p>", unsafe_allow_html=True)
            matches_range = st.slider('', 
                                    min_value=1, 
                                    max_value=max_matches, 
                                    value=(1, max_matches),
                                    label_visibility='collapsed',
                                    key='matches_slider')

        with col9:
            st.markdown("<p style='text-align: center;'>Average Range</p>", unsafe_allow_html=True)
            avg_range = st.slider('', 
                                min_value=0.0, 
                                max_value=max_avg, 
                                value=(0.0, max_avg),
                                label_visibility='collapsed',
                                key='avg_slider')

        with col10:
            st.markdown("<p style='text-align: center;'>Strike Rate Range</p>", unsafe_allow_html=True)
            sr_range = st.slider('', 
                                min_value=0.0, 
                                max_value=max_sr, 
                                value=(0.0, max_sr),
                                label_visibility='collapsed',
                                key='sr_slider')

###-------------------------------------APPLY FILTERS-------------------------------------###
        # Create filtered dataframe
        filtered_df = bowl_df.copy()

        # Apply basic filters
        if name_choice and 'All' not in name_choice:
            filtered_df = filtered_df[filtered_df['Name'].isin(name_choice)]
        if bowl_team_choice and 'All' not in bowl_team_choice:
            filtered_df = filtered_df[filtered_df['Bowl_Team'].isin(bowl_team_choice)]
        if bat_team_choice and 'All' not in bat_team_choice:
            filtered_df = filtered_df[filtered_df['Bat_Team'].isin(bat_team_choice)]
        if match_format_choice and 'All' not in match_format_choice:
            filtered_df = filtered_df[filtered_df['Match_Format'].isin(match_format_choice)]
        if comp_choice and 'All' not in comp_choice:
            filtered_df = filtered_df[filtered_df['comp'].isin(comp_choice)]

        # Apply year filter
        filtered_df = filtered_df[filtered_df['Year'].between(year_choice[0], year_choice[1])]

        # Apply range filters
        filtered_df = filtered_df.groupby('Name').filter(lambda x: 
            wickets_range[0] <= x['Bowler_Wkts'].sum() <= wickets_range[1] and
            matches_range[0] <= x['File Name'].nunique() <= matches_range[1] and
            (avg_range[0] <= (x['Bowler_Runs'].sum() / x['Bowler_Wkts'].sum()) <= avg_range[1] if x['Bowler_Wkts'].sum() > 0 else True) and
            (sr_range[0] <= (x['Bowler_Balls'].sum() / x['Bowler_Wkts'].sum()) <= sr_range[1] if x['Bowler_Wkts'].sum() > 0 else True)
        )

        filtered_df = filtered_df[filtered_df['Position'].between(position_choice[0], position_choice[1])]

        # Create a placeholder for tabs that will be lazily loaded
        main_container = st.container()
        
        # Create tabs for different views with clean, short names
        tabs = main_container.tabs([
            "Career", "Format", "Season", "Latest", "Opponent", 
            "Location", "Innings", "Position", "Cumulative", 
            "Block", "Home/Away", "Records"
        ])

        ###-------------------------------------CAREER STATS-------------------------------------###
        # Career Stats Tab
        with tabs[0]:
            # Calculate career statistics
            match_wickets = filtered_df.groupby(['Name', 'File Name'])['Bowler_Wkts'].sum().reset_index()
            ten_wickets = match_wickets[match_wickets['Bowler_Wkts'] >= 10].groupby('Name').size().reset_index(name='10W')

            bowlcareer_df = filtered_df.groupby('Name').agg({
                'File Name': 'nunique',
                'Bowler_Balls': 'sum',
                'Maidens': 'sum',
                'Bowler_Runs': 'sum',
                'Bowler_Wkts': 'sum'
            }).reset_index()

            bowlcareer_df.columns = ['Name', 'Matches', 'Balls', 'M/D', 'Runs', 'Wickets']

            # Calculate career metrics
            bowlcareer_df['Overs'] = (bowlcareer_df['Balls'] // 6) + (bowlcareer_df['Balls'] % 6) / 10
            bowlcareer_df['Strike Rate'] = (bowlcareer_df['Balls'] / bowlcareer_df['Wickets']).round(2)
            bowlcareer_df['Economy Rate'] = (bowlcareer_df['Runs'] / bowlcareer_df['Overs']).round(2)
            bowlcareer_df['Avg'] = (bowlcareer_df['Runs'] / bowlcareer_df['Wickets']).round(2)

            # Add additional statistics
            five_wickets = filtered_df[filtered_df['Bowler_Wkts'] >= 5].groupby('Name').size().reset_index(name='5W')
            bowlcareer_df = bowlcareer_df.merge(five_wickets, on='Name', how='left')
            bowlcareer_df['5W'] = bowlcareer_df['5W'].fillna(0).astype(int)
            
            bowlcareer_df = bowlcareer_df.merge(ten_wickets, on='Name', how='left')
            bowlcareer_df['10W'] = bowlcareer_df['10W'].fillna(0).astype(int)

            bowlcareer_df['WPM'] = (bowlcareer_df['Wickets'] / bowlcareer_df['Matches']).round(2)

            pom_counts = filtered_df[filtered_df['Player_of_the_Match'] == filtered_df['Name']].groupby('Name')['File Name'].nunique().reset_index(name='POM')
            bowlcareer_df = bowlcareer_df.merge(pom_counts, on='Name', how='left')
            bowlcareer_df['POM'] = bowlcareer_df['POM'].fillna(0).astype(int)

            bowlcareer_df = bowlcareer_df.replace([np.inf, -np.inf], np.nan)

            # Final column ordering
            bowlcareer_df = bowlcareer_df[[
                'Name', 'Matches', 'Balls', 'Overs', 'M/D', 'Runs', 'Wickets', 'Avg',
                'Strike Rate', 'Economy Rate', '5W', '10W', 'WPM', 'POM'
            ]]

            st.markdown("""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">🎳 Career Statistics</h3>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(bowlcareer_df, use_container_width=True, hide_index=True)

            # Create a new figure for the scatter plot
            scatter_fig = go.Figure()

            for name in bowlcareer_df['Name'].unique():
                player_stats = bowlcareer_df[bowlcareer_df['Name'] == name]
                
                # Get bowling statistics
                economy_rate = player_stats['Economy Rate'].iloc[0]
                strike_rate = player_stats['Strike Rate'].iloc[0]
                wickets = player_stats['Wickets'].iloc[0]
                
                # Add scatter point for the player
                scatter_fig.add_trace(go.Scatter(
                    x=[economy_rate],
                    y=[strike_rate],
                    mode='markers+text',
                    text=[name],
                    textposition='top center',
                    marker=dict(size=10),
                    name=name,
                    hovertemplate=(
                        f"<b>{name}</b><br><br>"
                        f"Economy Rate: {economy_rate:.2f}<br>"
                        f"Strike Rate: {strike_rate:.2f}<br>"
                        f"Wickets: {wickets}<br>"
                        "<extra></extra>"
                    )
                ))

            # Display the title using Streamlit's markdown
            st.markdown(
                "<h3 style='color:#f04f53; text-align: center;'>Economy Rate vs Strike Rate Analysis</h3>",
                unsafe_allow_html=True
            )

            # Update layout
            scatter_fig.update_layout(
                xaxis_title="Economy Rate",
                yaxis_title="Strike Rate",
                height=500,
                font=dict(size=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )

            # Show plot
            st.plotly_chart(scatter_fig, use_container_width=True)

        ###-------------------------------------FORMAT STATS-------------------------------------###
        # Format Stats Tab
        with tabs[1]:
            # Calculate format statistics
            match_wickets = filtered_df.groupby(['Name', 'Match_Format', 'File Name'])['Bowler_Wkts'].sum().reset_index()
            ten_wickets = match_wickets[match_wickets['Bowler_Wkts'] >= 10].groupby(['Name', 'Match_Format']).size().reset_index(name='10W')

            bowlformat_df = filtered_df.groupby(['Name', 'Match_Format']).agg({
                'File Name': 'nunique',
                'Bowler_Balls': 'sum',
                'Maidens': 'sum',
                'Bowler_Runs': 'sum',
                'Bowler_Wkts': 'sum'
            }).reset_index()

            bowlformat_df.columns = ['Name', 'Match_Format', 'Matches', 'Balls', 'M/D', 'Runs', 'Wickets']

            # Calculate format metrics
            bowlformat_df['Overs'] = (bowlformat_df['Balls'] // 6) + (bowlformat_df['Balls'] % 6) / 10
            bowlformat_df['Strike Rate'] = (bowlformat_df['Balls'] / bowlformat_df['Wickets']).round(2)
            bowlformat_df['Economy Rate'] = (bowlformat_df['Runs'] / bowlformat_df['Overs']).round(2)

            # Add additional statistics
            five_wickets = filtered_df[filtered_df['Bowler_Wkts'] >= 5].groupby(['Name', 'Match_Format']).size().reset_index(name='5W')
            bowlformat_df = bowlformat_df.merge(five_wickets, on=['Name', 'Match_Format'], how='left')
            bowlformat_df['5W'] = bowlformat_df['5W'].fillna(0).astype(int)
            
            bowlformat_df = bowlformat_df.merge(ten_wickets, on=['Name', 'Match_Format'], how='left')
            bowlformat_df['10W'] = bowlformat_df['10W'].fillna(0).astype(int)

            bowlformat_df['WPM'] = (bowlformat_df['Wickets'] / bowlformat_df['Matches']).round(2)

            pom_counts = filtered_df[filtered_df['Player_of_the_Match'] == filtered_df['Name']].groupby(['Name', 'Match_Format'])['File Name'].nunique().reset_index(name='POM')
            bowlformat_df = bowlformat_df.merge(pom_counts, on=['Name', 'Match_Format'], how='left')
            bowlformat_df['POM'] = bowlformat_df['POM'].fillna(0).astype(int)

            bowlformat_df = bowlformat_df.replace([np.inf, -np.inf], np.nan)
            bowlformat_df = bowlformat_df.rename(columns={'Match_Format': 'Format'})

            # Final column ordering
            bowlformat_df = bowlformat_df[[
                'Name', 'Format', 'Matches', 'Balls', 'Overs', 'M/D', 'Runs', 'Wickets',
                'Strike Rate', 'Economy Rate', '5W', '10W', 'WPM', 'POM'
            ]]

            st.markdown("""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 1rem; margin: 1rem 0; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(240, 147, 251, 0.3);
                        border: 1px solid rgba(255, 255, 255, 0.2);">
                <h3 style="color: white !important; margin: 0 !important; font-weight: bold; font-size: 1.3rem; text-align: center;">📋 Format Record</h3>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(bowlformat_df, use_container_width=True, hide_index=True)

        ###-------------------------------------SEASON STATS-------------------------------------###
        # Season Stats Tab
        with tabs[2]:
            # Calculate season statistics
            match_wickets = filtered_df.groupby(['Name', 'Year', 'File Name'])['Bowler_Wkts'].sum().reset_index()
            ten_wickets = match_wickets[match_wickets['Bowler_Wkts'] >= 10].groupby(['Name', 'Year']).size().reset_index(name='10W')

            bowlseason_df = filtered_df.groupby(['Name', 'Year']).agg({
                'File Name': 'nunique',
                'Bowler_Balls': 'sum',
                'Maidens': 'sum',
                'Bowler_Runs': 'sum',
                'Bowler_Wkts': 'sum'
            }).reset_index()

            bowlseason_df.columns = ['Name', 'Year', 'Matches', 'Balls', 'M/D', 'Runs', 'Wickets']

            # Calculate season metrics
            bowlseason_df['Overs'] = (bowlseason_df['Balls'] // 6) + (bowlseason_df['Balls'] % 6) / 10
            bowlseason_df['Strike Rate'] = (bowlseason_df['Balls'] / bowlseason_df['Wickets']).round(2)
            bowlseason_df['Economy Rate'] = (bowlseason_df['Runs'] / bowlseason_df['Overs']).round(2)

            # Add additional statistics
            five_wickets = filtered_df[filtered_df['Bowler_Wkts'] >= 5].groupby(['Name', 'Year']).size().reset_index(name='5W')
            bowlseason_df = bowlseason_df.merge(five_wickets, on=['Name', 'Year'], how='left')
            bowlseason_df['5W'] = bowlseason_df['5W'].fillna(0).astype(int)
            
            bowlseason_df = bowlseason_df.merge(ten_wickets, on=['Name', 'Year'], how='left')
            bowlseason_df['10W'] = bowlseason_df['10W'].fillna(0).astype(int)

            bowlseason_df['WPM'] = (bowlseason_df['Wickets'] / bowlseason_df['Matches']).round(2)

            pom_counts = filtered_df[filtered_df['Player_of_the_Match'] == filtered_df['Name']].groupby(['Name', 'Year'])['File Name'].nunique().reset_index(name='POM')
            bowlseason_df = bowlseason_df.merge(pom_counts, on=['Name', 'Year'], how='left')
            bowlseason_df['POM'] = bowlseason_df['POM'].fillna(0).astype(int)

            bowlseason_df = bowlseason_df.replace([np.inf, -np.inf], np.nan)
            bowlseason_df = bowlseason_df.sort_values(['Name', 'Year'])

            # Final column ordering
            bowlseason_df = bowlseason_df[[
                'Name', 'Year', 'Matches', 'Balls', 'Overs', 'M/D', 'Runs', 'Wickets',
                'Strike Rate', 'Economy Rate', '5W', '10W', 'WPM', 'POM'
            ]]

            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Season Statistics</h3>", unsafe_allow_html=True)
            st.dataframe(bowlseason_df, use_container_width=True, hide_index=True)

            ###-------------------------------------SEASON GRAPHS----------------------------------------###
            # Create subplots for Bowling Average, Strike Rate, and Economy Rate
            fig = make_subplots(rows=1, cols=3, subplot_titles=("Bowling Average", "Strike Rate", "Economy Rate"))

            # Handle 'All' selection
            if 'All' in name_choice:
                all_players_stats = filtered_df.groupby('Year').agg({
                    'Bowler_Runs': 'sum',
                    'Bowler_Wkts': 'sum',
                    'Bowler_Balls': 'sum'
                }).reset_index()

                all_players_stats['Avg'] = (all_players_stats['Bowler_Runs'] / all_players_stats['Bowler_Wkts']).round(2).fillna(0)
                all_players_stats['SR'] = (all_players_stats['Bowler_Balls'] / all_players_stats['Bowler_Wkts']).round(2).fillna(0)
                all_players_stats['Econ'] = (all_players_stats['Bowler_Runs'] / (all_players_stats['Bowler_Balls']/6)).round(2).fillna(0)

                # Use streamlit red if only 'All' selected, black if names also selected
                all_color = '#f84e4e' if not individual_players else 'black'

                # Add traces for 'All'
                fig.add_trace(go.Scatter(
                    x=all_players_stats['Year'], 
                    y=all_players_stats['Avg'], 
                    mode='lines+markers', 
                    name='All Players',
                    legendgroup='All',
                    marker=dict(color=all_color, size=8)
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=all_players_stats['Year'], 
                    y=all_players_stats['SR'], 
                    mode='lines+markers', 
                    name='All Players',
                    legendgroup='All',
                    marker=dict(color=all_color, size=8),
                    showlegend=False
                ), row=1, col=2)
                
                fig.add_trace(go.Scatter(
                    x=all_players_stats['Year'], 
                    y=all_players_stats['Econ'], 
                    mode='lines+markers', 
                    name='All Players',
                    legendgroup='All',
                    marker=dict(color=all_color, size=8),
                    showlegend=False
                ), row=1, col=3)

            # Add individual player traces
            for i, name in enumerate(individual_players):
                player_stats = filtered_df[filtered_df['Name'] == name]
                player_yearly_stats = player_stats.groupby('Year').agg({
                    'Bowler_Runs': 'sum',
                    'Bowler_Wkts': 'sum',
                    'Bowler_Balls': 'sum'
                }).reset_index()

                player_yearly_stats['Avg'] = (player_yearly_stats['Bowler_Runs'] / player_yearly_stats['Bowler_Wkts']).round(2).fillna(0)
                player_yearly_stats['SR'] = (player_yearly_stats['Bowler_Balls'] / player_yearly_stats['Bowler_Wkts']).round(2).fillna(0)
                player_yearly_stats['Econ'] = (player_yearly_stats['Bowler_Runs'] / (player_yearly_stats['Bowler_Balls']/6)).round(2).fillna(0)

                # First player gets streamlit red, others get random colors
                color = '#f84e4e' if i == 0 else f'#{random.randint(0, 0xFFFFFF):06x}'

                # Add bowling average trace
                fig.add_trace(go.Bar(
                    x=player_yearly_stats['Year'], 
                    y=player_yearly_stats['Avg'], 
                    name=name,
                    legendgroup=name,
                    marker_color=color,
                    showlegend=True
                ), row=1, col=1)

                # Add strike rate trace
                fig.add_trace(go.Bar(
                    x=player_yearly_stats['Year'], 
                    y=player_yearly_stats['SR'], 
                    name=name,
                    legendgroup=name,
                    marker_color=color,
                    showlegend=False
                ), row=1, col=2)
                
                # Add economy rate trace
                fig.add_trace(go.Bar(
                    x=player_yearly_stats['Year'], 
                    y=player_yearly_stats['Econ'], 
                    name=name,
                    legendgroup=name,
                    marker_color=color,
                    showlegend=False
                ), row=1, col=3)

            # Update layout
            fig.update_layout(
                title="<b>Season Performance Metrics</b>",
                title_x=0.5,
                showlegend=True,
                yaxis_title="Average (Runs/Wicket)",
                yaxis2_title="Strike Rate (Balls/Wicket)",
                yaxis3_title="Economy Rate (Runs/Over)",
                height=500,
                font=dict(size=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                )
            )

            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)', title_text="Year")
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

            st.plotly_chart(fig, use_container_width=True)

            # Create wickets per year chart
            fig = go.Figure()
            
            # Add wickets per year for 'All' if selected
            if 'All' in name_choice:
                wickets_all = filtered_df.groupby('Year')['Bowler_Wkts'].sum().reset_index()
                all_color = '#f84e4e' if not individual_players else 'black'
                
                fig.add_trace(
                    go.Bar(
                        x=wickets_all['Year'], 
                        y=wickets_all['Bowler_Wkts'],
                        name='All Players',
                        marker_color=all_color
                    )
                )

            # Add individual player wickets
            for i, name in enumerate(individual_players):
                player_wickets = filtered_df[filtered_df['Name'] == name].groupby('Year')['Bowler_Wkts'].sum().reset_index()
                color = '#f84e4e' if i == 0 else f'#{random.randint(0, 0xFFFFFF):06x}'
                
                fig.add_trace(
                    go.Bar(
                        x=player_wickets['Year'], 
                        y=player_wickets['Bowler_Wkts'],
                        name=name,
                        marker_color=color
                    )
                )

            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Wickets Per Year</h3>", unsafe_allow_html=True)

            fig.update_layout(
                showlegend=True,
                height=500,
                xaxis_title='Year',
                yaxis_title='Wickets',
                margin=dict(l=50, r=50, t=70, b=50),
                font=dict(size=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                barmode='group'
            )

            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

            st.plotly_chart(fig, use_container_width=True)

        ###-------------------------------------LATEST INNINGS-------------------------------------###
        # Latest Innings Tab
        with tabs[3]:
            # Create latest innings dataframe first to get the last 20 innings
            latest_inns_df = filtered_df.groupby(['Name', 'Match_Format', 'Date', 'Innings']).agg({
                'Bowl_Team': 'first',
                'Bat_Team': 'first',
                'Overs': 'sum',
                'Maidens': 'sum',
                'Bowler_Runs': 'sum',
                'Bowler_Wkts': 'sum',
                'File Name': 'first'  # Include the File Name for match counting
            }).reset_index()

            # Rename columns
            latest_inns_df = latest_inns_df.rename(columns={
                'Match_Format': 'Format',
                'Bowl_Team': 'Team',
                'Bat_Team': 'Opponent',
                'Bowler_Runs': 'Runs',
                'Bowler_Wkts': 'Wickets'
            })

            # Process and sort dates
            latest_inns_df['Date'] = pd.to_datetime(latest_inns_df['Date'])
            latest_inns_df = latest_inns_df.sort_values(by='Date', ascending=False).head(20)
            
            # Create a list of file names from the last 20 innings to filter the original dataframe
            last_20_match_files = latest_inns_df['File Name'].unique().tolist()
            
            # Filter the original dataframe to include only matches from the last 20 innings
            last_20_df = filtered_df[filtered_df['File Name'].isin(last_20_match_files)]
            
            # Calculate metrics from only these 20 innings
            total_matches = last_20_df['File Name'].nunique()
            total_innings = len(latest_inns_df)
            total_wickets = latest_inns_df['Wickets'].sum()
            total_runs = latest_inns_df['Runs'].sum()
            total_balls = last_20_df['Bowler_Balls'].sum()  # Calculate from original df for accurate ball count
            total_overs = (total_balls // 6) + (total_balls % 6) / 10
            total_maidens = latest_inns_df['Maidens'].sum()
            
            # Calculate strike rate and economy with safety checks
            strike_rate = (total_balls / total_wickets) if total_wickets > 0 else 0
            economy_rate = (total_runs / total_overs) if total_overs > 0 else 0
            bowling_avg = (total_runs / total_wickets) if total_wickets > 0 else 0
            
            # Count number of 5-wicket hauls in the last 20 innings
            five_wicket_hauls = len(latest_inns_df[latest_inns_df['Wickets'] >= 5])
            
            # Format date in the dataframe for display
            latest_inns_df['Date'] = latest_inns_df['Date'].dt.strftime('%d/%m/%Y')
            
            # Format Overs to 1 decimal place
            latest_inns_df['Overs'] = latest_inns_df['Overs'].apply(lambda x: f"{x:.1f}")
            
            # Display all metrics in a single row with borders
            col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
            
            with col1:
                st.metric("Matches", f"{total_matches}", border=True)
            with col2:
                st.metric("Innings", f"{total_innings}", border=True)
            with col3:
                st.metric("Wickets", f"{total_wickets}", border=True)
            with col4:
                st.metric("Average", f"{bowling_avg:.2f}", border=True)
            with col5:
                st.metric("Strike Rate", f"{strike_rate:.2f}", border=True)
            with col6:
                st.metric("Economy", f"{economy_rate:.2f}", border=True)
            with col7:
                st.metric("5 Wicket Hauls", f"{five_wicket_hauls}", border=True)
            with col8:
                st.metric("Maidens", f"{total_maidens}", border=True)

            # Reorder columns for display - remove File Name from display
            display_inns_df = latest_inns_df[[
                'Name', 'Format', 'Date', 'Innings', 'Team', 'Opponent',
                'Overs', 'Maidens', 'Runs', 'Wickets'
            ]]

            # Apply conditional formatting
            def color_wickets(value):
                if value == 0:
                    return 'background-color: #DE6A73'  # Light Red
                elif value <= 2:
                    return 'background-color: #DEAD68'  # Light Yellow
                elif value <= 4:
                    return 'background-color: #6977DE'  # Light Blue
                else:
                    return 'background-color: #69DE85'  # Light Green

            # Style the dataframe
            styled_df = display_inns_df.style.applymap(color_wickets, subset=['Wickets'])

            # Display section header and dataframe
            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Last 20 Bowling Innings</h3>", unsafe_allow_html=True)
            st.dataframe(styled_df, height=575, use_container_width=True, hide_index=True)
            
            # Create summary dataframe with bowling stats at the bottom of the tab
            try:
                # Calculate the player summary from the last 20 innings data only, now including Format
                player_summary = filtered_df[filtered_df['File Name'].isin(last_20_match_files)].groupby(['Name', 'Match_Format']).agg({
                    'File Name': 'nunique',  # Matches
                    'Bowler_Balls': 'sum',
                    'Bowler_Runs': 'sum',
                    'Bowler_Wkts': 'sum',
                    'Maidens': 'sum'
                }).reset_index()
                
                if not player_summary.empty:
                    # Rename Match_Format to Format for consistency
                    player_summary.columns = ['Name', 'Format', 'Matches', 'Balls', 'Runs', 'Wickets', 'Maidens']
                    
                    # Calculate additional metrics with safety checks
                    player_summary['Overs'] = (player_summary['Balls'] // 6) + (player_summary['Balls'] % 6) / 10
                    # Add safe division for these metrics
                    player_summary['Average'] = (player_summary['Runs'] / player_summary['Wickets']).replace([np.inf, -np.inf], np.nan).round(2)
                    player_summary['Strike Rate'] = (player_summary['Balls'] / player_summary['Wickets']).replace([np.inf, -np.inf], np.nan).round(2)
                    player_summary['Economy'] = (player_summary['Runs'] / player_summary['Overs']).replace([np.inf, -np.inf], np.nan).round(2)
                    
                    # Count 5W innings per player and format
                    fiveW_counts = filtered_df[
                        (filtered_df['File Name'].isin(last_20_match_files)) & 
                        (filtered_df['Bowler_Wkts'] >= 5)
                    ].groupby(['Name', 'Match_Format']).size().reset_index(name='5W')
                    
                    player_summary = player_summary.merge(fiveW_counts, 
                                                         left_on=['Name', 'Format'], 
                                                         right_on=['Name', 'Match_Format'], 
                                                         how='left')
                    
                    # Drop duplicate Match_Format column if it exists
                    if 'Match_Format' in player_summary.columns:
                        player_summary = player_summary.drop(columns=['Match_Format'])
                        
                    player_summary['5W'] = player_summary['5W'].fillna(0).astype(int)
                    
                    # Replace NaN with empty or "N/A"
                    player_summary = player_summary.fillna("N/A")
                    
                    # Add an "All Formats" summary row for each player
                    all_formats_summary = filtered_df[filtered_df['File Name'].isin(last_20_match_files)].groupby(['Name']).agg({
                        'File Name': 'nunique',  # Matches
                        'Bowler_Balls': 'sum',
                        'Bowler_Runs': 'sum',
                        'Bowler_Wkts': 'sum',
                        'Maidens': 'sum'
                    }).reset_index()
                    
                    all_formats_summary['Format'] = 'All Formats'
                    all_formats_summary.columns = ['Name', 'Matches', 'Balls', 'Runs', 'Wickets', 'Maidens', 'Format']
                    
                    # Calculate metrics for All Formats
                    all_formats_summary['Overs'] = (all_formats_summary['Balls'] // 6) + (all_formats_summary['Balls'] % 6) / 10
                    all_formats_summary['Average'] = (all_formats_summary['Runs'] / all_formats_summary['Wickets']).replace([np.inf, -np.inf], np.nan).round(2)
                    all_formats_summary['Strike Rate'] = (all_formats_summary['Balls'] / all_formats_summary['Wickets']).replace([np.inf, -np.inf], np.nan).round(2)
                    all_formats_summary['Economy'] = (all_formats_summary['Runs'] / all_formats_summary['Overs']).replace([np.inf, -np.inf], np.nan).round(2)
                    
                    # Count 5W for All Formats
                    fiveW_all = filtered_df[
                        (filtered_df['File Name'].isin(last_20_match_files)) & 
                        (filtered_df['Bowler_Wkts'] >= 5)
                    ].groupby(['Name']).size().reset_index(name='5W')
                    
                    all_formats_summary = all_formats_summary.merge(fiveW_all, on=['Name'], how='left')
                    all_formats_summary['5W'] = all_formats_summary['5W'].fillna(0).astype(int)
                    
                    # Ensure column order matches player_summary
                    all_formats_summary = all_formats_summary[[
                        'Name', 'Format', 'Matches', 'Balls', 'Runs', 'Wickets', 'Maidens',
                        'Overs', 'Average', 'Strike Rate', 'Economy', '5W'
                    ]]
                    
                    # Combine format-specific and all formats summaries
                    player_summary = pd.concat([player_summary, all_formats_summary])
                    
                    # Sort by Name and Format (with All Formats appearing first)
                    player_summary['Format_Order'] = player_summary['Format'].map(lambda x: 0 if x == 'All Formats' else 1)
                    player_summary = player_summary.sort_values(['Name', 'Format_Order', 'Format']).drop(columns=['Format_Order'])
                    
                    # Replace NaN with empty or "N/A"
                    player_summary = player_summary.fillna("N/A")
                    
                    # Display the summary dataframe at the bottom of the tab
                    st.markdown("<h3 style='color:#f04f53; text-align: center;'>Player Bowling Summary by Format (Last 20 Innings)</h3>", unsafe_allow_html=True)
                    st.dataframe(player_summary[[
                        'Name', 'Format', 'Matches', 'Overs', 'Maidens', 'Runs', 'Wickets', 
                        'Average', 'Strike Rate', 'Economy', '5W'
                    ]], use_container_width=True, hide_index=True)
                else:
                    st.warning("No player summary data available for the selected criteria.")
            except Exception as e:
                st.error(f"Error generating player summary: {str(e)}")

        ###-------------------------------------OPPONENT STATS-------------------------------------###
        # Opponent Stats Tab
        with tabs[4]:
            # Calculate statistics dataframe for opponents
            opponent_summary = filtered_df.groupby(['Name', 'Bat_Team']).agg({
                'File Name': 'nunique',      # Matches
                'Bowler_Balls': 'sum',
                'Maidens': 'sum',
                'Bowler_Runs': 'sum',
                'Bowler_Wkts': 'sum'
            }).reset_index()

            opponent_summary.columns = ['Name', 'Opposition', 'Matches', 'Balls', 'M/D', 'Runs', 'Wickets']

            # Calculate metrics
            opponent_summary['Overs'] = (opponent_summary['Balls'] // 6) + (opponent_summary['Balls'] % 6) / 10
            opponent_summary['Strike Rate'] = (opponent_summary['Balls'] / opponent_summary['Wickets']).round(2)
            opponent_summary['Economy Rate'] = (opponent_summary['Runs'] / opponent_summary['Overs']).round(2)
            opponent_summary['WPM'] = (opponent_summary['Wickets'] / opponent_summary['Matches']).round(2)
            opponent_summary['Average'] = (opponent_summary['Runs'] / opponent_summary['Wickets']).round(2)

            # Count 5W innings
            five_wickets = filtered_df[filtered_df['Bowler_Wkts'] >= 5].groupby(['Name', 'Bat_Team']).size().reset_index(name='5W')
            opponent_summary = opponent_summary.merge(five_wickets, left_on=['Name', 'Opposition'], right_on=['Name', 'Bat_Team'], how='left')
            opponent_summary['5W'] = opponent_summary['5W'].fillna(0).astype(int)

            # Count 10W matches
            match_wickets = filtered_df.groupby(['Name', 'Bat_Team', 'File Name'])['Bowler_Wkts'].sum().reset_index()
            ten_wickets = match_wickets[match_wickets['Bowler_Wkts'] >= 10].groupby(['Name', 'Bat_Team']).size().reset_index(name='10W')
            opponent_summary = opponent_summary.merge(ten_wickets, left_on=['Name', 'Opposition'], right_on=['Name', 'Bat_Team'], how='left')
            opponent_summary['10W'] = opponent_summary['10W'].fillna(0).astype(int)

            # Handle infinities and NaNs
            opponent_summary = opponent_summary.replace([np.inf, -np.inf], np.nan)
            # Final column ordering
            opponent_summary = opponent_summary[[
                'Name', 'Opposition', 'Matches', 'Overs', 'M/D', 'Runs', 'Wickets', 'Average',
                'Strike Rate', 'Economy Rate', '5W', '10W', 'WPM'
            ]]

            # Sort by Wickets high to low
            opponent_summary = opponent_summary.sort_values(by='Wickets', ascending=False)

            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Opposition Statistics</h3>", unsafe_allow_html=True)
            
            # Display the dataframe first (full width)
            st.dataframe(opponent_summary, use_container_width=True, hide_index=True)

            # Create opponent averages graph
            fig = go.Figure()

            # Calculate and show 'All' stats if it's selected
            if 'All' in name_choice:
                all_opponent_stats = filtered_df.groupby(['Bat_Team']).agg({
                    'Bowler_Runs': 'sum',
                    'Bowler_Wkts': 'sum',
                    'Bowler_Balls': 'sum'
                }).reset_index()

                all_opponent_stats['Overs'] = (all_opponent_stats['Bowler_Balls'] // 6) + (all_opponent_stats['Bowler_Balls'] % 6)/10
                all_opponent_stats['Average'] = (all_opponent_stats['Bowler_Runs'] / all_opponent_stats['Bowler_Wkts']).round(2).fillna(0)
                all_opponent_stats['Strike_Rate'] = (all_opponent_stats['Bowler_Balls'] / all_opponent_stats['Bowler_Wkts']).round(2).fillna(0)
                all_opponent_stats['Economy_Rate'] = (all_opponent_stats['Bowler_Runs'] / all_opponent_stats['Overs']).round(2).fillna(0)

                all_color = '#f84e4e' if not individual_players else 'black'
                
                fig.add_trace(
                    go.Bar(
                        x=all_opponent_stats['Bat_Team'],
                        y=all_opponent_stats['Average'],
                        name='All Players',
                        marker_color=all_color,
                        text=all_opponent_stats['Average'],
                        textposition='auto',
                        customdata=np.stack([
                            all_opponent_stats['Bowler_Wkts'],
                            all_opponent_stats['Strike_Rate'],
                            all_opponent_stats['Economy_Rate']
                        ], axis=-1),
                        hovertemplate=(
                            'Team: %{x}<br>'
                            'Average: %{y}<br>'
                            'Wickets: %{customdata[0]}<br>'
                            'Strike Rate: %{customdata[1]}<br>'
                            'Economy: %{customdata[2]}<extra></extra>'
                        )
                    )
                )

            # Add individual player traces with tooltips
            for i, name in enumerate(individual_players):
                player_data = filtered_df[filtered_df['Name'] == name].groupby('Bat_Team').agg({
                    'Bowler_Runs': 'sum',
                    'Bowler_Wkts': 'sum',
                    'Bowler_Balls': 'sum'
                }).reset_index()

                player_data['Overs'] = (player_data['Bowler_Balls'] // 6) + (player_data['Bowler_Balls'] % 6)/10
                player_data['Average'] = (player_data['Bowler_Runs'] / player_data['Bowler_Wkts']).round(2).fillna(0)
                player_data['Strike_Rate'] = (player_data['Bowler_Balls'] / player_data['Bowler_Wkts']).round(2).fillna(0)
                player_data['Economy_Rate'] = (player_data['Bowler_Runs'] / player_data['Overs']).round(2).fillna(0)

                color = '#f84e4e' if i == 0 else f'#{random.randint(0, 0xFFFFFF):06x}'

                fig.add_trace(
                    go.Bar(
                        x=player_data['Bat_Team'],
                        y=player_data['Average'],
                        name=name,
                        marker_color=color,
                        text=player_data['Bowler_Wkts'],
                        textposition='auto',
                        customdata=np.stack([
                            player_data['Bowler_Wkts'],
                            player_data['Strike_Rate'],
                            player_data['Economy_Rate']
                        ], axis=-1),
                        hovertemplate=(
                            'Team: %{x}<br>'
                            'Average: %{y}<br>'
                            'Wickets: %{customdata[0]}<br>'
                            'Strike Rate: %{customdata[1]}<br>'
                            'Economy: %{customdata[2]}<extra></extra>'
                        )
                    )
                )

            # Display chart title
            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Average vs Opponent Team</h3>", unsafe_allow_html=True)

            # Update layout
            fig.update_layout(
                showlegend=True,
                height=500,
                xaxis_title="Opposition Team",
                yaxis_title="Average",
                font=dict(size=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis={'categoryorder': 'total ascending'},
                barmode='group'
            )

            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

            # Display the bar chart (full width)
            st.plotly_chart(fig, use_container_width=True)

        ###-------------------------------------LOCATION STATS-------------------------------------###
        # Location Stats Tab
        with tabs[5]:
            # Calculate statistics dataframe for locations
            location_summary = filtered_df.groupby(['Name', 'Home_Team']).agg({
                'File Name': 'nunique',      # Matches
                'Bowler_Balls': 'sum',
                'Maidens': 'sum',
                'Bowler_Runs': 'sum',
                'Bowler_Wkts': 'sum'
            }).reset_index()

            location_summary.columns = ['Name', 'Location', 'Matches', 'Balls', 'M/D', 'Runs', 'Wickets']

            # Calculate metrics
            location_summary['Overs'] = (location_summary['Balls'] // 6) + (location_summary['Balls'] % 6) / 10
            location_summary['Strike Rate'] = (location_summary['Balls'] / location_summary['Wickets']).round(2)
            location_summary['Economy Rate'] = (location_summary['Runs'] / location_summary['Overs']).round(2)
            location_summary['WPM'] = (location_summary['Wickets'] / location_summary['Matches']).round(2)
            location_summary['Average'] = (location_summary['Runs'] / location_summary['Wickets']).round(2)  # Add average here

            # Count 5W innings
            five_wickets = filtered_df[filtered_df['Bowler_Wkts'] >= 5].groupby(['Name', 'Home_Team']).size().reset_index(name='5W')
            location_summary = location_summary.merge(five_wickets, left_on=['Name', 'Location'], right_on=['Name', 'Home_Team'], how='left')
            location_summary['5W'] = location_summary['5W'].fillna(0).astype(int)

            # Count 10W matches
            match_wickets = filtered_df.groupby(['Name', 'Home_Team', 'File Name'])['Bowler_Wkts'].sum().reset_index()
            ten_wickets = match_wickets[match_wickets['Bowler_Wkts'] >= 10].groupby(['Name', 'Home_Team']).size().reset_index(name='10W')
            location_summary = location_summary.merge(ten_wickets, left_on=['Name', 'Location'], right_on=['Name', 'Home_Team'], how='left')
            location_summary['10W'] = location_summary['10W'].fillna(0).astype(int)

            # Handle infinities and NaNs
            location_summary = location_summary.replace([np.inf, -np.inf], np.nan)

            # Final column ordering
            location_summary = location_summary[[
                'Name', 'Location', 'Matches', 'Balls', 'Overs', 'M/D', 'Runs', 'Wickets', 'Average',
                'Strike Rate', 'Economy Rate', '5W', '10W', 'WPM'
            ]]
            location_summary = location_summary.sort_values('Wickets', ascending=False)

            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Location Statistics</h3>", unsafe_allow_html=True)
            
            # Display the dataframe first (full width)
            st.dataframe(location_summary, use_container_width=True, hide_index=True)

            # Create location averages graph
            fig = go.Figure()

            # Add 'All' trace first if selected
            if 'All' in name_choice:
                all_location_stats = filtered_df.groupby(['Home_Team']).agg({
                    'Bowler_Runs': 'sum',
                    'Bowler_Wkts': 'sum'
                }).reset_index()
                
                all_location_stats['Average'] = (all_location_stats['Bowler_Runs'] / all_location_stats['Bowler_Wkts']).round(2)
                
                all_color = '#f84e4e' if not individual_players else 'black'
                
                fig.add_trace(
                    go.Bar(
                        x=all_location_stats['Home_Team'],
                        y=all_location_stats['Average'],
                        name='All Players',
                        marker_color=all_color,
                        text=all_location_stats['Bowler_Wkts'],
                        textposition='auto'
                    )
                )

            # Add individual player traces
            for i, name in enumerate(individual_players):
                player_data = filtered_df[filtered_df['Name'] == name].groupby('Home_Team').agg({
                    'Bowler_Runs': 'sum',
                    'Bowler_Wkts': 'sum'
                }).reset_index()
                
                player_data['Average'] = (player_data['Bowler_Runs'] / player_data['Bowler_Wkts']).round(2)
                
                color = '#f84e4e' if i == 0 else f'#{random.randint(0, 0xFFFFFF):06x}'
                
                fig.add_trace(
                    go.Bar(
                        x=player_data['Home_Team'],
                        y=player_data['Average'],
                        name=name,
                        marker_color=color,
                        text=player_data['Bowler_Wkts'],
                        textposition='auto'
                    )
                )

            # Display chart title
            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Average by Location</h3>", unsafe_allow_html=True)

            # Update layout
            fig.update_layout(
                showlegend=True,
                height=500,
                xaxis_title="Location",
                yaxis_title="Average",
                font=dict(size=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis={'categoryorder':'total ascending'},
                barmode='group'
            )

            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

            # Display the bar chart (full width)
            st.plotly_chart(fig, use_container_width=True)

        ###-------------------------------------INNINGS STATS-------------------------------------###
        # Innings Stats Tab
        with tabs[6]:
            # Calculate statistics dataframe for innings
            innings_summary = filtered_df.groupby(['Name', 'Innings']).agg({
                'File Name': 'nunique',      # Matches
                'Bowler_Balls': 'sum',
                'Maidens': 'sum',
                'Bowler_Runs': 'sum',
                'Bowler_Wkts': 'sum'
            }).reset_index()

            innings_summary.columns = ['Name', 'Innings', 'Matches', 'Balls', 'M/D', 'Runs', 'Wickets']

            # Calculate metrics
            innings_summary['Overs'] = (innings_summary['Balls'] // 6) + (innings_summary['Balls'] % 6) / 10
            innings_summary['Average'] = (innings_summary['Runs'] / innings_summary['Wickets']).round(2)  # Add bowling average
            innings_summary['Strike Rate'] = (innings_summary['Balls'] / innings_summary['Wickets']).round(2)
            innings_summary['Economy Rate'] = (innings_summary['Runs'] / innings_summary['Overs']).round(2)
            innings_summary['WPM'] = (innings_summary['Wickets'] / innings_summary['Matches']).round(2)

            # Count 5W innings
            five_wickets = filtered_df[filtered_df['Bowler_Wkts'] >= 5].groupby(['Name', 'Innings']). size().reset_index(name='5W')
            innings_summary = innings_summary.merge(five_wickets, on=['Name', 'Innings'], how='left')
            innings_summary['5W'] = innings_summary['5W'].fillna(0).astype(int)

            # Count 10W matches
            match_wickets = filtered_df.groupby(['Name', 'Innings', 'File Name'])['Bowler_Wkts'].sum().reset_index()
            ten_wickets = match_wickets[match_wickets['Bowler_Wkts'] >= 10].groupby(['Name', 'Innings']).size().reset_index(name='10W')
            innings_summary = innings_summary.merge(ten_wickets, on=['Name', 'Innings'], how='left')
            innings_summary['10W'] = innings_summary['10W'].fillna(0).astype(int)

            # Handle infinities and NaNs
            innings_summary = innings_summary.replace([np.inf, -np.inf], np.nan)

            # Final column ordering
            innings_summary = innings_summary[[
                'Name', 'Innings', 'Matches', 'Balls', 'Overs', 'M/D', 'Runs', 'Wickets', 'Average',
                'Strike Rate', 'Economy Rate', '5W', '10W', 'WPM'
            ]]

            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Innings Statistics</h3>", unsafe_allow_html=True)
            
            # Display the dataframe first (full width)
            st.dataframe(innings_summary, use_container_width=True, hide_index=True)

            # Create innings average graph
            fig = go.Figure()
            
            # Add 'All' trace first if selected
            if 'All' in name_choice:
                all_innings_stats = filtered_df.groupby('Innings').agg({
                    'Bowler_Runs': 'sum',
                    'Bowler_Wkts': 'sum',
                    'Bowler_Balls': 'sum'
                }).reset_index()
                
                # Calculate metrics
                all_innings_stats['Average'] = (all_innings_stats['Bowler_Runs'] / all_innings_stats['Bowler_Wkts']).round(2)
                all_innings_stats['Strike_Rate'] = (all_innings_stats['Bowler_Balls'] / all_innings_stats['Bowler_Wkts']).round(2)
                all_innings_stats['Economy_Rate'] = (all_innings_stats['Bowler_Runs'] / (all_innings_stats['Bowler_Balls']/6)).round(2)
                
                all_color = '#f84e4e' if not individual_players else 'black'
                
                fig.add_trace(
                    go.Bar(
                        y=all_innings_stats['Innings'],
                        x=all_innings_stats['Average'],
                        name='All Players',
                        marker_color=all_color,
                        text=all_innings_stats['Average'].round(2),
                        textposition='auto',
                        orientation='h',
                        customdata=np.stack((
                            all_innings_stats['Bowler_Wkts'],
                            all_innings_stats['Strike_Rate'],
                            all_innings_stats['Economy_Rate']
                        ), axis=-1),
                        hovertemplate=(
                            'Innings: %{y}<br>'
                            'Average: %{x:.2f}<br>'
                            'Wickets: %{customdata[0]}<br>'
                            'Strike Rate: %{customdata[1]:.2f}<br>'
                            'Economy Rate: %{customdata[2]:.2f}<extra></extra>'
                        )
                    )
                )

            # Add individual player traces
            for i, name in enumerate(individual_players):
                player_data = filtered_df[filtered_df['Name'] == name].groupby('Innings').agg({
                    'Bowler_Runs': 'sum',
                    'Bowler_Wkts': 'sum'
                }).reset_index()
                
                player_data['Average'] = (player_data['Bowler_Runs'] / player_data['Bowler_Wkts']).round(2)
                
                color = '#f84e4e' if i == 0 else f'#{random.randint(0, 0xFFFFFF):06x}'
                fig.add_trace(
                    go.Bar(
                        y=player_data['Innings'],  # Switched to y axis
                        x=player_data['Average'],  # Switched to x axis
                        name=name,
                        marker_color=color,
                        text=player_data['Bowler_Wkts'],
                        textposition='auto',
                        orientation='h'  # Make bars horizontal
                    )
                )

            # Display chart title
            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Average by Innings Number</h3>", unsafe_allow_html=True)

            # Update layout
            fig.update_layout(
                showlegend=True,
                height=500,
                yaxis_title="Innings",  # Switched axis titles
                xaxis_title="Average",
                font=dict(size=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis={
                    'type': 'category'
                },
                barmode='group'
            )

            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

            # Display the bar chart (full width)
            st.plotly_chart(fig, use_container_width=True)

        ###-------------------------------------POSITION STATS-------------------------------------###
        # Position Stats Tab
        with tabs[7]:
            # Calculate statistics dataframe for position
            position_summary = filtered_df.groupby(['Name', 'Position']).agg({
                'File Name': 'nunique',      # Matches
                'Bowler_Balls': 'sum',
                'Maidens': 'sum',
                'Bowler_Runs': 'sum',
                'Bowler_Wkts': 'sum'
            }).reset_index()

            position_summary.columns = ['Name', 'Position', 'Matches', 'Balls', 'M/D', 'Runs', 'Wickets']

            # Calculate metrics
            position_summary['Overs'] = (position_summary['Balls'] // 6) + (position_summary['Balls'] % 6) / 10
            position_summary['Average'] = (position_summary['Runs'] / position_summary['Wickets']).round(2)  # Added Average
            position_summary['Strike Rate'] = (position_summary['Balls'] / position_summary['Wickets']).round(2)
            position_summary['Economy Rate'] = (position_summary['Runs'] / position_summary['Overs']).round(2)
            position_summary['WPM'] = (position_summary['Wickets'] / position_summary['Matches']).round(2)

            # Count 5W innings
            five_wickets = filtered_df[filtered_df['Bowler_Wkts'] >= 5].groupby(['Name', 'Position']).size().reset_index(name='5W')
            position_summary = position_summary.merge(five_wickets, on=['Name', 'Position'], how='left')
            position_summary['5W'] = position_summary['5W'].fillna(0).astype(int)

            # Count 10W matches
            match_wickets = filtered_df.groupby(['Name', 'Position', 'File Name'])['Bowler_Wkts'].sum().reset_index()
            ten_wickets = match_wickets[match_wickets['Bowler_Wkts'] >= 10].groupby(['Name', 'Position']).size().reset_index(name='10W')
            position_summary = position_summary.merge(ten_wickets, on=['Name', 'Position'], how='left')
            position_summary['10W'] = position_summary['10W'].fillna(0).astype(int)

            # Handle infinities and NaNs
            position_summary = position_summary.replace([np.inf, -np.inf], np.nan)

            # Final column ordering
            position_summary = position_summary[[
                'Name', 'Position', 'Matches', 'Balls', 'Overs', 'M/D', 'Runs', 'Wickets', 'Average',
                'Strike Rate', 'Economy Rate', '5W', '10W', 'WPM'
            ]]

            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Position Statistics</h3>", unsafe_allow_html=True)
            
            # Display the dataframe first (full width)
            st.dataframe(position_summary, use_container_width=True, hide_index=True)

            # Create position averages graph
            fig = go.Figure()

            # Add 'All' trace first if selected
            if 'All' in name_choice:
                all_position_stats = filtered_df.groupby(['Position']).agg({
                    'Bowler_Runs': 'sum',
                    'Bowler_Wkts': 'sum',
                    'Bowler_Balls': 'sum'
                }).reset_index()
                
                all_position_stats['Average'] = (all_position_stats['Bowler_Runs'] / all_position_stats['Bowler_Wkts']).round(2)
                all_position_stats['Strike_Rate'] = (all_position_stats['Bowler_Balls'] / all_position_stats['Bowler_Wkts']).round(2)
                all_position_stats['Economy_Rate'] = (all_position_stats['Bowler_Runs'] / (all_position_stats['Bowler_Balls']/6)).round(2)
                all_position_stats['Position'] = all_position_stats['Position'].astype(int)
                
                all_color = '#f84e4e' if not individual_players else 'black'
                
                fig.add_trace(
                    go.Bar(
                        y=all_position_stats['Position'],  # Switch to Y axis
                        x=all_position_stats['Average'],   # Switch to X axis
                        name='All Players',
                        marker_color=all_color,
                        text=all_position_stats['Average'].round(2),
                        textposition='auto',
                        orientation='h',  # Make bars horizontal
                        customdata=np.stack((
                            all_position_stats['Bowler_Wkts'],
                            all_position_stats['Strike_Rate'],
                            all_position_stats['Economy_Rate']
                        ), axis=-1),
                        hovertemplate=(
                            'Position: %{y}<br>'
                            'Average: %{x:.2f}<br>'
                            'Wickets: %{customdata[0]}<br>'
                            'Strike Rate: %{customdata[1]:.2f}<br>'
                            'Economy Rate: %{customdata[2]:.2f}<extra></extra>'
                        )
                    )
                )

            # Add individual player traces
            for i, name in enumerate(individual_players):
                player_data = filtered_df[filtered_df['Name'] == name].groupby('Position').agg({
                    'Bowler_Runs': 'sum',
                    'Bowler_Wkts': 'sum',
                    'Bowler_Balls': 'sum'
                }).reset_index()
                
                player_data['Average'] = (player_data['Bowler_Runs'] / player_data['Bowler_Wkts']).round(2)
                player_data['Strike_Rate'] = (player_data['Bowler_Balls'] / player_data['Bowler_Wkts']).round(2)
                player_data['Economy_Rate'] = (player_data['Bowler_Runs'] / (player_data['Bowler_Balls']/6)).round(2)
                player_data['Position'] = player_data['Position'].astype(int)
                
                color = '#f84e4e' if i == 0 else f'#{random.randint(0, 0xFFFFFF):06x}'
                
                fig.add_trace(
                    go.Bar(
                        y=player_data['Position'],  # Switch to Y axis
                        x=player_data['Average'],   # Switch to X axis
                        name=name,
                        marker_color=color,
                        text=player_data['Average'].round(2),
                        textposition='auto',
                        orientation='h',  # Make bars horizontal
                        customdata=np.stack((
                            player_data['Bowler_Wkts'],
                            player_data['Strike_Rate'],
                            player_data['Economy_Rate']
                        ), axis=-1),
                        hovertemplate=(
                            'Position: %{y}<br>'
                            'Average: %{x:.2f}<br>'
                            'Wickets: %{customdata[0]}<br>'
                            'Strike Rate: %{customdata[1]:.2f}<br>'
                            'Economy Rate: %{customdata[2]:.2f}<extra></extra>'
                        )
                    )
                )

            # Display chart title
            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Average by Bowling Position</h3>", unsafe_allow_html=True)

            # Update layout
            fig.update_layout(
                showlegend=True,
                height=500,
                xaxis_title="Average",   # Switch axis titles
                yaxis_title="Position",
                font=dict(size=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis={
                    'categoryorder':'array',
                    'categoryarray': list(range(1,12)),  # Force positions 1-11
                    'dtick': 1  # Show all integer positions
                },
                barmode='group'
            )

            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

            # Display the bar chart (full width)
            st.plotly_chart(fig, use_container_width=True)

        ###--------------------------------------CUMULATIVE BOWLING STATS------------------------------------------#######
        # Cumulative Stats Tab
        with tabs[8]:
            # Create initial df_bowl from filtered bowl_df
            df_bowl = filtered_df.copy()

            # Convert the 'Date' column to datetime format for proper chronological sorting
            df_bowl['Date'] = pd.to_datetime(df_bowl['Date'], format='%d %b %Y').dt.date

            # Sort the DataFrame by 'Name', 'Match_Format', and the 'Date' column
            df_bowl = df_bowl.sort_values(by=['Name', 'Match_Format', 'Date'])

            # Only process if there is data for the selected player
            if not df_bowl.empty:
                # Sort by Name and Date to ensure chronological order
                df_bowl = df_bowl.sort_values(by=['Name', 'Match_Format', 'Date'])

                # Create innings number starting from 1 for each player and format
                df_bowl['Innings_Number'] = df_bowl.groupby(['Name', 'Match_Format']).cumcount() + 1

                # Calculate cumulative sums
                df_bowl['Cumulative Balls'] = df_bowl.groupby(['Name', 'Match_Format'])['Bowler_Balls'].cumsum()
                df_bowl['Cumulative Runs'] = df_bowl.groupby(['Name', 'Match_Format'])['Bowler_Runs'].cumsum()
                df_bowl['Cumulative Wickets'] = df_bowl.groupby(['Name', 'Match_Format'])['Bowler_Wkts'].cumsum()

                # Calculate cumulative overs
                df_bowl['Cumulative Overs'] = (df_bowl['Cumulative Balls'] // 6) + (df_bowl['Cumulative Balls'] % 6) / 10

                # Calculate cumulative metrics
                df_bowl['Cumulative SR'] = (df_bowl['Cumulative Balls'] / df_bowl['Cumulative Wickets'].replace(0, np.nan)).round(2)
                df_bowl['Cumulative Econ'] = (df_bowl['Cumulative Runs'] / df_bowl['Cumulative Overs'].replace(0, np.nan)).round(2)
                df_bowl['Cumulative Average'] = (df_bowl['Cumulative Runs'] / df_bowl['Cumulative Wickets'].replace(0, np.nan)).round(2)

                # Convert Bowler_Balls to Overs format
                df_bowl['Overs'] = (df_bowl['Bowler_Balls'] // 6) + (df_bowl['Bowler_Balls'] % 6) / 10

                # Create new columns with desired names
                df_bowl['Runs'] = df_bowl['Bowler_Runs']
                df_bowl['Wkts'] = df_bowl['Bowler_Wkts']
                df_bowl['Econ'] = df_bowl['Bowler_Econ']

                # Drop unwanted columns
                columns_to_drop = [
                    'Bat_Team', 'Bowl_Team', 'Total_Runs', 'Wickets', 
                    'Competition', 'Player_of_the_Match', 'Bowled', '5ws', '10ws',
                    'Runs_Per_Over', 'Balls_Per_Wicket', 'File Name',
                    'Dot_Ball_Percentage', 'Strike_Rate', 'Average', 'Year',
                    'Cumulative Balls', 'Bowler_Balls', 'Bowler_Runs', 'Bowler_Wkts',
                    'Bowler_Econ', 'Bowler_Overs'
                ]
                
                # Drop columns if they exist
                df_bowl = df_bowl.drop(columns=[col for col in columns_to_drop if col in df_bowl.columns])

                # Reorder columns
                column_order = [
                    'Date', 'Home_Team', 'Away_Team', 'Name', 'Innings', 'Position',
                    'Overs', 'Runs', 'Wkts', 'Econ', 'Innings_Number',
                    'Cumulative Overs', 'Cumulative Runs', 'Cumulative Wickets',
                    'Cumulative Average', 'Cumulative SR', 'Cumulative Econ'
                ]
                
                     # Only include columns that exist in the DataFrame
                final_columns = [col for col in column_order if col in df_bowl.columns]
                df_bowl = df_bowl[final_columns]
             # Handle infinities and NaNs
                df_bowl = df_bowl.replace([np.inf, -np.inf], np.nan)

                # Display the bowling statistics
                st.markdown("<h3 style='color:#f04f53; text-align: center;'>Cumulative Bowling Statistics</h3>", unsafe_allow_html=True)
                st.dataframe(df_bowl, use_container_width=True, hide_index=True)
            else:
                st.warning("No bowling data available for the selected player.")

            ###----------------------GRAPHS--------------------------###
            # Create subplots for Cumulative Average, Strike Rate, and Economy
            fig = make_subplots(rows=1, cols=3, subplot_titles=("Cumulative Average", "Cumulative Strike Rate", "Cumulative Economy"))

            # Get list of individual players (excluding 'All')
            individual_players = [name for name in name_choice if name != 'All']

            # Add individual player traces
            for i, name in enumerate(individual_players):
                player_stats = df_bowl[df_bowl['Name'] == name].sort_values('Innings_Number')
                
                # First player gets streamlit red, others get random colors
                color = '#f84e4e' if i == 0 else f'#{random.randint(0, 0xFFFFFF):06x}'

                # Add Cumulative Average trace
                fig.add_trace(go.Scatter(
                    x=player_stats['Innings_Number'],
                    y=player_stats['Cumulative Average'],
                    name=name,
                    legendgroup=name,
                    mode='lines+markers',
                    marker_color=color,
                    showlegend=True
                ), row=1, col=1)

                # Add Cumulative Strike Rate trace
                fig.add_trace(go.Scatter(
                    x=player_stats['Innings_Number'],
                    y=player_stats['Cumulative SR'],
                    name=name,
                    legendgroup=name,
                    mode='lines+markers',
                    marker_color=color,
                    showlegend=False
                ), row=1, col=2)

                # Add Cumulative Economy trace
                fig.add_trace(go.Scatter(
                    x=player_stats['Innings_Number'],
                    y=player_stats['Cumulative Econ'],
                    name=name,
                    legendgroup=name,
                    mode='lines+markers',
                    marker_color=color,
                    showlegend=False
                ), row=1, col=3)

            # Update layout
            fig.update_layout(
                showlegend=True,
                height=500,
                font=dict(size=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis_title="Average (Runs/Wicket)",
                yaxis2_title="Strike Rate (Balls/Wicket)",
                yaxis3_title="Economy Rate (Runs/Over)",
                xaxis_title="Innings Number",
                xaxis2_title="Innings Number",
                xaxis3_title="Innings Number"
            )

            # Add gridlines and update axes
            for i in range(1, 4):
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)', row=1, col=i)
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)', row=1, col=i)

            # Display the figure
            st.plotly_chart(fig, use_container_width=True)

        ###--------------------------------------BOWLING BLOCK STATS------------------------------------------#######
        # Block Stats Tab
        with tabs[9]:
            # Create DataFrame for block stats from filtered_df (notice this is at the same level as other main sections)
            df_blockbowl = filtered_df.copy()

            # Only process if there is data for the selected player
            if not df_blockbowl.empty:
                # Sort by Name and Date to ensure chronological order
                df_blockbowl = df_blockbowl.sort_values(by=['Name', 'Match_Format', 'Date'])

                # Create innings number and innings range
                df_blockbowl['Innings_Number'] = df_blockbowl.groupby(['Name', 'Match_Format']).cumcount() + 1
                df_blockbowl['Innings_Range'] = (((df_blockbowl['Innings_Number'] - 1) // 20) * 20).astype(str) + '-' + \
                                        ((((df_blockbowl['Innings_Number'] - 1) // 20) * 20 + 19)).astype(str)
                df_blockbowl['Range_Start'] = ((df_blockbowl['Innings_Number'] - 1) // 20) * 20

                # Group by blocks and calculate statistics
                block_stats_df = df_blockbowl.groupby(['Name', 'Match_Format', 'Innings_Range', 'Range_Start']).agg({
                    'Innings_Number': 'count',
                    'Bowler_Balls': 'sum',
                    'Bowler_Runs': 'sum',
                    'Bowler_Wkts': 'sum',
                    'Date': ['first', 'last']
                }).reset_index()

                # Flatten the column names
                block_stats_df.columns = ['Name', 'Match_Format', 'Innings_Range', 'Range_Start',
                                        'Innings', 'Balls', 'Runs', 'Wickets',
                                        'First_Date', 'Last_Date']

                # Calculate statistics for each block
                block_stats_df['Overs'] = (block_stats_df['Balls'] // 6) + (block_stats_df['Balls'] % 6) / 10
                block_stats_df['Average'] = (block_stats_df['Runs'] / block_stats_df['Wickets']).round(2)
                block_stats_df['Strike_Rate'] = (block_stats_df['Balls'] / block_stats_df['Wickets']).round(2)
                block_stats_df['Economy'] = (block_stats_df['Runs'] / block_stats_df['Overs']).round(2)

                # Format dates properly before creating date range
                block_stats_df['First_Date'] = pd.to_datetime(block_stats_df['First_Date']).dt.strftime('%d/%m/%Y')
                block_stats_df['Last_Date'] = pd.to_datetime(block_stats_df['Last_Date']).dt.strftime('%d/%m/%Y')
                
                # Create date range column
                block_stats_df['Date_Range'] = block_stats_df['First_Date'] + ' to ' + block_stats_df['Last_Date']

                # Sort the DataFrame
                block_stats_df = block_stats_df.sort_values(['Name', 'Match_Format', 'Range_Start'])

                # Select and order final columns
                final_columns = [
                    'Name', 'Match_Format', 'Innings_Range', 'Date_Range',
                    'Innings', 'Overs', 'Runs', 'Wickets',
                    'Average', 'Strike_Rate', 'Economy'
                ]
                block_stats_df = block_stats_df[final_columns]

                # Handle any infinities and NaN values
                block_stats_df = block_stats_df.replace([np.inf, -np.inf], np.nan)

                # Store the final DataFrame
                df_blocks = block_stats_df.copy()

                # Display the block statistics
                st.markdown("<h3 style='color:#f04f53; text-align: center;'>Block Statistics (Groups of 20 Innings)</h3>", unsafe_allow_html=True)
                st.dataframe(df_blocks, use_container_width=True, hide_index=True)

                # Create the figure for bowling averages by innings range
                fig = go.Figure()

                # Handle 'All' selection
                if 'All' in name_choice:
                    all_blocks = df_blocks.groupby('Innings_Range').agg({
                        'Runs': 'sum',
                        'Wickets': 'sum'
                    }).reset_index()
                    
                    all_blocks['Average'] = (all_blocks['Runs'] / all_blocks['Wickets']).round(2)
                    
                    all_blocks = all_blocks.sort_values('Innings_Range', 
                        key=lambda x: [int(i.split('-')[0]) for i in x])
                    
                    all_color = '#f84e4e' if not individual_players else 'black'
                    
                    fig.add_trace(
                        go.Bar(
                            x=all_blocks['Innings_Range'],
                            y=all_blocks['Average'],
                            name='All Players',
                            marker_color=all_color
                        )
                    )

                # Add individual player traces
                for i, name in enumerate(individual_players):
                    player_blocks = df_blocks[df_blocks['Name'] == name].sort_values('Innings_Range', 
                        key=lambda x: [int(i.split('-')[0]) for i in x])
                    
                    color = '#f84e4e' if i == 0 else f'#{random.randint(0, 0xFFFFFF):06x}'
                    
                    fig.add_trace(
                        go.Bar(
                            x=player_blocks['Innings_Range'],
                            y=player_blocks['Average'],
                            name=name,
                            marker_color=color
                        )
                    )

                # Update layout
                fig.update_layout(
                    showlegend=True,
                    height=500,
                    xaxis_title='Innings Range',
                    yaxis_title='Bowling Average',
                    margin=dict(l=50, r=50, t=70, b=50),
                    font=dict(size=12),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    barmode='group',
                    xaxis={'categoryorder': 'array', 
                        'categoryarray': sorted(df_blocks['Innings_Range'].unique(), 
                                            key=lambda x: int(x.split('-')[0]))}
                )

                # Add gridlines
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

                # Display title and graph
                st.markdown("<h3 style='color:#f04f53; text-align: center;'>Bowling Average by Innings Block</h3>", unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)

        ###-------------------------------------HOME/AWAY STATS-------------------------------------###
        # Home/Away Stats Tab
        with tabs[10]: # New tab index
            # Calculate home/away statistics
            match_wickets_ha = filtered_df.groupby(['Name', 'HomeOrAway', 'File Name'])['Bowler_Wkts'].sum().reset_index()
            # ten_wickets_ha calculation moved down

            bowlhomeaway_df = filtered_df.groupby(['Name', 'HomeOrAway']).agg({
                'File Name': 'nunique',
                'Bowler_Balls': 'sum',
                'Maidens': 'sum',
                'Bowler_Runs': 'sum',
                'Bowler_Wkts': 'sum'
            }).reset_index()

            # Check if bowlhomeaway_df is empty or HomeOrAway column is missing after initial group by
            if bowlhomeaway_df.empty or 'HomeOrAway' not in bowlhomeaway_df.columns:
                st.warning("No Home/Away data available for the selected filters.")
                # Create an empty structure or skip the rest of the tab
                bowlhomeaway_df = pd.DataFrame(columns=[ # Define expected columns
                    'Name', 'Home/Away', 'Matches', 'Balls', 'Overs', 'M/D', 'Runs', 'Wickets', 'Avg',
                    'Strike Rate', 'Economy Rate', '5W', '10W', 'WPM', 'POM'
                ])
                # Skip further processing for this tab if no base data
                st.dataframe(bowlhomeaway_df, use_container_width=True, hide_index=True)

            else: # Proceed if bowlhomeaway_df is valid

                bowlhomeaway_df.columns = ['Name', 'Home/Away', 'Matches', 'Balls', 'M/D', 'Runs', 'Wickets']

                # Calculate home/away metrics
                bowlhomeaway_df['Overs'] = (bowlhomeaway_df['Balls'] // 6) + (bowlhomeaway_df['Balls'] % 6) / 10
                bowlhomeaway_df['Avg'] = (bowlhomeaway_df['Runs'] / bowlhomeaway_df['Wickets']).replace([np.inf, -np.inf], np.nan).round(2)
                bowlhomeaway_df['Strike Rate'] = (bowlhomeaway_df['Balls'] / bowlhomeaway_df['Wickets']).replace([np.inf, -np.inf], np.nan).round(2)
                bowlhomeaway_df['Economy Rate'] = (bowlhomeaway_df['Runs'] / bowlhomeaway_df['Overs'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).round(2)


                # Add additional statistics safely
                five_wickets_ha = filtered_df[filtered_df['Bowler_Wkts'] >= 5].groupby(['Name', 'HomeOrAway']).size().reset_index(name='5W')
                # Check if 'HomeOrAway' exists in both dataframes before merging
                if 'HomeOrAway' in bowlhomeaway_df.columns and 'HomeOrAway' in five_wickets_ha.columns:
                    bowlhomeaway_df = bowlhomeaway_df.merge(five_wickets_ha, on=['Name', 'HomeOrAway'], how='left')
                elif 'HomeOrAway' in bowlhomeaway_df.columns: # Exists in left, but not right (no 5W found)
                    bowlhomeaway_df['5W'] = 0
                else: # HomeOrAway missing from the main aggregated df, add 5W column anyway
                    bowlhomeaway_df['5W'] = 0
                bowlhomeaway_df['5W'] = bowlhomeaway_df['5W'].fillna(0).astype(int)


                # Safely merge 10W stats
                ten_wickets_ha = match_wickets_ha[match_wickets_ha['Bowler_Wkts'] >= 10].groupby(['Name', 'HomeOrAway']).size().reset_index(name='10W')
                # Check if 'HomeOrAway' exists in both dataframes before merging
                if 'HomeOrAway' in bowlhomeaway_df.columns and 'HomeOrAway' in ten_wickets_ha.columns:
                    bowlhomeaway_df = bowlhomeaway_df.merge(ten_wickets_ha, on=['Name', 'HomeOrAway'], how='left')
                elif 'HomeOrAway' in bowlhomeaway_df.columns: # Exists in left, but not right (no 10W found)
                    bowlhomeaway_df['10W'] = 0
                else: # HomeOrAway missing from the main aggregated df, add 10W column anyway
                    bowlhomeaway_df['10W'] = 0
                bowlhomeaway_df['10W'] = bowlhomeaway_df['10W'].fillna(0).astype(int)


                bowlhomeaway_df['WPM'] = (bowlhomeaway_df['Wickets'] / bowlhomeaway_df['Matches']).replace([np.inf, -np.inf], np.nan).round(2)

                pom_counts_ha = filtered_df[filtered_df['Player_of_the_Match'] == filtered_df['Name']].groupby(['Name', 'HomeOrAway'])['File Name'].nunique().reset_index(name='POM')
                # Safely merge POM stats
                if 'HomeOrAway' in bowlhomeaway_df.columns and 'HomeOrAway' in pom_counts_ha.columns:
                    bowlhomeaway_df = bowlhomeaway_df.merge(pom_counts_ha, on=['Name', 'HomeOrAway'], how='left')
                elif 'HomeOrAway' in bowlhomeaway_df.columns: # Exists in left, but not right (no POM found)
                    bowlhomeaway_df['POM'] = 0
                else: # HomeOrAway missing from the main aggregated df, add POM column anyway
                    bowlhomeaway_df['POM'] = 0
                bowlhomeaway_df['POM'] = bowlhomeaway_df['POM'].fillna(0).astype(int)


                bowlhomeaway_df = bowlhomeaway_df.replace([np.inf, -np.inf], np.nan)
                bowlhomeaway_df = bowlhomeaway_df.sort_values(['Name', 'Home/Away'])

                # Final column ordering
                bowlhomeaway_df = bowlhomeaway_df[[
                    'Name', 'Home/Away', 'Matches', 'Balls', 'Overs', 'M/D', 'Runs', 'Wickets', 'Avg',
                    'Strike Rate', 'Economy Rate', '5W', '10W', 'WPM', 'POM'
                ]]

                st.markdown("<h3 style='color:#f04f53; text-align: center;'>Home/Away Statistics</h3>", unsafe_allow_html=True)
                st.dataframe(bowlhomeaway_df, use_container_width=True, hide_index=True)

                ###-------------------------------------HOME/AWAY GRAPHS----------------------------------------###
                # Create subplots for Bowling Average, Strike Rate, and Economy Rate by Home/Away
                fig_ha_metrics = make_subplots(rows=1, cols=3, subplot_titles=("Bowling Average", "Strike Rate", "Economy Rate"))

                # Handle 'All' selection
                if 'All' in name_choice:
                    # Check if HomeOrAway column exists and has data before grouping
                    if 'HomeOrAway' in filtered_df.columns and not filtered_df['HomeOrAway'].isnull().all():
                        all_players_ha_stats = filtered_df.groupby('HomeOrAway').agg({
                            'Bowler_Runs': 'sum',
                            'Bowler_Wkts': 'sum',
                            'Bowler_Balls': 'sum'
                        }).reset_index()

                        # Proceed only if all_players_ha_stats is not empty
                        if not all_players_ha_stats.empty:
                            all_players_ha_stats['Avg'] = (all_players_ha_stats['Bowler_Runs'] / all_players_ha_stats['Bowler_Wkts'].replace(0, np.nan)).round(2).fillna(0)
                            all_players_ha_stats['SR'] = (all_players_ha_stats['Bowler_Balls'] / all_players_ha_stats['Bowler_Wkts'].replace(0, np.nan)).round(2).fillna(0)
                            # Ensure Bowler_Balls/6 is not zero before division
                            all_players_ha_stats['Econ'] = (all_players_ha_stats['Bowler_Runs'] / (all_players_ha_stats['Bowler_Balls']/6).replace(0, np.nan)).round(2).fillna(0)

                            all_color = '#f84e4e' if not individual_players else 'black'

                            # Add traces for 'All' - Home and Away separately
                            for ha_status in ['Home', 'Away']:
                                data_subset = all_players_ha_stats[all_players_ha_stats['HomeOrAway'] == ha_status]
                                if not data_subset.empty:
                                    line_style = 'solid' if ha_status == 'Home' else 'dash'
                                    legend_name = f'All Players - {ha_status}'
                                    show_legend_main = True # Show both Home and Away in legend for the first plot

                                    fig_ha_metrics.add_trace(go.Scatter(
                                        x=data_subset['Year'],
                                        y=data_subset['Avg'],
                                        mode='lines+markers',
                                        name=legend_name,
                                        legendgroup='All',
                                        line=dict(color=all_color, dash=line_style),
                                        marker=dict(color=all_color, size=8),
                                        showlegend=show_legend_main # Apply change here
                                    ), row=1, col=1)

                                    fig_ha_metrics.add_trace(go.Scatter(
                                        x=data_subset['Year'],
                                        y=data_subset['SR'],
                                        mode='lines+markers',
                                        name=legend_name, # Name needed for hover, but legend entry hidden
                                        legendgroup='All',
                                        line=dict(color=all_color, dash=line_style),
                                        marker=dict(color=all_color, size=8),
                                        showlegend=False # Keep False for SR plot
                                    ), row=1, col=2)

                                    fig_ha_metrics.add_trace(go.Scatter(
                                        x=data_subset['Year'],
                                        y=data_subset['Econ'],
                                        mode='lines+markers',
                                        name=legend_name, # Name needed for hover, but legend entry hidden
                                        legendgroup='All',
                                        line=dict(color=all_color, dash=line_style),
                                        marker=dict(color=all_color, size=8),
                                        showlegend=False # Keep False for Econ plot
                                    ), row=1, col=3)
                    else:
                        st.warning("Not enough data to display 'All Players' Home/Away metrics.")


                # Add individual player traces
                for i, name in enumerate(individual_players):
                     # Check if HomeOrAway column exists and has data for the player before grouping
                    player_df_ha = filtered_df[(filtered_df['Name'] == name) & ('HomeOrAway' in filtered_df.columns) & (filtered_df['HomeOrAway'].notnull())]
                    if not player_df_ha.empty:
                        player_ha_stats = player_df_ha.groupby('HomeOrAway').agg({
                            'Bowler_Runs': 'sum',
                            'Bowler_Wkts': 'sum',
                            'Bowler_Balls': 'sum'
                        }).reset_index()

                        # Proceed only if player_ha_stats is not empty
                        if not player_ha_stats.empty:
                            player_ha_stats['Avg'] = (player_ha_stats['Bowler_Runs'] / player_ha_stats['Bowler_Wkts']).replace([np.inf, -np.inf], 0).fillna(0).round(2)
                            player_ha_stats['SR'] = (player_ha_stats['Bowler_Balls'] / player_ha_stats['Bowler_Wkts']).replace([np.inf, -np.inf], 0).fillna(0).round(2)
                            # Ensure Bowler_Balls/6 is not zero before division
                            player_ha_stats['Econ'] = (player_ha_stats['Bowler_Runs'] / (player_ha_stats['Bowler_Balls']/6).replace(0, np.nan)).replace([np.inf, -np.inf], 0).fillna(0).round(2)


                            color = '#f84e4e' if i == 0 else f'#{random.randint(0, 0xFFFFFF):06x}'

                            fig_ha_metrics.add_trace(go.Bar(
                                x=player_ha_stats['HomeOrAway'],
                                y=player_ha_stats['Avg'],
                                name=name,
                                legendgroup=name,
                                marker_color=color,
                                showlegend=True
                            ), row=1, col=1)

                            fig_ha_metrics.add_trace(go.Bar(
                                x=player_ha_stats['HomeOrAway'],
                                y=player_ha_stats['SR'],
                                name=name,
                                legendgroup=name,
                                marker_color=color,
                                showlegend=False
                            ), row=1, col=2)

                            fig_ha_metrics.add_trace(go.Bar(
                                x=player_ha_stats['HomeOrAway'],
                                y=player_ha_stats['Econ'],
                                name=name,
                                legendgroup=name,
                                marker_color=color,
                                showlegend=False
                            ), row=1, col=3)
                    # else: # Optional: Add a warning if a specific player has no Home/Away data
                    #     st.warning(f"Not enough data to display Home/Away metrics for {name}.")


                # Update layout
                fig_ha_metrics.update_layout(
                    title="<b>Home/Away Performance Metrics</b>",
                    title_x=0.5,
                    showlegend=True,
                    yaxis_title="Average (Runs/Wicket)",
                    yaxis2_title="Strike Rate (Balls/Wicket)",
                    yaxis3_title="Economy Rate (Runs/Over)",
                    height=500,
                    font=dict(size=12),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    barmode='group',
                    legend=dict(
                        orientation="h",
                        yanchor="top", # Anchor legend to its top
                        y=-0.15,       # Position below the plot area (adjust as needed)
                        xanchor="center",
                        x=0.5
                    )
                )

                fig_ha_metrics.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)', title_text="Home/Away")
                fig_ha_metrics.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

                st.plotly_chart(fig_ha_metrics, use_container_width=True)

                # Create wickets per Home/Away chart
                fig_ha_wickets = go.Figure()

                # Add wickets per Home/Away for 'All' if selected
                if 'All' in name_choice:
                     # Check if HomeOrAway column exists and has data before grouping
                    if 'HomeOrAway' in filtered_df.columns and not filtered_df['HomeOrAway'].isnull().all():
                        wickets_all_ha = filtered_df.groupby('HomeOrAway')['Bowler_Wkts'].sum().reset_index()
                        # Proceed only if wickets_all_ha is not empty
                        if not wickets_all_ha.empty:
                            all_color = '#f84e4e' if not individual_players else 'black'

                            fig_ha_wickets.add_trace(
                                go.Bar(
                                    x=wickets_all_ha['HomeOrAway'],
                                    y=wickets_all_ha['Bowler_Wkts'],
                                    name='All Players',
                                    marker_color=all_color
                                )
                            )
                    # else: # Optional warning handled in the metrics chart already
                    #     pass


                # Add individual player wickets
                for i, name in enumerate(individual_players):
                    # Check if HomeOrAway column exists and has data for the player before grouping
                    player_df_ha_wickets = filtered_df[(filtered_df['Name'] == name) & ('HomeOrAway' in filtered_df.columns) & (filtered_df['HomeOrAway'].notnull())]
                    if not player_df_ha_wickets.empty:
                        player_wickets_ha = player_df_ha_wickets.groupby('HomeOrAway')['Bowler_Wkts'].sum().reset_index()
                        # Proceed only if player_wickets_ha is not empty
                        if not player_wickets_ha.empty:
                            color = '#f84e4e' if i == 0 else f'#{random.randint(0, 0xFFFFFF):06x}'

                            fig_ha_wickets.add_trace(
                                go.Bar(
                                    x=player_wickets_ha['HomeOrAway'],
                                    y=player_wickets_ha['Bowler_Wkts'],
                                    name=name,
                                    marker_color=color
                                )
                            )
                    # else: # Optional warning handled in the metrics chart already
                    #     pass


                st.markdown("<h3 style='color:#f04f53; text-align: center;'>Wickets Home vs Away</h3>", unsafe_allow_html=True)

                fig_ha_wickets.update_layout(
                    showlegend=True,
                    height=500,
                    xaxis_title='Home/Away',
                    yaxis_title='Wickets',
                    margin=dict(l=50, r=50, t=70, b=50),
                    font=dict(size=12),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    barmode='group'
                )

                fig_ha_wickets.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                fig_ha_wickets.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

                st.plotly_chart(fig_ha_wickets, use_container_width=True)

                ###-------------------------------------SEASON GRAPHS (Modified for Home/Away)----------------------------------------###
                st.markdown("<hr>", unsafe_allow_html=True) # Add a separator
                st.markdown("<h2 style='color:#f04f53; text-align: center;'>Season Trends (Home vs Away)</h2>", unsafe_allow_html=True) # Updated Title

                # Create subplots for Bowling Average, Strike Rate, and Economy Rate
                fig_season_metrics_ha = make_subplots(rows=1, cols=3, subplot_titles=("Bowling Average", "Strike Rate", "Economy Rate")) # Renamed fig

                # Handle 'All' selection
                if 'All' in name_choice:
                    # Group by Year and HomeOrAway
                    all_players_stats_ha = filtered_df.groupby(['Year', 'HomeOrAway']).agg({
                        'Bowler_Runs': 'sum',
                        'Bowler_Wkts': 'sum',
                        'Bowler_Balls': 'sum'
                    }).reset_index()

                    # Check for division by zero or NaN results
                    all_players_stats_ha['Avg'] = (all_players_stats_ha['Bowler_Runs'] / all_players_stats_ha['Bowler_Wkts'].replace(0, np.nan)).round(2).fillna(0)
                    all_players_stats_ha['SR'] = (all_players_stats_ha['Bowler_Balls'] / all_players_stats_ha['Bowler_Wkts'].replace(0, np.nan)).round(2).fillna(0)
                    all_players_stats_ha['Econ'] = (all_players_stats_ha['Bowler_Runs'] / (all_players_stats_ha['Bowler_Balls']/6).replace(0, np.nan)).round(2).fillna(0)

                    # Use streamlit red if only 'All' selected, black if names also selected
                    all_color = '#f84e4e' if not individual_players else 'black'

                    # Add traces for 'All' - Home and Away separately
                    for ha_status in ['Home', 'Away']:
                        data_subset = all_players_stats_ha[all_players_stats_ha['HomeOrAway'] == ha_status]
                        if not data_subset.empty:
                            line_style = 'solid' if ha_status == 'Home' else 'dash'
                            legend_name = f'All Players - {ha_status}'
                            show_legend_main = True # Show both Home and Away in legend for the first plot

                            fig_season_metrics_ha.add_trace(go.Scatter(
                                x=data_subset['Year'],
                                y=data_subset['Avg'],
                                mode='lines+markers',
                                name=legend_name,
                                legendgroup='All',
                                line=dict(color=all_color, dash=line_style),
                                marker=dict(color=all_color, size=8),
                                showlegend=show_legend_main # Apply change here
                            ), row=1, col=1)

                            fig_season_metrics_ha.add_trace(go.Scatter(
                                x=data_subset['Year'],
                                y=data_subset['SR'],
                                mode='lines+markers',
                                name=legend_name, # Name needed for hover, but legend entry hidden
                                legendgroup='All',
                                line=dict(color=all_color, dash=line_style),
                                marker=dict(color=all_color, size=8),
                                showlegend=False # Keep False for SR plot
                            ), row=1, col=2)

                            fig_season_metrics_ha.add_trace(go.Scatter(
                                x=data_subset['Year'],
                                y=data_subset['Econ'],
                                mode='lines+markers',
                                name=legend_name, # Name needed for hover, but legend entry hidden
                                legendgroup='All',
                                line=dict(color=all_color, dash=line_style),
                                marker=dict(color=all_color, size=8),
                                showlegend=False # Keep False for Econ plot
                            ), row=1, col=3)

                # Add individual player traces
                for i, name in enumerate(individual_players):
                    player_stats = filtered_df[filtered_df['Name'] == name]
                    # Group by Year and HomeOrAway
                    player_yearly_stats_ha = player_stats.groupby(['Year', 'HomeOrAway']).agg({
                        'Bowler_Runs': 'sum',
                        'Bowler_Wkts': 'sum',
                        'Bowler_Balls': 'sum'
                    }).reset_index()

                    # Check for division by zero or NaN results
                    player_yearly_stats_ha['Avg'] = (player_yearly_stats_ha['Bowler_Runs'] / player_yearly_stats_ha['Bowler_Wkts'].replace(0, np.nan)).round(2).fillna(0)
                    player_yearly_stats_ha['SR'] = (player_yearly_stats_ha['Bowler_Balls'] / player_yearly_stats_ha['Bowler_Wkts'].replace(0, np.nan)).round(2).fillna(0)
                    player_yearly_stats_ha['Econ'] = (player_yearly_stats_ha['Bowler_Runs'] / (player_yearly_stats_ha['Bowler_Balls']/6).replace(0, np.nan)).round(2).fillna(0)

                    # First player gets streamlit red, others get random colors
                    base_color = '#f84e4e' if i == 0 else f'#{random.randint(0, 0xFFFFFF):06x}'

                    # Add traces for player - Home and Away separately
                    for ha_status in ['Home', 'Away']:
                        data_subset = player_yearly_stats_ha[player_yearly_stats_ha['HomeOrAway'] == ha_status]
                        if not data_subset.empty:
                            line_style = 'solid' if ha_status == 'Home' else 'dash'
                            legend_name = f'{name} - {ha_status}'
                            show_legend_main = True # Show both Home and Away in legend for the first plot

                            # Add bowling average trace
                            fig_season_metrics_ha.add_trace(go.Scatter(
                                x=data_subset['Year'],
                                y=data_subset['Avg'],
                                mode='lines+markers',
                                name=legend_name,
                                legendgroup=name, # Group by player name
                                line=dict(color=base_color, dash=line_style),
                                marker=dict(color=base_color, size=8),
                                showlegend=show_legend_main # Apply change here
                            ), row=1, col=1)

                            # Add strike rate trace
                            fig_season_metrics_ha.add_trace(go.Scatter(
                                x=data_subset['Year'],
                                y=data_subset['SR'],
                                mode='lines+markers',
                                name=legend_name, # Name needed for hover, but legend entry hidden
                                legendgroup=name,
                                line=dict(color=base_color, dash=line_style),
                                marker=dict(color=base_color, size=8),
                                showlegend=False # Keep False for SR plot
                            ), row=1, col=2)

                            # Add economy rate trace
                            fig_season_metrics_ha.add_trace(go.Scatter(
                                x=data_subset['Year'],
                                y=data_subset['Econ'],
                                mode='lines+markers',
                                name=legend_name, # Name needed for hover, but legend entry hidden
                                legendgroup=name,
                                line=dict(color=base_color, dash=line_style),
                                marker=dict(color=base_color, size=8),
                                showlegend=False # Keep False for Econ plot
                            ), row=1, col=3)

                # Update layout
                fig_season_metrics_ha.update_layout( # Use renamed fig
                    title="<b>Season Performance Metrics (Home vs Away)</b>", # Updated title
                    title_x=0.5,
                    showlegend=True,
                    yaxis_title="Average (Runs/Wicket)",
                    yaxis2_title="Strike Rate (Balls/Wicket)",
                    yaxis3_title="Economy Rate (Runs/Over)",
                    height=550, # Increased height slightly for legend space
                    font=dict(size=12),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(
                        orientation="h",
                        yanchor="top", # Anchor legend to its top
                        y=-0.15,       # Position below the plot area (adjust as needed)
                        xanchor="center",
                        x=0.5
                    )
                )

                fig_season_metrics_ha.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)', title_text="Year")
                fig_season_metrics_ha.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

                st.plotly_chart(fig_season_metrics_ha, use_container_width=True) # Use renamed fig

                # Create wickets per year chart (Home vs Away)
                fig_season_wickets_ha = go.Figure() # Renamed variable

                # Add wickets per year for 'All' if selected
                if 'All' in name_choice:
                    # Group by Year and HomeOrAway
                    wickets_all_ha = filtered_df.groupby(['Year', 'HomeOrAway'])['Bowler_Wkts'].sum().reset_index()
                    all_color = '#f84e4e' if not individual_players else 'black'

                    # Add traces for 'All' - Home and Away separately
                    for ha_status in ['Home', 'Away']:
                        data_subset = wickets_all_ha[wickets_all_ha['HomeOrAway'] == ha_status]
                        if not data_subset.empty:
                            legend_name = f'All Players - {ha_status}'
                            show_legend_main = True # Show both Home and Away in legend

                            fig_season_wickets_ha.add_trace( # Use renamed variable
                                go.Bar(
                                    x=data_subset['Year'],
                                    y=data_subset['Bowler_Wkts'],
                                    name=legend_name,
                                    legendgroup='All', # Group by 'All'
                                    marker_color=all_color,
                                    opacity=1.0 if ha_status == 'Home' else 0.7, # Differentiate Home/Away bars
                                    showlegend=show_legend_main # Apply change here
                                )
                            )

                # Add individual player wickets
                for i, name in enumerate(individual_players):
                    # Group by Year and HomeOrAway
                    player_wickets_ha = filtered_df[filtered_df['Name'] == name].groupby(['Year', 'HomeOrAway'])['Bowler_Wkts'].sum().reset_index()
                    base_color = '#f84e4e' if i == 0 else f'#{random.randint(0, 0xFFFFFF):06x}'

                    # Add traces for player - Home and Away separately
                    for ha_status in ['Home', 'Away']:
                        data_subset = player_wickets_ha[player_wickets_ha['HomeOrAway'] == ha_status]
                        if not data_subset.empty:
                            legend_name = f'{name} - {ha_status}'
                            show_legend_main = True # Show both Home and Away in legend

                            fig_season_wickets_ha.add_trace( # Use renamed variable
                                go.Bar(
                                    x=data_subset['Year'],
                                    y=data_subset['Bowler_Wkts'],
                                    name=legend_name,
                                    legendgroup=name, # Group by player name
                                    marker_color=base_color,
                                    opacity=1.0 if ha_status == 'Home' else 0.7, # Differentiate Home/Away bars
                                    showlegend=show_legend_main # Apply change here
                                )
                            )

                st.markdown("<h3 style='color:#f04f53; text-align: center;'>Wickets Per Year (Home vs Away)</h3>", unsafe_allow_html=True) # Updated title

                fig_season_wickets_ha.update_layout( # Use renamed variable
                    showlegend=True,
                    height=550, # Increased height slightly for legend space
                    xaxis_title='Year',
                    yaxis_title='Wickets',
                    margin=dict(l=50, r=50, t=70, b=50),
                    font=dict(size=12),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    barmode='group', # Keep as group to see Home/Away side-by-side per player/year
                    legend=dict(
                        orientation="h",
                        yanchor="top", # Anchor legend to its top
                        y=-0.15,       # Position below the plot area (adjust as needed)
                        xanchor="center",
                        x=0.5
                    )
                )

                fig_season_wickets_ha.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)') # Use renamed variable
                fig_season_wickets_ha.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)') # Use renamed variable

                st.plotly_chart(fig_season_wickets_ha, use_container_width=True) # Use renamed variable

        ###-------------------------------------RECORDS TAB-------------------------------------###
        # Records Tab
        with tabs[11]:
            st.markdown("<h2 style='color:#f04f53; text-align: center;'>Bowling Records</h2>", unsafe_allow_html=True)
            
            # Create columns for layout
            col1, col2 = st.columns(2)

            with col1:
                # --- Best Bowling in an Innings ---
                # Select relevant columns and sort by Wickets (descending) then Runs (ascending)
                best_bowling_df = filtered_df[['Name', 'Bowl_Team', 'Bowler_Wkts', 'Bowler_Runs', 'Bowler_Balls', 'Bat_Team', 'Year']].copy()
                
                # Calculate overs correctly from balls
                best_bowling_df['Overs'] = (best_bowling_df['Bowler_Balls'] // 6) + (best_bowling_df['Bowler_Balls'] % 6) / 10
                best_bowling_df['Overs'] = best_bowling_df['Overs'].round(1)  # Round to 1 decimal place
                
                best_bowling_df = best_bowling_df.sort_values(by=['Bowler_Wkts', 'Bowler_Runs'], ascending=[False, True]).head(10)
                
                # Calculate bowling figures string (e.g., "5/32")
                best_bowling_df['Bowling_Figures'] = best_bowling_df['Bowler_Wkts'].astype(str) + '/' + best_bowling_df['Bowler_Runs'].astype(str)

                # Add Rank column
                best_bowling_df.insert(0, 'Rank', range(1, 1 + len(best_bowling_df)))

                # Rename columns
                best_bowling_df = best_bowling_df.rename(columns={'Bat_Team': 'Opponent', 'Bowl_Team': 'Team'})

                # Reorder columns
                best_bowling_df = best_bowling_df[['Rank', 'Name', 'Team', 'Bowling_Figures', 'Overs', 'Opponent', 'Year']]

                # Display the Best Bowling header
                st.markdown("<h3 style='color:#f04f53; text-align: center;'>Best Bowling in an Innings</h3>", unsafe_allow_html=True)
                # Display the dataframe
                st.dataframe(best_bowling_df, use_container_width=True, hide_index=True)

            with col2:
                # --- Most Economical Spells (min overs by format) ---
                # Make a copy of the filtered dataframe with only needed columns to avoid any issues
                eco_df = filtered_df[['Name', 'Bowl_Team', 'Match_Format', 'Bowler_Runs', 'Bowler_Balls', 'Bowler_Wkts', 'Bat_Team', 'Year']].copy()
                
                # Set minimum overs required based on format
                def min_overs_required(format_name):
                    if format_name in ['T20', 'T20 Over International', '20 Over International']:
                        return 4  # T20 formats: min 4 overs
                    elif format_name in ['One Day International', 'List A']:
                        return 10  # ODI formats: min 10 overs
                    else:
                        return 15  # Test/FC formats: min 15 overs

                # Add column for minimum overs
                eco_df['Min_Overs'] = eco_df['Match_Format'].apply(min_overs_required)
                
                # Calculate overs properly from balls
                eco_df['Overs'] = (eco_df['Bowler_Balls'] // 6) + (eco_df['Bowler_Balls'] % 6) / 10
                eco_df['Overs'] = eco_df['Overs'].round(1)  # Round to 1 decimal place

                # Filter for spells meeting minimum over requirements
                eco_df = eco_df[eco_df['Overs'] >= eco_df['Min_Overs']]
                
                # Calculate economy rate - using correctly calculated overs value
                eco_df['Economy'] = (eco_df['Bowler_Runs'] / eco_df['Overs']).round(2)
                
                # Sort by Economy (ascending)
                eco_df = eco_df.sort_values(by='Economy', ascending=True).head(10)
                
                # Add Rank column
                eco_df.insert(0, 'Rank', range(1, 1 + len(eco_df)))
                
                # Create the final dataframe for display with properly renamed columns
                display_df = pd.DataFrame({
                    'Rank': eco_df['Rank'],
                    'Name': eco_df['Name'],
                    'Team': eco_df['Bowl_Team'],  # Add Team column
                    'Economy': eco_df['Economy'],
                    'Overs': eco_df['Overs'],
                    'Runs': eco_df['Bowler_Runs'],
                    'Wickets': eco_df['Bowler_Wkts'],
                    'Opponent': eco_df['Bat_Team'],
                    'Format': eco_df['Match_Format'],
                    'Year': eco_df['Year']
                })

                # Display the header
                st.markdown("<h3 style='color:#f04f53; text-align: center;'>Most Economical Spells</h3>", unsafe_allow_html=True)
                # Display the final dataframe
                st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Match bowling records section
            st.markdown("<h2 style='color:#f04f53; text-align: center;'>Match Bowling Records</h2>", unsafe_allow_html=True)
            
            # Create columns for layout - use two columns now
            col3, col4 = st.columns(2)

            with col3:
                # --- Best Match Bowling Figures ---
                # Group by player, match file and calculate total wickets and runs
                match_bowling = filtered_df.groupby(['Name', 'File Name', 'Bat_Team', 'Match_Format']).agg({
                    'Bowler_Wkts': 'sum',
                    'Bowler_Runs': 'sum',
                    'Bowler_Balls': 'sum',  # Use Bowler_Balls instead of Overs
                    'Year': 'first',
                    'Bowl_Team': 'first'  # Add Bowl_Team for display
                }).reset_index()
                
                # Calculate overs from balls
                match_bowling['Overs'] = (match_bowling['Bowler_Balls'] // 6) + (match_bowling['Bowler_Balls'] % 6) / 10
                match_bowling['Overs'] = match_bowling['Overs'].round(1)  # Round to 1 decimal place
                
                # Sort by wickets (descending) then runs (ascending)
                match_bowling = match_bowling.sort_values(by=['Bowler_Wkts', 'Bowler_Runs'], ascending=[False, True]).head(10)
                
                # Calculate match bowling figures
                match_bowling['Match_Figures'] = match_bowling['Bowler_Wkts'].astype(str) + '/' + match_bowling['Bowler_Runs'].astype(str)
                
                # Add rank column
                match_bowling.insert(0, 'Rank', range(1, 1 + len(match_bowling)))
                
                # Rename columns
                match_bowling = match_bowling.rename(columns={
                    'Bat_Team': 'Opponent', 
                    'Bowl_Team': 'Team',
                    'Bowler_Wkts': 'Wickets',
                    'Bowler_Runs': 'Runs',
                    'Match_Format': 'Format'
                })
                
                # Select and reorder columns - drop Bowler_Balls from display
                match_bowling_df = match_bowling[['Rank', 'Name', 'Team', 'Match_Figures', 'Wickets', 'Overs', 'Runs', 'Opponent', 'Format', 'Year']]

                # Display the header
                st.markdown("<h3 style='color:#f04f53; text-align: center;'>Best Match Bowling Figures</h3>", unsafe_allow_html=True)
                # Display the dataframe
                st.dataframe(match_bowling_df, use_container_width=True, hide_index=True)

            with col4:
                # --- Most Expensive Bowling Figures ---
                # Use the same grouped data but sort by most runs (descending)
                # Group by player, match file and calculate total wickets and runs
                expensive_bowling = filtered_df.groupby(['Name', 'File Name', 'Bat_Team', 'Match_Format']).agg({
                    'Bowler_Wkts': 'sum',
                    'Bowler_Runs': 'sum',
                    'Bowler_Balls': 'sum',
                    'Year': 'first',
                    'Bowl_Team': 'first'
                }).reset_index()
                
                # Calculate overs from balls
                expensive_bowling['Overs'] = (expensive_bowling['Bowler_Balls'] // 6) + (expensive_bowling['Bowler_Balls'] % 6) / 10
                expensive_bowling['Overs'] = expensive_bowling['Overs'].round(1)
                
                # Sort by runs (descending) - most expensive first
                expensive_bowling = expensive_bowling.sort_values(by='Bowler_Runs', ascending=False).head(10)
                
                # Calculate match bowling figures
                expensive_bowling['Match_Figures'] = expensive_bowling['Bowler_Wkts'].astype(str) + '/' + expensive_bowling['Bowler_Runs'].astype(str)
                
                # Add rank column
                expensive_bowling.insert(0, 'Rank', range(1, 1 + len(expensive_bowling)))
                
                # Rename columns
                expensive_bowling = expensive_bowling.rename(columns={
                    'Bat_Team': 'Opponent',
                    'Bowl_Team': 'Team',
                    'Bowler_Wkts': 'Wickets',
                    'Bowler_Runs': 'Runs',
                    'Match_Format': 'Format'
                })
                
                # Select and reorder columns
                expensive_bowling_df = expensive_bowling[['Rank', 'Name', 'Team', 'Match_Figures', 'Overs', 'Runs', 'Wickets', 'Opponent', 'Format', 'Year']]
                
                # Display the header
                st.markdown("<h3 style='color:#f04f53; text-align: center;'>Most Expensive Bowling Figures</h3>", unsafe_allow_html=True)
                # Display the dataframe
                st.dataframe(expensive_bowling_df, use_container_width=True, hide_index=True)

            # Seasonal Records Section
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<h2 style='color:#f04f53; text-align: center;'>Seasonal Bests (by Format)</h2>", unsafe_allow_html=True)
            
            # Create two rows with two columns each for the four seasonal stats
            row1_col1, row1_col2 = st.columns(2)
            row2_col1, row2_col2 = st.columns(2)

            with row1_col1:
                # --- Most Wickets in a Season by Format ---
                # Group by player, year, format
                season_wickets = filtered_df.groupby(['Name', 'Year', 'Match_Format']).agg({
                    'Bowler_Wkts': 'sum',
                    'File Name': 'nunique',
                    'Bowler_Runs': 'sum',
                    'Bowler_Balls': 'sum'
                }).reset_index()
                
                # Calculate average and economy - ensure proper overs calculation from balls
                season_wickets['Overs'] = (season_wickets['Bowler_Balls'] // 6) + (season_wickets['Bowler_Balls'] % 6) / 10
                season_wickets['Average'] = (season_wickets['Bowler_Runs'] / season_wickets['Bowler_Wkts']).round(2)
                season_wickets['Economy'] = (season_wickets['Bowler_Runs'] / season_wickets['Overs']).round(2)
                
                # Sort by wickets and get top seasons
                season_wickets = season_wickets.sort_values(by='Bowler_Wkts', ascending=False).head(10)
                
                # Add rank column
                season_wickets.insert(0, 'Rank', range(1, 1 + len(season_wickets)))
                
                # Rename columns
                season_wickets = season_wickets.rename(columns={
                    'Match_Format': 'Format',
                    'Bowler_Wkts': 'Wickets',
                    'File Name': 'Matches',
                    'Bowler_Runs': 'Runs'
                })
                
                # Select and reorder columns
                season_wickets_df = season_wickets[['Rank', 'Name', 'Year', 'Format', 'Matches', 'Wickets', 'Average', 'Economy']]

                # Display the header
                st.markdown("<h3 style='color:#f04f53; text-align: center;'>Most Wickets in a Season</h3>", unsafe_allow_html=True)
                # Display the dataframe
                st.dataframe(season_wickets_df, use_container_width=True, hide_index=True)

            with row1_col2:
                # --- Best Bowling Average in a Season (min 15 wickets) ---
                # Group by player, year, format
                season_avg = filtered_df.groupby(['Name', 'Year', 'Match_Format']).agg({
                    'Bowler_Wkts': 'sum',
                    'File Name': 'nunique',
                    'Bowler_Runs': 'sum',
                    'Bowler_Balls': 'sum'
                }).reset_index()
                
                # Filter for at least 15 wickets
                season_avg = season_avg[season_avg['Bowler_Wkts'] >= 15]
                
                # Calculate average and economy - ensure proper overs calculation
                season_avg['Overs'] = (season_avg['Bowler_Balls'] // 6) + (season_avg['Bowler_Balls'] % 6) / 10
                season_avg['Average'] = (season_avg['Bowler_Runs'] / season_avg['Bowler_Wkts']).round(2)
                season_avg['SR'] = (season_avg['Bowler_Balls'] / season_avg['Bowler_Wkts']).round(2)
                
                # Sort by average (ascending)
                season_avg = season_avg.sort_values(by='Average', ascending=True).head(10)
                
                # Add rank column
                season_avg.insert(0, 'Rank', range(1, 1 + len(season_avg)))
                
                # Rename columns
                season_avg = season_avg.rename(columns={
                    'Match_Format': 'Format',
                    'Bowler_Wkts': 'Wickets',
                    'File Name': 'Matches',
                    'Bowler_Runs': 'Runs'
                })
                
                # Select and reorder columns
                season_avg_df = season_avg[['Rank', 'Name', 'Year', 'Format', 'Wickets', 'Average', 'SR', 'Matches']]

                # Display the header
                st.markdown("<h3 style='color:#f04f53; text-align: center;'>Best Bowling Average in a Season (Min 15 Wickets)</h3>", unsafe_allow_html=True)
                # Display the dataframe
                st.dataframe(season_avg_df, use_container_width=True, hide_index=True)
                
            with row2_col1:
                # --- Best Strike Rate in a Season (min 15 wickets) ---
                # Group by player, year, format
                season_sr = filtered_df.groupby(['Name', 'Year', 'Match_Format']).agg({
                    'Bowler_Wkts': 'sum',
                    'File Name': 'nunique',
                    'Bowler_Runs': 'sum',
                    'Bowler_Balls': 'sum'
                }).reset_index()
                
                # Filter for at least 15 wickets
                season_sr = season_sr[season_sr['Bowler_Wkts'] >= 15]
                
                # Calculate average and economy - ensure proper overs calculation
                season_sr['Overs'] = (season_sr['Bowler_Balls'] // 6) + (season_sr['Bowler_Balls'] % 6) / 10
                season_sr['Average'] = (season_sr['Bowler_Runs'] / season_sr['Bowler_Wkts']).round(2)
                season_sr['SR'] = (season_sr['Bowler_Balls'] / season_sr['Bowler_Wkts']).round(2)
                
                # Sort by strike rate (ascending)
                season_sr = season_sr.sort_values(by='SR', ascending=True).head(10)
                
                # Add rank column
                season_sr.insert(0, 'Rank', range(1, 1 + len(season_sr)))
                
                # Rename columns
                season_sr = season_sr.rename(columns={
                    'Match_Format': 'Format',
                    'Bowler_Wkts': 'Wickets',
                    'File Name': 'Matches',
                    'Bowler_Runs': 'Runs',
                    'SR': 'Strike Rate'
                })
                
                # Select and reorder columns
                season_sr_df = season_sr[['Rank', 'Name', 'Year', 'Format', 'Wickets', 'Strike Rate', 'Average', 'Matches']]

                # Display the header
                st.markdown("<h3 style='color:#f04f53; text-align: center;'>Best Strike Rate in a Season (Min 15 Wickets)</h3>", unsafe_allow_html=True)
                # Display the dataframe
                st.dataframe(season_sr_df, use_container_width=True, hide_index=True)
                
            with row2_col2:
                # --- Best Economy Rate in a Season (min 15 wickets) ---
                # Group by player, year, format
                season_econ = filtered_df.groupby(['Name', 'Year', 'Match_Format']).agg({
                    'Bowler_Wkts': 'sum',
                    'File Name': 'nunique',
                    'Bowler_Runs': 'sum',
                    'Bowler_Balls': 'sum'
                }).reset_index()
                
                # Filter for at least 15 wickets
                season_econ = season_econ[season_econ['Bowler_Wkts'] >= 15]
                
                # Calculate average and economy - ensure proper overs calculation
                season_econ['Overs'] = (season_econ['Bowler_Balls'] // 6) + (season_econ['Bowler_Balls'] % 6) / 10
                season_econ['Economy'] = (season_econ['Bowler_Runs'] / season_econ['Overs']).round(2)
                season_econ['Average'] = (season_econ['Bowler_Runs'] / season_econ['Bowler_Wkts']).round(2)
                season_econ['SR'] = (season_econ['Bowler_Balls'] / season_econ['Bowler_Wkts']).round(2)
                
                # Sort by economy rate (ascending)
                season_econ = season_econ.sort_values(by='Economy', ascending=True).head(10)
                
                # Add rank column
                season_econ.insert(0, 'Rank', range(1, 1 + len(season_econ)))
                
                # Rename columns
                season_econ = season_econ.rename(columns={
                    'Match_Format': 'Format',
                    'Bowler_Wkts': 'Wickets',
                    'File Name': 'Matches',
                    'Bowler_Runs': 'Runs'
                })
                
                # Select and reorder columns
                season_econ_df = season_econ[['Rank', 'Name', 'Year', 'Format', 'Wickets', 'Economy', 'Average', 'Matches']]                # Display the header
                st.markdown("<h3 style='color:#f04f53; text-align: center;'>Best Economy Rate in a Season (Min 15 Wickets)</h3>", unsafe_allow_html=True)
                # Display the dataframe
                st.dataframe(season_econ_df, use_container_width=True, hide_index=True)

    else:
        st.error("No bowling data available. Please upload scorecards first.")

# Display the bowling view
display_bowl_view()
