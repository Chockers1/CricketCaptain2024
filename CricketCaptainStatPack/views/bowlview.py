import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

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
            
        except Exception as e:
            st.error(f"Error processing dates. Using original dates.")
            bowl_df = st.session_state['bowl_df'].copy()
            bowl_df['Date'] = pd.to_datetime(bowl_df['Date'], errors='coerce')
            bowl_df['Year'] = bowl_df['Date'].dt.year

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
            st.session_state.prev_bowl_teams = current_bowl_teams

        ###-------------------------------------HEADER AND FILTERS-------------------------------------###
        st.markdown("<h1 style='color:#f04f53; text-align: center;'>Bowling Statistics</h1>", unsafe_allow_html=True)
        
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
                        label_visibility='collapsed')

        # The rest of the sliders remain the same
        with col6:
            st.markdown("<p style='text-align: center;'>Choose Position:</p>", unsafe_allow_html=True)
            position_choice = st.slider('', 
                        min_value=1, 
                        max_value=11, 
                        value=(1, 11),
                        label_visibility='collapsed')

        with col7:
            st.markdown("<p style='text-align: center;'>Wickets Range</p>", unsafe_allow_html=True)
            wickets_range = st.slider('', 
                                    min_value=0, 
                                    max_value=max_wickets, 
                                    value=(0, max_wickets),
                                    label_visibility='collapsed')

        with col8:
            st.markdown("<p style='text-align: center;'>Matches Range</p>", unsafe_allow_html=True)
            matches_range = st.slider('', 
                                    min_value=1, 
                                    max_value=max_matches, 
                                    value=(1, max_matches),
                                    label_visibility='collapsed')

        with col9:
            st.markdown("<p style='text-align: center;'>Average Range</p>", unsafe_allow_html=True)
            avg_range = st.slider('', 
                                min_value=0.0, 
                                max_value=max_avg, 
                                value=(0.0, max_avg),
                                label_visibility='collapsed')

        with col10:
            st.markdown("<p style='text-align: center;'>Strike Rate Range</p>", unsafe_allow_html=True)
            sr_range = st.slider('', 
                                min_value=0.0, 
                                max_value=max_sr, 
                                value=(0.0, max_sr),
                                label_visibility='collapsed')

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

        ###-------------------------------------CAREER STATS-------------------------------------###
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

        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Career Statistics</h3>", unsafe_allow_html=True)
        st.dataframe(bowlcareer_df, use_container_width=True, hide_index=True)

###########--------------------------- SCATTER PLOT GRAPH--------------------------###################

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
        st.plotly_chart(scatter_fig)

###-------------------------------------FORMAT STATS-------------------------------------###
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

        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Format Record</h3>", unsafe_allow_html=True)
        st.dataframe(bowlformat_df, use_container_width=True, hide_index=True)

###-------------------------------------SEASON STATS-------------------------------------###
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

        ###-------------------------------------GRAPHS----------------------------------------###
        # Create subplots for Bowling Average and Strike Rate
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Bowling Average", "Bowling Strike Rate"))

        # Handle 'All' selection
        if 'All' in name_choice:
            all_players_stats = filtered_df.groupby('Year').agg({
                'Bowler_Runs': 'sum',
                'Bowler_Wkts': 'sum',
                'Bowler_Balls': 'sum'
            }).reset_index()

            all_players_stats['Avg'] = (all_players_stats['Bowler_Runs'] / all_players_stats['Bowler_Wkts']).round(2).fillna(0)
            all_players_stats['SR'] = (all_players_stats['Bowler_Balls'] / all_players_stats['Bowler_Wkts']).round(2).fillna(0)

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

        # Update layout
        fig.update_layout(
            showlegend=True,
            title_text=None,
            yaxis_title="Average (Runs/Wicket)",
            yaxis2_title="Strike Rate (Balls/Wicket)",
            height=500,
            font=dict(size=12),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

        st.plotly_chart(fig)

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

        st.plotly_chart(fig)

###-------------------------------------LATEST 15 INNINGS-------------------------------------###
        # Create latest innings dataframe
        latest_inns_df = filtered_df.groupby(['Name', 'Match_Format', 'Date', 'Innings']).agg({
            'Bowl_Team': 'first',
            'Bat_Team': 'first',
            'Overs': 'sum',
            'Maidens': 'sum',
            'Bowler_Runs': 'sum',
            'Bowler_Wkts': 'sum'
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
        latest_inns_df = latest_inns_df.sort_values(by='Date', ascending=False).head(15)
        latest_inns_df['Date'] = latest_inns_df['Date'].dt.strftime('%d/%m/%Y')

        # Format Overs to 1 decimal place
        latest_inns_df['Overs'] = latest_inns_df['Overs'].apply(lambda x: f"{x:.1f}")

        # Reorder columns
        latest_inns_df = latest_inns_df[[
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
        styled_df = latest_inns_df.style.applymap(color_wickets, subset=['Wickets'])

        # Display section header and dataframe
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Last 15 Bowling Innings</h3>", unsafe_allow_html=True)
        st.dataframe(styled_df, height=575, use_container_width=True, hide_index=True)

###-------------------------------------OPPONENT STATS-------------------------------------###
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
            'Name', 'Opposition', 'Matches', 'Balls', 'Overs', 'M/D', 'Runs', 'Wickets',
            'Strike Rate', 'Economy Rate', '5W', '10W', 'WPM'
        ]]

        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Opposition Statistics</h3>", unsafe_allow_html=True)
        st.dataframe(opponent_summary, use_container_width=True, hide_index=True)

        # Create opponent averages graph
        fig = go.Figure()

        # Calculate and show 'All' stats if it's selected
        if 'All' in name_choice:
            all_opponent_stats = filtered_df.groupby(['Bat_Team']).agg({
                'Bowler_Runs': 'sum',
                'Bowler_Wkts': 'sum'
            }).reset_index()
            
            all_opponent_stats['Average'] = (all_opponent_stats['Bowler_Runs'] / all_opponent_stats['Bowler_Wkts']).round(2)
            
            all_color = '#f84e4e' if not individual_players else 'black'
            
            fig.add_trace(
                go.Bar(
                    x=all_opponent_stats['Bat_Team'],
                    y=all_opponent_stats['Average'],
                    name='All Players',
                    marker_color=all_color,
                    text=all_opponent_stats['Bowler_Wkts'],
                    textposition='auto'
                )
            )

        # Add individual player traces
        for i, name in enumerate(individual_players):
            player_data = filtered_df[filtered_df['Name'] == name].groupby('Bat_Team').agg({
                'Bowler_Runs': 'sum',
                'Bowler_Wkts': 'sum'
            }).reset_index()
            
            player_data['Average'] = (player_data['Bowler_Runs'] / player_data['Bowler_Wkts']).round(2)
            
            color = '#f84e4e' if i == 0 else f'#{random.randint(0, 0xFFFFFF):06x}'
            
            fig.add_trace(
                go.Bar(
                    x=player_data['Bat_Team'],
                    y=player_data['Average'],
                    name=name,
                    marker_color=color,
                    text=player_data['Bowler_Wkts'],
                    textposition='auto'
                )
            )

        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Bowling Average vs Opposition</h3>", unsafe_allow_html=True)

        # Update layout
        fig.update_layout(
            showlegend=True,
            height=500,
            xaxis_title="Opposition Team",
            yaxis_title="Average",
            font=dict(size=12),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis={'categoryorder':'total ascending'},
            barmode='group'
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

        st.plotly_chart(fig)

 ###-------------------------------------LOCATION STATS-------------------------------------###
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
            'Name', 'Location', 'Matches', 'Balls', 'Overs', 'M/D', 'Runs', 'Wickets',
            'Strike Rate', 'Economy Rate', '5W', '10W', 'WPM'
        ]]

        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Location Statistics</h3>", unsafe_allow_html=True)
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

        # Add title
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Bowling Average by Location</h3>", unsafe_allow_html=True)

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

        # Add gridlines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

        # Display graph
        st.plotly_chart(fig)

###-------------------------------------INNINGS STATS-------------------------------------###
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
        innings_summary['Strike Rate'] = (innings_summary['Balls'] / innings_summary['Wickets']).round(2)
        innings_summary['Economy Rate'] = (innings_summary['Runs'] / innings_summary['Overs']).round(2)
        innings_summary['WPM'] = (innings_summary['Wickets'] / innings_summary['Matches']).round(2)

        # Count 5W innings
        five_wickets = filtered_df[filtered_df['Bowler_Wkts'] >= 5].groupby(['Name', 'Innings']).size().reset_index(name='5W')
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
            'Name', 'Innings', 'Matches', 'Balls', 'Overs', 'M/D', 'Runs', 'Wickets',
            'Strike Rate', 'Economy Rate', '5W', '10W', 'WPM'
        ]]

        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Innings Statistics</h3>", unsafe_allow_html=True)
        st.dataframe(innings_summary, use_container_width=True, hide_index=True)

        # Create innings averages graph
        fig = go.Figure()

        # Add 'All' trace first if selected
        if 'All' in name_choice:
            all_innings_stats = filtered_df.groupby(['Innings']).agg({
                'Bowler_Runs': 'sum',
                'Bowler_Wkts': 'sum'
            }).reset_index()
            
            all_innings_stats['Average'] = (all_innings_stats['Bowler_Runs'] / all_innings_stats['Bowler_Wkts']).round(2)
            
            all_color = '#f84e4e' if not individual_players else 'black'
            
            fig.add_trace(
                go.Bar(
                    x=all_innings_stats['Innings'],
                    y=all_innings_stats['Average'],
                    name='All Players',
                    marker_color=all_color,
                    text=all_innings_stats['Bowler_Wkts'],
                    textposition='auto'
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
                    x=player_data['Innings'],
                    y=player_data['Average'],
                    name=name,
                    marker_color=color,
                    text=player_data['Bowler_Wkts'],
                    textposition='auto'
                )
            )

        # Add title
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Bowling Average by Innings</h3>", unsafe_allow_html=True)

        # Update layout
        fig.update_layout(
            showlegend=True,
            height=500,
            xaxis_title="Innings",
            yaxis_title="Average",
            font=dict(size=12),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis={'categoryorder':'array', 'categoryarray':[1,2,3,4]},
            barmode='group'
        )

        # Add gridlines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

        # Display graph
        st.plotly_chart(fig)

###-------------------------------------POSITION STATS-------------------------------------###
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
            'Name', 'Position', 'Matches', 'Balls', 'Overs', 'M/D', 'Runs', 'Wickets',
            'Strike Rate', 'Economy Rate', '5W', '10W', 'WPM'
        ]]

        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Position Statistics</h3>", unsafe_allow_html=True)
        st.dataframe(position_summary, use_container_width=True, hide_index=True)

        # Create position averages graph
        fig = go.Figure()

        # Add 'All' trace first if selected
        if 'All' in name_choice:
            all_position_stats = filtered_df.groupby(['Position']).agg({
                'Bowler_Runs': 'sum',
                'Bowler_Wkts': 'sum'
            }).reset_index()
            
            all_position_stats['Average'] = (all_position_stats['Bowler_Runs'] / all_position_stats['Bowler_Wkts']).round(2)
            
            all_color = '#f84e4e' if not individual_players else 'black'
            
            fig.add_trace(
                go.Bar(
                    x=all_position_stats['Position'],
                    y=all_position_stats['Average'],
                    name='All Players',
                    marker_color=all_color,
                    text=all_position_stats['Bowler_Wkts'],
                    textposition='auto'
                )
            )

        # Add individual player traces
        for i, name in enumerate(individual_players):
            player_data = filtered_df[filtered_df['Name'] == name].groupby('Position').agg({
                'Bowler_Runs': 'sum',
                'Bowler_Wkts': 'sum'
            }).reset_index()
            
            player_data['Average'] = (player_data['Bowler_Runs'] / player_data['Bowler_Wkts']).round(2)
            
            color = '#f84e4e' if i == 0 else f'#{random.randint(0, 0xFFFFFF):06x}'
            
            fig.add_trace(
                go.Bar(
                    x=player_data['Position'],
                    y=player_data['Average'],
                    name=name,
                    marker_color=color,
                    text=player_data['Bowler_Wkts'],
                    textposition='auto'
                )
            )

        # Add title
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Bowling Average by Position</h3>", unsafe_allow_html=True)

        # Update layout
        fig.update_layout(
            showlegend=True,
            height=500,
            xaxis_title="Position",
            yaxis_title="Average",
            font=dict(size=12),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis={'categoryorder':'array', 'categoryarray':[1,2,3,4,5,6,7,8,9,10,11]},
            barmode='group'
        )

        # Add gridlines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

        # Display graph
        st.plotly_chart(fig)

###--------------------------------------CUMULATIVE BOWLING STATS------------------------------------------#######
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
        # Create DataFrame for block stats from filtered_df
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

    # Error handling for missing data
    else:
        st.error("Bowling statistics are not available. Please ensure you have processed the scorecards on the Home page.")

# Display the bowling view
display_bowl_view()
