# Section 1: Imports and Setup
###############################################################################
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def parse_date(date_str):
    """Helper function to parse dates in multiple formats"""
    try:
        # Try different date formats
        for fmt in ['%d/%m/%Y', '%d %b %Y', '%Y-%m-%d']:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except ValueError:
                continue
        # If none of the specific formats work, let pandas try to infer the format
        return pd.to_datetime(date_str)
    except Exception:
        return pd.NaT

###############################################################################
# Section 2: Page Header and Styling
###############################################################################
st.markdown("<h1 style='color:#f04f53; text-align: center;'>Records</h1>", unsafe_allow_html=True)

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
}
.stTabs [data-baseweb="tab"] {
    flex-grow: 1;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Section 1: Imports and Setup
###############################################################################
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def parse_date(date_str):
    """Helper function to parse dates in multiple formats"""
    try:
        # Try different date formats
        for fmt in ['%d/%m/%Y', '%d %b %Y', '%Y-%m-%d']:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except ValueError:
                continue
        # If none of the specific formats work, let pandas try to infer the format
        return pd.to_datetime(date_str)
    except Exception:
        return pd.NaT

###############################################################################
# Section 2: Page Header and Styling
###############################################################################
#st.markdown("<h1 style='color:#f04f53; text-align: center;'>Records</h1>", unsafe_allow_html=True)

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
}
.stTabs [data-baseweb="tab"] {
    flex-grow: 1;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Get unique formats from all dataframes
all_formats = set()

if 'game_df' in st.session_state:
    all_formats.update(st.session_state['game_df']['Match_Format'].unique())
if 'bat_df' in st.session_state:
    all_formats.update(st.session_state['bat_df']['Match_Format'].unique())
if 'bowl_df' in st.session_state:
    all_formats.update(st.session_state['bowl_df']['Match_Format'].unique())
if 'match_df' in st.session_state:
    all_formats.update(st.session_state['match_df']['Match_Format'].unique())

# Create the format filter (full width)
formats = ['All'] + sorted(list(all_formats))
format_choice = st.multiselect('Format:', formats, default='All', key='global_format_filter')

# Function to filter dataframe based on format
def filter_by_format(df):
    if 'All' not in format_choice:
        return df[df['Match_Format'].isin(format_choice)]
    return df
###############################################################################
# Section 3: Create Tabs
###############################################################################
tab1, tab2, tab3, tab4 = st.tabs(["Batting Records", "Bowling Records", "Match Records", "Game Records"])

###############################################################################
# Section 4: Batting Records Tab
###############################################################################
with tab1:
    if 'bat_df' in st.session_state:
        bat_df = st.session_state['bat_df']
        bat_df['Date'] = pd.to_datetime(bat_df['Date'])
        
        # Apply format filter to bat_df first
        bat_df = filter_by_format(bat_df)
        
        # Highest Scores DataFrame
        columns_to_drop = [
            'Bat_Team_x', 'Bowl_Team_x', 'File Name', 'Total_Runs', 
            'Overs', 'Wickets', 'Competition', 'Batted', 'Out',
            'Not Out', '50s', '100s', '200s', '<25&Out', 
            'Caught', 'Bowled', 'LBW', 'Stumped', 'Run Out',
            'Boundary Runs', 'Team Balls', 'Year', 'Player_of_the_Match'
        ]

        highest_scores_df = (bat_df[bat_df['Runs'] >= 100]
                            .drop(columns=columns_to_drop, errors='ignore')
                            .sort_values(by='Runs', ascending=False)
                            .rename(columns={
                                'Bat_Team_y': 'Bat Team',
                                'Bowl_Team_y': 'Bowl Team'
                            }))
        
        highest_scores_df['Date'] = highest_scores_df['Date'].dt.strftime('%d/%m/%Y')

        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Highest Scores</h3>", 
                   unsafe_allow_html=True)
        st.dataframe(highest_scores_df, 
                    use_container_width=True, 
                    hide_index=True)

        # 99 Not Out Club
        not_out_99s_df = (bat_df[(bat_df['Runs'] == 99) & (bat_df['Not Out'] == 1)]
                          .drop(columns=columns_to_drop, errors='ignore')
                          .sort_values(by='Runs', ascending=False)
                          .rename(columns={
                              'Bat_Team_y': 'Bat Team',
                              'Bowl_Team_y': 'Bowl Team'
                          }))
        
        not_out_99s_df['Date'] = not_out_99s_df['Date'].dt.strftime('%d/%m/%Y')

        st.markdown("<h3 style='color:#f04f53; text-align: center;'>99 Not Out Club</h3>", 
                   unsafe_allow_html=True)
        st.dataframe(not_out_99s_df, 
                    use_container_width=True, 
                    hide_index=True)

        # Carrying the Bat
        carrying_bat_df = (bat_df[
            (bat_df['Position'].isin([1, 2])) & 
            (bat_df['Not Out'] == 1) & 
            (bat_df['Wickets'] == 10)]
            .drop(columns=columns_to_drop, errors='ignore')
            .sort_values(by='Runs', ascending=False)
            .rename(columns={
                'Bat_Team_y': 'Bat Team',
                'Bowl_Team_y': 'Bowl Team'
            }))
        
        carrying_bat_df['Date'] = carrying_bat_df['Date'].dt.strftime('%d/%m/%Y')

        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Carrying the Bat</h3>", 
                   unsafe_allow_html=True)
        st.dataframe(carrying_bat_df, 
                    use_container_width=True, 
                    hide_index=True)

        # Hundred in Each Innings
        bat_match_100_df = (bat_df.groupby(['Name', 'Date', 'Home Team'])
                    .agg({
                        'Runs': lambda x: [
                            max([r for r, i in zip(x, bat_df.loc[x.index, 'Innings']) if i in [1, 2]], default=0),
                            max([r for r, i in zip(x, bat_df.loc[x.index, 'Innings']) if i in [3, 4]], default=0)
                        ],
                        'Bat_Team_y': 'first',
                        'Bowl_Team_y': 'first',
                        'Balls': 'sum',
                        '4s': 'sum',
                        '6s': 'sum'
                    })
                    .reset_index())

        bat_match_100_df['1st Innings'] = bat_match_100_df['Runs'].str[0]
        bat_match_100_df['2nd Innings'] = bat_match_100_df['Runs'].str[1]

        hundreds_both_df = (bat_match_100_df[
            (bat_match_100_df['1st Innings'] >= 100) & 
            (bat_match_100_df['2nd Innings'] >= 100)
        ]
        .drop(columns=['Runs'])
        .rename(columns={
            'Bat_Team_y': 'Bat Team',
            'Bowl_Team_y': 'Bowl Team',
            'Home Team': 'Country'
        }))

        hundreds_both_df['Date'] = pd.to_datetime(hundreds_both_df['Date'])
        hundreds_both_df = hundreds_both_df.sort_values(by='Date', ascending=False)
        hundreds_both_df['Date'] = hundreds_both_df['Date'].dt.strftime('%d/%m/%Y')

        hundreds_both_df = hundreds_both_df[
            ['Name', 'Bat Team', 'Bowl Team', 'Country', '1st Innings', '2nd Innings', 
            'Balls', '4s', '6s', 'Date']
        ]

        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Hundred in Each Innings</h3>", 
                unsafe_allow_html=True)
        st.dataframe(hundreds_both_df, 
                    use_container_width=True, 
                    hide_index=True)

        # Position Records and Centuries Analysis
        position_scores_df = (bat_df.sort_values('Runs', ascending=False)
                             .groupby('Position')
                             .agg({
                                 'Name': 'first',
                                 'Runs': 'first',
                                 'Balls': 'first',
                                 'How Out': 'first',
                                 '4s': 'first',
                                 '6s': 'first'
                             })
                             .reset_index()
                             .sort_values('Position')
                             [['Position', 'Name', 'Runs', 'Balls', 'How Out', '4s', '6s']])

        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Highest Score at Each Position</h3>", 
                   unsafe_allow_html=True)
        st.dataframe(position_scores_df, 
                    use_container_width=True, 
                    hide_index=True,
                    height=430)

        # Centuries Analysis Plot
        centuries_df = bat_df[bat_df['Runs'] >= 100].copy()
        scatter_fig = go.Figure()

        for name in centuries_df['Name'].unique():
            player_data = centuries_df[centuries_df['Name'] == name]
            scatter_fig.add_trace(go.Scatter(
                x=player_data['Balls'],
                y=player_data['Runs'],
                mode='markers+text',
                text=player_data['Name'],
                textposition='top center',
                marker=dict(size=10),
                name=name,
                hovertemplate=(
                    f"<b>{name}</b><br><br>"
                    "Runs: {y}<br>"
                    "Balls: {x}<br>"
                    f"4s: {player_data['4s'].iloc[0]}<br>"
                    f"6s: {player_data['6s'].iloc[0]}<br>"
                    "<extra></extra>"
                )
            ))

        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Centuries Analysis (Runs vs Balls)</h3>",
                    unsafe_allow_html=True)

        scatter_fig.update_layout(
            xaxis_title="Balls Faced",
            yaxis_title="Runs Scored",
            height=700,
            font=dict(size=12),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )

        st.plotly_chart(scatter_fig, use_container_width=True)

###############################################################################
# Section 5: Bowling Records Tab
###############################################################################
with tab2:
    if 'bowl_df' in st.session_state:
        bowl_df = st.session_state['bowl_df']
        bowl_df['Date'] = bowl_df['Date'].apply(parse_date)
        
        # Apply format filter to bowl_df first
        bowl_df = filter_by_format(bowl_df)
        
        # Best Bowling Figures
        best_bowling_df = (bowl_df[bowl_df['Bowler_Wkts'] >= 5]
                          .sort_values(by=['Bowler_Wkts', 'Bowler_Runs'], 
                                     ascending=[False, True])
                          [['Innings', 'Position', 'Name', 'Bowler_Overs', 'Maidens', 
                            'Bowler_Runs', 'Bowler_Wkts', 'Bowler_Econ', 'Bat_Team',
                            'Bowl_Team', 'Home_Team', 'Match_Format', 'Date']]
                          .rename(columns={
                              'Bowler_Overs': 'Overs',
                              'Bowler_Runs': 'Runs',
                              'Bowler_Wkts': 'Wickets',
                              'Bowler_Econ': 'Econ',
                              'Bat_Team': 'Bat Team',
                              'Bowl_Team': 'Bowl Team',
                              'Home_Team': 'Country',
                              'Match_Format': 'Format'
                          }))
        
        best_bowling_df['Date'] = best_bowling_df['Date'].dt.strftime('%d/%m/%Y')
        
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Best Bowling Figures </h3>", 
                   unsafe_allow_html=True)
        st.dataframe(best_bowling_df, 
                    use_container_width=True, 
                    hide_index=True)

        # 5+ Wickets in Both Innings (using filtered bowl_df)
        bowl_5w = (bowl_df.groupby(['Name', 'Date', 'Home_Team'])
                    .agg({
                        'Bowler_Wkts': lambda x: [
                            max([w for w, i in zip(x, bowl_df.loc[x.index, 'Innings']) if i in [1, 2]], default=0),
                            max([w for w, i in zip(x, bowl_df.loc[x.index, 'Innings']) if i in [3, 4]], default=0)
                        ],
                        'Bowler_Runs': lambda x: [
                            min([r for r, i, w in zip(x, bowl_df.loc[x.index, 'Innings'], bowl_df.loc[x.index, 'Bowler_Wkts']) 
                                if i in [1, 2] and w >= 5], default=0),
                            min([r for r, i, w in zip(x, bowl_df.loc[x.index, 'Innings'], bowl_df.loc[x.index, 'Bowler_Wkts']) 
                                if i in [3, 4] and w >= 5], default=0)
                        ],
                        'Bowler_Overs': lambda x: [
                            [o for o, i, w in zip(x, bowl_df.loc[x.index, 'Innings'], bowl_df.loc[x.index, 'Bowler_Wkts']) 
                             if i in [1, 2] and w >= 5][0] if any([w >= 5 for w, i in zip(bowl_df.loc[x.index, 'Bowler_Wkts'], 
                                                                                          bowl_df.loc[x.index, 'Innings']) if i in [1, 2]]) else 0,
                            [o for o, i, w in zip(x, bowl_df.loc[x.index, 'Innings'], bowl_df.loc[x.index, 'Bowler_Wkts']) 
                             if i in [3, 4] and w >= 5][0] if any([w >= 5 for w, i in zip(bowl_df.loc[x.index, 'Bowler_Wkts'], 
                                                                                          bowl_df.loc[x.index, 'Innings']) if i in [3, 4]]) else 0
                        ],
                        'Bat_Team': 'first',
                        'Bowl_Team': 'first',
                        'Match_Format': 'first'
                    })
                    .reset_index())
        
        # Rest of the code remains the same as it's working with the filtered data
        bowl_5w['1st Innings Wkts'] = bowl_5w['Bowler_Wkts'].str[0]
        bowl_5w['2nd Innings Wkts'] = bowl_5w['Bowler_Wkts'].str[1]
        bowl_5w['1st Innings Runs'] = bowl_5w['Bowler_Runs'].str[0]
        bowl_5w['2nd Innings Runs'] = bowl_5w['Bowler_Runs'].str[1]
        bowl_5w['1st Innings Overs'] = bowl_5w['Bowler_Overs'].str[0]
        bowl_5w['2nd Innings Overs'] = bowl_5w['Bowler_Overs'].str[1]
        
        five_wickets_both_df = (bowl_5w[
            (bowl_5w['1st Innings Wkts'] >= 5) & 
            (bowl_5w['2nd Innings Wkts'] >= 5)
        ]
        .drop(columns=['Bowler_Wkts', 'Bowler_Runs', 'Bowler_Overs'])
        .rename(columns={
            'Home_Team': 'Country',
            'Bat_Team': 'Bat Team',
            'Bowl_Team': 'Bowl Team',
            'Match_Format': 'Format'
        })
        [['Name', 'Bat Team', 'Bowl Team', 'Country', 'Format',
          '1st Innings Wkts', '1st Innings Runs', '1st Innings Overs',
          '2nd Innings Wkts', '2nd Innings Runs', '2nd Innings Overs', 
          'Date']]
        .sort_values(by='Date', ascending=False))
        
        five_wickets_both_df['Date'] = five_wickets_both_df['Date'].dt.strftime('%d/%m/%Y')
        
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>5+ Wickets in Both Innings</h3>", 
                   unsafe_allow_html=True)
        st.dataframe(five_wickets_both_df, 
                    use_container_width=True, 
                    hide_index=True)

        # Best Match Bowling Figures (using filtered bowl_df)
        match_bowling_df = (bowl_df.groupby(['Name', 'Date', 'Home_Team', 'Bat_Team', 'Bowl_Team', 'Match_Format'])
                           .agg({
                               'Bowler_Wkts': lambda x: [
                                   max([w for w, i in zip(x, bowl_df.loc[x.index, 'Innings']) if i in [1, 2]] or [0]),
                                   max([w for w, i in zip(x, bowl_df.loc[x.index, 'Innings']) if i in [3, 4]] or [0])
                               ],
                               'Bowler_Runs': lambda x: [
                                   sum([r for r, i in zip(x, bowl_df.loc[x.index, 'Innings']) if i in [1, 2]] or [0]),
                                   sum([r for r, i in zip(x, bowl_df.loc[x.index, 'Innings']) if i in [3, 4]] or [0])
                               ]
                           })
                           .reset_index())
        
        match_bowling_df['1st Innings Wkts'] = match_bowling_df['Bowler_Wkts'].str[0]
        match_bowling_df['2nd Innings Wkts'] = match_bowling_df['Bowler_Wkts'].str[1]
        match_bowling_df['1st Innings Runs'] = match_bowling_df['Bowler_Runs'].str[0]
        match_bowling_df['2nd Innings Runs'] = match_bowling_df['Bowler_Runs'].str[1]
        
        match_bowling_df['Match Wickets'] = match_bowling_df['1st Innings Wkts'] + match_bowling_df['2nd Innings Wkts']
        match_bowling_df['Match Runs'] = match_bowling_df['1st Innings Runs'] + match_bowling_df['2nd Innings Runs']
        
        best_match_bowling_df = (match_bowling_df
            .sort_values(by=['Match Wickets', 'Match Runs'], 
                        ascending=[False, True])
            .rename(columns={
                'Home_Team': 'Country',
                'Bat_Team': 'Bat Team',
                'Bowl_Team': 'Bowl Team',
                'Match_Format': 'Format'
            })
            [['Name', 'Bat Team', 'Bowl Team', 'Country', 'Format',
              '1st Innings Wkts', '1st Innings Runs',
              '2nd Innings Wkts', '2nd Innings Runs',
              'Match Wickets', 'Match Runs', 'Date']]
        )
        
        best_match_bowling_df['Date'] = best_match_bowling_df['Date'].dt.strftime('%d/%m/%Y')
        
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Best Match Bowling Figures</h3>", 
                   unsafe_allow_html=True)
        st.dataframe(best_match_bowling_df, 
                    use_container_width=True, 
                    hide_index=True)
        
###############################################################################
# Section 6: Match Records Tab
###############################################################################
with tab3:
    if 'match_df' in st.session_state:
        match_df = st.session_state['match_df']
        match_df['Date'] = match_df['Date'].apply(parse_date)
        
        # Apply format filter to match_df first
        match_df = filter_by_format(match_df)
        
        # Biggest Wins (Runs)
        bigwin_df = (match_df[
            (match_df['Margin_Runs'] > 0) & 
            (match_df['Innings_Win'] == 0) & 
            (match_df['Home_Drawn'] != 1)
        ].copy())
        
        bigwin_df['Win Team'] = np.where(
            bigwin_df['Home_Win'] == 1,
            bigwin_df['Home_Team'],
            bigwin_df['Away_Team']
        )
        
        bigwin_df['Opponent'] = np.where(
            bigwin_df['Home_Win'] == 1,
            bigwin_df['Away_Team'],
            bigwin_df['Home_Team']
        )
        
        bigwin_df = (bigwin_df
            .sort_values('Margin_Runs', ascending=False)
            [['Date', 'Win Team', 'Opponent', 'Margin_Runs', 'Match_Result', 'Match_Format']]
            .rename(columns={
                'Margin_Runs': 'Runs',
                'Match_Result': 'Match Result',
                'Match_Format': 'Format'
            }))
        
        bigwin_df['Date'] = bigwin_df['Date'].dt.strftime('%d/%m/%Y')
        
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Biggest Wins (Runs)</h3>", 
                   unsafe_allow_html=True)
        st.dataframe(bigwin_df, 
                    use_container_width=True, 
                    hide_index=True)

        # Biggest Wins (Wickets) - using filtered match_df
        bigwin_wickets_df = (match_df[
            (match_df['Margin_Wickets'] > 0) & 
            (match_df['Innings_Win'] == 0) & 
            (match_df['Home_Drawn'] != 1)
        ].copy())
        
        bigwin_wickets_df['Win Team'] = np.where(
            bigwin_wickets_df['Home_Win'] == 1,
            bigwin_wickets_df['Home_Team'],
            bigwin_wickets_df['Away_Team']
        )
        
        bigwin_wickets_df['Opponent'] = np.where(
            bigwin_wickets_df['Home_Win'] == 1,
            bigwin_wickets_df['Away_Team'],
            bigwin_wickets_df['Home_Team']
        )
        
        bigwin_wickets_df = (bigwin_wickets_df
            .sort_values('Margin_Wickets', ascending=False)
            [['Date', 'Win Team', 'Opponent', 'Margin_Wickets', 'Match_Result', 'Match_Format']]
            .rename(columns={
                'Margin_Wickets': 'Wickets',
                'Match_Result': 'Match Result',
                'Match_Format': 'Format'
            }))
        
        bigwin_wickets_df['Date'] = bigwin_wickets_df['Date'].dt.strftime('%d/%m/%Y')
        
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Biggest Wins (Wickets)</h3>", 
                   unsafe_allow_html=True)
        st.dataframe(bigwin_wickets_df, 
                    use_container_width=True, 
                    hide_index=True)

        # Innings Wins - using filtered match_df
        bigwin_innings_df = (match_df[
            (match_df['Innings_Win'] == 1) & 
            (match_df['Home_Drawn'] != 1)
        ].copy())
        
        bigwin_innings_df['Win Team'] = np.where(
            bigwin_innings_df['Home_Win'] == 1,
            bigwin_innings_df['Home_Team'],
            bigwin_innings_df['Away_Team']
        )
        
        bigwin_innings_df['Opponent'] = np.where(
            bigwin_innings_df['Home_Win'] == 1,
            bigwin_innings_df['Away_Team'],
            bigwin_innings_df['Home_Team']
        )
        
        bigwin_innings_df = (bigwin_innings_df
            .sort_values('Margin_Runs', ascending=False)
            [['Date', 'Win Team', 'Opponent', 'Margin_Runs', 'Innings_Win', 'Match_Result', 'Match_Format']]
            .rename(columns={
                'Margin_Runs': 'Runs',
                'Innings_Win': 'Innings',
                'Match_Result': 'Match Result',
                'Match_Format': 'Format'
            }))
        
        bigwin_innings_df['Date'] = bigwin_innings_df['Date'].dt.strftime('%d/%m/%Y')
        
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Biggest Wins (Innings)</h3>", 
                   unsafe_allow_html=True)
        st.dataframe(bigwin_innings_df, 
                    use_container_width=True, 
                    hide_index=True)
        
        # Narrowest Wins (Runs) - using filtered match_df
        narrow_wins_df = (match_df[
            (match_df['Margin_Runs'] > 0) & 
            (match_df['Innings_Win'] == 0) & 
            (match_df['Home_Drawn'] != 1)
        ].copy())
        
        narrow_wins_df['Win Team'] = np.where(
            narrow_wins_df['Home_Win'] == 1,
            narrow_wins_df['Home_Team'],
            narrow_wins_df['Away_Team']
        )
        
        narrow_wins_df['Opponent'] = np.where(
            narrow_wins_df['Home_Win'] == 1,
            narrow_wins_df['Away_Team'],
            narrow_wins_df['Home_Team']
        )
        
        narrow_wins_df = (narrow_wins_df
            .sort_values('Margin_Runs', ascending=True)
            [['Date', 'Win Team', 'Opponent', 'Margin_Runs', 'Match_Result', 'Match_Format']]
            .rename(columns={
                'Margin_Runs': 'Runs',
                'Match_Result': 'Match Result',
                'Match_Format': 'Format'
            }))
        
        narrow_wins_df['Date'] = narrow_wins_df['Date'].dt.strftime('%d/%m/%Y')
        
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Narrowest Wins (Runs)</h3>", 
                   unsafe_allow_html=True)
        st.dataframe(narrow_wins_df, 
                    use_container_width=True, 
                    hide_index=True)
        
        # Narrowest Wins (Wickets) - using filtered match_df
        narrowwin_wickets_df = (match_df[
            (match_df['Margin_Wickets'] > 0) & 
            (match_df['Innings_Win'] == 0) & 
            (match_df['Home_Drawn'] != 1)
        ].copy())
        
        narrowwin_wickets_df['Win Team'] = np.where(
            narrowwin_wickets_df['Home_Win'] == 1,
            narrowwin_wickets_df['Home_Team'],
            narrowwin_wickets_df['Away_Team']
        )
        
        narrowwin_wickets_df['Opponent'] = np.where(
            narrowwin_wickets_df['Home_Win'] == 1,
            narrowwin_wickets_df['Away_Team'],
            narrowwin_wickets_df['Home_Team']
        )
        
        narrowwin_wickets_df = (narrowwin_wickets_df
            .sort_values('Margin_Wickets', ascending=True)
            [['Date', 'Win Team', 'Opponent', 'Margin_Wickets', 'Match_Result', 'Match_Format']]
            .rename(columns={
                'Margin_Wickets': 'Wickets',
                'Match_Result': 'Match Result',
                'Match_Format': 'Format'
            }))
        
        narrowwin_wickets_df['Date'] = narrowwin_wickets_df['Date'].dt.strftime('%d/%m/%Y')
        
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Narrowest Wins (Wickets)</h3>", 
                   unsafe_allow_html=True)
        st.dataframe(narrowwin_wickets_df, 
                    use_container_width=True, 
                    hide_index=True)
        
###############################################################################
# Section 7: Game Records Tab
###############################################################################
with tab4:
    if 'game_df' in st.session_state and 'match_df' in st.session_state:
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Game Records</h3>", unsafe_allow_html=True)
        
        try:
            # Create copies of both dataframes
            gamerecords_df = st.session_state['game_df'].copy()
            matchgame_df = st.session_state['match_df'].copy()
            
            # Apply format filter to both dataframes
            gamerecords_df = filter_by_format(gamerecords_df)
            matchgame_df = filter_by_format(matchgame_df)
            
            # Convert Date column in gamerecords_df
            gamerecords_df['Date'] = pd.to_datetime(gamerecords_df['Date'])
            
            # Create Result column with margin details and additional columns
            def get_match_details(row):
                bat_team = row['Bat_Team']
                bowl_team = row['Bowl_Team']
                result = 'Unknown'
                margin_info = '-'
                innings_win = 0
                margin_runs = None
                margin_wickets = None
                
                # Check for matches where bat_team is home and bowl_team is away
                home_match = matchgame_df[
                    (matchgame_df['Home_Team'] == bat_team) & 
                    (matchgame_df['Away_Team'] == bowl_team)
                ]
                
                # Check for matches where bat_team is away and bowl_team is home
                away_match = matchgame_df[
                    (matchgame_df['Away_Team'] == bat_team) & 
                    (matchgame_df['Home_Team'] == bowl_team)
                ]
                
                if not home_match.empty:
                    match = home_match.iloc[0]
                    if match['Home_Win'] == 1:
                        result = 'Win'
                    elif match['Home_Lost'] == 1:
                        result = 'Lost'
                    elif match['Home_Drawn'] == 1:
                        result = 'Draw'
                    
                    # Get margin details
                    innings_win = match['Innings_Win']
                    margin_runs = match['Margin_Runs'] if pd.notna(match['Margin_Runs']) else None
                    margin_wickets = match['Margin_Wickets'] if pd.notna(match['Margin_Wickets']) else None
                    
                elif not away_match.empty:
                    match = away_match.iloc[0]
                    if match['Away_Won'] == 1:
                        result = 'Win'
                    elif match['Away_Lost'] == 1:
                        result = 'Lost'
                    elif match['Away_Drawn'] == 1:
                        result = 'Draw'
                    
                    # Get margin details
                    innings_win = match['Innings_Win']
                    margin_runs = match['Margin_Runs'] if pd.notna(match['Margin_Runs']) else None
                    margin_wickets = match['Margin_Wickets'] if pd.notna(match['Margin_Wickets']) else None
                
                # Add margin details
                if innings_win == 1:
                    margin_info = 'by an innings'
                    if pd.notna(margin_runs) and margin_runs > 0:
                        margin_info += f" and {int(margin_runs)} runs"
                elif pd.notna(margin_wickets) and margin_wickets > 0:
                    margin_info = f"by {int(margin_wickets)} wickets"
                elif pd.notna(margin_runs) and margin_runs > 0:
                    margin_info = f"by {int(margin_runs)} runs"
                
                return pd.Series([result, margin_info, innings_win, margin_runs, margin_wickets], 
                               index=['Result', 'Margin', 'Innings_Win', 'Margin_Runs', 'Margin_Wickets'])
            
            # Apply the function to create all columns
            gamerecords_df[['Result', 'Margin', 'Innings_Win', 'Margin_Runs', 'Margin_Wickets']] = \
                gamerecords_df.apply(get_match_details, axis=1)
            
            # Sort by date (descending) and then Innings (ascending)
            gamerecords_df = gamerecords_df.sort_values(
                by=['Date', 'Innings'], 
                ascending=[False, True]
            )
##################====================HIGH SCORES===================##################
            # Create Highest Scores Table
            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Highest Team Scores</h3>", unsafe_allow_html=True)

            # Create copy and select columns
            highest_scores_df = gamerecords_df[['Bat_Team', 'Bowl_Team', 'Total_Runs', 'Wickets', 
                                              'Overs', 'Run_Rate', 'Competition', 'Match_Format', 'Date']].copy()

            highest_scores_df['Date'] = highest_scores_df['Date'].dt.date

            # Rename columns
            highest_scores_df.columns = ['Bat Team', 'Bowl Team', 'Runs', 'Wickets', 'Overs', 
                                       'Run Rate', 'Competition', 'Format', 'Date']

            # Sort by runs (highest to lowest)
            highest_scores_df = highest_scores_df.sort_values('Runs', ascending=False)

            # Display the dataframe
            st.dataframe(highest_scores_df, use_container_width=True, hide_index=True)

##################====================LOW SCORES===================##################
            # Create Lowest Scores Table
            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Lowest Team Scores (All Out)</h3>", unsafe_allow_html=True)

            # Create copy and select columns
            lowest_scores_df = gamerecords_df[['Bat_Team', 'Bowl_Team', 'Total_Runs', 'Wickets', 
                                             'Overs', 'Run_Rate', 'Competition', 'Match_Format', 'Date']].copy()

            # Filter for all out innings
            lowest_scores_df = lowest_scores_df[lowest_scores_df['Wickets'] == 10]

            # Format date to remove time
            lowest_scores_df['Date'] = lowest_scores_df['Date'].dt.date

            # Rename columns
            lowest_scores_df.columns = ['Bat Team', 'Bowl Team', 'Runs', 'Wickets', 'Overs', 
                                      'Run Rate', 'Competition', 'Format', 'Date']

            # Sort by runs (lowest to highest)
            lowest_scores_df = lowest_scores_df.sort_values('Runs', ascending=True)

            # Display the dataframe
            st.dataframe(lowest_scores_df, use_container_width=True, hide_index=True)

##################====================HIGH CHASES===================##################
            # Create Highest Run Chases Table
            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Highest Successful Run Chases</h3>", unsafe_allow_html=True)

            # Create copy and select columns
            chases_df = gamerecords_df[['Bat_Team', 'Bowl_Team', 'Total_Runs', 'Wickets', 
                                    'Overs', 'Run_Rate', 'Competition', 'Match_Format', 
                                    'Date', 'Innings', 'Result', 'Margin']].copy()

            # Filter for successful chases based on format and exclude all-out innings
            successful_chases = (
                ((chases_df['Match_Format'].isin(['First Class', 'Test Match'])) & 
                 (chases_df['Innings'] == 4) & 
                 (chases_df['Result'] == 'Win') &
                 (chases_df['Wickets'] != 10)) |
                ((chases_df['Match_Format'].isin(['ODI', 'T20I', 'One Day', 'T20', '20'])) & 
                 (chases_df['Innings'] == 2) & 
                 (chases_df['Result'] == 'Win') &
                 (chases_df['Wickets'] != 10))
            )
            # Apply filter
            chases_df = chases_df[successful_chases]

            # Format date to remove time
            chases_df['Date'] = chases_df['Date'].dt.date

            def format_margin_wickets(row):
                if pd.notna(row['Margin']) and 'wicket' in str(row['Margin']).lower():
                    return row['Margin']
                elif pd.notna(row['Margin']):
                    wickets_left = 10 - row['Wickets']
                    return f"by {wickets_left} wickets"
                return "-"

            # Add formatted margin
            chases_df['Margin_Display'] = chases_df.apply(format_margin_wickets, axis=1)

            # Rename columns
            chases_df.columns = ['Bat Team', 'Bowl Team', 'Runs', 'Wickets', 'Overs', 
                                'Run Rate', 'Competition', 'Format', 'Date', 'Innings', 'Result', 'Margin', 'Margin_Display']

            # Sort by runs (highest to lowest)
            chases_df = chases_df.sort_values('Runs', ascending=False)

            # Select final display columns
            chases_df = chases_df[['Bat Team', 'Bowl Team', 'Innings', 'Runs', 'Wickets', 'Overs', 
                                   'Run Rate', 'Competition', 'Format', 'Date', 'Margin_Display']]

            # Rename the Margin_Display column to just Margin
            chases_df.columns = ['Bat Team', 'Bowl Team', 'Innings', 'Runs', 'Wickets', 'Overs', 
                                 'Run Rate', 'Competition', 'Format', 'Date', 'Margin']

            # Display the dataframe
            st.dataframe(chases_df, use_container_width=True, hide_index=True)

##################====================LOW 1ST INNINGS===================##################
            # Create Lowest First Innings Wins Table
            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Lowest First Innings Winning Scores</h3>", unsafe_allow_html=True)

            # Create copy and select columns
            low_wins_df = gamerecords_df[['Bat_Team', 'Bowl_Team', 'Total_Runs', 'Wickets', 
                                         'Overs', 'Run_Rate', 'Competition', 'Match_Format', 
                                         'Date', 'Innings', 'Result', 'Margin']].copy()

            # Filter for first innings wins
            first_innings_wins = (
                (low_wins_df['Innings'] == 1) & 
                (low_wins_df['Result'] == 'Win')
            )

            # Apply filter
            low_wins_df = low_wins_df[first_innings_wins]

            # Format date to remove time
            low_wins_df['Date'] = low_wins_df['Date'].dt.date

            # Rename columns
            low_wins_df.columns = ['Bat Team', 'Bowl Team', 'Runs', 'Wickets', 'Overs', 
                                  'Run Rate', 'Competition', 'Format', 'Date', 'Innings', 'Result', 'Margin']

            # Sort by runs (lowest to highest)
            low_wins_df = low_wins_df.sort_values('Runs', ascending=True)

            # Select final display columns
            low_wins_df = low_wins_df[['Bat Team', 'Bowl Team', 'Innings', 'Runs', 'Wickets', 'Overs', 
                                      'Run Rate', 'Competition', 'Format', 'Date', 'Margin']]

            # Display the dataframe
            st.dataframe(low_wins_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error processing records: {str(e)}")


    else:
        st.info("No game or match records available.")