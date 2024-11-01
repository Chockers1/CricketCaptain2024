# Section 1: Imports, Setup, and Cache Functions
###############################################################################
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

@st.cache_data
def parse_date(date_str):
    """Helper function to parse dates in multiple formats"""
    try:
        for fmt in ['%d/%m/%Y', '%d %b %Y', '%Y-%m-%d']:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except ValueError:
                continue
        return pd.to_datetime(date_str)
    except Exception:
        return pd.NaT

@st.cache_data
def process_dataframes():
    """Process all dataframes once and cache the results"""
    if 'bat_df' in st.session_state:
        bat_df = st.session_state['bat_df'].copy()
        bat_df['Date'] = pd.to_datetime(bat_df['Date'])
    else:
        bat_df = None

    if 'bowl_df' in st.session_state:
        bowl_df = st.session_state['bowl_df'].copy()
        bowl_df['Date'] = pd.to_datetime(bowl_df['Date'])
    else:
        bowl_df = None

    if 'match_df' in st.session_state:
        match_df = st.session_state['match_df'].copy()
        match_df['Date'] = pd.to_datetime(match_df['Date'])
    else:
        match_df = None

    if 'game_df' in st.session_state:
        game_df = st.session_state['game_df'].copy()
        game_df['Date'] = pd.to_datetime(game_df['Date'])
    else:
        game_df = None

    return bat_df, bowl_df, match_df, game_df

@st.cache_data
def filter_by_format(df, format_choice):
    if df is not None and 'All' not in format_choice:
        return df[df['Match_Format'].isin(format_choice)]
    return df

# Section 2: Page Header, Styling, and Format Selection
###############################################################################
def init_page():
   # Title
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

@st.cache_data
def get_formats():
   """Get unique formats from all dataframes"""
   all_formats = set()
   
   for df_name in ['game_df', 'bat_df', 'bowl_df', 'match_df']:
       if df_name in st.session_state:
           df = st.session_state[df_name]
           if 'Match_Format' in df.columns:
               all_formats.update(df['Match_Format'].unique())
   
   return ['All'] + sorted(list(all_formats))

def initialize_data():
   """Initialize main data and format filter"""
   # Process all dataframes
   bat_df, bowl_df, match_df, game_df = process_dataframes()
   
   # Get formats and create filter
   formats = get_formats()
   format_choice = st.multiselect('Format:', formats, default='All', key='global_format_filter')
   
   # Apply format filter to all dataframes
   filtered_bat_df = filter_by_format(bat_df, format_choice)
   filtered_bowl_df = filter_by_format(bowl_df, format_choice)
   filtered_match_df = filter_by_format(match_df, format_choice)
   filtered_game_df = filter_by_format(game_df, format_choice)
   
   return filtered_bat_df, filtered_bowl_df, filtered_match_df, filtered_game_df

# Initialize page and data
init_page()
filtered_bat_df, filtered_bowl_df, filtered_match_df, filtered_game_df = initialize_data()

# Create tabs
tab_names = ["Batting Records", "Bowling Records", "Match Records", "Game Records"]
tabs = st.tabs(tab_names)
###############################################################################
# Section 3: Create Tabs
###############################################################################
#tab1, tab2, tab3, tab4 = st.tabs(["Batting Records", "Bowling Records", "Match Records", "Game Records"])

###############################################################################
# Section 3: Create Tabs and Cache Functions for Batting Records
###############################################################################
###############################################################################
# Section 4: Batting Records Tab Functions
###############################################################################
@st.cache_data
def process_highest_scores(filtered_bat_df):
    """Process highest scores data"""
    if filtered_bat_df is None or filtered_bat_df.empty:
        return pd.DataFrame()
        
    columns_to_drop = [
        'Bat_Team_x', 'Bowl_Team_x', 'File Name', 'Total_Runs', 
        'Overs', 'Wickets', 'Competition', 'Batted', 'Out',
        'Not Out', '50s', '100s', '200s', '<25&Out', 
        'Caught', 'Bowled', 'LBW', 'Stumped', 'Run Out',
        'Boundary Runs', 'Team Balls', 'Year', 'Player_of_the_Match'
    ]

    df = (filtered_bat_df[filtered_bat_df['Runs'] >= 100]
          .drop(columns=columns_to_drop, errors='ignore')
          .sort_values(by='Runs', ascending=False)
          .rename(columns={
              'Bat_Team_y': 'Bat Team',
              'Bowl_Team_y': 'Bowl Team'
          }))
    
    df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')
    return df

@st.cache_data
def process_not_out_99s(filtered_bat_df):
    """Process 99 not out data"""
    if filtered_bat_df is None or filtered_bat_df.empty:
        return pd.DataFrame()
        
    columns_to_drop = [
        'Bat_Team_x', 'Bowl_Team_x', 'File Name', 'Total_Runs', 
        'Overs', 'Wickets', 'Competition', 'Batted', 'Out',
        'Not Out', '50s', '100s', '200s', '<25&Out', 
        'Caught', 'Bowled', 'LBW', 'Stumped', 'Run Out',
        'Boundary Runs', 'Team Balls', 'Year', 'Player_of_the_Match'
    ]

    df = (filtered_bat_df[(filtered_bat_df['Runs'] == 99) & (filtered_bat_df['Not Out'] == 1)]
          .drop(columns=columns_to_drop, errors='ignore')
          .sort_values(by='Runs', ascending=False)
          .rename(columns={
              'Bat_Team_y': 'Bat Team',
              'Bowl_Team_y': 'Bowl Team'
          }))
    
    df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')
    return df

@st.cache_data
def process_carrying_bat(filtered_bat_df):
    """Process carrying the bat data"""
    if filtered_bat_df is None or filtered_bat_df.empty:
        return pd.DataFrame()
        
    columns_to_drop = [
        'Bat_Team_x', 'Bowl_Team_x', 'File Name', 'Total_Runs', 
        'Overs', 'Wickets', 'Competition', 'Batted', 'Out',
        'Not Out', '50s', '100s', '200s', '<25&Out', 
        'Caught', 'Bowled', 'LBW', 'Stumped', 'Run Out',
        'Boundary Runs', 'Team Balls', 'Year', 'Player_of_the_Match'
    ]

    df = (filtered_bat_df[
        (filtered_bat_df['Position'].isin([1, 2])) & 
        (filtered_bat_df['Not Out'] == 1) & 
        (filtered_bat_df['Wickets'] == 10)]
        .drop(columns=columns_to_drop, errors='ignore')
        .sort_values(by='Runs', ascending=False)
        .rename(columns={
            'Bat_Team_y': 'Bat Team',
            'Bowl_Team_y': 'Bowl Team'
        }))
    
    df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')
    return df

@st.cache_data
def process_hundreds_both_innings(filtered_bat_df):
    """Process hundreds in both innings data"""
    if filtered_bat_df is None or filtered_bat_df.empty:
        return pd.DataFrame()
        
    bat_match_100_df = (filtered_bat_df.groupby(['Name', 'Date', 'Home Team'])
                .agg({
                    'Runs': lambda x: [
                        max([r for r, i in zip(x, filtered_bat_df.loc[x.index, 'Innings']) if i in [1, 2]], default=0),
                        max([r for r, i in zip(x, filtered_bat_df.loc[x.index, 'Innings']) if i in [3, 4]], default=0)
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

    hundreds_both_df = hundreds_both_df[
        ['Name', 'Bat Team', 'Bowl Team', 'Country', '1st Innings', '2nd Innings', 
         'Balls', '4s', '6s', 'Date']
    ]
    
    hundreds_both_df['Date'] = pd.to_datetime(hundreds_both_df['Date'])
    hundreds_both_df = hundreds_both_df.sort_values(by='Date', ascending=False)
    hundreds_both_df['Date'] = hundreds_both_df['Date'].dt.strftime('%d/%m/%Y')
    
    return hundreds_both_df

@st.cache_data
def process_position_scores(filtered_bat_df):
    """Process highest scores by position data"""
    if filtered_bat_df is None or filtered_bat_df.empty:
        return pd.DataFrame()
        
    return (filtered_bat_df.sort_values('Runs', ascending=False)
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

@st.cache_data
def create_centuries_plot(filtered_bat_df):
    """Create centuries scatter plot"""
    if filtered_bat_df is None or filtered_bat_df.empty:
        return None
        
    centuries_df = filtered_bat_df[filtered_bat_df['Runs'] >= 100].copy()
    fig = go.Figure()

    for name in centuries_df['Name'].unique():
        player_data = centuries_df[centuries_df['Name'] == name]
        fig.add_trace(go.Scatter(
            x=player_data['Balls'],
            y=player_data['Runs'],
            mode='markers+text',
            text=player_data['Name'],
            textposition='top center',
            marker=dict(size=10),
            name=name,
            hovertemplate=(
                f"<b>{name}</b><br><br>"
                "Runs: %{y}<br>"
                "Balls: %{x}<br>"
                f"4s: {player_data['4s'].iloc[0]}<br>"
                f"6s: {player_data['6s'].iloc[0]}<br>"
                "<extra></extra>"
            )
        ))

    fig.update_layout(
        xaxis_title="Balls Faced",
        yaxis_title="Runs Scored",
        height=700,
        font=dict(size=12),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )

    return fig

###############################################################################
# Section 4: Batting Records Tab Display
###############################################################################
with tabs[0]:
    if filtered_bat_df is not None and not filtered_bat_df.empty:
        # Highest Scores
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Highest Scores</h3>", 
                   unsafe_allow_html=True)
        highest_scores_df = process_highest_scores(filtered_bat_df)
        if not highest_scores_df.empty:
            st.dataframe(highest_scores_df, use_container_width=True, hide_index=True)

        # 99 Not Out Club
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>99 Not Out Club</h3>", 
                   unsafe_allow_html=True)
        not_out_99s_df = process_not_out_99s(filtered_bat_df)
        if not not_out_99s_df.empty:
            st.dataframe(not_out_99s_df, use_container_width=True, hide_index=True)

        # Carrying the Bat
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Carrying the Bat</h3>", 
                   unsafe_allow_html=True)
        carrying_bat_df = process_carrying_bat(filtered_bat_df)
        if not carrying_bat_df.empty:
            st.dataframe(carrying_bat_df, use_container_width=True, hide_index=True)

        # Hundred in Each Innings
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Hundred in Each Innings</h3>", 
                   unsafe_allow_html=True)
        hundreds_both_df = process_hundreds_both_innings(filtered_bat_df)
        if not hundreds_both_df.empty:
            st.dataframe(hundreds_both_df, use_container_width=True, hide_index=True)

        # Position Records
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Highest Score at Each Position</h3>", 
                   unsafe_allow_html=True)
        position_scores_df = process_position_scores(filtered_bat_df)
        if not position_scores_df.empty:
            st.dataframe(position_scores_df, use_container_width=True, 
                        hide_index=True, height=430)

        # Centuries Analysis Plot
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Centuries Analysis (Runs vs Balls)</h3>",
                    unsafe_allow_html=True)
        centuries_fig = create_centuries_plot(filtered_bat_df)
        if centuries_fig:
            st.plotly_chart(centuries_fig, use_container_width=True)
    else:
        st.info("No batting records available.")

###############################################################################
# Section 5: Bowling Records Functions
###############################################################################
@st.cache_data
def process_best_bowling(filtered_bowl_df):
    """Process best bowling figures data"""
    if filtered_bowl_df is None or filtered_bowl_df.empty:
        return pd.DataFrame()
    
    df = (filtered_bowl_df[filtered_bowl_df['Bowler_Wkts'] >= 5]
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
    
    df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')
    return df

@st.cache_data
def process_five_wickets_both(filtered_bowl_df):
    """Process 5+ wickets in both innings data"""
    if filtered_bowl_df is None or filtered_bowl_df.empty:
        return pd.DataFrame()
    
    bowl_5w = (filtered_bowl_df.groupby(['Name', 'Date', 'Home_Team'])
              .agg({
                  'Bowler_Wkts': lambda x: [
                      max([w for w, i in zip(x, filtered_bowl_df.loc[x.index, 'Innings']) 
                          if i in [1, 2]], default=0),
                      max([w for w, i in zip(x, filtered_bowl_df.loc[x.index, 'Innings']) 
                          if i in [3, 4]], default=0)
                  ],
                  'Bowler_Runs': lambda x: [
                      min([r for r, i, w in zip(x, filtered_bowl_df.loc[x.index, 'Innings'], 
                          filtered_bowl_df.loc[x.index, 'Bowler_Wkts']) 
                          if i in [1, 2] and w >= 5], default=0),
                      min([r for r, i, w in zip(x, filtered_bowl_df.loc[x.index, 'Innings'], 
                          filtered_bowl_df.loc[x.index, 'Bowler_Wkts']) 
                          if i in [3, 4] and w >= 5], default=0)
                  ],
                  'Bowler_Overs': lambda x: [
                      [o for o, i, w in zip(x, filtered_bowl_df.loc[x.index, 'Innings'], 
                       filtered_bowl_df.loc[x.index, 'Bowler_Wkts']) 
                       if i in [1, 2] and w >= 5][0] if any([w >= 5 for w, i in 
                       zip(filtered_bowl_df.loc[x.index, 'Bowler_Wkts'], 
                       filtered_bowl_df.loc[x.index, 'Innings']) if i in [1, 2]]) else 0,
                      [o for o, i, w in zip(x, filtered_bowl_df.loc[x.index, 'Innings'], 
                       filtered_bowl_df.loc[x.index, 'Bowler_Wkts']) 
                       if i in [3, 4] and w >= 5][0] if any([w >= 5 for w, i in 
                       zip(filtered_bowl_df.loc[x.index, 'Bowler_Wkts'], 
                       filtered_bowl_df.loc[x.index, 'Innings']) if i in [3, 4]]) else 0
                  ],
                  'Bat_Team': 'first',
                  'Bowl_Team': 'first',
                  'Match_Format': 'first'
              })
              .reset_index())
    
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
    return five_wickets_both_df

@st.cache_data
def process_match_bowling(filtered_bowl_df):
    """Process match bowling figures data"""
    if filtered_bowl_df is None or filtered_bowl_df.empty:
        return pd.DataFrame()
    
    match_bowling_df = (filtered_bowl_df
        .groupby(['Name', 'Date', 'Home_Team', 'Bat_Team', 'Bowl_Team', 'Match_Format'])
        .agg({
            'Bowler_Wkts': lambda x: [
                max([w for w, i in zip(x, filtered_bowl_df.loc[x.index, 'Innings']) 
                    if i in [1, 2]] or [0]),
                max([w for w, i in zip(x, filtered_bowl_df.loc[x.index, 'Innings']) 
                    if i in [3, 4]] or [0])
            ],
            'Bowler_Runs': lambda x: [
                sum([r for r, i in zip(x, filtered_bowl_df.loc[x.index, 'Innings']) 
                    if i in [1, 2]] or [0]),
                sum([r for r, i in zip(x, filtered_bowl_df.loc[x.index, 'Innings']) 
                    if i in [3, 4]] or [0])
            ]
        })
        .reset_index())
    
    match_bowling_df['1st Innings Wkts'] = match_bowling_df['Bowler_Wkts'].str[0]
    match_bowling_df['2nd Innings Wkts'] = match_bowling_df['Bowler_Wkts'].str[1]
    match_bowling_df['1st Innings Runs'] = match_bowling_df['Bowler_Runs'].str[0]
    match_bowling_df['2nd Innings Runs'] = match_bowling_df['Bowler_Runs'].str[1]
    
    match_bowling_df['Match Wickets'] = match_bowling_df['1st Innings Wkts'] + match_bowling_df['2nd Innings Wkts']
    match_bowling_df['Match Runs'] = match_bowling_df['1st Innings Runs'] + match_bowling_df['2nd Innings Runs']
    
    df = (match_bowling_df
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
    
    df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')
    return df

###############################################################################
# Section 5: Bowling Records Tab Display
###############################################################################
with tabs[1]:
    if filtered_bowl_df is not None and not filtered_bowl_df.empty:
        # Best Bowling Figures
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Best Bowling Figures</h3>", 
                   unsafe_allow_html=True)
        best_bowling_df = process_best_bowling(filtered_bowl_df)
        if not best_bowling_df.empty:
            st.dataframe(best_bowling_df, use_container_width=True, hide_index=True)

        # 5+ Wickets in Both Innings
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>5+ Wickets in Both Innings</h3>", 
                   unsafe_allow_html=True)
        five_wickets_both_df = process_five_wickets_both(filtered_bowl_df)
        if not five_wickets_both_df.empty:
            st.dataframe(five_wickets_both_df, use_container_width=True, hide_index=True)

        # Best Match Bowling Figures
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Best Match Bowling Figures</h3>", 
                   unsafe_allow_html=True)
        best_match_bowling_df = process_match_bowling(filtered_bowl_df)
        if not best_match_bowling_df.empty:
            st.dataframe(best_match_bowling_df, use_container_width=True, hide_index=True)
    else:
        st.info("No bowling records available.")

###############################################################################
# Section 6: Match Records Functions and Tab
###############################################################################
@st.cache_data
def process_wins_data(filtered_match_df, win_type='runs', margin_type='big'):
    """Process wins data by type and margin"""
    if filtered_match_df is None or filtered_match_df.empty:
        return pd.DataFrame()
    
    # Set conditions based on win type
    if win_type == 'runs':
        base_conditions = [
            (filtered_match_df['Margin_Runs'] > 0),
            (filtered_match_df['Innings_Win'] == 0),
            (filtered_match_df['Home_Drawn'] != 1)
        ]
        margin_column = 'Margin_Runs'
        margin_name = 'Runs'
    else:  # wickets
        base_conditions = [
            (filtered_match_df['Margin_Wickets'] > 0),
            (filtered_match_df['Innings_Win'] == 0),
            (filtered_match_df['Home_Drawn'] != 1)
        ]
        margin_column = 'Margin_Wickets'
        margin_name = 'Wickets'
    
    df = filtered_match_df[np.all(base_conditions, axis=0)].copy()
    
    # Add win team and opponent columns
    df['Win Team'] = np.where(
        df['Home_Win'] == 1,
        df['Home_Team'],
        df['Away_Team']
    )
    
    df['Opponent'] = np.where(
        df['Home_Win'] == 1,
        df['Away_Team'],
        df['Home_Team']
    )
    
    # Sort based on margin type
    ascending = True if margin_type == 'narrow' else False
    
    df = (df
        .sort_values(margin_column, ascending=ascending)
        [['Date', 'Win Team', 'Opponent', margin_column, 'Match_Result', 'Match_Format']]
        .rename(columns={
            margin_column: margin_name,
            'Match_Result': 'Match Result',
            'Match_Format': 'Format'
        }))
    
    df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')
    return df

@st.cache_data
def process_innings_wins(filtered_match_df):
    """Process innings wins data"""
    if filtered_match_df is None or filtered_match_df.empty:
        return pd.DataFrame()
    
    df = filtered_match_df[
        (filtered_match_df['Innings_Win'] == 1) & 
        (filtered_match_df['Home_Drawn'] != 1)
    ].copy()
    
    df['Win Team'] = np.where(
        df['Home_Win'] == 1,
        df['Home_Team'],
        df['Away_Team']
    )
    
    df['Opponent'] = np.where(
        df['Home_Win'] == 1,
        df['Away_Team'],
        df['Home_Team']
    )
    
    df = (df
        .sort_values('Margin_Runs', ascending=False)
        [['Date', 'Win Team', 'Opponent', 'Margin_Runs', 'Innings_Win', 'Match_Result', 'Match_Format']]
        .rename(columns={
            'Margin_Runs': 'Runs',
            'Innings_Win': 'Innings',
            'Match_Result': 'Match Result',
            'Match_Format': 'Format'
        }))
    
    df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')
    return df

###############################################################################
# Section 6: Match Records Tab Display
###############################################################################
with tabs[2]:
    if filtered_match_df is not None and not filtered_match_df.empty:
        # Biggest Wins (Runs)
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Biggest Wins (Runs)</h3>", 
                   unsafe_allow_html=True)
        bigwin_df = process_wins_data(filtered_match_df, 'runs', 'big')
        if not bigwin_df.empty:
            st.dataframe(bigwin_df, use_container_width=True, hide_index=True)

        # Biggest Wins (Wickets)
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Biggest Wins (Wickets)</h3>", 
                   unsafe_allow_html=True)
        bigwin_wickets_df = process_wins_data(filtered_match_df, 'wickets', 'big')
        if not bigwin_wickets_df.empty:
            st.dataframe(bigwin_wickets_df, use_container_width=True, hide_index=True)

        # Innings Wins
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Biggest Wins (Innings)</h3>", 
                   unsafe_allow_html=True)
        bigwin_innings_df = process_innings_wins(filtered_match_df)
        if not bigwin_innings_df.empty:
            st.dataframe(bigwin_innings_df, use_container_width=True, hide_index=True)

        # Narrowest Wins (Runs)
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Narrowest Wins (Runs)</h3>", 
                   unsafe_allow_html=True)
        narrow_wins_df = process_wins_data(filtered_match_df, 'runs', 'narrow')
        if not narrow_wins_df.empty:
            st.dataframe(narrow_wins_df, use_container_width=True, hide_index=True)

        # Narrowest Wins (Wickets)
        st.markdown("<h3 style='color:#f04f53; text-align: center;'>Narrowest Wins (Wickets)</h3>", 
                   unsafe_allow_html=True)
        narrowwin_wickets_df = process_wins_data(filtered_match_df, 'wickets', 'narrow')
        if not narrowwin_wickets_df.empty:
            st.dataframe(narrowwin_wickets_df, use_container_width=True, hide_index=True)
    else:
        st.info("No match records available.")

###############################################################################
# Section 7: Game Records Functions and Tab
###############################################################################
@st.cache_data
def get_match_details(filtered_game_df, filtered_match_df):
    """Process match details and add Result and Margin columns"""
    if filtered_game_df is None or filtered_match_df is None:
        return filtered_game_df
        
    def process_row(row):
        bat_team = row['Bat_Team']
        bowl_team = row['Bowl_Team']
        result = 'Unknown'
        margin_info = '-'
        innings_win = 0
        margin_runs = None
        margin_wickets = None
        
        # Check for matches where bat_team is home and bowl_team is away
        home_match = filtered_match_df[
            (filtered_match_df['Home_Team'] == bat_team) & 
            (filtered_match_df['Away_Team'] == bowl_team)
        ]
        
        # Check for matches where bat_team is away and bowl_team is home
        away_match = filtered_match_df[
            (filtered_match_df['Away_Team'] == bat_team) & 
            (filtered_match_df['Home_Team'] == bowl_team)
        ]
        
        if not home_match.empty:
            match = home_match.iloc[0]
            if match['Home_Win'] == 1:
                result = 'Win'
            elif match['Home_Lost'] == 1:
                result = 'Lost'
            elif match['Home_Drawn'] == 1:
                result = 'Draw'
            
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
    
    # Add the result columns
    result_df = filtered_game_df.copy()
    result_df[['Result', 'Margin', 'Innings_Win', 'Margin_Runs', 'Margin_Wickets']] = \
        filtered_game_df.apply(process_row, axis=1)
    
    return result_df

@st.cache_data
def process_team_scores(filtered_game_df, score_type='highest'):
    """Process team scores data"""
    if filtered_game_df is None or filtered_game_df.empty:
        return pd.DataFrame()
    
    df = filtered_game_df[['Bat_Team', 'Bowl_Team', 'Total_Runs', 'Wickets',
                          'Overs', 'Run_Rate', 'Competition', 'Match_Format',
                          'Date']].copy()
    
    if score_type == 'lowest':
        df = df[df['Wickets'] == 10]
        df = df.sort_values('Total_Runs', ascending=True)
    else:
        df = df.sort_values('Total_Runs', ascending=False)
    
    df['Date'] = df['Date'].dt.date
    df.columns = ['Bat Team', 'Bowl Team', 'Runs', 'Wickets', 'Overs',
                 'Run Rate', 'Competition', 'Format', 'Date']
    
    return df

###############################################################################
# Section 7: Game Records Tab Display
###############################################################################
with tabs[3]:
    if filtered_game_df is not None and filtered_match_df is not None and not filtered_game_df.empty:
        try:
            # Process match details first with caching
            @st.cache_data
            def get_processed_game_df(filtered_game_df, filtered_match_df):
                return get_match_details(filtered_game_df, filtered_match_df)
            
            processed_game_df = get_processed_game_df(filtered_game_df, filtered_match_df)
            
            # Highest Team Scores
            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Highest Team Scores</h3>", 
                       unsafe_allow_html=True)
            highest_scores_df = process_team_scores(processed_game_df, 'highest')
            if not highest_scores_df.empty:
                st.dataframe(highest_scores_df, use_container_width=True, hide_index=True)

            # Lowest Team Scores
            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Lowest Team Scores (All Out)</h3>", 
                       unsafe_allow_html=True)
            lowest_scores_df = process_team_scores(processed_game_df, 'lowest')
            if not lowest_scores_df.empty:
                st.dataframe(lowest_scores_df, use_container_width=True, hide_index=True)

            ##################====================HIGH CHASES===================##################
            # Highest Successful Run Chases with caching
            @st.cache_data
            def get_highest_chases(processed_game_df):
                chases_df = processed_game_df[['Bat_Team', 'Bowl_Team', 'Total_Runs', 'Wickets', 
                                               'Overs', 'Run_Rate', 'Competition', 'Match_Format', 
                                               'Date', 'Innings', 'Result', 'Margin']].copy()

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
                chases_df = chases_df[successful_chases]
                chases_df['Date'] = chases_df['Date'].dt.date

                def format_margin_wickets(row):
                    if pd.notna(row['Margin']) and 'wicket' in str(row['Margin']).lower():
                        return row['Margin']
                    elif pd.notna(row['Margin']):
                        wickets_left = 10 - row['Wickets']
                        return f"by {wickets_left} wickets"
                    return "-"
                
                chases_df['Margin_Display'] = chases_df.apply(format_margin_wickets, axis=1)
                chases_df.columns = ['Bat Team', 'Bowl Team', 'Runs', 'Wickets', 'Overs', 
                                     'Run Rate', 'Competition', 'Format', 'Date', 'Innings', 'Result', 'Margin', 'Margin_Display']
                chases_df = chases_df.sort_values('Runs', ascending=False)
                return chases_df[['Bat Team', 'Bowl Team', 'Innings', 'Runs', 'Wickets', 'Overs', 
                                  'Run Rate', 'Competition', 'Format', 'Date', 'Margin_Display']].rename(columns={'Margin_Display': 'Margin'})
            
            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Highest Successful Run Chases</h3>", unsafe_allow_html=True)
            chases_df = get_highest_chases(processed_game_df)
            st.dataframe(chases_df, use_container_width=True, hide_index=True)

            ##################====================LOW 1ST INNINGS===================##################
            # Lowest First Innings Wins with caching
            @st.cache_data
            def get_lowest_first_innings_wins(processed_game_df):
                low_wins_df = processed_game_df[['Bat_Team', 'Bowl_Team', 'Total_Runs', 'Wickets', 
                                                 'Overs', 'Run_Rate', 'Competition', 'Match_Format', 
                                                 'Date', 'Innings', 'Result', 'Margin']].copy()

                first_innings_wins = (low_wins_df['Innings'] == 1) & (low_wins_df['Result'] == 'Win')
                low_wins_df = low_wins_df[first_innings_wins]
                low_wins_df['Date'] = low_wins_df['Date'].dt.date

                low_wins_df.columns = ['Bat Team', 'Bowl Team', 'Runs', 'Wickets', 'Overs', 
                                       'Run Rate', 'Competition', 'Format', 'Date', 'Innings', 'Result', 'Margin']
                return low_wins_df.sort_values('Runs', ascending=True)[['Bat Team', 'Bowl Team', 'Innings', 'Runs', 'Wickets', 'Overs', 
                                                                       'Run Rate', 'Competition', 'Format', 'Date', 'Margin']]
            
            st.markdown("<h3 style='color:#f04f53; text-align: center;'>Lowest First Innings Winning Scores</h3>", unsafe_allow_html=True)
            low_wins_df = get_lowest_first_innings_wins(processed_game_df)
            st.dataframe(low_wins_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error processing records: {str(e)}")
    else:
        st.info("No game or match records available.")
