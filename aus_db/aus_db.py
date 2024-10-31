import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Welcome to Ultimate Red Ball Cricket Database",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------ PAGE SETUP ------------------------
home_page = st.Page(
    "views/home.py",
    title="Home",
    icon="🏠",
    default=True,
)
player_stats_page = st.Page(
    "views/player.py",
    title="Player",
    icon="👤",
)
batting_page = st.Page(
    "views/batting.py",
    title="Batting",
    icon="🏏",
)
bowling_page = st.Page(
    "views/bowling.py",
    title="Bowling",
    icon="🤾",
)
all_rounders_page = st.Page(
    "views/all_rounders.py",
    title="All Rounders",
    icon="🚀",
)
cumulative_page = st.Page(
    "views/cumulative.py",
    title="Cumulative Stats",
    icon="📊",
)
team_page = st.Page(
    "views/team.py",
    title="Team Stats",
    icon="🏆",
)
rankings_page = st.Page(
    "views/rankings.py",
    title="Rankings",
    icon="📈",
)
records_page = st.Page(
    "views/records.py",
    title="Records",
    icon="📜",
)
seasons_page = st.Page(
    "views/seasons.py",
    title="Seasons",
    icon="📅",
)

# -------------------- NAVIGATION SETUP [WITH SECTIONS] ------------------------
pg = st.navigation(
    {
        "Cricket Stats": [home_page, player_stats_page, batting_page, bowling_page, all_rounders_page, cumulative_page, team_page, rankings_page, records_page, seasons_page],
    }
)


# --- ADD LOGOS WITH LINKS IN SIDEBAR ---
st.sidebar.markdown("""
<div class="logo-link" style="text-align: center;">
    <a href="https://www.youtube.com/@RobTaylor1985" target="_blank">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTdEjIsQcn8NwYWhwISL74xXLGtRHeW1Mn67g&s" alt="YouTube Logo" style="width: 80px; height: auto;">
    </a>
    <a href="https://buymeacoffee.com/leadingedgepod" target="_blank">
        <img src="https://play-lh.googleusercontent.com/aMb_Qiolzkq8OxtQZ3Af2j8Zsp-ZZcNetR9O4xSjxH94gMA5c5gpRVbpg-3f_0L7vlo" alt="Buy Me a Coffee Logo" style="width: 80px; height: auto;">
    </a>
</div>
""", unsafe_allow_html=True)

# --- RUN NAVIGATION ---
pg.run()
