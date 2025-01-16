import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Welcome to The Cricket Captain Stats Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS FOR BACKGROUND COLOR ---
st.markdown("""
    <style>
    /* Multiselect tags */
    div[data-baseweb="tag"] {
        background-color: #6c24c0 !important;
    }
    
    /* Slider track and bar */
    .css-1p0q8wb, .css-eg6t2j {
        background-color: #6c24c0 !important;
    }
    
    /* Slider text */
    .css-1p0q8wb .stSlider p {
        color: #6c24c0 !important;
        background: transparent !important;
    }

    /* Markdown selection color */
    ::selection {
        background-color: #6c24c0 !important;
        color: white !important;
    }
    
    /* Logo container styling */
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
        margin-top: 20px;
    }
    
    .logo-link img {
        width: 80px;
        height: 80px;
        border-radius: 10px;
        transition: transform 0.3s ease;
    }
    
    .logo-link img:hover {
        transform: scale(1.1);
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------ PAGE SETUP ------------------------
home_page = st.Page(
    "views/Home.py",
    title="Home",
    icon="🏠",
)
batting_page = st.Page(
    "views/batview.py",
    title="Batting",
    icon="🏏",
)
bowling_page = st.Page(
    "views/bowlview.py",
    title="Bowling",
    icon="🤾",
)
all_rounders_page = st.Page(
    "views/allrounders.py",
    title="All Rounders",
    icon="🚀",
)
compare_page= st.Page(
    "views/compare.py",
    title="Compare Players",
    icon="⚖️",
)
team_page = st.Page(
    "views/teamview.py",
    title="Team",
    icon="🏆",
)
rankings_page = st.Page(
    "views/rankings.py",
    title="Int Rankings",
    icon="📈",
)
int_player_rankings_page = st.Page(
    "views/Playerrankings.py",
    title="Player Rankings",
    icon="📈",
)

records_page = st.Page(
    "views/recordsview.py",
    title="Records",
    icon="📜",
)
headtohead_page = st.Page(
    "views/headtohead.py",
    title="Head to Head",
    icon="🆚",
)
scorecard_page = st.Page(
    "views/scorecards.py",
    title="Scorecards",
    icon="📜",
)
domestictables_page = st.Page(
    "views/domestictables.py",
    title="Domestic Tables",
    icon="📅",
)
elorating_page = st.Page(
    "views/elorating.py",
    title="Elo Rating",
    icon="♟️",
)
versions_page = st.Page(
    "views/versions.py",
    title="Versions",
    icon="📖",
)


# -------------------- NAVIGATION SETUP ------------------------
pg = st.navigation(
    {
        "Dashboard Menu": [
            home_page,
            batting_page,
            bowling_page,
            all_rounders_page,
            compare_page,
            team_page,
            rankings_page,
            int_player_rankings_page,
            records_page,
            headtohead_page,
            domestictables_page,
            elorating_page,
            scorecard_page,
            versions_page,
        ],
    }
)

# --- ADD LOGOS WITH LINKS IN SIDEBAR ---
st.sidebar.markdown("""
    <div class="logo-container">
        <a href="https://www.youtube.com/@RobTaylor1985" target="_blank" class="logo-link">
            <img src="https://yt3.googleusercontent.com/584JjRp5QMuKbyduM_2k5RlXFqHJtQ0qLIPZpwbUjMJmgzZngHcam5JMuZQxyzGMV5ljwJRl0Q=s176-c-k-c0x00ffffff-no-rj"
                  alt="YouTube Logo">
        </a>
        <a href="https://www.buymeacoffee.com/leadingedgepod" target="_blank" class="logo-link">
            <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSXjJ3v0CkIIjXGm5rbmM84s6u0IurR-6khKq57cL1t601Xc4OrS93JTF_ZWBH5cBWrQ2I&usqp=CAU"
                  alt="Buy Me A Coffee">
        </a>
    </div>
""", unsafe_allow_html=True)

# --- RUN NAVIGATION ---
pg.run()
