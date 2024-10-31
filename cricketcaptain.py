import streamlit as st

# Password Protection
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    pwd = st.text_input("Enter password:", type="password")
    if pwd == st.secrets["passwords"]["access_pwd"]:
        st.session_state.authenticated = True
        st.rerun()
    elif pwd:  # Only show error if they've tried a password
        st.error("Incorrect password")
        st.stop()
    else:
        st.stop()

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
home_page = st.sidebar.page_link("views/Home.py", label="🏠 Home")
batting_page = st.sidebar.page_link("views/batview.py", label="🏏 Batting")
bowling_page = st.sidebar.page_link("views/bowlview.py", label="🤾 Bowling")
all_rounders_page = st.sidebar.page_link("views/allrounders.py", label="🚀 All Rounders")
team_page = st.sidebar.page_link("views/teamview.py", label="🏆 Team")
rankings_page = st.sidebar.page_link("views/rankings.py", label="📈 Int Rankings")
player_rankings_page = st.sidebar.page_link("views/Playerrankings.py", label="📈 Player Rankings")
records_page = st.sidebar.page_link("views/recordsview.py", label="📜 Records")
headtohead_page = st.sidebar.page_link("views/headtohead.py", label="🆚 Head to Head")
domestictables_page = st.sidebar.page_link("views/domestictables.py", label="📅 Domestic Tables")
elorating_page = st.sidebar.page_link("views/elorating.py", label="♟️ Elo Rating")

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