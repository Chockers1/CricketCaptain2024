import streamlit as st
import datetime
import time
import os

# Try to import dotenv, but make it optional
try:
    from dotenv import load_dotenv
    # Load environment variables from .env file (for local development)
    try:
        load_dotenv()
    except:
        pass  # Silently continue if .env doesn't exist
except ImportError:
    # Create a dummy load_dotenv function to avoid errors
    def load_dotenv():
        pass  # Do nothing if the module is not available

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
    
    /* Login form styling */
    .login-container {
        max-width: 500px;
        margin: 0 auto;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        background-color: white;
    }
    
    .login-header {
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Session timer styling */
    .session-timer {
        font-size: 12px;
        text-align: center;
        margin-top: 5px;
        color: #666;
    }
    </style>
""", unsafe_allow_html=True)

# --- SESSION DURATION CONSTANTS ---
SESSION_DURATION = 24 * 60 * 60  # 24 hours in seconds

# --- SECURE CREDENTIAL HANDLING ---
def get_credentials():
    """
    Get credentials from environment variables, Streamlit secrets, or fallback to hardcoded values.
    The order of precedence ensures the most secure method available is used.
    
    For production deployment on Streamlit Cloud:
    1. Set these values in the Streamlit Cloud secrets management interface
    
    Returns:
        tuple: (username, password)
    """
    # Production: Get from Streamlit Cloud secrets (preferred for production)
    try:
        username = st.secrets["login"]["username"]
        password = st.secrets["login"]["password"]
        return username, password
    except Exception as e:
        # Log the error for debugging, but don't expose details in the UI
        print(f"Error accessing secrets: {str(e)}")
        
    # Development: Try environment variables next
    try:
        username = os.environ.get("CC_USERNAME")
        password = os.environ.get("CC_PASSWORD")
        if username and password:
            return username, password
    except Exception as e:
        print(f"Error accessing environment variables: {str(e)}")
        
    # Fallback: Use hardcoded credentials as a last resort
    # This should only be used in development, never in production
    return "CC", "CCapril2025"

# --- LOGIN FUNCTIONALITY ---
def login():
    # Get credentials securely
    CORRECT_USERNAME, CORRECT_PASSWORD = get_credentials()
    
    st.markdown(
        """
        <div class="login-container">
            <div class="login-header">
                <h1 style="color: #f04f53;">Cricket Captain Stats Dashboard</h1>
                <p>To access the Cricket Captain Stats Dashboard, please login below.</p>
                <p><strong>Please note:</strong> There is a new password for July - check the latest <a href="https://buymeacoffee.com/leadingedgepod/cricket-captain-2025-stats-pack-password-update" target="_blank" style="color: #f04f53; text-decoration: none; font-weight: bold;">Buy Me A Coffee post</a> for details.</p>
                <p>To sign up: Subscribe to the <b>Cricket Captain 2024 Stats Pack tier</b> on <a href="https://buymeacoffee.com/leadingedgepod" target="_blank" style="color: #f04f53; text-decoration: none; font-weight: bold;">Buy Me A Coffee - Leading Edge</a></p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Create three columns for centering the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        username_input = st.text_input("Username")
        password_input = st.text_input("Password", type="password")
        
        if st.button("Login", use_container_width=True):
            if (
                username_input == CORRECT_USERNAME and
                password_input == CORRECT_PASSWORD
            ):
                # Set logged in state with timestamp
                st.session_state["logged_in"] = True
                st.session_state["login_time"] = datetime.datetime.now().timestamp()
                st.rerun()
            else:
                st.error("Incorrect username or password. Please check your subscription details for the correct credentials.")

# Function to check if the session is still valid
def is_session_valid():
    if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
        return False
    
    if "login_time" not in st.session_state:
        # No timestamp found, consider session expired
        return False
        
    current_time = datetime.datetime.now().timestamp()
    elapsed_time = current_time - st.session_state["login_time"]
    
    # Session is valid if less than SESSION_DURATION seconds have passed
    return elapsed_time < SESSION_DURATION

# Function to display time remaining in session
def display_session_timer():
    if "login_time" in st.session_state:
        current_time = datetime.datetime.now().timestamp()
        elapsed_time = current_time - st.session_state["login_time"]
        remaining_time = SESSION_DURATION - elapsed_time
        
        # Calculate hours, minutes, seconds
        hours = int(remaining_time // 3600)
        minutes = int((remaining_time % 3600) // 60)
        
        # Format the time display
        if hours > 0:
            time_display = f"Session expires in: {hours}h {minutes}m"
        else:
            time_display = f"Session expires in: {minutes}m"
            
        return time_display
    return ""

# Check if user is logged in, if not, show login screen and stop further execution
if not is_session_valid():
    # Clear session state if expired
    if "logged_in" in st.session_state and st.session_state["logged_in"]:
        st.warning("Your session has expired. Please log in again.")
    
    login()
    # This line will prevent the rest of the app from loading until logged in
    st.stop()

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
watch_page = st.Page(
    "views/watch.py",
    title="Videos",
    icon="📺",
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
            watch_page
        ],
    }
)

# --- ADD LOGOS WITH LINKS IN SIDEBAR ---
st.sidebar.markdown("""
    <div class="logo-container">
        <a href="https://www.youtube.com/@RobTaylor1985" target="_blank" class="logo-link">
            <img src="https://cdn3.iconfinder.com/data/icons/social-network-30/512/social-06-512.png"
                  alt="YouTube Logo">
        </a>
        <a href="https://www.buymeacoffee.com/leadingedgepod" target="_blank" class="logo-link">
            <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSXjJ3v0CkIIjXGm5rbmM84s6u0IurR-6khKq57cL1t601Xc4OrS93JTF_ZWBH5cBWrQ2I&usqp=CAU"
                  alt="Buy Me A Coffee">
        </a>
    </div>
""", unsafe_allow_html=True)

# --- ADD SESSION TIMER AND LOGOUT BUTTON TO SIDEBAR ---
session_timer = display_session_timer()
st.sidebar.markdown(f"<div class='session-timer'>{session_timer}</div>", unsafe_allow_html=True)

if st.sidebar.button("Logout"):
    st.session_state["logged_in"] = False
    if "login_time" in st.session_state:
        del st.session_state["login_time"]
    st.rerun()

# --- RUN NAVIGATION ---
pg.run() 
