import streamlit as st
import datetime
import time
import os

# PERFORMANCE OPTIMIZATION: Add memory management
try:
    from memory_optimization import add_memory_sidebar, optimize_dataframes_on_load
    MEMORY_OPTIMIZATION_AVAILABLE = True
except ImportError:
    MEMORY_OPTIMIZATION_AVAILABLE = False
    st.warning("⚠️ Memory optimization not available - install performance_utils.py for better performance")

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
    
    /* Preview grid styling */
    .preview-section {
        margin-top: 40px;
        padding: 20px;
        max-width: 1200px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .preview-header {
        text-align: center;
        margin-bottom: 30px;
    }
    
    .preview-header h2 {
        color: #333;
        font-size: 2rem;
        margin-bottom: 10px;
    }
    
    .preview-header p {
        color: #666;
        font-size: 1.1rem;
    }
    
    .preview-grid {
        display: grid;
        grid-template-columns: 1fr;
        gap: 30px;
        margin-top: 40px;
    }
    
    .preview-card {
        background: white;
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        border: 1px solid #e0e0e0;
    }
    
    .preview-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
    }
    
    .preview-card img {
        width: 100%;
        height: 350px;
        object-fit: cover;
        border-radius: 12px;
        margin-bottom: 20px;
        transition: transform 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .preview-card:hover img {
        transform: scale(1.02);
    }
    
    .preview-card h3 {
        color: #333;
        font-size: 1.5rem;
        margin-bottom: 15px;
        font-weight: bold;
    }
    
    .preview-card p {
        color: #666;
        font-size: 1.1rem;
        line-height: 1.7;
        margin: 0;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .preview-grid {
            grid-template-columns: 1fr;
            gap: 20px;
        }
        
        .preview-card {
            padding: 20px;
        }
        
        .preview-card img {
            height: 250px;
        }
        
        .preview-header h2 {
            font-size: 1.8rem;
        }
        
        .preview-header p {
            font-size: 1rem;
        }
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

def login():
    # Get credentials securely
    CORRECT_USERNAME, CORRECT_PASSWORD = get_credentials()
    
    st.markdown(
        """
        <div class="login-container">
            <div class="login-header">
                <h1 style="color: #f04f53; margin-bottom: 8px; font-size: 1.6rem;">🏏 Cricket Captain 2025 Stats Pack</h1>
                <p style="margin-bottom: 12px; font-size: 0.95rem; color: #666;">Access your comprehensive cricket analytics below</p>
                <div style="background: linear-gradient(135deg, #fff3cd, #ffeaa7); border-left: 4px solid #f39c12; padding: 10px 15px; border-radius: 8px; margin-bottom: 12px;">
                    <p style="margin: 0; font-size: 0.9rem; color: #856404;"><strong>🔐 New July Password:</strong> Check the latest <a href="https://buymeacoffee.com/leadingedgepod/cricket-captain-2025-stats-pack-password-update" target="_blank" style="color: #f04f53; text-decoration: none; font-weight: bold;">Buy Me A Coffee post</a></p>
                </div>
                <div style="background: linear-gradient(135deg, #d1ecf1, #bee5eb); border-left: 4px solid #17a2b8; padding: 10px 15px; border-radius: 8px; margin-bottom: 15px;">
                    <p style="margin: 0; font-size: 0.9rem; color: #0c5460;"><strong>🎯 New User?</strong> Subscribe to <b>Cricket Captain 2024 Stats Pack</b> on <a href="https://buymeacoffee.com/leadingedgepod" target="_blank" style="color: #f04f53; text-decoration: none; font-weight: bold;">Buy Me A Coffee - Leading Edge</a></p>
                </div>
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
    
    # Function to encode image to base64
    def get_base64_image(image_path):
        try:
            import base64
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        except:
            return None
    
    # Get base64 encoded images
    images = {
        'batting': get_base64_image('assets/images/batting.png'),
        'compare': get_base64_image('assets/images/compare.png'),
        'records': get_base64_image('assets/images/records.png'),
        'scorecards': get_base64_image('assets/images/scorecards.png'),
        'rankings': get_base64_image('assets/images/rankings.png'),
        'headtohead': get_base64_image('assets/images/headtohead.png'),
        'headtoheadform': get_base64_image('assets/images/headtoheadform.png'),
        'headtoheadtrends': get_base64_image('assets/images/headtoheadtrends.png'),
        'teams': get_base64_image('assets/images/teams.png')
    }
    
    # Add a quick preview section right after login
    st.markdown(f"""
        <div style="margin-top: 30px; padding: 20px; max-width: 800px; margin-left: auto; margin-right: auto;">
            <h3 style="text-align: center; color: #333; margin-bottom: 20px;">🏏 What's Inside the Dashboard</h3>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
                <div style="background: white; border-radius: 10px; padding: 15px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); text-align: center;">
                    <img src="data:image/png;base64,{images['batting'] if images['batting'] else ''}" alt="Batting Stats" style="width: 100%; height: 120px; object-fit: cover; border-radius: 8px; margin-bottom: 10px;">
                    <div style="background: linear-gradient(135deg, #6c24c0, #f04f53); color: white; padding: 4px 8px; border-radius: 15px; font-size: 0.7rem; font-weight: 600; margin-bottom: 8px; display: inline-block;">STATS</div>
                    <h4 style="margin: 0; font-size: 0.9rem; color: #333;">Batting & Bowling Stats</h4>
                </div>
                <div style="background: white; border-radius: 10px; padding: 15px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); text-align: center;">
                    <img src="data:image/png;base64,{images['compare'] if images['compare'] else ''}" alt="Compare Players" style="width: 100%; height: 120px; object-fit: cover; border-radius: 8px; margin-bottom: 10px;">
                    <div style="background: linear-gradient(135deg, #6c24c0, #f04f53); color: white; padding: 4px 8px; border-radius: 15px; font-size: 0.7rem; font-weight: 600; margin-bottom: 8px; display: inline-block;">COMPARE</div>
                    <h4 style="margin: 0; font-size: 0.9rem; color: #333;">Compare Players</h4>
                </div>
                <div style="background: white; border-radius: 10px; padding: 15px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); text-align: center;">
                    <img src="data:image/png;base64,{images['records'] if images['records'] else ''}" alt="Records" style="width: 100%; height: 120px; object-fit: cover; border-radius: 8px; margin-bottom: 10px;">
                    <div style="background: linear-gradient(135deg, #6c24c0, #f04f53); color: white; padding: 4px 8px; border-radius: 15px; font-size: 0.7rem; font-weight: 600; margin-bottom: 8px; display: inline-block;">RECORDS</div>
                    <h4 style="margin: 0; font-size: 0.9rem; color: #333;">Records Viewer</h4>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Add the preview section below the form
    st.markdown(f"""
        <div class="preview-section">
            <div class="preview-header">
                <h2>🏏 Discover What's Inside</h2>
                <p>Explore the comprehensive Cricket Captain 2025 stats dashboard features</p>
            </div>
            <div class="preview-grid">
                <div class="preview-card">
                    <img src="data:image/png;base64,{images['batting'] if images['batting'] else ''}" alt="Batting Stats" style="width: 100%; height: 350px; object-fit: cover; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
                    <div style="background: linear-gradient(135deg, #6c24c0, #f04f53); color: white; padding: 6px 16px; border-radius: 25px; font-size: 0.9rem; font-weight: 600; margin-bottom: 15px; display: inline-block;">STATS</div>
                    <h3>📊 Batting, Bowling, and All-Rounder Stats</h3>
                    <p>All the stats you need to make the key decisions in your Cricket Captain 2025 save. Detailed analysis for career, form, position, location and more.</p>
                </div>
                <div class="preview-card">
                    <img src="data:image/png;base64,{images['compare'] if images['compare'] else ''}" alt="Compare Players" style="width: 100%; height: 350px; object-fit: cover; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
                    <div style="background: linear-gradient(135deg, #6c24c0, #f04f53); color: white; padding: 6px 16px; border-radius: 25px; font-size: 0.9rem; font-weight: 600; margin-bottom: 15px; display: inline-block;">COMPARE</div>
                    <h3>🔍 Compare Players</h3>
                    <p>Compare players across formats and head-to-head. Find your best XI with data.</p>
                </div>
                <div class="preview-card">
                    <img src="data:image/png;base64,{images['records'] if images['records'] else ''}" alt="Records Viewer" style="width: 100%; height: 350px; object-fit: cover; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
                    <div style="background: linear-gradient(135deg, #6c24c0, #f04f53); color: white; padding: 6px 16px; border-radius: 25px; font-size: 0.9rem; font-weight: 600; margin-bottom: 15px; display: inline-block;">RECORDS</div>
                    <h3>🏆 Records Viewer</h3>
                    <p>Visualise every batting, bowling, match and innings record from your save.</p>
                </div>
                <div class="preview-card">
                    <img src="data:image/png;base64,{images['scorecards'] if images['scorecards'] else ''}" alt="Scorecard Archive" style="width: 100%; height: 350px; object-fit: cover; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
                    <div style="background: linear-gradient(135deg, #6c24c0, #f04f53); color: white; padding: 6px 16px; border-radius: 25px; font-size: 0.9rem; font-weight: 600; margin-bottom: 15px; display: inline-block;">ARCHIVE</div>
                    <h3>📖 Scorecard Archive</h3>
                    <p>Relive the history of every scorecard and revisit the big moments.</p>
                </div>
                <div class="preview-card">
                    <img src="data:image/png;base64,{images['rankings'] if images['rankings'] else ''}" alt="Player Rankings" style="width: 100%; height: 350px; object-fit: cover; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
                    <div style="background: linear-gradient(135deg, #6c24c0, #f04f53); color: white; padding: 6px 16px; border-radius: 25px; font-size: 0.9rem; font-weight: 600; margin-bottom: 15px; display: inline-block;">RANKINGS</div>
                    <h3>🥇 Player Rankings & Elo System</h3>
                    <p>Track player progression with our Hall of Fame system and team Elo ratings like chess.com.</p>
                </div>
                <div class="preview-card">
                    <img src="data:image/png;base64,{images['headtohead'] if images['headtohead'] else ''}" alt="Head-to-Head Overview" style="width: 100%; height: 350px; object-fit: cover; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
                    <div style="background: linear-gradient(135deg, #6c24c0, #f04f53); color: white; padding: 6px 16px; border-radius: 25px; font-size: 0.9rem; font-weight: 600; margin-bottom: 15px; display: inline-block;">ANALYSIS</div>
                    <h3>🧠 Head-to-Head Overview</h3>
                    <p>Review your coaching performance and win/loss records across formats.</p>
                </div>
                <div class="preview-card">
                    <img src="data:image/png;base64,{images['headtoheadform'] if images['headtoheadform'] else ''}" alt="Head-to-Head Form Tracker" style="width: 100%; height: 350px; object-fit: cover; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
                    <div style="background: linear-gradient(135deg, #6c24c0, #f04f53); color: white; padding: 6px 16px; border-radius: 25px; font-size: 0.9rem; font-weight: 600; margin-bottom: 15px; display: inline-block;">TRENDS</div>
                    <h3>🔄 Head-to-Head Form Tracker</h3>
                    <p>Monitor form trends over time and see how momentum shifts.</p>
                </div>
                <div class="preview-card">
                    <img src="data:image/png;base64,{images['headtoheadtrends'] if images['headtoheadtrends'] else ''}" alt="Performance Trends" style="width: 100%; height: 350px; object-fit: cover; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
                    <div style="background: linear-gradient(135deg, #6c24c0, #f04f53); color: white; padding: 6px 16px; border-radius: 25px; font-size: 0.9rem; font-weight: 600; margin-bottom: 15px; display: inline-block;">GRAPHS</div>
                    <h3>📈 Performance Trend Graphs</h3>
                    <p>See performance ups and downs by format or opposition.</p>
                </div>
                <div class="preview-card">
                    <img src="data:image/png;base64,{images['teams'] if images['teams'] else ''}" alt="Team Analytics" style="width: 100%; height: 350px; object-fit: cover; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
                    <div style="background: linear-gradient(135deg, #6c24c0, #f04f53); color: white; padding: 6px 16px; border-radius: 25px; font-size: 0.9rem; font-weight: 600; margin-bottom: 15px; display: inline-block;">TEAM</div>
                    <h3>⚔️ Team Match Analytics</h3>
                    <p>Compare your team's performance year-by-year against all others. Is your batting underperforming or is your bowling carrying you?</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

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
similar_players_page = st.Page(
    "views/similarplayers.py",
    title="Similar Players",
    icon="🔍",
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
            similar_players_page,
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
