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
    /* Your existing CSS */
    </style>
""", unsafe_allow_html=True)

# Create pages directory structure
import os
import sys
from pathlib import Path

# Add the views directory to Python path
views_path = Path(__file__).parent / "views"
sys.path.append(str(views_path))

# Create sidebar navigation
st.sidebar.title("Navigation")

# Define pages and their icons
pages = {
    "Home": {"icon": "🏠", "module": "Home"},
    "Batting": {"icon": "🏏", "module": "batview"},
    "Bowling": {"icon": "🤾", "module": "bowlview"},
    "All Rounders": {"icon": "🚀", "module": "allrounders"},
    "Team": {"icon": "🏆", "module": "teamview"},
    "Int Rankings": {"icon": "📈", "module": "rankings"},
    "Player Rankings": {"icon": "📈", "module": "Playerrankings"},
    "Records": {"icon": "📜", "module": "recordsview"},
    "Head to Head": {"icon": "🆚", "module": "headtohead"},
    "Domestic Tables": {"icon": "📅", "module": "domestictables"},
    "Elo Rating": {"icon": "♟️", "module": "elorating"}
}

# Create navigation
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Create sidebar navigation
selected_page = st.sidebar.selectbox(
    "Choose a page",
    list(pages.keys()),
    format_func=lambda x: f"{pages[x]['icon']} {x}"
)

# Update session state
st.session_state.page = selected_page

# Import and run selected page
try:
    module = __import__(pages[selected_page]["module"])
    # You might need to call a specific function in the module
    if hasattr(module, "run"):
        module.run()
except Exception as e:
    st.error(f"Error loading page: {str(e)}")

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