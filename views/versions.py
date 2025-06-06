# Imports and Setup
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import os

# Determine if we're running locally or on Streamlit Cloud
IS_LOCAL = os.path.exists("assets/images")  # Check if local assets folder exists

# Helper function to get correct image path
def get_image_path(filename):
    if IS_LOCAL:
        return f"assets/images/{filename}"
    return f"CricketCaptainStatPack/assets/images/{filename}"

st.markdown("<h1 style='color:#f04f53; text-align: center;'>Version History</h1>", unsafe_allow_html=True)
versions_data = [


    {
        "version": "1.20, 2025-02-016",
        "title": "International Tournament History",
        "description": "In the Head to Head tab you can now see (if scorecards loaded) the history for all International Tournaments played in your save. This is a matrix view of every teams progression at each tournament, including the winner and runner-up for each tournament.",
        "screenshot": get_image_path("v1.20_tournament_records.png")
    },
    {
        "version": "1.19, 2025-02-04",
        "title": "Historical Series Stats and Records",
        "description": "In Records tab under Series records explore all the international series records from your save, Most Runs / Wickets in a series, Most Hundreds / 5W's in a series, Highest Averages and much more",
        "screenshot": get_image_path("v1.19_records_series.png")
    },
    {
        "version": "1.18, 2025-01-18",
        "title": "Brand New Series Head-to-Head Visuals",
        "description": "Explore cricket like never before with our new Series Head-to-Head visuals! Dive into detailed series history, trends, and team form across multiple formats with intuitive and interactive graphics. These visuals make it easy to analyze matchups, identify patterns, and compare performance across Test, ODI, and T20I formats. Gain valuable insights and make informed decisions with our latest feature.",
        "screenshot": get_image_path("v1.18_series_headtohead.png")
    },
    {
        "version": "1.17, 2025-01-17",
        "title": "Fixed Column Navigation Enhancement",
        "description": ("Enhanced table navigation with a fixed first column that stays in view while scrolling horizontally through data. "
                       "This user-friendly improvement keeps player names or key identifiers visible at all times, making it easier to track and compare statistics across multiple columns without losing context of who you're analyzing.\n\n"
                       "This simple yet effective enhancement significantly improves the way you navigate through statistical tables and player comparisons."),
        "screenshot": get_image_path("v1.17_fixed_column.png")
    },
    {
        "version": "1.16, 2025-01-13",
        "title": "Form Guide Enhancements",
        "description": ("Introducing the Format-Specific Form Guide - a dynamic new addition to the Head to Head page that provides detailed performance insights across different cricket formats. "
                       "This enhancement gives you a complete picture of how players and teams perform across all game formats, displaying recent form and historical performance trends against all opponents. "
                       "Track momentum, identify format specialists, and make informed decisions with comprehensive form data now available at a glance.\n\n"
                       "This new feature empowers users to better understand performance patterns and make strategic decisions based on format-specific form trends."),
        "screenshots": [
            {
                "path": get_image_path("v1.16_form_guide_1.png"),
                "caption": "Add the formats you want to see in the form guide, by default All is selected and it doesn't breakdown by format"
            },
            {
                "path": get_image_path("v1.16_form_guide_2.png"),
                "caption": "Format Filter Options"
            }
        ]
    },
    {
        "version": "1.15, 2024-12-28",
        "title": "Scorecard Tab",
        "description": ("Introducing the Scorecards Archive - a comprehensive historical database that gives you instant access to every match scorecard in your cricket career. "
                       "This powerful new feature lets you dive deep into your cricketing history:\n\n"
                       "Key highlights:\n\n"
                       "• Access detailed scorecards from any match you've ever played\n"
                       "• Review complete batting and bowling performances from historical games\n"
                       "• Track player achievements and memorable moments from past matches\n"
                       "• Analyze match trends and team performances across your career\n"
                       "• Browse through your cricket journey with an easy-to-navigate interface"),
        "screenshot": get_image_path("v1.15_scorecard_tab.png")
    },
    {
        "version": "1.14, 2024-12-26",
        "title": "Compare Players Tab",
        "description": "Introducing the Compare Players feature - a powerful new tool that lets you make data-driven selection decisions by directly comparing any two players across their complete performance metrics.",
        "screenshot": get_image_path("v1.14_compare_players.png")
    },
    {
        "version": "1.13, 2024-12-25",
        "title": "New Head to Head Visuals",
        "description": ("Introducing a brand-new, easy-to-read head-to-head performance summary against each opponent. "
                       "This visualization displays Tasmania's last 20 outings versus each team, providing a quick snapshot of results:\n\n"
                       "• W (Win): Represented in green circles\n"
                       "• L (Loss): Represented in red circles\n"
                       "• D (Draw): Represented in yellow circles\n\n"
                       "Each section includes a concise win-loss-draw summary for added context (e.g., \"W 8, L 9, D 0\")."),
        "screenshot": get_image_path("v1.13_head_to_head.png")
    },
    {
        "version": "1.12, 2024-12-24",
        "title": "New Batting Percentile Rankings",
        "description": ("This table provides a detailed percentile-based analysis of player performance across all scorecards loaded into the dataset. "
                       "It evaluates key metrics such as batting average, balls per out (BPO), strike rate (SR), and scoring potential. "
                       "By calculating percentile rankings for each player, this analysis identifies top performers relative to the entire dataset.\n\n"
                       "Key metrics include:\n\n"
                       "• Average Percentile: Highlights consistent scorers with high batting averages.\n"
                       "• BPO Percentile: Identifies players who excel at staying at the crease, showcasing their reliability.\n"
                       "• SR Percentile: Showcases players with high strike rates, valuable for aggressive gameplay.\n"
                       "• 50+ and 100+ Scoring Percentiles: Highlights players capable of producing impactful innings.\n\n"
                       "The Total Score and Total Percentile aggregate these metrics to provide an overall assessment of each player's contributions."),
        "screenshot": get_image_path("v1.12_batting_percentile.png")
    }
]

# Add custom CSS for styling the expander headers
st.markdown("""
<style>
    .version-header {
        color: #666;
        font-size: 0.9em;
    }
    .title-header {
        color: #f04f53;
        font-weight: bold;
        font-size: 1.1em;
        margin-left: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Modified CSS to style the version headers
st.markdown("""
<style>
    .stExpander {
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    .version-text {
        color: #666;
        font-size: 0.9em;
    }
    .title-text {
        color: #f04f53;
        font-weight: bold;
        font-size: 1.1em;
        margin-left: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Modified expander display
for item in versions_data:
    # Create the header text
    st.markdown(f"<div class='version-text' style='margin-bottom: -1em;'>Version {item['version']}"
               f"<span class='title-text'>{item['title']}</span></div>", 
               unsafe_allow_html=True)
    
    with st.expander("", expanded=False):
        st.markdown(f"**Description:** {item['description']}")
        if "screenshots" in item:
            for screenshot in item["screenshots"]:
                col1, col2, col3 = st.columns([1,3,1])
                with col2:
                    st.image(screenshot["path"], caption=screenshot["caption"])
        elif item.get("screenshot"):
            col1, col2, col3 = st.columns([1,3,1])
            with col2:
                st.image(item["screenshot"], caption=f"Version {item['version']} screenshot")

