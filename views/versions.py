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

# Main Header with Modern Styling
st.markdown("""
<style>
    /* Custom CSS for modern UI styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0 2rem 0;
        text-align: center;
        box-shadow: 0 12px 48px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .main-header h1 {
        color: white !important;
        margin: 0 !important;
        font-weight: bold;
        font-size: 2.5rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .version-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.8);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .version-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    .version-header {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 6px 24px rgba(240, 147, 251, 0.3);
    }
    
    .version-number {
        color: white !important;
        font-size: 0.9rem;
        font-weight: 500;
        opacity: 0.9;
        margin: 0 !important;
    }
    
    .version-title {
        color: white !important;
        font-size: 1.4rem;
        font-weight: bold;
        margin: 0.3rem 0 0 0 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
    }
    
    .description-container {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
        border-left: 4px solid #f093fb;
    }
    
    .description-text {
        color: #2c3e50 !important;
        font-size: 1rem;
        line-height: 1.6;
        margin: 0 !important;
    }
    
    .screenshot-container {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(240, 147, 251, 0.2);
    }
    
    .screenshot-caption {
        color: #6c757d !important;
        font-style: italic;
        font-size: 0.9rem;
        text-align: center;
        margin-top: 0.5rem !important;
    }
    
    /* Hide default streamlit expander styling */
    .stExpander > div > div > div > div {
        padding: 0 !important;
    }
    
    /* Custom expander button styling */
    .stExpander > div > button {
        background: linear-gradient(135deg, #36d1dc 0%, #5b86e5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.8rem 1.2rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 16px rgba(54, 209, 220, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stExpander > div > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(54, 209, 220, 0.4) !important;
    }
</style>

<div class="main-header">
    <h1>🏏 Version History & Updates</h1>
</div>
""", unsafe_allow_html=True)
versions_data = [
    {
        "version": "1.22, 2025-07-17",
        "title": "All-Time Elo, Performance Charts & Head-to-Head Revamp",
        "description": ("Another major update focused on historical context, player comparisons, and deeper data insights:\n\n"
                       
                       "📈 **All-Time Elo Rating Table:**\n"
                       "• New section showing the highest Elo ratings in County Championship history\n"
                       "• See how teams rank historically across all eras\n"
                       "• Highlights sustained excellence and peak performance\n\n"
                       
                       "🆚 **Head-to-Head View Enhancements:**\n"
                       "• Team record and match summary moved to the top for instant visibility\n"
                       "• Improved layout flow for faster comparison\n\n"
                       "🏏 **Player Rankings Adjustments:**\n"
                       "• Hall of Fame thresholds lowered to make rankings more inclusive\n"
                       "• New cutoffs: 7.5k, 10k, 12.5k points (down from 10k, 15k, 20k)\n\n"
                       "📊 **Player Comparison Upgrade:**\n"
                       "• New performance breakdown sections:\n"
                       "   - Batting Performance\n"
                       "   - Batting Milestones\n"
                       "   - Bowling Performance\n"
                       "   - Bowling Milestones\n"
                       "   - Awards\n"
                       "• Combined performance totals added to help spot best all-round contributors\n\n"
                       "🛠️ **Bug Fixes & Classification Improvements:**\n"
                       "• Three Day Friendly matches now correctly classified as First Class\n\n"
                       "📌 **New Graphs & Analytics:**\n"
                       "• **Bowling Tab (Position):** Added Strike Rate, Economy Rate graphs + scatter plot for performance by position\n"
                       "• **Batting Tab:** Position and Innings tabs now include Strike Rate graphs alongside batting averages\n\n"
                       
                       "This version adds depth to analysis, cleans up key visual areas, and ensures broader recognition of player excellence.")
    },
    {
        "version": "1.21, 2025-06-29",
        "title": "UI Enhancements & Duplication Checker",
        "description": ("Major UI modernization and scorecard management improvements:\n\n"
                       "🎨 **Beautiful Modern UI:**\n"
                       "• Complete visual overhaul with stunning gradient backgrounds\n"
                       "• Each navigation tab now features unique, beautiful color schemes\n"
                       "• Professional card layouts with hover effects and shadows\n"
                       "• Consistent modern styling across all pages\n"
                       "• Enhanced typography and visual hierarchy\n\n"
                       
                       "🔍 **Smart Duplication Checker:**\n"
                       "• Automatic detection of duplicate scorecards during upload\n"
                       "• Real-time alerts when duplicates are found\n"
                       "• Prevention of data corruption from repeated imports\n"
                       "• Clean database management with duplicate removal options\n\n"
                       
                       "⚙️ **Cricket Captain 2025 Integration:**\n"
                       "• Updated step-by-step instructions for the new auto-save scorecard feature\n"
                       "• Streamlined workflow taking advantage of CC2025's automatic scorecard generation\n"
                       "• Simplified setup process for new users\n"
                       "• Enhanced compatibility with the latest Cricket Captain features\n\n"
                       
                       "This update represents a significant leap forward in both visual appeal and functionality, making the application more intuitive and reliable than ever before.")
    },

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

# Remove old CSS styling
# (removing the old custom CSS section)

# Modern version display with beautiful cards
for i, item in enumerate(versions_data):
    # Create unique gradient colors for each version
    gradient_colors = [
        ("linear-gradient(135deg, #667eea 0%, #764ba2 100%)", "rgba(102, 126, 234, 0.3)"),  # Purple-Blue
        ("linear-gradient(135deg, #f093fb 0%, #f5576c 100%)", "rgba(240, 147, 251, 0.3)"),  # Pink-Red
        ("linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%)", "rgba(78, 205, 196, 0.3)"),   # Teal-Green
        ("linear-gradient(135deg, #fa709a 0%, #fee140 100%)", "rgba(250, 112, 154, 0.3)"),  # Orange-Yellow
        ("linear-gradient(135deg, #a8caba 0%, #5d4e75 100%)", "rgba(168, 202, 186, 0.3)"),  # Green-Purple
        ("linear-gradient(135deg, #36d1dc 0%, #5b86e5 100%)", "rgba(54, 209, 220, 0.3)"),   # Cyan-Blue
        ("linear-gradient(135deg, #8360c3 0%, #2ebf91 100%)", "rgba(131, 96, 195, 0.3)"),   # Purple-Teal
        ("linear-gradient(135deg, #11998e 0%, #38ef7d 100%)", "rgba(17, 153, 142, 0.3)"),   # Dark Teal-Green
        ("linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%)", "rgba(255, 126, 95, 0.3)"),   # Orange-Peach
        ("linear-gradient(135deg, #f7971e 0%, #ffd200 100%)", "rgba(247, 151, 30, 0.3)")    # Gold-Yellow
    ]
    
    # Select gradient based on version index, cycling through available colors
    gradient, shadow_color = gradient_colors[i % len(gradient_colors)]
    
    # Version container with dynamic styling
    st.markdown(f"""
    <div class="version-container">
        <div class="version-header" style="background: {gradient}; box-shadow: 0 6px 24px {shadow_color};">
            <p class="version-number">Version {item['version']}</p>
            <h3 class="version-title">{item['title']}</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Expandable details section
    with st.expander("📖 View Details", expanded=False):
        # Description section
        st.markdown(f"""
        <div class="description-container">
            <p class="description-text"><strong>What's New:</strong><br>{item['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Screenshots section
        if "screenshots" in item:
            st.markdown("### 📸 Screenshots")
            for j, screenshot in enumerate(item["screenshots"]):
                st.markdown(f"""
                <div class="screenshot-container">
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1, 4, 1])
                with col2:
                    st.image(screenshot["path"], use_column_width=True)
                    st.markdown(f"""
                    <p class="screenshot-caption">{screenshot["caption"]}</p>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
        elif item.get("screenshot"):
            st.markdown("### 📸 Screenshot")
            st.markdown(f"""
            <div class="screenshot-container">
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 4, 1])
            with col2:
                st.image(item["screenshot"], use_column_width=True)
                st.markdown(f"""
                <p class="screenshot-caption">Version {item['version']} - {item['title']}</p>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Add some spacing between versions
        st.markdown("<br>", unsafe_allow_html=True)
 
