import streamlit as st
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import re

def get_latest_video_id():
    # Since YouTube scraping is unreliable, we'll use a fallback approach
    # You can manually update this with your latest video ID when needed
    fallback_video_id = "tQ3hMyWFHn8"  # Replace with your actual latest video ID
    
    try:
        # Attempt to get the channel page (this may not work reliably)
        response = requests.get('https://www.youtube.com/@RobTaylor1985/videos', timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Look for the first video link
            script_tag = soup.find('script', string=re.compile(r'"videoId":"([^"]+)"'))
            if script_tag:
                video_id = re.search(r'"videoId":"([^"]+)"', script_tag.string).group(1)
                return video_id
    except Exception as e:
        print(f"Error fetching latest video (using fallback): {e}")
    
    # Return fallback video ID if scraping fails
    return fallback_video_id

def get_playlist_videos(playlist_id):
    api_key = 'YOUR_YOUTUBE_API_KEY'  # Replace with your YouTube Data API key
    base_url = 'https://www.googleapis.com/youtube/v3/playlistItems'
    params = {
        'part': 'snippet',
        'playlistId': playlist_id,
        'maxResults': 50,
        'key': api_key
    }
    video_urls = []
    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            for item in data['items']:
                video_id = item['snippet']['resourceId']['videoId']
                video_urls.append(f"https://www.youtube.com/embed/{video_id}")
        else:
            st.error(f"Error fetching playlist videos: {response.json().get('error', {}).get('message', 'Unknown error')}")
    except Exception as e:
        st.error(f"Error fetching playlist videos: {e}")
    return video_urls

import streamlit as st
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import re

def get_latest_video_id():
    # Since YouTube scraping is unreliable, we'll use a fallback approach
    # You can manually update this with your latest video ID when needed
    fallback_video_id = "tQ3hMyWFHn8"  # Replace with your actual latest video ID
    
    try:
        # Attempt to get the channel page (this may not work reliably)
        response = requests.get('https://www.youtube.com/@RobTaylor1985/videos', timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Look for the first video link
            script_tag = soup.find('script', string=re.compile(r'"videoId":"([^"]+)"'))
            if script_tag:
                video_id = re.search(r'"videoId":"([^"]+)"', script_tag.string).group(1)
                return video_id
    except Exception as e:
        print(f"Error fetching latest video (using fallback): {e}")
    
    # Return fallback video ID if scraping fails
    return fallback_video_id

def get_playlist_videos(playlist_id):
    api_key = 'YOUR_YOUTUBE_API_KEY'  # Replace with your YouTube Data API key
    base_url = 'https://www.googleapis.com/youtube/v3/playlistItems'
    params = {
        'part': 'snippet',
        'playlistId': playlist_id,
        'maxResults': 50,
        'key': api_key
    }
    video_urls = []
    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            for item in data['items']:
                video_id = item['snippet']['resourceId']['videoId']
                video_urls.append(f"https://www.youtube.com/embed/{video_id}")
        else:
            st.error(f"Error fetching playlist videos: {response.json().get('error', {}).get('message', 'Unknown error')}")
    except Exception as e:
        st.error(f"Error fetching playlist videos: {e}")
    return video_urls

# Modern Header with Cricket Theme
st.markdown("""
<style>
    /* Modern UI Styling */
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
    
    .section-header {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem 1.5rem;
        border-radius: 15px;
        margin: 2rem 0 1.5rem 0;
        text-align: center;
        box-shadow: 0 8px 32px rgba(240, 147, 251, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .section-header h2 {
        color: white !important;
        margin: 0 !important;
        font-weight: bold;
        font-size: 1.8rem;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
    }
    
    .video-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.8);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .video-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.15);
    }
    
    .video-title {
        background: linear-gradient(135deg, #36d1dc 0%, #5b86e5 100%);
        color: white !important;
        padding: 0.8rem 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
        box-shadow: 0 4px 16px rgba(54, 209, 220, 0.3);
    }
    
    .video-frame {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        border: 3px solid rgba(255, 255, 255, 0.9);
    }
    
    .channel-button {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%) !important;
        color: white !important;
        padding: 1rem 2rem !important;
        border: none !important;
        border-radius: 15px !important;
        font-weight: bold !important;
        font-size: 1.2rem !important;
        text-decoration: none !important;
        display: inline-block !important;
        box-shadow: 0 8px 32px rgba(247, 151, 30, 0.4) !important;
        transition: all 0.3s ease !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2) !important;
    }
    
    .channel-button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 40px rgba(247, 151, 30, 0.5) !important;
        color: white !important;
        text-decoration: none !important;
    }
    
    .button-container {
        text-align: center;
        margin: 2rem 0;
        padding: 1rem;
    }
    
    /* Responsive iframe styling */
    .playlist-container {
        width: 100%;
        height: 350px;
        border: none;
        border-radius: 15px;
    }
    
    /* Hide default streamlit styling */
    .stApp > header {
        background-color: transparent;
    }
    
    .stApp {
        margin-top: -80px;
    }
</style>

<div class="main-header">
    <h1>🎬 Cricket Captain 2024 Videos</h1>
</div>
""", unsafe_allow_html=True)

# Channel link at the top with modern styling
st.markdown("""
<div class="button-container">
    <a href='https://www.youtube.com/@RobTaylor1985' target='_blank' class="channel-button">
    🎥 Visit YouTube Channel</a>
</div>
""", unsafe_allow_html=True)




# Career Rebuilds Section with modern styling
st.markdown("""
<div class="section-header">
    <h2>🏏 International Career Rebuilds</h2>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="video-container">
        <div class="video-title">🌴 West Indies Rebuild</div>
        <iframe class="playlist-container video-frame" 
        src="https://www.youtube.com/embed/videoseries?list=PLw134D7uCGyi4tgt4LNOlOS3_orWfrF-i" 
        frameborder="0" allowfullscreen></iframe>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="video-container">
        <div class="video-title">🏴󠁧󠁢󠁥󠁮󠁧󠁿 England Rebuild</div>
        <iframe class="playlist-container video-frame"
        src="https://www.youtube.com/embed/videoseries?list=PLw134D7uCGyjOeobNkICmzRFf64kCkuKD" 
        frameborder="0" allowfullscreen></iframe>
    </div>
    """, unsafe_allow_html=True)

# Second row for Career Rebuilds
col3, col4 = st.columns(2)

with col3:
    st.markdown("""
    <div class="video-container">
        <div class="video-title">🇿🇦 South Africa Rebuild</div>
        <iframe class="playlist-container video-frame"
        src="https://www.youtube.com/embed/QHnxvTANMj0" 
        frameborder="0" allowfullscreen></iframe>
    </div>
    """, unsafe_allow_html=True)

# Domestic Rebuilds Section with modern styling
st.markdown("""
<div class="section-header">
    <h2>🏟️ Domestic & County Rebuilds</h2>
</div>
""", unsafe_allow_html=True)

# First row
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="video-container">
        <div class="video-title">🏏 Northants County</div>
        <iframe class="playlist-container video-frame" 
        src="https://www.youtube.com/embed/videoseries?list=PLw134D7uCGygD492ht7mJEmGmylmg95JA" 
        frameborder="0" allowfullscreen></iframe>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="video-container">
        <div class="video-title">👶 Youth Only Challenge</div>
        <iframe class="playlist-container video-frame"
        src="https://www.youtube.com/embed/videoseries?list=PLw134D7uCGyhAMr6uriYFBtT7zbGUe_uQ" 
        frameborder="0" allowfullscreen></iframe>
    </div>
    """, unsafe_allow_html=True)

# Second row
col3, col4 = st.columns(2)

with col3:
    st.markdown("""
    <div class="video-container">
        <div class="video-title">🦅 Sussex County</div>
        <iframe class="playlist-container video-frame"
        src="https://www.youtube.com/embed/videoseries?list=PLw134D7uCGyhzHHpZmc-MldRf6d4JNuUV" 
        frameborder="0" allowfullscreen></iframe>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="video-container">
        <div class="video-title">⚡ Quick Rebuilds</div>
        <iframe class="playlist-container video-frame"
        src="https://www.youtube.com/embed/videoseries?list=PLw134D7uCGyivxSUSgwSzs4bow3K7R9gc" 
        frameborder="0" allowfullscreen></iframe>
    </div>
    """, unsafe_allow_html=True)

# Third row for Special Challenges
col5, col6 = st.columns(2)

with col5:
    st.markdown("""
    <div class="video-container">
        <div class="video-title">🔥 Impossible Challenge</div>
        <iframe class="playlist-container video-frame"
        src="https://www.youtube.com/embed/tQ3hMyWFHn8" 
        frameborder="0" allowfullscreen></iframe>
    </div>
    """, unsafe_allow_html=True)

# Channel link at the bottom with modern styling
st.markdown("""
<div class="button-container">
    <a href='https://www.youtube.com/@RobTaylor1985' target='_blank' class="channel-button">
    🔔 Subscribe for More Content</a>
</div>
""", unsafe_allow_html=True)
