import streamlit as st
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import re

def get_latest_video_id():
    try:
        # Get the channel page
        response = requests.get('https://www.youtube.com/@RobTaylor1985/videos')
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Look for the first video link
            script_tag = soup.find('script', string=re.compile(r'"videoId":"([^"]+)"'))
            if script_tag:
                video_id = re.search(r'"videoId":"([^"]+)"', script_tag.string).group(1)
                return video_id
    except Exception as e:
        st.error(f"Error fetching latest video: {e}")
    return None

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

st.markdown("<h1 style='color:#f04f53; text-align: center;'>Watch Cricket Captain 2024 Youtube Videos</h1>", unsafe_allow_html=True)

# Update CSS to control video widths
st.markdown("""
<style>
    .video-section {
        padding: 15px;
        margin: 10px auto;  /* Center the section */
        border-radius: 10px;
        background-color: #f8f9fa;
        max-width: 800px;   /* Limit maximum width */
    }
    .section-header {
        color: #f04f53;  /* Streamlit red */
        font-size: 20px;
        margin-bottom: 10px;
        text-align: center;
    }
    .playlist-container, .video-frame {
        width: 100%;
        height: 400px;
    }
    .video-column {
        max-width: 100%;    /* Ensure responsive behavior */
    }
</style>
""", unsafe_allow_html=True)

# Latest Video Section - with column wrapper for width control
st.markdown("<h2 class='section-header' style='color:#f04f53;'>Latest Video</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1,2,1])  # Create columns for centering
with col2:
    st.markdown("<div class='video-section'>", unsafe_allow_html=True)
    video_id = get_latest_video_id()
    if video_id:
        st.markdown(f"""
        <iframe class="video-frame"
        src="https://www.youtube.com/embed/{video_id}" 
        frameborder="0" 
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
        allowfullscreen></iframe>
        """, unsafe_allow_html=True)
    else:
        # Fallback to channel link if video fetch fails
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <p>Unable to load latest video. Click below to visit the channel:</p>
            <a href='https://www.youtube.com/@RobTaylor1985/videos' target='_blank'>View Latest Videos</a>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Alternative approach using a direct link
st.markdown("""
<div style='text-align: center; margin: 10px 0 30px;'>
    <a href='https://www.youtube.com/@RobTaylor1985/videos' target='_blank' 
    style='color: #666; text-decoration: none; font-size: 0.9em;'>
    View all recent videos →</a>
</div>
""", unsafe_allow_html=True)

# Career Rebuilds Section - Two columns with fixed height
st.markdown("<h2 class='section-header' style='margin-top: 30px; color:#f04f53;'>Career Rebuilds</h2>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='video-section'>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-header' style='color:#f04f53;'>West Indies</h3>", unsafe_allow_html=True)
    st.markdown("""
    <iframe class="playlist-container" 
    src="https://www.youtube.com/embed/videoseries?list=PLw134D7uCGyi4tgt4LNOlOS3_orWfrF-i" 
    frameborder="0" allowfullscreen></iframe>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='video-section'>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-header' style='color:#f04f53;'>England</h3>", unsafe_allow_html=True)
    st.markdown("""
    <iframe class="playlist-container"
    src="https://www.youtube.com/embed/videoseries?list=PLw134D7uCGyjOeobNkICmzRFf64kCkuKD" 
    frameborder="0" allowfullscreen></iframe>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# New row for additional rebuilds
st.markdown("<h2 class='section-header' style='margin-top: 30px; color:#f04f53;'>Domestic Rebuilds</h2>", unsafe_allow_html=True)

# First row
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='video-section'>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-header' style='color:#f04f53;'>Northants</h3>", unsafe_allow_html=True)
    st.markdown("""
    <iframe class="playlist-container" 
    src="https://www.youtube.com/embed/videoseries?list=PLw134D7uCGygD492ht7mJEmGmylmg95JA" 
    frameborder="0" allowfullscreen></iframe>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='video-section'>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-header' style='color:#f04f53;'>Youth Only Rebuild</h3>", unsafe_allow_html=True)
    st.markdown("""
    <iframe class="playlist-container"
    src="https://www.youtube.com/embed/videoseries?list=PLw134D7uCGyhAMr6uriYFBtT7zbGUe_uQ" 
    frameborder="0" allowfullscreen></iframe>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Second row
col3, col4 = st.columns(2)

with col3:
    st.markdown("<div class='video-section'>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-header' style='color:#f04f53;'>Sussex Rebuild</h3>", unsafe_allow_html=True)
    st.markdown("""
    <iframe class="playlist-container"
    src="https://www.youtube.com/embed/videoseries?list=PLw134D7uCGyhzHHpZmc-MldRf6d4JNuUV" 
    frameborder="0" allowfullscreen></iframe>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div class='video-section'>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-header' style='color:#f04f53;'>1 Episode Rebuilds</h3>", unsafe_allow_html=True)
    st.markdown("""
    <iframe class="playlist-container"
    src="https://www.youtube.com/embed/videoseries?list=PLw134D7uCGyivxSUSgwSzs4bow3K7R9gc" 
    frameborder="0" allowfullscreen></iframe>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Channel link at the bottom
st.markdown("""
<div style='text-align: center; margin-top: 20px;'>
    <a href='https://www.youtube.com/@RobTaylor1985' target='_blank' 
    style='background-color: #f04f53; color: white; padding: 10px 20px; 
    text-decoration: none; border-radius: 5px; font-weight: bold;'>
    Visit YouTube Channel</a>
</div>
""", unsafe_allow_html=True)


