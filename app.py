import streamlit as st
import pandas as pd
import requests
import pickle
import os
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

# Load environment variables
load_dotenv()

# Cache for poster URLs to avoid repeated API calls
poster_cache = {}

with open('movie_dict.pkl','rb') as file:
    movies,cosine_sim=pickle.load(file)

def create_session_with_retries(retries=5, backoff_factor=2, timeout=15):
    """Create a requests session with retry strategy"""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def get_recommendations(title,cosine_sim=cosine_sim):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6] #get top 5 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return movies[['title','movie_id']].iloc[movie_indices]

def fetch_poster(movie_id, max_retries=5):
    """Fetch poster URL from TMDB API with retry logic and caching"""
    global poster_cache
    
    # Check cache first
    if movie_id in poster_cache:
        return poster_cache[movie_id]
    
    api_key = os.getenv('TMDB_API_KEY')
    
    if not api_key:
        print("WARNING: TMDB_API_KEY not found in environment variables")
        placeholder = "https://via.placeholder.com/500x750?text=No+Image"
        poster_cache[movie_id] = placeholder
        return placeholder
    
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}'
    session = create_session_with_retries(retries=max_retries, backoff_factor=2)
    
    try:
        # Add delay to avoid rate limiting (0.25 seconds between requests)
        time.sleep(0.25)
        
        response = session.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        if 'poster_path' in data and data['poster_path']:
            poster_path = data['poster_path']
            full_path = f"https://image.tmdb.org/t/p/w500/{poster_path}"
            poster_cache[movie_id] = full_path
            return full_path
        else:
            placeholder = "https://via.placeholder.com/500x750?text=No+Image"
            poster_cache[movie_id] = placeholder
            return placeholder
    except requests.exceptions.Timeout:
        print(f"Timeout error for movie_id {movie_id} - retrying with increased timeout")
        placeholder = "https://via.placeholder.com/500x750?text=No+Image"
        poster_cache[movie_id] = placeholder
        return placeholder
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error for movie_id {movie_id}: {e}")
        placeholder = "https://via.placeholder.com/500x750?text=No+Image"
        poster_cache[movie_id] = placeholder
        return placeholder
    except requests.exceptions.RequestException as e:
        print(f"Request error for movie_id {movie_id}: {e}")
        placeholder = "https://via.placeholder.com/500x750?text=No+Image"
        poster_cache[movie_id] = placeholder
        return placeholder
    finally:
        session.close()

st.title('Movie Recommendation System')
select_movie_name=st.selectbox('Select a movie from the list',movies['title'].values)
if st.button('Recommend'):
    recommendations=get_recommendations(select_movie_name)
    st.write('Top 5 similar movies to '+select_movie_name)
    
    #creating a grid 1X5 layout
    cols=st.columns(5)
    for col,j in zip(cols,range(5)):
        if j<len(recommendations):
            movie_title=recommendations.iloc[j]['title']
            movie_id=recommendations.iloc[j]['movie_id']
            poster_url=fetch_poster(movie_id)
            with col:
                st.image(poster_url,width=130)
                st.write(movie_title)