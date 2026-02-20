import streamlit as st
import pandas as pd
import requests
import pickle
import os
import ast
import time
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Cache for poster URLs to avoid repeated API calls
if 'poster_cache' not in st.session_state:
    st.session_state.poster_cache = {}

@st.cache_data
def load_data():
    """
    Load data with caching to prevent reloading on every interaction.
    If pickle exists, use it. Otherwise, process CSVs.
    """
    pickle_path = 'movie_dict.pkl'
    
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, 'rb') as file:
                return pickle.load(file)
        except Exception as e:
            st.warning(f"Pickle corrupted, falling back to CSV: {e}")

    # Fallback to CSV Processing
    try:
        credits = pd.read_csv('tmdb_5000_credits.csv')
        movies_df = pd.read_csv('tmdb_5000_movies.csv')
        
        movies_df = movies_df.merge(credits, on='title')
        movies_df = movies_df[['movie_id','title','overview','genres','keywords','cast','crew']]
        
        def convert(obj):
            L = []
            try:
                for i in ast.literal_eval(obj):
                    L.append(i['name']) 
            except:
                pass
            return L
        
        movies_df['genres'] = movies_df['genres'].apply(convert)
        movies_df['keywords'] = movies_df['keywords'].apply(convert)
        movies_df['cast'] = movies_df['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[0:3]] if isinstance(x, str) else [])
        movies_df['crew'] = movies_df['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job']=='Director'] if isinstance(x, str) else [])
        
        movies_df['tags'] = movies_df['genres'] + movies_df['keywords'] + movies_df['cast'] + movies_df['crew']
        movies_df = movies_df[['movie_id','title','overview','tags']]
        movies_df['tags'] = movies_df['tags'].apply(lambda x: " ".join(x).lower())
        
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(movies_df['tags'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        return movies_df, cosine_sim
    except FileNotFoundError:
        st.error("Data files missing! Please ensure CSVs are in the repository.")
        return None, None

def create_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    return session

def fetch_poster(movie_id):
    if movie_id in st.session_state.poster_cache:
        return st.session_state.poster_cache[movie_id]
    
    api_key = os.getenv('TMDB_API_KEY')
    if not api_key:
        return "https://via.placeholder.com/500x750?text=Set+API+Key"
    
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}'
    try:
        session = create_session()
        response = session.get(url, timeout=5)
        data = response.json()
        path = data.get('poster_path')
        if path:
            full_path = f"https://image.tmdb.org/t/p/w500/{path}"
            st.session_state.poster_cache[movie_id] = full_path
            return full_path
    except:
        pass
    return "https://via.placeholder.com/500x750?text=No+Image"

def get_recommendations(title, movies, cosine_sim):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:6]]
    return movies.iloc[movie_indices]

# App Execution
movies, cosine_sim = load_data()

if movies is not None:
    st.title('Movie Recommender')
    selected_movie = st.selectbox('Type or select a movie', movies['title'].values)

    if st.button('Show Recommendations'):
        recs = get_recommendations(selected_movie, movies, cosine_sim)
        cols = st.columns(5)
        
        for i, col in enumerate(cols):
            with col:
                movie_id = recs.iloc[i]['movie_id']
                title = recs.iloc[i]['title']
                poster = fetch_poster(movie_id)
                st.image(poster)
                st.caption(title)