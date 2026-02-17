# Movie Recommendation System

A Streamlit-based movie recommendation engine using content-based filtering with cosine similarity.

## Features
- Recommend 5 similar movies based on selected movie
- Displays movie posters from TMDB API
- Graceful error handling with placeholder images
- Request caching to minimize API calls
- Automatic retry logic for network resilience

## Setup

### Local Development

1. **Clone/Extract the project**
   ```bash
   cd movie-recom
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate  # Windows
   # or
   source .venv/bin/activate  # Mac/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your TMDB API key**
   - Get API key from https://www.themoviedb.org/settings/api
   - Update `.env` file:
     ```
     TMDB_API_KEY=your_api_key_here
     ```

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

## Deployment

### Option 1: Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to https://streamlit.io/cloud
3. Click "New app" and select your GitHub repo
4. Add secrets in Streamlit Cloud dashboard:
   - Go to App settings → Secrets
   - Add: `TMDB_API_KEY = your_api_key`
5. Deploy!

### Option 2: Railway.app

1. Connect your GitHub repo at https://railway.app
2. Add environment variable `TMDB_API_KEY`
3. Deploy automatically

### Option 3: Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

## Files

- `app.py` - Main Streamlit application
- `MRS.ipynb` - Jupyter notebook with model training
- `tmdb_5000_movies.csv` - Movie metadata
- `tmdb_5000_credits.csv` - Credits data
- `movie_dict.pkl` - Serialized recommendations model
- `requirements.txt` - Python dependencies

## Tech Stack

- **Streamlit** - Web framework
- **Pandas** - Data manipulation
- **Scikit-learn** - ML (cosine similarity)
- **Requests** - API calls
- **TMDB API** - Movie data & posters

## Notes

- The app uses environment variables for API key security
- Implements request caching and retry logic for reliability
- Falls back to placeholder images if posters cannot be fetched
- Requires pre-trained `movie_dict.pkl` file (generated from MRS.ipynb)
