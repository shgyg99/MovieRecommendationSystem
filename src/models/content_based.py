import pandas as pd
from pathlib import Path
import sys
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.config_manager import config_manager

logger = get_logger("__name__")

def content_based(data_dir):
    """
    Build content-based recommendation model using movie genres and tags
    """
    logger.info(f"Building content-based model from {data_dir}")
    
    # Load data
    movies = pd.read_csv(f"{data_dir}/movies.csv")
    tags = pd.read_csv(f"{data_dir}/tags.csv")
    logger.info(f"Loaded {len(movies)} movies and {len(tags)} tags")
    
    # Create content from genres
    movies['content'] = movies['genres'].str.replace("|", " ", regex=False)
    
    # Add tags to content
    tags_grouped = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()
    movies = movies.merge(tags_grouped, on="movieId", how="left")
    movies['tag'] = movies['tag'].fillna("")
    movies['content'] = movies['genres'].str.replace("|", " ", regex=False) + " " + movies["tag"]
    logger.info("Created text content for each movie")
    
    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words="english")
    item_features = tfidf.fit_transform(movies['content'])
    logger.info(f"Created TF-IDF matrix with shape {item_features.shape}")
    
    # Calculate similarity matrix
    item_similarity = cosine_similarity(item_features)
    logger.info(f"Calculated item similarity matrix with shape {item_similarity.shape}")
    
    return {
        'movies': movies,
        'item_similarity': item_similarity,
        'tfidf': tfidf,
        'item_features': item_features
    }
    
def recommend_similar(movie_id, model_data, top_k=10):
    """Simple function to recommend similar movies"""
    movies = model_data['movies']
    similarity = model_data['item_similarity']
    
    # Find movie index
    idx = movies[movies['movieId'] == movie_id].index[0]
    
    # Get similar movies
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_k+1]
    
    # Return movie IDs
    similar_indices = [i[0] for i in scores]
    return movies.iloc[similar_indices][['movieId', 'title', 'genres']]

def save_model(model_data, save_path):
    """Save the content-based model"""
    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)
    logger.info(f"Model saved to {save_path}")

def load_model(load_path):
    """Load the content-based model"""
    with open(load_path, 'rb') as f:
        model_data = pickle.load(f)
    logger.info(f"Model loaded from {load_path}")
    return model_data


if __name__ == "__main__":
    data_dir = config_manager.get("data.processed_path")
    model_dir = config_manager.get("models.out_path")
    model_path = Path(f"{model_dir}/content_based.pkl")
    if not model_path.exists():
        model_data = content_based(data_dir)
        save_model(model_data, model_path)
    else:
        load_model(model_path)
    
    # Test with a movie
    recommendations = recommend_similar(1, model_data, top_k=5)
    print("Movies similar to movie 1:")
    print(recommendations)


    