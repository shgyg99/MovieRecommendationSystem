import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.config_manager import config_manager

logger = get_logger("__name__")

def popularity(data_dir: str):
    data_dir = Path(data_dir)
    logger.info(f"Calculating popularity scores from {data_dir}")
    
    ratings = pd.read_csv(f"{data_dir}/ratings.csv")
    logger.info(f"Loaded {len(ratings)} ratings")
    
    movie_stats = ratings.groupby('movieId').agg(
        rating_count=('rating', 'count'),
        rating_mean=('rating', 'mean')
    ).reset_index()
    
    movie_stats['popularity_score'] = (
        movie_stats["rating_count"] * movie_stats["rating_mean"]
    )
    popular_movies = movie_stats.sort_values(
        'popularity_score', ascending=False
    )
    
    logger.info(f"Calculated popularity for {len(popular_movies)} movies")
    return popular_movies


def recommend_popular(popular_movies, seen_movies, top_k=10):
    logger.info(f"Recommending top-{top_k}")
    
    candidates = popular_movies[
        ~popular_movies['movieId'].isin(seen_movies)
    ]
    
    recommendations = candidates.head(top_k)['movieId'].tolist()
    logger.info(f"Recommended {len(recommendations)} movies")
    
    return recommendations


if __name__ == "__main__":
    logger.info("Starting popularity-based recommendation")
    
    data_dir = config_manager.get("data.processed_path")
    popular_movies = popularity(data_dir)
    
    # Test recommendation
    rec = recommend_popular(popular_movies, [2, 222, 160], top_k=10)
    
    print(f"\nRecommendations: {rec}")
    logger.info("Recommendation completed successfully")