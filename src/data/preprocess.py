import pandas as pd
import pickle
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.logger import get_logger
from src.utils.config_manager import config_manager

logger = get_logger("__name__")

def preprocess(raw_dir: str, dest_dir: str):
    
    raw_dir = Path(raw_dir)
    dest_dir = Path(dest_dir)
    
    logger.info(f"Starting preprocessing from {raw_dir} to {dest_dir}")
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    logger.info("Loading datasets...")
    ratings = pd.read_csv(f"{raw_dir}/ratings.csv")
    movies = pd.read_csv(f"{raw_dir}/movies.csv")
    tags = pd.read_csv(f"{raw_dir}/tags.csv")
    logger.info(f"Loaded {len(ratings)} ratings, {len(movies)} movies, {len(tags)} tags")

    # Create labels
    logger.info("Creating binary labels (rating >= 4 = positive)")
    ratings['label'] = (ratings['rating'] >= 4).astype(int)
    positive_ratings = ratings[ratings["label"] == 1].copy()
    logger.info(f"Found {len(positive_ratings)} positive interactions")
    
    # Create mappings
    logger.info("Creating user and movie index mappings...")
    user_ids = positive_ratings["userId"].unique()
    movie_ids = positive_ratings["movieId"].unique()
    
    user2idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie2idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    
    idx2user = {idx: user_id for user_id, idx in user2idx.items()}
    idx2movie = {idx: movie_id for movie_id, idx in movie2idx.items()}
    
    positive_ratings['user_idx'] = positive_ratings['userId'].map(user2idx)
    positive_ratings['movie_idx'] = positive_ratings['movieId'].map(movie2idx)
    
    # Split data
    logger.info("Splitting data (80/20)...")
    positive_ratings = positive_ratings.sort_values('timestamp')
    split_idx = int(len(positive_ratings) * 0.8)
    
    train_df = positive_ratings.iloc[:split_idx]
    test_df = positive_ratings.iloc[split_idx:]
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Save files
    logger.info("Saving processed files...")
    train_df.to_csv(f"{dest_dir}/train.csv", index=False)
    test_df.to_csv(f"{dest_dir}/test.csv", index=False)
    movies.to_csv(f"{dest_dir}/movies.csv", index=False)
    ratings.to_csv(f"{dest_dir}/ratings.csv", index=False)
    tags.to_csv(f"{dest_dir}/tags.csv", index=False)
    
    
    with open(f'{dest_dir}/user2idx.pkl', 'wb') as file:
        pickle.dump(user2idx, file)
        
    with open(f'{dest_dir}/movie2idx.pkl', 'wb') as file:
        pickle.dump(movie2idx, file)
    
    logger.info(f"Preprocessing completed successfully! Files saved to {dest_dir}")


if __name__ == "__main__":
    raw_dir = config_manager.get("data.raw_path")
    dest_dir = config_manager.get("data.processed_path")
    preprocess(raw_dir, dest_dir)