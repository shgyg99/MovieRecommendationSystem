import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.logger import get_logger
from src.utils.config_manager import config_manager

logger = get_logger("__name__")

def create_negative_samples(user_items_dict, num_movies, num_negatives=4):

    negative_samples = []
    
    for user_idx, positive_items in user_items_dict.items():
        all_movies = set(range(num_movies))
        negative_candidates = list(all_movies - positive_items)
        
        if not negative_candidates:
            continue
        
        num_neg_total = num_negatives * len(positive_items)
        sampled_negatives = np.random.choice(negative_candidates, size=num_neg_total, replace=True)
        
        for movie_idx in sampled_negatives:
            negative_samples.append({
                'user_idx': user_idx,
                'movie_idx': movie_idx,
                'label': 0
            })
    
    return pd.DataFrame(negative_samples)

def preprocess(raw_dir: str, dest_dir: str, num_negatives: int = 4):
    
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

    # Create labels (implicit feedback: rating >= 4 = positive)
    logger.info("Creating binary labels (rating >= 4 = positive)")
    ratings['label'] = (ratings['rating'] >= 4).astype(int)
    positive_ratings = ratings[ratings["label"] == 1].copy()
    logger.info(f"Found {len(positive_ratings)} positive interactions")
    
    # Create mappings (ONLY on positive interactions)
    logger.info("Creating user and movie index mappings...")
    user_ids = positive_ratings["userId"].unique()
    movie_ids = positive_ratings["movieId"].unique()
    
    user2idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie2idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    
    idx2user = {idx: user_id for user_id, idx in user2idx.items()}
    idx2movie = {idx: movie_id for movie_id, idx in movie2idx.items()}
    
    # Add indices to positive ratings
    positive_ratings['user_idx'] = positive_ratings['userId'].map(user2idx)
    positive_ratings['movie_idx'] = positive_ratings['movieId'].map(movie2idx)
    
    # Create user_items dictionary for negative sampling
    logger.info("Creating user items dictionary...")
    user_items = defaultdict(set)
    for _, row in positive_ratings.iterrows():
        user_items[row['user_idx']].add(row['movie_idx'])
    
    # Create negative samples
    logger.info(f"Creating negative samples ({num_negatives} per positive)...")
    negative_df = create_negative_samples(user_items, len(movie_ids), num_negatives)
    logger.info(f"Created {len(negative_df)} negative samples")
    
    # Split positive data (80/20 based on timestamp)
    logger.info("Splitting positive data (80/20 based on timestamp)...")
    positive_ratings = positive_ratings.sort_values('timestamp')
    split_idx = int(len(positive_ratings) * 0.8)
    
    train_positives = positive_ratings.iloc[:split_idx]
    test_positives = positive_ratings.iloc[split_idx:]
    
    # Split negatives for train only (negatives only in training)
    logger.info("Splitting negatives for training...")
    train_negatives = negative_df.sample(frac=0.8, random_state=42)
    
    # Combine train data (positives + negatives)
    logger.info("Combining train data...")
    train_df = pd.concat([
        train_positives[['user_idx', 'movie_idx', 'label']],
        train_negatives[['user_idx', 'movie_idx', 'label']]
    ], ignore_index=True)
    
    # Shuffle train data
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Test data (only positives)
    test_df = test_positives[['user_idx', 'movie_idx', 'label']].copy()
    
    logger.info(f"Train: {len(train_df)} samples ({len(train_positives)} positive, {len(train_negatives)} negative)")
    logger.info(f"Test: {len(test_df)} samples (all positive)")
    
    # Save files
    logger.info("Saving processed files...")
    train_df.to_csv(f"{dest_dir}/train.csv", index=False)
    test_df.to_csv(f"{dest_dir}/test.csv", index=False)
    
    # Save original data for reference
    movies.to_csv(f"{dest_dir}/movies.csv", index=False)
    ratings.to_csv(f"{dest_dir}/ratings.csv", index=False)
    tags.to_csv(f"{dest_dir}/tags.csv", index=False)
    
    # Save mappings
    logger.info("Saving mappings...")
    with open(f'{dest_dir}/user2idx.pkl', 'wb') as file:
        pickle.dump(user2idx, file)    
    with open(f'{dest_dir}/idx2user.pkl', 'wb') as file:
        pickle.dump(idx2user, file)
    with open(f'{dest_dir}/movie2idx.pkl', 'wb') as file:
        pickle.dump(movie2idx, file)
    with open(f'{dest_dir}/idx2movie.pkl', 'wb') as file:
        pickle.dump(idx2movie, file)
    
    # Save user_items dict for future use
    with open(f'{dest_dir}/user_items.pkl', 'wb') as file:
        pickle.dump(dict(user_items), file)
    
    # Save statistics
    stats = {
        'num_users': len(user2idx),
        'num_movies': len(movie2idx),
        'num_train_samples': len(train_df),
        'num_train_positives': len(train_positives),
        'num_train_negatives': len(train_negatives),
        'num_test_samples': len(test_df),
        'negative_sampling_ratio': num_negatives
    }
    
    with open(f'{dest_dir}/stats.pkl', 'wb') as file:
        pickle.dump(stats, file)
    
    logger.info(f"Preprocessing completed successfully! Files saved to {dest_dir}")
    logger.info(f"Stats: {stats}")
    
    return stats


if __name__ == "__main__":
    raw_dir = config_manager.get("data.raw_path")
    dest_dir = config_manager.get("data.processed_path")
    
    num_negatives = config_manager.get("data.num_negatives", 4)
    
    stats = preprocess(raw_dir, dest_dir, num_negatives)
    
    print("\n" + "="*50)
    print("✅ Preprocessing Done!")
    print("="*50)
    print(f"Users: {stats['num_users']}")
    print(f"Movies: {stats['num_movies']}")
    print(f"Train samples: {stats['num_train_samples']:,} (pos:{stats['num_train_positives']:,} / neg:{stats['num_train_negatives']:,})")
    print(f"Test samples: {stats['num_test_samples']:,}")
    print(f"Negative ratio: 1:{stats['negative_sampling_ratio']}")
    print("="*50)