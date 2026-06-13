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

logger = get_logger(__name__)

def create_negative_samples(
    user_positive_items,
    excluded_user_items,
    num_movies,
    num_negatives=4,
    random_state=42,
):
    rng = np.random.default_rng(random_state)
    negative_samples = []
    all_movies = set(range(num_movies))

    for user_idx, positive_items in user_positive_items.items():
        excluded_items = excluded_user_items.get(user_idx, set())
        negative_candidates = list(all_movies - excluded_items)

        if not negative_candidates:
            logger.warning(f"User {user_idx} has no negative candidates! (watched all {num_movies} movies)")
            continue

        num_neg_total = num_negatives * len(positive_items)

        replace = len(negative_candidates) < num_neg_total
        sampled_negatives = rng.choice(negative_candidates, size=num_neg_total, replace=replace)

        for movie_idx in sampled_negatives:
            negative_samples.append({
                'user_idx': user_idx,
                'movie_idx': movie_idx,
                'label': 0
            })

    return pd.DataFrame(negative_samples, columns=["user_idx", "movie_idx", "label"])


def create_test_with_negatives(test_positives_df, user_items, num_movies, num_negatives=4, random_state=42):
    rng = np.random.default_rng(random_state)
    test_data = []

    for _, row in test_positives_df.iterrows():
        test_data.append({
            'user_idx': row['user_idx'],
            'movie_idx': row['movie_idx'],
            'label': 1
        })

    for user_idx in test_positives_df['user_idx'].unique():
        seen_movies = user_items.get(user_idx, set())
        user_test_positives = set(test_positives_df[test_positives_df['user_idx'] == user_idx]['movie_idx'].values)

        all_movies = set(range(num_movies))
        negative_candidates = list(all_movies - seen_movies)

        if negative_candidates:
            num_neg_total = min(num_negatives * len(user_test_positives), len(negative_candidates))
            if num_neg_total > 0:
                sampled_negatives = rng.choice(negative_candidates, size=num_neg_total, replace=False)

                for movie_idx in sampled_negatives:
                    test_data.append({
                        'user_idx': user_idx,
                        'movie_idx': movie_idx,
                        'label': 0
                    })

    return pd.DataFrame(test_data, columns=["user_idx", "movie_idx", "label"])


def split_positive_interactions(positive_ratings, test_size=0.2):
    train_parts = []
    test_parts = []

    for _, user_df in positive_ratings.sort_values("timestamp").groupby("user_idx", sort=False):
        if len(user_df) == 1:
            train_parts.append(user_df)
            continue

        num_test = max(1, int(np.ceil(len(user_df) * test_size)))
        num_test = min(num_test, len(user_df) - 1)
        train_parts.append(user_df.iloc[:-num_test])
        test_parts.append(user_df.iloc[-num_test:])

    train_df = pd.concat(train_parts, ignore_index=True)
    if test_parts:
        test_df = pd.concat(test_parts, ignore_index=True)
    else:
        test_df = positive_ratings.iloc[0:0].copy()

    return train_df, test_df


def build_user_items(interactions_df):
    user_items = defaultdict(set)
    for _, row in interactions_df.iterrows():
        user_items[int(row["user_idx"])].add(int(row["movie_idx"]))
    return user_items


def create_genre_matrix(movies_df, movie2idx):
    all_genres = set()
    for genres in movies_df['genres'].str.split('|'):
        all_genres.update(genres)

    all_genres.discard('(no genres listed)')

    genre2idx = {genre: idx for idx, genre in enumerate(sorted(all_genres))}
    num_genres = len(genre2idx)

    genre_matrix = np.zeros((len(movie2idx), num_genres), dtype=np.float32)

    for _, row in movies_df.iterrows():
        movie_idx = movie2idx.get(row["movieId"])
        if movie_idx is None:
            continue

        genres = str(row["genres"]).split("|")
        for genre in genres:
            if genre in genre2idx:
                genre_matrix[movie_idx, genre2idx[genre]] = 1.0

    return genre_matrix, genre2idx, num_genres


def preprocess(raw_dir: str, dest_dir: str, num_negatives: int = 4, add_test_negatives: bool = True):

    raw_dir = Path(raw_dir)
    dest_dir = Path(dest_dir)

    logger.info(f"Starting preprocessing from {raw_dir} to {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    logger.info("Loading datasets...")
    ratings = pd.read_csv(raw_dir / "ratings.csv")
    movies = pd.read_csv(raw_dir / "movies.csv")
    tags = pd.read_csv(raw_dir / "tags.csv")
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

    logger.info("Splitting positive data per user by timestamp...")
    train_positives, test_positives = split_positive_interactions(positive_ratings)

    logger.info("Creating user item dictionaries...")
    train_user_items = build_user_items(train_positives)
    all_user_items = build_user_items(positive_ratings)

    logger.info(f"Creating training negative samples ({num_negatives} per train positive)...")
    train_negatives = create_negative_samples(
        train_user_items,
        all_user_items,
        len(movie_ids),
        num_negatives,
    )
    logger.info(f"Created {len(train_negatives)} training negative samples")

    # Combine train data (positives + negatives)
    logger.info("Combining train data...")
    train_df = pd.concat([
        train_positives[['user_idx', 'movie_idx', 'label']],
        train_negatives[['user_idx', 'movie_idx', 'label']]
    ], ignore_index=True)

    # Shuffle train data
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Create test data (with or without negatives)
    if add_test_negatives:
        logger.info("Creating test data with negative samples...")
        test_df = create_test_with_negatives(test_positives, all_user_items, len(movie_ids), num_negatives)
        logger.info(f"Test data: {len(test_df)} samples ({test_df['label'].sum()} positive, {len(test_df) - test_df['label'].sum()} negative)")
    else:
        logger.info("Creating test data (only positives)...")
        test_df = test_positives[['user_idx', 'movie_idx', 'label']].copy()
        logger.info(f"Test data: {len(test_df)} samples (all positive)")

    logger.info(f"Train: {len(train_df)} samples ({train_df['label'].sum()} positive, {len(train_df) - train_df['label'].sum()} negative)")

    genre_matrix, genre2idx, num_genres = create_genre_matrix(movies, movie2idx)
    np.save(dest_dir / "genre_matrix.npy", genre_matrix)
    with open(dest_dir / "genre2idx.pkl", 'wb') as f:
        pickle.dump(genre2idx, f)

    # Save files
    logger.info("Saving processed files...")
    train_df.to_csv(dest_dir / "train.csv", index=False)
    test_df.to_csv(dest_dir / "test.csv", index=False)

    # Save original data for reference
    movies.to_csv(dest_dir / "movies.csv", index=False)
    ratings.to_csv(dest_dir / "ratings.csv", index=False)
    tags.to_csv(dest_dir / "tags.csv", index=False)

    # Save mappings
    logger.info("Saving mappings...")
    with open(dest_dir / "user2idx.pkl", "wb") as file:
        pickle.dump(user2idx, file)
    with open(dest_dir / "idx2user.pkl", "wb") as file:
        pickle.dump(idx2user, file)
    with open(dest_dir / "movie2idx.pkl", "wb") as file:
        pickle.dump(movie2idx, file)
    with open(dest_dir / "idx2movie.pkl", "wb") as file:
        pickle.dump(idx2movie, file)

    # Save user_items dict for future use
    with open(dest_dir / "user_items.pkl", "wb") as file:
        pickle.dump(dict(all_user_items), file)

    # Save statistics
    stats = {
        'num_users': len(user2idx),
        'num_movies': len(movie2idx),
        'num_genres': num_genres,
        'num_train_samples': len(train_df),
        'num_train_positives': int(train_df['label'].sum()),
        'num_train_negatives': int(len(train_df) - train_df['label'].sum()),
        'num_test_samples': len(test_df),
        'num_test_positives': int(test_df['label'].sum()) if add_test_negatives else len(test_df),
        'num_test_negatives': int(len(test_df) - test_df['label'].sum()) if add_test_negatives else 0,
        'negative_sampling_ratio': num_negatives,
        'add_test_negatives': add_test_negatives
    }

    with open(dest_dir / "stats.pkl", "wb") as file:
        pickle.dump(stats, file)

    logger.info(f"Preprocessing completed successfully! Files saved to {dest_dir}")
    logger.info(f"Stats: {stats}")

    return stats


if __name__ == "__main__":
    raw_dir = config_manager.get("data.raw_path")
    dest_dir = config_manager.get("data.processed_path")
    num_negatives = config_manager.get("data.num_negatives", 4)

    add_test_negatives = config_manager.get("data.add_test_negatives", True)

    stats = preprocess(raw_dir, dest_dir, num_negatives, add_test_negatives)

    print("\n" + "="*50)
    print("Preprocessing Done!")
    print("="*50)
    print(f"Users: {stats['num_users']}")
    print(f"Movies: {stats['num_movies']}")
    print(f"Genres: {stats['num_genres']}")
    print(f"Train samples: {stats['num_train_samples']:,} (pos:{stats['num_train_positives']:,} / neg:{stats['num_train_negatives']:,})")
    print(f"Test samples: {stats['num_test_samples']:,} (pos:{stats['num_test_positives']:,} / neg:{stats['num_test_negatives']:,})")
    print(f"Negative ratio: 1:{stats['negative_sampling_ratio']}")
    print("="*50)
