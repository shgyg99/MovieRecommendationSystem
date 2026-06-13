from pathlib import Path
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.config_manager import config_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_processed_data():
    processed_dir = Path(config_manager.get("data.processed_path"))

    logger.info(f"Loading data from {processed_dir}")

    train_df = pd.read_csv(processed_dir / "train.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")

    with open(processed_dir / "user2idx.pkl", 'rb') as f:
        user2idx = pickle.load(f)

    with open(processed_dir / "movie2idx.pkl", 'rb') as f:
        movie2idx = pickle.load(f)

    with open(processed_dir / "idx2movie.pkl", 'rb') as f:
        idx2movie = pickle.load(f)

    with open(processed_dir / "user_items.pkl", 'rb') as f:
        user_items = pickle.load(f)

    with open(processed_dir / "stats.pkl", 'rb') as f:
        stats = pickle.load(f)

    genre_matrix_path = processed_dir / "genre_matrix.npy"
    genre_matrix = np.load(genre_matrix_path) if genre_matrix_path.exists() else None

    logger.info(f"Train: {len(train_df)} samples (pos: {train_df['label'].sum()}, neg: {len(train_df) - train_df['label'].sum()})")
    logger.info(f"Test: {len(test_df)} samples (pos: {test_df['label'].sum()}, neg: {len(test_df) - test_df['label'].sum()})")
    logger.info(f"Users: {stats['num_users']}, Movies: {stats['num_movies']}")

    return {
        'train_df': train_df,
        'test_df': test_df,
        'user2idx': user2idx,
        'movie2idx': movie2idx,
        'idx2movie': idx2movie,
        'user_items': user_items,
        'stats': stats,
        'genre_matrix': genre_matrix
    }


class MovieLensDataset(Dataset):
    def __init__(self, df, genre_matrix=None):
        self.users = torch.LongTensor(df['user_idx'].values.copy())
        self.items = torch.LongTensor(df['movie_idx'].values.copy())
        self.labels = torch.FloatTensor(df['label'].values.copy())

        self.genre_matrix = genre_matrix
        if genre_matrix is not None:
            self.genres = torch.FloatTensor(genre_matrix[self.items.numpy()])

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        if self.genre_matrix is not None:
            return self.users[idx], self.items[idx], self.genres[idx], self.labels[idx]

        return self.users[idx], self.items[idx], self.labels[idx]
