import pandas as pd
import pickle
from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.config_manager import config_manager
from src.utils.logger import get_logger
from src.models.training import recommend_for_user, train_model

logger = get_logger(__name__)


class MovieLensDataset(Dataset):
    def __init__(self, df):
        self.users = torch.LongTensor(df['user_idx'].values.copy())
        self.items = torch.LongTensor(df['movie_idx'].values.copy())
        self.labels = torch.FloatTensor(df['label'].values.copy())

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids, item_ids):
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)

        score = (user_vec * item_vec).sum(dim=1)
        score = score + self.user_bias(user_ids).squeeze() + self.item_bias(item_ids).squeeze()

        return score

    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            score = self.forward(user_ids, item_ids)
            return torch.sigmoid(score)


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
        'stats': stats
    }




def main():
    data = load_processed_data()

    train_df = data['train_df']
    test_df = data['test_df']
    user_items = data['user_items']
    idx2movie = data['idx2movie']
    stats = data['stats']

    train_dataset = MovieLensDataset(train_df)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config_manager.get("model.batch_size", 1024),
        shuffle=True,
        num_workers=0
    )

    model = MatrixFactorization(
        num_users=stats['num_users'],
        num_items=stats['num_movies'],
        embedding_dim=config_manager.get("model.embedding_dim", 64)
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.BCEWithLogitsLoss()
    logger.info(f"Using device: {device}")

    train_losses, test_aucs = train_model(
        model=model,
        train_loader=train_loader,
        test_df=test_df,
        criterion=criterion,
        epochs=config_manager.get("model.epochs", 20),
        lr=config_manager.get("model.learning_rate", 0.01),
        device=device,
        eval_every=5
    )

    logger.info("\n" + "="*50)
    logger.info("Sample Recommendations")
    logger.info("="*50)

    sample_user_idx = 0
    recommendations = recommend_for_user(
        model=model,
        user_idx=sample_user_idx,
        user_items=user_items,
        idx2movie=idx2movie,
        num_movies=stats['num_movies'],
        n=10,
        device=device
    )

    logger.info(f"\nTop 10 Recommendations for User {sample_user_idx}:")

    movies_df = pd.read_csv(Path(config_manager.get("data.processed_path")) / "movies.csv")

    for i, (movie_id, score) in enumerate(recommendations, 1):
        movie_title = movies_df[movies_df['movieId'] == movie_id]['title'].values
        title = movie_title[0] if len(movie_title) > 0 else f"Movie_{movie_id}"
        logger.info(f"  {i:2d}. {title:<50} (score: {score:.4f})")

    model_dir = Path(config_manager.get("models.out_path", "./models"))
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'stats': stats,
        'embedding_dim': config_manager.get("model.embedding_dim", 64),
        'train_losses': train_losses,
        'test_aucs': test_aucs
    }, model_dir / "mf_model.pt")

    logger.info(f"\nModel saved to {model_dir / 'mf_model.pt'}")

    logger.info("\n" + "="*50)
    logger.info("Training Completed!")
    logger.info("="*50)
    logger.info(f"Final Train Loss: {train_losses[-1]:.4f}")
    if test_aucs:
        logger.info(f"Final Test AUC: {test_aucs[-1]:.4f}")

    return model, data


if __name__ == "__main__":
    model, data = main()
