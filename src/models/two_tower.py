from pathlib import Path
import sys

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loading import MovieLensDataset, load_processed_data
from src.models.training import recommend_for_user, train_model
from src.utils.config_manager import config_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TwoTower(nn.Module):
    def __init__(self, num_users, num_items, num_genres, embedding_dim=32):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Embedding(num_users, embedding_dim),
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
        )
        self.item_id_embedding = nn.Embedding(num_items, embedding_dim)
        self.genre_layer = nn.Sequential(
            nn.Linear(num_genres, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.combine_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, user_ids, item_ids, item_genres=None):
        user_vec = self.user_tower(user_ids)
        item_id_vec = self.item_id_embedding(item_ids)

        if item_genres is not None:
            genre_vec = self.genre_layer(item_genres)
            item_combined = torch.cat([item_id_vec, genre_vec], dim=1)
            item_vec = self.combine_layer(item_combined)
        else:
            item_vec = item_id_vec

        return (user_vec * item_vec).sum(dim=1)

    def predict(self, user_ids, item_ids, item_genres=None):
        with torch.no_grad():
            score = self.forward(user_ids, item_ids, item_genres)
            return torch.sigmoid(score)


def main():
    data = load_processed_data()
    genre_matrix = data["genre_matrix"]
    if genre_matrix is None:
        raise FileNotFoundError(
            "genre_matrix.npy not found. Run src/data/preprocess.py before training TwoTower."
        )

    train_df = data["train_df"]
    test_df = data["test_df"]
    user_items = data["user_items"]
    idx2movie = data["idx2movie"]
    stats = data["stats"]

    train_dataset = MovieLensDataset(train_df, genre_matrix)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config_manager.get("model.batch_size", 1024),
        shuffle=True,
        num_workers=0,
    )

    model = TwoTower(
        num_users=stats["num_users"],
        num_items=stats["num_movies"],
        num_genres=genre_matrix.shape[1],
        embedding_dim=config_manager.get("model.embedding_dim", 64),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = nn.BCEWithLogitsLoss()
    logger.info(f"Using device: {device}")

    train_losses, test_aucs = train_model(
        model=model,
        train_loader=train_loader,
        test_df=test_df,
        criterion=criterion,
        genre_matrix=genre_matrix,
        epochs=config_manager.get("model.epochs", 20),
        lr=config_manager.get("model.learning_rate", 0.01),
        device=device,
        eval_every=5,
    )

    logger.info("\n" + "=" * 50)
    logger.info("Sample Recommendations")
    logger.info("=" * 50)

    sample_user_idx = 0
    recommendations = recommend_for_user(
        model=model,
        user_idx=sample_user_idx,
        user_items=user_items,
        idx2movie=idx2movie,
        num_movies=stats["num_movies"],
        n=10,
        device=device,
        genre_matrix=genre_matrix,
    )

    logger.info(f"\nTop 10 Recommendations for User {sample_user_idx}:")

    movies_df = pd.read_csv(Path(config_manager.get("data.processed_path")) / "movies.csv")

    for i, (movie_id, score) in enumerate(recommendations, 1):
        movie_title = movies_df[movies_df["movieId"] == movie_id]["title"].values
        title = movie_title[0] if len(movie_title) > 0 else f"Movie_{movie_id}"
        logger.info(f"  {i:2d}. {title:<50} (score: {score:.4f})")

    model_dir = Path(config_manager.get("models.out_path", "./models"))
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "stats": stats,
            "embedding_dim": config_manager.get("model.embedding_dim", 64),
            "train_losses": train_losses,
            "test_aucs": test_aucs,
        },
        model_dir / "tt_model.pt",
    )

    logger.info(f"\nModel saved to {model_dir / 'tt_model.pt'}")

    logger.info("\n" + "=" * 50)
    logger.info("Training Completed!")
    logger.info("=" * 50)
    logger.info(f"Final Train Loss: {train_losses[-1]:.4f}")
    if test_aucs:
        logger.info(f"Final Test AUC: {test_aucs[-1]:.4f}")

    return model, data


if __name__ == "__main__":
    model, data = main()
