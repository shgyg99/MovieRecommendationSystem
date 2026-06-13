from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_model(
    model,
    train_loader,
    test_df,
    criterion,
    epochs: int = 20,
    lr: float = 0.01,
    device: str = "cpu",
    eval_every: int = 5,
    genre_matrix=None,
):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    train_losses = []
    test_aucs = []

    logger.info("Starting training on %s", device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            if len(batch) == 4:
                users, items, genres, labels = batch
                genres = genres.to(device)
            else:
                users, items, labels = batch
                genres = None

            users = users.to(device)
            items = items.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            scores = model(users, items, genres) if genres is not None else model(users, items)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        train_losses.append(avg_loss)

        if (epoch + 1) % eval_every == 0:
            auc = evaluate_model(model, test_df, device, genre_matrix=genre_matrix)
            test_aucs.append(auc)
            logger.info(
                "Epoch %s/%s - Loss: %.4f, Test AUC: %.4f",
                epoch + 1,
                epochs,
                avg_loss,
                auc,
            )
        else:
            logger.info("Epoch %s/%s - Loss: %.4f", epoch + 1, epochs, avg_loss)

    return train_losses, test_aucs


def evaluate_model(model, test_df, device: str = "cpu", genre_matrix=None) -> float:
    model.eval()

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for _, row in test_df.iterrows():
            user = torch.LongTensor([int(row["user_idx"])]).to(device)
            movie_idx = int(row["movie_idx"])
            item = torch.LongTensor([movie_idx]).to(device)
            label = row["label"]

            if genre_matrix is not None:
                genres = torch.FloatTensor(genre_matrix[[movie_idx]]).to(device)
                score = model.predict(user, item, genres)
            else:
                score = model.predict(user, item)
            all_scores.append(score.cpu().item())
            all_labels.append(label)

    if len(set(all_labels)) < 2:
        logger.warning("Only one class present in test data. Returning 0.5")
        return 0.5

    return roc_auc_score(all_labels, all_scores)


def recommend_for_user(
    model,
    user_idx: int,
    user_items: Dict[int, Set[int]],
    idx2movie: Dict[int, int],
    num_movies: int,
    n: int = 10,
    device: str = "cpu",
    genre_matrix=None,
) -> List[Tuple[int, float]]:
    model.eval()

    seen_movies = user_items.get(user_idx, set())

    all_movies = torch.LongTensor(list(range(num_movies))).to(device)
    user_tensor = torch.LongTensor([user_idx] * num_movies).to(device)

    with torch.no_grad():
        if genre_matrix is not None:
            genres = torch.FloatTensor(genre_matrix).to(device)
            scores = model.predict(user_tensor, all_movies, genres).cpu().numpy()
        else:
            scores = model.predict(user_tensor, all_movies).cpu().numpy()

    recommendations = []
    for movie_idx in np.argsort(scores)[::-1]:
        if movie_idx not in seen_movies:
            movie_id = idx2movie[movie_idx]
            recommendations.append((movie_id, float(scores[movie_idx])))
            if len(recommendations) >= n:
                break

    return recommendations
