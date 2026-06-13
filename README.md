# Movie Recommendation

A MovieLens-based recommendation project with data ingestion, preprocessing, classical baselines, and neural recommendation models.

The project currently uses the MovieLens `ml-latest-small` dataset and supports:

- Popularity-based recommendations
- Content-based recommendations using genres and tags
- Matrix Factorization with PyTorch
- Two-Tower neural recommendations using user IDs, item IDs, and movie genre features

## Project Structure

```text
MovieRecommendation/
|-- config/
|   `-- config.yaml
|-- data/
|   |-- raw/
|   `-- processed/
|-- logs/
|-- models/
|-- notebooks/
|-- reports/
|-- src/
|   |-- data/
|   |   |-- ingestion.py
|   |   |-- loading.py
|   |   `-- preprocess.py
|   |-- evaluation/
|   |   `-- metrics.py
|   |-- models/
|   |   |-- content_based.py
|   |   |-- matrix_factorization.py
|   |   |-- popularity.py
|   |   |-- training.py
|   |   `-- two_tower.py
|   `-- utils/
|       |-- config_manager.py
|       `-- logger.py
|-- requirements.txt
`-- README.md
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Configuration

Main settings live in `config/config.yaml`:

```yaml
data:
  url: "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
  raw_path: "./data/raw"
  processed_path: "./data/processed"
  num_negatives: 4
  add_test_negatives: true

model:
  batch_size: 1024
  embedding_dim: 64
  epochs: 20
  learning_rate: 0.001

models:
  out_path: "./models"
```

Relative paths are resolved from the project root by `ConfigManager`.

## Data Pipeline

Download and extract MovieLens:

```powershell
python src\data\ingestion.py
```

Preprocess the dataset:

```powershell
python src\data\preprocess.py
```

Preprocessing creates:

- `train.csv` and `test.csv`
- user and movie index mappings
- `user_items.pkl`
- `stats.pkl`
- `genre_matrix.npy`
- `genre2idx.pkl`

The split is user-aware and chronological: each user's latest positive interactions are held out for testing when possible. Negative samples are generated from unseen items while excluding known positive interactions.

## Models

### Popularity Baseline

Ranks movies by:

```text
rating_count * rating_mean
```

Run:

```powershell
python src\models\popularity.py
```

### Content-Based Model

Builds TF-IDF features from movie genres and tags, then computes item-item cosine similarity.

Run:

```powershell
python src\models\content_based.py
```

Output:

```text
models/content_based.pkl
```

### Matrix Factorization

PyTorch model with user embeddings, item embeddings, and bias terms.

Run:

```powershell
python src\models\matrix_factorization.py
```

Output:

```text
models/mf_model.pt
```

### Two-Tower Model

PyTorch model with:

- User tower from user IDs
- Item ID embedding
- Genre feature tower
- Combined item representation

Run:

```powershell
python src\models\two_tower.py
```

Output:

```text
models/tt_model.pt
```

## Evaluation

Available ranking metrics in `src/evaluation/metrics.py`:

- `precision_at_k`
- `recall_at_k`
- `hit_rate_at_k`
- `ndcg_at_k`

Neural models currently report AUC during training using the sampled test set.

## Useful Checks

Compile all source files:

```powershell
python -m compileall -q src
```

Quick import check:

```powershell
python -c "from src.models.matrix_factorization import MatrixFactorization; from src.models.two_tower import TwoTower; print('ok')"
```

## Notes

- `data/`, `models/`, and `logs/` are generated runtime artifacts and are ignored by Git.
- The content-based model can be large because it stores the full item similarity matrix.
- Run preprocessing again after changing negative sampling, genre processing, or split logic.
