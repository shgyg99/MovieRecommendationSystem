import numpy as np

def precision_at_k(recommended_items, relevant_items, k):
    if k <= 0:
        return 0.0

    recommended_at_k = recommended_items[:k]
    hits = len(set(recommended_at_k) & set(relevant_items))
    return hits / k

def recall_at_k(recommended_items, relevant_items, k):
    if k <= 0:
        return 0.0

    recommended_at_k = recommended_items[:k]
    hits = len(set(recommended_at_k) & set(relevant_items))
    if len(relevant_items) == 0:
        return 0.0
    return hits / len(relevant_items)

def hit_rate_at_k(recommended_items, relevant_items, k):
    if k <= 0:
        return 0

    recommended_at_k = recommended_items[:k]
    return int(len(set(recommended_at_k) & set(relevant_items)) > 0)

def ndcg_at_k(recommended_items, relevant_items, k):
    if k <= 0:
        return 0.0

    recommended_at_k = recommended_items[:k]
    dcg = 0.0

    for i, item in enumerate(recommended_at_k):
        if item in relevant_items:
            dcg += 1 / np.log2(i + 2)

    ideal_hits = min(len(relevant_items), k)
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0
    return dcg / idcg
