def precision_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    return len(set(recommended_k) & set(relevant)) / k

def recall_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    return len(set(recommended_k) & set(relevant)) / len(relevant)

def ndcg_at_k(recommended, relevant, k=10):
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1 / log2(i + 2)
    ideal_dcg = sum([1 / log2(i + 2) for i in range(min(len(relevant), k))])
    return dcg / ideal_dcg if ideal_dcg > 0 else 0