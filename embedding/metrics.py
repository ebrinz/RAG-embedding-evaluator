"""Metrics for evaluating embedding model performance."""
import numpy as np
from typing import List


def calculate_mrr(rankings: List[int]) -> float:
    """
    Calculate Mean Reciprocal Rank.
    
    Args:
        rankings: List of relevance scores in ranked order
    
    Returns:
        float: MRR score
    """
    for rank, item in enumerate(rankings, 1):
        if item == 1:  # Found relevant document
            return 1.0 / rank
    return 0.0


def calculate_ndcg(rankings: List[int], k: int = None) -> float:
    """
    Calculate normalized Discounted Cumulative Gain.
    
    Args:
        rankings: List of relevance scores in ranked order
        k: Number of positions to consider (optional)
    
    Returns:
        float: nDCG score
    """
    rankings = rankings[:k] if k else rankings
    dcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rankings))
    ideal_rankings = sorted(rankings, reverse=True)
    idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal_rankings))
    return dcg / idcg if idcg > 0 else 0.0