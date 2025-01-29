"""Tests for embedding evaluation metrics."""
import pytest
from ..metrics import calculate_mrr, calculate_ndcg

# Test data simulating movie plot relevance scenarios
@pytest.fixture
def perfect_ranking():
    return [1, 1, 1, 0, 0]  # First three are relevant

@pytest.fixture
def imperfect_ranking():
    return [0, 1, 1, 0, 1]  # Relevant docs at positions 2, 3, and 5

@pytest.fixture
def worst_ranking():
    return [0, 0, 0, 0, 1]  # Relevant doc at last position

@pytest.fixture
def binary_relevance_queries():
    """
    Simulated movie plot relevance scenarios.
    1 = relevant (similar plot/theme), 0 = irrelevant
    """
    return [
        # Perfect match (e.g., exact genre and theme match)
        [1, 0, 0, 0, 0],
        # Multiple relevant results (e.g., similar genre films)
        [1, 1, 0, 0, 0],
        # Scattered relevance (e.g., mixed genre/theme matches)
        [1, 0, 1, 0, 1],
    ]

def test_mrr_perfect_first():
    """Test MRR when first result is relevant."""
    rankings = [1, 0, 0, 0]
    assert calculate_mrr(rankings) == 1.0

def test_mrr_second_position():
    """Test MRR when relevant result is second."""
    rankings = [0, 1, 0, 0]
    assert calculate_mrr(rankings) == 0.5

def test_mrr_no_relevant():
    """Test MRR when no relevant results found."""
    rankings = [0, 0, 0, 0]
    assert calculate_mrr(rankings) == 0.0

def test_ndcg_perfect(perfect_ranking):
    """Test nDCG with perfect ranking."""
    assert calculate_ndcg(perfect_ranking) == 1.0

def test_ndcg_imperfect(imperfect_ranking):
    """Test nDCG with imperfect ranking."""
    score = calculate_ndcg(imperfect_ranking)
    assert 0 < score < 1  # Score should be between 0 and 1
    assert score == pytest.approx(0.7666, rel=1e-3)

def test_ndcg_worst(worst_ranking):
    """Test nDCG with worst possible ranking."""
    score = calculate_ndcg(worst_ranking)
    assert score < 0.5  # Should be a low score

def test_ndcg_empty():
    """Test nDCG with empty ranking."""
    assert calculate_ndcg([]) == 0.0

def test_ndcg_k_truncation():
    """Test nDCG with k-truncation."""
    rankings = [0, 1, 1, 0, 1]
    assert calculate_ndcg(rankings, k=3) != calculate_ndcg(rankings)  # Should differ when truncated

def test_ndcg_all_relevant():
    """Test nDCG when all documents are relevant."""
    rankings = [1, 1, 1]
    assert calculate_ndcg(rankings) == 1.0

def test_ndcg_all_irrelevant():
    """Test nDCG when no documents are relevant."""
    rankings = [0, 0, 0]
    assert calculate_ndcg(rankings) == 0.0