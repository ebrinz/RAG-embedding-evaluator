"""Tests for embedding evaluation metrics."""
import pytest
from ..metrics import calculate_mrr, calculate_ndcg

@pytest.fixture
def high_relevance_ranking():
    return [1, 1, 1, 0, 0]  # First three items are relevant

@pytest.fixture
def mixed_relevance_ranking():
    return [0, 1, 1, 0, 1]  # Scattered relevant items

@pytest.fixture
def low_relevance_ranking():
    return [0, 0, 0, 0, 1]  # Only last item relevant

@pytest.fixture
def relevance_patterns():
    """Different relevance patterns for testing."""
    return [
        [1, 0, 0, 0, 0],  # Single relevant at start
        [1, 1, 0, 0, 0],  # Multiple relevant at start
        [1, 0, 1, 0, 1],  # Scattered relevance pattern
    ]

def test_mrr_top_result():
    """Test MRR with relevant item at top."""
    rankings = [1, 0, 0, 0]
    assert calculate_mrr(rankings) == 1.0

def test_mrr_middle_result():
    """Test MRR with relevant item in middle."""
    rankings = [0, 1, 0, 0]
    assert calculate_mrr(rankings) == 0.5

def test_mrr_no_relevant():
    """Test MRR with no relevant items."""
    rankings = [0, 0, 0, 0]
    assert calculate_mrr(rankings) == 0.0

def test_ndcg_optimal(high_relevance_ranking):
    """Test nDCG with optimal ranking."""
    assert calculate_ndcg(high_relevance_ranking) == 1.0

def test_ndcg_suboptimal(mixed_relevance_ranking):
    """Test nDCG with suboptimal ranking."""
    score = calculate_ndcg(mixed_relevance_ranking)
    assert 0 < score < 1
    assert score == pytest.approx(0.7666, rel=1e-3)

def test_ndcg_minimal(low_relevance_ranking):
    """Test nDCG with minimal relevance."""
    score = calculate_ndcg(low_relevance_ranking)
    assert score < 0.5

def test_ndcg_empty():
    """Test nDCG with empty ranking."""
    assert calculate_ndcg([]) == 0.0

def test_ndcg_truncation():
    """Test nDCG with position cutoff."""
    rankings = [0, 1, 1, 0, 1]
    assert calculate_ndcg(rankings, k=3) != calculate_ndcg(rankings)

def test_ndcg_full_relevance():
    """Test nDCG with all relevant items."""
    rankings = [1, 1, 1]
    assert calculate_ndcg(rankings) == 1.0

def test_ndcg_no_relevance():
    """Test nDCG with no relevant items."""
    rankings = [0, 0, 0]
    assert calculate_ndcg(rankings) == 0.0