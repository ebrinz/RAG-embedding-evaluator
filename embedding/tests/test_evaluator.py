




"""Tests for the embedding evaluator."""
import pytest
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from ..evaluator import EmbeddingEvaluator

# Paths
TEST_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "test_config.yaml"

@pytest.fixture
def sample_movie_data():
    """Sample movie plot summaries and queries."""
    return {
        'queries': [
            "A science fiction movie about time travel and paradoxes",
            "A romantic comedy about mistaken identity",
            "An action movie with car chases and explosions"
        ],
        'documents': [
            "A scientist invents a time machine and travels to the future, only to discover the consequences of his actions have created a paradox that threatens his existence.",
            "Two strangers meet at a coffee shop and fall in love, but a case of mistaken identity leads to hilarious complications.",
            "A former spy must race against time in high-speed car chases while trying to prevent a terrorist attack.",
            "A family drama about coming to terms with loss and grief.",
            "A documentary about the migration patterns of butterflies."
        ],
        'relevance_scores': [
            [1, 0, 0, 0, 0],  # Relevant to sci-fi query
            [0, 1, 0, 0, 0],  # Relevant to rom-com query
            [0, 0, 1, 0, 0]   # Relevant to action query
        ]
    }

@pytest.fixture
def config():
    """Load test configuration."""
    with open(TEST_CONFIG_PATH) as f:
        return yaml.safe_load(f)

def test_evaluator_initialization(config):
    """Test evaluator initialization with config."""
    evaluator = EmbeddingEvaluator(TEST_CONFIG_PATH)
    assert len(evaluator.models) == len(config['models'])
    assert evaluator.batch_size == config['batch_size']
    assert evaluator.device == config['device']

def test_evaluate_model(config, sample_movie_data):
    """Test single model evaluation."""
    evaluator = EmbeddingEvaluator(TEST_CONFIG_PATH)
    results = evaluator.evaluate_model(
        config['models'][0],
        sample_movie_data['queries'],
        sample_movie_data['documents'],
        sample_movie_data['relevance_scores']
    )
    
    # Check result structure
    assert 'model' in results
    assert 'description' in results
    
    # Check configured metrics are present
    for metric in config['metrics']:
        metric_key = f"avg_{metric}"
        assert metric_key in results
        assert 0 <= results[metric_key] <= 1

def test_run_evaluation(config, sample_movie_data):
    """Test full evaluation pipeline."""
    evaluator = EmbeddingEvaluator(TEST_CONFIG_PATH)
    results_df = evaluator.run_evaluation(
        sample_movie_data['queries'],
        sample_movie_data['documents'],
        sample_movie_data['relevance_scores']
    )
    
    # Check DataFrame structure
    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) == len(config['models'])
    
    # Check all configured metrics are present
    expected_columns = ['model', 'description'] + [f"avg_{metric}" for metric in config['metrics']]
    assert all(col in results_df.columns for col in expected_columns)

def test_invalid_config_path():
    """Test error handling with invalid config path."""
    with pytest.raises(FileNotFoundError):
        EmbeddingEvaluator("nonexistent_config.yaml")

def test_missing_required_fields():
    """Test error handling with invalid config structure."""
    invalid_config = Path(__file__).parent / "invalid_config.yaml"
    invalid_config.write_text("models: []")  # Missing required fields
    
    with pytest.raises(ValueError):
        EmbeddingEvaluator(invalid_config)
    
    invalid_config.unlink()  # Clean up

def test_document_length_limits(config, sample_movie_data):
    """Test handling of document length limits from config."""
    # Create a document that exceeds max_length
    max_length = config.get('test_data', {}).get('max_length', 512)
    long_doc = " ".join(["word"] * max_length)
    
    test_data = {
        'queries': ["test query"],
        'documents': [long_doc],
        'relevance_scores': [[1]]
    }
    
    evaluator = EmbeddingEvaluator(TEST_CONFIG_PATH)
    results = evaluator.evaluate_model(
        config['models'][0],
        test_data['queries'],
        test_data['documents'],
        test_data['relevance_scores']
    )
    
    assert 'avg_mrr' in results  # Should handle long document without errors





############## older

@pytest.mark.parametrize("query_idx", [0, 1, 2])
def test_genre_specific_evaluation(model_configs, sample_movie_data, query_idx):
    """Test evaluation for specific movie genres."""
    evaluator = EmbeddingEvaluator([model_configs[0]])  # Test with one model
    results = evaluator.evaluate_model(
        model_configs[0]['name'],
        [sample_movie_data['queries'][query_idx]],
        sample_movie_data['documents'],
        [sample_movie_data['relevance_scores'][query_idx]]
    )
    
    assert results['avg_mrr'] >= 0  # Should find at least some relevance

def test_error_handling(model_configs):
    """Test error handling with invalid inputs."""
    evaluator = EmbeddingEvaluator(model_configs)
    
    with pytest.raises(Exception):
        # Test with mismatched query and relevance score lengths
        evaluator.evaluate_model(
            model_configs[0]['name'],
            ["query1", "query2"],
            ["doc1"],
            [[1]]
        )

def test_empty_documents(model_configs):
    """Test handling of empty document list."""
    evaluator = EmbeddingEvaluator(model_configs)
    
    with pytest.raises(Exception):
        evaluator.evaluate_model(
            model_configs[0]['name'],
            ["query"],
            [],
            [[]]
        )

def test_large_document_handling(model_configs):
    """Test handling of large documents."""
    long_text = " ".join(["word"] * 1000)  # Create a long document
    evaluator = EmbeddingEvaluator([model_configs[0]])
    
    results = evaluator.evaluate_model(
        model_configs[0]['name'],
        ["short query"],
        [long_text],
        [[1]]
    )
    
    assert 'avg_mrr' in results  # Should process without errors
