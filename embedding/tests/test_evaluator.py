"""Tests for the embedding evaluator."""
import pytest
from pathlib import Path
import pandas as pd
import numpy as np
import json
import yaml
from ..evaluator import EmbeddingEvaluator

# Paths
TEST_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "embedding_eval.yaml"

@pytest.fixture
def test_cases():
    """Load test cases from configuration."""
    with open(TEST_CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    
    test_cases_path = Path(config['dataset']['evaluation']['test_cases_path'])
    if not test_cases_path.is_absolute():
        test_cases_path = Path(TEST_CONFIG_PATH).parent / test_cases_path
    
    with open(test_cases_path) as f:
        return json.load(f)

@pytest.fixture
def sample_evaluation_data(test_cases):
    """Generate evaluation data from test cases."""
    return {
        'queries': [case['query'] for case in test_cases['cases']],
        'content': [case['content'] for case in test_cases['cases']],
        'relevance_scores': [
            [case['expected_relevance']] for case in test_cases['cases']
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

def test_evaluate_model(config, sample_evaluation_data):
    """Test single model evaluation."""
    evaluator = EmbeddingEvaluator(TEST_CONFIG_PATH)
    results = evaluator.evaluate_model(
        config['models'][0],
        sample_evaluation_data['queries'],
        sample_evaluation_data['content'],
        sample_evaluation_data['relevance_scores']
    )
    
    # Check result structure
    assert 'model' in results
    assert 'description' in results
    
    # Check configured metrics are present
    for metric in config['metrics']:
        metric_key = f"avg_{metric}"
        assert metric_key in results
        assert 0 <= results[metric_key] <= 1

def test_run_evaluation(config, sample_evaluation_data):
    """Test full evaluation pipeline."""
    evaluator = EmbeddingEvaluator(TEST_CONFIG_PATH)
    results_df = evaluator.run_evaluation(
        sample_evaluation_data['queries'],
        sample_evaluation_data['content'],
        sample_evaluation_data['relevance_scores']
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

def test_content_length_limits(config, test_cases):
    """Test handling of content length limits."""
    max_length = config.get('test_data', {}).get('max_length', 512)
    long_content = " ".join(["word"] * max_length)
    
    test_data = {
        'queries': [test_cases['cases'][0]['query']],
        'content': [long_content],
        'relevance_scores': [[1]]
    }
    
    evaluator = EmbeddingEvaluator(TEST_CONFIG_PATH)
    results = evaluator.evaluate_model(
        config['models'][0],
        test_data['queries'],
        test_data['content'],
        test_data['relevance_scores']
    )
    
    assert 'avg_mrr' in results

def test_similarity_distribution(test_cases):
    """Test similarity distribution requirements."""
    distribution = test_cases['similarity_distribution']
    total = sum(distribution.values())
    assert pytest.approx(total, 0.01) == 1.0

def test_test_requirements(test_cases):
    """Test statistical requirements."""
    requirements = test_cases['test_requirements']
    assert requirements['min_queries_per_category'] > 0
    assert 0 < requirements['max_query_content_ratio'] <= 1.0
    assert requirements['min_relevant_per_query'] > 0