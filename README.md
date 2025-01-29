
# RAG Embedding Model Evaluator

This tool helps evaluate different embedding models for RAG (Retrieval-Augmented Generation) applications. It provides a framework for testing how well different embedding models perform on your specific dataset and use case.

## Overview

The evaluator allows you to:
- Test multiple embedding models against your dataset
- Measure performance using standard metrics (MRR, nDCG)
- Compare results to choose the best model for your use case

## Setup

1. Create and activate the evaluation environment:
```bash
make setup-evals-env
```

2. Clean up environment (if needed):
```bash
make clean-evals-env
```

## Usage

### 1. Data Preparation

Use the Jupyter notebook in `evals/notebooks/data_filterer.ipynb` to:
- Load your raw dataset
- Clean and preprocess the data
- Save the processed dataset to your data directory

Your processed dataset should include:
- Content to be embedded (e.g., text, documents)
- Unique identifiers for each item
- Metadata for measuring relevance (e.g., categories, tags)

### 2. Configuration

Update `evals/config/embedding_eval.yaml` with:
```yaml
models:
  - name: model1-name
    description: Description of first model
  - name: model2-name
    description: Description of second model

dataset:
  path: path/to/your/clean/data.csv
  cache_dir: data/cache
  
  # Map your dataset columns
  columns:
    content: YourContentColumn       # Main content to embed
    identifier: YourIDColumn         # Unique identifier
    metadata: YourMetadataColumn     # For relevance scoring
    
  sampling:
    n_samples: 1000                  # Number of content samples
    n_queries: 100                   # Number of queries to generate

  evaluation:
    query_template: "Find content similar to '{identifier}' ({metadata})"
    similarity_threshold: 0.5        # For binary relevance scoring
```

### 3. Running Evaluations

```bash
# Load and validate dataset
make download-data

# Prepare evaluation data
make prepare-evals

# Run evaluation
make run-evals
```

Results will be saved to the specified output directory with:
- Performance metrics for each model
- Comparison visualizations
- Detailed evaluation logs

## Evaluation Metrics

The tool calculates:
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG@5, nDCG@10)
- Additional metrics can be configured in the YAML file

## Output

Evaluation results include:
- CSV file with metrics for each model
- Performance comparison tables
- Logs with detailed evaluation information

## Adding New Models

To evaluate additional embedding models:
1. Add them to the models list in the config file
2. Ensure they're compatible with the HuggingFace transformers library
3. Run the evaluation pipeline

## Tips

- Start with a small sample size for quick testing
- Use the notebook to explore your data before running full evaluations
- Cache embeddings to speed up repeated evaluations
- Monitor the logs for insights into model performance

## Customization

The tool is designed to be dataset-agnostic and can be adapted for:
- Different content types
- Custom relevance metrics
- Specific evaluation criteria

## Requirements

See `evals/requirements.txt` for complete list of dependencies.


## Sitemap

```
├── Makefile
├── README.md
├── __init__.py
├── config
│   └── embedding_eval.yaml
├── data
│   ├── cache
│   │   ├── intfloat_e5-large-v2_content.json
│   │   ├── intfloat_e5-large-v2_embeddings.npy
│   │   ├── sentence-transformers_all-MiniLM-L6-v2_content.json
│   │   ├── sentence-transformers_all-MiniLM-L6-v2_embeddings.npy
│   │   ├── sentence-transformers_all-mpnet-base-v2_content.json
│   │   ├── sentence-transformers_all-mpnet-base-v2_embeddings.npy
│   │   └── test_data.json
│   ├── processed_dataset.csv
│   ├── processed_dataset.stats.json
│   └── wiki_movie_plots_deduped_with_summaries.csv
├── embedding
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-311.pyc
│   │   ├── cli.cpython-311.pyc
│   │   ├── data_loader.cpython-311.pyc
│   │   ├── evaluator.cpython-311.pyc
│   │   ├── logger_config.cpython-311.pyc
│   │   └── metrics.cpython-311.pyc
│   ├── cli.py
│   ├── data_loader.py
│   ├── evaluator.py
│   ├── logger_config.py
│   ├── metrics.py
│   └── tests
│       ├── __init__.py
│       ├── test_evaluator.py
│       └── test_metrics.py
├── notebooks
│   ├── data_filterer.ipynb
│   └── embedding_comparison.ipynb
├── requirements.txt
└── results
    └── embedding_benchmarks
        └── results.csv
```