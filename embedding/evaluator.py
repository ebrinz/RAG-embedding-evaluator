"""Core embedding model evaluation functionality."""
from typing import List, Dict
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger

from .metrics import calculate_mrr, calculate_ndcg
from .data_loader import DataLoader

class EmbeddingEvaluator:
    def __init__(self, config_path: str | Path):
        """Initialize evaluator with configuration file."""
        self.config = self._load_config(config_path)
        self.models = self.config['models']
        self.device = self.config.get('device', 'cpu')
        self.batch_size = self.config.get('batch_size', 32)
        self.data_loader = DataLoader(config_path)
        self.results = {}
        
        logger.info(f"Initialized evaluator with {len(self.models)} models")
        logger.info(f"Using device: {self.device}")

    def _load_config(self, config_path: str | Path) -> dict:
        """Load and validate configuration file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        # Validate required fields
        required_fields = ['models', 'metrics']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in config: {field}")
                
        return config

    def generate_embeddings(self, 
                          model_name: str, 
                          documents: List[str],
                          progress=None) -> np.ndarray:
        """Generate embeddings for documents using specified model."""
        # Check cache first
        cached_embeddings = self.data_loader.load_cached_embeddings(model_name, documents)
        if cached_embeddings is not None:
            logger.info(f"Loaded cached embeddings for {model_name}")
            return cached_embeddings

        logger.info(f"Generating embeddings for {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(self.device)
        
        embeddings = []
        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i:i + self.batch_size]
            inputs = tokenizer(batch_docs, 
                             return_tensors="pt", 
                             padding=True, 
                             truncation=True,
                             max_length=self.config.get('test_data', {}).get('max_length', 512))
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings.extend(batch_embeddings.cpu().numpy())
                
            if progress:
                progress.advance(1)
        
        embeddings = np.array(embeddings)
        
        # Cache the embeddings
        self.data_loader.cache_embeddings(model_name, documents, embeddings)
        
        return embeddings

    def evaluate_model(self, 
                      model_config: Dict,
                      test_queries: List[str],
                      documents: List[str],
                      relevance_scores: List[List[int]]) -> Dict:
        """Evaluate a single embedding model."""
        model_name = model_config['name']
        logger.info(f"Evaluating model: {model_name}")
        
        # Generate or load cached document embeddings
        doc_embeddings = self.generate_embeddings(model_name, documents)
        
        # Setup model for query encoding
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(self.device)
        
        results = {metric: [] for metric in self.config['metrics']}
        
        # Process queries in batches
        for i in range(0, len(test_queries), self.batch_size):
            batch_queries = test_queries[i:i + self.batch_size]
            batch_relevance = relevance_scores[i:i + self.batch_size]
            
            # Encode queries
            query_inputs = tokenizer(batch_queries, 
                                   return_tensors="pt", 
                                   padding=True, 
                                   truncation=True,
                                   max_length=self.config.get('test_data', {}).get('max_length', 512))
            query_inputs = {k: v.to(self.device) for k, v in query_inputs.items()}
            
            with torch.no_grad():
                query_outputs = model(**query_inputs)
                query_embeddings = query_outputs.last_hidden_state.mean(dim=1)
            
            # Calculate similarities for batch
            similarities = cosine_similarity(query_embeddings.cpu().numpy(), doc_embeddings)
            
            # Calculate metrics for each query in batch
            for sim, rel in zip(similarities, batch_relevance):
                ranked_indices = sim.argsort()[::-1]
                ranked_relevance = [rel[idx] for idx in ranked_indices]
                
                if 'mrr' in self.config['metrics']:
                    results['mrr'].append(calculate_mrr(ranked_relevance))
                if 'ndcg@5' in self.config['metrics']:
                    results['ndcg@5'].append(calculate_ndcg(ranked_relevance, k=5))
                if 'ndcg@10' in self.config['metrics']:
                    results['ndcg@10'].append(calculate_ndcg(ranked_relevance, k=10))
        
        # Average results
        final_results = {
            'model': model_name,
            'description': model_config.get('description', ''),
            **{f"avg_{k}": np.mean(v) for k, v in results.items() if v}
        }
        
        logger.info(f"Evaluation complete for {model_name}")
        return final_results

    def run_evaluation(self) -> pd.DataFrame:
        """Run complete evaluation pipeline."""
        # Load and prepare data
        logger.info("Loading dataset...")
        df = self.data_loader.load_dataset()
        
        # Prepare evaluation samples
        logger.info("Preparing evaluation samples...")
        queries, documents, relevance_scores = self.data_loader.prepare_evaluation_samples(
            df,
            n_samples=self.config.get('n_samples', 1000),
            n_queries=self.config.get('n_queries', 100)
        )
        
        # Evaluate each model
        all_results = []
        for model_config in self.models:
            results = self.evaluate_model(
                model_config,
                queries,
                documents,
                relevance_scores
            )
            all_results.append(results)
        
        return pd.DataFrame(all_results)