"""Data loading and preprocessing utilities for embedding evaluation."""
from pathlib import Path
import pandas as pd
from typing import Tuple, List, Dict, Optional
from loguru import logger
import json
import yaml
import numpy as np
from rich.progress import Progress

class DataLoader:
    def __init__(self, config_path: str | Path):
        """
        Initialize data loader with config.
        
        Args:
            config_path: Path to the evaluation configuration file
        """
        self.config = self._load_config(config_path)
        
        # Set up data directories relative to config file location
        config_dir = Path(config_path).parent
        self.dataset_path = config_dir / self.config['dataset']['path']
        self.cache_dir = config_dir / self.config['dataset']['cache_dir']
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Get column mappings
        self.columns = self.config['dataset']['columns']
        
        # Get evaluation settings
        self.query_template = self.config['dataset']['evaluation']['query_template']
        self.similarity_threshold = self.config['dataset']['evaluation']['similarity_threshold']
        
        logger.info(f"Using dataset: {self.dataset_path}")
        logger.info(f"Cache directory: {self.cache_dir}")

    def _load_config(self, config_path: str | Path) -> dict:
        """
        Load and validate configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Validated configuration dictionary
        """
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            # Validate required fields
            required_fields = ['models', 'metrics', 'dataset']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field in config: {field}")
            
            # Validate dataset config
            dataset_required = ['path', 'cache_dir', 'columns', 'evaluation']
            for field in dataset_required:
                if field not in config['dataset']:
                    raise ValueError(f"Missing required dataset field: {field}")
            
            logger.info(f"Loaded configuration from {config_path}")
            return config
            
        except Exception as e:
            raise RuntimeError(f"Error loading config from {config_path}: {str(e)}")

    def load_dataset(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load dataset for evaluation.
        
        Args:
            force_reload: Whether to bypass cache
            
        Returns:
            DataFrame ready for evaluation
        """
        try:
            logger.info(f"Loading dataset from {self.dataset_path}")
            df = pd.read_csv(self.dataset_path)
            logger.info(f"Loaded {len(df)} rows")
            
            # Verify required columns exist
            missing_columns = [
                col for col in self.columns.values() 
                if col not in df.columns
            ]
            if missing_columns:
                raise ValueError(
                    f"Missing required columns in dataset: {missing_columns}\n"
                    f"Available columns: {df.columns.tolist()}"
                )
            
            # Log some basic stats about the data
            logger.info(f"Content length stats:")
            content_lengths = df[self.columns['content']].str.len()
            logger.info(f"  Min: {content_lengths.min()}")
            logger.info(f"  Max: {content_lengths.max()}")
            logger.info(f"  Mean: {content_lengths.mean():.0f}")
            
            unique_metadata = df[self.columns['metadata']].nunique()
            logger.info(f"Unique metadata values: {unique_metadata}")
            
            return df
            
        except Exception as e:
            raise RuntimeError(
                f"Error loading dataset ({str(e)}). "
                "Please check the CSV file exists and is properly formatted."
            )
        
    def prepare_test_data(self, progress: Progress = None) -> Path:
        """
        Prepare evaluation test data.
        
        Args:
            progress: Progress bar instance for tracking
        
        Returns:
            Path to saved test data file
        """
        test_data_path = self.cache_dir / 'test_data.json'
        
        # Check if test data already exists
        if test_data_path.exists():
            logger.info(f"Test data already exists at {test_data_path}")
            return test_data_path
            
        task_id = None
        if progress:
            task_id = progress.add_task(
                "[cyan]Preparing test data...",
                total=4  # Number of main steps
            )
            
        # Load dataset
        logger.info("Loading dataset...")
        df = self.load_dataset()
        if progress and task_id:
            progress.advance(task_id)
            
        # Prepare evaluation samples
        logger.info("Preparing evaluation samples...")
        queries, content_samples, relevance_matrix = self.prepare_evaluation_samples(df)
        if progress and task_id:
            progress.advance(task_id)
            
        # Prepare test data dictionary
        test_data = {
            'queries': queries,
            'content': content_samples,
            'relevance_matrix': relevance_matrix,
            'metadata': {
                'n_samples': len(content_samples),
                'n_queries': len(queries),
                'creation_date': pd.Timestamp.now().isoformat(),
                'columns': {
                    'content': self.columns['content'],
                    'identifier': self.columns['identifier'],
                    'metadata': self.columns['metadata']
                }
            }
        }
        
        # Save test data
        logger.info(f"Saving test data to {test_data_path}")
        test_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(test_data_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        if progress and task_id:
            progress.advance(task_id)
            
        logger.info("Test data preparation complete!")
        if progress and task_id:
            progress.advance(task_id)
            
        return test_data_path

    def prepare_evaluation_samples(self, 
                                 df: pd.DataFrame, 
                                 n_samples: Optional[int] = None,
                                 n_queries: Optional[int] = None) -> Tuple[List[str], List[str], List[List[int]]]:
        """
        Prepare evaluation samples from the dataset.
        
        Args:
            df: Input DataFrame
            n_samples: Number of content samples (optional)
            n_queries: Number of queries to generate (optional)
            
        Returns:
            Tuple of (queries, content_samples, relevance_matrix)
        """
        # Get sampling parameters from config
        sampling_config = self.config['dataset']['sampling']
        n_samples = n_samples or sampling_config['n_samples']
        n_queries = n_queries or sampling_config['n_queries']
        
        content_col = self.columns['content']
        id_col = self.columns['identifier']
        metadata_col = self.columns['metadata']
        
        # Sample content
        content_samples = df.sample(n_samples, random_state=42)[content_col].tolist()
        
        # Generate queries from items not in content samples
        remaining_df = df[~df[content_col].isin(content_samples)]
        query_items = remaining_df.sample(n_queries, random_state=42)
        
        # Create queries using template from config
        queries = [
            self.query_template.format(
                identifier=row[id_col],
                metadata=row[metadata_col]
            )
            for _, row in query_items.iterrows()
        ]
        
        # Generate relevance matrix
        relevance_matrix = self._generate_relevance_matrix(
            query_items, 
            content_samples, 
            df,
            content_col,
            metadata_col
        )
        
        return queries, content_samples, relevance_matrix

    def _generate_relevance_matrix(self, 
                                query_items: pd.DataFrame, 
                                content_samples: List[str], 
                                full_df: pd.DataFrame,
                                content_col: str,
                                metadata_col: str) -> List[List[int]]:
        """
        Generate relevance matrix based on metadata similarity.
        
        Args:
            query_items: DataFrame of items used to generate queries
            content_samples: List of content strings
            full_df: Complete dataset
            content_col: Name of content column
            metadata_col: Name of metadata column
            
        Returns:
            Binary relevance matrix [n_queries x n_samples]
        """
        relevance_matrix = []
        
        for _, query_item in query_items.iterrows():
            query_metadata = set(query_item[metadata_col].split(','))
            
            # Find items in content samples and their metadata
            sample_relevance = []
            for content in content_samples:
                # Find the item in full_df that matches this content
                item_row = full_df[full_df[content_col] == content].iloc[0]
                item_metadata = set(item_row[metadata_col].split(','))
                
                # Calculate metadata similarity
                similarity = len(query_metadata & item_metadata) / len(query_metadata | item_metadata)
                
                # Convert to binary relevance using threshold from config
                sample_relevance.append(1 if similarity > self.similarity_threshold else 0)
            
            relevance_matrix.append(sample_relevance)
        
        return relevance_matrix

    def cache_embeddings(self, 
                        model_name: str, 
                        content: List[str], 
                        embeddings: np.ndarray) -> None:
        """
        Cache embeddings for a model.
        
        Args:
            model_name: Name of the embedding model
            content: List of content strings
            embeddings: NumPy array of embeddings
        """
        # Create safe filename from model name
        safe_name = model_name.replace('/', '_')
        cache_path = self.cache_dir / f"{safe_name}_embeddings.npy"
        content_path = self.cache_dir / f"{safe_name}_content.json"

        # Save embeddings
        logger.info(f"Caching embeddings for {model_name}")
        np.save(cache_path, embeddings)
        
        # Save content mapping
        with open(content_path, 'w') as f:
            json.dump(content, f)
        
        logger.info(f"Cached {len(content)} embeddings to {cache_path}")

    def load_cached_embeddings(self, 
                             model_name: str, 
                             content: List[str]) -> Optional[np.ndarray]:
        """
        Load cached embeddings if available and matching content.
        
        Args:
            model_name: Name of the embedding model
            content: List of content strings to verify against cache
            
        Returns:
            NumPy array of embeddings if cache exists and matches, None otherwise
        """
        # Create safe filename from model name
        safe_name = model_name.replace('/', '_')
        cache_path = self.cache_dir / f"{safe_name}_embeddings.npy"
        content_path = self.cache_dir / f"{safe_name}_content.json"
        
        # Check if cache exists
        if not cache_path.exists() or not content_path.exists():
            return None
            
        # Load and verify content
        try:
            with open(content_path) as f:
                cached_content = json.load(f)
                
            # Check if content matches exactly
            if cached_content == content:
                logger.info(f"Found matching cache for {model_name}")
                embeddings = np.load(cache_path)
                logger.info(f"Loaded {len(embeddings)} embeddings from cache")
                return embeddings
            else:
                logger.info("Cache exists but content doesn't match")
                return None
                
        except Exception as e:
            logger.warning(f"Error loading cache: {str(e)}")
            return None