"""
Base loader class for multimodal benchmarking data.
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Iterator
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class BaseLoader(ABC):
    """Base class for all data loaders."""
    
    def __init__(self, 
                data_dir: Optional[str] = None, 
                cache_dir: Optional[str] = None,
                file_format: str = "json"):
        """
        Initialize the base loader.
        
        Args:
            data_dir: Directory containing the raw data files
            cache_dir: Directory to cache processed data
            file_format: Format of the data files (json, csv, etc.)
        """
        self.data_dir = data_dir or os.getenv("DATA_DIR", "benchmark/data")
        self.cache_dir = cache_dir or os.getenv("PROCESSED_DATA_DIR", "outputs/processed")
        self.file_format = file_format
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"Initialized {self.__class__.__name__} with data_dir={self.data_dir}, cache_dir={self.cache_dir}")
    
    def _validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Validate a data sample.
        
        Args:
            sample: A data sample to validate
            
        Returns:
            True if the sample is valid, False otherwise
        """
        # Check if the sample has the required fields
        required_fields = ["id", "input", "expected_output"]
        
        for field in required_fields:
            if field not in sample:
                logger.warning(f"Sample missing required field: {field}")
                return False
        
        return True
    
    def _normalize_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a data sample to a standard format.
        
        Args:
            sample: A data sample to normalize
            
        Returns:
            The normalized sample
        """
        # Implement basic normalization that applies to all loaders
        normalized = {
            "id": sample.get("id", "unknown"),
            "input": sample.get("input", {}),
            "expected_output": sample.get("expected_output", ""),
            "metadata": sample.get("metadata", {})
        }
        
        # Add timestamp if not present
        if "timestamp" not in normalized["metadata"]:
            normalized["metadata"]["timestamp"] = None
            
        return normalized
    
    def load_file(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load data from a file.
        
        Args:
            filepath: Path to the data file
            
        Returns:
            List of data samples from the file
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if self.file_format == 'json':
                    data = json.load(f)
                    
                    # If data is a dict, convert to list of dict
                    if isinstance(data, dict):
                        if "samples" in data:
                            data = data["samples"]
                        else:
                            data = [data]
                            
                    return data
                else:
                    raise NotImplementedError(f"File format {self.file_format} not supported yet")
        except Exception as e:
            logger.error(f"Error loading file {filepath}: {e}")
            return []
    
    @abstractmethod
    def load_data(self, 
                 filepath: Optional[str] = None, 
                 split: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load and preprocess data.
        
        Args:
            filepath: Path to the data file or directory
            split: Data split to load (train, val, test)
            
        Returns:
            List of preprocessed data samples
        """
        pass
    
    @abstractmethod
    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess a data sample.
        
        Args:
            sample: A raw data sample
            
        Returns:
            The preprocessed sample
        """
        pass
    
    def batch_iterator(self, 
                      data: List[Dict[str, Any]], 
                      batch_size: int = 1) -> Iterator[List[Dict[str, Any]]]:
        """
        Create batches from a list of samples.
        
        Args:
            data: List of data samples
            batch_size: Size of each batch
            
        Returns:
            Iterator yielding batches of samples
        """
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]
    
    def save_to_cache(self, data: List[Dict[str, Any]], filename: str) -> str:
        """
        Save processed data to the cache directory.
        
        Args:
            data: List of processed data samples
            filename: Name of the cache file
            
        Returns:
            Path to the cached file
        """
        cache_path = os.path.join(self.cache_dir, filename)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(data)} samples to {cache_path}")
            return cache_path
        except Exception as e:
            logger.error(f"Error saving to cache {cache_path}: {e}")
            return ""
    
    def load_from_cache(self, filename: str) -> List[Dict[str, Any]]:
        """
        Load processed data from the cache directory.
        
        Args:
            filename: Name of the cache file
            
        Returns:
            List of processed data samples
        """
        cache_path = os.path.join(self.cache_dir, filename)
        
        if not os.path.exists(cache_path):
            logger.info(f"Cache file {cache_path} not found")
            return []
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} samples from {cache_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading from cache {cache_path}: {e}")
            return [] 