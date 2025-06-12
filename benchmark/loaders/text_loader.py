"""
Text data loader for benchmarking.
"""
import os
import re
import json
import logging
from typing import Dict, List, Any, Optional, Union, Iterator
from .base import BaseLoader

logger = logging.getLogger(__name__)

class TextLoader(BaseLoader):
    """Loader for text-based benchmark data."""
    
    def __init__(self, 
                data_dir: Optional[str] = None, 
                cache_dir: Optional[str] = None,
                file_format: str = "json",
                max_length: int = 8192):
        """
        Initialize the text loader.
        
        Args:
            data_dir: Directory containing the raw data files
            cache_dir: Directory to cache processed data
            file_format: Format of the data files (json, csv, etc.)
            max_length: Maximum length of text to load
        """
        super().__init__(data_dir, cache_dir, file_format)
        self.max_length = max_length
        
    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess a text data sample.
        
        Args:
            sample: A raw text data sample
            
        Returns:
            The preprocessed sample
        """
        # Normalize the sample using the base method
        normalized = self._normalize_sample(sample)
        
        # Extract text from input
        text_input = normalized["input"]
        if isinstance(text_input, dict) and "text" in text_input:
            text_input = text_input["text"]
        elif isinstance(text_input, str):
            pass
        else:
            logger.warning(f"Unexpected input format for sample {normalized['id']}")
            text_input = str(text_input)
        
        # Apply text-specific preprocessing
        processed_text = self._clean_text(text_input)
        
        # Truncate if needed
        if len(processed_text) > self.max_length:
            logger.warning(f"Truncating text for sample {normalized['id']}: {len(processed_text)} -> {self.max_length}")
            processed_text = processed_text[:self.max_length]
        
        # Update the normalized sample with processed text
        normalized["input"] = {"text": processed_text}
        
        # Also process expected output if it's text
        if isinstance(normalized["expected_output"], str):
            normalized["expected_output"] = self._clean_text(normalized["expected_output"])
        
        return normalized
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
            
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def load_data(self, 
                 filepath: Optional[str] = None, 
                 split: Optional[str] = None,
                 use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Load and preprocess text data.
        
        Args:
            filepath: Path to the data file or directory
            split: Data split to load (train, val, test)
            use_cache: Whether to use cached data if available
            
        Returns:
            List of preprocessed text data samples
        """
        # Determine the file path
        if filepath is None:
            if split is not None:
                filepath = os.path.join(self.data_dir, f"{split}.json")
            else:
                filepath = os.path.join(self.data_dir, "data.json")
        
        # Generate cache file name
        cache_filename = os.path.basename(filepath)
        cache_filename = f"processed_text_{cache_filename}"
        
        # Try to load from cache first
        if use_cache:
            cached_data = self.load_from_cache(cache_filename)
            if cached_data:
                return cached_data
        
        # Load the raw data
        raw_data = self.load_file(filepath)
        
        # Preprocess each sample
        processed_data = []
        for sample in raw_data:
            if self._validate_sample(sample):
                processed_sample = self.preprocess(sample)
                processed_data.append(processed_sample)
            else:
                logger.warning(f"Skipping invalid sample: {sample.get('id', 'unknown')}")
        
        # Save to cache if we have processed data
        if processed_data and use_cache:
            self.save_to_cache(processed_data, cache_filename)
        
        return processed_data
    
    def create_debug_sample(self) -> Dict[str, Any]:
        """
        Create a debug sample for testing.
        
        Returns:
            A sample text data point
        """
        return {
            "id": "debug_text_001",
            "input": {
                "text": "What is the capital of France?"
            },
            "expected_output": "The capital of France is Paris.",
            "metadata": {
                "source": "debug",
                "difficulty": "easy",
                "category": "geography"
            }
        } 