"""
Accuracy metric for evaluating benchmark results.
"""
import re
import logging
import numpy as np
from typing import Dict, List, Any, Union, Optional
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class AccuracyMetric:
    """
    Accuracy metric for evaluating model responses against expected outputs.
    
    This metric can compute:
    - Exact match accuracy
    - Normalized accuracy (case-insensitive, punctuation-free)
    - Fuzzy match accuracy (using similarity threshold)
    """
    
    def __init__(self, 
                normalize: bool = True,
                fuzzy_threshold: float = 0.8,
                keyword_match: bool = False,
                keywords_weight: float = 0.5):
        """
        Initialize the accuracy metric.
        
        Args:
            normalize: Whether to normalize text before comparison (case, punctuation)
            fuzzy_threshold: Threshold for fuzzy matching (0.0 to 1.0)
            keyword_match: Whether to consider keyword matches in scoring
            keywords_weight: Weight of keyword matching in the final score (0.0 to 1.0)
        """
        self.normalize = normalize
        self.fuzzy_threshold = fuzzy_threshold
        self.keyword_match = keyword_match
        self.keywords_weight = keywords_weight
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by removing punctuation, extra whitespace, and converting to lowercase.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _fuzzy_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate fuzzy similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract important keywords from text.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """
        # Simple implementation - split and filter out common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'was', 'were'}
        words = self._normalize_text(text).split()
        return [word for word in words if word not in common_words and len(word) > 2]
    
    def _keyword_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate keyword similarity between two texts.
        
        Args:
            text1: First text (typically the expected output)
            text2: Second text (typically the model response)
            
        Returns:
            Keyword similarity score between 0.0 and 1.0
        """
        keywords1 = set(self._extract_keywords(text1))
        keywords2 = set(self._extract_keywords(text2))
        
        if not keywords1:
            return 0.0
            
        # Calculate Jaccard similarity
        intersection = keywords1.intersection(keywords2)
        union = keywords1.union(keywords2)
        
        return len(intersection) / len(keywords1)  # Use expected keywords as denominator
    
    def compute(self, 
               expected: str, 
               actual: str,
               return_details: bool = False) -> Union[float, Dict[str, float]]:
        """
        Compute accuracy score between expected and actual outputs.
        
        Args:
            expected: Expected output text
            actual: Actual model response text
            return_details: Whether to return detailed scores
            
        Returns:
            Accuracy score or dictionary of detailed scores
        """
        if not expected or not actual:
            logger.warning("Empty input to accuracy metric")
            return 0.0 if not return_details else {"score": 0.0, "exact_match": 0.0, "fuzzy_match": 0.0, "keyword_match": 0.0}
        
        # Compute exact match
        exact_match = 1.0 if expected == actual else 0.0
        
        # Compute normalized match
        if self.normalize:
            norm_expected = self._normalize_text(expected)
            norm_actual = self._normalize_text(actual)
            norm_match = 1.0 if norm_expected == norm_actual else 0.0
        else:
            norm_match = exact_match
        
        # Compute fuzzy match
        fuzzy_match = self._fuzzy_similarity(
            self._normalize_text(expected) if self.normalize else expected,
            self._normalize_text(actual) if self.normalize else actual
        )
        
        # Compute keyword match if enabled
        if self.keyword_match:
            keyword_match = self._keyword_similarity(expected, actual)
        else:
            keyword_match = 0.0
        
        # Compute final score
        if fuzzy_match >= self.fuzzy_threshold:
            # If fuzzy similarity is high, consider it correct
            score = 1.0
        elif self.keyword_match and keyword_match > 0:
            # Weighted score based on keyword matches
            score = keyword_match * self.keywords_weight
        else:
            # Default to normalized match
            score = norm_match
        
        if return_details:
            return {
                "score": score,
                "exact_match": exact_match,
                "normalized_match": norm_match,
                "fuzzy_match": fuzzy_match,
                "keyword_match": keyword_match
            }
        else:
            return score
    
    def compute_batch(self, 
                     expected_list: List[str], 
                     actual_list: List[str],
                     return_details: bool = False) -> Union[float, Dict[str, Any]]:
        """
        Compute accuracy scores for a batch of outputs.
        
        Args:
            expected_list: List of expected outputs
            actual_list: List of actual model responses
            return_details: Whether to return detailed scores
            
        Returns:
            Average accuracy score or dictionary with detailed information
        """
        if len(expected_list) != len(actual_list):
            raise ValueError(f"Mismatched lists: expected {len(expected_list)}, got {len(actual_list)}")
        
        scores = []
        details = []
        
        for expected, actual in zip(expected_list, actual_list):
            result = self.compute(expected, actual, return_details=True)
            scores.append(result["score"])
            details.append(result)
        
        average_score = np.mean(scores) if scores else 0.0
        
        if return_details:
            return {
                "score": average_score,
                "individual_scores": scores,
                "details": details,
                "metrics": {
                    "exact_match_rate": np.mean([d["exact_match"] for d in details]),
                    "normalized_match_rate": np.mean([d["normalized_match"] for d in details]),
                    "avg_fuzzy_match": np.mean([d["fuzzy_match"] for d in details]),
                    "avg_keyword_match": np.mean([d["keyword_match"] for d in details])
                }
            }
        else:
            return average_score 