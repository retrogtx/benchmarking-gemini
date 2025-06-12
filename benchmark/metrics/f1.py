"""
F1, Precision, and Recall metrics for evaluating benchmark results.
"""
import re
import logging
import numpy as np
from typing import Dict, List, Any, Union, Optional, Set, Tuple

logger = logging.getLogger(__name__)

class BaseClassificationMetric:
    """Base class for classification metrics like F1, Precision, and Recall."""
    
    def __init__(self, normalize: bool = True, token_based: bool = False):
        """
        Initialize the base classification metric.
        
        Args:
            normalize: Whether to normalize text before comparison (case, punctuation)
            token_based: Whether to evaluate based on tokens instead of exact answers
        """
        self.normalize = normalize
        self.token_based = token_based
    
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
    
    def _tokenize(self, text: str) -> Set[str]:
        """
        Tokenize text into a set of words.
        
        Args:
            text: Text to tokenize
            
        Returns:
            Set of tokens
        """
        if not text:
            return set()
            
        # Normalize if needed
        if self.normalize:
            text = self._normalize_text(text)
        
        # Split into words and return as a set
        return set(text.split())
    
    def _get_tokens(self, text: str) -> Set[str]:
        """
        Get tokens from text based on configuration.
        
        Args:
            text: Text to process
            
        Returns:
            Set of tokens or the text itself as a singleton set
        """
        if self.token_based:
            return self._tokenize(text)
        else:
            # For non-token-based evaluation, use the whole text as one "token"
            normalized = self._normalize_text(text) if self.normalize else text
            return {normalized} if normalized else set()
    
    def _compute_classification_metrics(self, 
                                      expected: str, 
                                      actual: str) -> Tuple[int, int, int]:
        """
        Compute true positives, false positives, and false negatives.
        
        Args:
            expected: Expected output text
            actual: Actual model response text
            
        Returns:
            Tuple of (true_positives, false_positives, false_negatives)
        """
        expected_tokens = self._get_tokens(expected)
        actual_tokens = self._get_tokens(actual)
        
        # If both are empty, return zeros
        if not expected_tokens and not actual_tokens:
            return 0, 0, 0
        
        # Compute metrics
        true_positives = len(expected_tokens.intersection(actual_tokens))
        false_positives = len(actual_tokens - expected_tokens)
        false_negatives = len(expected_tokens - actual_tokens)
        
        return true_positives, false_positives, false_negatives
    
    def compute_batch(self, 
                     expected_list: List[str], 
                     actual_list: List[str],
                     return_details: bool = False) -> Union[float, Dict[str, Any]]:
        """
        Compute metrics for a batch of outputs.
        
        Args:
            expected_list: List of expected outputs
            actual_list: List of actual model responses
            return_details: Whether to return detailed scores
            
        Returns:
            Average score or dictionary with detailed information
        """
        if len(expected_list) != len(actual_list):
            raise ValueError(f"Mismatched lists: expected {len(expected_list)}, got {len(actual_list)}")
        
        # Initialize counters
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        individual_metrics = []
        
        # Process each pair
        for expected, actual in zip(expected_list, actual_list):
            tp, fp, fn = self._compute_classification_metrics(expected, actual)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            # Compute individual metrics
            individual_result = self._compute_from_counts(tp, fp, fn)
            individual_metrics.append(individual_result)
        
        # Compute aggregated metrics
        result = self._compute_from_counts(total_tp, total_fp, total_fn)
        
        if return_details:
            return {
                "score": result["score"],
                "true_positives": total_tp,
                "false_positives": total_fp,
                "false_negatives": total_fn,
                "individual_scores": [m["score"] for m in individual_metrics],
                "details": individual_metrics
            }
        else:
            return result["score"]
    
    def _compute_from_counts(self, 
                           tp: int, 
                           fp: int, 
                           fn: int) -> Dict[str, float]:
        """
        Compute metrics from counts. To be implemented by subclasses.
        
        Args:
            tp: True positives
            fp: False positives
            fn: False negatives
            
        Returns:
            Dictionary with metric scores
        """
        raise NotImplementedError("Subclasses must implement this method")


class PrecisionMetric(BaseClassificationMetric):
    """Precision metric for evaluating model responses."""
    
    def compute(self, 
               expected: str, 
               actual: str,
               return_details: bool = False) -> Union[float, Dict[str, float]]:
        """
        Compute precision score between expected and actual outputs.
        
        Args:
            expected: Expected output text
            actual: Actual model response text
            return_details: Whether to return detailed scores
            
        Returns:
            Precision score or dictionary of detailed scores
        """
        tp, fp, fn = self._compute_classification_metrics(expected, actual)
        result = self._compute_from_counts(tp, fp, fn)
        
        if return_details:
            return result
        else:
            return result["score"]
    
    def _compute_from_counts(self, 
                           tp: int, 
                           fp: int, 
                           fn: int) -> Dict[str, float]:
        """
        Compute precision from counts.
        
        Args:
            tp: True positives
            fp: False positives
            fn: False negatives
            
        Returns:
            Dictionary with precision score
        """
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        return {
            "score": precision,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        }


class RecallMetric(BaseClassificationMetric):
    """Recall metric for evaluating model responses."""
    
    def compute(self, 
               expected: str, 
               actual: str,
               return_details: bool = False) -> Union[float, Dict[str, float]]:
        """
        Compute recall score between expected and actual outputs.
        
        Args:
            expected: Expected output text
            actual: Actual model response text
            return_details: Whether to return detailed scores
            
        Returns:
            Recall score or dictionary of detailed scores
        """
        tp, fp, fn = self._compute_classification_metrics(expected, actual)
        result = self._compute_from_counts(tp, fp, fn)
        
        if return_details:
            return result
        else:
            return result["score"]
    
    def _compute_from_counts(self, 
                           tp: int, 
                           fp: int, 
                           fn: int) -> Dict[str, float]:
        """
        Compute recall from counts.
        
        Args:
            tp: True positives
            fp: False positives
            fn: False negatives
            
        Returns:
            Dictionary with recall score
        """
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return {
            "score": recall,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        }


class F1Metric(BaseClassificationMetric):
    """F1 score metric for evaluating model responses."""
    
    def compute(self, 
               expected: str, 
               actual: str,
               return_details: bool = False) -> Union[float, Dict[str, float]]:
        """
        Compute F1 score between expected and actual outputs.
        
        Args:
            expected: Expected output text
            actual: Actual model response text
            return_details: Whether to return detailed scores
            
        Returns:
            F1 score or dictionary of detailed scores
        """
        tp, fp, fn = self._compute_classification_metrics(expected, actual)
        result = self._compute_from_counts(tp, fp, fn)
        
        if return_details:
            return result
        else:
            return result["score"]
    
    def _compute_from_counts(self, 
                           tp: int, 
                           fp: int, 
                           fn: int) -> Dict[str, float]:
        """
        Compute F1 score from counts.
        
        Args:
            tp: True positives
            fp: False positives
            fn: False negatives
            
        Returns:
            Dictionary with F1 score and component scores
        """
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Compute F1
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "score": f1,
            "precision": precision,
            "recall": recall,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        } 