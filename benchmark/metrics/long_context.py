"""
Long context metric for evaluating understanding of long inputs.
"""
import re
import logging
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple, Set
from .accuracy import AccuracyMetric

logger = logging.getLogger(__name__)

class LongContextMetric:
    """
    Metric for evaluating understanding of long contexts.
    
    Measures the model's ability to retain and reason about information
    across different parts of a long input (beginning, middle, end).
    """
    
    def __init__(self, 
                 normalize: bool = True,
                 segment_count: int = 3):  # Default: beginning, middle, end
        """
        Initialize the long context metric.
        
        Args:
            normalize: Whether to normalize text before comparison
            segment_count: Number of segments to divide the context into
        """
        self.normalize = normalize
        self.segment_count = segment_count
        self.accuracy_metric = AccuracyMetric(normalize=normalize)
    
    def _segment_text(self, text: str) -> List[str]:
        """
        Divide text into segments.
        
        Args:
            text: Text to segment
            
        Returns:
            List of text segments
        """
        if not text:
            return []
            
        # Split by sentences or paragraphs
        if '\n\n' in text:
            # Split by paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            # If too few paragraphs, fall back to sentences
            if len(paragraphs) < self.segment_count:
                sentences = re.split(r'(?<=[.!?])\s+', text)
                sentences = [s.strip() for s in sentences if s.strip()]
                return self._group_into_segments(sentences)
            else:
                return self._group_into_segments(paragraphs)
        else:
            # Split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            return self._group_into_segments(sentences)
    
    def _group_into_segments(self, items: List[str]) -> List[str]:
        """
        Group items into the desired number of segments.
        
        Args:
            items: List of text items (sentences or paragraphs)
            
        Returns:
            List of segment texts
        """
        if not items:
            return []
            
        # If fewer items than segments, return as is
        if len(items) <= self.segment_count:
            return items
            
        # Calculate items per segment
        items_per_segment = len(items) // self.segment_count
        remainder = len(items) % self.segment_count
        
        segments = []
        start_idx = 0
        
        for i in range(self.segment_count):
            # Add one extra item to some segments if there's a remainder
            extra = 1 if i < remainder else 0
            end_idx = start_idx + items_per_segment + extra
            
            # Join items for this segment
            segment = ' '.join(items[start_idx:end_idx])
            segments.append(segment)
            
            start_idx = end_idx
            
        return segments
    
    def _extract_segment_keywords(self, segments: List[str]) -> List[Set[str]]:
        """
        Extract important keywords from each segment.
        
        Args:
            segments: List of text segments
            
        Returns:
            List of sets containing keywords for each segment
        """
        # Common words to exclude
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                       'for', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 
                       'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 
                       'could', 'may', 'might', 'must', 'can'}
        
        segment_keywords = []
        
        for segment in segments:
            # Normalize and tokenize
            if self.normalize:
                segment = segment.lower()
                segment = re.sub(r'[^\w\s]', '', segment)
                
            # Split into words
            words = segment.split()
            
            # Filter out common words and short words
            keywords = {word for word in words if word not in common_words and len(word) > 3}
            
            segment_keywords.append(keywords)
            
        return segment_keywords
    
    def _evaluate_segment_coverage(self, 
                                  response: str, 
                                  segment_keywords: List[Set[str]]) -> List[float]:
        """
        Evaluate how well the response covers each segment.
        
        Args:
            response: Model response text
            segment_keywords: Keywords from each context segment
            
        Returns:
            List of coverage scores for each segment
        """
        # Normalize response
        if self.normalize:
            response = response.lower()
            response = re.sub(r'[^\w\s]', '', response)
            
        # Get response words
        response_words = set(response.split())
        
        # Calculate coverage for each segment
        coverage_scores = []
        
        for keywords in segment_keywords:
            if not keywords:
                coverage_scores.append(0.0)
                continue
                
            # Count keywords found in response
            found_keywords = keywords.intersection(response_words)
            coverage = len(found_keywords) / len(keywords)
            coverage_scores.append(coverage)
            
        return coverage_scores
    
    def compute(self, 
               context: str,
               expected: str, 
               actual: str,
               return_details: bool = False) -> Union[float, Dict[str, Any]]:
        """
        Compute long context understanding score.
        
        Args:
            context: The long context text
            expected: Expected output text
            actual: Actual model response text
            return_details: Whether to return detailed scores
            
        Returns:
            Long context score or dictionary of detailed scores
        """
        if not context or not actual:
            logger.warning("Empty input to long context metric")
            return 0.0 if not return_details else {"score": 0.0}
        
        # Segment the context
        segments = self._segment_text(context)
        
        # If segmentation failed, return basic accuracy
        if len(segments) < 2:
            accuracy = self.accuracy_metric.compute(expected, actual)
            return accuracy if not return_details else {"score": accuracy, "segments": 1}
        
        # Extract keywords from each segment
        segment_keywords = self._extract_segment_keywords(segments)
        
        # Evaluate coverage of each segment in the response
        coverage_scores = self._evaluate_segment_coverage(actual, segment_keywords)
        
        # Calculate standard deviation to measure balance
        coverage_std = np.std(coverage_scores)
        coverage_mean = np.mean(coverage_scores)
        
        # Penalize high standard deviation (unbalanced coverage)
        balance_score = max(0, 1.0 - (coverage_std * 2))
        
        # Calculate accuracy
        accuracy = self.accuracy_metric.compute(expected, actual)
        
        # Combine scores - 60% accuracy, 40% balanced coverage
        score = 0.6 * accuracy + 0.4 * (0.7 * coverage_mean + 0.3 * balance_score)
        
        if return_details:
            return {
                "score": score,
                "accuracy": accuracy,
                "coverage_mean": coverage_mean,
                "coverage_balance": balance_score,
                "segment_coverage": dict(enumerate(coverage_scores)),
                "segment_count": len(segments)
            }
        else:
            return score
    
    def compute_batch(self, 
                     contexts: List[str],
                     expected_list: List[str], 
                     actual_list: List[str],
                     return_details: bool = False) -> Union[float, Dict[str, Any]]:
        """
        Compute long context scores for a batch of outputs.
        
        Args:
            contexts: List of context texts
            expected_list: List of expected outputs
            actual_list: List of actual model responses
            return_details: Whether to return detailed scores
            
        Returns:
            Average score or dictionary with detailed information
        """
        if len(contexts) != len(expected_list) or len(expected_list) != len(actual_list):
            raise ValueError(f"Mismatched lists: contexts={len(contexts)}, expected={len(expected_list)}, actual={len(actual_list)}")
        
        scores = []
        details = []
        
        for context, expected, actual in zip(contexts, expected_list, actual_list):
            result = self.compute(context, expected, actual, return_details=True)
            scores.append(result["score"])
            details.append(result)
        
        average_score = np.mean(scores) if scores else 0.0
        
        if return_details:
            return {
                "score": average_score,
                "individual_scores": scores,
                "details": details,
                "avg_accuracy": np.mean([d["accuracy"] for d in details]),
                "avg_coverage": np.mean([d["coverage_mean"] for d in details]),
                "avg_balance": np.mean([d["coverage_balance"] for d in details])
            }
        else:
            return average_score 