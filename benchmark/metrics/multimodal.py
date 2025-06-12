"""
Multimodal reasoning metric for evaluating multimodal benchmarks.
"""
import re
import logging
import numpy as np
from typing import Dict, List, Any, Union, Optional, Set
from .accuracy import AccuracyMetric
from .f1 import F1Metric

logger = logging.getLogger(__name__)

class MultimodalReasoningMetric:
    """
    Metric for evaluating multimodal reasoning capabilities.
    
    This metric combines multiple factors to evaluate how well a model reasons
    across different modalities and integrates information.
    """
    
    def __init__(self, 
                reasoning_weight: float = 0.6,
                accuracy_weight: float = 0.3,
                consistency_weight: float = 0.1,
                normalize: bool = True):
        """
        Initialize the multimodal reasoning metric.
        
        Args:
            reasoning_weight: Weight for reasoning component
            accuracy_weight: Weight for factual accuracy component
            consistency_weight: Weight for cross-modal consistency
            normalize: Whether to normalize text before comparison
        """
        self.reasoning_weight = reasoning_weight
        self.accuracy_weight = accuracy_weight
        self.consistency_weight = consistency_weight
        self.normalize = normalize
        
        # Validate weights sum to 1
        total_weight = reasoning_weight + accuracy_weight + consistency_weight
        if not np.isclose(total_weight, 1.0):
            logger.warning(f"Weights sum to {total_weight}, not 1.0. Normalizing.")
            self.reasoning_weight /= total_weight
            self.accuracy_weight /= total_weight
            self.consistency_weight /= total_weight
            
        # Initialize component metrics
        self.accuracy_metric = AccuracyMetric(normalize=normalize)
        self.f1_metric = F1Metric(normalize=normalize, token_based=True)
    
    def _extract_reasoning_indicators(self, text: str) -> List[str]:
        """
        Extract indicators of reasoning from text.
        
        Looks for phrases that indicate reasoning like "because", "therefore", etc.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of reasoning indicators found
        """
        # List of reasoning indicators to look for
        indicators = [
            r'\bbecause\b',
            r'\btherefore\b',
            r'\bhence\b',
            r'\bthus\b',
            r'\bas a result\b',
            r'\bdue to\b',
            r'\bexplains\b',
            r'\bcaused by\b',
            r'\bleads to\b',
            r'\bimplies\b',
            r'\bsuggests\b',
            r'\bindicates\b',
            r'\bshows\b',
            r'\bdemonstrates\b',
            r'\bevidenced by\b',
            r'\bsupports\b',
            r'\bconcludes\b',
            r'\binfer\b',
            r'\binference\b',
            r'\breasoning\b',
            r'\banalysis\b',
            r'\brelationship\b',
            r'\bcorrelation\b',
            r'\bconnection\b'
        ]
        
        found_indicators = []
        
        for pattern in indicators:
            matches = re.findall(pattern, text, re.IGNORECASE)
            found_indicators.extend(matches)
        
        return found_indicators
    
    def _evaluate_reasoning(self, text: str) -> float:
        """
        Evaluate the quality of reasoning in text.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Reasoning score between 0.0 and 1.0
        """
        if not text:
            return 0.0
            
        # Get reasoning indicators
        indicators = self._extract_reasoning_indicators(text)
        
        # Basic implementation - count the number of reasoning indicators
        # and normalize by text length
        indicator_count = len(indicators)
        word_count = len(text.split())
        
        # Avoid division by zero
        if word_count == 0:
            return 0.0
            
        # Calculate density of reasoning indicators
        density = indicator_count / (word_count / 100)  # Per 100 words
        
        # Normalize to [0, 1] range - assuming good reasoning has at least 
        # 5 indicators per 100 words
        score = min(density / 5.0, 1.0)
        
        return score
    
    def _evaluate_modality_references(self, text: str, modalities: List[str]) -> float:
        """
        Evaluate references to different modalities in the text.
        
        Args:
            text: Text to evaluate
            modalities: List of modalities in the input (e.g., ['image', 'text'])
            
        Returns:
            Cross-modal consistency score between 0.0 and 1.0
        """
        if not text or not modalities:
            return 0.0
            
        # Look for references to each modality
        modality_references = {
            'image': [
                r'\bimage\b', r'\bpicture\b', r'\bphoto\b', r'\bvisual\b', 
                r'\bshown\b', r'\bdisplayed\b', r'\bvisible\b', r'\bsee\b',
                r'\bappears\b', r'\blooks like\b', r'\bvisually\b'
            ],
            'text': [
                r'\btext\b', r'\bsays\b', r'\bmentioned\b', r'\bstated\b',
                r'\bwritten\b', r'\bdescribed\b', r'\bpassage\b', r'\bdocument\b'
            ],
            'audio': [
                r'\baudio\b', r'\bsound\b', r'\bheard\b', r'\blistening\b',
                r'\bspoken\b', r'\bvoice\b', r'\bnoise\b', r'\bspeech\b'
            ],
            'video': [
                r'\bvideo\b', r'\bclip\b', r'\bfootage\b', r'\bmotion\b',
                r'\bplaying\b', r'\bscene\b', r'\bmoving\b', r'\bframe\b'
            ]
        }
        
        # Count references to each modality
        references_found = {}
        
        for modality in modalities:
            if modality not in modality_references:
                continue
                
            # Count references to this modality
            count = 0
            for pattern in modality_references[modality]:
                count += len(re.findall(pattern, text, re.IGNORECASE))
            
            references_found[modality] = count
        
        # Calculate consistency score based on whether each modality is referenced
        modalities_referenced = sum(1 for m in modalities if references_found.get(m, 0) > 0)
        score = modalities_referenced / len(modalities) if modalities else 0.0
        
        return score
    
    def compute(self, 
               expected: str, 
               actual: str,
               modalities: Optional[List[str]] = None,
               return_details: bool = False) -> Union[float, Dict[str, float]]:
        """
        Compute multimodal reasoning score.
        
        Args:
            expected: Expected output text
            actual: Actual model response text
            modalities: List of modalities in the input (e.g., ['image', 'text'])
            return_details: Whether to return detailed scores
            
        Returns:
            Multimodal reasoning score or dictionary of detailed scores
        """
        if not expected or not actual:
            logger.warning("Empty input to multimodal reasoning metric")
            return 0.0 if not return_details else {"score": 0.0}
        
        # Default modalities if not provided
        if not modalities:
            modalities = ['text']
        
        # Calculate component scores
        accuracy_score = self.accuracy_metric.compute(expected, actual)
        reasoning_score = self._evaluate_reasoning(actual)
        
        # Only evaluate cross-modal consistency if there are multiple modalities
        if len(modalities) > 1:
            consistency_score = self._evaluate_modality_references(actual, modalities)
        else:
            consistency_score = 1.0  # Perfect by default for single modality
        
        # Calculate weighted score
        score = (self.accuracy_weight * accuracy_score +
                self.reasoning_weight * reasoning_score +
                self.consistency_weight * consistency_score)
        
        if return_details:
            return {
                "score": score,
                "accuracy": accuracy_score,
                "reasoning": reasoning_score,
                "consistency": consistency_score,
                "modalities": modalities
            }
        else:
            return score
    
    def compute_batch(self, 
                     expected_list: List[str], 
                     actual_list: List[str],
                     modalities_list: Optional[List[List[str]]] = None,
                     return_details: bool = False) -> Union[float, Dict[str, Any]]:
        """
        Compute multimodal reasoning scores for a batch of outputs.
        
        Args:
            expected_list: List of expected outputs
            actual_list: List of actual model responses
            modalities_list: List of modality lists for each sample
            return_details: Whether to return detailed scores
            
        Returns:
            Average score or dictionary with detailed information
        """
        if len(expected_list) != len(actual_list):
            raise ValueError(f"Mismatched lists: expected {len(expected_list)}, got {len(actual_list)}")
        
        # Use default modalities if not provided
        if not modalities_list:
            modalities_list = [['text']] * len(expected_list)
        elif len(modalities_list) != len(expected_list):
            raise ValueError(f"Mismatched modalities list: expected {len(expected_list)}, got {len(modalities_list)}")
        
        scores = []
        details = []
        
        for expected, actual, modalities in zip(expected_list, actual_list, modalities_list):
            result = self.compute(expected, actual, modalities, return_details=True)
            scores.append(result["score"])
            details.append(result)
        
        average_score = np.mean(scores) if scores else 0.0
        
        if return_details:
            # Calculate average component scores
            avg_accuracy = np.mean([d["accuracy"] for d in details])
            avg_reasoning = np.mean([d["reasoning"] for d in details])
            avg_consistency = np.mean([d["consistency"] for d in details])
            
            return {
                "score": average_score,
                "individual_scores": scores,
                "details": details,
                "avg_components": {
                    "accuracy": avg_accuracy,
                    "reasoning": avg_reasoning,
                    "consistency": avg_consistency
                }
            }
        else:
            return average_score 