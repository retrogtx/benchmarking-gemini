"""
Metrics for evaluating benchmark results.
"""
from .accuracy import AccuracyMetric
from .f1 import F1Metric, PrecisionMetric, RecallMetric
from .multimodal import MultimodalReasoningMetric
from .long_context import LongContextMetric

# Registry of metrics
METRICS_REGISTRY = {
    "accuracy": AccuracyMetric,
    "f1": F1Metric,
    "precision": PrecisionMetric,
    "recall": RecallMetric,
    "multimodal_reasoning": MultimodalReasoningMetric,
    "long_context": LongContextMetric,
}

def get_metric(name: str, **kwargs):
    """
    Get a metric by name.
    
    Args:
        name: Name of the metric
        **kwargs: Additional arguments to pass to the metric constructor
        
    Returns:
        A metric instance
    """
    if name not in METRICS_REGISTRY:
        raise ValueError(f"Metric '{name}' not found. Available: {list(METRICS_REGISTRY.keys())}")
    
    metric_class = METRICS_REGISTRY[name]
    return metric_class(**kwargs)

def get_all_metrics(**kwargs):
    """
    Get all available metrics.
    
    Args:
        **kwargs: Additional arguments to pass to all metric constructors
        
    Returns:
        Dictionary mapping metric names to metric instances
    """
    return {name: cls(**kwargs) for name, cls in METRICS_REGISTRY.items()} 