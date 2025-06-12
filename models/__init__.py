"""
Model registry for benchmarking models.
"""
from typing import Dict, Type, Any, Optional
from .gemini import GeminiModel

# Model registry dictionary that maps model names to their class
MODEL_REGISTRY = {
    "gemini": GeminiModel,
    # Add more models here as they are implemented
}

def get_model(name: str, **kwargs) -> Any:
    """
    Get a model instance by name.
    
    Args:
        name: Name of the model to get
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        An instance of the requested model
        
    Raises:
        ValueError: If the model name is not found in the registry
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry. Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[name]
    return model_class(**kwargs)

def list_available_models() -> Dict[str, Type]:
    """
    List all available models in the registry.
    
    Returns:
        Dictionary mapping model names to their classes
    """
    return MODEL_REGISTRY.copy()
