"""
Data loaders for multimodal benchmarking.
"""
from .base import BaseLoader
from .text_loader import TextLoader
from .image_loader import ImageLoader
from .audio_loader import AudioLoader

# Registry of loaders by modality
LOADER_REGISTRY = {
    "text": TextLoader,
    "image": ImageLoader,
    "audio": AudioLoader,
}

def get_loader(modality: str, **kwargs):
    """
    Get a loader by modality.
    
    Args:
        modality: The modality type (text, image, audio)
        **kwargs: Additional arguments to pass to the loader
        
    Returns:
        A loader instance for the specified modality
    """
    if modality not in LOADER_REGISTRY:
        raise ValueError(f"Modality '{modality}' not supported. Available: {list(LOADER_REGISTRY.keys())}")
    
    loader_class = LOADER_REGISTRY[modality]
    return loader_class(**kwargs) 