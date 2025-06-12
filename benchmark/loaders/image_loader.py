"""
Image data loader for benchmarking.
"""
import os
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from PIL import Image
from .base import BaseLoader

logger = logging.getLogger(__name__)

class ImageLoader(BaseLoader):
    """Loader for image-based benchmark data."""
    
    def __init__(self, 
                data_dir: Optional[str] = None, 
                cache_dir: Optional[str] = None,
                file_format: str = "json",
                image_size: Tuple[int, int] = (512, 512),
                normalize: bool = True):
        """
        Initialize the image loader.
        
        Args:
            data_dir: Directory containing the raw data files
            cache_dir: Directory to cache processed data
            file_format: Format of the data files (json, csv, etc.)
            image_size: Target size for images (width, height)
            normalize: Whether to normalize pixel values
        """
        super().__init__(data_dir, cache_dir, file_format)
        self.image_size = image_size
        self.normalize = normalize
        
    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess an image data sample.
        
        Args:
            sample: A raw image data sample
            
        Returns:
            The preprocessed sample
        """
        # Normalize the sample using the base method
        normalized = self._normalize_sample(sample)
        
        # Extract image path from input
        input_data = normalized["input"]
        
        # Handle different input formats
        image_path = None
        prompt = ""
        
        if isinstance(input_data, dict):
            image_path = input_data.get("image_path")
            prompt = input_data.get("text", "")
        elif isinstance(input_data, str) and (input_data.endswith('.jpg') or 
                                           input_data.endswith('.jpeg') or
                                           input_data.endswith('.png')):
            image_path = input_data
        
        # Process the image if path is provided
        if image_path:
            # Make sure the path is absolute or relative to data_dir
            if not os.path.isabs(image_path):
                image_path = os.path.join(self.data_dir, image_path)
                
            # Verify the image exists
            if not os.path.exists(image_path):
                logger.warning(f"Image file not found: {image_path} for sample {normalized['id']}")
                normalized["input"] = {"text": prompt, "image_path": None, "error": "Image not found"}
                return normalized
            
            # Process the image but don't load the pixels here to save memory
            # Just validate and get basic metadata
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    format = img.format
                    mode = img.mode
                
                # Create a processed version with target size if needed
                processed_path = None
                if width != self.image_size[0] or height != self.image_size[1]:
                    # Construct processed file path
                    filename = os.path.basename(image_path)
                    processed_path = os.path.join(self.cache_dir, f"processed_{filename}")
                    
                    # Resize and save the image
                    with Image.open(image_path) as img:
                        resized_img = img.resize(self.image_size, Image.Resampling.LANCZOS)
                        resized_img.save(processed_path)
                    
                    logger.info(f"Resized image from {width}x{height} to {self.image_size[0]}x{self.image_size[1]}")
                    image_path = processed_path
                
                # Update the normalized input
                normalized["input"] = {
                    "text": prompt,
                    "image_path": image_path,
                    "metadata": {
                        "original_size": (width, height),
                        "format": format,
                        "mode": mode
                    }
                }
                
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                normalized["input"] = {"text": prompt, "image_path": None, "error": str(e)}
        else:
            logger.warning(f"No image path found for sample {normalized['id']}")
            normalized["input"] = {"text": prompt, "image_path": None}
        
        return normalized
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from file into a numpy array.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image as a numpy array
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if needed
                if img.size != self.image_size:
                    img = img.resize(self.image_size, Image.Resampling.LANCZOS)
                
                # Convert to numpy array
                img_array = np.array(img)
                
                # Normalize if requested
                if self.normalize:
                    img_array = img_array.astype(np.float32) / 255.0
                
                return img_array
                
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return empty array with correct shape
            return np.zeros((self.image_size[1], self.image_size[0], 3), 
                          dtype=np.float32 if self.normalize else np.uint8)
    
    def load_data(self, 
                 filepath: Optional[str] = None, 
                 split: Optional[str] = None,
                 use_cache: bool = True,
                 load_images: bool = False) -> List[Dict[str, Any]]:
        """
        Load and preprocess image data.
        
        Args:
            filepath: Path to the data file or directory
            split: Data split to load (train, val, test)
            use_cache: Whether to use cached data if available
            load_images: Whether to load image pixels into memory
            
        Returns:
            List of preprocessed image data samples
        """
        # Determine the file path
        if filepath is None:
            if split is not None:
                filepath = os.path.join(self.data_dir, f"{split}.json")
            else:
                filepath = os.path.join(self.data_dir, "data.json")
        
        # Generate cache file name
        cache_filename = os.path.basename(filepath)
        cache_filename = f"processed_image_{cache_filename}"
        
        # Try to load from cache first
        if use_cache:
            cached_data = self.load_from_cache(cache_filename)
            if cached_data:
                logger.info(f"Loaded {len(cached_data)} samples from cache")
                
                # Load actual image pixels if requested
                if load_images:
                    for sample in cached_data:
                        image_path = sample["input"].get("image_path")
                        if image_path:
                            sample["input"]["image"] = self._load_image(image_path)
                
                return cached_data
        
        # Load the raw data
        raw_data = self.load_file(filepath)
        
        # Preprocess each sample
        processed_data = []
        for sample in raw_data:
            if self._validate_sample(sample):
                processed_sample = self.preprocess(sample)
                
                # Load actual image pixels if requested
                if load_images:
                    image_path = processed_sample["input"].get("image_path")
                    if image_path:
                        processed_sample["input"]["image"] = self._load_image(image_path)
                
                processed_data.append(processed_sample)
            else:
                logger.warning(f"Skipping invalid sample: {sample.get('id', 'unknown')}")
        
        # Save to cache if we have processed data
        if processed_data and use_cache:
            # Create a copy without image data for caching
            cache_data = []
            for sample in processed_data:
                cache_sample = sample.copy()
                if "image" in cache_sample["input"]:
                    # Remove the image pixels before caching
                    cache_sample["input"] = cache_sample["input"].copy()
                    del cache_sample["input"]["image"]
                cache_data.append(cache_sample)
            
            self.save_to_cache(cache_data, cache_filename)
        
        return processed_data
    
    def create_debug_sample(self, include_image: bool = True) -> Dict[str, Any]:
        """
        Create a debug sample for testing.
        
        Args:
            include_image: Whether to include a placeholder image
            
        Returns:
            A sample image data point
        """
        debug_sample = {
            "id": "debug_image_001",
            "input": {
                "text": "What is shown in this image?",
                "image_path": "placeholder.jpg" if include_image else None
            },
            "expected_output": "The image shows a placeholder test pattern.",
            "metadata": {
                "source": "debug",
                "difficulty": "medium",
                "category": "image_recognition"
            }
        }
        
        # If include_image is True, generate a simple test pattern
        if include_image and not os.path.exists(os.path.join(self.data_dir, "placeholder.jpg")):
            try:
                # Create a directory if it doesn't exist
                os.makedirs(self.data_dir, exist_ok=True)
                
                # Create a simple test pattern
                img = Image.new('RGB', self.image_size, color=(73, 109, 137))
                
                # Draw a pattern
                from PIL import ImageDraw
                draw = ImageDraw.Draw(img)
                
                # Draw a grid
                for i in range(0, self.image_size[0], 50):
                    draw.line([(i, 0), (i, self.image_size[1])], fill=(255, 255, 255), width=1)
                
                for i in range(0, self.image_size[1], 50):
                    draw.line([(0, i), (self.image_size[0], i)], fill=(255, 255, 255), width=1)
                
                # Draw text
                from PIL import ImageFont
                try:
                    font = ImageFont.truetype("Arial.ttf", 40)
                except:
                    font = ImageFont.load_default()
                
                draw.text((self.image_size[0]//2 - 100, self.image_size[1]//2 - 20), 
                         "Test Image", fill=(255, 255, 255), font=font)
                
                # Save the image
                img.save(os.path.join(self.data_dir, "placeholder.jpg"))
                
                logger.info(f"Created debug test image at {os.path.join(self.data_dir, 'placeholder.jpg')}")
                
            except Exception as e:
                logger.error(f"Error creating debug image: {e}")
                debug_sample["input"]["image_path"] = None
        
        return debug_sample 