"""
Gemini API wrapper for benchmarking.
Supports multi-modal inputs (text, image, audio).
"""
import os
import base64
from typing import Dict, List, Optional, Union, Any
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class GeminiModel:
    """Wrapper for Google's Gemini model with multi-modal input support."""
    
    def __init__(self, model_name: str = None, api_key: str = None):
        """
        Initialize the Gemini model wrapper.
        
        Args:
            model_name: Name of the Gemini model to use (e.g., 'gemini-1.5-pro')
            api_key: Google API key. If None, will try to load from environment.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API key not provided and not found in environment.")
            
        self.model_name = model_name or os.getenv("MODEL_NAME", "gemini-1.5-pro")
        
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        
        # Load generation configuration from environment or use defaults
        self.generation_config = {
            "max_output_tokens": int(os.getenv("MAX_OUTPUT_TOKENS", 2048)),
            "temperature": float(os.getenv("TEMPERATURE", 0.2)),
            "top_p": float(os.getenv("TOP_P", 0.95)),
            "top_k": int(os.getenv("TOP_K", 40)),
        }
        
        # Initialize the model
        try:
            self.model = genai.GenerativeModel(model_name=self.model_name)
            logger.info(f"Initialized Gemini model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise
    
    def _process_image(self, image_path: str) -> Any:
        """Process image for input to Gemini API."""
        try:
            image = Image.open(image_path)
            return image
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise
    
    def _prepare_contents(self, 
                         text: str = None, 
                         image_paths: List[str] = None,
                         audio_paths: List[str] = None) -> List[Union[str, Dict]]:
        """Prepare multimodal content for the Gemini API."""
        contents = []
        
        # Add text if provided
        if text:
            contents.append(text)
            
        # Add images if provided
        if image_paths:
            for img_path in image_paths:
                image = self._process_image(img_path)
                contents.append(image)
        
        # TODO: Add audio support once Gemini API supports it natively
        if audio_paths:
            logger.warning("Audio input support is currently limited. Using placeholders.")
            # For now, just mention the audio in the prompt
            audio_desc = f"\n[Referenced audio files: {', '.join(audio_paths)}]"
            if text:
                contents[0] += audio_desc
            else:
                contents.append(audio_desc)
                
        return contents
    
    def generate(self, 
                text: str = None, 
                image_paths: List[str] = None,
                audio_paths: List[str] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Generate a response from Gemini model using multimodal inputs.
        
        Args:
            text: Text prompt to send to the model
            image_paths: List of paths to image files
            audio_paths: List of paths to audio files
            **kwargs: Additional keyword arguments to override generation config
            
        Returns:
            Dictionary containing:
                - 'response': The model's response text
                - 'metadata': Additional response metadata
        """
        if not any([text, image_paths, audio_paths]):
            raise ValueError("At least one input modality (text, image, audio) must be provided.")
        
        # Prepare the multimodal contents
        contents = self._prepare_contents(text, image_paths, audio_paths)
        
        # Update generation config with any provided kwargs
        gen_config = {**self.generation_config, **kwargs}
        
        try:
            # Generate the response
            response = self.model.generate_content(
                contents=contents,
                generation_config=gen_config
            )
            
            # Extract the text response
            response_text = response.text
            
            # Return the response with metadata
            return {
                "response": response_text,
                "metadata": {
                    "model": self.model_name,
                    "generation_config": gen_config,
                    "input_modalities": {
                        "text": bool(text),
                        "image": bool(image_paths),
                        "audio": bool(audio_paths)
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": None,
                "error": str(e),
                "metadata": {
                    "model": self.model_name,
                    "generation_config": gen_config
                }
            }

# Example usage
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Initialize the model
    model = GeminiModel()
    
    # Generate a response
    response = model.generate(
        text="Describe what you see in this image in detail.",
        image_paths=["path/to/image.jpg"]
    )
    
    print(response["response"])
