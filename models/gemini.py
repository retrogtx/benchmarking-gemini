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
import time
import random
from google.api_core.exceptions import ResourceExhausted
from google.api_core.exceptions import TooManyRequests as GoogleTooManyRequests  

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class GeminiModel:
    """Wrapper for Google's Gemini model with multi-modal input support (default: Gemini 2.0 Flash-Lite)."""
    
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
            
        # Default to Gemini 2.0 Flash-Lite if not specified
        self.model_name = model_name or os.getenv("MODEL_NAME", "gemini-2.0-flash-lite")
        
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
                         images: List[Image.Image] = None,
                         image_paths: List[str] = None,
                         audio_paths: List[str] = None) -> List[Union[str, Dict]]:
        """Prepare multimodal content for the Gemini API."""
        contents = []
        
        # Add text if provided
        if text:
            contents.append(text)
            
        # Add images if provided as objects
        if images:
            contents.extend(images)
        
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
                images: List[Image.Image] = None,
                image_paths: List[str] = None,
                audio_paths: List[str] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Generate a response from Gemini model using multimodal inputs.
        
        Args:
            text: Text prompt to send to the model
            images: List of PIL Image objects
            image_paths: List of paths to image files
            audio_paths: List of paths to audio files
            **kwargs: Additional keyword arguments to override generation config
            
        Returns:
            Dictionary containing:
                - 'response': The model's response text
                - 'metadata': Additional response metadata
        """
        if not any([text, images, image_paths, audio_paths]):
            raise ValueError("At least one input modality (text, image, image_paths, audio) must be provided.")
        
        # Prepare the multimodal contents
        contents = self._prepare_contents(text, images, image_paths, audio_paths)
        
        gen_config = {**self.generation_config, **kwargs}
        
        max_retries = int(os.getenv("MAX_RETRIES", 5))
        backoff_base = float(os.getenv("BACKOFF_BASE", 1.0))  # seconds

        attempt = 0
        while True:
            try:
                response = self.model.generate_content(
                    contents=contents,
                    generation_config=gen_config
                )

                response_text = response.text

                return {
                    "response": response_text,
                    "metadata": {
                        "model": self.model_name,
                        "generation_config": gen_config,
                        "input_modalities": {
                            "text": bool(text),
                            "image": bool(image_paths or images),
                            "audio": bool(audio_paths)
                        }
                    }
                }

            except (ResourceExhausted, GoogleTooManyRequests) as rate_err:
                error_msg = getattr(rate_err, "message", str(rate_err))

                if attempt >= max_retries:
                    logger.error(f"Rate/Quota limit hit and max retries exceeded. Last error: {error_msg}")
                    raise

                sleep_time = backoff_base * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(
                    f"Rate/Quota limit encountered (attempt {attempt + 1}/{max_retries}). "
                    f"Error: {error_msg} | Sleeping for {sleep_time:.2f}s before retrying."
                )
                time.sleep(sleep_time)
                attempt += 1
                continue
            except Exception as e:
                error_str = str(e).lower()
                if ("429" in error_str or "rate limit" in error_str or "exhausted" in error_str) and attempt < max_retries:
                    sleep_time = backoff_base * (2 ** attempt) + random.uniform(0, 0.5)
                    logger.warning(
                        f"Possible rate limit encountered (attempt {attempt + 1}/{max_retries}). "
                        f"Error: {e}. Sleeping for {sleep_time:.2f}s before retrying."
                    )
                    time.sleep(sleep_time)
                    attempt += 1
                    continue

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
        images=[Image.new('RGB', (60, 30), color = 'red')] # Example with a dummy image
    )
    
    print(response["response"])
