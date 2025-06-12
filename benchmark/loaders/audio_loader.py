"""
Audio data loader for benchmarking.
"""
import os
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from .base import BaseLoader

logger = logging.getLogger(__name__)

class AudioLoader(BaseLoader):
    """Loader for audio-based benchmark data."""
    
    def __init__(self, 
                data_dir: Optional[str] = None, 
                cache_dir: Optional[str] = None,
                file_format: str = "json",
                max_duration: float = 30.0,  # Max duration in seconds
                sample_rate: int = 16000):   # Target sample rate
        """
        Initialize the audio loader.
        
        Args:
            data_dir: Directory containing the raw data files
            cache_dir: Directory to cache processed data
            file_format: Format of the data files (json, csv, etc.)
            max_duration: Maximum duration of audio in seconds
            sample_rate: Target sample rate in Hz
        """
        super().__init__(data_dir, cache_dir, file_format)
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        
        # Check if we have audio processing libraries
        self.has_librosa = False
        try:
            import librosa
            self.has_librosa = True
            logger.info("Using librosa for audio processing")
        except ImportError:
            logger.warning("librosa not found. Audio processing capabilities will be limited.")
    
    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess an audio data sample.
        
        Args:
            sample: A raw audio data sample
            
        Returns:
            The preprocessed sample
        """
        # Normalize the sample using the base method
        normalized = self._normalize_sample(sample)
        
        # Extract audio path from input
        input_data = normalized["input"]
        
        # Handle different input formats
        audio_path = None
        prompt = ""
        
        if isinstance(input_data, dict):
            audio_path = input_data.get("audio_path")
            prompt = input_data.get("text", "")
        elif isinstance(input_data, str) and (input_data.endswith('.wav') or 
                                           input_data.endswith('.mp3') or
                                           input_data.endswith('.flac')):
            audio_path = input_data
        
        # Process the audio if path is provided
        if audio_path:
            # Make sure the path is absolute or relative to data_dir
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(self.data_dir, audio_path)
                
            # Verify the audio exists
            if not os.path.exists(audio_path):
                logger.warning(f"Audio file not found: {audio_path} for sample {normalized['id']}")
                normalized["input"] = {"text": prompt, "audio_path": None, "error": "Audio not found"}
                return normalized
            
            # Process the audio metadata
            try:
                duration = None
                sample_rate = None
                channels = None
                
                if self.has_librosa:
                    import librosa
                    # Get audio duration and properties without loading the full file
                    duration = librosa.get_duration(path=audio_path)
                    y, sr = librosa.load(audio_path, sr=None, duration=0.1)  # Load just a small snippet
                    sample_rate = sr
                    channels = 1 if len(y.shape) == 1 else y.shape[0]
                else:
                    # Try to get basic info using other methods
                    try:
                        from pydub import AudioSegment
                        audio = AudioSegment.from_file(audio_path)
                        duration = len(audio) / 1000.0  # Convert to seconds
                        sample_rate = audio.frame_rate
                        channels = audio.channels
                    except ImportError:
                        logger.warning("Neither librosa nor pydub available. Limited audio metadata.")
                        # Just use file size as a proxy for duration
                        file_size = os.path.getsize(audio_path)
                        # Rough estimate: 16kHz, 16-bit, mono ~ 32 KB per second
                        duration = file_size / (32 * 1024)
                
                # Create a processed version with target sample rate if needed
                processed_path = None
                if self.has_librosa and (duration > self.max_duration or sample_rate != self.sample_rate):
                    import librosa
                    import soundfile as sf
                    
                    # Construct processed file path
                    filename = os.path.basename(audio_path)
                    base, ext = os.path.splitext(filename)
                    processed_path = os.path.join(self.cache_dir, f"processed_{base}.wav")
                    
                    # Load, trim, and resample
                    y, sr = librosa.load(audio_path, sr=self.sample_rate, 
                                        duration=self.max_duration if duration > self.max_duration else None)
                    
                    # Save the processed audio
                    sf.write(processed_path, y, self.sample_rate)
                    
                    actual_duration = len(y) / self.sample_rate
                    logger.info(f"Processed audio from {duration:.2f}s to {actual_duration:.2f}s at {self.sample_rate}Hz")
                    audio_path = processed_path
                    duration = actual_duration
                    sample_rate = self.sample_rate
                
                # Update the normalized input
                normalized["input"] = {
                    "text": prompt,
                    "audio_path": audio_path,
                    "metadata": {
                        "duration": duration,
                        "sample_rate": sample_rate,
                        "channels": channels
                    }
                }
                
            except Exception as e:
                logger.error(f"Error processing audio {audio_path}: {e}")
                normalized["input"] = {"text": prompt, "audio_path": None, "error": str(e)}
        else:
            logger.warning(f"No audio path found for sample {normalized['id']}")
            normalized["input"] = {"text": prompt, "audio_path": None}
        
        return normalized
    
    def _load_audio(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Load an audio file into a numpy array.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Audio as a numpy array of shape (samples,) or None if loading fails
        """
        if not self.has_librosa:
            logger.warning("Cannot load audio data without librosa")
            return None
            
        try:
            import librosa
            
            # Load audio file with resampling if needed
            y, sr = librosa.load(audio_path, sr=self.sample_rate, 
                               duration=self.max_duration)
            
            return y
                
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            return None
    
    def load_data(self, 
                 filepath: Optional[str] = None, 
                 split: Optional[str] = None,
                 use_cache: bool = True,
                 load_audio: bool = False) -> List[Dict[str, Any]]:
        """
        Load and preprocess audio data.
        
        Args:
            filepath: Path to the data file or directory
            split: Data split to load (train, val, test)
            use_cache: Whether to use cached data if available
            load_audio: Whether to load audio data into memory
            
        Returns:
            List of preprocessed audio data samples
        """
        # Determine the file path
        if filepath is None:
            if split is not None:
                filepath = os.path.join(self.data_dir, f"{split}.json")
            else:
                filepath = os.path.join(self.data_dir, "data.json")
        
        # Generate cache file name
        cache_filename = os.path.basename(filepath)
        cache_filename = f"processed_audio_{cache_filename}"
        
        # Try to load from cache first
        if use_cache:
            cached_data = self.load_from_cache(cache_filename)
            if cached_data:
                logger.info(f"Loaded {len(cached_data)} samples from cache")
                
                # Load actual audio data if requested
                if load_audio and self.has_librosa:
                    for sample in cached_data:
                        audio_path = sample["input"].get("audio_path")
                        if audio_path:
                            sample["input"]["audio"] = self._load_audio(audio_path)
                
                return cached_data
        
        # Load the raw data
        raw_data = self.load_file(filepath)
        
        # Preprocess each sample
        processed_data = []
        for sample in raw_data:
            if self._validate_sample(sample):
                processed_sample = self.preprocess(sample)
                
                # Load actual audio data if requested
                if load_audio and self.has_librosa:
                    audio_path = processed_sample["input"].get("audio_path")
                    if audio_path:
                        processed_sample["input"]["audio"] = self._load_audio(audio_path)
                
                processed_data.append(processed_sample)
            else:
                logger.warning(f"Skipping invalid sample: {sample.get('id', 'unknown')}")
        
        # Save to cache if we have processed data
        if processed_data and use_cache:
            # Create a copy without audio data for caching
            cache_data = []
            for sample in processed_data:
                cache_sample = sample.copy()
                if "audio" in cache_sample["input"]:
                    # Remove the audio data before caching
                    cache_sample["input"] = cache_sample["input"].copy()
                    del cache_sample["input"]["audio"]
                cache_data.append(cache_sample)
            
            self.save_to_cache(cache_data, cache_filename)
        
        return processed_data
    
    def create_debug_sample(self) -> Dict[str, Any]:
        """
        Create a debug sample for testing.
        
        Returns:
            A sample audio data point
        """
        return {
            "id": "debug_audio_001",
            "input": {
                "text": "Transcribe this audio file.",
                "audio_path": "sample.wav"
            },
            "expected_output": "This is a test audio file for benchmarking speech recognition.",
            "metadata": {
                "source": "debug",
                "difficulty": "medium",
                "category": "speech_recognition",
                "language": "english"
            }
        } 