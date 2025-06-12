#!/usr/bin/env python3
"""
Preprocessing script for benchmark data.

This script processes raw data into a format suitable for benchmarking:
- Resizes images
- Normalizes text
- Trims audio
- Generates statistics about the dataset
"""
import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.loaders import get_loader
from benchmark.loaders.base import BaseLoader
from benchmark.loaders.text_loader import TextLoader
from benchmark.loaders.image_loader import ImageLoader
from benchmark.loaders.audio_loader import AudioLoader

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess benchmark data')
    
    parser.add_argument('--input', '-i', type=str, required=False,
                        default=os.getenv('DATA_DIR', 'benchmark/data'),
                        help='Input data directory')
    
    parser.add_argument('--output', '-o', type=str, required=False,
                        default=os.getenv('PROCESSED_DATA_DIR', 'outputs/processed'),
                        help='Output directory for processed data')
    
    parser.add_argument('--log-dir', type=str, required=False,
                        default=os.getenv('LOGS_DIR', 'outputs/logs'),
                        help='Directory for log output')
    
    parser.add_argument('--modalities', '-m', type=str, nargs='+', 
                        default=['text', 'image', 'audio'],
                        help='Modalities to process (text, image, audio)')
    
    parser.add_argument('--splits', '-s', type=str, nargs='+',
                        default=['train', 'val', 'test'],
                        help='Data splits to process')
    
    parser.add_argument('--plot', '-p', action='store_true',
                        help='Generate plots of dataset statistics')
    
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Enable debug mode with sample data')
    
    return parser.parse_args()


def setup_dirs(args):
    """Set up necessary directories."""
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set up file logger
    file_handler = logging.FileHandler(os.path.join(args.log_dir, 'preprocess.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Processing data from {args.input} to {args.output}")


def process_data(args):
    """Process data for each modality and split."""
    stats = {}
    all_processed_data = []
    
    for modality in args.modalities:
        try:
            logger.info(f"Processing {modality} data")
            
            # Get the appropriate loader
            loader = get_loader(modality, data_dir=args.input, cache_dir=args.output)
            
            # Process each split
            for split in args.splits:
                logger.info(f"Processing {split} split for {modality}")
                
                try:
                    # If in debug mode, generate sample data instead of loading
                    if args.debug:
                        if hasattr(loader, 'create_debug_sample'):
                            # Generate 5 debug samples
                            samples = [loader.create_debug_sample() for _ in range(5)]
                            # Process them
                            processed_samples = []
                            for sample in samples:
                                if loader._validate_sample(sample):
                                    processed_sample = loader.preprocess(sample)
                                    processed_samples.append(processed_sample)
                            
                            # Save to cache
                            loader.save_to_cache(processed_samples, f"debug_{modality}_{split}.json")
                            
                            # Track stats
                            if modality not in stats:
                                stats[modality] = {}
                            stats[modality][split] = len(processed_samples)
                            
                            # Add to all processed data
                            all_processed_data.extend(processed_samples)
                            
                            logger.info(f"Created {len(processed_samples)} debug samples for {modality} {split}")
                        else:
                            logger.warning(f"Loader for {modality} doesn't support debug samples")
                    else:
                        # Normal processing
                        input_path = os.path.join(args.input, f"{split}.json")
                        
                        # Skip if file doesn't exist
                        if not os.path.exists(input_path):
                            logger.warning(f"Input file not found: {input_path}")
                            continue
                        
                        # Process the data
                        processed_data = loader.load_data(
                            filepath=input_path, 
                            split=split, 
                            use_cache=True
                        )
                        
                        # Track stats
                        if modality not in stats:
                            stats[modality] = {}
                        stats[modality][split] = len(processed_data)
                        
                        # Add to all processed data
                        all_processed_data.extend(processed_data)
                        
                        logger.info(f"Processed {len(processed_data)} samples for {modality} {split}")
                
                except Exception as e:
                    logger.error(f"Error processing {modality} {split}: {e}")
        
        except Exception as e:
            logger.error(f"Error setting up loader for {modality}: {e}")
    
    return stats, all_processed_data


def generate_stats(stats, processed_data, args):
    """Generate and save statistics about the processed data."""
    # Save stats to JSON
    stats_path = os.path.join(args.output, 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved statistics to {stats_path}")
    
    # If plotting is enabled, generate plots
    if args.plot and processed_data:
        try:
            plot_dir = os.path.join(args.output, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            
            # Plot distribution of modalities
            modality_counts = {}
            for sample in processed_data:
                input_data = sample.get('input', {})
                modalities = []
                
                if isinstance(input_data, dict):
                    if 'text' in input_data:
                        modalities.append('text')
                    if 'image_path' in input_data:
                        modalities.append('image')
                    if 'audio_path' in input_data:
                        modalities.append('audio')
                
                modality_key = '+'.join(sorted(modalities)) if modalities else 'unknown'
                modality_counts[modality_key] = modality_counts.get(modality_key, 0) + 1
            
            # Create pie chart of modality distribution
            plt.figure(figsize=(10, 6))
            plt.pie(modality_counts.values(), labels=modality_counts.keys(), autopct='%1.1f%%')
            plt.title('Distribution of Modalities')
            plt.savefig(os.path.join(plot_dir, 'modality_distribution.png'))
            
            # Plot difficulty distribution if available
            difficulty_counts = {}
            for sample in processed_data:
                difficulty = sample.get('metadata', {}).get('difficulty')
                if difficulty:
                    difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
            
            if difficulty_counts:
                plt.figure(figsize=(10, 6))
                plt.bar(difficulty_counts.keys(), difficulty_counts.values())
                plt.title('Distribution of Difficulty Levels')
                plt.savefig(os.path.join(plot_dir, 'difficulty_distribution.png'))
            
            logger.info(f"Generated plots in {plot_dir}")
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")


def main():
    """Main entry point."""
    args = parse_args()
    setup_dirs(args)
    
    # Process the data
    stats, processed_data = process_data(args)
    
    # Generate statistics
    generate_stats(stats, processed_data, args)
    
    logger.info("Preprocessing complete")


if __name__ == "__main__":
    main()
