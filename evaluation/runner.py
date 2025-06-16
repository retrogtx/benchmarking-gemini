"""
Unified pipeline runner for model evaluation.
"""
import os
import sys
import json
import time
import logging
import argparse
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
import numpy as np

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model, list_available_models
from benchmark.loaders import get_loader
from benchmark.loaders.mmmu_loader import load_mmmu_data, normalize_mmmu_sample
from benchmark.metrics import get_metric, get_all_metrics

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run model evaluation')
    
    parser.add_argument('--model', '-m', type=str, required=False,
                        default=os.getenv('MODEL_NAME', 'gemini'),
                        help='Model to evaluate')
    
    parser.add_argument('--dataset', type=str, required=False,
                        default='default',
                        choices=['default', 'mmmu'],
                        help='Dataset to use for evaluation.')
    
    parser.add_argument('--data', '-d', type=str, required=False,
                        default=os.getenv('PROCESSED_DATA_DIR', 'outputs/processed'),
                        help='Directory with processed data')
    
    parser.add_argument('--output', '-o', type=str, required=False,
                        default=os.getenv('RESULTS_DIR', 'outputs/results'),
                        help='Output directory for results')
    
    parser.add_argument('--log-dir', type=str, required=False,
                        default=os.getenv('LOGS_DIR', 'outputs/logs'),
                        help='Directory for log output')
    
    parser.add_argument('--split', '-s', type=str, required=False,
                        default='test',
                        help='Data split to evaluate on (train, val, test)')
    
    parser.add_argument('--modalities', type=str, nargs='+', 
                        default=['text', 'image', 'audio'],
                        help='Modalities to evaluate (text, image, audio)')
    
    parser.add_argument('--metrics', type=str, nargs='+',
                        default=['accuracy', 'f1', 'multimodal_reasoning'],
                        help='Metrics to compute')
    
    parser.add_argument('--batch-size', '-b', type=int, required=False,
                        default=int(os.getenv('BATCH_SIZE', 8)),
                        help='Batch size for evaluation')
    
    parser.add_argument('--sample-count', '-n', type=int, required=False,
                        default=None,
                        help='Number of samples to evaluate (default: all)')
    
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with more verbose output')
    
    parser.add_argument('--list-models', action='store_true',
                        help='List available models and exit')
    
    parser.add_argument('--list-metrics', action='store_true',
                        help='List available metrics and exit')
    
    return parser.parse_args()


def setup_environment(args):
    """Set up the evaluation environment."""
    # Create output directories
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set up file logger
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.log_dir, f'eval_{args.model}_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Set log level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    logger.info(f"Evaluating model: {args.model}")
    logger.info(f"Data directory: {args.data}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Modalities: {args.modalities}")
    logger.info(f"Metrics: {args.metrics}")
    logger.info(f"Batch size: {args.batch_size}")


def load_evaluation_data(args):
    """Load data for evaluation."""
    if args.dataset == 'mmmu':
        logger.info("Loading MMMU dataset.")
        data_dir = "benchmark/data/mmmu"
        if args.debug:
            split_file = os.path.join(data_dir, "debug", "debug_sample.jsonl")
        else:
            split_file = os.path.join(data_dir, f"{args.split}.jsonl")

        if not os.path.exists(split_file):
            logger.error(f"MMMU data file not found at {split_file}. "
                         f"Please run scripts/download_mmmu.py first.")
            return []

        raw_samples = load_mmmu_data(split_file)
        all_samples = [normalize_mmmu_sample(s) for s in raw_samples]
    else:
        all_samples = []
        for modality in args.modalities:
            try:
                logger.info(f"Loading {modality} data")
                
                # Get the appropriate loader
                loader = get_loader(modality, data_dir=args.data, cache_dir=args.data)
                
                # Look for cached processed data
                cache_files = [f for f in os.listdir(args.data) 
                             if (f.startswith(f"processed_{modality}_") or 
                                (args.debug and f.startswith(f"debug_{modality}_"))) 
                             and f.endswith(".json")]
                
                # If we have a specific split, look for that
                split_files = [f for f in cache_files if args.split in f]
                target_files = split_files if split_files else cache_files
                
                if not target_files:
                    logger.warning(f"No processed {modality} data found for split {args.split}")
                    continue
                
                # Load data from each file
                for filename in target_files:
                    samples = loader.load_from_cache(filename)
                    if samples:
                        logger.info(f"Loaded {len(samples)} samples from {filename}")
                        all_samples.extend(samples)
            
            except Exception as e:
                logger.error(f"Error loading {modality} data: {e}")
    
    # Limit sample count if specified
    if args.sample_count and args.sample_count < len(all_samples):
        logger.info(f"Limiting to {args.sample_count} samples (from {len(all_samples)} total)")
        all_samples = all_samples[:args.sample_count]
    
    logger.info(f"Loaded {len(all_samples)} total samples for evaluation")
    return all_samples


def prepare_model_inputs(samples):
    """Prepare inputs for the model from the samples."""
    inputs = []
    expected_outputs = []
    sample_ids = []
    sample_modalities = []
    
    for sample in samples:
        sample_id = sample.get('id', 'unknown')

        # Handle MMMU sample format
        if 'question' in sample and 'options' in sample:
            # For MMMU, construct the prompt from question and options
            options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(sample['options'])])
            prompt = f"{sample['question']}\n\n{options_str}\n\nAnswer:"
            
            model_input = {
                'text': prompt,
                'images': sample.get('images'),
                'image_paths': None,
                'audio_paths': None,
            }
            expected_output = sample.get('answer', '')
            modalities = ['text', 'image'] if sample.get('images') else ['text']

        # Handle default sample format
        else:
            input_data = sample.get('input', {})
            expected_output = sample.get('expected_output', '')
            
            # Build input dictionary
            model_input = {
                'text': None,
                'images': None,
                'image_paths': None,
                'audio_paths': None
            }
            
            # Track modalities present
            modalities = []
            
            # Extract text input
            if isinstance(input_data, dict) and 'text' in input_data:
                model_input['text'] = input_data['text']
                modalities.append('text')
            
            # Extract image paths
            if isinstance(input_data, dict) and 'image_path' in input_data:
                if input_data['image_path']:
                    model_input['image_paths'] = [input_data['image_path']]
                    modalities.append('image')
            
            # Extract audio paths
            if isinstance(input_data, dict) and 'audio_path' in input_data:
                if input_data['audio_path']:
                    model_input['audio_paths'] = [input_data['audio_path']]
                    modalities.append('audio')
        
        inputs.append(model_input)
        expected_outputs.append(expected_output)
        sample_ids.append(sample_id)
        sample_modalities.append(modalities)
    
    return inputs, expected_outputs, sample_ids, sample_modalities


def run_model_evaluation(args, samples):
    """Run the model evaluation pipeline."""
    # Prepare inputs
    inputs, expected_outputs, sample_ids, sample_modalities = prepare_model_inputs(samples)
    
    try:
        # Initialize model
        model = get_model(args.model)
        logger.info(f"Initialized model: {args.model}")
        
        # Initialize metrics
        if args.metrics and args.metrics[0] == 'all':
            metrics = get_all_metrics()
            logger.info("Using all available metrics")
        else:
            metrics = {name: get_metric(name) for name in args.metrics}
            logger.info(f"Using metrics: {list(metrics.keys())}")
        
        # Run evaluation in batches
        all_responses = []
        all_latencies = []
        
        logger.info(f"Running evaluation on {len(inputs)} samples in batches of {args.batch_size}")
        
        for i in tqdm(range(0, len(inputs), args.batch_size)):
            batch_inputs = inputs[i:i + args.batch_size]
            batch_ids = sample_ids[i:i + args.batch_size]
            
            batch_responses = []
            batch_latencies = []
            
            for j, input_dict in enumerate(batch_inputs):
                sample_id = batch_ids[j]
                try:
                    # Time the model response
                    start_time = time.time()
                    response = model.generate(**input_dict)
                    end_time = time.time()
                    
                    latency = end_time - start_time
                    
                    batch_responses.append(response)
                    batch_latencies.append(latency)
                    
                    if args.debug:
                        logger.debug(f"Sample {sample_id} - Latency: {latency:.2f}s")
                        
                except Exception as e:
                    logger.error(f"Error generating response for sample {sample_id}: {e}")
                    # Add a placeholder for failed generations
                    batch_responses.append({
                        "response": None, 
                        "error": str(e),
                        "metadata": {"error": True}
                    })
                    batch_latencies.append(0.0)
            
            all_responses.extend(batch_responses)
            all_latencies.extend(batch_latencies)
        
        # Extract actual responses
        actual_outputs = [r.get("response", "") if isinstance(r, dict) else str(r) for r in all_responses]
        
        # Clean up None values
        actual_outputs = [output if output is not None else "" for output in actual_outputs]
        
        # Compute metrics
        results = {
            "model": args.model,
            "split": args.split,
            "sample_count": len(inputs),
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "latency": {
                "mean": float(np.mean(all_latencies)),
                "median": float(np.median(all_latencies)),
                "min": float(np.min(all_latencies)),
                "max": float(np.max(all_latencies)),
            },
        }
        
        for metric_name, metric in metrics.items():
            try:
                # For multimodal reasoning, we need to pass modalities
                if metric_name == 'multimodal_reasoning':
                    score = metric.compute_batch(
                        expected_outputs, 
                        actual_outputs,
                        modalities_list=sample_modalities,
                        return_details=True
                    )
                # For long context, we need to pass contexts too
                elif metric_name == 'long_context':
                    # Extract contexts (using text inputs as contexts)
                    contexts = [inp.get('text', '') for inp in inputs]
                    score = metric.compute_batch(
                        contexts,
                        expected_outputs, 
                        actual_outputs,
                        return_details=True
                    )
                else:
                    score = metric.compute_batch(
                        expected_outputs, 
                        actual_outputs,
                        return_details=True
                    )
                
                results["metrics"][metric_name] = score
                logger.info(f"{metric_name.capitalize()} score: {score['score']:.4f}")
            
            except Exception as e:
                logger.error(f"Error computing {metric_name}: {e}")
                results["metrics"][metric_name] = {"score": 0.0, "error": str(e)}
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(args.output, f'{args.model}_{args.split}_{timestamp}.json')
        
        # Add individual sample results
        sample_results = []
        for i, (sample_id, expected, actual, latency, response) in enumerate(
            zip(sample_ids, expected_outputs, actual_outputs, all_latencies, all_responses)
        ):
            sample_result = {
                "id": sample_id,
                "expected": expected,
                "actual": actual,
                "latency": latency,
                "modalities": sample_modalities[i],
            }
            
            # Add individual metric scores
            sample_result["metrics"] = {}
            for metric_name, metric in metrics.items():
                try:
                    # For multimodal reasoning, we need to pass modalities
                    if metric_name == 'multimodal_reasoning':
                        score = metric.compute(
                            expected, 
                            actual,
                            modalities=sample_modalities[i],
                            return_details=False
                        )
                    # For long context, we need to pass contexts too
                    elif metric_name == 'long_context':
                        context = inputs[i].get('text', '')
                        score = metric.compute(
                            context,
                            expected, 
                            actual,
                            return_details=False
                        )
                    else:
                        score = metric.compute(expected, actual, return_details=False)
                    
                    sample_result["metrics"][metric_name] = float(score)
                except Exception as e:
                    sample_result["metrics"][metric_name] = 0.0
            
            # Add response metadata if available
            if isinstance(response, dict) and "metadata" in response:
                sample_result["metadata"] = response["metadata"]
            
            sample_results.append(sample_result)
        
        results["samples"] = sample_results
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results to {results_file}")
        return results
    
    except Exception as e:
        logger.error(f"Error in evaluation pipeline: {e}")
        raise


def main():
    """Main entry point."""
    args = parse_args()
    
    # Handle utility commands
    if args.list_models:
        print("Available models:")
        for model_name in list_available_models():
            print(f"  - {model_name}")
        return
    
    if args.list_metrics:
        from benchmark.metrics import METRICS_REGISTRY
        print("Available metrics:")
        for metric_name in METRICS_REGISTRY:
            print(f"  - {metric_name}")
        return
    
    # Run the evaluation pipeline
    setup_environment(args)
    samples = load_evaluation_data(args)
    
    if not samples:
        logger.error("No samples found for evaluation")
        return
    
    try:
        results = run_model_evaluation(args, samples)
        
        # Print summary
        print("\nEvaluation Summary:")
        print(f"Model: {args.model}")
        print(f"Split: {args.split}")
        print(f"Samples: {len(samples)}")
        print("\nMetrics:")
        for metric_name, metric_result in results["metrics"].items():
            print(f"  {metric_name}: {metric_result['score']:.4f}")
        
        print("\nLatency:")
        print(f"  Mean: {results['latency']['mean']:.2f}s")
        print(f"  Median: {results['latency']['median']:.2f}s")
        print(f"  Min: {results['latency']['min']:.2f}s")
        print(f"  Max: {results['latency']['max']:.2f}s")
        
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")


if __name__ == "__main__":
    main()
