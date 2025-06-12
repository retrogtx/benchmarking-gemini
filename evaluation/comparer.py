"""
Model comparison tool for comparing benchmark results across different models.
"""
import os
import sys
import json
import logging
import argparse
import glob
from typing import Dict, List, Any, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dotenv import load_dotenv

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare model benchmark results')
    
    parser.add_argument('--results-dir', '-d', type=str, required=False,
                        default=os.getenv('RESULTS_DIR', 'outputs/results'),
                        help='Directory containing results files')
    
    parser.add_argument('--output', '-o', type=str, required=False,
                        default=os.getenv('RESULTS_DIR', 'outputs/results'),
                        help='Output directory for comparison results')
    
    parser.add_argument('--models', '-m', type=str, nargs='+', 
                        default=None,
                        help='Models to compare (if not specified, use all available)')
    
    parser.add_argument('--metrics', type=str, nargs='+',
                        default=None,
                        help='Metrics to compare (if not specified, use all available)')
    
    parser.add_argument('--split', '-s', type=str, required=False,
                        default=None,
                        help='Data split to compare (if not specified, use all available)')
    
    parser.add_argument('--files', '-f', type=str, nargs='+',
                        default=None,
                        help='Specific result files to compare')
    
    parser.add_argument('--output-format', type=str, choices=['json', 'csv', 'both'],
                        default='both',
                        help='Output format for comparison results')
    
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable plot generation')
    
    return parser.parse_args()


def load_results(args):
    """Load result files for comparison."""
    results = []
    
    # If specific files are provided, load those
    if args.files:
        for file_path in args.files:
            try:
                with open(file_path, 'r') as f:
                    result = json.load(f)
                    results.append(result)
                    logger.info(f"Loaded result file: {file_path}")
            except Exception as e:
                logger.error(f"Error loading result file {file_path}: {e}")
    else:
        # Otherwise, search for results in the directory
        pattern = os.path.join(args.results_dir, '*.json')
        result_files = glob.glob(pattern)
        
        if not result_files:
            logger.error(f"No result files found in {args.results_dir}")
            return []
        
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    result = json.load(f)
                    
                    # Filter by model if specified
                    if args.models and result.get('model') not in args.models:
                        continue
                        
                    # Filter by split if specified
                    if args.split and result.get('split') != args.split:
                        continue
                    
                    results.append(result)
                    logger.info(f"Loaded result file: {file_path}")
            except Exception as e:
                logger.error(f"Error loading result file {file_path}: {e}")
    
    logger.info(f"Loaded {len(results)} result files for comparison")
    return results


def extract_comparison_data(results, args):
    """Extract data for comparison from the results."""
    if not results:
        return {}
    
    # Determine metrics to compare
    all_metrics = set()
    for result in results:
        metrics = result.get('metrics', {})
        all_metrics.update(metrics.keys())
    
    metrics_to_compare = args.metrics if args.metrics else list(all_metrics)
    
    # Group results by model
    models_data = {}
    
    for result in results:
        model = result.get('model', 'unknown')
        split = result.get('split', 'unknown')
        sample_count = result.get('sample_count', 0)
        timestamp = result.get('timestamp', '')
        
        if model not in models_data:
            models_data[model] = []
        
        # Extract metric scores
        metrics_data = {}
        for metric_name in metrics_to_compare:
            if metric_name in result.get('metrics', {}):
                metric_data = result['metrics'][metric_name]
                # Extract score (could be a dictionary or a scalar)
                if isinstance(metric_data, dict) and 'score' in metric_data:
                    score = metric_data['score']
                else:
                    score = metric_data
                metrics_data[metric_name] = score
        
        # Extract latency data
        latency = result.get('latency', {})
        
        # Add to model data
        models_data[model].append({
            'split': split,
            'sample_count': sample_count,
            'timestamp': timestamp,
            'metrics': metrics_data,
            'latency': latency
        })
    
    # Calculate aggregated statistics for each model
    comparison_data = {
        'metrics': metrics_to_compare,
        'models': {}
    }
    
    for model, runs in models_data.items():
        model_metrics = {metric: [] for metric in metrics_to_compare}
        latencies = []
        
        for run in runs:
            # Collect metric scores
            for metric in metrics_to_compare:
                if metric in run['metrics']:
                    model_metrics[metric].append(run['metrics'][metric])
            
            # Collect latencies
            if 'latency' in run and 'mean' in run['latency']:
                latencies.append(run['latency']['mean'])
        
        # Calculate average scores
        avg_metrics = {}
        for metric, scores in model_metrics.items():
            if scores:
                avg_metrics[metric] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'runs': len(scores)
                }
        
        # Calculate average latency
        avg_latency = None
        if latencies:
            avg_latency = {
                'mean': float(np.mean(latencies)),
                'std': float(np.std(latencies)),
                'min': float(np.min(latencies)),
                'max': float(np.max(latencies))
            }
        
        comparison_data['models'][model] = {
            'metrics': avg_metrics,
            'latency': avg_latency,
            'runs': len(runs)
        }
    
    return comparison_data


def generate_comparison_plots(comparison_data, output_dir):
    """Generate comparison plots for the models."""
    if not comparison_data or not comparison_data.get('models'):
        logger.warning("No data available for plotting")
        return
    
    metrics = comparison_data.get('metrics', [])
    models = comparison_data.get('models', {})
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Bar chart for each metric comparing models
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # Extract data
        model_names = []
        scores = []
        errors = []
        
        for model_name, model_data in models.items():
            if metric in model_data.get('metrics', {}):
                metric_data = model_data['metrics'][metric]
                model_names.append(model_name)
                scores.append(metric_data['mean'])
                errors.append(metric_data['std'])
        
        if not model_names:
            continue
            
        # Create the bar plot
        bars = plt.bar(model_names, scores, yerr=errors, capsize=10)
        
        # Add labels and title
        plt.xlabel('Models')
        plt.ylabel(f'{metric.capitalize()} Score')
        plt.title(f'Comparison of {metric.capitalize()} Across Models')
        plt.ylim(0, 1.0)  # Assume metrics are in [0, 1] range
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_{metric}.png'))
        plt.close()
    
    # 2. Radar chart comparing models across all metrics
    # Only create if we have multiple metrics
    if len(metrics) > 1:
        try:
            # Set up the radar chart
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, polar=True)
            
            # Compute angle for each metric
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
            # Close the polygon
            angles += angles[:1]
            
            # Add labels at each angle
            plt.xticks(angles[:-1], metrics)
            
            # Plot each model
            for model_name, model_data in models.items():
                values = []
                for metric in metrics:
                    if metric in model_data.get('metrics', {}):
                        values.append(model_data['metrics'][metric]['mean'])
                    else:
                        values.append(0)
                
                # Close the polygon
                values += values[:1]
                
                # Plot the model data
                ax.plot(angles, values, linewidth=2, label=model_name)
                ax.fill(angles, values, alpha=0.1)
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            plt.title('Model Performance Across Metrics')
            plt.savefig(os.path.join(output_dir, 'comparison_radar.png'))
            plt.close()
        except Exception as e:
            logger.error(f"Error generating radar chart: {e}")
    
    # 3. Latency comparison
    plt.figure(figsize=(10, 6))
    
    # Extract data
    model_names = []
    latencies = []
    errors = []
    
    for model_name, model_data in models.items():
        if model_data.get('latency'):
            model_names.append(model_name)
            latencies.append(model_data['latency']['mean'])
            errors.append(model_data['latency']['std'])
    
    if model_names:
        # Create the bar plot
        bars = plt.bar(model_names, latencies, yerr=errors, capsize=10)
        
        # Add labels and title
        plt.xlabel('Models')
        plt.ylabel('Latency (seconds)')
        plt.title('Comparison of Latency Across Models')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}s', ha='center', va='bottom')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison_latency.png'))
        plt.close()
    
    logger.info(f"Generated comparison plots in {output_dir}")


def save_comparison_results(comparison_data, args):
    """Save comparison results to file."""
    if not comparison_data:
        return
    
    os.makedirs(args.output, exist_ok=True)
    
    # Save as JSON
    if args.output_format in ['json', 'both']:
        json_path = os.path.join(args.output, 'comparison_results.json')
        with open(json_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        logger.info(f"Saved comparison results to {json_path}")
    
    # Save as CSV
    if args.output_format in ['csv', 'both']:
        csv_path = os.path.join(args.output, 'comparison_results.csv')
        
        try:
            import pandas as pd
            
            # Prepare data for DataFrame
            rows = []
            
            for model_name, model_data in comparison_data.get('models', {}).items():
                row = {'Model': model_name, 'Runs': model_data.get('runs', 0)}
                
                # Add metrics
                for metric in comparison_data.get('metrics', []):
                    if metric in model_data.get('metrics', {}):
                        metric_data = model_data['metrics'][metric]
                        row[f'{metric}_mean'] = metric_data['mean']
                        row[f'{metric}_std'] = metric_data['std']
                
                # Add latency
                if model_data.get('latency'):
                    row['latency_mean'] = model_data['latency']['mean']
                    row['latency_std'] = model_data['latency']['std']
                
                rows.append(row)
            
            # Create DataFrame and save
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved comparison results to {csv_path}")
            
        except ImportError:
            logger.warning("pandas not available, skipping CSV export")
    
    # Generate plots
    if not args.no_plots:
        plot_dir = os.path.join(args.output, 'plots')
        generate_comparison_plots(comparison_data, plot_dir)


def display_comparison_summary(comparison_data):
    """Display a summary of the comparison results."""
    if not comparison_data or not comparison_data.get('models'):
        print("No comparison data available")
        return
    
    print("\nModel Comparison Summary:")
    print("==========================")
    
    # Determine the best model for each metric
    metrics = comparison_data.get('metrics', [])
    models = comparison_data.get('models', {})
    
    best_models = {}
    for metric in metrics:
        best_score = -1
        best_model = None
        
        for model_name, model_data in models.items():
            if metric in model_data.get('metrics', {}):
                score = model_data['metrics'][metric]['mean']
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        if best_model:
            best_models[metric] = (best_model, best_score)
    
    # Display results
    print(f"Compared {len(models)} models across {len(metrics)} metrics\n")
    
    # Summary table
    print(f"{'Model':<15} {'Runs':<5} " + " ".join([f"{m[:10]:<10}" for m in metrics]) + " Latency(s)")
    print("-" * (15 + 5 + 10 * len(metrics) + 11))
    
    for model_name, model_data in models.items():
        runs = model_data.get('runs', 0)
        metric_scores = []
        
        for metric in metrics:
            if metric in model_data.get('metrics', {}):
                score = model_data['metrics'][metric]['mean']
                # Highlight best model for this metric
                if best_models.get(metric) and best_models[metric][0] == model_name:
                    metric_scores.append(f"\033[1m{score:.3f}\033[0m")
                else:
                    metric_scores.append(f"{score:.3f}")
            else:
                metric_scores.append("N/A")
        
        # Add latency
        if model_data.get('latency'):
            latency = model_data['latency']['mean']
            latency_str = f"{latency:.2f}"
        else:
            latency_str = "N/A"
        
        print(f"{model_name:<15} {runs:<5} " + " ".join([f"{s:<10}" for s in metric_scores]) + f" {latency_str}")
    
    # Best model by metric
    print("\nBest model by metric:")
    for metric, (model, score) in best_models.items():
        print(f"  {metric}: {model} ({score:.3f})")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load results
    results = load_results(args)
    
    if not results:
        logger.error("No results found for comparison")
        return
    
    # Extract comparison data
    comparison_data = extract_comparison_data(results, args)
    
    # Save comparison results
    save_comparison_results(comparison_data, args)
    
    # Display summary
    display_comparison_summary(comparison_data)


if __name__ == "__main__":
    main()
