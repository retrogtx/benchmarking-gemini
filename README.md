# 🧠 Gemini Benchmark

A comprehensive benchmarking framework for evaluating Google's Gemini multimodal AI models across different tasks and modalities.

## 📋 Overview

This framework provides tools to:

1. 📊 **Benchmark Gemini models** across text, image, and audio modalities
2. 🔍 **Evaluate model performance** using a variety of metrics
3. 📈 **Compare results** between different models or model versions
4. 🧪 **Process and normalize data** for consistent evaluation

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Google API key with access to Gemini models

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/retrogtx/benchmarking-gemini.git
   cd benchmarking-gemini
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` to add your Google API key and other configuration.

## ⚡ Quick Start Guide

This guide walks through the complete end-to-end process for running the benchmarking system.

### 1. Setup Environment Variables

First, ensure your `.env` file is configured properly:

```bash
# Required
GOOGLE_API_KEY=your_google_api_key_here
MODEL_NAME=gemini  # IMPORTANT: Use "gemini" not "gemini-1.5-pro"

# Optional
MAX_OUTPUT_TOKENS=2048
TEMPERATURE=0.2
TOP_P=0.95
TOP_K=40
```

### 2. Generate Debug Data

For testing, generate sample data across different modalities:

```bash
# Generate test data for text and image modalities
python scripts/preprocess.py --modalities text image --splits test --debug

# If you want to generate data for all modalities and splits
python scripts/preprocess.py --modalities text image audio --splits test --debug
```

⚠️ **IMPORTANT**: The `--debug` flag creates files with the naming pattern `debug_{modality}_{split}.json` (e.g., `debug_text_test.json`). These files are saved in the `outputs/processed/` directory.

### 3. Run the Evaluation

Run the evaluation pipeline using the generated debug data:

```bash
# CRITICAL: Must use --debug flag to match the debug file naming pattern
python evaluation/runner.py --debug --data outputs/processed --split test --modalities text image --model gemini

# Full example with all modalities
python evaluation/runner.py --debug --data outputs/processed --split test --modalities text image audio --model gemini
```

⚠️ **CRITICAL**: The `--debug` flag is **REQUIRED** when using data generated with the `--debug` flag in the preprocessing step. Without this flag, the evaluation will fail with "No samples found" errors.

### 4. View Results

The evaluation results are saved in the `outputs/results/` directory as JSON files named with the pattern:
`{model}_{split}_{timestamp}.json`

You can examine the detailed results, or if you've run multiple evaluations, compare them:

```bash
# Compare results from different model runs (if you have multiple result files)
python evaluation/comparer.py --results outputs/results/gemini_test_*.json
```

### 5. Debug with Jupyter Notebook

For interactive debugging and visualization:

```bash
# Start Jupyter notebook
jupyter notebook notebooks/debug_pipeline.ipynb
```

### 6. Using Real Data

When you're ready to use real data instead of debug samples:

1. Place your data files in `benchmark/data/` with the naming pattern `{split}.json` (e.g., `test.json`)
2. Run preprocessing without the `--debug` flag:

```bash
python scripts/preprocess.py --modalities text image --splits test
```

3. Run evaluation on the processed real data (without the `--debug` flag):

```bash
python evaluation/runner.py --data outputs/processed --split test --modalities text image --model gemini
```

### 7. Troubleshooting

#### "No samples found for evaluation" Error

If you see this error, check:

1. **File naming mismatch**: Make sure you're using the `--debug` flag consistently in both preprocessing and evaluation scripts.
   - With `--debug`, files are named `debug_text_test.json`
   - Without `--debug`, files are named `processed_text_test.json`

2. **Verify files exist**: Check the contents of `outputs/processed/` directory to confirm the files were created:
   ```bash
   ls -la outputs/processed/
   ```

3. **Regenerate data**: If the files are missing, run the preprocessing script again:
   ```bash
   python scripts/preprocess.py --modalities text image audio --splits test --debug
   ```

4. **Check logs**: For more detailed error information, examine the log files in `outputs/logs/`

## 📊 Using the Framework

### Data Preparation

1. Place your benchmark data in the `benchmark/data/` directory
2. Run the preprocessing script to normalize and prepare the data:
   ```bash
   python scripts/preprocess.py --modalities text image --splits test
   ```

### Running Benchmarks

Run the evaluation pipeline:

```bash
python evaluation/runner.py --model gemini --split test --metrics accuracy f1 multimodal_reasoning
```

### Comparing Models

Compare results from multiple model runs:

```bash
python evaluation/comparer.py --models gemini-1.0-pro gemini-1.5-pro
```

## 📁 Project Structure

```
benchmarking-gemini/
├── benchmark/                 # Benchmark components
│   ├── data/                  # Benchmark datasets
│   ├── loaders/               # Data loaders for different modalities
│   └── metrics/               # Evaluation metrics
├── evaluation/                # Evaluation pipeline
│   ├── runner.py              # Main evaluation runner
│   └── comparer.py            # Tool for comparing results
├── models/                    # Model wrappers
│   ├── __init__.py            # Model registry
│   └── gemini.py              # Gemini model wrapper
├── notebooks/                 # Jupyter notebooks for exploration
├── outputs/                   # Output directories
│   ├── processed/             # Processed benchmark data
│   ├── results/               # Benchmark results
│   └── logs/                  # Log files
├── scripts/                   # Utility scripts
│   └── preprocess.py          # Data preprocessing script
├── .env.example               # Example environment variables
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

## 🔧 Configuration

The system can be configured through:

1. **Environment variables** (in `.env` file)
2. **Command-line arguments** to the various scripts
3. **Code-level configuration** for advanced scenarios

Key configuration options:

- `GOOGLE_API_KEY`: Your Google API key
- `MODEL_NAME`: Gemini model to use (e.g., "gemini-1.5-pro")
- `MAX_OUTPUT_TOKENS`: Maximum tokens in generated responses
- `TEMPERATURE`: Temperature for response generation

## 📊 Supported Metrics

The framework includes several evaluation metrics:

- **Accuracy**: Basic accuracy for text responses
- **F1/Precision/Recall**: For evaluating answers with multiple parts
- **Multimodal Reasoning**: Special metric for evaluating reasoning across modalities
- **Long Context**: Evaluates handling of long context inputs

## 📄 Supported Data Formats

Data should be in JSON format with the following structure (for now):

```json
[
  {
    "id": "unique_sample_id",
    "input": {
      "text": "Question or instruction text",
      "image_path": "path/to/image.jpg"  // Optional
    },
    "expected_output": "Expected model response",
    "metadata": {
      "source": "dataset_name",
      "difficulty": "easy|medium|hard",
      "category": "category_name"
    }
  }
]
```

## 📈 Example Workflow

1. **Prepare your benchmark data** in JSON format
2. **Run preprocessing** to normalize inputs:
   ```bash
   python scripts/preprocess.py
   ```
3. **Run the benchmark**:
   ```bash
   python evaluation/runner.py --model gemini --split test
   ```
4. **Analyze results** in the `outputs/results/` directory
5. **Compare different models**:
   ```bash
   python evaluation/comparer.py
   ```

## 🔍 Debugging

For debugging, use the provided notebook:

```bash
jupyter notebook notebooks/debug_pipeline.ipynb
```

This notebook walks through the complete pipeline with a single example for easier debugging.
