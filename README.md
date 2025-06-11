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
