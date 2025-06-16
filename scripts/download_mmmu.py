import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def download_and_split_mmmu(output_dir: str = "benchmark/data/mmmu"):
    """
    Downloads the MMMU dataset from Hugging Face, saves the splits,
    and creates a small debug sample.
    """
    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    debug_dir = Path(output_dir) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Importing datasets library...")
        import datasets
        print(f"Using datasets version: {datasets.__version__}")
    except ImportError:
        print("Error: The 'datasets' library is not installed.")
        print("Install it with: pip install datasets")
        print("Note: This will also install a compatible version of pyarrow automatically.")
        sys.exit(1)

    try:
        # The correct repository for MMMU dataset
        repo_id = "MMMU/MMMU"
        print(f"Loading MMMU dataset from repository: {repo_id}")
        
        # Try to list all available configs directly
        try:
            print(f"Checking if repository {repo_id} exists...")
            available_configs = datasets.get_dataset_config_names(repo_id)
            print(f"Available configurations: {available_configs}")
        except Exception as e:
            print(f"Error getting configurations from {repo_id}: {e}")
            print("Trying alternative repository 'mmmu/mmmu'...")
            repo_id = "mmmu/mmmu"
            try:
                available_configs = datasets.get_dataset_config_names(repo_id)
                print(f"Available configurations from {repo_id}: {available_configs}")
            except Exception as e2:
                print(f"Error getting configurations from alternative repository: {e2}")
                print("Falling back to direct loading...")
                # Try direct loading approach
                try:
                    print("Attempting to load default configuration...")
                    test_load = datasets.load_dataset(repo_id, split="test")
                    print(f"Successfully loaded test split with {len(test_load)} samples")
                    available_configs = ["default"]
                except Exception as e3:
                    print(f"Error loading default configuration: {e3}")
                    raise Exception("Cannot access MMMU dataset. Please check your internet connection and try again.")
        
        # Load a few representative subjects instead of all (which would be too large)
        # This selection covers different disciplines: Math, CS, and Economics
        selected_configs = ["Math", "Computer_Science", "Economics"]
        
        # Check if selected configs are available
        valid_configs = []
        for config in selected_configs:
            if config in available_configs:
                valid_configs.append(config)
            else:
                print(f"Warning: Config '{config}' not found in available configurations.")
        
        if not valid_configs and "default" not in available_configs:
            if "dev" in available_configs:
                valid_configs = ["dev"]
                print(f"Using 'dev' configuration instead.")
            elif len(available_configs) > 0:
                valid_configs = [available_configs[0]]
                print(f"Using '{valid_configs[0]}' configuration instead.")
            else:
                print("Error: None of the selected configurations are available.")
                print(f"Please choose from: {available_configs}")
                sys.exit(1)
            
        print(f"Loading selected configurations: {valid_configs}")
        
        # Dictionary to hold datasets by split
        merged_datasets = {}
        
        # Load each selected configuration and merge by split
        if "default" in available_configs or not valid_configs:
            print("Loading default configuration...")
            try:
                # Try to load the dataset with default config and standard splits
                for split_name in ["train", "validation", "test"]:
                    try:
                        print(f"Loading {split_name} split...")
                        split_data = datasets.load_dataset(repo_id, split=split_name)
                        print(f"Loaded {len(split_data)} samples from {split_name} split")
                        merged_datasets[split_name] = split_data
                    except Exception as e:
                        print(f"Error loading {split_name} split: {e}")
            except Exception as e:
                print(f"Error loading dataset with default configuration: {e}")
                sys.exit(1)
        else:
            for config in valid_configs:
                print(f"Loading configuration: {config}")
                try:
                    config_dataset = datasets.load_dataset(repo_id, config)
                    print(f"Loaded configuration {config} with splits: {list(config_dataset.keys())}")
                    
                    for split_name, split_data in config_dataset.items():
                        print(f"Processing {split_name} split with {len(split_data)} samples")
                        if split_name not in merged_datasets:
                            merged_datasets[split_name] = split_data
                        else:
                            # Concatenate with existing split data
                            print(f"Merging with existing {split_name} split")
                            merged_datasets[split_name] = datasets.concatenate_datasets([
                                merged_datasets[split_name], 
                                split_data
                            ])
                except Exception as e:
                    print(f"Error processing configuration {config}: {e}")
        
        if not merged_datasets:
            print("Error: No data was loaded. Cannot proceed.")
            sys.exit(1)
            
        print("Datasets loaded successfully.")
        print(f"Splits available: {list(merged_datasets.keys())}")
        for split_name, data in merged_datasets.items():
            print(f"  - {split_name}: {len(data)} samples")
    except Exception as e:
        print(f"Error downloading or processing the MMMU dataset: {e}")
        print("\nPossible solutions:")
        print("1. Check your internet connection")
        print("2. Ensure you have a compatible version of datasets and pyarrow installed")
        print("3. Try running: pip install 'datasets>=2.14.0' 'pyarrow>=14.0.0'")
        sys.exit(1)

    try:
        # Save the merged splits manually to avoid encoding issues
        for split_name, split_data in merged_datasets.items():
            output_path = os.path.join(output_dir, f"{split_name}.jsonl")
            print(f"Saving {split_name} split with {len(split_data)} samples to {output_path}...")
            
            # Convert to list of dictionaries and sanitize
            samples = []
            for idx, sample in enumerate(split_data):
                try:
                    # Convert to dict and handle potential encoding issues
                    sample_dict = dict(sample)

                    # Remove image data if present, as it's not JSON serializable
                    if "image" in sample_dict:
                        # We can't save the raw image, so we remove it.
                        # The loader will handle fetching images later.
                        del sample_dict["image"]
                        
                    # Replace problematic fields or convert types if needed
                    samples.append(sample_dict)
                    if idx > 0 and idx % 1000 == 0:
                        print(f"  Processed {idx}/{len(split_data)} samples")
                except Exception as e:
                    print(f"Error processing sample {idx}, skipping: {e}")
                    continue
            
            # Write manually line by line
            with open(output_path, 'w', encoding='utf-8') as f:
                for idx, sample in enumerate(samples):
                    try:
                        # Use ensure_ascii=False to properly handle non-ASCII characters
                        json_line = json.dumps(sample, ensure_ascii=False)
                        f.write(json_line + '\n')
                        if idx > 0 and idx % 1000 == 0:
                            print(f"  Written {idx}/{len(samples)} samples")
                    except Exception as e:
                        print(f"Error writing sample {idx}, skipping: {e}")
                        continue
            
            # Verify the file was written successfully
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"{split_name} split saved with {len(samples)} samples.")
            else:
                print(f"Warning: {split_name} split file is empty or not created!")

        # Create and save a debug sample from the validation split
        if "validation" in merged_datasets:
            print("Creating a debug sample from the validation split...")
            
            debug_count = min(5, len(merged_datasets["validation"]))
            debug_output_path = os.path.join(debug_dir, "debug_sample.jsonl")
            
            # Get first few samples
            validation_samples = merged_datasets["validation"].select(range(debug_count))
            
            # Write debug samples manually
            with open(debug_output_path, 'w', encoding='utf-8') as f:
                for i in range(debug_count):
                    try:
                        sample = dict(validation_samples[i])
                        # Remove image data if present
                        if "image" in sample:
                            del sample["image"]
                        json_line = json.dumps(sample, ensure_ascii=False)
                        f.write(json_line + '\n')
                    except Exception as e:
                        print(f"Error writing debug sample {i}, skipping: {e}")
                        continue
            
            # Verify the debug file was written successfully
            if os.path.exists(debug_output_path) and os.path.getsize(debug_output_path) > 0:
                print(f"Debug sample saved to {debug_output_path} with {debug_count} samples")
            else:
                print(f"Warning: Debug sample file is empty or not created!")
        else:
            print("Could not create debug sample: 'validation' split not found.")
    except Exception as e:
        print(f"Error saving the dataset: {e}")
        sys.exit(1)

    print("\nAll tasks complete.")

if __name__ == "__main__":
    download_and_split_mmmu() 