import json
from PIL import Image
from typing import List, Dict, Any

def load_mmmu_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads MMMU data from a jsonl file.
    
    Args:
        file_path: The path to the jsonl file.

    Returns:
        A list of dictionaries, where each dictionary represents a sample.
    """
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def normalize_mmmu_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes a single MMMU sample to a unified schema.

    Args:
        sample: A dictionary representing a single sample from the MMMU dataset.

    Returns:
        A dictionary with the normalized sample data.
    """
    images = []
    for i in range(1, 8):  # Check for image_1 up to image_7
        image_key = f"image_{i}"
        if image_key in sample and sample[image_key]:
            # The 'bytes' field of the image dictionary contains the image data
            image_data = sample[image_key]['bytes']
            if image_data:
                import io
                images.append(Image.open(io.BytesIO(image_data)))

    return {
        "question": sample.get("question", ""),
        "options": sample.get("options", []),
        "answer": sample.get("answer", ""),
        "images": images,
        "id": sample.get("id", "")
    }

def get_mmmu_sample_for_debug():
    """
    Returns a dummy MMMU-style data sample for testing purposes.
    """
    return {
        "id": "debug_sample_1",
        "question": "This is a test question with an image <image 1>.",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "answer": "A",
        "image_1": {"bytes": None}, # In a real scenario, this would have image bytes
        "image_2": None,
        "explanation": "This is a dummy explanation."
    } 