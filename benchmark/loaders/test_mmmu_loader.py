import unittest
import os
import json
from PIL import Image

from benchmark.loaders.mmmu_loader import (
    load_mmmu_data,
    normalize_mmmu_sample,
    get_mmmu_sample_for_debug
)

class TestMmmuLoader(unittest.TestCase):

    def test_normalize_mmmu_sample(self):
        """
        Tests the normalization of a single MMMU sample.
        """
        dummy_sample = get_mmmu_sample_for_debug()
        normalized_sample = normalize_mmmu_sample(dummy_sample)

        self.assertIn("question", normalized_sample)
        self.assertIn("options", normalized_sample)
        self.assertIn("answer", normalized_sample)
        self.assertIn("images", normalized_sample)
        self.assertIn("id", normalized_sample)

        self.assertEqual(normalized_sample["question"], dummy_sample["question"])
        self.assertEqual(normalized_sample["options"], dummy_sample["options"])
        self.assertEqual(normalized_sample["answer"], dummy_sample["answer"])
        self.assertEqual(normalized_sample["id"], dummy_sample["id"])
        self.assertEqual(len(normalized_sample["images"]), 0) # No image bytes in dummy data

    def test_load_mmmu_data(self):
        """
        Tests loading of MMMU data from a JSONL file.
        """
        # Create a dummy jsonl file for testing
        dummy_data = [get_mmmu_sample_for_debug()]
        file_path = "test_data.jsonl"
        with open(file_path, 'w') as f:
            for item in dummy_data:
                f.write(json.dumps(item) + "\n")
        
        loaded_data = load_mmmu_data(file_path)
        self.assertEqual(len(loaded_data), 1)
        self.assertEqual(loaded_data[0]['id'], dummy_data[0]['id'])

        # Clean up the dummy file
        os.remove(file_path)

if __name__ == "__main__":
    unittest.main() 