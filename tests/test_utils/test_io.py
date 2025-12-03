import os
import tempfile
import unittest

import torch

from src.utils.io import load_state, save_state


class TestIO(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ckpt_path = os.path.join(self.temp_dir.name, "model.pt")
        self.model = torch.nn.Linear(10, 2)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_and_load_state(self):
        # Save
        save_state(self.model, self.ckpt_path)
        self.assertTrue(os.path.exists(self.ckpt_path))

        # Modify model
        original_weight = self.model.weight.clone()
        torch.nn.init.xavier_uniform_(self.model.weight)
        self.assertFalse(torch.equal(self.model.weight, original_weight))

        # Load
        load_state(self.model, self.ckpt_path)
        self.assertTrue(torch.equal(self.model.weight, original_weight))

    def test_save_creates_directories(self):
        nested_path = os.path.join(self.temp_dir.name, "subdir", "model.pt")
        save_state(self.model, nested_path)
        self.assertTrue(os.path.exists(nested_path))


if __name__ == "__main__":
    unittest.main()
