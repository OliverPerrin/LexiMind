import os
import tempfile
import unittest

import yaml

from src.utils.config import Config, load_yaml


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.yaml_path = os.path.join(self.temp_dir.name, "config.yaml")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_yaml_valid(self):
        data = {"key": "value", "nested": {"k": 1}}
        with open(self.yaml_path, "w") as f:
            yaml.dump(data, f)

        config = load_yaml(self.yaml_path)
        self.assertIsInstance(config, Config)
        self.assertEqual(config.data["key"], "value")
        self.assertEqual(config.data["nested"]["k"], 1)

    def test_load_yaml_invalid_structure(self):
        # List at root instead of dict
        data = ["item1", "item2"]
        with open(self.yaml_path, "w") as f:
            yaml.dump(data, f)

        with self.assertRaises(ValueError):
            load_yaml(self.yaml_path)

    def test_load_yaml_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_yaml("non_existent_file.yaml")


if __name__ == "__main__":
    unittest.main()
