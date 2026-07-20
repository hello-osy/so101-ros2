from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "data_collection"))
sys.path.insert(0, str(ROOT / "scripts" / "training"))
sys.path.insert(0, str(ROOT / "scripts" / "inference"))
sys.path.insert(0, str(ROOT / "scripts" / "models"))

from project_utils import deep_merge, load_yaml, reject_placeholders  # noqa: E402
from run_collection import build_native as build_collection  # noqa: E402
from run_inference import build_native as build_inference  # noqa: E402
from run_training import build_native as build_training  # noqa: E402
from run_benchmark import resolve_config as resolve_profile  # noqa: E402


class ProjectConfigTest(unittest.TestCase):
    def test_deep_merge_keeps_nested_defaults(self):
        self.assertEqual(
            deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}}),
            {"a": {"x": 1, "y": 3}},
        )

    def test_placeholder_detection(self):
        with self.assertRaises(ValueError):
            reject_placeholders({"port": "/dev/CHANGE_ME"})
        with self.assertRaises(ValueError):
            reject_placeholders([("follower", {"port": "CHANGE_ME"})])

    def test_all_configs_resolve_without_hardware(self):
        configs = ROOT / "scripts" / "configs"
        collection = load_yaml(configs / "data_collection.yaml")
        collection_native = build_collection(collection, "/tmp/test_dataset")
        self.assertEqual(collection_native["dataset"]["root"], "/tmp/test_dataset")
        self.assertEqual(collection_native["robot"]["type"], "so101_follower")

        training = load_yaml(configs / "training.yaml")
        model = load_yaml(ROOT / training["model_config"])
        training_native = build_training(training, model, "/tmp/test_training")
        self.assertEqual(training_native["peft"]["method_type"], "LORA")
        self.assertEqual(training_native["policy"]["path"], "lerobot/smolvla_base")

        inference_native = build_inference(load_yaml(configs / "inference.yaml"))
        self.assertEqual(inference_native["inference"]["type"], "sync")
        self.assertGreater(inference_native["robot"]["max_relative_target"], 0)

        profile = resolve_profile(load_yaml(configs / "profiling.yaml"))
        self.assertEqual(profile["benchmark"]["device"], "cuda")

    def test_yaml_loader_rejects_non_mapping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.yaml"
            path.write_text("- item\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_yaml(path)


if __name__ == "__main__":
    unittest.main()
