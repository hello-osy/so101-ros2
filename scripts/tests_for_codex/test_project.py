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

from project_utils import deep_merge, load_yaml, reject_placeholders  # noqa: E402
from system_config import (  # noqa: E402
    benchmark_config,
    collection_config,
    inference_config,
    load_system,
    training_config,
)
from benchmark_child import current_rss_bytes, memory_available_bytes  # noqa: E402
from camera_preview import camera_command  # noqa: E402


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
        config = load_system(ROOT / "config/system.yaml")
        collection_native = collection_config(config, "/tmp/test_dataset")
        self.assertEqual(collection_native["dataset"]["root"], "/tmp/test_dataset")
        self.assertEqual(collection_native["robot"]["type"], "so101_follower")
        self.assertEqual(
            collection_native["dataset"]["single_task"],
            "Pick up the blue bottle cap inside the yellow square and place it inside the green square.",
        )

        training_native = training_config(config, "/tmp/test_training")
        self.assertEqual(training_native["peft"]["method_type"], "LORA")
        self.assertEqual(training_native["policy"]["path"], str(ROOT / "data/models/smolvla_base"))

        desktop_training = training_config(config, "/tmp/test_desktop_training", "training_desktop")
        self.assertEqual(desktop_training["batch_size"], 8)
        self.assertEqual(desktop_training["peft"]["r"], 64)
        self.assertEqual(desktop_training["dataset"]["eval_split"], 0.1)
        self.assertTrue(desktop_training["policy"]["freeze_vision_encoder"])

        inference_native = inference_config(config)
        self.assertEqual(inference_native["inference"]["type"], "sync")
        self.assertGreater(inference_native["robot"]["max_relative_target"], 0)
        self.assertEqual(inference_native["task"], collection_native["dataset"]["single_task"])

        profile = benchmark_config(config)
        self.assertEqual(profile["benchmark"]["device"], "cuda")

    def test_system_yaml_is_the_only_user_config(self):
        self.assertTrue((ROOT / "config/system.yaml").is_file())
        self.assertEqual(list((ROOT / "scripts/configs").rglob("*.yaml")), [])

    def test_camera_preview_uses_configured_low_latency_mode(self):
        config = load_system(ROOT / "config/system.yaml")
        wrist = camera_command(config, "wrist")
        self.assertIn("/dev/video0", wrist)
        self.assertIn("nobuffer", wrist)
        self.assertIn("mjpeg", wrist)
        self.assertIn("640x480", wrist)
        front = camera_command(config, "front")
        self.assertIn("/dev/video2", front)
        self.assertIn("yuyv422", front)
        self.assertIn("640x480", front)

    def test_yaml_loader_rejects_non_mapping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.yaml"
            path.write_text("- item\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_yaml(path)

    def test_benchmark_memory_probes_are_non_negative(self):
        self.assertGreater(current_rss_bytes(), 0)
        available = memory_available_bytes()
        self.assertTrue(available is None or available > 0)


if __name__ == "__main__":
    unittest.main()
