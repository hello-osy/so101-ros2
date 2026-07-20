from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "scripts"),
    str(ROOT / "scripts" / "calibration"),
    str(ROOT / "scripts" / "data_collection"),
    str(ROOT / "scripts" / "inference"),
    str(ROOT / "scripts" / "models"),
    str(ROOT / "scripts" / "training"),
]


@unittest.skipUnless(importlib.util.find_spec("lerobot") and importlib.util.find_spec("datasets"), "LeRobot extras not installed")
class LeRobotNativeConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import draccus
        from lerobot.scripts import lerobot_rollout, lerobot_train  # noqa: F401

        cls.draccus = draccus

    def _parse(self, config_class, data):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as stream:
            yaml.safe_dump(data, stream, sort_keys=False)
            stream.flush()
            with self.draccus.config_type("yaml"):
                return self.draccus.parse(config_class, stream.name, args=[])

    def test_record_and_calibration_native_dataclasses(self):
        from lerobot.scripts.lerobot_calibrate import CalibrateConfig
        from lerobot.scripts.lerobot_record import RecordConfig
        from project_utils import load_yaml
        from run_calibration import build_configs
        from run_collection import build_native

        record = build_native(
            load_yaml(ROOT / "scripts/configs/data_collection.yaml"),
            "/tmp/so101-native-record-check",
        )
        parsed_record = self._parse(RecordConfig, record)
        self.assertEqual(parsed_record.dataset.repo_id, "local/so101_pick_place")
        self.assertEqual(parsed_record.robot.max_relative_target, 5.0)

        calibration = build_configs(load_yaml(ROOT / "scripts/configs/calibration.yaml"))
        for _, native in calibration:
            self.assertIsNotNone(self._parse(CalibrateConfig, native).device)

    def test_cached_smolvla_training_and_rollout_configs(self):
        snapshots = sorted(
            (ROOT / "data/models/huggingface/hub/models--lerobot--smolvla_base/snapshots").glob("*")
        )
        snapshots = [path for path in snapshots if (path / "config.json").exists()]
        if not snapshots:
            self.skipTest("SmolVLA config is not cached yet")
        local_policy = str(snapshots[-1])

        from lerobot.configs import parser
        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.rollout.configs import RolloutConfig
        from model_registry import load_model_profile
        from project_utils import load_yaml
        from run_inference import build_native as build_inference
        from run_training import build_native as build_training

        training = load_yaml(ROOT / "scripts/configs/training.yaml")
        model = load_model_profile(ROOT / training["model_config"])
        model["policy"]["path"] = local_policy
        native_training = build_training(training, model, "/tmp/so101-native-training-check")

        inference = build_inference(load_yaml(ROOT / "scripts/configs/inference.yaml"))
        inference["policy"]["path"] = local_policy

        for config_class, native, should_validate in (
            (TrainPipelineConfig, native_training, True),
            (RolloutConfig, inference, False),
        ):
            with tempfile.NamedTemporaryFile("w", suffix=".yaml") as stream:
                yaml.safe_dump(native, stream, sort_keys=False)
                stream.flush()
                parser._config_path_args.clear()
                parser._config_yaml_overrides.clear()
                cleaned = parser.extract_path_fields_from_config(stream.name, ["policy"])
                with self.draccus.config_type("yaml"):
                    parsed = self.draccus.parse(config_class, cleaned, args=[])
                if should_validate:
                    parsed.validate()
                    self.assertEqual(parsed.peft.method_type, "LORA")
                    self.assertIsNone(parsed.policy.input_features)
                    self.assertIsNone(parsed.policy.output_features)
                else:
                    self.assertEqual(parsed.inference.type, "sync")
                    self.assertEqual(parsed.robot.max_relative_target, 5.0)


if __name__ == "__main__":
    unittest.main()
