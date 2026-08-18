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
        from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig
        from system_config import (
            calibration_configs,
            collection_config,
            load_system,
            teleoperation_config,
        )

        system = load_system(ROOT / "config/system.yaml")
        record = collection_config(system, "/tmp/so101-native-record-check")
        parsed_record = self._parse(RecordConfig, record)
        self.assertEqual(parsed_record.dataset.repo_id, "local/so101_pick_place")
        self.assertEqual(parsed_record.robot.max_relative_target, 5.0)

        calibration = calibration_configs(system)
        for _, native in calibration:
            self.assertIsNotNone(self._parse(CalibrateConfig, native).device)

        teleoperation = self._parse(TeleoperateConfig, teleoperation_config(system))
        # SO100/SO101 share one registered dataclass, so the parsed alias reports
        # the first registered type even though the SO101 YAML alias is accepted.
        self.assertEqual(teleoperation.teleop.id, "osy_so101_leader")
        self.assertEqual(teleoperation.robot.cameras, {})
        self.assertFalse(teleoperation.display_data)
        self.assertEqual(teleoperation.fps, 60)

    def test_cached_smolvla_training_and_rollout_configs(self):
        local_model = ROOT / "data/models/smolvla_base"
        snapshots = sorted(
            (ROOT / "data/models/huggingface/hub/models--lerobot--smolvla_base/snapshots").glob("*")
        )
        snapshots = [path for path in snapshots if (path / "config.json").exists()]
        if (local_model / "config.json").exists():
            local_policy = str(local_model)
        elif snapshots:
            local_policy = str(snapshots[-1])
        else:
            self.skipTest("SmolVLA config is not cached yet")

        from lerobot.configs import parser
        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.rollout.configs import RolloutConfig
        from system_config import inference_config, load_system, training_config

        system = load_system(ROOT / "config/system.yaml")
        system["model"]["base"]["path"] = local_policy
        native_training = training_config(system, "/tmp/so101-native-training-check")

        inference = inference_config(system)
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
                    self.assertEqual(parsed.inference.type, "rtc")
                    self.assertEqual(parsed.inference.queue_threshold, 30)
                    self.assertEqual(parsed.inference.rtc.execution_horizon, 10)
                    self.assertEqual(parsed.policy.num_steps, 5)
                    self.assertEqual(parsed.fps, 30)
                    self.assertEqual(parsed.interpolation_multiplier, 1)
                    self.assertFalse(parsed.use_torch_compile)
                    self.assertEqual(parsed.compile_warmup_inferences, 0)
                    self.assertEqual(parsed.torch_compile_mode, "default")
                    self.assertEqual(parsed.robot.max_relative_target, 5.0)


if __name__ == "__main__":
    unittest.main()
