from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts" / "calibration")]

from calibrate_device import install_write_retries  # noqa: E402
from run_calibration import get_communication_retries, validate_hardware  # noqa: E402
from system_config import calibration_configs  # noqa: E402


class CalibrationLauncherTest(unittest.TestCase):
    def test_write_retry_wrapper_applies_minimum_without_lowering_explicit_retry(self):
        class FakeBus:
            def __init__(self):
                self.retries = []

            def write(self, *args, **kwargs):
                self.retries.append(kwargs["num_retry"])

        bus = FakeBus()
        install_write_retries(bus, 3)
        bus.write("Lock", "gripper", 1)
        bus.write("Lock", "gripper", 1, num_retry=5)
        self.assertEqual(bus.retries, [3, 5])

    def test_follower_only_does_not_require_leader_config(self):
        configs = calibration_configs(
            {
                "runs": {"calibration": {"target": "follower"}},
                "devices": {
                    "follower": {
                        "type": "so101_follower",
                        "port": "/dev/null",
                        "id": "test_follower",
                        "calibration_dir": "data/calibration/robots",
                    },
                    "leader": {},
                    "cameras": {},
                },
            }
        )
        self.assertEqual([name for name, _ in configs], ["follower"])

    def test_duplicate_ports_are_rejected(self):
        configs = [
            ("follower", {"robot": {"port": "/dev/null", "id": "follower"}}),
            ("leader", {"teleop": {"port": "/dev/null", "id": "leader"}}),
        ]
        with self.assertRaisesRegex(RuntimeError, "같은 포트"):
            validate_hardware(configs)

    def test_missing_port_is_rejected(self):
        configs = [
            ("follower", {"robot": {"port": "/dev/so101-port-that-does-not-exist", "id": "follower"}})
        ]
        with self.assertRaisesRegex(RuntimeError, "포트가 없습니다"):
            validate_hardware(configs)

    def test_invalid_retry_count_is_rejected_by_check(self):
        with self.assertRaisesRegex(ValueError, "0~10"):
            get_communication_retries({"communication_retries": 11})


if __name__ == "__main__":
    unittest.main()
