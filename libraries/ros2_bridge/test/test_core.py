import math
import unittest

from so101_ros2.core import (
    JOINT_NAMES,
    JointStateValidationError,
    ReplayGuard,
    ReplaySafetyError,
    lerobot_to_ros_positions,
    ros_to_lerobot_action,
)


class ConversionTest(unittest.TestCase):
    def test_ros_to_lerobot_reorders_and_converts_units(self):
        names = ("gripper", *reversed(JOINT_NAMES[:-1]))
        positions = (0.42, *[math.radians(value) for value in (50, 40, 30, 20, 10)])

        action = ros_to_lerobot_action(names, positions)

        self.assertAlmostEqual(action["shoulder_pan.pos"], 10.0)
        self.assertAlmostEqual(action["wrist_roll.pos"], 50.0)
        self.assertAlmostEqual(action["gripper.pos"], 42.0)

    def test_lerobot_to_ros_converts_units(self):
        observation = {f"{name}.pos": 90.0 for name in JOINT_NAMES[:-1]}
        observation["gripper.pos"] = 25.0

        positions = lerobot_to_ros_positions(observation)

        self.assertEqual(len(positions), 6)
        self.assertTrue(all(math.isclose(value, math.pi / 2) for value in positions[:-1]))
        self.assertAlmostEqual(positions[-1], 0.25)

    def test_missing_joint_is_rejected(self):
        with self.assertRaisesRegex(JointStateValidationError, "missing required joints"):
            ros_to_lerobot_action(JOINT_NAMES[:-1], [0.0] * 5)

    def test_duplicate_joint_is_rejected(self):
        with self.assertRaisesRegex(JointStateValidationError, "duplicates"):
            ros_to_lerobot_action((*JOINT_NAMES[:-1], JOINT_NAMES[-2]), [0.0] * 6)

    def test_non_finite_value_is_rejected(self):
        with self.assertRaisesRegex(JointStateValidationError, "finite"):
            ros_to_lerobot_action(JOINT_NAMES, [0.0, 0.0, math.nan, 0.0, 0.0, 0.5])

    def test_gripper_out_of_range_is_rejected(self):
        with self.assertRaisesRegex(JointStateValidationError, "normalized range"):
            ros_to_lerobot_action(JOINT_NAMES, [0.0] * 5 + [1.1])


class ReplayGuardTest(unittest.TestCase):
    def test_nearby_start_and_small_step_are_accepted(self):
        guard = ReplayGuard()
        first = (0.1, 0.0, 0.0, 0.0, 0.0, 0.5)
        second = (0.2, 0.0, 0.0, 0.0, 0.0, 0.55)

        self.assertEqual(guard.check(first, (0.0,) * 5 + (0.5,)), first)
        self.assertEqual(guard.check(second), second)
        self.assertTrue(guard.started)

    def test_far_start_latches_fault(self):
        guard = ReplayGuard(max_start_delta_rad=0.1)

        with self.assertRaisesRegex(ReplaySafetyError, "first bag pose"):
            guard.check((0.2, 0.0, 0.0, 0.0, 0.0, 0.5), (0.0,) * 5 + (0.5,))
        with self.assertRaisesRegex(ReplaySafetyError, "latched off"):
            guard.check((0.0,) * 5 + (0.5,))

    def test_large_inter_frame_jump_latches_fault(self):
        guard = ReplayGuard(max_step_delta_rad=0.1)
        guard.check((0.0,) * 5 + (0.5,))

        with self.assertRaisesRegex(ReplaySafetyError, "adjacent bag poses"):
            guard.check((0.0, 0.0, 0.2, 0.0, 0.0, 0.5))

    def test_dry_run_can_start_without_current_hardware_pose(self):
        guard = ReplayGuard()
        target = (0.0,) * 5 + (0.5,)
        self.assertEqual(guard.check(target), target)


if __name__ == "__main__":
    unittest.main()
