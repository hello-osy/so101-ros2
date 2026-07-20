"""ROS 2 node for recording, replaying, or bridging an SO-101 follower."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState

from .core import (
    JOINT_NAMES,
    ReplayGuard,
    lerobot_to_ros_positions,
    ordered_ros_positions,
)
from .lerobot_adapter import LeRobotAdapter


class SO101HardwareNode(Node):
    """Single-owner hardware node; never open the follower port from two processes."""

    def __init__(self) -> None:
        super().__init__("so101_hardware")
        self._declare_parameters()
        self._mode = str(self.get_parameter("mode").value)
        if self._mode not in {"record", "replay", "bridge"}:
            raise ValueError("mode must be one of: record, replay, bridge")

        self._dry_run = bool(self.get_parameter("dry_run").value)
        if self._mode == "record" and self._dry_run:
            raise ValueError("record mode needs hardware, so dry_run cannot be true")
        self._validate_positive_parameters()

        self._guard = ReplayGuard(
            max_start_delta_rad=float(self.get_parameter("max_start_delta_rad").value),
            max_start_delta_gripper=float(self.get_parameter("max_start_delta_gripper").value),
            max_step_delta_rad=float(self.get_parameter("max_step_delta_rad").value),
            max_step_delta_gripper=float(self.get_parameter("max_step_delta_gripper").value),
        )
        self._adapter: LeRobotAdapter | None = None
        self._state_publisher = None
        self._camera_publishers: dict[str, Any] = {}
        self._command_count = 0
        self._command_error_count = 0
        self._last_command_error = ""

        if not self._dry_run:
            self._adapter = LeRobotAdapter(
                port=str(self.get_parameter("port").value),
                robot_id=str(self.get_parameter("robot_id").value),
                calibration_dir=str(self.get_parameter("calibration_dir").value),
                camera_configs_json=str(self.get_parameter("camera_configs_json").value),
                max_relative_target_deg=float(
                    self.get_parameter("max_relative_target_deg").value
                ),
            )
            self._adapter.connect(hand_guided=self._mode == "record")

        if self._mode in {"record", "bridge"}:
            state_topic = str(self.get_parameter("state_topic").value)
            self._state_publisher = self.create_publisher(JointState, state_topic, 10)
            if self._adapter is not None:
                topic_prefix = str(self.get_parameter("camera_topic_prefix").value).rstrip("/")
                for camera_name in self._adapter.camera_names:
                    topic = f"{topic_prefix}/{camera_name}/image_raw"
                    self._camera_publishers[camera_name] = self.create_publisher(
                        Image, topic, qos_profile_sensor_data
                    )
            rate_hz = float(self.get_parameter("publish_rate_hz").value)
            self._timer = self.create_timer(1.0 / rate_hz, self._publish_observation)

        if self._mode in {"replay", "bridge"}:
            command_topic = str(self.get_parameter("command_topic").value)
            self._command_subscription = self.create_subscription(
                JointState, command_topic, self._handle_command, 10
            )

        if self._mode == "record":
            self.get_logger().warning(
                "RECORD mode: follower torque is disabled. "
                "Physically support the arm before moving it."
            )
        elif self._dry_run:
            self.get_logger().warning(
                "DRY RUN: replay messages are checked but the arm will not move."
            )
        else:
            self.get_logger().warning(
                "MOTION ENABLED: keep an emergency stop/power switch within reach."
            )

    def _declare_parameters(self) -> None:
        self.declare_parameter("mode", "record")
        self.declare_parameter("dry_run", True)
        self.declare_parameter("port", "")
        self.declare_parameter("robot_id", "osy_so101_follower")
        self.declare_parameter("calibration_dir", "")
        self.declare_parameter("camera_configs_json", "{}")
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("state_topic", "/so101/record/joint_states")
        self.declare_parameter("command_topic", "/so101/record/joint_states")
        self.declare_parameter("camera_topic_prefix", "/so101/record/camera")
        self.declare_parameter("max_relative_target_deg", 5.0)
        self.declare_parameter("max_start_delta_rad", math.radians(20.0))
        self.declare_parameter("max_start_delta_gripper", 0.25)
        self.declare_parameter("max_step_delta_rad", math.radians(15.0))
        self.declare_parameter("max_step_delta_gripper", 0.25)

    def _validate_positive_parameters(self) -> None:
        names = (
            "publish_rate_hz",
            "max_relative_target_deg",
            "max_start_delta_rad",
            "max_start_delta_gripper",
            "max_step_delta_rad",
            "max_step_delta_gripper",
        )
        invalid = [name for name in names if float(self.get_parameter(name).value) <= 0.0]
        if invalid:
            raise ValueError(f"parameters must be positive: {', '.join(invalid)}")

    def _publish_observation(self) -> None:
        if self._adapter is None or self._state_publisher is None:
            return
        try:
            observation = self._adapter.observation()
            stamp = self.get_clock().now().to_msg()
            joint_message = JointState()
            joint_message.header.stamp = stamp
            joint_message.header.frame_id = "so101_base"
            joint_message.name = list(JOINT_NAMES)
            joint_message.position = list(lerobot_to_ros_positions(observation))
            self._state_publisher.publish(joint_message)

            for camera_name, publisher in self._camera_publishers.items():
                publisher.publish(self._image_message(observation[camera_name], camera_name, stamp))
        except Exception as exc:  # Hardware errors must be visible without killing cleanup.
            self.get_logger().error(f"failed to read follower observation: {exc}")

    def _handle_command(self, message: JointState) -> None:
        try:
            target = ordered_ros_positions(message.name, message.position)
            current = None
            if not self._guard.started and self._adapter is not None:
                current = self._adapter.current_ros_positions()
            checked_target = self._guard.check(target, current)

            if self._adapter is not None:
                self._adapter.send_ros_positions(checked_target)
            self._command_count += 1
            self._command_error_count = 0
            self._last_command_error = ""
            if self._dry_run and (self._command_count == 1 or self._command_count % 100 == 0):
                self.get_logger().info(f"dry-run validated {self._command_count} command(s)")
        except Exception as exc:
            # ReplaySafetyError remains latched in ReplayGuard until this node is restarted.
            error = str(exc)
            self._command_error_count += 1
            if error != self._last_command_error or self._command_error_count % 100 == 0:
                self.get_logger().error(f"command rejected: {error}")
            self._last_command_error = error

    @staticmethod
    def _image_message(frame: object, camera_name: str, stamp: Any) -> Image:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"camera '{camera_name}' returned shape {array.shape}, expected HxWx3")
        if array.dtype != np.uint8:
            array = array.astype(np.uint8)
        array = np.ascontiguousarray(array)

        message = Image()
        message.header.stamp = stamp
        message.header.frame_id = f"so101_{camera_name}_optical_frame"
        message.height = int(array.shape[0])
        message.width = int(array.shape[1])
        message.encoding = "rgb8"
        message.is_bigendian = 0
        message.step = message.width * 3
        message.data = array.tobytes()
        return message

    def close(self) -> None:
        if self._adapter is not None:
            self._adapter.close()
            self._adapter = None


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: SO101HardwareNode | None = None
    try:
        node = SO101HardwareNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node is not None:
                node.close()
        finally:
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
