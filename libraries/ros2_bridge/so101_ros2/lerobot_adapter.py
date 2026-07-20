"""Small adapter around the LeRobot SOFollower hardware API."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .core import JOINT_NAMES, lerobot_to_ros_positions, ros_to_lerobot_action

logger = logging.getLogger(__name__)


class LeRobotAdapter:
    """Own the serial port and optional cameras for one follower arm."""

    def __init__(
        self,
        *,
        port: str,
        robot_id: str,
        calibration_dir: str,
        camera_configs_json: str,
        max_relative_target_deg: float,
    ) -> None:
        if not port:
            raise ValueError("the 'port' parameter is required when hardware is enabled")
        if max_relative_target_deg <= 0.0:
            raise ValueError("max_relative_target_deg must be positive")

        try:
            from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
            from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
            from lerobot.robots.so_follower.so_follower import SOFollower
        except ImportError as exc:
            raise RuntimeError(
                "LeRobot is not importable. Source scripts/setup_env.bash and rerun, "
                "or run scripts/repair_environment.bash after moving the repository."
            ) from exc

        camera_configs = self._make_camera_configs(camera_configs_json, OpenCVCameraConfig)
        config = SOFollowerRobotConfig(
            port=port,
            id=robot_id,
            calibration_dir=Path(calibration_dir).expanduser() if calibration_dir else None,
            cameras=camera_configs,
            use_degrees=True,
            max_relative_target=max_relative_target_deg,
            disable_torque_on_disconnect=True,
        )
        self._robot = SOFollower(config)
        self._connected = False

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(self._robot.cameras)

    def connect(self, *, hand_guided: bool) -> None:
        try:
            # Calibration is deliberately non-interactive inside a ROS node.
            self._robot.connect(calibrate=False)
            self._connected = True
            # SOFollower.configure() ends by enabling torque. Disable it again
            # immediately so we can preload the current pose as the goal before
            # deliberately enabling motion.
            self._robot.bus.disable_torque()
            if not self._robot.is_calibrated:
                raise RuntimeError(
                    "no matching calibration was loaded. Run lerobot-calibrate with the same "
                    "robot id first"
                )
            if not hand_guided:
                observation = self._robot.get_observation()
                current_action = {
                    key: float(value)
                    for key, value in observation.items()
                    if key.endswith(".pos")
                }
                self._robot.send_action(current_action)
                self._robot.bus.enable_torque()
        except Exception:
            self.close()
            raise

    def observation(self) -> dict[str, Any]:
        return self._robot.get_observation()

    def current_ros_positions(self) -> tuple[float, ...]:
        return lerobot_to_ros_positions(self.observation())

    def send_ros_positions(self, positions: tuple[float, ...]) -> None:
        action = ros_to_lerobot_action(JOINT_NAMES, positions)
        self._robot.send_action(action)

    def close(self) -> None:
        if self._connected and self._robot.is_connected:
            try:
                self._robot.disconnect()
            finally:
                self._connected = False
            return

        # Also clean up a partial connect (for example, serial opened but a
        # camera failed). SOFollower.disconnect() requires every device to be
        # connected, so partial cleanup is performed device by device.
        try:
            if self._robot.bus.is_connected:
                try:
                    self._robot.bus.disconnect(disable_torque=True)
                except Exception as exc:
                    logger.warning("motor bus cleanup failed; forcing the port closed: %s", exc)
                    self._robot.bus.port_handler.closePort()
            for camera in self._robot.cameras.values():
                if camera.is_connected:
                    try:
                        camera.disconnect()
                    except Exception as exc:
                        logger.warning("camera cleanup failed: %s", exc)
        finally:
            self._connected = False

    @staticmethod
    def _make_camera_configs(camera_configs_json: str, config_class: type) -> dict[str, Any]:
        try:
            raw_configs = json.loads(camera_configs_json or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"camera_configs_json is not valid JSON: {exc}") from exc
        if not isinstance(raw_configs, dict):
            raise ValueError("camera_configs_json must be a JSON object keyed by camera name")

        result: dict[str, Any] = {}
        for name, raw in raw_configs.items():
            if not isinstance(name, str) or not name or not isinstance(raw, dict):
                raise ValueError("each camera entry must have a non-empty name and an object value")
            required = {"device", "fps", "width", "height"}
            missing = sorted(required - raw.keys())
            if missing:
                raise ValueError(f"camera '{name}' is missing: {', '.join(missing)}")

            device = raw["device"]
            if isinstance(device, str):
                device = int(device) if device.isdecimal() else Path(device).expanduser()
            kwargs = {
                "index_or_path": device,
                "fps": int(raw["fps"]),
                "width": int(raw["width"]),
                "height": int(raw["height"]),
                "rotation": int(raw.get("rotation", 0)),
                "warmup_s": int(raw.get("warmup_s", 1)),
                "fourcc": raw.get("fourcc"),
            }
            result[name] = config_class(**kwargs)
        return result
