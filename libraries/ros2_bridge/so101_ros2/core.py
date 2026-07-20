"""ROS-independent conversions and replay safety checks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
REVOLUTE_JOINT_NAMES = JOINT_NAMES[:-1]


class JointStateValidationError(ValueError):
    """Raised when a ROS joint-state payload cannot safely command the arm."""


class ReplaySafetyError(RuntimeError):
    """Raised and latched when a replay trajectory violates a safety limit."""


def ordered_ros_positions(names: Sequence[str], positions: Sequence[float]) -> tuple[float, ...]:
    """Validate and order a possibly shuffled ROS JointState payload."""

    if len(names) != len(positions):
        raise JointStateValidationError(
            f"joint name/position lengths differ ({len(names)} != {len(positions)})"
        )
    if len(set(names)) != len(names):
        raise JointStateValidationError("joint names contain duplicates")

    by_name = dict(zip(names, positions, strict=True))
    missing = [name for name in JOINT_NAMES if name not in by_name]
    if missing:
        raise JointStateValidationError(f"missing required joints: {', '.join(missing)}")

    ordered = tuple(float(by_name[name]) for name in JOINT_NAMES)
    if not all(math.isfinite(value) for value in ordered):
        raise JointStateValidationError("joint positions must all be finite numbers")
    if not 0.0 <= ordered[-1] <= 1.0:
        raise JointStateValidationError("gripper position must be in the normalized range [0, 1]")
    return ordered


def ros_to_lerobot_action(names: Sequence[str], positions: Sequence[float]) -> dict[str, float]:
    """Convert ROS units (radians, normalized gripper) to LeRobot units."""

    ordered = ordered_ros_positions(names, positions)
    action = {
        f"{name}.pos": math.degrees(ordered[index])
        for index, name in enumerate(REVOLUTE_JOINT_NAMES)
    }
    action["gripper.pos"] = ordered[-1] * 100.0
    return action


def lerobot_to_ros_positions(observation: Mapping[str, object]) -> tuple[float, ...]:
    """Convert LeRobot degrees/percent observations to ROS-standard positions."""

    try:
        degrees = [float(observation[f"{name}.pos"]) for name in REVOLUTE_JOINT_NAMES]
        gripper = float(observation["gripper.pos"]) / 100.0
    except (KeyError, TypeError, ValueError) as exc:
        raise JointStateValidationError(f"invalid LeRobot observation: {exc}") from exc

    result = tuple(math.radians(value) for value in degrees) + (gripper,)
    if not all(math.isfinite(value) for value in result):
        raise JointStateValidationError("LeRobot observation contains a non-finite position")
    return result


@dataclass
class ReplayGuard:
    """Latch replay faults so a bad bag cannot resume motion by itself."""

    max_start_delta_rad: float = math.radians(20.0)
    max_start_delta_gripper: float = 0.25
    max_step_delta_rad: float = math.radians(15.0)
    max_step_delta_gripper: float = 0.25
    _previous: tuple[float, ...] | None = field(default=None, init=False, repr=False)
    _fault: str | None = field(default=None, init=False, repr=False)

    @property
    def fault(self) -> str | None:
        return self._fault

    @property
    def started(self) -> bool:
        return self._previous is not None

    def check(
        self,
        target: Sequence[float],
        current: Sequence[float] | None = None,
    ) -> tuple[float, ...]:
        if self._fault is not None:
            raise ReplaySafetyError(f"replay is latched off: {self._fault}")

        target_tuple = tuple(float(value) for value in target)
        if len(target_tuple) != len(JOINT_NAMES) or not all(
            math.isfinite(value) for value in target_tuple
        ):
            self._trip("target has an invalid shape or a non-finite value")
        if not 0.0 <= target_tuple[-1] <= 1.0:
            self._trip("target gripper position is outside [0, 1]")

        if self._previous is None and current is not None:
            current_tuple = tuple(float(value) for value in current)
            self._check_delta(
                target_tuple,
                current_tuple,
                self.max_start_delta_rad,
                self.max_start_delta_gripper,
                "first bag pose is too far from the arm's current pose",
            )
        elif self._previous is not None:
            self._check_delta(
                target_tuple,
                self._previous,
                self.max_step_delta_rad,
                self.max_step_delta_gripper,
                "adjacent bag poses contain an unsafe jump",
            )

        self._previous = target_tuple
        return target_tuple

    def _check_delta(
        self,
        target: tuple[float, ...],
        reference: tuple[float, ...],
        max_joint_delta: float,
        max_gripper_delta: float,
        reason: str,
    ) -> None:
        if len(reference) != len(JOINT_NAMES) or not all(
            math.isfinite(value) for value in reference
        ):
            self._trip("reference pose has an invalid shape or a non-finite value")
        joint_delta = max(abs(target[index] - reference[index]) for index in range(5))
        gripper_delta = abs(target[-1] - reference[-1])
        if joint_delta > max_joint_delta or gripper_delta > max_gripper_delta:
            self._trip(
                f"{reason} (joint={math.degrees(joint_delta):.1f} deg, "
                f"gripper={gripper_delta:.3f})"
            )

    def _trip(self, reason: str) -> None:
        self._fault = reason
        raise ReplaySafetyError(reason)
