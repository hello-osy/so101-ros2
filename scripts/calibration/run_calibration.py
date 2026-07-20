#!/usr/bin/env python3
"""Calibrate the SO-101 follower, leader, or both from one YAML file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_utils import (  # noqa: E402
    absolute_path,
    command_path,
    create_run,
    load_yaml,
    print_check,
    project_environment,
    reject_placeholders,
    require_keys,
    run_logged,
    snapshot_configs,
    update_latest,
    write_yaml,
)


def build_configs(config: dict) -> list[tuple[str, dict]]:
    project = config.get("project", {})
    devices = config.get("devices", {})
    target = project.get("target", "both")
    if target not in {"follower", "leader", "both"}:
        raise ValueError("project.target은 follower, leader, both 중 하나여야 합니다.")
    require_keys(devices, "follower", "leader", context="devices")

    result: list[tuple[str, dict]] = []
    if target in {"follower", "both"}:
        robot = dict(devices["follower"])
        robot["calibration_dir"] = absolute_path(robot["calibration_dir"])
        result.append(("follower", {"robot": robot}))
    if target in {"leader", "both"}:
        teleop = dict(devices["leader"])
        teleop["calibration_dir"] = absolute_path(teleop["calibration_dir"])
        result.append(("leader", {"teleop": teleop}))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    native_configs = build_configs(config)
    command = [command_path("lerobot-calibrate"), "--config_path=RESOLVED_CONFIG"]
    if args.check:
        for name, native in native_configs:
            print_check(f"calibration:{name}", native, command)
        return 0

    reject_placeholders(native_configs)
    project = config.get("project", {})
    run_dir, artifacts = create_run(
        project.get("output_root", "data/calibration_runs"),
        project.get("run_name", "calibration"),
    )
    # Keep the exact user YAML once at the top level.
    snapshot_configs(config_path, {name: native for name, native in native_configs}, artifacts)

    for name, native in native_configs:
        child = run_dir / name
        (child / "artifacts").mkdir(parents=True)
        resolved = child / "artifacts" / "resolved_lerobot.yaml"
        write_yaml(resolved, native)
        print(f"\n{name} calibration을 시작합니다. 화면 안내에 따라 천천히 전체 관절을 움직이세요.")
        code = run_logged(
            [command_path("lerobot-calibrate"), f"--config_path={resolved}"],
            child,
            project_environment(),
        )
        if code != 0:
            return code

    update_latest(project.get("output_root", "data/calibration_runs"), run_dir)
    print(f"calibration run: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
