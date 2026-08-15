#!/usr/bin/env python3
"""Control the follower from the leader and optionally show live cameras."""

from __future__ import annotations

import argparse
from pathlib import Path

from camera_preview import CameraProcess, start_camera, stop_camera

from project_utils import (
    command_path,
    create_run,
    print_check,
    project_environment,
    reject_placeholders,
    run_logged,
    snapshot_configs,
    update_latest,
)
from system_config import load_system, run_settings, teleoperation_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="통합 system YAML")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_system(config_path)
    native = teleoperation_config(config)
    command = [command_path("lerobot-teleoperate"), "--config_path=RESOLVED_CONFIG"]
    if args.check:
        print_check("teleoperation", native, command)
        return 0

    reject_placeholders(native)
    settings = run_settings(config, "teleoperation")
    output_root = settings.get("output_root", "data/teleoperation_logs")
    run_dir, artifacts = create_run(output_root, settings.get("run_name", "teleoperation"))
    resolved = snapshot_configs(config_path, native, artifacts)
    print("leader를 움직이면 follower가 따라갑니다. 종료: Ctrl+C")
    camera_name = settings.get("camera_preview")
    preview: CameraProcess | None = None
    if camera_name:
        preview = start_camera(config, str(camera_name), artifacts / "camera_preview.log")
        print(f"저지연 카메라 창: {camera_name}")
    try:
        code = run_logged(
            [command_path("lerobot-teleoperate"), f"--config_path={resolved}"],
            run_dir,
            project_environment(),
        )
    finally:
        stop_camera(preview)
    if code == 0:
        update_latest(output_root, run_dir)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
