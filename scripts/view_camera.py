#!/usr/bin/env python3
"""Show one camera from the unified system YAML without opening either arm."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from camera_preview import start_camera, stop_camera
from system_config import load_system, run_settings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="통합 system YAML")
    args = parser.parse_args()

    config = load_system(args.config)
    settings = run_settings(config, "camera_viewer")
    camera_names = settings.get("cameras", [settings.get("camera", "wrist")])
    if not isinstance(camera_names, list) or not camera_names:
        raise ValueError("runs.camera_viewer.cameras는 카메라 이름 목록이어야 합니다.")
    previews = []
    try:
        for name in camera_names:
            previews.append(
                start_camera(config, str(name), Path(f"data/camera_preview_{name}.log").resolve())
            )
    except Exception:
        for preview in previews:
            stop_camera(preview)
        raise
    print(f"카메라: {', '.join(map(str, camera_names))} (창 또는 Ctrl+C로 종료)")
    try:
        while any(process.poll() is None for process, _ in previews):
            time.sleep(0.2)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        for preview in previews:
            stop_camera(preview)


if __name__ == "__main__":
    raise SystemExit(main())
