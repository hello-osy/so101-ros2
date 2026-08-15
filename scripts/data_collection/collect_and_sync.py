#!/usr/bin/env python3
"""Collect demonstrations while periodically syncing partial data to the desktop."""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

from project_utils import command_path  # noqa: E402
from system_config import load_system  # noqa: E402


def sync(config: Path) -> int:
    return subprocess.run(
        [
            command_path("python"),
            str(SCRIPTS / "sync_artifacts.py"),
            "push-dataset",
            str(config),
            "--quiet",
        ],
        cwd=ROOT,
        check=False,
    ).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="통합 system YAML")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_system(config_path)
    interval = max(5, int(config["transfer"].get("sync_interval_s", 30)))

    # Fail before moving the robot if the SSH destination is not configured/reachable.
    if sync(config_path) != 0:
        return 1
    child = subprocess.Popen(
        [command_path("python"), str(Path(__file__).with_name("run_collection.py")), str(config_path)],
        cwd=ROOT,
        start_new_session=True,
    )
    try:
        while child.poll() is None:
            time.sleep(interval)
            if child.poll() is None:
                sync(config_path)
    except KeyboardInterrupt:
        # Let run_collection handle SIGINT and finish its dataset metadata.
        try:
            child.send_signal(signal.SIGINT)
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.terminate()
            child.wait()
    final_sync = sync(config_path)
    return child.returncode or final_sync


if __name__ == "__main__":
    raise SystemExit(main())
