#!/usr/bin/env python3
"""Record keyboard-delimited SO-101 demonstrations as LeRobotDataset v3."""

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
)


def build_native(config: dict, dataset_root: str) -> dict:
    project = config.get("project", {})
    native = dict(config.get("lerobot", {}))
    require_keys(project, "dataset_format", "output_root", context="project")
    require_keys(native, "robot", "teleop", "dataset", context="lerobot")
    if project["dataset_format"] != "lerobot_v3":
        raise ValueError("현재 수집기는 dataset_format: lerobot_v3만 지원합니다.")

    native["robot"] = dict(native["robot"])
    native["teleop"] = dict(native["teleop"])
    for device in (native["robot"], native["teleop"]):
        if device.get("calibration_dir"):
            device["calibration_dir"] = absolute_path(device["calibration_dir"])
    native["dataset"] = dict(native["dataset"])
    native["dataset"]["root"] = dataset_root
    # LeRobot 0.6.1 records v3 and calls finalize() even on Ctrl+C.
    native["dataset"]["push_to_hub"] = bool(native["dataset"].get("push_to_hub", False))
    return native


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--check", action="store_true", help="hardware 없이 config만 검사")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    project = config.get("project", {})
    provisional_root = absolute_path(Path(project.get("output_root", "data/collected_datasets")) / "CHECK_RUN" / "dataset")
    native = build_native(config, provisional_root)
    check_command = [command_path("lerobot-record"), "--config_path=RESOLVED_CONFIG"]
    if args.check:
        print_check("collection", native, check_command)
        return 0

    reject_placeholders(native)
    run_name = str(project.get("run_name", "pick_place"))
    run_dir, artifacts = create_run(project["output_root"], run_name)
    dataset_root = run_dir / "dataset"
    native = build_native(config, str(dataset_root))
    resolved = snapshot_configs(config_path, native, artifacts)

    print("\n[데이터 수집 키]")
    print("  Enter       : 첫 episode 시작")
    print("  오른쪽 화살표: 현재 episode 종료 / reset 종료 후 다음 episode 시작")
    print("  왼쪽 화살표 : 현재 episode 폐기 후 다시 기록")
    print("  Esc          : 전체 수집 종료 (저장 중에는 전원을 끄지 마세요)\n")
    input("팔과 작업물을 준비한 뒤 Enter를 누르세요: ")

    code = run_logged(
        [command_path("lerobot-record"), f"--config_path={resolved}"],
        run_dir,
        project_environment(),
    )
    if code == 0 and (dataset_root / "meta" / "info.json").exists():
        update_latest(project["output_root"], run_dir)
        print(f"dataset: {dataset_root}")
    elif code == 0:
        print("오류: 명령은 끝났지만 LeRobotDataset meta/info.json이 없습니다.", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
