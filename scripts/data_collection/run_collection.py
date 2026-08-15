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
    print_check,
    project_environment,
    reject_placeholders,
    run_logged,
    snapshot_configs,
    update_latest,
)
from system_config import collection_config, load_system, run_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="통합 system YAML")
    parser.add_argument("--check", action="store_true", help="hardware 없이 config만 검사")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_system(config_path)
    dataset = config["dataset"]
    settings = run_settings(config, "collection")
    output_root = dataset.get("storage_root", "data/collected_datasets")
    provisional_root = absolute_path(Path(output_root) / "CHECK_RUN" / "dataset")
    native = collection_config(config, provisional_root)
    check_command = [command_path("lerobot-record"), "--config_path=RESOLVED_CONFIG"]
    if args.check:
        print_check("collection", native, check_command)
        return 0

    reject_placeholders(native)
    run_name = str(settings.get("run_name", "pick_place"))
    run_dir, artifacts = create_run(output_root, run_name)
    dataset_root = run_dir / "dataset"
    native = collection_config(config, str(dataset_root))
    resolved = snapshot_configs(config_path, native, artifacts)

    print("\n[데이터 수집 키]")
    print("  Enter       : 첫 episode 시작")
    print("  오른쪽 화살표: 현재 episode 종료 / reset 종료 후 다음 episode 시작")
    print("  왼쪽 화살표 : 현재 episode 폐기 후 다시 기록")
    print("  Esc          : 전체 수집 종료 (저장 중에는 전원을 끄지 마세요)\n")
    print(f"  목표 episode : {int(dataset.get('num_episodes', 1))}개")
    print("  카메라       : Rerun 창에서 wrist/front 실시간 표시\n")
    input("팔과 작업물을 준비한 뒤 Enter를 누르세요: ")

    env = project_environment()
    if not bool(settings.get("show_clamp_warnings", False)):
        env["SO101_SUPPRESS_CLAMP_WARNINGS"] = "1"
    code = run_logged(
        [command_path("lerobot-record"), f"--config_path={resolved}"],
        run_dir,
        env,
    )
    if code == 0 and (dataset_root / "meta" / "info.json").exists():
        update_latest(output_root, run_dir)
        print(f"dataset: {dataset_root}")
    elif code == 0:
        print("오류: 명령은 끝났지만 LeRobotDataset meta/info.json이 없습니다.", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
