#!/usr/bin/env python3
"""Collect demonstrations while periodically syncing partial data to the desktop."""

from __future__ import annotations

import argparse
import subprocess
import sys
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


def collect_and_sync(config_path: Path, interval: int) -> int:
    print("데스크탑 연결 확인 및 기존 데이터셋 동기화 중...", flush=True)
    if sync(config_path) != 0:
        print(
            "오류: 데스크탑 사전 동기화에 실패해 수집을 시작하지 않았습니다. "
            "SSH 연결과 transfer.desktop 설정을 확인하세요.",
            file=sys.stderr,
        )
        return 1

    # Keep the collector in this terminal's foreground process group. Both
    # run_collection and lerobot-record need direct access to stdin for Enter,
    # arrow, and Escape key handling.
    child = subprocess.Popen(
        [command_path("python"), str(Path(__file__).with_name("run_collection.py")), str(config_path)],
        cwd=ROOT,
    )
    try:
        while True:
            try:
                collection_code = child.wait(timeout=interval)
                break
            except subprocess.TimeoutExpired:
                if sync(config_path) != 0:
                    print(
                        "경고: 중간 동기화에 실패했습니다. 수집은 계속하며 종료 후 다시 시도합니다.",
                        file=sys.stderr,
                    )
    except KeyboardInterrupt:
        # SIGINT is delivered to every process in the foreground group. The
        # collector therefore already received it and must only be given time
        # to finalize episode/video metadata here.
        print("\n수집 종료 및 데이터셋 메타데이터 저장을 기다리는 중...", flush=True)
        try:
            collection_code = child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("경고: 수집 프로세스가 30초 안에 끝나지 않아 종료합니다.", file=sys.stderr)
            child.terminate()
            collection_code = child.wait()

    print("최종 데이터셋 동기화 중...", flush=True)
    final_sync_code = sync(config_path)
    if final_sync_code != 0:
        print(
            "경고: 최종 동기화에 실패했습니다. 로컬 데이터셋은 유지되어 있습니다.",
            file=sys.stderr,
        )
    return collection_code or final_sync_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="통합 system YAML")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_system(config_path)
    interval = max(5, int(config["transfer"].get("sync_interval_s", 30)))
    return collect_and_sync(config_path, interval)


if __name__ == "__main__":
    raise SystemExit(main())
