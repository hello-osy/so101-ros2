#!/usr/bin/env python3
"""Calibrate the SO-101 follower, leader, or both from one YAML file."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_utils import (  # noqa: E402
    create_run,
    print_check,
    project_environment,
    reject_placeholders,
    run_logged,
    snapshot_configs,
    update_latest,
    write_yaml,
)
from system_config import calibration_configs, load_system, run_settings  # noqa: E402

CALIBRATE_DEVICE = Path(__file__).with_name("calibrate_device.py")


def _device(native: dict) -> dict:
    return native.get("robot") or native.get("teleop") or {}


def validate_hardware(native_configs: list[tuple[str, dict]]) -> None:
    """Fail early with actionable serial-port errors before LeRobot starts."""
    errors: list[str] = []
    used_ports: dict[str, str] = {}

    for name, native in native_configs:
        device = _device(native)
        raw_port = str(device["port"])
        port = Path(raw_port).expanduser()

        previous = used_ports.get(raw_port)
        if previous:
            errors.append(f"{previous}와 {name}에 같은 포트가 지정됐습니다: {raw_port}")
        used_ports[raw_port] = name

        if not port.exists():
            errors.append(
                f"{name} 포트가 없습니다: {raw_port}\n"
                "  USB를 연결한 뒤 ./libraries/venv/bin/lerobot-find-port로 다시 확인하세요."
            )
            continue
        if not port.is_char_device():
            errors.append(f"{name} 포트가 문자 장치가 아닙니다: {raw_port}")
        if not os.access(port, os.R_OK | os.W_OK):
            errors.append(
                f"{name} 포트 읽기/쓰기 권한이 없습니다: {raw_port}\n"
                "  사용자가 dialout 그룹인지 확인하고, 그룹 변경 후에는 로그아웃/로그인이 필요합니다."
            )

    if errors:
        raise RuntimeError("캘리브레이션 사전검사 실패:\n- " + "\n- ".join(errors))


def print_workflow(native_configs: list[tuple[str, dict]]) -> None:
    print("\n=== SO-101 캘리브레이션 사전 안내 ===")
    for name, native in native_configs:
        device = _device(native)
        print(f"- {name}: port={device['port']}, id={device['id']}")
    print("\n진행 순서")
    print("1. 팔 주변을 비우고, 전원과 USB가 안정적으로 연결됐는지 확인합니다.")
    print("2. 기존 calibration 파일 질문이 나오면: 새로 측정은 c + Enter, 기존 값 사용은 Enter입니다.")
    print("3. '가운데 위치가 맞으면 Enter' 프롬프트가 보일 때 관절을 중간 위치로 맞추고 Enter를 누릅니다.")
    print("4. wrist_roll을 제외한 관절과 gripper를 하나씩 천천히 최소/최대 범위로 움직입니다.")
    print("5. 전체 범위 기록을 끝내려면 Enter를 누릅니다. wrist_roll은 전체 회전 범위로 자동 설정됩니다.")
    print("이상한 저항, 케이블 걸림, 급격한 움직임이 있으면 즉시 Ctrl+C로 중단하세요.\n")


def confirm_start() -> bool:
    answer = input("준비됐으면 Enter를 누르세요. 취소하려면 q + Enter: ").strip().lower()
    return answer not in {"q", "quit", "취소"}


def get_communication_retries(settings: dict) -> int:
    retries = int(settings.get("communication_retries", 3))
    if retries < 0 or retries > 10:
        raise ValueError("runs.calibration.communication_retries는 0~10 사이여야 합니다.")
    return retries


def print_failure_hint(log_path: Path) -> None:
    try:
        log = log_path.read_text(encoding="utf-8")
    except OSError:
        return

    if "Incorrect status packet" in log:
        print("통신 응답 패킷 오류입니다. 환경 재설치나 calibration 초기화 문제는 아닙니다.")
        if "id_=6" in log or "- 6 " in log:
            print("gripper(ID 6)와 그 앞 모터 사이의 3핀 케이블, 커넥터 및 전원을 우선 확인하세요.")
        print("일시적 오류는 설정된 재시도로 처리됩니다. 반복되면 전원을 끄고 케이블을 다시 결합하세요.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="통합 system YAML")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--yes", action="store_true", help="사전 확인만 생략합니다. LeRobot 단계별 Enter 입력은 유지됩니다.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_system(config_path)
    native_configs = calibration_configs(config)
    settings = run_settings(config, "calibration")
    communication_retries = get_communication_retries(settings)
    command = [
        sys.executable,
        str(CALIBRATE_DEVICE),
        "--config_path=RESOLVED_CONFIG",
        f"--communication-retries={communication_retries}",
    ]
    if args.check:
        for name, native in native_configs:
            print_check(f"calibration:{name}", native, command)
        return 0

    reject_placeholders(native_configs)
    validate_hardware(native_configs)
    print_workflow(native_configs)
    if not args.yes and not confirm_start():
        print("캘리브레이션을 시작하지 않았습니다.")
        return 0

    run_dir, artifacts = create_run(
        settings.get("output_root", "data/calibration_runs"),
        settings.get("run_name", "calibration"),
    )
    # Keep the exact user YAML once at the top level.
    snapshot_configs(config_path, {name: native for name, native in native_configs}, artifacts)

    for name, native in native_configs:
        child = run_dir / name
        (child / "artifacts").mkdir(parents=True)
        resolved = child / "artifacts" / "resolved_lerobot.yaml"
        write_yaml(resolved, native)
        print(f"\n=== {name} 캘리브레이션 시작 ===", flush=True)
        print("줄바꿈 없는 Enter 프롬프트도 그대로 표시됩니다. 화면 문구를 확인한 뒤 입력하세요.", flush=True)
        attempt_log = child / "artifacts" / "console.log"
        code = run_logged(
            [
                sys.executable,
                str(CALIBRATE_DEVICE),
                f"--config_path={resolved}",
                f"--communication-retries={communication_retries}",
            ],
            child,
            project_environment(),
        )
        if code != 0:
            print(f"\n{name} 캘리브레이션 실패(종료 코드 {code})")
            print_failure_hint(attempt_log)
            print(f"로그: {attempt_log}")
            return code
        print(f"\n{name} 캘리브레이션 완료")
        print(f"calibration 저장 폴더: {_device(native)['calibration_dir']}")

    update_latest(settings.get("output_root", "data/calibration_runs"), run_dir)
    print("\n모든 캘리브레이션이 완료됐습니다.")
    print(f"실행 기록: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
