#!/usr/bin/env python3
"""Transfer datasets, trained checkpoints, and profiler reports over SSH/rsync."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))

from project_utils import PROJECT_ROOT, absolute_path  # noqa: E402
from system_config import load_system  # noqa: E402


def desktop(config: dict) -> tuple[str, int, PurePosixPath]:
    value = config.get("transfer", {}).get("desktop", {})
    host = value.get("host")
    user = value.get("user")
    repo = str(value.get("repo_path", ""))
    if not host or not user:
        raise ValueError(
            "config/system.yaml의 transfer.desktop.host와 user를 설정하세요. "
            "데스크탑에서 `hostname -I`로 유선 IPv4를 확인할 수 있습니다."
        )
    for label, item in (("host", str(host)), ("user", str(user)), ("repo_path", repo)):
        if any(ch.isspace() for ch in item) or any(ch in item for ch in "\n\r\0"):
            raise ValueError(f"transfer.desktop.{label}에 공백/제어문자를 사용할 수 없습니다.")
    if ":" in str(host) or ":" in str(user) or not repo.startswith("/"):
        raise ValueError("host/user는 IPv4 또는 이름이어야 하고 repo_path는 절대 경로여야 합니다.")
    return f"{user}@{host}", int(value.get("ssh_port", 22)), PurePosixPath(repo)


def rsync(source: str, destination: str, port: int, *, check: bool) -> int:
    command = [
        "rsync",
        "--archive",
        "--partial",
        "--mkpath",
        "--human-readable",
        "--info=progress2",
        "--exclude=*.tmp",
        "-e",
        f"ssh -p {port}",
        source,
        destination,
    ]
    print("$", " ".join(command))
    if check:
        return 0
    try:
        return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode
    except FileNotFoundError as exc:
        raise SystemExit("rsync가 없습니다. `sudo apt install rsync openssh-client`를 실행하세요.") from exc


def push_dataset(config: dict, check: bool) -> int:
    target, port, remote_repo = desktop(config)
    source = Path(absolute_path(config["dataset"]["storage_root"]))
    source.mkdir(parents=True, exist_ok=True)
    destination = f"{target}:{remote_repo / 'data/collected_datasets'}/"
    return rsync(f"{source}/", destination, port, check=check)


def pull_model(config: dict, check: bool) -> int:
    target, port, remote_repo = desktop(config)
    output_root = PurePosixPath(config["runs"]["training_desktop"]["output_root"])
    remote = remote_repo / output_root / "latest/training/checkpoints/last/pretrained_model"
    model_root = Path(absolute_path(config["transfer"]["model_root"]))
    run_dir = model_root / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_5070ti")
    if check:
        local = run_dir
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        local = run_dir
    code = rsync(f"{target}:{remote}/", f"{local}/", port, check=check)
    if code == 0 and not check:
        if not (run_dir / "config.json").is_file():
            raise FileNotFoundError(
                f"전송은 끝났지만 checkpoint config.json이 없습니다: {run_dir}. "
                "데스크탑 학습이 정상 종료되었는지 확인하세요."
            )
        latest = model_root / "latest"
        temporary = model_root / ".latest.tmp"
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(run_dir.name)
        os.replace(temporary, latest)
        print(f"Orin inference model: {latest}")
    return code


def push_ncu(config: dict, check: bool) -> int:
    target, port, remote_repo = desktop(config)
    output_root = Path(absolute_path(config["runs"]["benchmark"]["output_root"]))
    def successful_ncu(path: Path) -> bool:
        metadata = path / "artifacts/run_metadata.json"
        try:
            return json.loads(metadata.read_text(encoding="utf-8")).get("exit_code") == 0 and any(
                path.glob("*.ncu-rep")
            )
        except (OSError, ValueError):
            return False

    candidates = sorted(
        (path for path in output_root.glob("*_ncu") if path.is_dir() and successful_ncu(path)),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        if check:
            source = output_root / "LATEST_NCU_RUN"
        else:
            raise FileNotFoundError(
                "전송할 NCU run이 없습니다. 먼저 ./launchfiles/profile_ncu.bash config/system.yaml을 실행하세요."
            )
    else:
        source = candidates[0]
    remote_root = PurePosixPath(config["transfer"].get("profiling_root", "data/profiling_from_orin"))
    destination = f"{target}:{remote_repo / remote_root / socket.gethostname() / source.name}/"
    return rsync(f"{source}/", destination, port, check=check)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("push-dataset", "pull-model", "push-ncu"))
    parser.add_argument("config", help="통합 system YAML")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = load_system(args.config)
    functions = {
        "push-dataset": push_dataset,
        "pull-model": pull_model,
        "push-ncu": push_ncu,
    }
    try:
        return functions[args.action](config, args.check)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
