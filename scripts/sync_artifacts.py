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


def rsync(source: str, destination: str, port: int, *, check: bool, quiet: bool = False) -> int:
    ssh_command = (
        f"ssh -p {port} -o BatchMode=yes -o ConnectTimeout=5 "
        "-o ServerAliveInterval=5 -o ServerAliveCountMax=1"
    )
    command = [
        "rsync",
        "--archive",
        "--partial",
        "--mkpath",
        "--human-readable",
        "--exclude=*.tmp",
        "-e",
        ssh_command,
        source,
        destination,
    ]
    if not quiet:
        command.insert(6, "--info=progress2")
        print("$", " ".join(command))
    if check:
        return 0
    try:
        return subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            stdout=subprocess.DEVNULL if quiet else None,
        ).returncode
    except FileNotFoundError as exc:
        raise SystemExit("rsync가 없습니다. `sudo apt install rsync openssh-client`를 실행하세요.") from exc


def push_dataset(config: dict, check: bool, quiet: bool = False) -> int:
    target, port, remote_repo = desktop(config)
    source = Path(absolute_path(config["dataset"]["storage_root"]))
    source.mkdir(parents=True, exist_ok=True)
    destination = f"{target}:{remote_repo / 'data/collected_datasets'}/"
    return rsync(f"{source}/", destination, port, check=check, quiet=quiet)


def rebase_model_paths(config: dict, checkpoint_dir: Path) -> None:
    """Replace training-machine paths embedded in a PEFT checkpoint."""
    base_path = absolute_path(config["model"]["base"]["path"])
    vlm_path = absolute_path(config["model"]["vlm"]["path"])
    replacements = {
        "config.json": {
            "pretrained_path": base_path,
            "vlm_model_name": vlm_path,
        },
        "adapter_config.json": {
            "base_model_name_or_path": base_path,
        },
    }
    for filename, values in replacements.items():
        path = checkpoint_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint 파일이 없습니다: {path}")
        document = json.loads(path.read_text(encoding="utf-8"))
        document.update(values)
        path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def pull_model(config: dict, check: bool, quiet: bool = False) -> int:
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
        rebase_model_paths(config, run_dir)
        latest = model_root / "latest"
        temporary = model_root / ".latest.tmp"
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(run_dir.name)
        os.replace(temporary, latest)
        print(f"Orin inference model: {latest}")
    return code


def _profiling_destination(config: dict, source: Path) -> tuple[str, int, str]:
    target, port, remote_repo = desktop(config)
    remote_root = PurePosixPath(config["transfer"].get("profiling_root", "data/profiling_from_orin"))
    destination = f"{target}:{remote_repo / remote_root / socket.gethostname() / source.name}/"
    return target, port, destination


def push_profile_run(config: dict, source: Path, check: bool, quiet: bool = False) -> int:
    source = source.resolve()
    if not check and not source.is_dir():
        raise FileNotFoundError(f"전송할 profiling run이 없습니다: {source}")
    _target, port, destination = _profiling_destination(config, source)
    return rsync(f"{source}/", destination, port, check=check, quiet=quiet)


def push_profiling(config: dict, check: bool, quiet: bool = False) -> int:
    output_root = Path(absolute_path(config["runs"]["benchmark"]["output_root"]))

    def has_profile(path: Path) -> bool:
        metadata = path / "artifacts/run_metadata.json"
        try:
            run_metadata = json.loads(metadata.read_text(encoding="utf-8"))
            return bool(run_metadata.get("live_robot_actions")) or any(
                path.glob("*.ncu-rep")
            ) or any(path.glob("*.nsys-rep")) or any(
                (path / "artifacts").glob("torch_trace*.json")
            )
        except (OSError, ValueError):
            return False

    candidates = sorted(
        (path for path in output_root.iterdir() if path.is_dir() and has_profile(path)),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ) if output_root.is_dir() else []
    if not candidates:
        if check:
            source = output_root / "LATEST_PROFILE_RUN"
        else:
            raise FileNotFoundError(
                "전송할 profiling run이 없습니다. 먼저 성능 측정 명령을 실행하세요."
            )
    else:
        source = candidates[0]
    return push_profile_run(config, source, check, quiet)


def push_ncu(config: dict, check: bool, quiet: bool = False) -> int:
    """Backward-compatible alias; now sends the latest run of any profiler type."""
    return push_profiling(config, check, quiet)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("push-dataset", "pull-model", "push-ncu", "push-profiling")
    )
    parser.add_argument("config", help="통합 system YAML")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="성공 진행률 숨김 (자동 동기화용)")
    args = parser.parse_args()
    config = load_system(args.config)
    functions = {
        "push-dataset": push_dataset,
        "pull-model": pull_model,
        "push-ncu": push_ncu,
        "push-profiling": push_profiling,
    }
    try:
        return functions[args.action](config, args.check, args.quiet)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
