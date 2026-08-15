"""Shared, dependency-light helpers for all project launchers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_ROOT = PROJECT_ROOT / "libraries" / "lerobot"
VENV_ROOT = PROJECT_ROOT / "libraries" / "venv"


def runtime_venv() -> Path:
    """Use the desktop venv when a launcher selects it, otherwise use Jetson's."""
    configured = os.environ.get("SO101_VENV_DIR")
    return Path(configured).expanduser().resolve() if configured else VENV_ROOT


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"YAML 최상위 값은 mapping이어야 합니다: {path}")
    return data


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive merge without mutating either input."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def absolute_path(value: str | Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def local_path_or_hub_id(value: str) -> str:
    """Resolve project-local paths while leaving Hugging Face repo IDs unchanged."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    if value.startswith(("data/", "libraries/", "scripts/", "./", "../")):
        return absolute_path(value)
    if (PROJECT_ROOT / path).exists():
        return absolute_path(value)
    return value


def timestamp_id(name: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name).strip("-")
    return f"{datetime.now().astimezone():%Y%m%d_%H%M%S}_{clean or 'run'}"


def create_run(output_root: str | Path, name: str) -> tuple[Path, Path]:
    root = Path(absolute_path(output_root))
    run_dir = root / timestamp_id(name)
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{timestamp_id(name)}_{suffix:02d}"
        suffix += 1
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)
    return run_dir, artifacts


def update_latest(output_root: str | Path, run_dir: Path) -> None:
    """Atomically update the ignored latest symlink after a successful run."""
    latest = Path(absolute_path(output_root)) / "latest"
    temporary = latest.with_name(".latest.tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(run_dir.name)
    temporary.replace(latest)


def snapshot_configs(
    source_config: str | Path,
    generated_config: dict[str, Any],
    artifacts: Path,
    extra_configs: list[str | Path] | None = None,
) -> Path:
    source = Path(source_config).resolve()
    shutil.copy2(source, artifacts / source.name)
    for extra in extra_configs or []:
        extra_path = Path(extra).resolve()
        shutil.copy2(extra_path, artifacts / extra_path.name)
    generated_path = artifacts / "resolved_lerobot.yaml"
    write_yaml(generated_path, generated_config)
    return generated_path


def require_keys(mapping: dict[str, Any], *keys: str, context: str = "config") -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{context}에 필수 키가 없습니다: {', '.join(missing)}")


def reject_placeholders(data: Any) -> None:
    if isinstance(data, dict):
        for value in data.values():
            reject_placeholders(value)
    elif isinstance(data, (list, tuple)):
        for value in data:
            reject_placeholders(value)
    elif isinstance(data, str) and "CHANGE_ME" in data:
        raise ValueError("YAML의 CHANGE_ME 값을 실제 장치/사용자 값으로 바꿔주세요.")


def command_path(name: str) -> str:
    candidate = runtime_venv() / "bin" / name
    if candidate.exists():
        return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"{name}을 찾지 못했습니다. 먼저 ./scripts/bootstrap_environment.bash config/system.yaml을 실행하세요."
    )


def project_environment() -> dict[str, str]:
    env = os.environ.copy()
    venv_root = runtime_venv()
    env.update(
        {
            "SO101_PROJECT_ROOT": str(PROJECT_ROOT),
            "SO101_LEROBOT_DIR": str(LEROBOT_ROOT),
            "SO101_VENV_DIR": str(venv_root),
            "HF_HOME": str(PROJECT_ROOT / "data" / "models" / "huggingface"),
            "HF_LEROBOT_HOME": str(PROJECT_ROOT / "data" / "downloaded_datasets" / "lerobot"),
            "HF_LEROBOT_CALIBRATION": str(PROJECT_ROOT / "data" / "calibration"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    python_path = str(LEROBOT_ROOT / "src")
    if env.get("PYTHONPATH"):
        python_path += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = python_path
    env["PATH"] = str(venv_root / "bin") + os.pathsep + env.get("PATH", "")
    return env


def _short_command(args: list[str]) -> str:
    return shlex.join(str(arg) for arg in args)


def run_logged(args: list[str], run_dir: Path, env: dict[str, str] | None = None) -> int:
    """Stream a subprocess to the terminal and persist the same output."""
    artifacts = run_dir / "artifacts"
    log_path = artifacts / "console.log"
    command = [str(arg) for arg in args]
    (artifacts / "command.txt").write_text(_short_command(command) + "\n", encoding="utf-8")

    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write(f"$ {_short_command(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env or project_environment(),
            stdin=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            # input() prompts often have no trailing newline. Reading a line at
            # a time hides the prompt while the child is already waiting.
            while True:
                chunk = process.stdout.read(1)
                if chunk == "":
                    break
                print(chunk, end="", flush=True)
                log.write(chunk)
            code = process.wait()
        except KeyboardInterrupt:
            process.send_signal(2)
            code = process.wait()

    elapsed = time.perf_counter() - started
    metadata = system_metadata()
    metadata.update({"exit_code": code, "elapsed_seconds": elapsed})
    write_json(artifacts / "run_metadata.json", metadata)
    return code


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _capture(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.run(
            args,
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=8,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def system_metadata() -> dict[str, Any]:
    """Collect reproducibility metadata without requiring CUDA to be available."""
    try:
        jetson_model = Path("/proc/device-tree/model").read_text(encoding="utf-8").strip("\x00\n")
    except OSError:
        jetson_model = None
    try:
        memory_total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        memory_total = None
    metadata: dict[str, Any] = {
        "recorded_at": datetime.now().astimezone().isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "memory_total_bytes": memory_total,
        "jetson_model": jetson_model,
        "jetpack_l4t": _capture(
            ["dpkg-query", "--showformat=${Version}", "--show", "nvidia-l4t-core"]
        ),
        "nvpmodel": _capture(["nvpmodel", "-q"]),
        "jetson_clocks": _capture(["jetson_clocks", "--show"]),
        "project_git_commit": _capture(["git", "rev-parse", "HEAD"], PROJECT_ROOT),
        "project_git_status": _capture(["git", "status", "--short"], PROJECT_ROOT),
        "lerobot_git_commit": _capture(["git", "rev-parse", "HEAD"], LEROBOT_ROOT),
        "nsys_version": _capture(["nsys", "--version"]),
        "ncu_version": _capture(["ncu", "--version"]),
    }
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        metadata["torch"] = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": cuda_available,
            "cudnn": torch.backends.cudnn.version(),
            "gpu_count": torch.cuda.device_count() if cuda_available else 0,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if cuda_available
            else [],
        }
    except Exception as exc:  # metadata collection must never hide the real run result
        metadata["torch_error"] = repr(exc)
    return metadata


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def print_check(name: str, native: dict[str, Any], command: list[str]) -> None:
    print(f"[{name}] config validation OK")
    print(yaml.safe_dump(native, allow_unicode=True, sort_keys=False).rstrip())
    print(f"command: {_short_command(command)}")
