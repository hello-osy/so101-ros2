#!/usr/bin/env python3
"""Profile the real camera-to-policy-to-robot rollout."""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_utils import (  # noqa: E402
    command_path,
    create_run,
    print_check,
    project_environment,
    reject_placeholders,
    snapshot_configs,
    system_metadata,
    update_latest,
    write_json,
)
from sync_artifacts import push_profile_run  # noqa: E402
from system_config import benchmark_config, load_system  # noqa: E402


def profiling_is_admin_only(parameters: str) -> bool:
    return any(
        line.strip() == "RmProfilingAdminOnly: 1" for line in parameters.splitlines()
    )


def is_jetson_platform() -> bool:
    try:
        model = Path("/proc/device-tree/model").read_text(encoding="utf-8").strip("\x00\n")
    except OSError:
        return False
    return "NVIDIA Jetson" in model


def require_gpu_profiling_permission(profiler: str) -> None:
    """Fail before connecting the robot when CUPTI cannot collect GPU activity."""
    if (
        profiler == "none"
        or is_jetson_platform()  # Tegra profiler child is elevated separately below.
        or not hasattr(os, "geteuid")
        or os.geteuid() == 0
    ):
        return
    parameters_path = Path("/proc/driver/nvidia/params")
    try:
        parameters = parameters_path.read_text(encoding="utf-8")
    except OSError:
        return
    if profiling_is_admin_only(parameters):
        raise PermissionError(
            "NVIDIA GPU profiler 권한이 제한되어 있습니다 "
            "(/proc/driver/nvidia/params: RmProfilingAdminOnly=1).\n"
            "다음 명령을 한 번 실행하고 재부팅하세요:\n"
            "  echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | "
            "sudo tee /etc/modprobe.d/so101-profiler.conf\n"
            "  sudo update-initramfs -u\n"
            "  sudo reboot\n"
            "재부팅 후 확인:\n"
            "  grep RmProfilingAdminOnly /proc/driver/nvidia/params"
        )


def profiler_needs_root(profiler: str, gpu_activity: bool = False) -> bool:
    return profiler == "ncu" or gpu_activity


def elevate_profiler_on_jetson(
    command: list[str], profiler: str, gpu_activity: bool = False
) -> list[str]:
    """Run only the profiled rollout as root; keep rsync and artifacts owned by the user."""
    if (
        not profiler_needs_root(profiler, gpu_activity)
        or not is_jetson_platform()
        or os.geteuid() == 0
    ):
        return command
    environment = project_environment()
    names = (
        "PATH",
        "PYTHONPATH",
        "HF_HOME",
        "HF_LEROBOT_HOME",
        "HF_LEROBOT_CALIBRATION",
        "SO101_PROJECT_ROOT",
        "SO101_LEROBOT_DIR",
        "SO101_VENV_DIR",
        "PYTHONUNBUFFERED",
    )
    assignments = [f"{name}={environment[name]}" for name in names]
    return [command_path("sudo"), "--", command_path("env"), *assignments, *command]


def authorize_jetson_profiler(profiler: str, gpu_activity: bool = False) -> None:
    """Cache sudo credentials before the child enters its detached process group."""
    if (
        not profiler_needs_root(profiler, gpu_activity)
        or not is_jetson_platform()
        or os.geteuid() == 0
    ):
        return
    print("Jetson CUDA trace 권한을 위해 sudo 인증이 필요합니다.")
    subprocess.run([command_path("sudo"), "-v"], check=True)


def signal_process_group(
    process: subprocess.Popen[str], requested_signal: signal.Signals, *, elevated: bool
) -> None:
    """Signal the complete detached profiler tree, including root-owned descendants."""
    try:
        os.killpg(process.pid, requested_signal)
    except (ProcessLookupError, PermissionError):
        pass
    if elevated:
        # sudo/ncu/Python share the detached process group. The ordinary kill above
        # cannot necessarily reach root-owned descendants after sudo has forked.
        subprocess.run(
            [
                command_path("sudo"), "-n", "--", command_path("kill"),
                f"-{requested_signal.name}", "--", f"-{process.pid}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=False,
        )


def process_group_is_alive(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # A root-owned NCU/Python descendant can remain after sudo exits.
        return True
    return True


def safe_gpu_profile_is_ready(status_path: Path | None) -> bool:
    if status_path is None:
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return status.get("status") == "ready" and int(
        status.get("live_inferences_before_capture", 0)
    ) > 0


def profiler_prefix(profiler: str, config: dict, run_dir: Path) -> list[str]:
    profiling = config.get("profiling", {})
    if profiler in {"none", "torch"}:
        return []
    if profiler == "nsys":
        options = profiling.get("nsys", {})
        return [
            command_path("nsys"), "profile",
            f"--trace={options.get('trace', 'cuda,nvtx,osrt')}",
            f"--sample={options.get('sample', 'none')}",
            "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
            "--force-overwrite=true", f"--output={run_dir / 'nsys_profile'}",
        ]
    if profiler == "ncu":
        options = profiling.get("ncu", {})
        command = [
            command_path("ncu"), "--target-processes", str(options.get("target_processes", "all")),
            "--profile-from-start", "off", "--set", str(options.get("set", "basic")),
        ]
        launch_count = int(options.get("launch_count", 0))
        if launch_count > 0:
            command.extend(["--launch-count", str(launch_count)])
        command.extend(["--force-overwrite", "--export", str(run_dir / "ncu_profile")])
        return command
    raise ValueError(f"지원하지 않는 profiler: {profiler}")


@contextlib.contextmanager
def jetson_telemetry(config: dict, artifacts: Path):
    options = config.get("profiling", {}).get("tegrastats", {})
    executable = shutil.which("tegrastats") if bool(options.get("enable", True)) else None
    if not executable:
        (artifacts / "tegrastats_unavailable.txt").write_text(
            "tegrastats is disabled or unavailable.\n", encoding="utf-8"
        )
        yield
        return
    errors = (artifacts / "tegrastats.stderr.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        [executable, "--interval", str(int(options.get("interval_ms", 250))),
         "--logfile", str(artifacts / "tegrastats.log")],
        stdout=subprocess.DEVNULL, stderr=errors, start_new_session=True,
    )
    try:
        yield
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        errors.close()


def run_interactive(
    command: list[str],
    run_dir: Path,
    safe_profile_request: Path | None = None,
    safe_profile_status: Path | None = None,
    *,
    elevated: bool = False,
) -> tuple[int, bool, float]:
    """Turn Ctrl-C into a safe GPU capture request, or forward it for normal cleanup."""
    artifacts = run_dir / "artifacts"
    command_text = shlex.join(command)
    (artifacts / "command.txt").write_text(command_text + "\n", encoding="utf-8")
    started = time.perf_counter()
    interrupted = False
    with (artifacts / "console.log").open("w", encoding="utf-8", buffering=1) as log:
        log.write(f"$ {command_text}\n")
        process = subprocess.Popen(
            command, cwd=Path(__file__).resolve().parents[2], env=project_environment(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            start_new_session=True,
        )
        try:
            assert process.stdout is not None
            while True:
                chunk = process.stdout.read(1)
                if chunk == "":
                    break
                print(chunk, end="", flush=True)
                log.write(chunk)
            code = process.wait()
        except KeyboardInterrupt:
            interrupted = True
            if safe_profile_request is not None and safe_gpu_profile_is_ready(
                safe_profile_status
            ):
                safe_profile_request.write_text(
                    datetime.now().astimezone().isoformat() + "\n", encoding="utf-8"
                )
                print(
                    "\nCtrl-C: action 전송을 차단했습니다. 동일 프로세스에서 안전 GPU capture 후 종료합니다...",
                    flush=True,
                )
            elif safe_profile_request is not None:
                print(
                    "\nCtrl-C: 실제 inference가 아직 시작되지 않아 GPU capture 없이 종료합니다...",
                    flush=True,
                )
                signal_process_group(process, signal.SIGINT, elevated=elevated)
            else:
                print("\nCtrl-C: 로봇과 profiler를 안전하게 정리하는 중입니다...", flush=True)
                signal_process_group(process, signal.SIGINT, elevated=elevated)
            force_count = 0

            def force_cleanup(_signum, _frame):
                nonlocal force_count
                force_count += 1
                if force_count == 1:
                    print(
                        "\n두 번째 Ctrl-C: 전체 profiler process group을 종료합니다 "
                        "(5초 뒤 남으면 강제 종료)...",
                        flush=True,
                    )
                    signal_process_group(process, signal.SIGTERM, elevated=elevated)

                    def kill_if_still_running() -> None:
                        time.sleep(5)
                        if process_group_is_alive(process.pid):
                            print(
                                "profiler가 종료되지 않아 전체 process group을 SIGKILL로 정리합니다...",
                                flush=True,
                            )
                            signal_process_group(process, signal.SIGKILL, elevated=elevated)

                    threading.Thread(target=kill_if_still_running, daemon=True).start()
                else:
                    print(
                        "\n추가 Ctrl-C: 전체 profiler process group을 즉시 SIGKILL로 정리합니다...",
                        flush=True,
                    )
                    signal_process_group(process, signal.SIGKILL, elevated=elevated)

            previous = signal.signal(signal.SIGINT, force_cleanup)
            try:
                code = process.wait()
            finally:
                signal.signal(signal.SIGINT, previous)
    return code, interrupted, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="통합 system YAML")
    parser.add_argument("--profiler", choices=("none", "torch", "nsys", "ncu"), default="none")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    system = load_system(config_path)
    resolved = benchmark_config(system)
    child = Path(__file__).with_name("inference_child.py")
    example = profiler_prefix(args.profiler, resolved, Path("PROFILE_RUN")) + [
        command_path("python"), str(child), "--config", "RESOLVED_CONFIG"
    ]
    if args.check:
        print_check(f"moving-profile:{args.profiler}", resolved, example)
        return 0

    try:
        require_gpu_profiling_permission(args.profiler)
    except PermissionError as exc:
        parser.error(str(exc))
    reject_placeholders(resolved["inference"])
    policy_path = Path(resolved["inference"]["policy"]["path"])
    if policy_path.is_absolute() and not policy_path.exists():
        raise FileNotFoundError(f"정책 checkpoint가 없습니다: {policy_path}")
    torch_options = resolved["profiling"].get("torch", {})
    torch_cuda_activity = bool(torch_options.get("cuda_activity", False))
    nsys_trace = {
        item.strip() for item in str(resolved["profiling"].get("nsys", {}).get("trace", "")).split(",")
    }
    gpu_activity = torch_cuda_activity if args.profiler == "torch" else (
        args.profiler == "nsys" and "cuda" in nsys_trace
    )
    safe_gpu_profile = args.profiler in {"nsys", "ncu"}
    authorize_jetson_profiler(args.profiler, gpu_activity)

    project = resolved["project"]
    run_dir, artifacts = create_run(
        project.get("output_root", "data/inference_logs"),
        f"{project.get('run_name', 'moving_profile')}_{args.profiler}",
    )
    if safe_gpu_profile:
        # Actions remain gated after capture; teardown must not move the robot.
        resolved["inference"]["return_to_initial_position"] = False
    rollout = snapshot_configs(config_path, resolved["inference"], artifacts)
    benchmark = resolved["benchmark"]
    command = profiler_prefix(args.profiler, resolved, run_dir) + [
        command_path("python"), str(child), "--config", str(rollout),
        "--metrics-output", str(artifacts / "latency.jsonl"),
        "--summary-output", str(artifacts / "latency_summary.json"),
        "--warmup-inferences", "0",
        # 0 keeps latency, memory, and profiler capture active until Ctrl-C.
        "--profile-iterations", "0",
    ]
    if bool(benchmark.get("cuda_synchronize", True)):
        command.append("--cuda-synchronize")
    if args.profiler in {"nsys", "ncu"}:
        command.append("--cuda-profiler-api")
    safe_request = artifacts / "safe_gpu_profile.request" if safe_gpu_profile else None
    safe_status = artifacts / "safe_gpu_profile_status.json" if safe_gpu_profile else None
    if safe_request is not None:
        safe_options = resolved["profiling"].get("safe_gpu", {})
        command.extend([
            "--safe-gpu-profile-request", str(safe_request),
            "--safe-gpu-profile-status", str(safe_status),
            "--safe-gpu-profile-iterations", str(int(safe_options.get("iterations", 1))),
        ])
    if args.profiler == "torch":
        command.extend([
            "--torch-profile-output", str(artifacts / "torch_trace.json"),
            "--torch-profile-table", str(artifacts / "torch_operators.txt"),
        ])
        if torch_cuda_activity:
            command.append("--torch-profile-cuda")
        if bool(torch_options.get("record_shapes", False)):
            command.append("--torch-profile-record-shapes")
        if bool(torch_options.get("profile_memory", False)):
            command.append("--torch-profile-memory")
    elevated = (
        profiler_needs_root(args.profiler, gpu_activity)
        and is_jetson_platform()
        and os.geteuid() != 0
    )
    command = elevate_profiler_on_jetson(command, args.profiler, gpu_activity)

    if safe_gpu_profile:
        print(
            "실제 rollout로 GPU를 warm-up합니다. Ctrl-C 한 번: action 차단 → 같은 프로세스 GPU capture → 전송/종료"
        )
    else:
        print("실제 로봇 동작 추론 측정을 시작합니다. 종료 및 데스크탑 전송: Ctrl-C 한 번")
    with jetson_telemetry(resolved, artifacts):
        code, interrupted, elapsed = run_interactive(
            command,
            run_dir,
            safe_request,
            safe_status,
            elevated=elevated,
        )
    metadata = system_metadata()
    metadata.update({"exit_code": code, "elapsed_seconds": elapsed,
                     "interrupted_by_user": interrupted, "live_robot_actions": True})
    write_json(artifacts / "run_metadata.json", metadata)
    update_latest(project.get("output_root", "data/inference_logs"), run_dir)
    print(f"profiling output: {run_dir}")

    if bool(resolved["profiling"].get("auto_push_on_exit", True)) and (interrupted or code == 0):
        print("5070 Ti 데스크탑으로 profiling 데이터를 전송합니다...")
        sync_code = push_profile_run(system, run_dir, check=False)
        if sync_code:
            print(f"전송 실패(rsync exit {sync_code}); 로컬 결과는 보존했습니다.")
            return sync_code
        print("profiling 전송 완료")
    return 0 if interrupted else code


if __name__ == "__main__":
    raise SystemExit(main())
