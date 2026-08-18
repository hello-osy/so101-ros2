#!/usr/bin/env python3
"""Profile the real camera-to-policy-to-robot rollout."""

from __future__ import annotations

import argparse
import contextlib
import os
import shlex
import shutil
import signal
import subprocess
import sys
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
        return [
            command_path("ncu"), "--target-processes", str(options.get("target_processes", "all")),
            "--profile-from-start", "off", "--set", str(options.get("set", "basic")),
            "--launch-count", str(int(options.get("launch_count", 10))),
            "--force-overwrite", "--export", str(run_dir / "ncu_profile"),
        ]
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


def run_interactive(command: list[str], run_dir: Path) -> tuple[int, bool, float]:
    """Forward the first Ctrl-C to the rollout and wait for safe cleanup."""
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
            print("\nCtrl-C: 로봇과 profiler를 안전하게 정리하는 중입니다...", flush=True)
            os.killpg(process.pid, signal.SIGINT)
            # Ignore another terminal SIGINT while the rollout disconnects the robot.
            previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
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

    reject_placeholders(resolved["inference"])
    policy_path = Path(resolved["inference"]["policy"]["path"])
    if policy_path.is_absolute() and not policy_path.exists():
        raise FileNotFoundError(f"정책 checkpoint가 없습니다: {policy_path}")

    project = resolved["project"]
    run_dir, artifacts = create_run(
        project.get("output_root", "data/inference_logs"),
        f"{project.get('run_name', 'moving_profile')}_{args.profiler}",
    )
    rollout = snapshot_configs(config_path, resolved["inference"], artifacts)
    benchmark = resolved["benchmark"]
    command = profiler_prefix(args.profiler, resolved, run_dir) + [
        command_path("python"), str(child), "--config", str(rollout),
        "--metrics-output", str(artifacts / "latency.jsonl"),
        "--summary-output", str(artifacts / "latency_summary.json"),
        "--warmup-inferences", "0",
        "--profile-iterations", str(int(benchmark.get("iterations", 20))),
    ]
    if bool(benchmark.get("cuda_synchronize", True)):
        command.append("--cuda-synchronize")
    if args.profiler in {"nsys", "ncu"}:
        command.append("--cuda-profiler-api")
    if args.profiler == "torch":
        torch_options = resolved["profiling"].get("torch", {})
        command.extend([
            "--torch-profile-output", str(artifacts / "torch_trace.json"),
            "--torch-profile-table", str(artifacts / "torch_operators.txt"),
            "--torch-profile-iterations", str(int(torch_options.get("active_iterations", 5))),
        ])

    print("실제 로봇 동작 추론 측정을 시작합니다. 종료 및 데스크탑 전송: Ctrl-C 한 번")
    with jetson_telemetry(resolved, artifacts):
        code, interrupted, elapsed = run_interactive(command, run_dir)
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
