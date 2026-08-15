#!/usr/bin/env python3
"""Run offline inference timing directly or under NVIDIA profilers."""

from __future__ import annotations

import argparse
import contextlib
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_utils import (  # noqa: E402
    command_path,
    create_run,
    print_check,
    project_environment,
    reject_placeholders,
    run_logged,
    update_latest,
    write_yaml,
)
from system_config import benchmark_config, load_system  # noqa: E402


def profiler_prefix(profiler: str, config: dict, run_dir: Path) -> list[str]:
    profile_cfg = config.get("profiling", {})
    if profiler in {"none", "torch"}:
        return []
    if profiler == "nsys":
        nsys = profile_cfg.get("nsys", {})
        return [
            command_path("nsys"),
            "profile",
            f"--trace={nsys.get('trace', 'cuda,nvtx,osrt')}",
            f"--sample={nsys.get('sample', 'none')}",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--force-overwrite=true",
            f"--output={run_dir / 'nsys_profile'}",
        ]
    if profiler == "ncu":
        ncu = profile_cfg.get("ncu", {})
        return [
            command_path("ncu"),
            "--target-processes",
            str(ncu.get("target_processes", "all")),
            "--profile-from-start",
            "off",
            "--set",
            str(ncu.get("set", "basic")),
            "--launch-count",
            str(int(ncu.get("launch_count", 10))),
            "--force-overwrite",
            "--export",
            str(run_dir / "ncu_profile"),
        ]
    raise ValueError(f"지원하지 않는 profiler: {profiler}")


@contextlib.contextmanager
def jetson_telemetry(config: dict, artifacts: Path):
    """Capture unified-memory, clocks, temperature, and power alongside a run."""
    telemetry = config.get("profiling", {}).get("tegrastats", {})
    if not bool(telemetry.get("enable", True)):
        yield
        return
    executable = shutil.which("tegrastats")
    if not executable:
        (artifacts / "tegrastats_unavailable.txt").write_text(
            "tegrastats was not found; GPU timing is still available in benchmark_metrics.json.\n",
            encoding="utf-8",
        )
        yield
        return

    log_path = artifacts / "tegrastats.log"
    error_stream = (artifacts / "tegrastats.stderr.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            executable,
            "--interval",
            str(int(telemetry.get("interval_ms", 250))),
            "--logfile",
            str(log_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=error_stream,
        start_new_session=True,
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
        error_stream.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="통합 system YAML")
    parser.add_argument("--profiler", choices=("none", "torch", "nsys", "ncu"), default="none")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    resolved_config = benchmark_config(load_system(config_path))
    child = Path(__file__).with_name("benchmark_child.py")
    command = profiler_prefix(args.profiler, resolved_config, Path("PROFILE_RUN")) + [
        command_path("python"),
        str(child),
        "--config",
        "RESOLVED_CONFIG",
        "--metrics-output",
        "METRICS_OUTPUT",
    ]
    if args.profiler in {"nsys", "ncu"}:
        command.append("--cuda-profiler-api")
    if args.check:
        print_check(f"benchmark:{args.profiler}", resolved_config, command)
        return 0

    reject_placeholders(resolved_config)
    benchmark = resolved_config["benchmark"]
    dataset_root = Path(benchmark["dataset"]["root"])
    if not dataset_root.exists():
        raise FileNotFoundError(f"benchmark dataset이 없습니다: {dataset_root}")
    policy_path = benchmark["policy"]["path"]
    if Path(policy_path).is_absolute() and not Path(policy_path).exists():
        raise FileNotFoundError(f"benchmark policy가 없습니다: {policy_path}")

    project = resolved_config.get("project", {})
    run_dir, artifacts = create_run(
        project.get("output_root", "data/inference_logs"),
        f"{project.get('run_name', 'benchmark')}_{args.profiler}",
    )
    shutil.copy2(config_path, artifacts / config_path.name)
    resolved_path = artifacts / "resolved_profile.yaml"
    write_yaml(resolved_path, resolved_config)
    child_command = profiler_prefix(args.profiler, resolved_config, run_dir) + [
        command_path("python"),
        str(child),
        "--config",
        str(resolved_path),
        "--metrics-output",
        str(artifacts / "benchmark_metrics.json"),
    ]
    if args.profiler in {"nsys", "ncu"}:
        child_command.append("--cuda-profiler-api")
    if args.profiler == "torch":
        torch_cfg = resolved_config.get("profiling", {}).get("torch", {})
        child_command.extend(
            [
                "--torch-profile-output",
                str(artifacts / "torch_trace.json"),
                "--torch-profile-table",
                str(artifacts / "torch_operators.txt"),
                "--torch-profile-iterations",
                str(int(torch_cfg.get("active_iterations", 5))),
            ]
        )
    with jetson_telemetry(resolved_config, artifacts):
        code = run_logged(child_command, run_dir, project_environment())
    if code == 0:
        update_latest(project.get("output_root", "data/inference_logs"), run_dir)
        print(f"benchmark output: {run_dir}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
