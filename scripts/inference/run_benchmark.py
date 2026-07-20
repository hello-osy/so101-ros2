#!/usr/bin/env python3
"""Run offline inference timing directly or under NVIDIA profilers."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_utils import (  # noqa: E402
    absolute_path,
    command_path,
    create_run,
    load_yaml,
    local_path_or_hub_id,
    print_check,
    project_environment,
    reject_placeholders,
    require_keys,
    run_logged,
    update_latest,
    write_yaml,
)


def resolve_config(config: dict) -> dict:
    resolved = dict(config)
    benchmark = dict(resolved.get("benchmark", {}))
    require_keys(benchmark, "dataset", "policy", "device", context="benchmark")
    dataset = dict(benchmark["dataset"])
    policy = dict(benchmark["policy"])
    require_keys(dataset, "repo_id", "root", context="benchmark.dataset")
    require_keys(policy, "path", context="benchmark.policy")
    dataset["root"] = absolute_path(dataset["root"])
    policy["path"] = local_path_or_hub_id(str(policy["path"]))
    benchmark["dataset"] = dataset
    benchmark["policy"] = policy
    resolved["benchmark"] = benchmark
    return resolved


def profiler_prefix(profiler: str, config: dict, run_dir: Path) -> list[str]:
    profile_cfg = config.get("profiling", {})
    if profiler == "none":
        return []
    if profiler == "nsys":
        nsys = profile_cfg.get("nsys", {})
        return [
            command_path("nsys"),
            "profile",
            f"--trace={nsys.get('trace', 'cuda,nvtx,osrt')}",
            f"--sample={nsys.get('sample', 'none')}",
            "--force-overwrite=true",
            f"--output={run_dir / 'nsys_profile'}",
        ]
    if profiler == "ncu":
        ncu = profile_cfg.get("ncu", {})
        return [
            command_path("ncu"),
            "--target-processes",
            str(ncu.get("target_processes", "all")),
            "--set",
            str(ncu.get("set", "basic")),
            "--launch-count",
            str(int(ncu.get("launch_count", 10))),
            "--force-overwrite",
            "--export",
            str(run_dir / "ncu_profile"),
        ]
    raise ValueError(f"지원하지 않는 profiler: {profiler}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--profiler", choices=("none", "nsys", "ncu"), default="none")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    resolved_config = resolve_config(load_yaml(config_path))
    child = Path(__file__).with_name("benchmark_child.py")
    command = profiler_prefix(args.profiler, resolved_config, Path("PROFILE_RUN")) + [
        command_path("python"),
        str(child),
        "--config",
        "RESOLVED_CONFIG",
        "--metrics-output",
        "METRICS_OUTPUT",
    ]
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
    code = run_logged(child_command, run_dir, project_environment())
    if code == 0:
        update_latest(project.get("output_root", "data/inference_logs"), run_dir)
        print(f"benchmark output: {run_dir}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
