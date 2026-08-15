#!/usr/bin/env python3
"""Run live SO-101 policy inference and persist per-decision latency."""

from __future__ import annotations

import argparse
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
    snapshot_configs,
    update_latest,
)
from system_config import inference_config, load_system, run_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="통합 system YAML")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_system(config_path)
    settings = run_settings(config, "inference")
    native = inference_config(config)
    child = Path(__file__).with_name("inference_child.py")
    command = [command_path("python"), str(child), "--config", "RESOLVED_CONFIG"]
    if args.check:
        print_check("inference", native, command)
        return 0

    reject_placeholders(native)
    policy_path = native["policy"]["path"]
    if Path(policy_path).is_absolute() and not Path(policy_path).exists():
        raise FileNotFoundError(
            f"정책 checkpoint가 없습니다: {policy_path}\n"
            "먼저 ./launchfiles/train.bash config/system.yaml을 실행하세요."
        )

    run_dir, artifacts = create_run(
        settings.get("output_root", "data/inference_logs"),
        settings.get("run_name", "live_inference"),
    )
    resolved = snapshot_configs(config_path, native, artifacts)
    metrics = settings.get("metrics", {})
    child_command = [
        command_path("python"),
        str(child),
        "--config",
        str(resolved),
        "--metrics-output",
        str(artifacts / "latency.jsonl"),
        "--summary-output",
        str(artifacts / "latency_summary.json"),
        "--warmup-inferences",
        str(int(metrics.get("warmup_inferences", 3))),
    ]
    if bool(metrics.get("cuda_synchronize", True)):
        child_command.append("--cuda-synchronize")
    code = run_logged(child_command, run_dir, project_environment())
    if code == 0:
        update_latest(settings.get("output_root", "data/inference_logs"), run_dir)
        print(f"inference log: {run_dir}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
