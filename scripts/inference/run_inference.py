#!/usr/bin/env python3
"""Run live SO-101 policy inference and persist per-decision latency."""

from __future__ import annotations

import argparse
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
    snapshot_configs,
    update_latest,
)


def build_native(config: dict) -> dict:
    native = dict(config.get("lerobot", {}))
    require_keys(native, "robot", "policy", "strategy", "inference", context="inference config")
    require_keys(native["policy"], "path", context="policy")
    native["robot"] = dict(native["robot"])
    if native["robot"].get("calibration_dir"):
        native["robot"]["calibration_dir"] = absolute_path(native["robot"]["calibration_dir"])
    native["policy"] = dict(native["policy"])
    native["policy"]["path"] = local_path_or_hub_id(str(native["policy"]["path"]))
    return native


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    require_keys(config, "project", "lerobot", "metrics", context="inference YAML")
    native = build_native(config)
    child = Path(__file__).with_name("inference_child.py")
    command = [command_path("python"), str(child), "--config", "RESOLVED_CONFIG"]
    if args.check:
        print_check("inference", native, command)
        return 0

    reject_placeholders(native)
    policy_path = native["policy"]["path"]
    if Path(policy_path).is_absolute() and not Path(policy_path).exists():
        raise FileNotFoundError(
            f"정책 checkpoint가 없습니다: {policy_path}\n먼저 ./launchfiles/train.bash를 실행하세요."
        )

    project = config["project"]
    run_dir, artifacts = create_run(
        project.get("output_root", "data/inference_logs"),
        project.get("run_name", "live_inference"),
    )
    resolved = snapshot_configs(config_path, native, artifacts)
    metrics = config["metrics"]
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
        update_latest(project.get("output_root", "data/inference_logs"), run_dir)
        print(f"inference log: {run_dir}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
