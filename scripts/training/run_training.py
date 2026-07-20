#!/usr/bin/env python3
"""Run reproducible LeRobot training with a swappable model profile."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from project_utils import (  # noqa: E402
    absolute_path,
    command_path,
    create_run,
    deep_merge,
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
from model_registry import load_model_profile  # noqa: E402


def build_native(config: dict, model: dict, output_dir: str) -> dict:
    native = deep_merge(config.get("lerobot", {}), model)
    require_keys(native, "dataset", "policy", "peft", context="training config")
    require_keys(native["dataset"], "repo_id", context="dataset")
    require_keys(native["policy"], "path", context="policy")

    native["dataset"] = dict(native["dataset"])
    if native["dataset"].get("root"):
        native["dataset"]["root"] = absolute_path(native["dataset"]["root"])
    native["policy"] = dict(native["policy"])
    native["policy"]["path"] = local_path_or_hub_id(str(native["policy"]["path"]))
    native["output_dir"] = output_dir
    return native


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    require_keys(config, "project", "model_config", "lerobot", context="training YAML")
    model_path = Path(absolute_path(config["model_config"]))
    model = load_model_profile(model_path)
    native = build_native(config, model, absolute_path("data/training_outputs/CHECK_RUN/training"))
    command = [command_path("lerobot-train"), "--config_path=RESOLVED_CONFIG"]
    if args.check:
        print_check("training", native, command)
        return 0

    reject_placeholders(native)
    dataset_root = native["dataset"].get("root")
    if dataset_root and not Path(dataset_root).exists():
        raise FileNotFoundError(
            f"학습 dataset이 없습니다: {dataset_root}\n먼저 ./launchfiles/collect_data.bash를 실행하세요."
        )

    project = config["project"]
    run_dir, artifacts = create_run(
        project.get("output_root", "data/training_outputs"),
        project.get("run_name", "smolvla_lora"),
    )
    native = build_native(config, model, str(run_dir / "training"))
    resolved = snapshot_configs(config_path, native, artifacts, [model_path])
    code = run_logged(
        [command_path("lerobot-train"), f"--config_path={resolved}"],
        run_dir,
        project_environment(),
    )
    if code == 0:
        update_latest(project.get("output_root", "data/training_outputs"), run_dir)
        print(f"training output: {run_dir / 'training'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
