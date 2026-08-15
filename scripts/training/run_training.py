#!/usr/bin/env python3
"""Run reproducible LeRobot training from the unified system YAML."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_utils import (  # noqa: E402
    absolute_path,
    command_path,
    create_run,
    print_check,
    project_environment,
    reject_placeholders,
    run_logged,
    snapshot_configs,
    update_latest,
)
from system_config import load_system, run_settings, training_config  # noqa: E402
from dataset_merge import (  # noqa: E402
    planned_training_root,
    prepare_training_dataset,
    usable_training_roots,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="통합 system YAML")
    parser.add_argument("--profile", default="training", help="runs 아래의 학습 프로필")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_system(config_path)
    settings = run_settings(config, args.profile)
    dataset_roots, skipped_roots = usable_training_roots(config)
    planned_root = planned_training_root(config, dataset_roots)
    native = training_config(
        config,
        absolute_path("data/training_outputs/CHECK_RUN/training"),
        args.profile,
        dataset_root=planned_root,
    )
    command = [command_path("lerobot-train"), "--config_path=RESOLVED_CONFIG"]
    if args.check:
        print(f"[training] source datasets: {len(dataset_roots)}")
        for root in dataset_roots:
            print(f"  - {root}")
        for root in skipped_roots:
            print(f"  - skipped empty dataset: {root}")
        print_check("training", native, command)
        return 0

    reject_placeholders(native)
    policy_path = native["policy"]["path"]
    if Path(policy_path).is_absolute() and not Path(policy_path, "config.json").exists():
        raise FileNotFoundError(
            f"SmolVLA checkpoint가 없습니다: {policy_path}\n"
            "먼저 ./launchfiles/download_models.bash config/system.yaml을 실행하세요."
        )

    run_dir, artifacts = create_run(
        settings.get("output_root", "data/training_outputs"),
        settings.get("run_name", "smolvla_lora"),
    )
    for root in skipped_roots:
        print(f"[training] skipped empty dataset: {root}")
    dataset_root = prepare_training_dataset(config, dataset_roots)
    native = training_config(
        config,
        str(run_dir / "training"),
        args.profile,
        dataset_root=dataset_root,
    )
    resolved = snapshot_configs(config_path, native, artifacts)
    code = run_logged(
        [command_path("lerobot-train"), f"--config_path={resolved}"],
        run_dir,
        project_environment(),
    )
    if code == 0:
        update_latest(settings.get("output_root", "data/training_outputs"), run_dir)
        print(f"training output: {run_dir / 'training'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
