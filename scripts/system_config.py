"""Translate the single user-facing system YAML into LeRobot configs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from project_utils import absolute_path, deep_merge, load_yaml, local_path_or_hub_id, require_keys


def load_system(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    require_keys(
        config, "system", "transfer", "devices", "dataset", "model", "runs", context="system YAML"
    )
    require_keys(
        config["system"],
        "torch",
        "torchvision",
        "cuda",
        "lerobot_revision",
        context="system",
    )
    require_keys(config["devices"], "follower", "leader", "cameras", context="devices")
    require_keys(
        config["dataset"],
        "format",
        "repo_id",
        "storage_root",
        "training_root",
        "task",
        context="dataset",
    )
    require_keys(config["model"], "base", "vlm", "trained_policy_path", "peft", context="model")
    for name in ("base", "vlm"):
        require_keys(config["model"][name], "repo_id", "revision", "path", context=f"model.{name}")
    for name in (
        "camera_viewer",
        "calibration",
        "teleoperation",
        "collection",
        "training",
        "training_desktop",
        "inference",
        "benchmark",
        "profiling",
    ):
        require_keys(config["runs"], name, context="runs")
    return config


def run_settings(config: dict, name: str) -> dict:
    runs = config["runs"]
    require_keys(runs, name, context="runs")
    return deepcopy(runs[name])


def device(config: dict, name: str, *, with_cameras: bool = False) -> dict:
    value = deepcopy(config["devices"][name])
    require_keys(value, "type", "port", "id", "calibration_dir", context=f"devices.{name}")
    value["calibration_dir"] = absolute_path(value["calibration_dir"])
    if with_cameras:
        value["cameras"] = deepcopy(config["devices"]["cameras"])
    return value


def calibration_configs(config: dict) -> list[tuple[str, dict]]:
    settings = run_settings(config, "calibration")
    target = settings.get("target", "both")
    if target not in {"follower", "leader", "both"}:
        raise ValueError("runs.calibration.target은 follower, leader, both 중 하나여야 합니다.")
    result: list[tuple[str, dict]] = []
    if target in {"follower", "both"}:
        follower = device(config, "follower")
        follower["cameras"] = {}
        result.append(("follower", {"robot": follower}))
    if target in {"leader", "both"}:
        result.append(("leader", {"teleop": device(config, "leader")}))
    return result


def teleoperation_config(config: dict) -> dict:
    settings = run_settings(config, "teleoperation")
    display_data = bool(settings.get("display_data", True))
    return {
        "robot": device(config, "follower", with_cameras=display_data),
        "teleop": device(config, "leader"),
        "fps": int(settings.get("fps", config["dataset"].get("fps", 30))),
        "teleop_time_s": settings.get("teleop_time_s"),
        "display_data": display_data,
        "display_mode": str(settings.get("display_mode", "rerun")),
    }


def collection_config(config: dict, dataset_root: str) -> dict:
    dataset = deepcopy(config["dataset"])
    if dataset.pop("format", None) != "lerobot_v3":
        raise ValueError("dataset.format은 lerobot_v3여야 합니다.")
    dataset.pop("storage_root", None)
    dataset.pop("training_root", None)
    dataset.pop("training_roots", None)
    dataset.pop("training_merged_root", None)
    dataset.pop("eval_holdout_root", None)
    dataset["single_task"] = dataset.pop("task")
    dataset["root"] = dataset_root
    dataset["push_to_hub"] = bool(dataset.get("push_to_hub", False))
    settings = run_settings(config, "collection")
    settings.pop("run_name", None)
    # Project-only launcher behavior; not part of LeRobot's RecordConfig.
    settings.pop("show_clamp_warnings", None)
    return {
        "robot": device(config, "follower", with_cameras=True),
        "teleop": device(config, "leader"),
        "dataset": dataset,
        **settings,
    }


def training_config(
    config: dict,
    output_dir: str,
    profile: str = "training",
    *,
    dataset_root: str | Path | None = None,
    eval_split: float | None = None,
) -> dict:
    dataset = config["dataset"]
    model = config["model"]
    settings = run_settings(config, profile)
    for key in ("run_name", "output_root"):
        settings.pop(key, None)
    dataset_overrides = settings.pop("dataset", {})
    policy_overrides = settings.pop("policy", {})
    peft = settings.pop("peft", deepcopy(model["peft"]))
    native_dataset = deep_merge(
        {
            "repo_id": dataset["repo_id"],
            "root": absolute_path(dataset_root or dataset["training_root"]),
            "use_imagenet_stats": True,
            "return_uint8": True,
            "eval_split": 0.0,
        },
        dataset_overrides,
    )
    if eval_split is not None:
        native_dataset["eval_split"] = eval_split
    native_policy = deep_merge(
        {
            "path": local_path_or_hub_id(str(model["base"]["path"])),
            "input_features": None,
            "output_features": None,
            "device": model.get("device", "cuda"),
            "use_amp": bool(model.get("use_amp", True)),
            "vlm_model_name": local_path_or_hub_id(str(model["vlm"]["path"])),
            "load_vlm_weights": bool(model.get("load_vlm_weights", False)),
            "push_to_hub": bool(model.get("push_to_hub", False)),
        },
        policy_overrides,
    )
    native = {
        "dataset": native_dataset,
        "policy": native_policy,
        "peft": peft,
        "output_dir": output_dir,
        **settings,
    }
    if str(native["peft"].get("method_type", "")).upper() != "LORA":
        raise ValueError("model.peft.method_type은 LORA여야 합니다.")
    return native


def inference_config(config: dict) -> dict:
    dataset = config["dataset"]
    model = config["model"]
    settings = run_settings(config, "inference")
    for key in ("run_name", "output_root", "metrics"):
        settings.pop(key, None)
    policy_overrides = settings.pop("policy", {})
    native_policy = deep_merge(
        {
            "path": local_path_or_hub_id(str(model["trained_policy_path"])),
            "device": model.get("device", "cuda"),
            "use_amp": bool(model.get("use_amp", True)),
        },
        policy_overrides,
    )
    return {
        "robot": device(config, "follower", with_cameras=True),
        "policy": native_policy,
        "device": model.get("device", "cuda"),
        "task": dataset["task"],
        "fps": int(dataset.get("fps", 30)),
        **settings,
    }


def benchmark_config(config: dict) -> dict:
    settings = run_settings(config, "benchmark")
    return {
        "project": run_settings(config, "benchmark"),
        "benchmark": settings,
        "inference": inference_config(config),
        "profiling": run_settings(config, "profiling"),
    }
