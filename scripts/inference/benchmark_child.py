#!/usr/bin/env python3
"""Load one LeRobotDataset sample and repeatedly benchmark a real policy forward."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import yaml


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percent
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    fraction = index - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--metrics-output", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = config["benchmark"]

    import torch
    from torch.utils.data._utils.collate import default_collate

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    device = str(cfg.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark를 요청했지만 torch.cuda.is_available()이 False입니다.")

    dataset_cfg = cfg["dataset"]
    repo_id = str(dataset_cfg["repo_id"])
    root = str(dataset_cfg["root"])
    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    dataset = LeRobotDataset(repo_id, root=root)
    sample_index = int(cfg.get("sample_index", 0))
    if not 0 <= sample_index < len(dataset):
        raise IndexError(f"sample_index={sample_index}, dataset length={len(dataset)}")
    batch = default_collate([dataset[sample_index]])

    policy_path = str(cfg["policy"]["path"])
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = Path(policy_path)
    policy_cfg.device = device
    if "use_amp" in cfg["policy"]:
        policy_cfg.use_amp = bool(cfg["policy"]["use_amp"])
    policy = make_policy(policy_cfg, ds_meta=metadata)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg,
        pretrained_path=policy_path,
        dataset_stats=metadata.stats,
        dataset_meta=metadata,
    )
    policy.eval()

    warmup = int(cfg.get("warmup_inferences", 3))
    iterations = int(cfg.get("iterations", 20))
    synchronize = bool(cfg.get("cuda_synchronize", True))
    records: list[dict] = []

    def one(index: int, is_warmup: bool) -> None:
        # reset() empties action queues, so every sample measures an actual model forward.
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        if device.startswith("cuda") and synchronize:
            torch.cuda.synchronize()
        if device.startswith("cuda"):
            torch.cuda.nvtx.range_push("so101_offline_policy_forward")
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        started = time.perf_counter_ns()
        with torch.inference_mode():
            processed = preprocessor(batch)
            action = policy.select_action(processed)
            postprocessor(action)
        if device.startswith("cuda"):
            end_event.record()
        if device.startswith("cuda") and synchronize:
            torch.cuda.synchronize()
        end_to_end_ms = (time.perf_counter_ns() - started) / 1_000_000
        cuda_ms = start_event.elapsed_time(end_event) if device.startswith("cuda") and synchronize else None
        if device.startswith("cuda"):
            torch.cuda.nvtx.range_pop()
        records.append(
            {
                "index": index,
                "warmup": is_warmup,
                "end_to_end_ms": end_to_end_ms,
                "cuda_ms": cuda_ms,
            }
        )

    for index in range(warmup):
        one(index, True)
    for index in range(iterations):
        one(index, False)

    measured = [row["end_to_end_ms"] for row in records if not row["warmup"]]
    cuda_values = [row["cuda_ms"] for row in records if not row["warmup"] and row["cuda_ms"]]
    result = {
        "recorded_at": datetime.now().astimezone().isoformat(),
        "dataset_repo_id": repo_id,
        "dataset_root": root,
        "sample_index": sample_index,
        "policy_path": policy_path,
        "device": device,
        "iterations": iterations,
        "warmup_inferences": warmup,
        "summary": {
            "mean_ms": statistics.fmean(measured),
            "min_ms": min(measured),
            "p50_ms": percentile(measured, 0.50),
            "p95_ms": percentile(measured, 0.95),
            "p99_ms": percentile(measured, 0.99),
            "max_ms": max(measured),
            "mean_hz": 1000 / statistics.fmean(measured),
            "mean_cuda_ms": statistics.fmean(cuda_values) if cuda_values else None,
        },
        "samples": records,
    }
    Path(args.metrics_output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
