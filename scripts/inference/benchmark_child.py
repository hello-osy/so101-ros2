#!/usr/bin/env python3
"""Load one LeRobotDataset sample and repeatedly benchmark a real policy forward."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
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


def current_rss_bytes() -> int | None:
    try:
        pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def memory_available_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--torch-profile-output")
    parser.add_argument("--torch-profile-table")
    parser.add_argument("--torch-profile-iterations", type=int, default=5)
    parser.add_argument("--cuda-profiler-api", action="store_true")
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
            torch.cuda.reset_peak_memory_stats()
        if device.startswith("cuda"):
            torch.cuda.nvtx.range_push("so101_offline_policy_forward")
            events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            events[0].record()
        started = time.perf_counter_ns()
        autocast = (
            torch.autocast(device_type="cuda")
            if device.startswith("cuda") and bool(cfg["policy"].get("use_amp", False))
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast:
            processed = preprocessor(batch)
            if device.startswith("cuda"):
                events[1].record()
            action = policy.select_action(processed)
            if device.startswith("cuda"):
                events[2].record()
            postprocessor(action)
        if device.startswith("cuda"):
            events[3].record()
        if device.startswith("cuda") and synchronize:
            torch.cuda.synchronize()
        end_to_end_ms = (time.perf_counter_ns() - started) / 1_000_000
        cuda_stages = None
        if device.startswith("cuda") and synchronize:
            cuda_stages = {
                "preprocess_ms": events[0].elapsed_time(events[1]),
                "policy_ms": events[1].elapsed_time(events[2]),
                "postprocess_ms": events[2].elapsed_time(events[3]),
                "total_ms": events[0].elapsed_time(events[3]),
            }
        if device.startswith("cuda"):
            torch.cuda.nvtx.range_pop()
        cuda_memory = None
        if device.startswith("cuda"):
            cuda_memory = {
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        records.append(
            {
                "index": index,
                "warmup": is_warmup,
                "end_to_end_ms": end_to_end_ms,
                "cuda_ms": cuda_stages["total_ms"] if cuda_stages else None,
                "cuda_stages": cuda_stages,
                "cuda_memory": cuda_memory,
                "process_rss_bytes": current_rss_bytes(),
                "system_memory_available_bytes": memory_available_bytes(),
            }
        )

    for index in range(warmup):
        one(index, True)
    if args.cuda_profiler_api:
        torch.cuda.cudart().cudaProfilerStart()
    try:
        if args.torch_profile_output:
            profile_iterations = min(iterations, max(1, args.torch_profile_iterations))
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.startswith("cuda"):
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
            ) as profiler:
                for index in range(profile_iterations):
                    one(index, False)
                    profiler.step()
            profiler.export_chrome_trace(args.torch_profile_output)
            sort_by = "self_cuda_time_total" if device.startswith("cuda") else "self_cpu_time_total"
            table = profiler.key_averages().table(sort_by=sort_by, row_limit=100)
            if args.torch_profile_table:
                Path(args.torch_profile_table).write_text(table + "\n", encoding="utf-8")
        else:
            for index in range(iterations):
                one(index, False)
    finally:
        if args.cuda_profiler_api:
            torch.cuda.cudart().cudaProfilerStop()

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
        "measured_iterations": len(measured),
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
            "peak_cuda_allocated_bytes": max(
                (row["cuda_memory"]["peak_allocated_bytes"] for row in records if row["cuda_memory"]),
                default=None,
            ),
            "peak_cuda_reserved_bytes": max(
                (row["cuda_memory"]["peak_reserved_bytes"] for row in records if row["cuda_memory"]),
                default=None,
            ),
            "peak_process_rss_bytes": max(
                (row["process_rss_bytes"] for row in records if row["process_rss_bytes"] is not None),
                default=None,
            ),
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
