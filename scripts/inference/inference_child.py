#!/usr/bin/env python3
"""Instrument LeRobot's synchronous rollout engine without forking LeRobot."""

from __future__ import annotations

import argparse
import atexit
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
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
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--warmup-inferences", type=int, default=3)
    parser.add_argument("--cuda-synchronize", action="store_true")
    args = parser.parse_args()

    import torch
    from lerobot.rollout.inference.sync import SyncInferenceEngine

    metrics_path = Path(args.metrics_output)
    summary_path = Path(args.summary_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    stream = metrics_path.open("w", encoding="utf-8", buffering=1)
    samples: list[float] = []
    counter = 0
    original = SyncInferenceEngine.get_action

    def measured_get_action(self, obs_frame):
        nonlocal counter
        is_cuda = self._device.type == "cuda" and torch.cuda.is_available()
        if is_cuda and args.cuda_synchronize:
            torch.cuda.synchronize(self._device)
        if is_cuda:
            torch.cuda.nvtx.range_push("so101_live_select_action")
        started = time.perf_counter_ns()
        try:
            return original(self, obs_frame)
        finally:
            if is_cuda and args.cuda_synchronize:
                torch.cuda.synchronize(self._device)
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            if is_cuda:
                torch.cuda.nvtx.range_pop()
            warmup = counter < args.warmup_inferences
            record = {
                "index": counter,
                "recorded_at": datetime.now().astimezone().isoformat(),
                "latency_ms": elapsed_ms,
                "warmup": warmup,
                "cuda_synchronized": bool(is_cuda and args.cuda_synchronize),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if not warmup:
                samples.append(elapsed_ms)
            counter += 1

    def finish() -> None:
        summary = {
            "measured_inferences": len(samples),
            "warmup_inferences": min(counter, args.warmup_inferences),
            "mean_ms": statistics.fmean(samples) if samples else None,
            "min_ms": min(samples) if samples else None,
            "p50_ms": percentile(samples, 0.50),
            "p95_ms": percentile(samples, 0.95),
            "p99_ms": percentile(samples, 0.99),
            "max_ms": max(samples) if samples else None,
            "mean_hz": 1000 / statistics.fmean(samples) if samples else None,
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        stream.close()

    atexit.register(finish)
    SyncInferenceEngine.get_action = measured_get_action
    sys.argv = ["lerobot-rollout", f"--config_path={Path(args.config).resolve()}"]
    from lerobot.scripts.lerobot_rollout import main as rollout_main

    rollout_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
