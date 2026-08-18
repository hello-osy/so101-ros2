#!/usr/bin/env python3
"""Instrument real sync/RTC policy inference while the robot rollout is moving."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, TypeVar


T = TypeVar("T")


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
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
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--warmup-inferences", type=int, default=0)
    parser.add_argument("--cuda-synchronize", action="store_true")
    parser.add_argument(
        "--profile-iterations", type=int, default=0,
        help="0이면 종료할 때까지, 양수면 첫 N회 실제 추론만 summary/profile에 포함",
    )
    parser.add_argument("--cuda-profiler-api", action="store_true")
    parser.add_argument("--torch-profile-output")
    parser.add_argument("--torch-profile-table")
    parser.add_argument("--torch-profile-iterations", type=int, default=5)
    args = parser.parse_args()

    import torch
    from lerobot.rollout.inference.rtc import RTCInferenceEngine
    from lerobot.rollout.inference.sync import SyncInferenceEngine

    metrics_path = Path(args.metrics_output)
    summary_path = Path(args.summary_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    stream = metrics_path.open("w", encoding="utf-8", buffering=1)
    samples: list[float] = []
    records: list[dict] = []
    counter = 0
    profiler = None
    profiler_started = False
    profiler_finished = False
    cuda_profiler_started = False
    cuda_profiler_finished = False
    finished = False

    def stop_profilers() -> None:
        nonlocal profiler_finished, cuda_profiler_finished
        if profiler is not None and profiler_started and not profiler_finished:
            profiler.__exit__(None, None, None)
            profiler_finished = True
            profiler.export_chrome_trace(args.torch_profile_output)
            sort_by = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
            table = profiler.key_averages().table(sort_by=sort_by, row_limit=100)
            if args.torch_profile_table:
                Path(args.torch_profile_table).write_text(table + "\n", encoding="utf-8")
        if args.cuda_profiler_api and cuda_profiler_started and not cuda_profiler_finished:
            torch.cuda.cudart().cudaProfilerStop()
            cuda_profiler_finished = True

    def measured(operation: Callable[[], T], device_value: object, engine: str) -> T:
        nonlocal counter, profiler, profiler_started, cuda_profiler_started
        device = torch.device(device_value or "cpu")
        is_cuda = device.type == "cuda" and torch.cuda.is_available()
        in_range = (
            counter >= args.warmup_inferences
            and (
                args.profile_iterations <= 0
                or counter < args.warmup_inferences + args.profile_iterations
            )
        )
        profile_index = counter - args.warmup_inferences
        if in_range and args.cuda_profiler_api and not cuda_profiler_started:
            torch.cuda.cudart().cudaProfilerStart()
            cuda_profiler_started = True
        if in_range and args.torch_profile_output and not profiler_started:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if is_cuda:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            profiler = torch.profiler.profile(
                activities=activities, record_shapes=True, profile_memory=True
            )
            profiler.__enter__()
            profiler_started = True
        if is_cuda and args.cuda_synchronize:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        if is_cuda:
            torch.cuda.nvtx.range_push(f"so101_moving_robot_{engine}_policy_inference")
        started = time.perf_counter_ns()
        try:
            return operation()
        finally:
            if is_cuda and args.cuda_synchronize:
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            if is_cuda:
                torch.cuda.nvtx.range_pop()
            warmup = counter < args.warmup_inferences
            record = {
                "index": counter,
                "recorded_at": datetime.now().astimezone().isoformat(),
                "engine": engine,
                "latency_ms": elapsed_ms,
                "warmup": warmup,
                "profile_range": in_range,
                "robot_is_live": True,
                "cuda_synchronized": bool(is_cuda and args.cuda_synchronize),
                "process_rss_bytes": current_rss_bytes(),
                "system_memory_available_bytes": memory_available_bytes(),
                "cuda_memory": {
                    "allocated_bytes": torch.cuda.memory_allocated(device),
                    "reserved_bytes": torch.cuda.memory_reserved(device),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                } if is_cuda else None,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
            if not warmup and in_range:
                samples.append(elapsed_ms)
            if profiler is not None and profiler_started and not profiler_finished and in_range:
                profiler.step()
            counter += 1
            torch_limit = (
                min(args.profile_iterations, max(1, args.torch_profile_iterations))
                if args.profile_iterations > 0
                else max(1, args.torch_profile_iterations)
            )
            if (args.torch_profile_output and profile_index + 1 >= torch_limit) or (
                args.cuda_profiler_api and profile_index + 1 >= args.profile_iterations
            ):
                stop_profilers()

    original_sync_get_action = SyncInferenceEngine.get_action

    def measured_sync_get_action(self, obs_frame):
        return measured(lambda: original_sync_get_action(self, obs_frame), self._device, "sync")

    SyncInferenceEngine.get_action = measured_sync_get_action

    # RTC inference runs in its own thread. Wrapping get_action would only time
    # a queue pop, so wrap the actual policy chunk generation on each instance.
    original_rtc_init = RTCInferenceEngine.__init__

    def measured_rtc_init(self, *init_args, **init_kwargs):
        original_rtc_init(self, *init_args, **init_kwargs)
        original_predict = self._policy.predict_action_chunk

        def measured_predict(*predict_args, **predict_kwargs):
            return measured(
                lambda: original_predict(*predict_args, **predict_kwargs), self._device, "rtc"
            )

        self._policy.predict_action_chunk = measured_predict

    RTCInferenceEngine.__init__ = measured_rtc_init

    def finish() -> None:
        nonlocal finished
        if finished:
            return
        finished = True
        stop_profilers()
        cuda_rows = [row["cuda_memory"] for row in records if row["cuda_memory"]]
        rss_rows = [row["process_rss_bytes"] for row in records if row["process_rss_bytes"]]
        summary = {
            "measured_inferences": len(samples),
            "warmup_inferences": min(counter, args.warmup_inferences),
            "profile_iterations": len(samples),
            "live_robot_actions": True,
            "mean_ms": statistics.fmean(samples) if samples else None,
            "min_ms": min(samples) if samples else None,
            "p50_ms": percentile(samples, 0.50),
            "p95_ms": percentile(samples, 0.95),
            "p99_ms": percentile(samples, 0.99),
            "max_ms": max(samples) if samples else None,
            "mean_hz": 1000 / statistics.fmean(samples) if samples else None,
            "peak_cuda_allocated_bytes": max(
                (row["peak_allocated_bytes"] for row in cuda_rows), default=None
            ),
            "peak_cuda_reserved_bytes": max(
                (row["peak_reserved_bytes"] for row in cuda_rows), default=None
            ),
            "peak_process_rss_bytes": max(rss_rows, default=None),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        stream.close()

    atexit.register(finish)
    sys.argv = ["lerobot-rollout", f"--config_path={Path(args.config).resolve()}"]
    from lerobot.scripts.lerobot_rollout import main as rollout_main

    rollout_main()
    finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
