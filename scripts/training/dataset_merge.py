"""Discover and cache a merged LeRobot dataset for training."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from project_utils import PROJECT_ROOT, absolute_path


def _configured_patterns(config: dict[str, Any]) -> list[str]:
    dataset = config["dataset"]
    configured = dataset.get("training_roots")
    if configured is None:
        configured = [dataset["training_root"]]
    if not isinstance(configured, list) or not configured:
        raise ValueError("dataset.training_roots는 비어 있지 않은 경로 목록이어야 합니다.")
    if not all(isinstance(value, str) and value.strip() for value in configured):
        raise ValueError("dataset.training_roots의 각 항목은 비어 있지 않은 문자열이어야 합니다.")
    return configured


def discover_training_roots(config: dict[str, Any]) -> list[Path]:
    """Expand configured globs and remove aliases such as the ``latest`` symlink."""
    roots: list[Path] = []
    seen: set[Path] = set()
    for pattern in _configured_patterns(config):
        expanded = Path(pattern).expanduser()
        if not expanded.is_absolute():
            expanded = PROJECT_ROOT / expanded
        matches = sorted(Path(value) for value in glob.glob(str(expanded)))
        if not matches and not glob.has_magic(str(expanded)):
            matches = [expanded]
        for match in matches:
            canonical = match.resolve()
            if canonical not in seen:
                roots.append(canonical)
                seen.add(canonical)
    if not roots:
        raise FileNotFoundError(
            "dataset.training_roots와 일치하는 데이터셋이 없습니다: "
            + ", ".join(_configured_patterns(config))
        )
    return roots


def usable_training_roots(config: dict[str, Any]) -> tuple[list[Path], list[Path]]:
    """Return finalized, non-empty roots and the empty roots that were skipped."""
    usable: list[Path] = []
    skipped: list[Path] = []
    for root in discover_training_roots(config):
        info_path = root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"LeRobot dataset meta/info.json이 없습니다: {root}")
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            total_episodes = int(info["total_episodes"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"유효하지 않은 LeRobot dataset metadata입니다: {info_path}") from exc
        if total_episodes <= 0:
            skipped.append(root)
        else:
            usable.append(root)
    if not usable:
        raise ValueError("학습 가능한 episode가 있는 데이터셋이 하나도 없습니다.")
    holdout = configured_eval_holdout_root(config, usable)
    if holdout is not None:
        if holdout not in usable:
            raise ValueError(
                "dataset.eval_holdout_root는 비어 있지 않은 training_roots 중 하나여야 합니다: "
                f"{holdout}"
            )
        # LeRobot holds out the final episodes. Keep one complete collection run
        # at the end so no session is split across train and evaluation.
        usable = [root for root in usable if root != holdout] + [holdout]
    return usable, skipped


def configured_eval_holdout_root(config: dict[str, Any], roots: list[Path] | None = None) -> Path | None:
    configured = config["dataset"].get("eval_holdout_root")
    if configured is None:
        return None
    if not isinstance(configured, str) or not configured.strip():
        raise ValueError("dataset.eval_holdout_root는 비어 있지 않은 경로여야 합니다.")
    if configured == "auto_latest":
        if not roots:
            raise ValueError("auto_latest eval holdout을 선택할 유효한 dataset이 없습니다.")
        # Collection run directory names begin with sortable timestamps.
        return max(roots, key=lambda root: root.parent.name)
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def holdout_eval_split(config: dict[str, Any], roots: list[Path]) -> float | None:
    """Return the exact episode fraction for the configured whole-run holdout."""
    holdout = configured_eval_holdout_root(config, roots)
    if holdout is None:
        return None
    if not roots or roots[-1] != holdout:
        raise ValueError("eval holdout dataset은 병합 순서의 마지막이어야 합니다.")
    episode_counts = [
        int(json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))["total_episodes"])
        for root in roots
    ]
    return episode_counts[-1] / sum(episode_counts)


def _fingerprint(roots: list[Path]) -> str:
    digest = hashlib.sha256()
    for root in roots:
        digest.update(str(root).encode())
        for path in sorted((root / "meta").rglob("*")):
            if path.is_file():
                digest.update(str(path.relative_to(root)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def planned_training_root(config: dict[str, Any], roots: list[Path]) -> Path:
    if len(roots) == 1:
        return roots[0]
    cache_root = Path(absolute_path(config["dataset"].get("training_merged_root", "data/merged_datasets")))
    return cache_root / _fingerprint(roots) / "dataset"


def prepare_training_dataset(config: dict[str, Any], roots: list[Path]) -> Path:
    """Return one dataset root, merging multiple roots into a reusable cache when needed."""
    target = planned_training_root(config, roots)
    if len(roots) == 1:
        return target

    expected_episodes = sum(
        int(json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))["total_episodes"])
        for root in roots
    )
    target_info = target / "meta" / "info.json"
    if target_info.is_file():
        cached = json.loads(target_info.read_text(encoding="utf-8"))
        if int(cached.get("total_episodes", -1)) == expected_episodes:
            print(f"[training] cached merged dataset: {target}")
            return target
        raise ValueError(f"병합 dataset cache가 불완전합니다: {target}")

    from lerobot.datasets.aggregate import aggregate_datasets

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".dataset.tmp-{os.getpid()}"
    repo_id = str(config["dataset"]["repo_id"])
    source_repo_ids = [f"{repo_id}_part_{index:03d}" for index in range(len(roots))]
    print(f"[training] merging {len(roots)} datasets ({expected_episodes} episodes) -> {target}")
    try:
        aggregate_datasets(
            repo_ids=source_repo_ids,
            aggr_repo_id=f"{repo_id}_merged",
            roots=roots,
            aggr_root=temporary,
            concatenate_videos=False,
            concatenate_data=False,
        )
        temporary.replace(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target
