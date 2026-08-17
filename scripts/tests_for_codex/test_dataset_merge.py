from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts" / "training")]

from dataset_merge import (  # noqa: E402
    discover_training_roots,
    holdout_eval_split,
    planned_training_root,
    prepare_training_dataset,
    usable_training_roots,
)


class DatasetMergeConfigTest(unittest.TestCase):
    @staticmethod
    def _dataset(root: Path, episodes: int) -> Path:
        dataset = root / "dataset"
        (dataset / "meta").mkdir(parents=True)
        (dataset / "meta" / "info.json").write_text(
            json.dumps({"total_episodes": episodes, "total_frames": episodes * 10}),
            encoding="utf-8",
        )
        return dataset

    def test_glob_deduplicates_latest_and_skips_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._dataset(root / "run-a", 3)
            second = self._dataset(root / "run-b", 2)
            self._dataset(root / "empty", 0)
            (root / "latest").symlink_to(root / "run-b", target_is_directory=True)
            config = {
                "dataset": {
                    "training_root": str(first),
                    "training_roots": [str(root / "*" / "dataset")],
                    "training_merged_root": str(root / "merged"),
                }
            }

            discovered = discover_training_roots(config)
            self.assertEqual(len(discovered), 3)
            usable, skipped = usable_training_roots(config)
            self.assertEqual(set(usable), {first.resolve(), second.resolve()})
            self.assertEqual(len(skipped), 1)
            self.assertTrue(str(planned_training_root(config, usable)).endswith("/dataset"))

    def test_single_root_needs_no_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory) / "run", 1)
            config = {"dataset": {"training_root": str(dataset)}}
            usable, _ = usable_training_roots(config)
            self.assertEqual(planned_training_root(config, usable), dataset.resolve())

    def test_whole_run_holdout_is_merged_last_with_exact_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            holdout = self._dataset(root / "run-a", 2)
            training = self._dataset(root / "run-b", 8)
            config = {
                "dataset": {
                    "training_root": str(training),
                    "training_roots": [str(holdout), str(training)],
                    "eval_holdout_root": str(holdout),
                }
            }
            usable, _ = usable_training_roots(config)
            self.assertEqual(usable, [training.resolve(), holdout.resolve()])
            self.assertAlmostEqual(holdout_eval_split(config, usable), 0.2)

    def test_auto_latest_holdout_survives_collection_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            older = self._dataset(root / "20260817_170901_run", 5)
            latest = self._dataset(root / "20260817_210543_run", 5)
            config = {
                "dataset": {
                    "training_root": str(older),
                    "training_roots": [str(latest), str(older)],
                    "eval_holdout_root": "auto_latest",
                }
            }
            usable, _ = usable_training_roots(config)
            self.assertEqual(usable, [older.resolve(), latest.resolve()])
            self.assertAlmostEqual(holdout_eval_split(config, usable), 0.5)

    def test_multiple_roots_are_merged_once_and_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._dataset(root / "run-a", 3)
            second = self._dataset(root / "run-b", 2)
            config = {
                "dataset": {
                    "repo_id": "local/test",
                    "training_root": str(first),
                    "training_roots": [str(first), str(second)],
                    "training_merged_root": str(root / "merged"),
                }
            }
            usable, _ = usable_training_roots(config)

            def fake_aggregate(**kwargs):
                output = Path(kwargs["aggr_root"])
                (output / "meta").mkdir(parents=True)
                (output / "meta" / "info.json").write_text(
                    json.dumps({"total_episodes": 5, "total_frames": 50}),
                    encoding="utf-8",
                )

            with patch(
                "lerobot.datasets.aggregate.aggregate_datasets", side_effect=fake_aggregate
            ) as aggregate:
                merged = prepare_training_dataset(config, usable)
                self.assertEqual(aggregate.call_count, 1)
                self.assertTrue((merged / "meta" / "info.json").is_file())
                self.assertEqual(prepare_training_dataset(config, usable), merged)
                self.assertEqual(aggregate.call_count, 1)


if __name__ == "__main__":
    unittest.main()
