from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "data_collection"))

import collect_and_sync  # noqa: E402
import sync_artifacts  # noqa: E402


class CollectAndSyncTest(unittest.TestCase):
    @patch.object(collect_and_sync.subprocess, "Popen")
    @patch.object(collect_and_sync, "sync", side_effect=[0, 0, 0])
    def test_collector_inherits_foreground_terminal(self, sync_mock: Mock, popen_mock: Mock):
        child = popen_mock.return_value
        child.wait.side_effect = [subprocess.TimeoutExpired("collector", 30), 0]

        code = collect_and_sync.collect_and_sync(Path("/tmp/system.yaml"), 30)

        self.assertEqual(code, 0)
        self.assertEqual(sync_mock.call_count, 3)
        _, kwargs = popen_mock.call_args
        self.assertNotIn("start_new_session", kwargs)
        self.assertNotIn("stdin", kwargs)
        self.assertEqual(child.wait.call_args_list, [call(timeout=30), call(timeout=30)])

    @patch.object(collect_and_sync.subprocess, "Popen")
    @patch.object(collect_and_sync, "sync", side_effect=[0, 0])
    def test_ctrl_c_waits_for_collector_cleanup_without_resending_signal(
        self, sync_mock: Mock, popen_mock: Mock
    ):
        child = popen_mock.return_value
        child.wait.side_effect = [KeyboardInterrupt, 0]

        code = collect_and_sync.collect_and_sync(Path("/tmp/system.yaml"), 30)

        self.assertEqual(code, 0)
        self.assertEqual(sync_mock.call_count, 2)
        child.send_signal.assert_not_called()
        child.terminate.assert_not_called()

    @patch.object(sync_artifacts.subprocess, "run")
    def test_rsync_uses_noninteractive_bounded_ssh(self, run_mock: Mock):
        run_mock.return_value.returncode = 0

        code = sync_artifacts.rsync("source/", "user@host:/target/", 22, check=False)

        self.assertEqual(code, 0)
        command = run_mock.call_args.args[0]
        ssh_command = command[command.index("-e") + 1]
        self.assertIn("BatchMode=yes", ssh_command)
        self.assertIn("ConnectTimeout=5", ssh_command)
        self.assertIn("ServerAliveCountMax=1", ssh_command)


if __name__ == "__main__":
    unittest.main()
