import asyncio
import json
import logging
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from config import PPOConfig
from train import run_training_battle
from training_errors import TrainingDiagnosticsMixin, TrainingErrorRecorder, _ClientErrorHandler


class RecorderIOTests(unittest.TestCase):
    def test_open_write_and_close_failures_preserve_record_in_fallback(self):
        for failure_at in ("open", "write", "close", "mkdir"):
            with self.subTest(failure_at=failure_at), TemporaryDirectory() as directory:
                primary = Path(directory) / "primary.jsonl"
                fallback = Path(directory) / "fallback.jsonl"
                recorder = TrainingErrorRecorder(primary, fallback_path=fallback)
                real_open, real_mkdir = Path.open, Path.mkdir
                attempted = []

                class BadStream:
                    def __enter__(self):
                        return self

                    def write(self, text):
                        if failure_at == "write":
                            raise OSError(22, "Invalid argument")

                    def __exit__(self, *args):
                        if failure_at == "close":
                            raise OSError(22, "Invalid argument")

                def faulty_open(path, *args, **kwargs):
                    attempted.append(path)
                    if path == primary:
                        if failure_at == "open":
                            raise OSError(22, "Invalid argument")
                        return BadStream()
                    return real_open(path, *args, **kwargs)

                mkdir_calls = []

                def faulty_mkdir(path, *args, **kwargs):
                    mkdir_calls.append(path)
                    if failure_at == "mkdir" and len(mkdir_calls) == 1:
                        raise OSError(22, "Invalid argument")
                    return real_mkdir(path, *args, **kwargs)

                with patch.object(Path, "open", faulty_open), patch.object(Path, "mkdir", faulty_mkdir), patch("training_errors.sys.stderr"):
                    self.assertTrue(recorder.record(None, None, "team_validation_rejected", "original rejection",
                                                    selected_moves=["thunderbolt"]))
                    self.assertTrue(recorder.record(None, None, "second_event", "second"))
                rows = [json.loads(line) for line in fallback.read_text(encoding="utf-8").splitlines()]
                self.assertEqual([r["kind"] for r in rows], ["team_validation_rejected", "second_event"])
                self.assertEqual(rows[0]["error"], "original rejection")
                self.assertEqual(rows[0]["candidate_moves"], ["thunderbolt"])
                self.assertEqual(recorder.active_path, fallback)
                self.assertLessEqual(attempted.count(primary), 1)

    def test_both_paths_fail_then_buffered_records_are_written_on_recovery(self):
        with TemporaryDirectory() as directory:
            recorder = TrainingErrorRecorder(Path(directory) / "primary.jsonl",
                                             fallback_path=Path(directory) / "fallback.jsonl")
            with patch.object(Path, "open", side_effect=OSError(22, "Invalid argument")), patch("training_errors.sys.stderr"):
                self.assertFalse(recorder.record(None, None, "first", "first error"))
            self.assertTrue(recorder.record(None, None, "second", "second error"))
            rows = [json.loads(line) for line in recorder.active_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["kind"] for r in rows], ["first", "second"])
            self.assertNotEqual(rows[0]["event_id"], rows[1]["event_id"])

    def test_buffer_is_bounded_and_broken_stderr_does_not_raise(self):
        with TemporaryDirectory() as directory:
            recorder = TrainingErrorRecorder(Path(directory) / "primary.jsonl",
                                             fallback_path=Path(directory) / "fallback.jsonl")
            with patch.object(Path, "open", side_effect=OSError(22, "Invalid argument")), patch("training_errors.sys.stderr") as stderr:
                stderr.write.side_effect = OSError(22, "Invalid argument")
                for i in range(101):
                    self.assertFalse(recorder.record(None, None, "test", str(i)))
            self.assertEqual(len(recorder._pending), 100)
            self.assertEqual(recorder.dropped_records, 1)


class FakeBase:
    async def _handle_ots_request(self, tag):
        self.sheet_replies = getattr(self, "sheet_replies", 0) + 1

    async def _handle_battle_message(self, messages):
        self.handled_messages = getattr(self, "handled_messages", 0) + 1
        if self.fail:
            raise ValueError("simulated parser failure")


class FakePlayer(TrainingDiagnosticsMixin, FakeBase):
    def __init__(self, recorder):
        self.username = "diagnostic-test"
        self.error_recorder = recorder
        self.training_error = threading.Event()
        self._protocol_tail = {}
        self._selected_orders = {"battle-test": (["protect"], "/choose move 1, move 2")}
        self.battles = {"battle-test": SimpleNamespace(battle_tag="battle-test", turn=4,
                                                     team={}, opponent_team={}, finished=False)}
        self.fail = False


class DiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    async def test_logging_io_failure_cannot_mask_original_battle_exception(self):
        with TemporaryDirectory() as directory:
            recorder = TrainingErrorRecorder(Path(directory) / "primary.jsonl",
                                             fallback_path=Path(directory) / "fallback.jsonl")
            player = FakePlayer(recorder)
            player.fail = True
            with patch.object(Path, "open", side_effect=OSError(22, "Invalid argument")), patch("training_errors.sys.stderr"):
                with self.assertRaisesRegex(ValueError, "simulated parser failure"):
                    await player._handle_battle_message([[">battle-test"], ["", "turn", "5"]])
            self.assertTrue(player.training_error.is_set())

    async def test_late_team_sheet_notice_does_not_enter_fatal_error_handler(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "errors.jsonl"
            player = FakePlayer(TrainingErrorRecorder(path))
            await player._handle_battle_message([
                [">battle-test"], ["", "error", "You cannot reject open team sheets after Team Preview."]])
            self.assertFalse(player.training_error.is_set())
            self.assertEqual(getattr(player, "handled_messages", 0), 0)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["kind"], "late_team_sheet_reply")

    async def test_team_sheet_reply_only_sent_before_battle_starts(self):
        player = FakePlayer(Mock())
        await player._handle_ots_request("battle-test")
        self.assertEqual(getattr(player, "sheet_replies", 0), 0)
        player.battles["battle-test"].turn = 0
        await player._handle_ots_request("battle-test")
        self.assertEqual(player.sheet_replies, 1)

    async def test_repeated_server_errors_are_deduplicated_and_bounded(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "errors.jsonl"
            player = FakePlayer(TrainingErrorRecorder(path))
            for _ in range(100):
                await player._handle_battle_message([
                    [">battle-test"], ["", "error", "[Invalid choice] Can't move: Outrage needs a target"]])
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["kind"] for r in rows], ["server_error", "server_error_retry_limit"])
            self.assertTrue(player.training_error.is_set())
            self.assertEqual(player.handled_messages, 2)

    async def test_new_request_in_same_turn_resets_error_budget(self):
        with TemporaryDirectory() as directory:
            player = FakePlayer(TrainingErrorRecorder(Path(directory) / "errors.jsonl"))
            for request_id in range(4):
                player.battles["battle-test"]._last_request = {"rqid": request_id}
                await player._handle_battle_message([
                    [">battle-test"], ["", "error", "[Invalid choice] test"]])
            self.assertFalse(player.training_error.is_set())
            self.assertEqual(player.handled_messages, 4)

    def test_specific_move_failure_keeps_move_id_even_without_protocol(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "errors.jsonl"
            player = FakePlayer(TrainingErrorRecorder(path))
            player.record_training_error(player.battles["battle-test"], "move_score_exception",
                                         ValueError("invalid move data"), related_moves=["thunderbolt"])
            row = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(row["related_moves"], ["thunderbolt"])
            self.assertIn("thunderbolt", row["candidate_moves"])

    async def test_protocol_exception_records_moves_and_traceback_and_sets_failure(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "errors.jsonl"
            player = FakePlayer(TrainingErrorRecorder(path))
            player.fail = True
            with self.assertRaises(ValueError):
                await player._handle_battle_message([
                    [">battle-test"], ["", "move", "p2a: Pikachu", "Thunderbolt", "p1a: target"]])
            row = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(row["candidate_moves"], ["protect", "thunderbolt"])
            self.assertFalse(row["cause_confirmed"])
            self.assertEqual(row["battle_tag"], "battle-test")
            self.assertEqual(row["turn"], 4)
            self.assertIn("ValueError", row["traceback"])
            self.assertTrue(player.training_error.is_set())

    async def test_server_error_is_recorded_without_crashing_recovery(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "errors.jsonl"
            player = FakePlayer(TrainingErrorRecorder(path))
            await player._handle_battle_message([
                [">battle-test"], ["", "error", "[Invalid choice] test"]])
            row = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(row["kind"], "server_error")
            self.assertFalse(player.training_error.is_set())

    async def test_battle_timeout_cancels_wait_and_records_error(self):
        async def wait_forever(*args, **kwargs):
            await asyncio.Event().wait()

        player = SimpleNamespace(battle_against=wait_forever, training_error=threading.Event(),
                                 battles={}, record_training_error=Mock())
        opponent = SimpleNamespace(training_error=threading.Event(), battles={}, record_training_error=Mock())
        with self.assertRaises(TimeoutError):
            await run_training_battle(player, opponent, PPOConfig(battle_timeout_seconds=0.01))
        self.assertEqual(player.record_training_error.call_args.args[1], "training_battle_aborted")
        opponent.record_training_error.assert_called_once()

    async def test_background_failure_aborts_wait(self):
        async def wait_forever(*args, **kwargs):
            await asyncio.Event().wait()

        player = SimpleNamespace(battle_against=wait_forever, training_error=threading.Event(),
                                 battles={}, record_training_error=Mock())
        player.training_error.set()
        with self.assertRaisesRegex(RuntimeError, "Training battle failed"):
            await run_training_battle(player, player, PPOConfig())

    def test_challenge_rejection_popup_is_fatal_and_logged(self):
        player = SimpleNamespace(training_error=threading.Event(), record_training_error=Mock())
        handler = _ClientErrorHandler(player)
        handler.emit(logging.LogRecord("test", logging.WARNING, "", 0,
                                       "Popup message received: Your team was rejected", (), None))
        self.assertTrue(player.training_error.is_set())
        player.record_training_error.assert_called_once()


if __name__ == "__main__":
    unittest.main()
