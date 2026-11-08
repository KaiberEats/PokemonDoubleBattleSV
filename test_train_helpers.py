import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from config import ACTION_DIM, OBS_DIM, PPOConfig
from ppo import RolloutBuffer
from train import drain_buffers, load_existing_model, update_and_save, training_account, training_failure_message


class FakePlayer:
    def __init__(self, action, reward):
        self.buffer = RolloutBuffer()
        self.buffer.add(
            np.zeros(OBS_DIM, dtype=np.float32),
            action,
            0.0,
            reward,
            True,
            0.0,
            np.ones(ACTION_DIM, dtype=np.bool_),
        )


class FakeAgent:
    def __init__(self):
        self.updated_actions = []
        self.saved_path = None
        self.loaded_path = None
        self.net = FakeNet()

    def update(self, buffer):
        self.updated_actions = list(buffer.actions)

    def save(self, path):
        self.saved_path = path

    def load(self, path):
        self.loaded_path = path


class FakeNet:
    def __init__(self):
        self.train_called = False

    def train(self):
        self.train_called = True


class TrainHelperTests(unittest.TestCase):
    def test_default_logs_are_fresh_per_run_and_custom_path_is_preserved(self):
        first, second = PPOConfig(), PPOConfig()
        self.assertNotEqual(first.error_log_path, second.error_log_path)
        self.assertNotEqual(Path(first.error_log_path).name, "training_errors.jsonl")
        self.assertEqual(PPOConfig(error_log_path="custom.jsonl").error_log_path, "custom.jsonl")

    def test_failure_message_uses_actual_fallback_log_path(self):
        cfg = PPOConfig(error_log_path="old.jsonl")
        player = SimpleNamespace(error_recorder=SimpleNamespace(active_path=Path("fallback.jsonl")))
        message = training_failure_message(cfg, player, player)
        self.assertIn("fallback.jsonl", message)
        self.assertNotIn("old.jsonl", message)
        self.assertEqual(message.count("fallback.jsonl"), 1)

    def test_new_training_players_do_not_reuse_stale_showdown_accounts(self):
        first = training_account("maxdamage")
        second = training_account("maxdamage")
        self.assertNotEqual(first.username, second.username)
        self.assertLessEqual(len(first.username), 18)
        self.assertTrue(first.username.startswith("maxdamage"))

    def test_drain_buffers_combines_and_resets_both_players(self):
        player_1 = FakePlayer(10, 1.0)
        player_2 = FakePlayer(20, -1.0)

        combined = drain_buffers([player_1, player_2])

        self.assertEqual(combined.actions, [10, 20])
        self.assertEqual(len(player_1.buffer), 0)
        self.assertEqual(len(player_2.buffer), 0)

    def test_update_and_save_uses_both_self_play_buffers(self):
        cfg = PPOConfig(model_path="test-model.pt")
        agent = FakeAgent()

        buffer_size = update_and_save(
            agent,
            cfg,
            [FakePlayer(10, 1.0), FakePlayer(20, -1.0)],
        )

        self.assertEqual(buffer_size, 2)
        self.assertEqual(agent.updated_actions, [10, 20])
        self.assertEqual(agent.saved_path, "test-model.pt")

    def test_load_existing_model_continues_from_saved_model(self):
        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            model_path.write_bytes(b"placeholder")
            cfg = PPOConfig(model_path=str(model_path))
            agent = FakeAgent()

            loaded = load_existing_model(agent, cfg)

        self.assertTrue(loaded)
        self.assertEqual(agent.loaded_path, str(model_path))
        self.assertTrue(agent.net.train_called)

    def test_load_existing_model_starts_fresh_when_missing(self):
        cfg = PPOConfig(model_path="missing-model.pt")
        agent = FakeAgent()

        loaded = load_existing_model(agent, cfg)

        self.assertFalse(loaded)
        self.assertIsNone(agent.loaded_path)
        self.assertFalse(agent.net.train_called)


if __name__ == "__main__":
    unittest.main()
