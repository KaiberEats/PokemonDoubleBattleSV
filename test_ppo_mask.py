import unittest

import numpy as np

from config import ACTION_DIM, OBS_DIM, PPOConfig
from ppo import PPOAgent, RolloutBuffer


class PPOMaskTests(unittest.TestCase):
    def test_agent_samples_and_updates_with_legal_actions_only(self):
        cfg = PPOConfig(hidden_dim=16, epochs=1)
        agent = PPOAgent(cfg)
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        mask = np.zeros(ACTION_DIM, dtype=np.bool_)
        mask[123] = True

        action, logprob, value = agent.act(obs, mask)
        self.assertEqual(action, 123)

        buffer = RolloutBuffer()
        buffer.add(obs, action, logprob, 1.0, True, value, mask)
        agent.update(buffer)

    def test_rollout_buffer_extend_combines_self_play_experience(self):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        mask = np.ones(ACTION_DIM, dtype=np.bool_)
        player_1_buffer = RolloutBuffer()
        player_2_buffer = RolloutBuffer()
        combined = RolloutBuffer()

        player_1_buffer.add(obs, 1, 0.0, 1.0, True, 0.0, mask)
        player_2_buffer.add(obs, 2, 0.0, -1.0, True, 0.0, mask)
        combined.extend(player_1_buffer)
        combined.extend(player_2_buffer)

        self.assertEqual(len(combined), 2)
        self.assertEqual(combined.actions, [1, 2])
        self.assertEqual(combined.rewards, [1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
