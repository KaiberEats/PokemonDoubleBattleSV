import unittest
from unittest.mock import AsyncMock, Mock, patch

import train_max


class MaxTrainingTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_continues_one_model_for_10000_maxdamage_battles(self):
        agent, teams = Mock(), {"team1": "fixed"}
        with patch('train_max.PPOAgent', return_value=agent), \
             patch('train_max.load_teams', return_value=teams), \
             patch('train_max.load_existing_model') as load, \
             patch('train_max.Path.write_text'), \
             patch('train_max.train_stage', new_callable=AsyncMock) as stage:
            await train_max.main()
        load.assert_called_once()
        cfg = load.call_args.args[1]
        self.assertEqual(cfg.parallel_battles, 5)
        self.assertEqual(cfg.update_every_episodes, 50)
        self.assertEqual(cfg.model_path, 'ppo_doubles_full_actions.pt')
        stage.assert_awaited_once()
        args = stage.await_args.args
        self.assertIs(args[0], agent)
        self.assertIs(args[2], teams)
        self.assertEqual(args[3], 'maxdamage')
        self.assertEqual(args[5], 10000)
        agent.save.assert_called_once_with(cfg.model_path)

    async def test_invalid_counts_do_not_create_agent(self):
        with patch('train_max.PPOAgent') as factory:
            for parallel, battles in ((0, 10), (5, 0), (-1, 10)):
                with self.assertRaises(ValueError):
                    await train_max.main(parallel, battles)
            factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
