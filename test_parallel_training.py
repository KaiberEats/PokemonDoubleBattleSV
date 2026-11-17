import asyncio
import random
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from config import PPOConfig
from ppo import PPOAgent, RolloutBuffer
from train import update_and_save
from train_team import train_stage


class ParallelTrainingTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, kind, failure=False):
        players, rounds, completed = [], [], []
        active = peak = 0

        def make_player(**kwargs):
            player = NS(buffer=RolloutBuffer(), update_team=Mock(), kwargs=kwargs)
            players.append(player)
            return player

        async def battle(learner, opponent, cfg):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(.01)
                if failure and learner is players[0]:
                    raise RuntimeError('battle failed')
                if failure:
                    await asyncio.sleep(10)
                for player in (learner, opponent) if kind == 'selfplay' else (learner,):
                    player.buffer.add([0.0], 0, 0, 1, True, 0, [True])
                completed.append(learner)
            finally:
                active -= 1

        agent = NS(update=Mock(), save=Mock())
        cfg = PPOConfig(update_every_episodes=7, parallel_battles=5)

        def update(agent, cfg, learners):
            self.assertEqual(active, 0, 'model updated during live battles')
            rounds.append(len(completed))
            return update_and_save(agent, cfg, learners)

        async def stop(*clients):
            self.assertEqual(active, 0, 'clients closed before cancelled battles finished')
            self.assertEqual(list(clients), players)

        with patch('train_team.PPODoublesPlayer', side_effect=make_player), \
             patch('train_team.FixedTeamRandomDoublesPlayer', side_effect=make_player), \
             patch('train_team.MaxDamageDoublesPlayer', side_effect=make_player), \
             patch('train_team.run_training_battle', side_effect=battle), \
             patch('train_team.finalize_finished_battles'), \
             patch('train_team.update_and_save', side_effect=update), \
             patch('train_team.stop_players', side_effect=stop) as stopped:
            if failure:
                with self.assertRaisesRegex(RuntimeError, 'battle failed'):
                    await train_stage(agent, cfg, {'team1': 'one', 'team2': 'two'}, kind, random.Random(42), 13)
                agent.update.assert_not_called()
                agent.save.assert_not_called()
            else:
                await train_stage(agent, cfg, {'team1': 'one', 'team2': 'two'}, kind, random.Random(42), 13)
                self.assertEqual(rounds, [7, 13])
                self.assertEqual(agent.save.call_count, 2)
                sizes = [len(call.args[0]) for call in agent.update.call_args_list]
                self.assertEqual(sizes, [14, 12] if kind == 'selfplay' else [7, 6])
                self.assertTrue(all(len(p.buffer) == 0 for p in players))
            stopped.assert_awaited_once()
        self.assertEqual(peak, 5)
        self.assertEqual(len(players), 10)
        accounts = [p.kwargs['account_configuration'].username for p in players]
        self.assertEqual(len(set(accounts)), 10)

    async def test_five_live_battles_one_shared_update_and_partial_round(self):
        for kind in ('random', 'maxdamage', 'selfplay'):
            with self.subTest(kind=kind):
                await self.exercise(kind)

    async def test_failure_cancels_other_battles_before_shutdown_without_update(self):
        await self.exercise('random', failure=True)

    async def test_invalid_parallelism_rejected_before_connections(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                await train_stage(Mock(), PPOConfig(parallel_battles=value), {'team1': 'one'}, 'random', random.Random(42), 1)

    def test_gae_does_not_cross_worker_or_selection_episode_boundaries(self):
        agent = PPOAgent(PPOConfig(hidden_dim=8))
        a = ([0, 1, 1], [False, True, True], [.2, .3, .1])
        b = ([0, -1, -1], [False, True, True], [.4, .5, .2])
        separate = [agent.compute_gae(*episode) for episode in (a, b)]
        joined = agent.compute_gae(*[a[i] + b[i] for i in range(3)])
        for i in range(2):
            np.testing.assert_allclose(joined[i], np.concatenate([item[i] for item in separate]))


if __name__ == '__main__':
    unittest.main()
