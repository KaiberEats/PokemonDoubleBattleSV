import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from action_rules import FakeOutTracker
from action_mapper import legal_action_mask
from config import PPOConfig, SLOT_ACTION_DIM
from test_action_mapper import FakeBattle, FakeMove
from train_max import main
from train_team import train_stage


class RuleTests(unittest.TestCase):
    def test_initial_ordinary_and_replacement_entries_and_reentry(self):
        tracker = FakeOutTracker()
        tracker.record([['', 'switch', 'p1a: Hands'], ['', 'turn', '1']])
        battle = NS(player_role='p1', turn=1)
        self.assertEqual(tracker.allowed(battle), (True, False))
        tracker.record([['', 'turn', '2']])
        battle.turn = 2
        self.assertFalse(tracker.allowed(battle)[0])
        # A normal turn-2 switch has its first actionable turn at turn 3.
        tracker.record([['', 'switch', 'p1a: Other'], ['', 'turn', '3']])
        battle.turn = 3
        self.assertTrue(tracker.allowed(battle)[0])
        tracker.record([['', 'faint', 'p1a: Other'], ['', 'upkeep'],
                        ['', 'switch', 'p1a: Hands'], ['', 'turn', '4']])
        battle.turn = 4
        self.assertTrue(tracker.allowed(battle)[0])
        tracker.record([['', 'turn', '5']])
        battle.turn = 5
        self.assertFalse(tracker.allowed(battle)[0])

    def test_mask_changes_only_fakeout_choices_on_the_disallowed_slot(self):
        battle = FakeBattle()
        battle.available_moves[0] = [FakeMove('fakeout'), FakeMove('drainpunch')]
        baseline = legal_action_mask(battle)
        allowed = legal_action_mask(battle, fake_out_allowed=(True, True))
        blocked = legal_action_mask(battle, fake_out_allowed=(False, True))
        np.testing.assert_array_equal(baseline, allowed)
        changed = np.flatnonzero(baseline != blocked)
        self.assertGreater(len(changed), 0)
        self.assertTrue(all(1 <= action // SLOT_ACTION_DIM <= 5 for action in changed))
        self.assertTrue(all(not blocked[action] for action in changed))


class ExperimentTests(unittest.IsolatedAsyncioTestCase):
    async def test_evaluation_restricts_only_player_one(self):
        from evaluate import evaluate_against
        learner, opponent = Mock(), Mock()
        learner.n_finished_battles = learner.n_won_battles = learner.n_lost_battles = 0
        async def battle(*args, **kwargs):
            learner.n_finished_battles += 1
            learner.n_won_battles += 1
        factory = Mock(return_value=opponent)
        with patch('evaluate.PPODoublesPlayer', return_value=learner) as create, \
             patch('evaluate.run_training_battle', side_effect=battle), \
             patch('evaluate.stop_players', new_callable=AsyncMock):
            await evaluate_against(Mock(), factory, 'random', 1,
                                   (('team1', 'one'), ('team2', 'two')),
                                   PPOConfig(block_fake_out=True))
        self.assertTrue(create.call_args.kwargs['block_fake_out'])
        self.assertNotIn('block_fake_out', factory.call_args.kwargs)

    async def test_input_output_separation_flags_and_manifest(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'base.pt'
            output = Path(directory) / 'b.pt'
            source.write_bytes(b'baseline')
            agent = Mock()
            with patch('train_max.load_teams', return_value={'team1': 'fixed'}), \
                 patch('train_max.PPOAgent', return_value=agent), \
                 patch('train_max.load_existing_model') as load, \
                 patch('train_max.train_stage', new_callable=AsyncMock) as stage:
                await main(battles=10000, load_model=str(source), save_model=str(output), block_fake_out=True)
            self.assertEqual(source.read_bytes(), b'baseline')
            self.assertEqual(load.call_args.args[1].model_path, str(source))
            cfg = stage.await_args.args[1]
            self.assertEqual(cfg.model_path, str(output))
            self.assertTrue(cfg.block_fake_out)
            self.assertEqual(cfg.checkpoint_every, 5000)
            agent.save.assert_called_once_with(str(output))
            metadata = json.loads(output.with_suffix('.run.json').read_text())
            self.assertTrue(metadata['completed'])
            self.assertEqual(metadata['completed_battles'], 10000)
            with self.assertRaises(ValueError):
                await main(load_model=str(source), save_model=str(source))
            with self.assertRaises(FileNotFoundError):
                await main(load_model=str(source) + '.missing', save_model=str(output))

    async def test_checkpoint_boundaries_are_exact_even_across_update_intervals(self):
        players = [Mock(buffer=[]), Mock(buffer=[])]
        with tempfile.TemporaryDirectory() as directory:
            cfg = PPOConfig(parallel_battles=1, update_every_episodes=3,
                            checkpoint_every=5, model_path=str(Path(directory) / 'a.pt'))
            agent = Mock()
            with patch('train_team.PPODoublesPlayer', return_value=players[0]), \
                 patch('train_team.MaxDamageDoublesPlayer', return_value=players[1]), \
                 patch('train_team.run_training_battle', new_callable=AsyncMock) as battle, \
                 patch('train_team.finalize_finished_battles'), \
                 patch('train_team.update_and_save'), \
                 patch('train_team.stop_players', new_callable=AsyncMock):
                await train_stage(agent, cfg, {'team1': 'one'}, 'maxdamage', random.Random(42), 10)
            self.assertEqual(battle.await_count, 10)
            self.assertEqual([Path(c.args[0]).name for c in agent.save.call_args_list],
                             ['a_00005.pt', 'a_00010.pt'])


if __name__ == '__main__':
    unittest.main()
