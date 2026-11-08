import unittest
from types import SimpleNamespace as NS
from pathlib import Path
import tempfile
import numpy as np
import torch

from switch_history import SwitchHistory
from state_encoder import switch_history_features, species_features, embed_battle
from config import SWITCH_SLOT_DIM, MOVE_HISTORY_OBS_DIM, PPOConfig, OBS_DIM
from ppo import PPOAgent


class SwitchHistoryTests(unittest.TestCase):
    def test_initial_entry_switch_faint_and_forced_replacement(self):
        h = SwitchHistory()
        h.record([['', 'switch', 'p1a: Flutter Mane', 'Flutter Mane, L50'],
                  ['', 'switch', 'p2b: Pelipper', 'Pelipper, L50'], ['', 'turn', '1']])
        b = NS(player_role='p1', turn=1)
        x = switch_history_features(b, h)
        self.assertEqual(x[0], 1)
        self.assertEqual(x[2], 0)
        self.assertEqual(x[3], 0)  # Initial deployment is not a switch-out.
        h.record([['', 'turn', '3'], ['', 'switch', 'p1a: Landorus', 'Landorus-Therian, L50']])
        b.turn = 5
        x = switch_history_features(b, h)
        np.testing.assert_allclose(x[:6], [1, .75, 2/3, 1, .75, 0])
        np.testing.assert_array_equal(x[6:42], species_features(NS(species='fluttermane')))
        np.testing.assert_array_equal(x[42:78], species_features(NS(species='landorustherian')))
        h.record([['', 'faint', 'p1a: Landorus']])
        self.assertEqual(switch_history_features(b, h)[0], 0)
        h.record([['', 'turn', '6'], ['', 'switch', 'p1a: Iron Hands', 'Iron Hands, L50']])
        self.assertTrue(h.slots['p1a']['forced'])
        self.assertEqual(h.slots['p1a']['entry_turn'], 6)

    def test_perspective_and_form_change_do_not_reset_entry(self):
        h = SwitchHistory()
        h.record([['', 'switch', 'p2b: X', 'Palafin, L50'], ['', 'turn', '4'],
                  ['', 'detailschange', 'p2b: X', 'Palafin-Hero, L50']])
        b = NS(player_role='p2', turn=4)
        x = switch_history_features(b, h)
        self.assertEqual(x[SWITCH_SLOT_DIM], 1)
        self.assertAlmostEqual(x[SWITCH_SLOT_DIM + 2], .75)
        self.assertFalse(x[:SWITCH_SLOT_DIM].any())
        self.assertFalse(switch_history_features(b, SwitchHistory()).any())
        self.assertEqual(h.slots['p2b']['species'], 'palafinhero')

    def test_previous_history_checkpoint_loads_without_changing_old_weights(self):
        agent = PPOAgent(PPOConfig(hidden_dim=16))
        state = agent.net.state_dict()
        state['body.0.weight'] = state['body.0.weight'][:, :MOVE_HISTORY_OBS_DIM].clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save(state, path)
            agent.load(path)
        torch.testing.assert_close(agent.net.body[0].weight[:, :MOVE_HISTORY_OBS_DIM], state['body.0.weight'])
        self.assertEqual(agent.net.body[0].weight[:, MOVE_HISTORY_OBS_DIM:].count_nonzero(), 0)
        h = SwitchHistory()
        h.record([['', 'switch', 'p1a: X', 'Torkoal, L50']])
        obs = embed_battle(NS(player_role='p1', turn=1), switch_history=h)
        self.assertEqual(len(obs), OBS_DIM)
        np.testing.assert_array_equal(obs[MOVE_HISTORY_OBS_DIM:], switch_history_features(NS(player_role='p1', turn=1), h))
