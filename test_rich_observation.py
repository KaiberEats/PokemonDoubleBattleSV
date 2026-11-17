import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch
from poke_env.environment.move import Move

from config import OBS_DIM, PRE_RICH_OBS_DIM, PPOConfig, OWN_SET_OFFSET
from ppo import PPOAgent
from rich_observation import (BattleFlow, HISTORY_DIM, TIMER_DIM, TIMER_NAMES,
                              SET_DIM, rich_features, static_move, own_set)
from state_encoder import embed_battle


class RichObservationTests(unittest.TestCase):
    def test_history_preserves_order_actor_target_outcome_and_long_prefix(self):
        a = ['', 'move', 'p1a: A', 'Fake Out', 'p2b: B']
        b = ['', '-fail', 'p1a: A']
        flow, reversed_flow = BattleFlow(), BattleFlow()
        flow.record([a, b])
        reversed_flow.record([b, a])
        battle = NS(player_role='p1', turn=1)
        self.assertFalse(np.array_equal(flow.features(battle), reversed_flow.features(battle)))
        self.assertEqual(flow.events[0][1], tuple(a))
        prefix = flow.features(battle).copy()
        flow.record([['', 'turn', str(i)] for i in range(1, 1001)])
        self.assertEqual(len(flow.events), 1002)
        self.assertEqual(flow.events[0][1], tuple(a))
        self.assertFalse(np.array_equal(prefix, flow.features(battle)))
        other = BattleFlow()
        other.record([['', 'move', 'p1a: A', 'Fake Out', 'p1b: B'], b])
        self.assertFalse(np.array_equal(other.features(battle), reversed_flow.features(battle)))

    def test_timers_layers_and_explicit_end(self):
        flow = BattleFlow()
        flow.record([['', 'turn', '2'], ['', '-weather', 'RainDance'],
                     ['', '-fieldstart', 'move: Trick Room'],
                     ['', '-sidestart', 'p1: learner', 'Reflect'],
                     ['', '-sidestart', 'p2: opponent', 'Spikes'],
                     ['', '-sidestart', 'p2: opponent', 'Spikes']])
        out = flow.features(NS(player_role='p1', turn=4))
        def timer(side, name):
            i = HISTORY_DIM + (side * len(TIMER_NAMES) + TIMER_NAMES.index(name)) * 5
            return out[i:i+5]
        np.testing.assert_allclose(timer(0, 'raindance'), [1, 3/8, 6/8, 1, 1/3])
        np.testing.assert_allclose(timer(0, 'trickroom'), [1, 3/8, 3/8, 0, 1/3])
        self.assertAlmostEqual(timer(2, 'spikes')[-1], 2/3)
        flow.record([['', '-sideend', 'p2: opponent', 'Spikes']])
        self.assertNotIn(('p2', 'spikes'), flow.timers)

    def test_category_pp_hp_condition_and_own_sets(self):
        move = Move('eruption', gen=9)
        move.use()
        mon = NS(current_hp_fraction=.4, moves={'eruption': move}, item='charcoal', ability='drought')
        battle = NS(player_role='p1', turn=1, active_pokemon=[mon], available_moves=[[move]], team={'a': mon})
        flow = BattleFlow()
        flow.record([['', 'switch', 'p1a: A', 'Torkoal, L50'], ['', 'turn', '1']])
        out = rich_features(battle, flow)
        start = HISTORY_DIM + TIMER_DIM
        np.testing.assert_array_equal(out[start:start+3], [0, 1, 0])
        self.assertAlmostEqual(out[start+5], move.current_pp / move.max_pp)
        self.assertAlmostEqual(out[start+6], .4)
        self.assertEqual(out[start+8], 1)
        self.assertEqual(static_move('protect')[3], 1)
        self.assertEqual(static_move('fakeout')[1], 1)
        changed = NS(**{**mon.__dict__, 'item': 'heatrock'})
        self.assertFalse(np.array_equal(own_set(mon), own_set(changed)))
        x = embed_battle(battle, flow=flow)
        self.assertEqual(x.shape, (OBS_DIM,))
        np.testing.assert_array_equal(x[OWN_SET_OFFSET:OWN_SET_OFFSET+SET_DIM], own_set(mon))
        self.assertTrue(np.isfinite(x).all())

    def test_checkpoint_preserves_previous_policy_and_enables_new_selection_inputs(self):
        agent = PPOAgent(PPOConfig(hidden_dim=16))
        state = agent.net.state_dict()
        state['body.0.weight'] = state['body.0.weight'][:, :PRE_RICH_OBS_DIM].clone()
        state.pop('selection.set_encoder.weight')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.pt'
            torch.save(state, path)
            agent.load(path)
        torch.testing.assert_close(agent.net.body[0].weight[:, :PRE_RICH_OBS_DIM], state['body.0.weight'])
        self.assertEqual(agent.net.body[0].weight[:, PRE_RICH_OBS_DIM:].count_nonzero(), 0)
        self.assertEqual(agent.net.selection.set_encoder.weight.count_nonzero(), 0)
        mons = {i: NS(species=s, moves={'protect': None}, item='leftovers', ability='pressure')
                for i, s in enumerate(['torkoal', 'pelipper', 'rillaboom', 'kingambit', 'incineroar', 'primarina'])}
        obs = torch.tensor(embed_battle(NS(team=mons, teampreview=True)))[None]
        logits, value = agent.net(obs)
        (logits[0, 0] + value.sum()).backward()
        self.assertGreater(agent.net.selection.set_encoder.weight.grad.abs().sum(), 0)


if __name__ == '__main__':
    unittest.main()
