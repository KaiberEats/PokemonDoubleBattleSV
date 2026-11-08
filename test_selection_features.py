import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from itertools import permutations
from unittest.mock import Mock

import numpy as np
import torch

from config import LEGACY_OBS_DIM, OBS_DIM, SPECIES_OBS_DIM, PPOConfig, SPECIES_FEATURE_DIM
from ppo import ActorCritic, PPOAgent
from state_encoder import embed_battle, selection_features, species_features, TYPES
from action_mapper import team_preview_order_from_action, team_preview_action_mask
from players import PPODoublesPlayer
from ppo import RolloutBuffer


class SelectionTests(unittest.TestCase):
    def preview(self, species):
        mons = [NS(species=s) for s in species]
        return NS(team=dict(enumerate(mons)), teampreview=True,
                  teampreview_opponent_team=mons, opponent_team={}, battle_tag='test')

    def test_species_tuple_scores_survive_team_permutations(self):
        torch.manual_seed(7)
        net = ActorCritic(hidden_dim=16).eval()
        names = ['torkoal','pelipper','rillaboom','kingambit','incineroar','primarina']

        def scores(order):
            battle = self.preview(order)
            # Keep the opponent identical while reordering our team.
            battle.teampreview_opponent_team = [NS(species=s) for s in names]
            with torch.no_grad():
                logits, value = net(torch.tensor(embed_battle(battle))[None])
            return {tuple(order[i] for i in choice): logits[0, j].item()
                    for j, choice in enumerate(permutations(range(6), 4))}, value

        expected, value = scores(names)
        for ordering in (list(reversed(names)), names[2:] + names[:2]):
            actual, actual_value = scores(ordering)
            np.testing.assert_allclose(list(expected.values()), [actual[k] for k in expected], atol=1e-6)
            torch.testing.assert_close(actual_value, value)
            self.assertEqual(max(expected, key=expected.get), max(actual, key=actual.get))

    def test_selection_uses_species_and_trains_its_own_head(self):
        torch.manual_seed(9)
        net = ActorCritic(hidden_dim=16)
        names = ['torkoal','pelipper','rillaboom','kingambit','incineroar','primarina']
        obs = torch.tensor(embed_battle(self.preview(names)))[None]
        logits, value = net(obs)
        changed = torch.tensor(embed_battle(self.preview(['garchomp'] + names[1:])))[None]
        self.assertFalse(torch.allclose(logits[:, :360], net(changed)[0][:, :360]))
        (logits[0, 0] + value.sum()).backward()
        self.assertGreater(net.selection.encoder[0].weight.grad.abs().sum().item(), 0)
        self.assertTrue(net.pi.weight.grad is None or net.pi.weight.grad.abs().sum() == 0)

    def test_selection_reward_is_terminal_and_not_mixed_with_move_trajectory(self):
        for reward in (1.0, -1.0):
            player = object.__new__(PPODoublesPlayer)
            player.training = True
            player.ppo_agent = Mock()
            player.ppo_agent.act.return_value = (0, -1.0, 0.2)
            player.buffer = RolloutBuffer()
            for name in ('_last_obs', '_last_action', '_last_logprob', '_last_value', '_last_action_mask', '_selection_decisions'):
                setattr(player, name, {})
            battle = self.preview(['torkoal','pelipper','rillaboom','kingambit','incineroar','primarina'])
            battle.won, battle.lost = reward > 0, reward < 0
            player.teampreview(battle)
            self.assertNotIn('test', player._last_obs)
            player._last_obs['test'] = np.zeros(OBS_DIM, dtype=np.float32)
            player._last_action['test'] = 42
            player._last_logprob['test'] = -2.0
            player._last_value['test'] = 0.1
            player._last_action_mask['test'] = np.ones(729, dtype=np.bool_)
            player.finalize_battle(battle)
            player.finalize_battle(battle)
            self.assertEqual(player.buffer.actions, [42, 0])
            self.assertEqual(player.buffer.rewards, [reward, reward])
            self.assertEqual(player.buffer.dones, [True, True])

    def test_small_team_candidate_indices_match_showdown_orders(self):
        net = ActorCritic(hidden_dim=16)
        for size in (1, 2, 3, 4, 5):
            battle = self.preview(['torkoal','pelipper','rillaboom','kingambit','incineroar'][:size])
            logits, _ = net(torch.tensor(embed_battle(battle))[None])
            count = len(list(permutations(range(size), min(4, size))))
            self.assertTrue(torch.isfinite(logits[0, :count]).all())
            self.assertTrue((logits[0, count:] == -1e9).all())
            self.assertTrue(team_preview_order_from_action(battle, count-1).startswith('/team '))

    def test_ppo_updates_species_selection_with_terminal_result(self):
        agent = PPOAgent(PPOConfig(hidden_dim=16, epochs=1))
        battle = self.preview(['torkoal','pelipper','rillaboom','kingambit','incineroar','primarina'])
        obs = embed_battle(battle)
        mask = team_preview_action_mask(battle)
        action, logprob, value = agent.act(obs, mask)
        buffer = RolloutBuffer()
        buffer.add(obs, action, logprob, 1.0, True, value, mask)
        # Include a battle action in the same update to exercise both policies.
        battle_obs = obs.copy()
        battle_obs[LEGACY_OBS_DIM] = 0
        battle_mask = np.ones_like(mask)
        battle_action, battle_logprob, battle_value = agent.act(battle_obs, battle_mask)
        buffer.add(battle_obs, battle_action, battle_logprob, -1.0, True, battle_value, battle_mask)
        before = agent.net.selection.encoder[0].weight.detach().clone()
        agent.update(buffer)
        self.assertFalse(torch.equal(before, agent.net.selection.encoder[0].weight))
        self.assertTrue(torch.isfinite(agent.net.selection.encoder[0].weight).all())

    def test_public_species_and_form_stats(self):
        mon = NS(species='fluttermane', item='secret', ability='secret')
        data = species_features(mon)
        self.assertEqual(data[12 + TYPES.index('GHOST')], 1)
        self.assertEqual(data[12 + TYPES.index('FAIRY')], 1)
        np.testing.assert_allclose(data[30:], np.array([55,55,55,135,135,135])/255)
        mon.item = 'other'
        np.testing.assert_array_equal(data, species_features(mon))
        self.assertFalse(np.array_equal(species_features(NS(species='landorus')),
                                        species_features(NS(species='landorustherian'))))

    def test_own_order_and_all_six_opponents(self):
        mons = [NS(species=s) for s in ['torkoal','pelipper','rillaboom','kingambit','incineroar','primarina']]
        battle = NS(team=dict(enumerate(mons)), teampreview=True,
                    teampreview_opponent_team=mons, opponent_team={})
        first = selection_features(battle)
        self.assertEqual(len(first), SPECIES_OBS_DIM - LEGACY_OBS_DIM)
        self.assertEqual(first[0], 1)
        for i, mon in enumerate(mons):
            np.testing.assert_array_equal(first[1+i*SPECIES_FEATURE_DIM:1+(i+1)*SPECIES_FEATURE_DIM], species_features(mon))
        battle.teampreview_opponent_team = list(reversed(mons))
        np.testing.assert_array_equal(first, selection_features(battle))
        battle.teampreview_opponent_team = mons[:-1] + [NS(species='garchomp')]
        self.assertFalse(np.array_equal(first, selection_features(battle)))

    def test_legacy_logits_preserved_and_new_weights_trainable(self):
        cfg = PPOConfig(hidden_dim=16)
        old = ActorCritic(obs_dim=LEGACY_OBS_DIM, hidden_dim=16)
        agent = PPOAgent(cfg)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save(old.state_dict(), path)
            agent.load(path)
            x = torch.randn(2, OBS_DIM, device=agent.device)
            x[:, LEGACY_OBS_DIM] = 0  # Legacy battle policy is preserved; preview is intentionally replaced.
            with torch.no_grad():
                expected = old(x[:, :LEGACY_OBS_DIM].cpu())[0]
                actual = agent.net(x)[0].cpu()
            torch.testing.assert_close(actual, expected)
            agent.net(x)[0].sum().backward()
            self.assertGreater(agent.net.body[0].weight.grad[:, LEGACY_OBS_DIM:].abs().sum().item(), 0)
            agent.save(path)
            restored = PPOAgent(cfg)
            restored.load(path)
            torch.testing.assert_close(restored.net(x)[0], agent.net(x)[0])
