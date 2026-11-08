import unittest
from types import SimpleNamespace
from collections import Counter
from unittest.mock import AsyncMock, patch
import tempfile
from pathlib import Path
import numpy as np
import torch
from poke_env.environment import Status, Weather, Field, SideCondition

from config import OBS_DIM, SPECIES_OBS_DIM, MOVE_HISTORY_OBS_DIM, HISTORY_MOVE_IDS, PPOConfig
from state_encoder import (MOVE_FEATURE_DIM, TYPES, _available_move_features,
                           embed_battle, _status_onehot, _field_features,
                           move_history_features, MOVE_HISTORY_INDEX, STATUS_LIST)
from players import PPODoublesPlayer
from training_errors import TrainingDiagnosticsMixin
from ppo import PPOAgent


class FakeMove:
    def __init__(self, base_power, move_type, accuracy=1.0, priority=0):
        self.base_power = base_power
        self.type = move_type
        self.accuracy = accuracy
        self.priority = priority


class MissingPriorityMove:
    base_power = 80
    type = "FIRE"
    accuracy = 1.0

    @property
    def priority(self):
        raise KeyError("priority")


class FakePokemon:
    def __init__(self, type_1=None, type_2=None, multipliers=None):
        self.current_hp_fraction = 1.0
        self.status = None
        self.boosts = {}
        self.fainted = False
        self.active = True
        self.type_1 = type_1
        self.type_2 = type_2
        self._multipliers = multipliers or {}

    def damage_multiplier(self, move):
        return self._multipliers.get(move.type, 1.0)


class FakeBattle:
    weather = None
    fields = None
    side_conditions = None
    opponent_side_conditions = None
    turn = 1

    def __init__(self):
        self.active_pokemon = [
            FakePokemon(type_1="FIRE"),
            FakePokemon(type_1="WATER"),
        ]
        self.opponent_active_pokemon = [
            FakePokemon(multipliers={"FIRE": 2.0}),
            FakePokemon(multipliers={"FIRE": 0.5}),
        ]
        self.team = {str(i): FakePokemon() for i in range(6)}
        self.opponent_team = {str(i): FakePokemon() for i in range(6)}
        self.available_moves = [
            [FakeMove(90, "FIRE", accuracy=85, priority=1)],
            [],
        ]


class StateEncoderTests(unittest.TestCase):
    def test_real_status_enums(self):
        for status in Status:
            features = _status_onehot(SimpleNamespace(status=status))
            self.assertEqual(features[STATUS_LIST.index(status.name.lower())], 1)
        self.assertEqual(_status_onehot(SimpleNamespace(status=None))[-1], 1)

    def test_real_weather_field_and_side_enums(self):
        battle = FakeBattle()
        battle.weather = {Weather.RAINDANCE: 1}
        battle.fields = {Field.TRICK_ROOM: 1}
        battle.side_conditions = {SideCondition.TAILWIND: 1}
        battle.opponent_side_conditions = {SideCondition.REFLECT: 1}
        features = _field_features(battle)
        for index in (0, 8, 20, 25):
            self.assertEqual(features[index], 1)
        self.assertEqual(features[1], 0)

    def test_move_history_counts_and_side_perspective(self):
        battle = FakeBattle()
        battle.player_role = 'p1'
        history = Counter({('p1', 'protect'): 2, ('p2', 'moonblast'): 1})
        width = len(HISTORY_MOVE_IDS) + 1
        features = move_history_features(battle, history)
        self.assertAlmostEqual(features[MOVE_HISTORY_INDEX['protect']], 2/3)
        self.assertEqual(features[width + MOVE_HISTORY_INDEX['moonblast']], .5)
        np.testing.assert_array_equal(embed_battle(battle, history)[SPECIES_OBS_DIM:MOVE_HISTORY_OBS_DIM], features)
        battle.player_role = 'p2'
        swapped = move_history_features(battle, history)
        np.testing.assert_array_equal(swapped[:width], features[width:])
        self.assertFalse(move_history_features(battle).any())

    def test_previous_checkpoint_preserves_logits_and_zero_pads_history(self):
        agent = PPOAgent(PPOConfig(hidden_dim=16))
        state = agent.net.state_dict()
        state['body.0.weight'] = state['body.0.weight'][:, :SPECIES_OBS_DIM].clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.pt'
            torch.save(state, path)
            agent.load(path)
        torch.testing.assert_close(agent.net.body[0].weight[:, :SPECIES_OBS_DIM], state['body.0.weight'])
        self.assertEqual(agent.net.body[0].weight[:, SPECIES_OBS_DIM:].count_nonzero(), 0)

    def test_embed_battle_keeps_configured_observation_size(self):
        obs = embed_battle(FakeBattle())

        self.assertEqual(obs.shape[0], OBS_DIM)

    def test_available_move_features_include_requested_move_information(self):
        features = _available_move_features(FakeBattle())
        first_move = features[:MOVE_FEATURE_DIM]

        self.assertEqual(features.shape[0], 2 * 4 * MOVE_FEATURE_DIM)
        self.assertEqual(first_move[0], 1.0)
        self.assertAlmostEqual(first_move[1], 90 / 250)
        self.assertEqual(first_move[2 + TYPES.index("FIRE")], 1.0)
        self.assertAlmostEqual(first_move[2 + len(TYPES)], 0.85)
        self.assertAlmostEqual(first_move[3 + len(TYPES)], 0.5)
        self.assertAlmostEqual(first_move[4 + len(TYPES)], 0.125)
        self.assertEqual(first_move[5 + len(TYPES)], 1.0)
        self.assertAlmostEqual(first_move[6 + len(TYPES)], 8 / 14)

    def test_missing_move_priority_defaults_to_zero(self):
        battle = FakeBattle()
        battle.available_moves[0] = [MissingPriorityMove()]

        features = _available_move_features(battle)
        first_move = features[:MOVE_FEATURE_DIM]

        self.assertAlmostEqual(first_move[6 + len(TYPES)], 7 / 14)


class MoveHistoryMessageTests(unittest.IsolatedAsyncioTestCase):
    async def test_server_events_are_separate_per_battle_and_cleared_at_end(self):
        player = object.__new__(PPODoublesPlayer)
        player._move_histories = {}
        player._battles = {}
        async def handled(messages):
            if messages[0][0] == '>battle-a':
                self.assertEqual(player._move_histories['battle-a'][('p2', 'icebeam')], 1)
        with patch.object(TrainingDiagnosticsMixin, '_handle_battle_message', side_effect=handled):
            await player._handle_battle_message([
                ['>battle-a'], ['', 'move', 'p2a: Porygon2', 'Ice Beam', 'p1a: X'],
                ['', 'request', '{}']])
            await player._handle_battle_message([['>battle-b'], ['', 'move', 'p1b: X', 'Protect']])
            self.assertNotIn(('p2', 'icebeam'), player._move_histories['battle-b'])
            player._battles['battle-a'] = SimpleNamespace(finished=True)
            await player._handle_battle_message([['>battle-a'], ['', 'win', 'player']])
        self.assertNotIn('battle-a', player._move_histories)
        self.assertIn('battle-b', player._move_histories)
        self.assertNotIn('battle-a', player._switch_histories)
        self.assertIn('battle-b', player._switch_histories)


if __name__ == "__main__":
    unittest.main()
