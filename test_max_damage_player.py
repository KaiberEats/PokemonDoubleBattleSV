import unittest
from unittest.mock import Mock, patch
from poke_env.environment import Move

from max_damage_player import MaxDamageDoublesPlayer, estimate_damage_score


class FakeMove:
    def __init__(
        self,
        move_id,
        base_power,
        move_type,
        category="SPECIAL",
        accuracy=1.0,
        expected_hits=1,
        flags=None,
    ):
        self.id = move_id
        self.base_power = base_power
        self.type = move_type
        self.category = category
        self.accuracy = accuracy
        self.expected_hits = expected_hits
        self.flags = flags or {}


class FakePokemon:
    def __init__(
        self,
        species,
        type_1=None,
        type_2=None,
        ability=None,
        hp=1.0,
        multipliers=None,
    ):
        self.species = species
        self.type_1 = type_1
        self.type_2 = type_2
        self.ability = ability
        self.current_hp_fraction = hp
        self.fainted = False
        self.stats = {"atk": 100, "def": 100, "spa": 100, "spd": 100}
        self.boosts = {}
        self._multipliers = multipliers or {}

    def damage_multiplier(self, move):
        return self._multipliers.get(move.type, 1.0)


class FakeBattle:
    teampreview = False
    weather = None

    def __init__(self):
        self.active_pokemon = [
            FakePokemon("attacker-left", type_1="Fire", ability="technician"),
            FakePokemon("attacker-right", type_1="Water"),
        ]
        self.opponent_active_pokemon = [
            FakePokemon("target-left", multipliers={"Fire": 2.0, "Water": 0.5}),
            FakePokemon("target-right", hp=0.4, multipliers={"Fire": 1.0, "Water": 2.0}),
        ]
        self.available_moves = [
            [
                FakeMove("weak-fire", 50, "Fire", category="SPECIAL"),
                FakeMove("strong-water", 90, "Water", category="SPECIAL"),
            ],
            [
                FakeMove("water-hit", 80, "Water", category="SPECIAL"),
            ],
        ]
        self.available_switches = [
            [FakePokemon("bench-low", hp=0.2), FakePokemon("bench-high", hp=0.9)],
            [FakePokemon("bench-right", hp=0.7)],
        ]
        self.force_switch = [False, False]

    def get_possible_showdown_targets(self, move, pokemon):
        return [1, 2]


class MaxDamagePlayerTests(unittest.TestCase):
    def test_helping_hand_is_selectable_with_or_without_ally(self):
        for ally_present in (True, False):
            with self.subTest(ally_present=ally_present):
                battle = FakeBattle()
                if not ally_present:
                    battle.active_pokemon[1] = None
                battle.available_switches = [[], []]
                battle.available_moves[0] = [Move('helpinghand', gen=9)]
                if not ally_present:
                    battle.available_moves[1] = []
                battle.get_possible_showdown_targets = lambda move, mon: [-2] if move.id == 'helpinghand' else [1, 2]
                player = object.__new__(MaxDamageDoublesPlayer)
                player.record_training_error = Mock()
                order = player.choose_move(battle)
                self.assertEqual(order.first_order.order.id, 'helpinghand')
                self.assertEqual(order.first_order.move_target, -2)
                fallback = player.choose_random_doubles_move(battle)
                self.assertEqual(fallback.first_order.order.id, 'helpinghand')
                player.record_training_error.assert_not_called()

    def test_struggle_remains_selectable_after_pp_exhaustion(self):
        battle = FakeBattle()
        battle.active_pokemon[1] = None
        battle.available_moves = [[Move('struggle', gen=9)], []]
        battle.available_switches = [[], []]
        player = object.__new__(MaxDamageDoublesPlayer)
        self.assertEqual(player.choose_move(battle).first_order.order.id, 'struggle')

    def test_fallback_failure_is_not_retried(self):
        player = object.__new__(MaxDamageDoublesPlayer)
        player.record_training_error = Mock()
        player._best_slot_order = Mock(return_value=None)
        player.choose_random_doubles_move = Mock(side_effect=ValueError('no legal action'))
        with self.assertRaisesRegex(ValueError, 'no legal action'):
            player.choose_move(FakeBattle())
        player.choose_random_doubles_move.assert_called_once()

    def test_weak_ally_is_not_a_damage_target(self):
        battle = FakeBattle()
        battle.active_pokemon[1].current_hp_fraction = 0.01
        battle.get_possible_showdown_targets = lambda move, mon: [-2, -1, 1, 2]
        player = object.__new__(MaxDamageDoublesPlayer)
        order = player.choose_move(battle)
        self.assertGreater(order.first_order.move_target, 0)
        self.assertGreater(order.second_order.move_target, 0)
        self.assertEqual(estimate_damage_score(battle, battle.active_pokemon[0], battle.available_moves[0][0], -2), 0)

    def test_spread_friendly_fire_is_excluded_but_foe_only_spread_is_allowed(self):
        battle = FakeBattle()
        attacker = battle.active_pokemon[0]
        self.assertEqual(estimate_damage_score(battle, attacker, Move('earthquake', gen=9), 0), 0)
        self.assertEqual(estimate_damage_score(battle, attacker, Move('surf', gen=9), 0), 0)
        self.assertGreater(estimate_damage_score(battle, attacker, Move('heatwave', gen=9), 0), 0)

    def test_exception_fallback_also_excludes_allied_targets(self):
        battle = FakeBattle()
        battle.get_possible_showdown_targets = lambda move, mon: [-2, -1, 1, 2]
        player = object.__new__(MaxDamageDoublesPlayer)
        player.record_training_error = Mock()
        with patch('max_damage_player.estimate_damage_score', side_effect=ValueError('test')):
            order = player.choose_move(battle)
        for slot, selected in enumerate((order.first_order, order.second_order)):
            self.assertTrue(selected.order in battle.available_switches[slot] or selected.move_target > 0)

    def test_both_slots_can_use_same_move_without_random_fallback(self):
        battle = FakeBattle()
        battle.available_moves = [[FakeMove('water-hit', 80, 'Water')]] * 2
        player = object.__new__(MaxDamageDoublesPlayer)
        player.choose_random_doubles_move = Mock(side_effect=AssertionError('unnecessary fallback'))
        order = player.choose_move(battle)
        self.assertEqual(order.first_order.order.id, order.second_order.order.id)

    def test_move_score_error_records_the_specific_move_before_fallback(self):
        player = object.__new__(MaxDamageDoublesPlayer)
        player.record_training_error = Mock()
        player.choose_random_doubles_move = Mock(return_value="fallback")
        with patch("max_damage_player.estimate_damage_score", side_effect=ValueError("bad move")):
            self.assertEqual(player.choose_move(FakeBattle()), "fallback")
        calls = player.record_training_error.call_args_list
        self.assertEqual(calls[0].args[1], "move_score_exception")
        self.assertEqual(calls[0].kwargs["related_moves"], ["weak-fire"])
        self.assertEqual(calls[1].args[1], "max_damage_fallback")

    def test_damage_score_uses_type_stab_ability_and_hp_pressure(self):
        battle = FakeBattle()
        attacker = battle.active_pokemon[0]
        weak_fire = battle.available_moves[0][0]
        strong_water = battle.available_moves[0][1]

        fire_score = estimate_damage_score(battle, attacker, weak_fire, 1)
        water_score = estimate_damage_score(battle, attacker, strong_water, 1)

        self.assertGreater(fire_score, water_score)

    def test_choose_move_selects_best_move_and_target(self):
        player = object.__new__(MaxDamageDoublesPlayer)
        battle = FakeBattle()

        order = player.choose_move(battle)

        self.assertEqual(order.first_order.order.id, "strong-water")
        self.assertEqual(order.first_order.move_target, 2)
        self.assertEqual(order.second_order.order.id, "water-hit")
        self.assertEqual(order.second_order.move_target, 2)

    def test_forced_switch_picks_healthiest_available_switch(self):
        player = object.__new__(MaxDamageDoublesPlayer)
        battle = FakeBattle()
        battle.force_switch = [True, False]

        order = player.choose_move(battle)

        self.assertEqual(order.first_order.order.species, "bench-high")
        self.assertIsNone(order.second_order)


if __name__ == "__main__":
    unittest.main()
