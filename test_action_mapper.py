import unittest
from unittest.mock import patch

import numpy as np

from action_mapper import (
    SWITCH_ACTION_START,
    TARGETS,
    action_to_order,
    legal_action_mask,
    random_team_preview_order,
    team_preview_order_from_action,
)
from config import ACTION_DIM, SLOT_ACTION_DIM


class FakeMove:
    def __init__(self, move_id):
        self.id = move_id


class FakePokemon:
    def __init__(self, species):
        self.species = species


class FakeBattle:
    teampreview = False

    def __init__(self):
        bench_a = FakePokemon("bench-a")
        bench_b = FakePokemon("bench-b")
        self.active_pokemon = [FakePokemon("left"), FakePokemon("right")]
        self.available_moves = [[FakeMove("leftmove")], [FakeMove("rightmove")]]
        self.available_switches = [
            [bench_a, bench_b],
            [bench_a, bench_b],
        ]
        self.force_switch = [False, False]
        self.team = {str(i): FakePokemon(str(i)) for i in range(6)}

    def get_possible_showdown_targets(self, move, pokemon):
        if pokemon is self.active_pokemon[0]:
            return [1, 2]
        return [-1, 1]


class FakePlayer:
    def choose_random_doubles_move(self, battle):
        raise AssertionError("A legal action should not fall back to random")


def joint_action(left_action, right_action):
    return left_action * SLOT_ACTION_DIM + right_action


def move_action(move_index, target):
    return 1 + move_index * len(TARGETS) + TARGETS.index(target)


class ActionMapperTests(unittest.TestCase):
    def test_supplied_mask_avoids_second_full_validation(self):
        battle = FakeBattle()
        mask = legal_action_mask(battle)
        action = int(np.flatnonzero(mask)[0])
        with patch('action_mapper.legal_action_mask', side_effect=AssertionError('mask recomputed')):
            order = action_to_order(FakePlayer(), battle, action, action_mask=mask)
        self.assertIsNotNone(order)

    def test_slot_orders_are_built_only_once_per_choice(self):
        from action_mapper import _slot_order
        with patch('action_mapper._slot_order', wraps=_slot_order) as build:
            legal_action_mask(FakeBattle())
        self.assertEqual(build.call_count, 2 * SLOT_ACTION_DIM)

    def test_illegal_or_malformed_supplied_mask_still_falls_back(self):
        from unittest.mock import Mock
        for mask in (np.zeros(ACTION_DIM, dtype=bool), np.ones(1, dtype=bool)):
            player = Mock()
            player.choose_random_doubles_move.return_value = 'fallback'
            self.assertEqual(action_to_order(player, FakeBattle(), 0, action_mask=mask), 'fallback')
            player.choose_random_doubles_move.assert_called_once()

    def test_mask_contains_only_available_targets(self):
        battle = FakeBattle()
        mask = legal_action_mask(battle)

        legal = joint_action(move_action(0, 2), move_action(0, -1))
        illegal_target = joint_action(move_action(0, -2), move_action(0, -1))

        self.assertEqual(mask.shape, (ACTION_DIM,))
        self.assertTrue(mask[legal])
        self.assertFalse(mask[illegal_target])

        order = action_to_order(FakePlayer(), battle, legal)
        self.assertEqual(order.first_order.move_target, 2)
        self.assertEqual(order.second_order.move_target, -1)

    def test_regular_turn_allows_switch_and_rejects_duplicate_switch(self):
        battle = FakeBattle()
        mask = legal_action_mask(battle)

        move_and_switch = joint_action(move_action(0, 1), SWITCH_ACTION_START)
        duplicate_switch = joint_action(SWITCH_ACTION_START, SWITCH_ACTION_START)

        self.assertTrue(mask[move_and_switch])
        self.assertFalse(mask[duplicate_switch])

    def test_forced_switch_requires_other_slot_to_pass(self):
        battle = FakeBattle()
        battle.force_switch = [True, False]
        mask = legal_action_mask(battle)

        forced_switch = joint_action(SWITCH_ACTION_START, 0)
        regular_move = joint_action(SWITCH_ACTION_START, move_action(0, 1))

        self.assertTrue(mask[forced_switch])
        self.assertFalse(mask[regular_move])
        self.assertGreater(np.count_nonzero(mask), 0)

    def test_teampreview_masks_ordered_four_pokemon_choices(self):
        battle = FakeBattle()
        battle.teampreview = True

        mask = legal_action_mask(battle)

        self.assertEqual(np.count_nonzero(mask), 360)
        self.assertEqual(team_preview_order_from_action(battle, 0), "/team 1234")
        self.assertEqual(team_preview_order_from_action(battle, 1), "/team 1235")
        self.assertRegex(random_team_preview_order(battle), r"^/team [1-6]{4}$")

    def test_double_replacement_with_one_bench_allows_switch_and_pass(self):
        battle = FakeBattle()
        battle.force_switch = [True, True]
        bench = battle.available_switches[0][0]
        battle.available_switches = [[bench], [bench]]
        mask = legal_action_mask(battle)
        expected = [joint_action(SWITCH_ACTION_START, 0),
                    joint_action(0, SWITCH_ACTION_START)]
        self.assertEqual(set(np.flatnonzero(mask)), set(expected))
        for action in expected:
            order = action_to_order(FakePlayer(), battle, action)
            self.assertIsNotNone(order)

    def test_double_replacement_with_two_bench_requires_distinct_switches(self):
        battle = FakeBattle()
        battle.force_switch = [True, True]
        mask = legal_action_mask(battle)
        expected = [joint_action(SWITCH_ACTION_START, SWITCH_ACTION_START + 1),
                    joint_action(SWITCH_ACTION_START + 1, SWITCH_ACTION_START)]
        self.assertEqual(set(np.flatnonzero(mask)), set(expected))

    def test_single_replacement_cannot_pass_with_unused_bench(self):
        battle = FakeBattle()
        battle.force_switch = [True, False]
        self.assertFalse(legal_action_mask(battle)[0])

    def test_replacement_without_bench_allows_pass_only(self):
        battle = FakeBattle()
        battle.force_switch = [True, True]
        battle.available_switches = [[], []]
        self.assertEqual(list(np.flatnonzero(legal_action_mask(battle))), [0])


if __name__ == "__main__":
    unittest.main()
