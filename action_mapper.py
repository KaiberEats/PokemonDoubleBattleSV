from itertools import permutations
import random

import numpy as np

from config import ACTION_DIM, SLOT_ACTION_DIM
from poke_env.player.battle_order import BattleOrder, DoubleBattleOrder


TARGETS = (-2, -1, 0, 1, 2)
MOVE_ACTIONS = 4 * len(TARGETS)
SWITCH_ACTION_START = 1 + MOVE_ACTIONS
TEAM_PREVIEW_SIZE = 4


def decode_action(action_id: int) -> tuple[int, int]:
    action_id = int(action_id) % ACTION_DIM
    return action_id // SLOT_ACTION_DIM, action_id % SLOT_ACTION_DIM


def _decode_slot_action(slot_action: int):
    if slot_action == 0:
        return "pass", None, None
    if slot_action < SWITCH_ACTION_START:
        offset = slot_action - 1
        return "move", offset // len(TARGETS), TARGETS[offset % len(TARGETS)]
    return "switch", slot_action - SWITCH_ACTION_START, None


def _teampreview_orders(battle):
    team_size = len(getattr(battle, "team", {}) or {})
    picked_size = min(TEAM_PREVIEW_SIZE, team_size)
    if picked_size <= 0:
        return []
    return list(permutations(range(1, team_size + 1), picked_size))


def team_preview_action_mask(battle) -> np.ndarray:
    mask = np.zeros(ACTION_DIM, dtype=np.bool_)
    orders = _teampreview_orders(battle)
    mask[: min(len(orders), ACTION_DIM)] = True
    return mask


def team_preview_order_from_action(battle, action_id: int) -> str:
    orders = _teampreview_orders(battle)
    if not orders:
        return "/team"
    action_id = int(action_id)
    if action_id < 0 or action_id >= min(len(orders), ACTION_DIM):
        action_id = random.randrange(min(len(orders), ACTION_DIM))
    return "/team " + "".join(str(index) for index in orders[action_id])


def random_team_preview_order(battle) -> str:
    orders = _teampreview_orders(battle)
    if not orders:
        return "/team"
    return "/team " + "".join(str(index) for index in random.choice(orders))


def _slot_order(battle, slot: int, slot_action: int):
    kind, index, target = _decode_slot_action(slot_action)
    active = battle.active_pokemon[slot]

    if kind == "pass":
        return None
    if kind == "move":
        moves = battle.available_moves[slot]
        if active is None or index >= len(moves):
            raise ValueError("Unavailable move")
        move = moves[index]
        if target not in battle.get_possible_showdown_targets(move, active):
            raise ValueError("Unavailable move target")
        return BattleOrder(move, move_target=target)

    switches = battle.available_switches[slot]
    if index >= len(switches):
        raise ValueError("Unavailable switch")
    return BattleOrder(switches[index])


def legal_action_mask(battle, *, fake_out_allowed=None) -> np.ndarray:
    if getattr(battle, "teampreview", False):
        return team_preview_action_mask(battle)

    mask = np.zeros(ACTION_DIM, dtype=np.bool_)
    force_switch = list(getattr(battle, "force_switch", [False, False]))
    replacement_turn = any(force_switch)
    # A slot has only 27 choices. Build each once rather than reconstructing
    # the same order and querying poke-env's properties for every joint action.
    unavailable = object()
    slot_orders = []
    for slot in range(2):
        orders = []
        for slot_action in range(SLOT_ACTION_DIM):
            try:
                order = _slot_order(battle, slot, slot_action)
                if (fake_out_allowed is not None and not fake_out_allowed[slot]
                        and slot_action < SWITCH_ACTION_START and order is not None
                        and getattr(order.order, "id", None) == "fakeout"):
                    orders.append(unavailable)
                else:
                    orders.append(order)
            except Exception:
                orders.append(unavailable)
        slot_orders.append(orders)
    available_switches = battle.available_switches
    active_pokemon = battle.active_pokemon

    for action_id in range(ACTION_DIM):
        left_action, right_action = decode_action(action_id)
        try:
            left_order = slot_orders[0][left_action]
            right_order = slot_orders[1][right_action]
            if left_order is unavailable or right_order is unavailable:
                continue
            orders = (left_order, right_order)
            valid = True

            for slot, order in enumerate(orders):
                must_switch = slot < len(force_switch) and force_switch[slot]
                if must_switch:
                    switches = available_switches[slot]
                    if order is None:
                        # Both slots may request replacement with only one bench
                        # Pokemon left. Pass only when the other slot uses the
                        # last available replacement; never skip an unused one.
                        other = orders[1 - slot]
                        other_action = (right_action, left_action)[slot]
                        valid = valid and all(
                            other is not None
                            and other_action >= SWITCH_ACTION_START
                            and other.order == candidate
                            for candidate in switches
                        )
                    elif switches:
                        valid = (
                            valid
                            and (left_action, right_action)[slot] >= SWITCH_ACTION_START
                            and order.order in available_switches[slot]
                        )
                    else:
                        valid = False
                elif replacement_turn:
                    valid = valid and order is None
                elif active_pokemon[slot] is None:
                    valid = valid and order is None
                else:
                    valid = valid and order is not None

            if (
                valid
                and left_order is not None
                and right_order is not None
                and left_action >= SWITCH_ACTION_START
                and right_action >= SWITCH_ACTION_START
                and left_order.order == right_order.order
            ):
                valid = False

            mask[action_id] = valid
        except Exception:
            continue

    return mask


def action_to_order(player, battle, action_id: int, *, action_mask=None):
    if getattr(battle, "teampreview", False):
        return team_preview_order_from_action(battle, action_id)

    try:
        action_id = int(action_id) % ACTION_DIM
        # The supplied mask belongs to this immediate decision, never a later
        # request. Other callers retain the checked, standalone conversion API.
        mask = legal_action_mask(battle) if action_mask is None else np.asarray(action_mask)
        if mask.shape != (ACTION_DIM,) or not mask[action_id]:
            raise ValueError("Illegal joint action")
        left_action, right_action = decode_action(action_id)
        return DoubleBattleOrder(
            _slot_order(battle, 0, left_action),
            _slot_order(battle, 1, right_action),
        )
    except Exception as exc:
        if hasattr(player, "record_training_error"):
            player.record_training_error(battle, "action_to_order_fallback", exc)
        return player.choose_random_doubles_move(battle)
