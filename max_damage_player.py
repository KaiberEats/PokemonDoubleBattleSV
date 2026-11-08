from poke_env import Player
import random
from poke_env.player.battle_order import BattleOrder, DoubleBattleOrder
from training_errors import TrainingDiagnosticsMixin


OFFENSIVE_STAT_BY_CATEGORY = {
    "PHYSICAL": "atk",
    "SPECIAL": "spa",
}

DEFENSIVE_STAT_BY_CATEGORY = {
    "PHYSICAL": "def",
    "SPECIAL": "spd",
}


def _enum_name(value):
    return str(getattr(value, "name", getattr(value, "value", value))).upper()


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _boost_multiplier(stage):
    stage = max(-6, min(6, int(_safe_float(stage, 0))))
    if stage >= 0:
        return (2 + stage) / 2
    return 2 / (2 - stage)


def _accuracy_multiplier(move):
    accuracy = getattr(move, "accuracy", 1.0)
    if accuracy is True:
        return 1.0
    accuracy = _safe_float(accuracy, 1.0)
    if accuracy > 1:
        accuracy /= 100.0
    return max(0.0, min(1.0, accuracy))


def _expected_hits(move):
    return max(1.0, _safe_float(getattr(move, "expected_hits", 1), 1.0))


def _flag_enabled(flags, name):
    try:
        if isinstance(flags, dict):
            return bool(flags.get(name))
        return name in flags
    except Exception:
        return False


def _ability_multiplier(attacker, move, category_name):
    ability = str(getattr(attacker, "ability", "") or "").lower().replace(" ", "")
    base_power = _safe_float(getattr(move, "base_power", 0), 0)
    move_flags = getattr(move, "flags", {}) or {}
    move_id = str(getattr(move, "id", "") or "")

    multiplier = 1.0
    if ability == "adaptability" and _has_stab(attacker, move):
        multiplier *= 4 / 3
    if ability == "technician" and 0 < base_power <= 60:
        multiplier *= 1.5
    if ability in {"hugepower", "purepower"} and category_name == "PHYSICAL":
        multiplier *= 2.0
    if ability == "toughclaws" and _flag_enabled(move_flags, "contact"):
        multiplier *= 1.3
    if ability == "ironfist" and _flag_enabled(move_flags, "punch"):
        multiplier *= 1.2
    if ability == "strongjaw" and _flag_enabled(move_flags, "bite"):
        multiplier *= 1.5
    if ability == "megalauncher" and _flag_enabled(move_flags, "pulse"):
        multiplier *= 1.5
    if ability == "sharpness" and _flag_enabled(move_flags, "slicing"):
        multiplier *= 1.5
    if ability == "reckless" and getattr(move, "recoil", 0):
        multiplier *= 1.2
    if ability == "sheerforce" and getattr(move, "secondary", None):
        multiplier *= 1.3
    if ability == "gorillatactics" and category_name == "PHYSICAL":
        multiplier *= 1.5
    if ability == "supremeoverlord" and category_name in OFFENSIVE_STAT_BY_CATEGORY:
        multiplier *= 1.1
    if ability == "pixilate" and move_id in {"quickattack", "hypervoice", "facade", "doubleedge"}:
        multiplier *= 1.2

    return multiplier


def _has_stab(attacker, move):
    move_type = getattr(move, "type", None)
    return move_type is not None and move_type in (getattr(attacker, "type_1", None), getattr(attacker, "type_2", None))


def _stab_multiplier(attacker, move):
    if not _has_stab(attacker, move):
        return 1.0
    try:
        return _safe_float(attacker.stab_multiplier, 1.5)
    except Exception:
        return 1.5


def _stat_ratio(attacker, target, move):
    category_name = _enum_name(getattr(move, "category", ""))
    offensive_stat = OFFENSIVE_STAT_BY_CATEGORY.get(category_name)
    defensive_stat = DEFENSIVE_STAT_BY_CATEGORY.get(category_name)
    if offensive_stat is None or defensive_stat is None:
        return 1.0

    attacker_stats = getattr(attacker, "stats", {}) or {}
    target_stats = getattr(target, "stats", {}) or {}
    attacker_boosts = getattr(attacker, "boosts", {}) or {}
    target_boosts = getattr(target, "boosts", {}) or {}

    attack = _safe_float(attacker_stats.get(offensive_stat), 100.0)
    defense = _safe_float(target_stats.get(defensive_stat), 100.0)
    attack *= _boost_multiplier(attacker_boosts.get(offensive_stat, 0))
    defense *= _boost_multiplier(target_boosts.get(defensive_stat, 0))
    return attack / max(1.0, defense)


def _weather_multiplier(battle, move):
    weather = str(getattr(battle, "weather", "") or "").lower()
    move_type = str(getattr(getattr(move, "type", None), "name", getattr(move, "type", ""))).lower()
    if "rain" in weather or "primordial" in weather:
        if "water" in move_type:
            return 1.5
        if "fire" in move_type:
            return 0.5
    if "sun" in weather or "desolate" in weather:
        if "fire" in move_type:
            return 1.5
        if "water" in move_type:
            return 0.5
    return 1.0


def _target_for_showdown_position(battle, target_position):
    if target_position == 1:
        return _active_at(getattr(battle, "opponent_active_pokemon", []), 0)
    if target_position == 2:
        return _active_at(getattr(battle, "opponent_active_pokemon", []), 1)
    if target_position == -1:
        return _active_at(getattr(battle, "active_pokemon", []), 0)
    if target_position == -2:
        return _active_at(getattr(battle, "active_pokemon", []), 1)
    return None


def _active_at(active, index):
    try:
        if index < len(active):
            return active[index]
    except Exception:
        pass
    return None


def estimate_damage_score(battle, attacker, move, target_position):
    if not _safe_damage_target(move, target_position):
        return 0.0
    base_power = _safe_float(getattr(move, "base_power", 0), 0)
    if base_power <= 0:
        return 0.0

    target = _target_for_showdown_position(battle, target_position)
    targets = [target] if target is not None else list(getattr(battle, "opponent_active_pokemon", []) or [])
    targets = [pokemon for pokemon in targets if pokemon is not None and not getattr(pokemon, "fainted", False)]
    if not targets:
        return 0.0

    category_name = _enum_name(getattr(move, "category", ""))
    attacker_score = base_power
    attacker_score *= _accuracy_multiplier(move)
    attacker_score *= _expected_hits(move)
    attacker_score *= _stab_multiplier(attacker, move)
    attacker_score *= _ability_multiplier(attacker, move, category_name)
    attacker_score *= _weather_multiplier(battle, move)

    total = 0.0
    for target in targets:
        try:
            type_multiplier = _safe_float(target.damage_multiplier(move), 1.0)
        except Exception:
            type_multiplier = 1.0
        if type_multiplier <= 0:
            continue
        hp_fraction = max(0.05, _safe_float(getattr(target, "current_hp_fraction", 1.0), 1.0))
        ko_pressure = 1.0 / hp_fraction
        total += attacker_score * type_multiplier * _stat_ratio(attacker, target, move) * ko_pressure

    return total


def _safe_damage_target(move, target):
    # Ally-targeted support (e.g. Helping Hand) does not cause friendly fire.
    # It must remain selectable even when it will fail, so its PP can run out.
    if _enum_name(getattr(move, "category", "")) == "STATUS":
        return True
    # Negative positions explicitly select our side. Earthquake/Surf etc. also
    # hit allies even though their target position is zero.
    move_target = _enum_name(getattr(move, "deduced_target", ""))
    if target < 0 or move_target in {"ALL_ADJACENT", "ALL", "ADJACENT_ALLY", "ADJACENT_ALLY_OR_SELF"}:
        return False
    return move_target != "SELF" or _enum_name(getattr(move, "category", "")) == "STATUS"


def _best_switch_order(switches):
    if not switches:
        return None
    best_switch = max(
        switches,
        key=lambda pokemon: _safe_float(getattr(pokemon, "current_hp_fraction", 1.0), 1.0),
    )
    return BattleOrder(best_switch)


class MaxDamageDoublesPlayer(TrainingDiagnosticsMixin, Player):
    @staticmethod
    def choose_random_doubles_move(battle):
        """Fallback must obey the same no-friendly-fire rule as the scorer."""
        choices = [[], []]
        forced = any(battle.force_switch)
        for slot in range(2):
            active = _active_at(battle.active_pokemon, slot)
            if forced:
                if battle.force_switch[slot]:
                    choices[slot] = [BattleOrder(mon) for mon in battle.available_switches[slot]]
                continue
            if active is None:
                continue
            for move in battle.available_moves[slot]:
                try:
                    targets = battle.get_possible_showdown_targets(move, active)
                except Exception:
                    continue
                choices[slot].extend(BattleOrder(move, move_target=target)
                                     for target in targets if _safe_damage_target(move, target))
            choices[slot].extend(BattleOrder(mon) for mon in battle.available_switches[slot])
        for slot, available in enumerate(choices):
            if (not forced and _active_at(battle.active_pokemon, slot) is not None
                    and not available):
                raise ValueError("No action available without friendly fire")
        orders = []
        for left in choices[0] or [None]:
            for right in choices[1] or [None]:
                if (left is not None and right is not None
                        and left.order in battle.available_switches[0]
                        and right.order in battle.available_switches[1]
                        and left.order == right.order):
                    continue
                orders.append(DoubleBattleOrder(left, right))
        if not orders:
            raise ValueError("No joint action available without friendly fire")
        return random.choice(orders)

    def _choose_move(self, battle):
        order = self._choose_scored_move(battle)
        if order is not None:
            return order
        # Run fallback once. Its own failure must propagate rather than retrying
        # the identical failing operation through the scorer's exception path.
        return self.choose_random_doubles_move(battle)

    def _choose_scored_move(self, battle):
        try:
            if getattr(battle, "teampreview", False):
                return self.random_teampreview(battle)

            if any(getattr(battle, "force_switch", [False, False])):
                first_order = None
                second_order = None
                if battle.force_switch[0]:
                    first_order = _best_switch_order(battle.available_switches[0])
                if battle.force_switch[1]:
                    second_switches = [
                        pokemon
                        for pokemon in battle.available_switches[1]
                        if first_order is None or pokemon != first_order.order
                    ]
                    second_order = _best_switch_order(second_switches)
                return DoubleBattleOrder(first_order, second_order)

            orders = []
            for slot in range(2):
                orders.append(self._best_slot_order(battle, slot))

            for slot, active in enumerate(getattr(battle, "active_pokemon", []) or []):
                if active is not None and not getattr(active, "fainted", False) and orders[slot] is None:
                    return None

            if (
                orders[0] is not None
                and orders[1] is not None
                and orders[0].order in battle.available_switches[0]
                and orders[1].order in battle.available_switches[1]
                and orders[0].order == orders[1].order
            ):
                return None

            if orders[0] is None and orders[1] is None:
                return None
            return DoubleBattleOrder(orders[0], orders[1])
        except Exception as exc:
            self.record_training_error(battle, "max_damage_fallback", exc)
            return None

    def _best_slot_order(self, battle, slot):
        active = _active_at(getattr(battle, "active_pokemon", []), slot)
        if active is None:
            return None

        best_order = None
        best_score = float("-inf")
        moves = battle.available_moves[slot] if slot < len(battle.available_moves) else []

        for move in moves:
            try:
                targets = battle.get_possible_showdown_targets(move, active)
            except Exception as exc:
                self.record_training_error(battle, "move_target_exception", exc,
                                           related_moves=[getattr(move, "id", "unknown")])
                targets = []
            for target in targets:
                if not _safe_damage_target(move, target):
                    continue
                try:
                    score = estimate_damage_score(battle, active, move, target)
                except Exception as exc:
                    self.record_training_error(battle, "move_score_exception", exc,
                                               related_moves=[getattr(move, "id", "unknown")])
                    raise
                if score > best_score:
                    best_score = score
                    best_order = BattleOrder(move, move_target=target)

        if best_order is not None and best_score > 0:
            return best_order
        switches = battle.available_switches[slot] if slot < len(battle.available_switches) else []
        # Preserve a legal zero-damage move when switching is impossible.
        return _best_switch_order(switches) or best_order
