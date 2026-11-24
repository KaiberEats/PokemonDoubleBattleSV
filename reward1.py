"""Conservative public-event penalties and bounded board-potential shaping."""
import json
from pathlib import Path

from poke_env.data import GenData, to_id_str
from players import PPODoublesPlayer
from rich_observation import BattleFlow

PROTECT = {"protect", "detect", "spikyshield", "banefulbunker", "silktrap",
           "burningbulwark", "kingsshield", "obstruct", "wideguard", "quickguard"}
SINGLE = {"normal", "adjacentFoe", "any"}
BENEFICIAL_ABILITIES = {"justified", "rattled", "stamina", "watercompaction",
                       "steamengine", "angerpoint", "weakarmor", "berserk"}


class RewardEvents:
    def __init__(self):
        self.flow = BattleFlow()
        self.pending = []
        self.turn = 0
        self.turn_switches = {}
        self.turn_protect = {}

    def record(self, messages):
        self.flow.record(messages)
        for msg in messages:
            if len(msg) < 2:
                continue
            if msg[1] == "turn":
                self.turn = int(msg[2])
            self.pending.append((self.turn, tuple(msg)))
            if msg[1] in ("switch", "drag"):
                self.turn_switches.setdefault(self.turn, set()).add(msg[2][:3])
            if msg[1] == "move" and to_id_str(msg[3]) in PROTECT:
                self.turn_protect.setdefault(self.turn, set()).add(msg[2][:2])

    def penalties(self, battle):
        role = battle.player_role
        data = GenData.from_gen(9).moves
        immune = ally = 0
        current = None
        seen = set()
        for index, (turn, msg) in enumerate(self.pending):
            event = msg[1]
            if event == "move":
                current = (index, turn, msg)
            elif event in ("turn", "switch", "drag", "upkeep", "cant"):
                current = None
            elif event in ("-immune", "-damage") and current is not None:
                key, move_turn, move = current
                if len(move) < 5 or move[2][:2] != role or key in seen:
                    continue
                target = move[4][:3]
                details = data.get(to_id_str(move[3]), {})
                if details.get("target") not in SINGLE or msg[2][:3] != target:
                    continue
                # Secondary residual damage, weather, recoil, etc. are not hits.
                if event == "-damage" and any(part.startswith("[from]") for part in msg[3:]):
                    continue
                if event == "-immune" and target[:2] != role:
                    if (target not in self.turn_switches.get(move_turn, set())
                            and target[:2] not in self.turn_protect.get(move_turn, set())):
                        immune += 1
                        seen.add(key)
                elif event == "-damage" and target[:2] == role and target != move[2][:3]:
                    # Restrict to direct damaging moves. Exempt recognizable
                    # intentional ability/item activation; avoid guessing intent.
                    if details.get("category") not in ("Physical", "Special"):
                        continue
                    slot = 0 if target.endswith("a") else 1
                    active = list(battle.active_pokemon or [])
                    mon = active[slot] if slot < len(active) else None
                    ability = to_id_str(str(getattr(mon, "ability", "") or ""))
                    item = to_id_str(str(getattr(mon, "item", "") or ""))
                    # Consumed items can already be absent from the parsed
                    # Pokemon at the next decision. Check their public events.
                    activated_policy = any(
                        len(follow) > 3 and follow[1] == "-enditem"
                        and follow[2][:3] == target
                        and to_id_str(follow[3]) == "weaknesspolicy"
                        for t, follow in self.pending[index:] if t == move_turn)
                    if ability in BENEFICIAL_ABILITIES or item == "weaknesspolicy" or activated_policy:
                        continue
                    ally += 1
                    seen.add(key)
        self.pending.clear()
        # Keep current-turn context through forced replacement requests.
        self.turn_switches = {t: v for t, v in self.turn_switches.items() if t >= self.turn}
        self.turn_protect = {t: v for t, v in self.turn_protect.items() if t >= self.turn}
        return {"ally_attack": -5.0 * ally, "single_immune": -3.0 * immune}


def board_potential(battle, flow):
    """0..10 own board utility minus opponent utility: total -10..10.

    Durations use public lower bounds (no assumed weather rocks/light clay).
    This is a heuristic, not an exact damage or win-probability estimator.
    """
    def utility(team, side):
        mons = [m for m in team.values() if not m.fainted]
        if not mons:
            return 0.0
        value = sum(float(m.current_hp_fraction) for m in mons) / 4 * 4
        turn = battle.turn
        for (owner, name), (start, lower, upper, layers) in flow.timers.items():
            remaining = max(0, lower - max(0, turn - max(1, start)))
            fraction = remaining / max(1, lower)
            if owner == side and name in ("reflect", "lightscreen", "auroraveil", "tailwind"):
                value += fraction
            elif owner == "global" and name in ("raindance", "sunnyday"):
                good, bad = (("WATER", "FIRE") if name == "raindance" else ("FIRE", "WATER"))
                balance = 0
                for mon in mons:
                    types = {getattr(getattr(move, "type", None), "name", "")
                             for move in (getattr(mon, "moves", {}) or {}).values()}
                    balance += int(good in types) - int(bad in types)
                value += fraction * balance / 4
            elif owner == "global" and name == "trickroom":
                # Base speed is public; actual EVs/opponent speed are unknown.
                slow = sum((80 - (getattr(m, "base_stats", {}) or {}).get("spe", 80)) / 80 for m in mons) / 4
                value += fraction * slow
        return max(0.0, min(10.0, value))
    role = battle.player_role
    other = "p2" if role == "p1" else "p1"
    return utility(battle.team, role) - utility(battle.opponent_team, other)


class Reward1Player(PPODoublesPlayer):
    def __init__(self, *args, reward_log_path, gamma=.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_log_path = Path(reward_log_path)
        self.reward_gamma = gamma
        self.reward_events = {}
        self.reward_previous = {}
        self.reward_totals = {}

    async def _handle_battle_message(self, split_messages):
        tag = split_messages[0][0].lstrip(">")
        self.reward_events.setdefault(tag, RewardEvents()).record(split_messages[1:])
        return await super()._handle_battle_message(split_messages)

    def _choose_move(self, battle):
        tag = battle.battle_tag
        if tag not in self._last_obs:
            tracker = self.reward_events.setdefault(tag, RewardEvents())
            self.reward_previous[tag] = board_potential(battle, tracker.flow)
            tracker.pending.clear()  # No action yet: no action penalties.
        return super()._choose_move(battle)

    def result_reward(self, battle, reward):
        return reward * 100.0

    def transition_reward(self, battle):
        tag = battle.battle_tag
        tracker = self.reward_events.setdefault(tag, RewardEvents())
        parts = tracker.penalties(battle)
        # Terminal potential is zero; avoids paying for ending with a buff.
        current = 0.0 if battle.finished else board_potential(battle, tracker.flow)
        previous = self.reward_previous.get(tag, current)
        parts["board"] = self.reward_gamma * current - previous
        self.reward_previous[tag] = current
        totals = self.reward_totals.setdefault(tag, {"ally_attack": 0., "single_immune": 0., "board": 0.})
        for name, value in parts.items():
            totals[name] += value
        return sum(parts.values())

    def finalize_battle(self, battle):
        tag = battle.battle_tag
        pending = tag in self._last_obs or tag in self._selection_decisions
        super().finalize_battle(battle)
        if pending:
            totals = self.reward_totals.pop(tag, {})
            terminal = 100.0 if battle.won else -100.0 if battle.lost else 0.0
            self.reward_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.reward_log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(battle=tag, terminal=terminal, **totals)) + "\n")
            self.reward_events.pop(tag, None)
            self.reward_previous.pop(tag, None)
