"""Version 2 observations: public ordered protocol, timers and own move sets.

The complete public protocol is retained per battle (requests are excluded).
Its fixed-width projection is lossy, not a lossless transcript or a damage engine.
Unknown opposing items/durations/conditions must never be treated as known.
"""
import hashlib
import json
import math
from functools import lru_cache

import numpy as np
from poke_env.data import GenData, to_id_str

HISTORY_DIM = 256
STATIC_MOVE_DIM = 128
DYNAMIC_MOVE_DIM = 64 + STATIC_MOVE_DIM
SET_DIM = 4 * STATIC_MOVE_DIM + 64
TIMER_NAMES = ("raindance", "sunnyday", "sandstorm", "snow", "hail",
               "trickroom", "electricterrain", "grassyterrain", "mistyterrain",
               "psychicterrain", "reflect", "lightscreen", "auroraveil", "tailwind",
               "stealthrock", "spikes", "toxicspikes", "stickyweb")
TIMER_DIM = len(TIMER_NAMES) * 3 * 5
RICH_DIM = HISTORY_DIM + TIMER_DIM + 8 * DYNAMIC_MOVE_DIM + 6 * SET_DIM


def sketch(tokens, width):
    """Stable signed feature hashing, independent of Python hash randomization."""
    out = np.zeros(width, dtype=np.float32)
    for token in tokens:
        digest = hashlib.blake2b(str(token).encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        out[value % width] += 1 if value & (1 << 63) else -1
    return out


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from flatten(value[key], prefix + "/" + key)
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            yield from flatten(child, prefix + "/" + str(i))
    else:
        yield prefix + "=" + str(value)


@lru_cache(maxsize=2048)
def static_move(move_id):
    data = GenData.from_gen(9).moves.get(move_id, {})
    out = np.zeros(STATIC_MOVE_DIM, dtype=np.float32)
    if not data:
        return out
    out[0] = 1
    out[1:4] = [float(data.get("category") == kind) for kind in ("Physical", "Special", "Status")]
    out[4] = data.get("basePower", 0) / 250
    out[5] = data.get("pp", 0) / 64
    out[6] = (data.get("priority", 0) + 7) / 14
    out[7] = float(data.get("accuracy", True)) / (100 if data.get("accuracy") is not True else 1)
    out[8] = float("basePowerCallback" in data)
    out[9] = float("onTry" in data or "onTryHit" in data)
    out[10] = float(bool(data.get("secondary") or data.get("secondaries")))
    out[11] = float(bool(data.get("selfSwitch") or data.get("forceSwitch")))
    # Includes type, target, boosts, statuses, drain, recoil, flags, callback
    # presence and move identity. Callback names are not executable semantics.
    out[12:] = np.tanh(sketch(["id=" + move_id, *flatten(data)], STATIC_MOVE_DIM - 12))
    return out


def own_set(mon):
    out = np.zeros(SET_DIM, dtype=np.float32)
    if mon is None:
        return out
    moves = list((getattr(mon, "moves", {}) or {}).keys())
    for i, move in enumerate(moves[:4]):
        out[i * STATIC_MOVE_DIM:(i + 1) * STATIC_MOVE_DIM] = static_move(to_id_str(move))
    offset = 4 * STATIC_MOVE_DIM
    out[offset:offset + 32] = sketch(["item=" + str(getattr(mon, "item", None))], 32)
    out[offset + 32:] = sketch(["ability=" + str(getattr(mon, "ability", None))], 32)
    return out


class BattleFlow:
    def __init__(self):
        self.events = []
        self.turn = 0
        self.timers = {}
        self.entries = {}
        self.last_moves = {}
        self.protect_streak = {}
        self.memory = {role: np.zeros(HISTORY_DIM, dtype=np.float64) for role in ("p1", "p2")}
        self.between_turns = True

    def record(self, messages):
        for message in messages:
            if len(message) < 2 or message[1] in ("request", "uhtml", "uhtmlchange", "j", "l", "t:"):
                continue
            msg = tuple(message)
            event = msg[1]
            if event == "turn":
                self.turn = int(msg[2])
                self.between_turns = False
            elif event == "upkeep":
                self.between_turns = True
            self.events.append((self.turn, msg))
            # All events participate, including actor/target, move, fail, miss,
            # immune, cant, damage, effects and switch order. No recent-N cutoff.
            position = len(self.events)
            tokens = [f"{i}:{part}" for i, part in enumerate(msg[1:])]
            tokens += ["previous=" + json.dumps(self.events[-2][1]) if position > 1 else "start"]
            for role in ("p1", "p2"):
                other = "p2" if role == "p1" else "p1"
                relative = [token.replace(role, "own").replace(other, "opponent") for token in tokens]
                base = sketch(relative, HISTORY_DIM // 2)
                self.memory[role][:HISTORY_DIM // 2] += base
                self.memory[role][HISTORY_DIM // 2:] += base * np.sin(position * np.arange(1, HISTORY_DIM // 2 + 1) * 0.017)
            if event in ("switch", "drag"):
                self.entries[msg[2][:3]] = max(1, self.turn + int(self.between_turns))
                self.last_moves.pop(msg[2][:3], None)
                self.protect_streak[msg[2][:3]] = 0
            elif event == "move":
                actor, move = msg[2][:3], to_id_str(msg[3])
                self.last_moves[actor] = (self.turn, move)
                self.protect_streak[actor] = self.protect_streak.get(actor, 0) + 1 if move in ("protect", "detect", "spikyshield", "banefulbunker", "silktrap", "burningbulwark") else 0
            elif event == "-weather" and "[upkeep]" not in msg:
                self.timers = {k: v for k, v in self.timers.items() if k[1] not in TIMER_NAMES[:5]}
                name = to_id_str(msg[2])
                if name in TIMER_NAMES[:5]:
                    self.timers[("global", name)] = (self.turn, 5, 8, 1)
            elif event in ("-fieldstart", "-fieldend", "-sidestart", "-sideend"):
                is_side = event.startswith("-side")
                side = msg[2][:2] if is_side else "global"
                name = to_id_str(msg[3 if is_side else 2].removeprefix("move: "))
                key = (side, name)
                if event.endswith("end"):
                    self.timers.pop(key, None)
                elif name in TIMER_NAMES:
                    if name in ("spikes", "toxicspikes"):
                        layers = min(3 if name == "spikes" else 2, self.timers.get(key, (0, 0, 0, 0))[3] + 1)
                        self.timers[key] = (self.turn, 0, 0, layers)
                    else:
                        duration = 4 if name == "tailwind" else 5
                        upper = 8 if name in ("reflect", "lightscreen", "auroraveil", "electricterrain", "grassyterrain", "mistyterrain", "psychicterrain") else duration
                        self.timers[key] = (self.turn, duration, upper, 1)

    def features(self, battle):
        out = np.zeros(HISTORY_DIM + TIMER_DIM, dtype=np.float32)
        role = getattr(battle, "player_role", "p1")
        role = role if role in ("p1", "p2") else "p1"
        out[:HISTORY_DIM] = np.tanh(self.memory[role] / math.sqrt(max(1, len(self.events))))
        sides = ("global", role, "p2" if role == "p1" else "p1")
        turn = getattr(battle, "turn", self.turn)
        for s, side in enumerate(sides):
            for n, name in enumerate(TIMER_NAMES):
                value = self.timers.get((side, name))
                if value is None:
                    continue
                start, lower, upper, layers = value
                # Start before |turn|1 counts turn1 as the first affected turn.
                elapsed = max(0, turn - max(1, start))
                remaining_low, remaining_high = max(0, lower - elapsed), max(0, upper - elapsed)
                index = HISTORY_DIM + (s * len(TIMER_NAMES) + n) * 5
                out[index:index + 5] = [1, remaining_low / 8, remaining_high / 8,
                                      float(lower != upper), layers / 3]
        return out


def rich_features(battle, flow=None):
    out = np.zeros(RICH_DIM, dtype=np.float32)
    if flow is not None:
        out[:HISTORY_DIM + TIMER_DIM] = flow.features(battle)
    active = getattr(battle, "active_pokemon", []) or []
    available = getattr(battle, "available_moves", []) or []
    role = getattr(battle, "player_role", "p1")
    start = HISTORY_DIM + TIMER_DIM
    for slot in range(2):
        mon = active[slot] if slot < len(active) else None
        moves = available[slot] if slot < len(available) else []
        for index, move in enumerate(moves[:4]):
            values = np.zeros(DYNAMIC_MOVE_DIM, dtype=np.float32)
            move_id = getattr(move, "id", "")
            data = GenData.from_gen(9).moves.get(move_id, {})
            values[:3] = [data.get("category") == kind for kind in ("Physical", "Special", "Status")]
            pp, max_pp = getattr(move, "current_pp", 0), getattr(move, "max_pp", 0)
            values[3:6] = [pp / 64, max_pp / 64, pp / max(1, max_pp)]
            hp = getattr(mon, "current_hp_fraction", 0)
            values[6] = hp if move_id in ("eruption", "waterspout", "dragonenergy") else 1 - hp if move_id in ("flail", "reversal") else 0
            key = role + ("a" if slot == 0 else "b")
            if flow is not None:
                first_turn = max(1, flow.entries.get(key, -100)) == getattr(battle, "turn", 0)
                values[7:10] = [move_id in ("fakeout", "firstimpression"), first_turn,
                                flow.protect_streak.get(key, 0) / (1 + flow.protect_streak.get(key, 0))]
            values[10:13] = [move_id == "suckerpunch", move_id in ("protect", "detect"), float("basePowerCallback" in data)]
            values[13] = float(bool(data.get("secondary") or data.get("secondaries")))
            values[14] = float("onTry" in data or "onTryHit" in data)
            values[15] = float(bool(data.get("heal") or data.get("drain")))
            values[16:24] = np.tanh(sketch(flatten(data), 8))
            values[64:] = static_move(move_id)
            # Explicit secondary effect / self-effect summaries alongside the
            # complete hashed metadata. Values are move mechanics, not forecasts.
            secondary = data.get("secondary") or {}
            secondaries = data.get("secondaries") or ([secondary] if secondary else [])
            values[24] = max((effect.get("chance", 100) / 100 for effect in secondaries), default=0)
            values[25] = float(any(effect.get("volatileStatus") == "flinch" for effect in secondaries))
            values[26] = float(bool(data.get("status") or any(effect.get("status") for effect in secondaries)))
            for i, effect in enumerate(("drain", "recoil", "heal")):
                ratio = data.get(effect)
                if isinstance(ratio, (list, tuple)) and len(ratio) == 2:
                    values[27+i] = ratio[0] / max(1, ratio[1])
            values[30] = float(bool(data.get("forceSwitch")))
            values[31] = float(bool(data.get("selfSwitch")))
            values[32] = float(bool(data.get("selfdestruct")))
            stats = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
            boosts = data.get("boosts") or {}
            self_boosts = (data.get("self") or {}).get("boosts") or {}
            for i, stat in enumerate(stats):
                values[33+i] = boosts.get(stat, 0) / 6
                values[40+i] = self_boosts.get(stat, 0) / 6
            # Known conditions are separate from conditions requiring the
            # opponent's unrevealed choice. Unknown is never labelled success.
            values[47] = float(move_id in ("fakeout", "firstimpression", "snore", "sleeptalk"))
            if move_id in ("fakeout", "firstimpression") and flow is not None:
                values[48] = values[8]
            elif move_id in ("snore", "sleeptalk"):
                values[48] = float(str(getattr(getattr(mon, "status", None), "name", "")).lower() == "slp")
            values[49] = float(move_id in ("suckerpunch", "counter", "mirrorcoat", "metalburst", "focuspunch"))
            values[50] = float(bool(data.get("onTry") or data.get("onTryHit")) and not values[47] and not values[49])
            values[51] = float(bool(data.get("basePowerCallback")) and move_id not in ("eruption", "waterspout", "dragonenergy", "weatherball"))
            power = data.get("basePower", 0)
            if move_id in ("eruption", "waterspout", "dragonenergy"):
                power = max(1, int(150 * hp))
            elif move_id == "weatherball" and getattr(battle, "weather", {}):
                power = 100
            values[52] = power / 250
            values[53] = float(move_id in ("eruption", "waterspout", "dragonenergy", "weatherball"))
            flags = data.get("flags") or {}
            for i, flag in enumerate(("contact", "sound", "bullet", "powder", "punch", "protect")):
                values[54+i] = bool(flags.get(flag))
            values[60] = float(bool(data.get("multihit")))
            values[61] = float(bool(data.get("weather") or data.get("terrain") or data.get("pseudoWeather")))
            values[62] = float(bool(data.get("sideCondition")))
            values[63] = float(bool(data.get("volatileStatus")))
            out[start:start + DYNAMIC_MOVE_DIM] = values
            start += DYNAMIC_MOVE_DIM
        start = HISTORY_DIM + TIMER_DIM + (slot + 1) * 4 * DYNAMIC_MOVE_DIM
    start = HISTORY_DIM + TIMER_DIM + 8 * DYNAMIC_MOVE_DIM
    for i, mon in enumerate(list((getattr(battle, "team", {}) or {}).values())[:6]):
        out[start + i * SET_DIM:start + (i + 1) * SET_DIM] = own_set(mon)
    return out
