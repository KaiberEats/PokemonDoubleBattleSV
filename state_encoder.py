import numpy as np
from types import SimpleNamespace
from config import (OBS_DIM, LEGACY_OBS_DIM, SPECIES_FEATURE_DIM,
                    HISTORY_MOVE_IDS, MOVE_HISTORY_DIM, SWITCH_SLOT_DIM, SWITCH_HISTORY_DIM)
from poke_env.data import GenData, to_id_str

STATUS_LIST = ["brn", "frz", "par", "psn", "slp", "tox", "fnt", None]

WEATHERS = [
    "raindance", "sunnyday", "sandstorm", "hail", "snow",
    "desolateland", "primordialsea", "deltastream",
]

FIELDS = [
    "trickroom", "electricterrain", "grassyterrain",
    "mistyterrain", "psychicterrain",
]

SIDE_CONDITIONS = [
    "stealthrock", "spikes", "toxicspikes", "stickyweb",
    "reflect", "lightscreen", "auroraveil", "tailwind",
]

TYPES = [
    "BUG", "DARK", "DRAGON", "ELECTRIC", "FAIRY", "FIGHTING",
    "FIRE", "FLYING", "GHOST", "GRASS", "GROUND", "ICE",
    "NORMAL", "POISON", "PSYCHIC", "ROCK", "STEEL", "WATER",
]

MOVE_FEATURE_DIM = 1 + 1 + len(TYPES) + 1 + 2 + 1 + 1

def _safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def _safe_attr(obj, name, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default

def _hp_fraction(pokemon):
    if pokemon is None:
        return 0.0
    return _safe_float(getattr(pokemon, "current_hp_fraction", 0.0), 0.0)

def _status_onehot(pokemon):
    arr = np.zeros(len(STATUS_LIST), dtype=np.float32)
    if pokemon is None:
        arr[-1] = 1.0
        return arr
    status = getattr(pokemon, "status", None)
    key = getattr(status, "name", status)
    if isinstance(key, str):
        key = key.lower()
    if key in STATUS_LIST:
        arr[STATUS_LIST.index(key)] = 1.0
    else:
        arr[-1] = 1.0
    return arr

def _boosts(pokemon):
    stats = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
    arr = []
    boosts = getattr(pokemon, "boosts", {}) or {}
    for s in stats:
        # -6〜+6 を 0〜1 に正規化
        arr.append((_safe_float(boosts.get(s, 0)) + 6.0) / 12.0)
    return np.array(arr, dtype=np.float32)

def _pokemon_features(pokemon):
    if pokemon is None:
        return np.zeros(1 + len(STATUS_LIST) + 7 + 2, dtype=np.float32)
    fainted = 1.0 if getattr(pokemon, "fainted", False) else 0.0
    active = 1.0 if getattr(pokemon, "active", False) else 0.0
    return np.concatenate([
        np.array([_hp_fraction(pokemon)], dtype=np.float32),
        _status_onehot(pokemon),
        _boosts(pokemon),
        np.array([fainted, active], dtype=np.float32),
    ])

def _contains_key(container, key: str) -> float:
    try:
        for x in container:
            name = getattr(x, "name", x)
            name = str(name).lower().replace("_", "").replace(" ", "")
            if key in name:
                return 1.0
    except Exception:
        pass
    return 0.0

def _field_features(battle):
    out = []

    weather = getattr(battle, "weather", None)
    for w in WEATHERS:
        out.append(_contains_key(weather or [], w))

    fields = getattr(battle, "fields", None)
    for f in FIELDS:
        out.append(_contains_key(fields or [], f))

    side = getattr(battle, "side_conditions", None)
    opp_side = getattr(battle, "opponent_side_conditions", None)
    for c in SIDE_CONDITIONS:
        out.append(_contains_key(side or [], c))
    for c in SIDE_CONDITIONS:
        out.append(_contains_key(opp_side or [], c))

    # ターン数をざっくり正規化
    turn = _safe_float(getattr(battle, "turn", 0))
    out.append(min(turn / 100.0, 1.0))

    return np.array(out, dtype=np.float32)

def _enum_name(value):
    return str(getattr(value, "name", getattr(value, "value", value))).upper()

def _move_type_onehot(move):
    arr = np.zeros(len(TYPES), dtype=np.float32)
    move_type = _enum_name(_safe_attr(move, "type", ""))
    if move_type in TYPES:
        arr[TYPES.index(move_type)] = 1.0
    return arr

def _accuracy(move):
    accuracy = _safe_attr(move, "accuracy", 1.0)
    if accuracy is True:
        return 1.0
    accuracy = _safe_float(accuracy, 1.0)
    if accuracy > 1:
        accuracy /= 100.0
    return max(0.0, min(1.0, accuracy))

def _stab(attacker, move):
    move_type = _safe_attr(move, "type", None)
    if move_type is None or attacker is None:
        return 0.0
    move_type_name = _enum_name(move_type)
    attacker_types = {
        _enum_name(getattr(attacker, "type_1", "")),
        _enum_name(getattr(attacker, "type_2", "")),
    }
    return float(move_type_name in attacker_types)

def _target_multiplier(target, move):
    if target is None or getattr(target, "fainted", False):
        return 0.0
    try:
        multiplier = _safe_float(target.damage_multiplier(move), 1.0)
    except Exception:
        multiplier = 1.0
    return max(0.0, min(4.0, multiplier)) / 4.0

def _move_features(battle, attacker, move):
    if attacker is None or move is None:
        return np.zeros(MOVE_FEATURE_DIM, dtype=np.float32)

    opp_active = getattr(battle, "opponent_active_pokemon", []) or []
    target_multipliers = [
        _target_multiplier(opp_active[i] if i < len(opp_active) else None, move)
        for i in range(2)
    ]
    base_power = max(0.0, min(250.0, _safe_float(_safe_attr(move, "base_power", 0), 0))) / 250.0
    priority = (_safe_float(_safe_attr(move, "priority", 0), 0) + 7.0) / 14.0
    priority = max(0.0, min(1.0, priority))

    return np.concatenate([
        np.array([1.0, base_power], dtype=np.float32),
        _move_type_onehot(move),
        np.array([_accuracy(move)], dtype=np.float32),
        np.array(target_multipliers, dtype=np.float32),
        np.array([_stab(attacker, move), priority], dtype=np.float32),
    ])

def _available_move_features(battle):
    feats = []
    active = getattr(battle, "active_pokemon", []) or []
    available_moves = getattr(battle, "available_moves", []) or []
    for slot in range(2):
        attacker = active[slot] if slot < len(active) else None
        moves = available_moves[slot] if slot < len(available_moves) else []
        for move_index in range(4):
            move = moves[move_index] if move_index < len(moves) else None
            feats.append(_move_features(battle, attacker, move))
    return np.concatenate(feats).astype(np.float32)

def species_features(pokemon):
    """Public species facts only; never infer the actual item/ability/moves/EVs."""
    result = np.zeros(SPECIES_FEATURE_DIM, dtype=np.float32)
    species = to_id_str(getattr(pokemon, "species", "") or "")
    entry = GenData.from_gen(9).pokedex.get(species)
    if not entry:
        return result
    result[0] = 1
    number = max(0, entry.get("num", 0))
    result[1:12] = [(number >> bit) & 1 for bit in range(11)]
    for kind in entry.get("types", []):
        result[12 + TYPES.index(kind.upper())] = 1
    result[30:36] = [entry["baseStats"][stat] / 255.0
                     for stat in ("hp", "atk", "def", "spa", "spd", "spe")]
    return result


def selection_features(battle):
    # Own order must match action_mapper._teampreview_orders. The opponent's
    # preview is a set in poke-env, so sort it for deterministic observations.
    own = list((getattr(battle, "team", {}) or {}).values())
    preview = getattr(battle, "teampreview_opponent_team", None)
    opponents = list(preview or (getattr(battle, "opponent_team", {}) or {}).values())
    opponents.sort(key=lambda mon: getattr(mon, "species", "") or "")
    features = [np.array([float(bool(getattr(battle, "teampreview", False)))], dtype=np.float32)]
    for team in (own, opponents):
        features.extend(species_features(team[i] if i < len(team) else None) for i in range(6))
    return np.concatenate(features)


MOVE_HISTORY_INDEX = {move: i for i, move in enumerate(HISTORY_MOVE_IDS)}


def move_history_features(battle, history=None):
    """Cumulative executed move counts, own side then opponent, scaled to [0, 1)."""
    result = np.zeros(MOVE_HISTORY_DIM, dtype=np.float32)
    role = getattr(battle, "player_role", None)
    if role not in ("p1", "p2"):
        return result
    width = len(HISTORY_MOVE_IDS) + 1
    for (side, move), count in (history or {}).items():
        index = MOVE_HISTORY_INDEX.get(move, len(HISTORY_MOVE_IDS))
        offset = 0 if side == role else width
        result[offset + index] += count
    return result / (1.0 + result)


def switch_history_features(battle, history=None):
    result = np.zeros(SWITCH_HISTORY_DIM, dtype=np.float32)
    role = getattr(battle, "player_role", None)
    if history is None or role not in ("p1", "p2"):
        return result
    other = "p2" if role == "p1" else "p1"
    turn = max(1, int(getattr(battle, "turn", 0)))
    for index, key in enumerate((role + "a", role + "b", other + "a", other + "b")):
        slot = history.slots.get(key)
        if not slot:
            continue
        start = index * SWITCH_SLOT_DIM
        # Saturation-free normalization preserves differences beyond turn 100.
        scale = lambda value: value / (1.0 + value)
        elapsed = max(0, turn - slot["entry_turn"]) if slot["present"] else 0
        result[start:start + 6] = [slot["present"], scale(slot["entry_turn"]),
            scale(elapsed), slot["changed"], scale(slot["switch_turn"]), slot["forced"]]
        result[start + 6:start + 6 + SPECIES_FEATURE_DIM] = species_features(
            SimpleNamespace(species=slot["outgoing"]))
        result[start + 6 + SPECIES_FEATURE_DIM:start + SWITCH_SLOT_DIM] = species_features(
            SimpleNamespace(species=slot["species"]))
    return result


def embed_battle(battle, move_history=None, switch_history=None) -> np.ndarray:
    """
    PPOに入れる固定長ベクトル。
    記憶したい情報：
    - 天候
    - フィールド/トリックルーム
    - ステルスロック/毒びし等の場の状態
    - 自分/相手のHP、状態異常、能力変化
    """
    feats = []

    # 自分の場に出ている2体
    active = getattr(battle, "active_pokemon", []) or []
    for i in range(2):
        feats.append(_pokemon_features(active[i] if i < len(active) else None))

    # 相手の場に出ている2体
    opp_active = getattr(battle, "opponent_active_pokemon", []) or []
    for i in range(2):
        feats.append(_pokemon_features(opp_active[i] if i < len(opp_active) else None))

    # 自分の手持ち最大6体
    team = list((getattr(battle, "team", {}) or {}).values())
    for i in range(6):
        feats.append(_pokemon_features(team[i] if i < len(team) else None))

    # 相手の判明済み手持ち最大6体
    opp_team = list((getattr(battle, "opponent_team", {}) or {}).values())
    for i in range(6):
        feats.append(_pokemon_features(opp_team[i] if i < len(opp_team) else None))

    feats.append(_field_features(battle))
    feats.append(_available_move_features(battle))

    vec = np.concatenate(feats).astype(np.float32)

    if vec.size < LEGACY_OBS_DIM:
        vec = np.pad(vec, (0, LEGACY_OBS_DIM - vec.size))
    else:
        vec = vec[:LEGACY_OBS_DIM]

    return np.concatenate([vec, selection_features(battle),
                           move_history_features(battle, move_history),
                           switch_history_features(battle, switch_history)]).astype(np.float32)
