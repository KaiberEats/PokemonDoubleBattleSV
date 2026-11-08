from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path
from poke_env.ps_client import ServerConfiguration
from poke_env.data import GenData

LOCAL_SERVER = ServerConfiguration(
    "ws://localhost:8000/showdown/websocket",
    "http://localhost:8000/action.php?",
)
TRAIN_FORMAT = "gen9vgc2024regg"
EVAL_FORMAT = TRAIN_FORMAT
TEAM_PATH = Path(__file__).with_name("team1.txt")


def new_error_log_path():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return str(Path(__file__).with_name("logs") / f"training_errors_{stamp}_{uuid4().hex[:8]}.jsonl")

@dataclass
class PPOConfig:
    total_battles: int = 100
    max_damage_battles: int = 1000
    self_play_battles: int = 100
    update_every_episodes: int = 10
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    epochs: int = 4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden_dim: int = 256
    seed: int = 42
    model_path: str = "ppo_doubles_full_actions.pt"
    training_bans_path: str = str(Path(__file__).with_name("training_bans.json"))
    error_log_path: str = field(default_factory=new_error_log_path)
    battle_timeout_seconds: float = 180.0

# 1スロット: pass 1 + 技4 × 対象5 + 交代6 = 27。2スロットで729。
SLOT_ACTION_DIM = 27
ACTION_DIM = SLOT_ACTION_DIM * SLOT_ACTION_DIM
LEGACY_OBS_DIM = 768
SPECIES_FEATURE_DIM = 1 + 11 + 18 + 6  # present, dex-number bits, types, base stats
SPECIES_OBS_DIM = LEGACY_OBS_DIM + 1 + 12 * SPECIES_FEATURE_DIM
# Sorted IDs give every move a distinct, reproducible feature; final slot is unknown.
HISTORY_MOVE_IDS = tuple(sorted(GenData.from_gen(9).moves))
MOVE_HISTORY_DIM = 2 * (len(HISTORY_MOVE_IDS) + 1)
MOVE_HISTORY_OBS_DIM = SPECIES_OBS_DIM + MOVE_HISTORY_DIM
# Four active slots: presence, entry/elapsed/switch turns, change/forced flags,
# and the outgoing/incoming species features.
SWITCH_SLOT_DIM = 6 + 2 * SPECIES_FEATURE_DIM
SWITCH_HISTORY_DIM = 4 * SWITCH_SLOT_DIM
OBS_DIM = MOVE_HISTORY_OBS_DIM + SWITCH_HISTORY_DIM
