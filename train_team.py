"""Train with uniformly sampled fixed teams: 100 battles x 3 opponents = 300."""

import asyncio
import random
from pathlib import Path

from config import LOCAL_SERVER, TRAIN_FORMAT, PPOConfig
from max_damage_player import MaxDamageDoublesPlayer
from players import FixedTeamRandomDoublesPlayer, PPODoublesPlayer
from ppo import PPOAgent
from team_loader import load_team
from train import (finalize_finished_battles, load_existing_model,
                   run_training_battle, stop_players, training_account,
                   update_and_save)
from training_errors import TrainingErrorRecorder


BATTLES_PER_OPPONENT = 3000
OPPONENT_TYPES = ("random", "maxdamage", "selfplay")
TEAM_DIRECTORY = Path(__file__).resolve().parent


def load_teams():
    # Resolve relative to this script, regardless of the working directory.
    return {f"team{i}": load_team(TEAM_DIRECTORY / f"team{i}.txt")
            for i in range(1, 6)}


async def train_stage(agent, cfg, teams, opponent_type, rng,
                      battles=BATTLES_PER_OPPONENT):
    if opponent_type not in OPPONENT_TYPES:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
    if battles <= 0:
        return
    if cfg.update_every_episodes <= 0:
        raise ValueError("update_every_episodes must be positive")
    if not teams:
        raise ValueError("At least one team is required")

    names = tuple(teams)
    recorder = TrainingErrorRecorder(cfg.error_log_path)
    players = []
    label = opponent_type
    common = dict(error_recorder=recorder, battle_format=TRAIN_FORMAT,
                  server_configuration=LOCAL_SERVER, max_concurrent_battles=1)
    try:
        learner = PPODoublesPlayer(
            account_configuration=training_account("teamppo"),
            ppo_agent=agent, training=True, team=teams[names[0]], **common)
        players.append(learner)
        opponent_class = {"random": FixedTeamRandomDoublesPlayer,
                          "maxdamage": MaxDamageDoublesPlayer,
                          "selfplay": PPODoublesPlayer}[opponent_type]
        extra = dict(ppo_agent=agent, training=True) if opponent_type == "selfplay" else {}
        opponent = opponent_class(
            account_configuration=training_account("teamopp"),
            team=teams[names[0]], **extra, **common)
        players.append(opponent)
        learners = players if opponent_type == "selfplay" else [learner]

        for finished in range(1, battles + 1):
            # Independent uniform draws with replacement allow mirror matches.
            learner_name = rng.choice(names)
            opponent_name = rng.choice(names)
            learner.update_team(teams[learner_name])
            opponent.update_team(teams[opponent_name])
            print(f"[{label}] battle={finished}/{battles} | "
                  f"learner={learner_name} vs opponent={opponent_name}", flush=True)
            await run_training_battle(learner, opponent, cfg)
            for player in learners:
                finalize_finished_battles(player)
            if finished % cfg.update_every_episodes == 0:
                size = update_and_save(agent, cfg, learners)
                print(f"[{label}] battles={finished}/{battles} | buffer={size}")

        if any(len(player.buffer) > 0 for player in learners):
            update_and_save(agent, cfg, learners)
        print(f"[{label}] completed {battles} battles")
    finally:
        await stop_players(*players)


async def train_random_teams(agent, cfg, teams, battles=BATTLES_PER_OPPONENT):
    rng = random.Random(cfg.seed)
    total = len(OPPONENT_TYPES) * battles
    print(f"starting random fixed-team training: {total} battles")
    for opponent_type in OPPONENT_TYPES:
        await train_stage(agent, cfg, teams, opponent_type, rng, battles)


async def main():
    teams = load_teams()
    cfg = PPOConfig(update_every_episodes=50,
                    model_path="ppo_doubles_full_actions.pt")
    agent = PPOAgent(cfg)
    print(f"training error log: {cfg.error_log_path}")
    load_existing_model(agent, cfg)
    await train_random_teams(agent, cfg, teams)
    agent.save(cfg.model_path)
    print(f"saved: {cfg.model_path}")


if __name__ == "__main__":
    asyncio.run(main())
