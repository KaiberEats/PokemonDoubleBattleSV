"""Collect concurrent fixed-team battles, then update one shared PPO model."""

import asyncio
import argparse
import random
from collections import deque
from pathlib import Path

from config import LOCAL_SERVER, TRAIN_FORMAT, PPOConfig
from max_damage_player import MaxDamageDoublesPlayer
from players import FixedTeamRandomDoublesPlayer, PPODoublesPlayer
from ppo import PPOAgent, RolloutBuffer
from team_loader import load_team
from train import (finalize_finished_battles, load_existing_model,
                   run_training_battle, stop_players, training_account,
                   update_and_save)
from training_errors import TrainingErrorRecorder
from performance_timing import TIMINGS


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
    parallel = getattr(cfg, "parallel_battles", 5)
    if not isinstance(parallel, int) or isinstance(parallel, bool) or parallel <= 0:
        raise ValueError("parallel_battles must be a positive integer")

    names = tuple(teams)
    recorder = TrainingErrorRecorder(cfg.error_log_path)
    players = []
    workers = []
    learners = []
    tasks = []
    label = opponent_type
    common = dict(error_recorder=recorder, battle_format=TRAIN_FORMAT,
                  server_configuration=LOCAL_SERVER, max_concurrent_battles=1)
    try:
        opponent_class = {"random": FixedTeamRandomDoublesPlayer,
                          "maxdamage": MaxDamageDoublesPlayer,
                          "selfplay": PPODoublesPlayer}[opponent_type]
        extra = dict(ppo_agent=agent, training=True) if opponent_type == "selfplay" else {}
        for worker_id in range(min(parallel, battles, cfg.update_every_episodes)):
            learner = PPODoublesPlayer(
                account_configuration=training_account(f"ppo{worker_id}"),
                ppo_agent=agent, training=True, team=teams[names[0]],
                block_fake_out=getattr(cfg, "block_fake_out", False), **common)
            players.append(learner)
            opponent = opponent_class(
                account_configuration=training_account(f"opp{worker_id}"),
                team=teams[names[0]], **extra, **common)
            players.append(opponent)
            worker_learners = [learner, opponent] if opponent_type == "selfplay" else [learner]
            learners.extend(worker_learners)
            workers.append((learner, opponent, worker_learners))

        print(f"[{label}] parallel battles={len(workers)}; update every {cfg.update_every_episodes} total battles", flush=True)
        finished = 0

        async def collect(worker_id, worker, jobs):
            nonlocal finished
            learner, opponent, worker_learners = worker
            while jobs:
                # No await between checking and claiming a job. Sampling was
                # done in schedule order, independent of completion timing.
                number, learner_name, opponent_name = jobs.popleft()
                learner.update_team(teams[learner_name])
                opponent.update_team(teams[opponent_name])
                print(f"[{label}] worker={worker_id + 1} battle={number}/{battles} | "
                      f"learner={learner_name} vs opponent={opponent_name}", flush=True)
                await run_training_battle(learner, opponent, cfg)
                for player in worker_learners:
                    finalize_finished_battles(player)
                    # Each worker owns its buffer, with complete episodes in
                    # sequence. Terminal flags isolate GAE across concatenation.
                    if isinstance(player.buffer, RolloutBuffer) and len(player.buffer) and not player.buffer.dones[-1]:
                        raise RuntimeError("Unfinished rollout at parallel collection barrier")
                finished += 1

        while finished < battles:
            count = min(cfg.update_every_episodes, battles - finished)
            checkpoint_every = getattr(cfg, "checkpoint_every", 0)
            if checkpoint_every > 0:
                count = min(count, checkpoint_every - finished % checkpoint_every)
            jobs = deque((number, rng.choice(names), rng.choice(names))
                         for number in range(finished + 1, finished + count + 1))
            tasks = [asyncio.create_task(collect(i, worker, jobs))
                     for i, worker in enumerate(workers)]
            try:
                with TIMINGS.measure("collection_wall"):
                    await asyncio.gather(*tasks)
            finally:
                # gather does not cancel siblings when one battle fails.
                # Await cancellation before closing clients or touching weights.
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                tasks = []
            size = update_and_save(agent, cfg, learners)
            if checkpoint_every > 0 and finished % checkpoint_every == 0:
                checkpoint = Path(cfg.model_path)
                checkpoint = checkpoint.with_name(f"{checkpoint.stem}_{finished:05d}{checkpoint.suffix}")
                agent.save(str(checkpoint))
                print(f"checkpoint: {checkpoint}", flush=True)
            print(f"[{label}] battles={finished}/{battles} | buffer={size}", flush=True)
        print(f"[{label}] completed {battles} battles")
    finally:
        await stop_players(*players)


async def train_random_teams(agent, cfg, teams, battles=BATTLES_PER_OPPONENT):
    rng = random.Random(cfg.seed)
    total = len(OPPONENT_TYPES) * battles
    print(f"starting random fixed-team training: {total} battles")
    for opponent_type in OPPONENT_TYPES:
        await train_stage(agent, cfg, teams, opponent_type, rng, battles)


async def main(parallel_battles=5, battles=BATTLES_PER_OPPONENT):
    teams = load_teams()
    cfg = PPOConfig(update_every_episodes=50, parallel_battles=parallel_battles,
                    model_path="ppo_doubles_full_actions.pt")
    agent = PPOAgent(cfg)
    print(f"training error log: {cfg.error_log_path}")
    load_existing_model(agent, cfg)
    await train_random_teams(agent, cfg, teams, battles=battles)
    agent.save(cfg.model_path)
    print(f"saved: {cfg.model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one PPO model with concurrent battles")
    parser.add_argument("--parallel-battles", type=int, default=5,
                        help="Maximum simultaneous battles (default: 5)")
    parser.add_argument("--battles-per-opponent", type=int, default=BATTLES_PER_OPPONENT,
                        help=f"Total battles for each opponent type (default: {BATTLES_PER_OPPONENT})")
    args = parser.parse_args()
    if args.parallel_battles < 1 or args.battles_per_opponent < 1:
        parser.error("parallel-battles and battles-per-opponent must be positive")
    asyncio.run(main(args.parallel_battles, args.battles_per_opponent))
