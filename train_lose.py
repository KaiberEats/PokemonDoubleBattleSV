"""Continue model A against MaxDamage, emphasizing its weakest matchups."""

import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from config import PPOConfig
from ppo import PPOAgent
from train import load_existing_model
from train_team import load_teams, train_stage


# Ordered pairs: learner first, MaxDamage opponent second. A's evaluation:
# 18%, 26%, 36% respectively. These are not reversed automatically.
WEAK_PAIRS = (("team4", "team1"), ("team3", "team4"), ("team2", "team4"))


def build_schedule(names, battles, focus_ratio, seed):
    if battles < 1 or not 0 <= focus_ratio <= 1:
        raise ValueError("battles must be positive and focus_ratio must be between 0 and 1")
    pairs = [(left, right) for left in names for right in names if left != right]
    if not pairs or any(left not in names or right not in names for left, right in WEAK_PAIRS):
        raise ValueError("All five training teams are required")
    rng = random.Random(seed)

    def balanced(pool, count):
        result = []
        while len(result) < count:
            cycle = list(pool)
            rng.shuffle(cycle)
            result.extend(cycle[:count - len(result)])
        return result

    focused = int(battles * focus_ratio)
    schedule = balanced(WEAK_PAIRS, focused) + balanced(pairs, battles - focused)
    rng.shuffle(schedule)
    return schedule


async def main(parallel_battles=20, battles=10000, *, load_model="models/a.pt",
               save_model="models/lose.pt", focus_ratio=0.5, checkpoint_every=1000,
               seed=42):
    if parallel_battles < 1 or checkpoint_every < 0:
        raise ValueError("parallel_battles must be positive; checkpoint_every must be nonnegative")
    source, output = Path(load_model), Path(save_model)
    if not source.is_file():
        raise FileNotFoundError(f"Starting model not found: {source}")
    if source.resolve() == output.resolve():
        raise ValueError("Use a separate output to preserve model A")
    teams = load_teams()
    schedule = build_schedule(tuple(teams), battles, focus_ratio, seed)
    counts = Counter(schedule)
    random.seed(seed)
    cfg = PPOConfig(update_every_episodes=50, parallel_battles=parallel_battles,
                    model_path=str(output), seed=seed, block_fake_out=False,
                    checkpoint_every=checkpoint_every)
    agent = PPOAgent(cfg)
    # A missing source is an error, never a silently initialized model.
    from dataclasses import replace
    load_existing_model(agent, replace(cfg, model_path=str(source)))
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(load_model=str(source.resolve()), save_model=str(output.resolve()),
                    initial_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    opponent="maxdamage", seed=seed, battles=battles,
                    parallel_battles=parallel_battles, focus_ratio=focus_ratio,
                    weak_pairs=WEAK_PAIRS, block_fake_out=False,
                    checkpoint_every=checkpoint_every,
                    scheduled_matchups={f"{a}_vs_{b}": n for (a, b), n in sorted(counts.items())},
                    error_log=str(cfg.error_log_path), completed=False)
    manifest = output.with_suffix(".run.json")
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"training error log: {cfg.error_log_path}", flush=True)
    print(f"starting focused MaxDamage training: {battles} battles; parallel={parallel_battles}", flush=True)
    for (left, right), count in sorted(counts.items()):
        print(f"schedule: learner={left} opponent={right}: {count}", flush=True)
    await train_stage(agent, cfg, teams, "maxdamage", random.Random(seed), battles,
                      team_schedule=schedule)
    agent.save(str(output))
    metadata.update(completed=True, completed_battles=battles,
                    final_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved: {output}; manifest: {manifest}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-model", default="models/a.pt")
    parser.add_argument("--save-model", default="models/lose.pt")
    parser.add_argument("--battles", type=int, default=10000)
    parser.add_argument("--parallel-battles", type=int, default=20)
    parser.add_argument("--focus-ratio", type=float, default=0.5,
                        help="Fraction reserved for the three weak pairs; the rest covers all 20 ordered pairs")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(main(**vars(args)))
