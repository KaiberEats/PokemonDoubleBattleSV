"""Continue PPO training against MaxDamage only, using shared parallel collection."""

import argparse
import asyncio
import random
import hashlib
import json
from pathlib import Path
from dataclasses import replace

from config import PPOConfig
from ppo import PPOAgent
from train import load_existing_model
from train_team import load_teams, train_stage


TOTAL_BATTLES = 10000


async def main(parallel_battles=5, battles=TOTAL_BATTLES, *, load_model=None,
               save_model=None, block_fake_out=False, checkpoint_every=5000, seed=42):
    if parallel_battles < 1 or battles < 1:
        raise ValueError("parallel_battles and battles must be positive")
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be nonnegative")
    if load_model is not None and not Path(load_model).is_file():
        raise FileNotFoundError(f"Starting model not found: {load_model}")
    output = Path(save_model or "ppo_doubles_full_actions.pt")
    source = Path(load_model or "ppo_doubles_full_actions.pt")
    if load_model is not None and source.resolve() == output.resolve():
        raise ValueError("Use a separate --save-model to preserve the starting model")
    random.seed(seed)
    teams = load_teams()
    cfg = PPOConfig(update_every_episodes=50, parallel_battles=parallel_battles,
                    model_path=str(output), seed=seed, block_fake_out=block_fake_out,
                    checkpoint_every=checkpoint_every)
    agent = PPOAgent(cfg)
    print(f"training error log: {cfg.error_log_path}", flush=True)
    load_existing_model(agent, replace(cfg, model_path=str(source)))
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(load_model=str(source.resolve()), save_model=str(output.resolve()),
                    initial_sha256=hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else None,
                    seed=seed, block_fake_out=block_fake_out, opponent_block_fake_out=False,
                    battles=battles, parallel_battles=parallel_battles,
                    checkpoint_every=checkpoint_every, completed=False)
    manifest = output.with_suffix('.run.json')
    manifest.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f"experiment: {manifest}; block_fake_out={block_fake_out}", flush=True)
    print(f"starting MaxDamage-only training: {battles} total battles; "
          f"parallel={parallel_battles}", flush=True)
    await train_stage(agent, cfg, teams, "maxdamage", random.Random(cfg.seed), battles)
    agent.save(cfg.model_path)
    metadata['completed'] = True
    metadata['completed_battles'] = battles
    manifest.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f"saved: {cfg.model_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO against MaxDamage only")
    parser.add_argument("--parallel-battles", type=int, default=5,
                        help="Maximum simultaneous battles (default: 5)")
    parser.add_argument("--battles", "--battles-per-opponent", type=int, default=TOTAL_BATTLES,
                        help=f"Total battles, across all workers (default: {TOTAL_BATTLES})")
    parser.add_argument("--load-model", help="Starting model (must exist)")
    parser.add_argument("--save-model", help="Output model, separate from the starting model")
    parser.add_argument("--block-fake-out", action="store_true", help="Block Fake Out after the first actionable turn (learner only)")
    parser.add_argument("--checkpoint-every", type=int, default=5000, help="Save numbered checkpoints every N battles; 0 disables")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.parallel_battles < 1 or args.battles < 1:
        parser.error("parallel-battles and battles must be positive")
    if args.checkpoint_every < 0:
        parser.error("checkpoint-every must be nonnegative")
    asyncio.run(main(args.parallel_battles, args.battles, load_model=args.load_model,
                     save_model=args.save_model, block_fake_out=args.block_fake_out,
                     checkpoint_every=args.checkpoint_every, seed=args.seed))
