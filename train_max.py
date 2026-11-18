"""Continue PPO training against MaxDamage only, using shared parallel collection."""

import argparse
import asyncio
import random

from config import PPOConfig
from ppo import PPOAgent
from train import load_existing_model
from train_team import load_teams, train_stage


TOTAL_BATTLES = 10000


async def main(parallel_battles=5, battles=TOTAL_BATTLES):
    if parallel_battles < 1 or battles < 1:
        raise ValueError("parallel_battles and battles must be positive")
    teams = load_teams()
    cfg = PPOConfig(update_every_episodes=50, parallel_battles=parallel_battles,
                    model_path="ppo_doubles_full_actions.pt")
    agent = PPOAgent(cfg)
    print(f"training error log: {cfg.error_log_path}", flush=True)
    load_existing_model(agent, cfg)
    print(f"starting MaxDamage-only training: {battles} total battles; "
          f"parallel={parallel_battles}", flush=True)
    await train_stage(agent, cfg, teams, "maxdamage", random.Random(cfg.seed), battles)
    agent.save(cfg.model_path)
    print(f"saved: {cfg.model_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO against MaxDamage only")
    parser.add_argument("--parallel-battles", type=int, default=5,
                        help="Maximum simultaneous battles (default: 5)")
    parser.add_argument("--battles", "--battles-per-opponent", type=int, default=TOTAL_BATTLES,
                        help=f"Total battles, across all workers (default: {TOTAL_BATTLES})")
    args = parser.parse_args()
    if args.parallel_battles < 1 or args.battles < 1:
        parser.error("parallel-battles and battles must be positive")
    asyncio.run(main(args.parallel_battles, args.battles))
