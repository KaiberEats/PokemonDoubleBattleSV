"""Run all three training stages without changing the research checkpoint."""

import argparse
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from config import PPOConfig
from ppo import PPOAgent
from train import train_against_random, train_against_max_damage, train_self_play


async def main(battles, resume_model=None):
    with TemporaryDirectory() as directory:
        cfg = PPOConfig(total_battles=battles, max_damage_battles=battles,
                        self_play_battles=battles, update_every_episodes=battles,
                        model_path=str(Path(directory) / "smoke.pt"),
                        error_log_path=str(Path(__file__).with_name("logs") / "smoke_errors.jsonl"))
        agent = PPOAgent(cfg)
        if resume_model is not None:
            agent.load(resume_model)
            agent.net.train()
        await train_against_random(agent, cfg)
        await train_against_max_damage(agent, cfg)
        await train_self_play(agent, cfg)
        print(f"Smoke training passed: {battles} battles per stage; original checkpoint unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battles", type=int, default=1)
    parser.add_argument("--resume-model", help="Read an existing checkpoint; smoke updates are saved only to a temporary file")
    args = parser.parse_args()
    if args.battles < 1:
        parser.error("--battles must be positive")
    asyncio.run(main(args.battles, args.resume_model))
