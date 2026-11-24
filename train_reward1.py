"""Train from A against MaxDamage with reward1; no matchup weighting or Fake Out restriction."""
import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random

from config import PPOConfig
from ppo import PPOAgent
from reward1 import Reward1Player
from train import load_existing_model
from train_team import load_teams, train_stage


async def main(parallel_battles=20, battles=10000, *, load_model="models/a.pt",
               save_model="models/reward1.pt", checkpoint_every=1000, seed=42):
    if parallel_battles < 1 or battles < 1 or checkpoint_every < 0:
        raise ValueError("battle counts must be positive; checkpoint_every must be nonnegative")
    source, output = Path(load_model), Path(save_model)
    if not source.is_file():
        raise FileNotFoundError(f"Starting model not found: {source}")
    if source.resolve() == output.resolve():
        raise ValueError("Use a separate output to preserve the starting model")
    random.seed(seed)
    teams = load_teams()
    cfg = PPOConfig(update_every_episodes=50, parallel_battles=parallel_battles,
                    model_path=str(output), seed=seed, block_fake_out=False,
                    checkpoint_every=checkpoint_every)
    agent = PPOAgent(cfg)
    load_existing_model(agent, replace(cfg, model_path=str(source)))
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = output.with_suffix(".run.json")
    reward_log = output.with_suffix(".rewards.jsonl")
    metadata = dict(load_model=str(source.resolve()), save_model=str(output.resolve()),
                    initial_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    battles=battles, parallel_battles=parallel_battles, seed=seed,
                    checkpoint_every=checkpoint_every, opponent="maxdamage",
                    sampling="independent uniform learner/opponent teams, same as train_max",
                    block_fake_out=False, reward=dict(win=100, loss=-100,
                    ally_attack=-5, single_immune=-3, board="gamma*Phi(next)-Phi(previous)",
                    potential_bound=10, gamma=cfg.gamma),
                    reward_log=str(reward_log.resolve()), error_log=str(cfg.error_log_path),
                    completed=False)
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"training error log: {cfg.error_log_path}", flush=True)
    print(f"reward components: {reward_log}", flush=True)
    await train_stage(agent, cfg, teams, "maxdamage", random.Random(seed), battles,
                      learner_class=Reward1Player,
                      learner_kwargs=dict(reward_log_path=str(reward_log), gamma=cfg.gamma))
    agent.save(str(output))
    metadata.update(completed=True, completed_battles=battles,
                    final_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel-battles", type=int, default=20)
    parser.add_argument("--battles", "--battles-per-opponent", type=int, default=10000)
    parser.add_argument("--load-model", default="models/a.pt")
    parser.add_argument("--save-model", default="models/reward1.pt")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    asyncio.run(main(**vars(parser.parse_args())))
