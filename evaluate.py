import asyncio
import argparse
from functools import partial
from itertools import combinations
from pathlib import Path
import re
import torch

from config import LOCAL_SERVER, TEAM_PATH, EVAL_FORMAT, PPOConfig, new_error_log_path
from max_damage_player import MaxDamageDoublesPlayer
from ppo import PPOAgent
from players import FixedTeamRandomDoublesPlayer, PPODoublesPlayer as BasePPODoublesPlayer
from team_loader import load_team
from train import training_account, run_training_battle, stop_players
from training_errors import TrainingErrorRecorder
from random_team import TrainingBans

def format_summary(results, team_names, opponent_names):
    """Every result is counted from the player1 PPO agent's perspective."""
    def rate(rows):
        return f"{sum(r['result'] == 'win' for r in rows) / len(rows):.1%}" if rows else "--"

    def cell(rows):
        return f"{rate(rows)} ({sum(r['result'] == 'win' for r in rows)}/{len(rows)})" if rows else "--"

    def table(headers, rows):
        data = [headers, *rows]
        widths = [max(len(str(row[i])) for row in data) for i in range(len(headers))]
        lines = [" | ".join(str(value).ljust(width) for value, width in zip(row, widths)) for row in data]
        lines.insert(1, "-+-".join("-" * width for width in widths))
        return "\n".join(lines)

    lines = ["\n===== FINAL WIN RATE SUMMARY (player1: trained PPO) =====",
             "Win rate = wins / all finished battles (draws included). Cells: win rate (wins/battles).",
             "selfplay = trained PPO opponent; -- = no battles."]
    rows = []
    for team in team_names:
        own = [r for r in results if r['player1_team'] == team]
        rows.append([team, *[cell([r for r in own if r['opponent'] == mode]) for mode in opponent_names], cell(own)])
    rows.append(["TOTAL", *[cell([r for r in results if r['opponent'] == mode]) for mode in opponent_names], cell(results)])
    lines.append(table(["PPO team", *opponent_names, "Overall"], rows))
    lines.append(table(["Opponent", "Battles", "Wins", "Losses", "Draws", "Win rate"], [
        [mode, len(subset), sum(r['result'] == 'win' for r in subset),
         sum(r['result'] == 'loss' for r in subset), sum(r['result'] == 'draw' for r in subset), rate(subset)]
        for mode in [*opponent_names, "TOTAL"]
        for subset in [[r for r in results if mode == "TOTAL" or r['opponent'] == mode]]
    ]))
    for mode in opponent_names:
        lines.append(f"\n{mode}: rows = player1 PPO team, columns = player2 team")
        lines.append(table(["PPO \\ P2", *team_names], [
            [team, *[cell([r for r in results if r['opponent'] == mode and
                          r['player1_team'] == team and r['player2_team'] == other])
                     for other in team_names]] for team in team_names
        ]))
    return "\n\n".join(lines)


class PPODoublesPlayer(BasePPODoublesPlayer):
    """Keep the server's battle events, including both players' actions."""

    def __init__(self, *args, action_log_dir=None, **kwargs):
        self.action_log_dir = Path(action_log_dir) if action_log_dir else None
        super().__init__(*args, **kwargs)

    async def _handle_battle_message(self, split_messages):
        if self.action_log_dir is not None and split_messages:
            tag = re.sub(r"[^a-zA-Z0-9_-]", "_", split_messages[0][0].lstrip(">"))
            # Requests contain private state rather than executed actions.
            events = ["|".join(parts) for parts in split_messages[1:]
                      if len(parts) > 1 and parts[1] != "request"]
            if events:
                self.action_log_dir.mkdir(parents=True, exist_ok=True)
                with (self.action_log_dir / f"{tag}.txt").open("a", encoding="utf-8") as stream:
                    stream.write("\n".join(events) + "\n")
        return await super()._handle_battle_message(split_messages)


class DeterministicPPOAgent(PPOAgent):
    @torch.no_grad()
    def act(self, obs_np, action_mask_np):
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_mask = torch.tensor(
            action_mask_np, dtype=torch.bool, device=self.device
        ).unsqueeze(0)
        logits, value = self.net(obs)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        if not action_mask.any():
            action_mask[:] = True
        logits = logits.masked_fill(~action_mask, -1e9)
        action = torch.argmax(logits, dim=-1)
        return (
            int(action.item()),
            0.0,
            float(value.item()),
        )


async def evaluate_against(agent, opponent_type, opponent_name, n_eval, teams, cfg):
    """Play an unordered team pair; PPO always controls player1.

    Alternate the teams assigned to each agent, not the player1 agent itself.
    """
    recorder = TrainingErrorRecorder(cfg.error_log_path)
    (first_name, first_team), (second_name, second_team) = teams
    action_log_dir = (Path(cfg.error_log_path).with_suffix("") / opponent_name /
                      f"{first_name}_vs_{second_name}")
    print(f"evaluation action logs: {action_log_dir}", flush=True)
    ppo_player = PPODoublesPlayer(
        action_log_dir=action_log_dir,
        account_configuration=training_account("evalppo"),
        error_recorder=recorder,
        ppo_agent=agent,
        training=False,
        battle_format=EVAL_FORMAT,
        server_configuration=LOCAL_SERVER,
        max_concurrent_battles=1,
        team=first_team,
    )

    opponent = None
    results = []
    try:
        opponent = opponent_type(
            account_configuration=training_account(f"eval{opponent_name}"),
            error_recorder=recorder,
            battle_format=EVAL_FORMAT,
            server_configuration=LOCAL_SERVER,
            max_concurrent_battles=1,
            team=second_team,
        )
        print(f"before {opponent_name} evaluation")
        for index in range(n_eval):
            (p1_name, p1_team), (p2_name, p2_team) = teams if index % 2 == 0 else teams[::-1]
            ppo_player.update_team(p1_team)
            opponent.update_team(p2_team)
            ppo_player.action_log_dir = action_log_dir / f"p1_{p1_name}_vs_p2_{p2_name}"
            print(f"[{opponent_name}] {index + 1}/{n_eval}: PPO {p1_name} vs {p2_name}", flush=True)
            before = (ppo_player.n_finished_battles, ppo_player.n_won_battles,
                      ppo_player.n_lost_battles)
            await run_training_battle(ppo_player, opponent, cfg,
                                      battle_format=EVAL_FORMAT)
            delta = (ppo_player.n_finished_battles - before[0],
                     ppo_player.n_won_battles - before[1],
                     ppo_player.n_lost_battles - before[2])
            if delta not in ((1, 1, 0), (1, 0, 1), (1, 0, 0)):
                raise RuntimeError(f"Unexpected battle counters: {delta}")
            results.append(dict(opponent=opponent_name, player1_team=p1_name,
                                player2_team=p2_name,
                                result="win" if delta[1] else "loss" if delta[2] else "draw"))
        print(f"after {opponent_name} evaluation")
    finally:
        await stop_players(*[p for p in (ppo_player, opponent) if p is not None])

    print(f"===== {opponent_name}: {first_name} vs {second_name} (player1 PPO) =====")
    print(f"battles: {ppo_player.n_finished_battles}")
    print(f"wins:    {ppo_player.n_won_battles}")
    print(f"losses:  {ppo_player.n_lost_battles}")
    print(
        f"winrate: "
        f"{ppo_player.n_won_battles / max(1, ppo_player.n_finished_battles):.3f}"
    )
    return results


async def main(n_eval=10):
    if n_eval < 1:
        raise ValueError("n_eval must be positive")
    cfg = PPOConfig(model_path=str(TEAM_PATH.with_name("ppo_doubles_full_actions.pt")),
                    error_log_path=new_error_log_path().replace("training_errors_", "evaluation_errors_"))
    agent = DeterministicPPOAgent(cfg)
    agent.load(cfg.model_path)
    agent.net.eval()
    teams = [(f"team{i}", load_team(TEAM_PATH.with_name(f"team{i}.txt")))
             for i in range(1, 6)]
    bans = TrainingBans(cfg.training_bans_path)
    for _, team in teams:
        bans.validate_team(team)

    print(f"evaluation error log: {cfg.error_log_path}")
    opponents = [(FixedTeamRandomDoublesPlayer, "random"),
                 (MaxDamageDoublesPlayer, "maxdamage"),
                 (partial(PPODoublesPlayer, ppo_agent=agent, training=False), "selfplay")]
    pairs = list(combinations(teams, 2))
    print(f"round robin: {len(pairs)} pairs x {n_eval} battles x {len(opponents)} agents "
          f"= {len(pairs) * n_eval * len(opponents)} battles", flush=True)
    results = []
    for opponent_type, opponent_name in opponents:
        print(f"\n===== player2: {opponent_name} =====", flush=True)
        for pair in pairs:
            results.extend(await evaluate_against(agent, opponent_type, opponent_name, n_eval, pair, cfg))
    print(format_summary(results, [name for name, _ in teams],
                         [name for _, name in opponents]), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Round robin of team1-team5: PPO vs random, maxdamage, and PPO")
    parser.add_argument("--battles", type=int, default=10, help="Battles per team pair and player2 agent; teams alternate (default: 10)")
    args = parser.parse_args()
    if args.battles < 1:
        parser.error("--battles must be positive")
    asyncio.run(main(args.battles))
