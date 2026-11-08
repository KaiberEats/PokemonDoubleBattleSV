"""Round-robin evaluation of fixed teams with the saved PPO policy."""

import argparse
import asyncio
from itertools import combinations
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from uuid import uuid4

from poke_env.teambuilder import Teambuilder

from config import EVAL_FORMAT, LOCAL_SERVER, PPOConfig, new_error_log_path
from evaluate import DeterministicPPOAgent
from players import PPODoublesPlayer
from random_team import PackedTeam, TrainingBans, validated_random_team
from team_loader import load_team
from train import run_training_battle, stop_players, training_account
from training_errors import TrainingErrorRecorder

BASE = Path(__file__).resolve().parent


def summarize(names, battles):
    rows = {name: dict(battles=0, wins=0, losses=0, draws=0) for name in names}
    for battle in battles:
        a, b = battle['teams']
        winner = battle['winner']
        for name in (a, b):
            row = rows[name]
            row['battles'] += 1
            row['draws' if winner is None else 'wins' if winner == name else 'losses'] += 1
    for row in rows.values():
        row['winrate'] = row['wins'] / row['battles'] if row['battles'] else None
    return rows


def save_results(path, report):
    report['summary'] = summarize(report['teams'], report['battles'])
    payload = json.dumps(report, ensure_ascii=False, indent=2)

    def write(destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + '.tmp')
        temporary.write_text(payload, encoding='utf-8')
        temporary.replace(destination)

    for attempt in range(3):
        try:
            write(path)
            return path
        except OSError as exc:
            failure = exc
            if attempt < 2:
                time.sleep(0.1 * (attempt + 1))
    # Cloud sync and Windows file locks can prevent atomic replacement. Keep
    # the old snapshot intact and switch subsequent saves to a unique path.
    fallback = Path(tempfile.gettempdir()) / 'pokemonshowdown-team-results' / (
        f'{path.stem}_{uuid4().hex[:8]}.json')
    write(fallback)
    print(f'[WARN] Results save failed at {path}: {failure}; '
          f'results saved to {fallback}', flush=True)
    return fallback


async def play_pair(agent, cfg, recorder, names, teams, count, on_result):
    players = []
    try:
        for name, team in zip(names, teams):
            players.append(PPODoublesPlayer(
                account_configuration=training_account(name),
                error_recorder=recorder, ppo_agent=agent, training=False,
                battle_format=EVAL_FORMAT, server_configuration=LOCAL_SERVER,
                max_concurrent_battles=1, team=team,
            ))
        for index in range(count):
            # Alternate the challenger to reduce player-side bias.
            first, second = players if index % 2 == 0 else players[::-1]
            before = (players[0].n_finished_battles, players[0].n_won_battles,
                      players[0].n_lost_battles)
            await run_training_battle(first, second, cfg, battle_format=EVAL_FORMAT)
            delta = (players[0].n_finished_battles - before[0],
                     players[0].n_won_battles - before[1],
                     players[0].n_lost_battles - before[2])
            if delta not in ((1, 1, 0), (1, 0, 1), (1, 0, 0)):
                raise RuntimeError(f'Unexpected battle counters: {delta}')
            winner = names[0] if delta[1] else names[1] if delta[2] else None
            on_result(dict(teams=list(names), game=index + 1,
                           challenger=names[index % 2], winner=winner))
    finally:
        await stop_players(*players)


async def main(battles=100, model=BASE / 'ppo_doubles_full_actions.pt', output=None):
    if battles < 1:
        raise ValueError('battles must be positive')
    model = Path(model).resolve()
    if not model.is_file():
        raise FileNotFoundError(f'Saved model not found: {model}')
    cfg = PPOConfig(model_path=str(model), error_log_path=new_error_log_path().replace(
        'training_errors_', 'team_battle_errors_'))
    output = Path(output) if output else Path(cfg.error_log_path).with_name(
        Path(cfg.error_log_path).name.replace('team_battle_errors_', 'team_battle_results_')).with_suffix('.json')
    if output.resolve() == model:
        raise ValueError('The results path must not overwrite the model')
    names = [f'team{i}' for i in range(1, 6)]
    bans = TrainingBans(cfg.training_bans_path)
    texts = {name: load_team(BASE / f'{name}.txt') for name in names}
    for text in texts.values():
        bans.validate_team(text)
    agent = DeterministicPPOAgent(cfg)
    agent.load(str(model))
    agent.net.eval()
    recorder = TrainingErrorRecorder(cfg.error_log_path)
    teams = {}
    # Validate the exact fixed sets once; never substitute random teams.
    for name, text in texts.items():
        builder = PackedTeam(Teambuilder.join_team(Teambuilder.parse_showdown_team(text)))
        teams[name] = await validated_random_team(
            builder, LOCAL_SERVER.websocket_url, EVAL_FORMAT,
            SimpleNamespace(error_recorder=None), attempts=1)
    report = dict(format=EVAL_FORMAT, model=str(model), policy='deterministic argmax',
                  battles_per_pair=battles, teams=texts, battles=[], complete=False)
    output = save_results(output, report)
    print(f'Model: {model}\nResults: {output}\nError log: {cfg.error_log_path}', flush=True)

    def record(result):
        nonlocal output
        report['battles'].append(result)
        output = save_results(output, report)
        print(f"[{len(report['battles'])}/{10 * battles}] "
              f"{' vs '.join(result['teams'])}: {result['winner'] or 'draw'}", flush=True)

    try:
        for a, b in combinations(names, 2):
            await play_pair(agent, cfg, recorder, (a, b), (teams[a], teams[b]), battles, record)
        report['complete'] = True
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        output = save_results(output, report)
        print(f'Results: {output}', flush=True)
        print('\nTeam     Battles  Wins  Losses  Draws  Winrate')
        for name, row in report['summary'].items():
            rate = f"{row['winrate']:.1%}" if row['winrate'] is not None else '-'
            print(f"{name:8} {row['battles']:7} {row['wins']:5} {row['losses']:7} "
                  f"{row['draws']:6} {rate:>8}")
        print('\nPair results (wins / losses / draws from first team):')
        for a, b in combinations(names, 2):
            games = [g for g in report['battles'] if g['teams'] == [a, b]]
            row = summarize((a, b), games)[a]
            print(f"{a} vs {b}: {row['wins']} / {row['losses']} / {row['draws']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--battles', type=int, default=100, help='Battles per pair (default: 100; total: 1000)')
    parser.add_argument('--model', type=Path, default=BASE / 'ppo_doubles_full_actions.pt')
    parser.add_argument('--output', type=Path, help='Results JSON (default: timestamped file in logs/)')
    args = parser.parse_args()
    if args.battles < 1:
        parser.error('--battles must be positive')
    asyncio.run(main(args.battles, args.model, args.output))
