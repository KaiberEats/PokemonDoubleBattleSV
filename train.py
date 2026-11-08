import asyncio
from pathlib import Path
from uuid import uuid4

from poke_env.ps_client import AccountConfiguration

from config import LOCAL_SERVER, TRAIN_FORMAT, PPOConfig
from max_damage_player import MaxDamageDoublesPlayer
from players import FixedTeamRandomDoublesPlayer, PPODoublesPlayer
from ppo import PPOAgent, RolloutBuffer
from random_team import SafeRandomTeamBuilder, TrainingBans, validated_random_team
from training_errors import TrainingErrorRecorder


def training_account(prefix):
    # Reusing a username reconnects to its unfinished Showdown rooms and replays
    # old requests into the new player's state. Start each player in fresh rooms.
    return AccountConfiguration(f"{prefix[:9]}{uuid4().hex[:8]}", None)


def training_setup(cfg, team, seed_offset):
    bans = TrainingBans(cfg.training_bans_path)
    if team is not None:
        bans.validate_team(team)
    return (
        SafeRandomTeamBuilder(seed=cfg.seed + seed_offset, bans=bans),
        TrainingErrorRecorder(cfg.error_log_path),
    )


def random_training_setup(cfg, seed_offset):
    opponent_builder, recorder = training_setup(cfg, None, seed_offset)
    player_builder = SafeRandomTeamBuilder(
        seed=cfg.seed + 1000 + seed_offset, bans=opponent_builder.bans)
    return player_builder, opponent_builder, recorder


def training_failure_message(cfg, *players):
    paths = []
    for player in players:
        recorder = getattr(player, "error_recorder", None)
        path = str(recorder.active_path) if recorder is not None else cfg.error_log_path
        if path not in paths:
            paths.append(path)
    return "Training battle failed; inspect " + ", ".join(paths)


async def run_training_battle(player, opponent, cfg, opponent_builder=None, *,
                              player_builder=None, battle_format=TRAIN_FORMAT):
    """Background poke-env failures must not leave battle_against waiting forever."""
    for current, builder in ((player, player_builder), (opponent, opponent_builder)):
        if builder is None:
            continue
        try:
            async with asyncio.timeout(cfg.battle_timeout_seconds):
                team = await validated_random_team(
                    builder, LOCAL_SERVER.websocket_url, battle_format, current)
            current.update_team(team)
        except Exception as exc:
            current.record_training_error(None, "team_generation_failed", exc)
            raise
    task = asyncio.create_task(player.battle_against(opponent, n_battles=1))
    deadline = asyncio.get_running_loop().time() + cfg.battle_timeout_seconds
    try:
        while not task.done():
            if any(p.training_error.is_set() for p in (player, opponent)):
                raise RuntimeError(training_failure_message(cfg, player, opponent))
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"Training battle exceeded {cfg.battle_timeout_seconds}s")
            await asyncio.wait({task}, timeout=0.25)
        await task
        if any(p.training_error.is_set() for p in (player, opponent)):
            raise RuntimeError(training_failure_message(cfg, player, opponent))
    except Exception as exc:
        for current in (player, opponent):
            active = [b for b in current.battles.values() if not b.finished]
            for battle in active or [None]:
                current.record_training_error(battle, "training_battle_aborted", exc)
        raise
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def load_existing_model(agent, cfg):
    model_path = Path(cfg.model_path)
    if not model_path.exists():
        print(f"no existing model found: {cfg.model_path}")
        return False

    agent.load(cfg.model_path)
    agent.net.train()
    print(f"loaded existing model: {cfg.model_path}")
    return True


def finalize_finished_battles(player):
    for battle in player.battles.values():
        if battle.finished:
            player.finalize_battle(battle)


def drain_buffers(players):
    combined = RolloutBuffer()
    for player in players:
        combined.extend(player.buffer)
        player.buffer = RolloutBuffer()
    return combined


def update_and_save(agent, cfg, players):
    update_buffer = drain_buffers(players)
    agent.update(update_buffer)
    agent.save(cfg.model_path)
    return len(update_buffer)


async def stop_players(*players):
    for player in players:
        try:
            await player.ps_client.stop_listening()
        except Exception as exc:
            print(f"[WARN] failed to stop {player.username}: {exc}")
        finally:
            player.detach_error_handler()


async def train_against_random(agent, cfg):
    if cfg.total_battles <= 0:
        return

    player_team, opponent_team, error_recorder = random_training_setup(cfg, 0)

    ppo_player = PPODoublesPlayer(
        account_configuration=training_account("ppoagent"),
        error_recorder=error_recorder,
        ppo_agent=agent,
        training=True,
        battle_format=TRAIN_FORMAT,
        server_configuration=LOCAL_SERVER,
        max_concurrent_battles=1,
        team=player_team,
    )
    random_player = FixedTeamRandomDoublesPlayer(
        account_configuration=training_account("random"),
        error_recorder=error_recorder,
        battle_format=TRAIN_FORMAT,
        server_configuration=LOCAL_SERVER,
        max_concurrent_battles=1,
        team=opponent_team,
    )

    try:
        for finished in range(1, cfg.total_battles + 1):
            await run_training_battle(ppo_player, random_player, cfg, opponent_team,
                                      player_builder=player_team)
            finalize_finished_battles(ppo_player)

            if finished % cfg.update_every_episodes == 0:
                buffer_size = update_and_save(agent, cfg, [ppo_player])
                winrate = ppo_player.n_won_battles / max(1, ppo_player.n_finished_battles)
                print(
                    f"[RANDOM UPDATE] battles={finished} | "
                    f"wins={ppo_player.n_won_battles} | "
                    f"losses={ppo_player.n_lost_battles} | "
                    f"winrate={winrate:.3f} | buffer={buffer_size}"
                )

        if len(ppo_player.buffer) > 0:
            update_and_save(agent, cfg, [ppo_player])
    finally:
        await stop_players(ppo_player, random_player)


async def train_against_max_damage(agent, cfg):
    if cfg.max_damage_battles <= 0:
        return

    player_team, opponent_team, error_recorder = random_training_setup(cfg, 1)

    ppo_player = PPODoublesPlayer(
        account_configuration=training_account("ppovsmax"),
        error_recorder=error_recorder,
        ppo_agent=agent,
        training=True,
        battle_format=TRAIN_FORMAT,
        server_configuration=LOCAL_SERVER,
        max_concurrent_battles=1,
        team=player_team,
    )
    max_damage_player = MaxDamageDoublesPlayer(
        account_configuration=training_account("maxdamage"),
        error_recorder=error_recorder,
        battle_format=TRAIN_FORMAT,
        server_configuration=LOCAL_SERVER,
        max_concurrent_battles=1,
        team=opponent_team,
    )

    try:
        for finished in range(1, cfg.max_damage_battles + 1):
            await run_training_battle(ppo_player, max_damage_player, cfg, opponent_team,
                                      player_builder=player_team)
            finalize_finished_battles(ppo_player)

            if finished % cfg.update_every_episodes == 0:
                buffer_size = update_and_save(agent, cfg, [ppo_player])
                winrate = ppo_player.n_won_battles / max(1, ppo_player.n_finished_battles)
                print(
                    f"[MAX-DAMAGE UPDATE] battles={finished} | "
                    f"wins={ppo_player.n_won_battles} | "
                    f"losses={ppo_player.n_lost_battles} | "
                    f"winrate={winrate:.3f} | buffer={buffer_size}"
                )

        if len(ppo_player.buffer) > 0:
            update_and_save(agent, cfg, [ppo_player])
    finally:
        await stop_players(ppo_player, max_damage_player)


async def train_self_play(agent, cfg):
    if cfg.self_play_battles <= 0:
        return

    player_team, opponent_team, error_recorder = random_training_setup(cfg, 2)

    player_1 = PPODoublesPlayer(
        account_configuration=training_account("selfplay1"),
        error_recorder=error_recorder,
        ppo_agent=agent,
        training=True,
        battle_format=TRAIN_FORMAT,
        server_configuration=LOCAL_SERVER,
        max_concurrent_battles=1,
        team=player_team,
    )
    player_2 = PPODoublesPlayer(
        account_configuration=training_account("selfplay2"),
        error_recorder=error_recorder,
        ppo_agent=agent,
        training=True,
        battle_format=TRAIN_FORMAT,
        server_configuration=LOCAL_SERVER,
        max_concurrent_battles=1,
        team=opponent_team,
    )

    try:
        for finished in range(1, cfg.self_play_battles + 1):
            await run_training_battle(player_1, player_2, cfg, opponent_team,
                                      player_builder=player_team)
            finalize_finished_battles(player_1)
            finalize_finished_battles(player_2)

            if finished % cfg.update_every_episodes == 0:
                buffer_size = update_and_save(agent, cfg, [player_1, player_2])
                print(
                    f"[SELF-PLAY UPDATE] battles={finished} | "
                    f"player1_wins={player_1.n_won_battles} | "
                    f"player2_wins={player_2.n_won_battles} | "
                    f"buffer={buffer_size}"
                )

        if len(player_1.buffer) > 0 or len(player_2.buffer) > 0:
            update_and_save(agent, cfg, [player_1, player_2])
    finally:
        await stop_players(player_1, player_2)


async def main():
    cfg = PPOConfig(
        total_battles=1000,
        max_damage_battles=1000,
        self_play_battles=1000,
        update_every_episodes=50,
        model_path="ppo_doubles_full_actions.pt",
    )
    agent = PPOAgent(cfg)
    print(f"training error log: {cfg.error_log_path}")
    load_existing_model(agent, cfg)

    print(f"starting random-opponent training: {cfg.total_battles} battles")
    await train_against_random(agent, cfg)

    print(f"starting max-damage-opponent training: {cfg.max_damage_battles} battles")
    await train_against_max_damage(agent, cfg)

    print(f"starting self-play training: {cfg.self_play_battles} battles")
    await train_self_play(agent, cfg)

    agent.save(cfg.model_path)
    print(f"saved: {cfg.model_path}")


if __name__ == "__main__":
    asyncio.run(main())
