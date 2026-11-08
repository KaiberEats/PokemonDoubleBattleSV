import asyncio

from poke_env import RandomPlayer
from poke_env.ps_client import AccountConfiguration

from config import LOCAL_SERVER, TRAIN_FORMAT


async def main():
    print("creating players...")

    player_1 = RandomPlayer(
        account_configuration=AccountConfiguration("playerone", None),
        battle_format=TRAIN_FORMAT,
        server_configuration=LOCAL_SERVER,
        max_concurrent_battles=1,
    )

    player_2 = RandomPlayer(
        account_configuration=AccountConfiguration("playertwo", None),
        battle_format=TRAIN_FORMAT,
        server_configuration=LOCAL_SERVER,
        max_concurrent_battles=1,
    )

    print("starting battle...")
    await player_1.battle_against(player_2, n_battles=1)

    print("battle finished")
    print(f"player_1 battles: {player_1.n_finished_battles}")
    print(f"player_1 wins: {player_1.n_won_battles}")
    print(f"player_2 battles: {player_2.n_finished_battles}")
    print(f"player_2 wins: {player_2.n_won_battles}")


if __name__ == "__main__":
    asyncio.run(main())