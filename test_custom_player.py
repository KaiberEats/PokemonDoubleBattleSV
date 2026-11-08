import asyncio

from poke_env import Player, RandomPlayer
from poke_env.ps_client import AccountConfiguration

from config import LOCAL_SERVER, TRAIN_FORMAT


class CustomRandomDoublesPlayer(Player):
    def choose_move(self, battle):
        return self.choose_random_doubles_move(battle)


async def main():
    player_1 = CustomRandomDoublesPlayer(
        account_configuration=AccountConfiguration("customtest001", None),
        battle_format=TRAIN_FORMAT,
        server_configuration=LOCAL_SERVER,
        max_concurrent_battles=1,
    )

    player_2 = RandomPlayer(
        account_configuration=AccountConfiguration("randomtest001", None),
        battle_format=TRAIN_FORMAT,
        server_configuration=LOCAL_SERVER,
        max_concurrent_battles=1,
    )

    print("before battle")
    await player_1.battle_against(player_2, n_battles=1)
    print("after battle")

    print(f"player_1 battles: {player_1.n_finished_battles}")
    print(f"player_1 wins: {player_1.n_won_battles}")


if __name__ == "__main__":
    asyncio.run(main())