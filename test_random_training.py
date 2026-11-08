import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from config import PPOConfig
from ppo import RolloutBuffer
from random_team import PackedTeam
from train import (random_training_setup, run_training_battle, train_against_random,
                   train_against_max_damage, train_self_play)


class RandomTrainingTests(unittest.IsolatedAsyncioTestCase):
    async def test_both_teams_are_validated_and_replaced_before_every_battle(self):
        cfg = PPOConfig()
        left, right, _ = random_training_setup(cfg, 0)
        self.assertIs(left.bans, right.bans)
        events = []

        def player(label):
            return SimpleNamespace(training_error=threading.Event(), battles={},
                                   update_team=lambda team: events.append((label, team.packed)),
                                   record_training_error=Mock())

        p1, p2 = player("left"), player("right")

        async def battle(*args, **kwargs):
            self.assertEqual([label for label, _ in events[-2:]], ["left", "right"])
            events.append(("battle", None))

        p1.battle_against = battle

        async def validate(builder, url, battle_format, owner):
            self.assertIs(builder, left if owner is p1 else right)
            packed = builder.yield_team()
            for mon in builder.parse_packed_team(packed):
                self.assertFalse(builder.bans.bans_species(mon.species, builder.data.pokedex))
                self.assertFalse(set(mon.moves) & builder.bans.moves)
                self.assertNotIn(mon.ability, builder.bans.abilities)
            return PackedTeam(packed)

        with patch("train.validated_random_team", side_effect=validate) as validator:
            for _ in range(2):
                await run_training_battle(p1, p2, cfg, right, player_builder=left)
        self.assertEqual(validator.await_count, 4)
        self.assertEqual(len({team for label, team in events if label != "battle"}), 4)

    async def test_all_training_stages_supply_both_random_builders(self):
        cfg = PPOConfig(total_battles=2, max_damage_battles=2, self_play_battles=2,
                        update_every_episodes=10)
        for stage in (train_against_random, train_against_max_damage, train_self_play):
            created = []

            def create(**kwargs):
                obj = SimpleNamespace(buffer=RolloutBuffer(), battles={}, **kwargs)
                created.append(obj)
                return obj

            with patch("train.PPODoublesPlayer", side_effect=create), \
                 patch("train.FixedTeamRandomDoublesPlayer", side_effect=create), \
                 patch("train.MaxDamageDoublesPlayer", side_effect=create), \
                 patch("train.run_training_battle", new_callable=AsyncMock) as run, \
                 patch("train.stop_players", new_callable=AsyncMock):
                await stage(Mock(), cfg)
            self.assertEqual(run.await_count, 2)
            for call in run.await_args_list:
                self.assertIs(call.kwargs["player_builder"], created[0].team)
                self.assertIs(call.args[3], created[1].team)
                self.assertIsNot(created[0].team, created[1].team)


if __name__ == "__main__":
    unittest.main()
