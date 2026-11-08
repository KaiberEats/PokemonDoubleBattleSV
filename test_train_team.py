import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from train_team import train_stage, train_random_teams


class TeamTrainingTests(unittest.IsolatedAsyncioTestCase):
    async def test_schedule_has_300_battles_in_requested_opponent_order(self):
        teams = {f"team{i}": f"fixed{i}" for i in range(1, 6)}
        with patch("train_team.train_stage", new_callable=AsyncMock) as run:
            await train_random_teams(Mock(), SimpleNamespace(seed=42), teams)
        self.assertEqual(run.await_count, 3)
        self.assertEqual(sum(c.args[5] for c in run.await_args_list), 300)
        self.assertEqual([c.args[3] for c in run.await_args_list],
                         ["random", "maxdamage", "selfplay"])
        for call in run.await_args_list:
            self.assertIs(call.args[2], teams)
            self.assertIs(call.args[4], run.await_args_list[0].args[4])

    async def test_independent_team_draws_allow_mirrors_and_selfplay_updates_both(self):
        for kind in ("random", "maxdamage", "selfplay"):
            with self.subTest(kind=kind):
                players = [Mock(buffer=[]), Mock(buffer=[])]
                cfg = SimpleNamespace(error_log_path="unused", update_every_episodes=2)
                teams = {f"team{i}": f"fixed{i}" for i in range(1, 6)}
                rng = Mock()
                rng.choice.side_effect = ["team5", "team5", "team1", "team3"]
                with patch("train_team.PPODoublesPlayer", side_effect=players), \
                     patch("train_team.FixedTeamRandomDoublesPlayer", return_value=players[1]), \
                     patch("train_team.MaxDamageDoublesPlayer", return_value=players[1]), \
                     patch("train_team.TrainingErrorRecorder"), \
                     patch("train_team.run_training_battle", new_callable=AsyncMock) as battle, \
                     patch("train_team.finalize_finished_battles") as finalize, \
                     patch("train_team.update_and_save") as update, \
                     patch("train_team.stop_players", new_callable=AsyncMock) as stop:
                    await train_stage(Mock(), cfg, teams, kind, rng, 2)
                self.assertEqual(rng.choice.call_count, 4)
                for call in rng.choice.call_args_list:
                    self.assertEqual(call.args[0], tuple(teams))
                self.assertEqual(battle.await_count, 2)
                for call in battle.await_args_list:
                    self.assertEqual(len(call.args), 3)
                    self.assertEqual(call.kwargs, {})
                self.assertEqual([c.args[0] for c in players[0].update_team.call_args_list],
                                 ["fixed5", "fixed1"])
                self.assertEqual([c.args[0] for c in players[1].update_team.call_args_list],
                                 ["fixed5", "fixed3"])
                learners = players if kind == "selfplay" else players[:1]
                self.assertEqual(update.call_args.args[2], learners)
                self.assertEqual(finalize.call_count, 2 * len(learners))
                stop.assert_awaited_once_with(*players)


if __name__ == "__main__":
    unittest.main()
