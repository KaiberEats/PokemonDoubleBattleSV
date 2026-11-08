import tempfile
import unittest
from functools import partial
from itertools import combinations
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from config import EVAL_FORMAT, PPOConfig, TEAM_PATH
from evaluate import PPODoublesPlayer, evaluate_against, main, format_summary
from players import FixedTeamRandomDoublesPlayer
from max_damage_player import MaxDamageDoublesPlayer
from team_loader import load_team


class EvaluationTests(unittest.IsolatedAsyncioTestCase):
    def pair(self):
        return tuple((f'team{i}', load_team(TEAM_PATH.with_name(f'team{i}.txt'))) for i in (1, 2))

    async def test_default_schedule_has_300_battles_and_three_modes(self):
        with patch('evaluate.DeterministicPPOAgent') as agent, patch('evaluate.evaluate_against', new_callable=AsyncMock, return_value=[]) as play:
            await main()
        calls = play.await_args_list
        self.assertEqual(len(calls), 30)
        expected = list(combinations([f'team{i}' for i in range(1, 6)], 2))
        for index, mode in enumerate(('random', 'maxdamage', 'selfplay')):
            group = calls[index*10:(index+1)*10]
            self.assertEqual([c.args[2] for c in group], [mode]*10)
            self.assertEqual([tuple(name for name, team in c.args[4]) for c in group], expected)
            for call in group:
                self.assertEqual(call.args[3], 10)
                self.assertIs(call.args[0], agent.return_value)
                for name, team in call.args[4]:
                    self.assertEqual(team, load_team(TEAM_PATH.with_name(f'{name}.txt')))
        self.assertIs(calls[0].args[1], FixedTeamRandomDoublesPlayer)
        self.assertIs(calls[10].args[1], MaxDamageDoublesPlayer)
        self.assertIs(calls[20].args[1].keywords['ppo_agent'], agent.return_value)
        self.assertFalse(calls[20].args[1].keywords['training'])
        agent.return_value.update.assert_not_called()
        agent.return_value.save.assert_not_called()

    async def test_alternate_teams_and_logs_without_changing_player1_agent(self):
        agent, ppo, opponent = Mock(), Mock(), Mock()
        ppo.n_finished_battles = ppo.n_won_battles = ppo.n_lost_battles = 0
        pair = self.pair()
        assignments = []
        async def play(player, other, cfg, *, battle_format):
            self.assertIs(player, ppo)
            self.assertIs(other, opponent)
            self.assertEqual(battle_format, EVAL_FORMAT)
            assignments.append((player.update_team.call_args.args[0], other.update_team.call_args.args[0], player.action_log_dir.name))
            ppo.n_finished_battles += 1
        with patch('evaluate.PPODoublesPlayer', return_value=ppo) as factory, patch('evaluate.run_training_battle', side_effect=play) as run, patch('evaluate.stop_players', new_callable=AsyncMock) as stop:
            other_factory = Mock(return_value=opponent)
            results = await evaluate_against(agent, other_factory, 'random', 10, pair, PPOConfig())
        self.assertEqual(run.await_count, 10)
        self.assertEqual([r['player1_team'] for r in results], ['team1', 'team2'] * 5)
        self.assertTrue(all(r['result'] == 'draw' for r in results))
        self.assertEqual(assignments, [(pair[0][1], pair[1][1], 'p1_team1_vs_p2_team2'), (pair[1][1], pair[0][1], 'p1_team2_vs_p2_team1')]*5)
        self.assertIs(factory.call_args.kwargs['ppo_agent'], agent)
        self.assertFalse(factory.call_args.kwargs['training'])
        self.assertEqual(other_factory.call_args.kwargs['team'], pair[1][1])
        stop.assert_awaited_once_with(ppo, opponent)
        agent.update.assert_not_called()
        agent.save.assert_not_called()

    async def test_selfplay_uses_same_frozen_agent_for_both_players(self):
        agent, ppo, opponent = Mock(), Mock(), Mock()
        ppo.n_finished_battles = ppo.n_won_battles = ppo.n_lost_battles = 0
        async def play(*args, **kwargs):
            ppo.n_finished_battles += 1
        with patch('evaluate.PPODoublesPlayer', side_effect=[ppo, opponent]) as factory, patch('evaluate.run_training_battle', side_effect=play), patch('evaluate.stop_players', new_callable=AsyncMock):
            await evaluate_against(agent, partial(factory, ppo_agent=agent, training=False), 'selfplay', 2, self.pair(), PPOConfig())
        for call in factory.call_args_list:
            self.assertIs(call.kwargs['ppo_agent'], agent)
            self.assertFalse(call.kwargs['training'])

    async def test_battle_protocol_is_saved_as_text(self):
        player = object.__new__(PPODoublesPlayer)
        messages = [['>battle-test-1'], ['', 'turn', '1'], ['', 'move', 'p1a: Pikachu', 'Thunderbolt', 'p2a: Eevee'], ['', 'request', 'private state'], ['', 'win', 'PPO']]
        with tempfile.TemporaryDirectory() as directory:
            player.action_log_dir = Path(directory) / 'random' / 'team1_vs_team2'
            with patch('evaluate.BasePPODoublesPlayer._handle_battle_message', new_callable=AsyncMock) as handle:
                await player._handle_battle_message(messages)
            handle.assert_awaited_once_with(messages)
            content = (player.action_log_dir / 'battle-test-1.txt').read_text(encoding='utf-8')
            self.assertIn('|move|p1a: Pikachu|Thunderbolt|p2a: Eevee', content)
            self.assertIn('|win|PPO', content)
            self.assertNotIn('private state', content)

    async def test_failure_closes_both_players(self):
        ppo, opponent = Mock(), Mock()
        with patch('evaluate.PPODoublesPlayer', return_value=ppo), patch('evaluate.run_training_battle', side_effect=RuntimeError('test failure')), patch('evaluate.stop_players', new_callable=AsyncMock) as stop:
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                await evaluate_against(Mock(), Mock(return_value=opponent), 'random', 1, self.pair(), PPOConfig())
        stop.assert_awaited_once_with(ppo, opponent)

    async def test_battle_count_override_and_validation(self):
        with patch('evaluate.DeterministicPPOAgent'), patch('evaluate.evaluate_against', new_callable=AsyncMock, return_value=[]) as play:
            await main(n_eval=2)
        self.assertTrue(all(c.args[3] == 2 for c in play.await_args_list))
        with self.assertRaises(ValueError):
            await main(n_eval=0)

    def test_summary_counts_draws_and_uses_weighted_totals(self):
        results = [dict(opponent='random', player1_team='team1', player2_team='team2', result=r)
                   for r in ('win', 'loss', 'draw')]
        results.append(dict(opponent='random', player1_team='team2', player2_team='team1', result='win'))
        summary = format_summary(results, ['team1', 'team2'], ['random', 'maxdamage', 'selfplay'])
        self.assertIn('33.3% (1/3)', summary)
        self.assertIn('100.0% (1/1)', summary)
        self.assertIn('50.0% (2/4)', summary)
        self.assertIn('Draws', summary)
        self.assertIn('--', summary)

    async def test_completed_battle_results_track_wins_losses_and_team_assignment(self):
        ppo, opponent = Mock(), Mock()
        ppo.n_finished_battles = ppo.n_won_battles = ppo.n_lost_battles = 0
        async def play(*args, **kwargs):
            ppo.n_finished_battles += 1
            if ppo.n_finished_battles == 1:
                ppo.n_won_battles += 1
            elif ppo.n_finished_battles == 2:
                ppo.n_lost_battles += 1
        with patch('evaluate.PPODoublesPlayer', return_value=ppo), patch('evaluate.run_training_battle', side_effect=play), patch('evaluate.stop_players', new_callable=AsyncMock):
            results = await evaluate_against(Mock(), Mock(return_value=opponent), 'random', 3, self.pair(), PPOConfig())
        self.assertEqual([r['result'] for r in results], ['win', 'loss', 'draw'])
        self.assertEqual([r['player1_team'] for r in results], ['team1', 'team2', 'team1'])


if __name__ == '__main__':
    unittest.main()
