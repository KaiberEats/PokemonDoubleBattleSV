import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from test_battle_team import play_pair, summarize, save_results


class TournamentTests(unittest.IsolatedAsyncioTestCase):
    def test_transient_replace_failure_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results.json'
            original = Path.replace
            calls = []

            def replace(source, target):
                calls.append(target)
                if len(calls) == 1:
                    raise PermissionError(13, 'locked')
                return original(source, target)

            report = dict(teams=['a', 'b'], battles=[dict(teams=['a', 'b'], winner='a')])
            with patch.object(Path, 'replace', replace), patch('test_battle_team.time.sleep'):
                self.assertEqual(save_results(path, report), path)
            self.assertEqual(len(calls), 2)
            self.assertEqual(json.loads(path.read_text())['summary']['a']['wins'], 1)

    def test_persistent_failure_switches_path_and_preserves_results(self):
        for operation in ('replace', 'write_text'):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'results.json'
                path.write_text('old snapshot')
                original = getattr(Path, operation)

                def fail(source, *args, **kwargs):
                    if source == path.with_suffix('.json.tmp'):
                        raise PermissionError(13, 'locked')
                    return original(source, *args, **kwargs)

                report = dict(teams=['a', 'b'], battles=[dict(teams=['a', 'b'], winner='a')])
                with patch.object(Path, operation, fail), \
                     patch('test_battle_team.tempfile.gettempdir', return_value=directory), \
                     patch('test_battle_team.time.sleep'):
                    active = save_results(path, report)
                    self.assertNotEqual(active, path)
                    report['battles'].append(dict(teams=['a', 'b'], winner=None))
                    self.assertEqual(save_results(active, report), active)
                self.assertEqual(path.read_text(), 'old snapshot')
                self.assertEqual(len(json.loads(active.read_text())['battles']), 2)

    def test_summary_counts_draws_in_denominator(self):
        games = [dict(teams=['a', 'b'], winner=w) for w in ('a', 'b', None)]
        rows = summarize(['a', 'b', 'c'], games)
        self.assertEqual(rows['a'], dict(battles=3, wins=1, losses=1, draws=1, winrate=1/3))
        self.assertEqual(rows['b'], rows['a'])
        self.assertIsNone(rows['c']['winrate'])

    async def test_fixed_teams_frozen_agent_alternation_and_results(self):
        agent = Mock()
        p, q = [SimpleNamespace(n_finished_battles=0, n_won_battles=0,
                                 n_lost_battles=0) for _ in range(2)]
        records, sides = [], []

        async def battle(first, second, cfg, **kwargs):
            sides.append(first is p)
            p.n_finished_battles += 1
            if len(sides) == 1:
                p.n_won_battles += 1
            elif len(sides) == 2:
                p.n_lost_battles += 1

        with patch('test_battle_team.PPODoublesPlayer', side_effect=[p, q]) as factory, \
             patch('test_battle_team.run_training_battle', side_effect=battle), \
             patch('test_battle_team.stop_players', new_callable=AsyncMock) as stop:
            await play_pair(agent, Mock(), Mock(), ('a', 'b'), ('fixed-a', 'fixed-b'), 3, records.append)
        self.assertEqual(sides, [True, False, True])
        self.assertEqual([r['winner'] for r in records], ['a', 'b', None])
        for call, team in zip(factory.call_args_list, ('fixed-a', 'fixed-b')):
            self.assertIs(call.kwargs['ppo_agent'], agent)
            self.assertFalse(call.kwargs['training'])
            self.assertEqual(call.kwargs['team'], team)
        stop.assert_awaited_once_with(p, q)
        agent.update.assert_not_called()
        agent.save.assert_not_called()

    async def test_failure_closes_players_without_counting_loss(self):
        p, q = Mock(), Mock()
        record = Mock()
        with patch('test_battle_team.PPODoublesPlayer', side_effect=[p, q]), \
             patch('test_battle_team.run_training_battle', side_effect=RuntimeError('failed')), \
             patch('test_battle_team.stop_players', new_callable=AsyncMock) as stop:
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                await play_pair(Mock(), Mock(), Mock(), ('a', 'b'), ('a', 'b'), 1, record)
        stop.assert_awaited_once_with(p, q)
        record.assert_not_called()
