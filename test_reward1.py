import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch

from ppo import RolloutBuffer
from reward1 import RewardEvents, Reward1Player, board_potential
import train_reward1


def battle():
    mon = lambda: NS(fainted=False, current_hp_fraction=1., moves={}, base_stats={"spe": 80}, ability="", item="")
    return NS(player_role="p1", turn=2, team={str(i): mon() for i in range(4)},
              opponent_team={str(i): mon() for i in range(4)},
              active_pokemon=[mon(), mon()], battle_tag="test", finished=False,
              won=False, lost=False)


def record(tracker, text):
    tracker.record([line.split('|') for line in text.splitlines()])


class RewardTests(unittest.TestCase):
    def test_single_immune_but_not_switch_protect_or_spread(self):
        for extra, move, expected in [('', 'Thunderbolt', -3),
                ('|switch|p2a: Target|Gastrodon|100/100\n', 'Thunderbolt', 0),
                ('|move|p2b: Guard|Protect|p2b: Guard\n', 'Thunderbolt', 0),
                ('', 'Earthquake', 0)]:
            tracker = RewardEvents()
            record(tracker, '|turn|2\n' + extra + f'|move|p1a: Actor|{move}|p2a: Target\n|-immune|p2a: Target')
            self.assertEqual(tracker.penalties(battle())['single_immune'], expected)
            self.assertEqual(tracker.penalties(battle())['single_immune'], 0)

    def test_direct_ally_damage_exemptions_and_opponent_attacks(self):
        for ability, item, actor, expected in [('', '', 'p1a', -5),
                ('stamina', '', 'p1a', 0), ('', 'weaknesspolicy', 'p1a', 0),
                ('', '', 'p2a', 0)]:
            b = battle()
            b.active_pokemon[1].ability = ability
            b.active_pokemon[1].item = item
            tracker = RewardEvents()
            record(tracker, f'|turn|2\n|move|{actor}: Actor|Tera Blast|p1b: Ally\n|-damage|p1b: Ally|90/100')
            self.assertEqual(tracker.penalties(b)['ally_attack'], expected)

    def test_field_value_expires_and_is_relative(self):
        b, tracker = battle(), RewardEvents()
        record(tracker, '|turn|1\n|-sidestart|p1: User|move: Tailwind')
        early = board_potential(b, tracker.flow)
        b.turn = 4
        late = board_potential(b, tracker.flow)
        b.turn = 5
        self.assertGreater(early, late)
        self.assertGreater(late, board_potential(b, tracker.flow))
        b.turn = 2
        b.player_role = 'p2'
        self.assertAlmostEqual(board_potential(b, tracker.flow), -early)

    def test_consumed_policy_and_later_protect_are_not_penalized(self):
        tracker = RewardEvents()
        record(tracker, '|turn|2\n|move|p1a: Actor|Tera Blast|p1b: Ally\n|-damage|p1b: Ally|90/100\n|-enditem|p1b: Ally|Weakness Policy')
        self.assertEqual(tracker.penalties(battle())['ally_attack'], 0)
        tracker = RewardEvents()
        record(tracker, '|turn|2\n|move|p1a: Actor|Thunderbolt|p2a: Target\n|-immune|p2a: Target')
        record(tracker, '|move|p2b: Guard|Protect|p2b: Guard\n|turn|3')
        self.assertEqual(tracker.penalties(battle())['single_immune'], 0)

    def test_terminal_rewards_selection_and_idempotent_cleanup(self):
        for won in (True, False):
            with TemporaryDirectory() as directory:
                player = object.__new__(Reward1Player)
                player.training = True
                player.buffer = RolloutBuffer()
                for name in ('_last_obs', '_last_action', '_last_logprob', '_last_value', '_last_action_mask'):
                    setattr(player, name, {'test': 0})
                player._selection_decisions = {'test': (0, 0, 0, 0, 0)}
                player.reward_events = {'test': RewardEvents()}
                player.reward_previous = {'test': 2.}
                player.reward_totals = {}
                player.reward_gamma = .99
                player.reward_log_path = Path(directory) / 'rewards.jsonl'
                b = battle()
                b.finished, b.won, b.lost = True, won, not won
                player.finalize_battle(b)
                terminal = 100 if won else -100
                self.assertEqual(player.buffer.rewards, [terminal - 2, terminal])
                player.finalize_battle(b)
                self.assertEqual(len(player.buffer.rewards), 2)
                self.assertFalse(player.reward_events)
                self.assertEqual(json.loads(player.reward_log_path.read_text())['terminal'], terminal)


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_a_uniform_sampling_and_reward_player(self):
        with TemporaryDirectory() as directory:
            source, output = Path(directory)/'a.pt', Path(directory)/'reward1.pt'
            source.write_bytes(b'base')
            agent = Mock()
            agent.save.side_effect = lambda path: Path(path).write_bytes(b'trained')
            with patch('train_reward1.PPOAgent', return_value=agent), \
                 patch('train_reward1.load_existing_model'), \
                 patch('train_reward1.load_teams', return_value={'team1': 'team'}), \
                 patch('train_reward1.train_stage', new_callable=AsyncMock) as stage:
                await train_reward1.main(battles=50, load_model=str(source), save_model=str(output))
            self.assertIs(stage.await_args.kwargs['learner_class'], Reward1Player)
            self.assertNotIn('team_schedule', stage.await_args.kwargs)
            self.assertFalse(stage.await_args.args[1].block_fake_out)
            manifest = json.loads(output.with_suffix('.run.json').read_text())
            self.assertTrue(manifest['completed'])
            self.assertEqual(source.read_bytes(), b'base')


if __name__ == '__main__':
    unittest.main()
