import json
import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, Mock, AsyncMock
from types import SimpleNamespace

from poke_env.data import GenData, to_id_str

from random_team import SafeRandomTeamBuilder, TrainingBans, validated_random_team


class RandomTeamTests(unittest.TestCase):
    def test_many_teams_respect_bans_learnsets_and_species_item_clauses(self):
        builder = SafeRandomTeamBuilder(seed=123)
        teams = set()
        for _ in range(200):
            packed = builder.yield_team()
            teams.add(packed)
            team = builder.parse_packed_team(packed)
            self.assertEqual(len(team), 6)
            self.assertEqual(len({builder.data.pokedex[p.species]["num"] for p in team}), 6)
            self.assertEqual(len({p.item for p in team}), 6)
            self.assertLessEqual(sum(builder._restricted(p.species) for p in team), 1)
            for mon in team:
                self.assertFalse(builder.bans.bans_species(mon.species, builder.data.pokedex))
                self.assertNotIn(mon.ability, builder.bans.abilities)
                self.assertEqual(len(set(mon.moves)), 4)
                self.assertFalse(set(mon.moves) & builder.bans.moves)
                for move in mon.moves:
                    sources = builder.data.learnset[mon.species]["learnset"][move]
                    self.assertTrue(any(s.startswith(("9L", "9M")) for s in sources))
        self.assertEqual(len(teams), 200)

    def test_seed_is_reproducible_and_changes_each_battle(self):
        first, second = SafeRandomTeamBuilder(9), SafeRandomTeamBuilder(9)
        samples = [first.yield_team() for _ in range(3)]
        self.assertEqual(samples, [second.yield_team() for _ in range(3)])
        self.assertEqual(len(set(samples)), 3)

    def test_banned_families_and_abilities_never_enter_pool(self):
        builder = SafeRandomTeamBuilder()
        self.assertIn("lingeringaroma", builder.bans.abilities)
        self.assertIn("oinkologne", builder.pool)
        self.assertNotIn("Lingering Aroma", builder.pool["oinkologne"][0])
        for species in ("zorua", "zoruahisui", "zoroark", "zoroarkhisui", "ditto", "tatsugiri", "palafin"):
            self.assertNotIn(species, builder.pool)
        for abilities, _, _ in builder.pool.values():
            self.assertFalse({to_id_str(a) for a in abilities} & builder.bans.abilities)
        self.assertTrue(builder._restricted("calyrexice"))
        self.assertNotIn("zarudedada", builder.pool)
        self.assertNotIn("magearnaoriginal", builder.pool)

    def test_confirmed_failing_locked_moves_are_banned_from_every_candidate(self):
        builder = SafeRandomTeamBuilder()
        failing = {"outrage", "uproar", "petaldance", "thrash", "ragingfury"}
        self.assertTrue(failing <= builder.bans.moves)
        for _, moves, _ in builder.pool.values():
            self.assertFalse(failing & set(moves))
        for move in failing:
            with self.assertRaisesRegex(ValueError, "Banned training moves"):
                builder.bans.validate_team(f"Tauros\nAbility: Intimidate\n- {move}")

    def test_revival_blessing_is_excluded_from_random_and_fixed_teams(self):
        builder = SafeRandomTeamBuilder()
        self.assertIn("revivalblessing", builder.bans.moves)
        for species in ("rabsca", "pawmot"):
            self.assertIn(species, builder.pool)
            self.assertNotIn("revivalblessing", builder.pool[species][1])
        for _, moves, _ in builder.pool.values():
            self.assertNotIn("revivalblessing", moves)
        with self.assertRaisesRegex(ValueError, "Banned training moves"):
            builder.bans.validate_team("Rabsca\nAbility: Telepathy\n- Revival Blessing")

    def test_target_family_ban_covers_new_moves_without_manual_name_updates(self):
        data = GenData.from_gen(9)
        with patch.dict(data.moves, {"futurelockedmove": {"target": "randomNormal"}}):
            bans = TrainingBans()
            self.assertIn("futurelockedmove", bans.moves)
            for move, entry in data.moves.items():
                if entry.get("target") == "randomNormal":
                    self.assertIn(move, bans.moves)

    def test_server_rejected_formes_are_excluded_but_normal_formes_remain(self):
        builder = SafeRandomTeamBuilder()
        for species in ("pikachustarter", "eeveestarter", "floetteeternal"):
            self.assertNotIn(species, builder.pool)
        for species in ("pikachu", "eevee", "floette"):
            self.assertIn(species, builder.pool)

    def test_fixed_team_rejects_banned_species_moves_and_abilities(self):
        bans = TrainingBans()
        for team in (
            "Zoroark-Hisui\nAbility: Illusion\n- Shadow Ball",
            "Pikachu\nAbility: Static\n- Metronome",
            "Pikachu\nAbility: Illusion\n- Thunderbolt",
        ):
            with self.assertRaisesRegex(ValueError, "Banned training"):
                bans.validate_team(team)

    def test_user_move_ban_removes_move_from_all_candidates(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bans.json"
            path.write_text(json.dumps({"species": {}, "abilities": {}, "moves": {"Protect": "test"}}))
            builder = SafeRandomTeamBuilder(bans=TrainingBans(path))
            self.assertTrue(all("protect" not in moves for _, moves, _ in builder.pool.values()))


class ValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejected_team_is_retried_even_when_log_file_is_unwritable(self):
        from training_errors import TrainingErrorRecorder
        first = self.Socket("|popup|Your team was rejected for the following reasons: bad set")
        second = self.Socket("|popup|Your team is valid for VGC.")
        with TemporaryDirectory() as directory:
            primary = Path(directory) / "unwritable.jsonl"
            fallback = Path(directory) / "fallback.jsonl"
            recorder = TrainingErrorRecorder(primary, fallback_path=fallback)
            real_open = Path.open

            def faulty_open(path, *args, **kwargs):
                if path == primary:
                    raise OSError(22, "Invalid argument")
                return real_open(path, *args, **kwargs)

            with patch("random_team.connect", side_effect=[first, second]), patch.object(Path, "open", faulty_open), patch("training_errors.sys.stderr"):
                team = await validated_random_team(SafeRandomTeamBuilder(), "ws://local", "gen9vgc2024regg",
                                                   SimpleNamespace(error_recorder=recorder))
            self.assertEqual(team.yield_team(), second.sent[0].removeprefix("|/utm "))
            self.assertEqual(json.loads(fallback.read_text(encoding="utf-8"))["kind"], "team_validation_rejected")

    class Socket:
        def __init__(self, reply):
            self.sent = []
            self.ready = False
            self.reply = reply

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def send(self, message):
            if not self.ready:
                raise AssertionError("Command sent before Showdown initialization")
            self.sent.append(message)

        async def recv(self):
            if not self.ready:
                self.ready = True
                return "|updateuser| Guest 1|0|1\n|challstr|4|test-token"
            if self.reply is None:
                await asyncio.Event().wait()
            return self.reply

    async def test_rejected_team_is_logged_and_replaced_with_validated_team(self):
        first = self.Socket("|popup|Your team was rejected for the following reasons: bad set")
        second = self.Socket("|popup|Your team is valid for [Gen 9] VGC 2024 Reg G.")
        player = SimpleNamespace(error_recorder=Mock())
        with patch("random_team.connect", side_effect=[first, second]):
            team = await validated_random_team(SafeRandomTeamBuilder(), "ws://local", "gen9vgc2024regg", player)
        self.assertEqual(team.yield_team(), second.sent[0].removeprefix("|/utm "))
        self.assertNotEqual(first.sent[0], second.sent[0])
        player.error_recorder.record.assert_called_once()

    async def test_timeout_reconnects_and_revalidates_same_team(self):
        first = self.Socket(None)
        second = self.Socket("|popup|Your team is valid for VGC.")
        player = SimpleNamespace(error_recorder=Mock())
        with patch("random_team.connect", side_effect=[first, second]), patch("random_team.asyncio.sleep", new_callable=AsyncMock):
            team = await validated_random_team(SafeRandomTeamBuilder(), "ws://local", "gen9vgc2024regg",
                                               player, response_timeout=0.01)
        self.assertEqual(first.sent, second.sent)
        self.assertEqual(team.yield_team(), first.sent[0].removeprefix("|/utm "))
        self.assertEqual(player.error_recorder.record.call_args.args[2], "team_validation_timeout")

    async def test_persistent_timeout_is_bounded_and_explains_phase(self):
        sockets = [self.Socket(None) for _ in range(3)]
        with patch("random_team.connect", side_effect=sockets) as connect_mock, patch("random_team.asyncio.sleep", new_callable=AsyncMock):
            with self.assertRaisesRegex(TimeoutError, "team response"):
                await validated_random_team(SafeRandomTeamBuilder(), "ws://local", "gen9vgc2024regg",
                                             SimpleNamespace(error_recorder=None), response_timeout=0.01)
        self.assertEqual(connect_mock.call_count, 3)

    async def test_server_error_is_not_ignored_until_timeout(self):
        socket = self.Socket("|error|Command rejected")
        with patch("random_team.connect", return_value=socket):
            with self.assertRaisesRegex(RuntimeError, "Command rejected"):
                await validated_random_team(SafeRandomTeamBuilder(), "ws://local", "gen9vgc2024regg",
                                             SimpleNamespace(error_recorder=None))


if __name__ == "__main__":
    unittest.main()
