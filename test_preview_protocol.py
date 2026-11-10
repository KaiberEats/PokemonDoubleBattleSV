import json
import logging
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from poke_env.player import Player
from poke_env.environment.double_battle import DoubleBattle
from action_mapper import random_team_preview_order
from training_errors import TrainingDiagnosticsMixin


class ProtocolBase:
    # Exercise poke-env's actual event dispatcher and action routing.
    _handle_battle_message = Player._handle_battle_message
    _handle_battle_request = Player._handle_battle_request
    MESSAGES_TO_IGNORE = Player.MESSAGES_TO_IGNORE

    async def _get_battle(self, tag):
        return self.battles[tag.lstrip(">")]


class PreviewPlayer(TrainingDiagnosticsMixin, ProtocolBase):
    def teampreview(self, battle):
        return "/team 1234"

    def choose_move(self, battle):
        return SimpleNamespace(message="/choose move 1, move 1")


class PreviewProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_double_battle_parses_preview_before_selecting(self):
        player = self.make_player()
        battle = DoubleBattle("battle-test-1", "learner", player.logger, 9)
        player.battles[battle.battle_tag] = battle
        player.teampreview = random_team_preview_order
        pokemon = []
        for species in ("Pikachu", "Raichu", "Eevee", "Umbreon", "Espeon", "Sylveon"):
            pokemon.append({"ident": "p1: " + species, "details": species + ", L50",
                            "condition": "100/100", "active": False,
                            "stats": {stat: 100 for stat in ("atk", "def", "spa", "spd", "spe")},
                            "moves": ["protect"], "baseAbility": "static", "item": ""})
        request = {"teamPreview": True, "maxChosenTeamSize": 4, "rqid": 3,
                   "side": {"name": "learner", "id": "p1", "pokemon": pokemon}}
        await player._handle_battle_message([[">battle-test-1"], ["", "teampreview", "4"]])
        player.ps_client.send_message.assert_not_awaited()
        await player._handle_battle_message([
            [">battle-test-1"], ["", "request", json.dumps(request)]])
        player.ps_client.send_message.assert_awaited_once()
        message, tag = player.ps_client.send_message.await_args.args
        self.assertEqual(tag, battle.battle_tag)
        self.assertTrue(message.startswith("/team "))
        choices = message.removeprefix("/team ")
        self.assertEqual(len(choices), 4)
        self.assertEqual(len(set(choices)), 4)

    def make_player(self):
        player = PreviewPlayer()
        battle = SimpleNamespace(battle_tag="battle-test-1", teampreview=False,
                                 finished=False, _last_request={}, _wait=False,
                                 move_on_next_request=False)

        def parse_request(request):
            battle._last_request = request
            battle.teampreview = request.get("teamPreview", False)

        battle.parse_request = parse_request
        battle.parse_message = lambda message: None
        player.battles = {battle.battle_tag: battle}
        player.ps_client = SimpleNamespace(send_message=AsyncMock())
        player.logger = logging.getLogger("preview-test")
        return player

    async def test_both_event_orders_and_batch_boundaries(self):
        notice = ["", "teampreview", "4"]
        request = ["", "request", json.dumps({"teamPreview": True, "rqid": 3})]
        for events in ([notice, request], [request, notice]):
            for batched in (False, True):
                with self.subTest(events=events, batched=batched):
                    player = self.make_player()
                    batches = [events] if batched else [[event] for event in events]
                    for batch in batches:
                        await player._handle_battle_message([[">battle-test-1"], *batch])
                    # Repeated request/notification must not select again.
                    await player._handle_battle_message([[">battle-test-1"], request, notice])
                    player.ps_client.send_message.assert_awaited_once_with(
                        "/team 1234", "battle-test-1")
                    # Normal turns must still use poke-env's action routing.
                    battle = player.battles["battle-test-1"]
                    battle.parse_request({"active": [], "rqid": 4})
                    await player._handle_battle_request(battle)
                    self.assertEqual(player.ps_client.send_message.await_args.args,
                                     ("/choose move 1, move 1", "battle-test-1"))

    async def test_turn_before_request_dispatches_once_using_new_request(self):
        for batched in (False, True):
            with self.subTest(batched=batched):
                player = self.make_player()
                battle = player.battles["battle-test-1"]
                battle.parse_request({"teamPreview": True, "rqid": 3})
                turn = ["", "turn", "1"]
                request = ["", "request", json.dumps({"active": [], "rqid": 4})]
                if batched:
                    await player._handle_battle_message([[">battle-test-1"], turn, request])
                else:
                    await player._handle_battle_message([[">battle-test-1"], turn])
                    player.ps_client.send_message.assert_not_awaited()
                    await player._handle_battle_message([[">battle-test-1"], request])
                player.ps_client.send_message.assert_awaited_once_with(
                    "/choose move 1, move 1", "battle-test-1")
                await player._handle_battle_message([[">battle-test-1"], request])
                # A later turn notice must not reuse the old active request.
                await player._handle_battle_message([[">battle-test-1"], ["", "turn", "2"]])
                self.assertEqual(player.ps_client.send_message.await_count, 1)
                request[2] = json.dumps({"active": [], "rqid": 5})
                await player._handle_battle_message([[">battle-test-1"], request])
                self.assertEqual(player.ps_client.send_message.await_count, 2)


if __name__ == "__main__":
    unittest.main()
