import json
import logging
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from poke_env.player import Player
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


if __name__ == "__main__":
    unittest.main()
