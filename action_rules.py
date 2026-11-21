"""Track the first actionable turn after each public switch-in."""


class FakeOutTracker:
    def __init__(self):
        self.turn = 0
        self.eligible_turn = {}

    def record(self, messages):
        for msg in messages:
            if len(msg) < 3:
                continue
            if msg[1] == 'turn':
                self.turn = int(msg[2])
            elif msg[1] in ('switch', 'drag'):
                # Initial deployment, ordinary switches and replacements all
                # precede the incoming Pokemon's next actionable turn.
                self.eligible_turn[msg[2][:3]] = self.turn + 1
            elif msg[1] == 'faint':
                self.eligible_turn.pop(msg[2][:3], None)

    def allowed(self, battle):
        role = getattr(battle, 'player_role', None)
        return tuple(self.eligible_turn.get(str(role) + slot) == battle.turn
                     for slot in ('a', 'b'))
