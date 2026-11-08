"""Track public entry/exit events independently for each battle and active slot."""
from poke_env.data import to_id_str


class SwitchHistory:
    def __init__(self):
        self.turn = 0
        self.slots = {}

    def record(self, messages):
        for msg in messages:
            if len(msg) < 3:
                continue
            event = msg[1]
            if event == "turn":
                self.turn = int(msg[2])
            elif event == "faint":
                slot = self.slots.get(msg[2][:3])
                if slot:
                    slot["present"] = False
            elif event in ("switch", "drag") and len(msg) > 3:
                key = msg[2][:3]
                if key not in ("p1a", "p1b", "p2a", "p2b"):
                    continue
                previous = self.slots.get(key)
                species = to_id_str(msg[3].split(",")[0])
                entry_turn = max(1, self.turn)
                self.slots[key] = dict(
                    present=True, species=species, entry_turn=entry_turn,
                    changed=previous is not None,
                    switch_turn=self.turn if previous else 0,
                    outgoing=previous["species"] if previous else "",
                    forced=bool(previous and (not previous["present"] or event == "drag")),
                )
            elif event in ("detailschange", "replace") and len(msg) > 3:
                # Form changes / revealed Illusion are not new entries.
                slot = self.slots.get(msg[2][:3])
                if slot:
                    slot["species"] = to_id_str(msg[3].split(",")[0])
