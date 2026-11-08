"""Locally generated Gen 9 teams; no unrestricted server random-team fallback."""

import json
import random
import asyncio
from pathlib import Path

from poke_env.data import GenData, to_id_str
from poke_env.teambuilder import Teambuilder
from poke_env.teambuilder.teambuilder_pokemon import TeambuilderPokemon
from websockets.asyncio.client import connect

DEFAULT_BANS_PATH = Path(__file__).with_name("training_bans.json")
ITEMS = ("Sitrus Berry", "Leftovers", "Focus Sash", "Life Orb", "Lum Berry", "Safety Goggles")


class TrainingBans:
    def __init__(self, path=DEFAULT_BANS_PATH):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.species = {to_id_str(x) for x in data["species"]}
        self.abilities = {to_id_str(x) for x in data["abilities"]}
        self.moves = {to_id_str(x) for x in data["moves"]}
        banned_targets = set(data.get("move_targets", {}))
        self.moves.update(
            move for move, entry in GenData.from_gen(9).moves.items()
            if entry.get("target") in banned_targets
        )

    def bans_species(self, species, dex):
        entry = dex.get(to_id_str(species), {})
        return bool({to_id_str(species), to_id_str(entry.get("baseSpecies", species))} & self.species)

    def validate_team(self, team):
        """Reject unsafe fixed teams as well as unsafe generated teams."""
        dex = GenData.from_gen(9).pokedex
        for mon in Teambuilder.parse_showdown_team(team):
            species = mon.species or mon.nickname or ""
            if self.bans_species(species, dex):
                raise ValueError(f"Banned training species: {species}")
            if to_id_str(mon.ability or "") in self.abilities:
                raise ValueError(f"Banned training ability: {mon.ability}")
            banned_moves = {to_id_str(x) for x in mon.moves} & self.moves
            if banned_moves:
                raise ValueError(f"Banned training moves on {species}: {sorted(banned_moves)}")


class SafeRandomTeamBuilder(Teambuilder):
    """Sample six species with four directly learnable SV moves each.

    Only Gen 9 level/TM sources are used: no event, egg, transfer, Sketch,
    inherited-form learnsets or required-item forms. This intentionally trades
    complete dex coverage for simpler, reproducible legal set construction.
    """

    def __init__(self, seed=42, bans=None):
        self.rng = random.Random(seed)
        self.bans = bans or TrainingBans()
        self.data = GenData.from_gen(9)
        self.pool = {}
        for species, entry in sorted(self.data.pokedex.items()):
            if self.bans.bans_species(species, self.data.pokedex):
                continue
            if entry.get("num", 0) <= 0 or entry.get("isNonstandard"):
                continue
            if entry.get("battleOnly") or entry.get("requiredItem") or entry.get("requiredItems") or entry.get("requiredMove"):
                continue
            base = self.data.pokedex.get(to_id_str(entry.get("baseSpecies", species)), entry)
            if "Mythical" in entry.get("tags", []) + base.get("tags", []):
                continue
            # Exclude cosmetic aliases; they need not have independent learnsets.
            if to_id_str(entry["name"]) != species:
                continue
            abilities = [a for a in entry["abilities"].values() if to_id_str(a) not in self.bans.abilities]
            required = entry.get("requiredAbility")
            if required:
                abilities = [a for a in abilities if a == required]
            learnset = self.data.learnset.get(species, {}).get("learnset", {})
            moves = sorted(
                move for move, sources in learnset.items()
                if move not in self.bans.moves
                and move in self.data.moves
                and not self.data.moves[move].get("isNonstandard")
                and any(source.startswith(("9L", "9M")) for source in sources)
            )
            attacks = [m for m in moves if self.data.moves[m].get("basePower", 0) > 0]
            if abilities and len(moves) >= 4 and attacks:
                self.pool[species] = (abilities, moves, attacks)
        ordinary = {self.data.pokedex[s]["num"] for s in self.pool if not self._restricted(s)}
        if len(ordinary) < 6:
            raise ValueError("BAN configuration leaves fewer than six ordinary species")

    def _restricted(self, species):
        entry = self.data.pokedex[species]
        base = self.data.pokedex.get(to_id_str(entry.get("baseSpecies", species)), entry)
        return "Restricted Legendary" in entry.get("tags", []) + base.get("tags", [])

    def yield_team(self):
        candidates = list(self.pool)
        self.rng.shuffle(candidates)
        selected, numbers = [], set()
        has_restricted = False
        for species in candidates:
            entry = self.data.pokedex[species]
            restricted = self._restricted(species)
            if entry["num"] in numbers or (restricted and has_restricted):
                continue
            selected.append(species)
            numbers.add(entry["num"])
            has_restricted |= restricted
            if len(selected) == 6:
                break
        items = self.rng.sample(ITEMS, 6)
        team = []
        for species, item in zip(selected, items):
            entry = self.data.pokedex[species]
            abilities, moves, attacks = self.pool[species]
            first = self.rng.choice(attacks)
            chosen_moves = [first] + self.rng.sample([m for m in moves if m != first], 3)
            physical = entry["baseStats"]["atk"] >= entry["baseStats"]["spa"]
            team.append(TeambuilderPokemon(
                species=entry["name"], item=item, ability=self.rng.choice(abilities),
                moves=chosen_moves, nature="Adamant" if physical else "Modest",
                evs=[4, 252 if physical else 0, 0, 0 if physical else 252, 0, 252],
                gender=entry.get("gender"), level=50,
                tera_type=entry.get("forceTeraType", entry["types"][0]),
            ))
        return self.join_team(team)


class PackedTeam(Teambuilder):
    """Use exactly the team that the server has validated."""

    def __init__(self, packed):
        self.packed = packed

    def yield_team(self):
        return self.packed


async def _validation_message(websocket, marker, phase, timeout):
    """Wait for a protocol line, retaining useful diagnostics on timeout."""
    recent = []
    try:
        async with asyncio.timeout(timeout):
            while True:
                message = str(await websocket.recv())
                for line in message.splitlines():
                    if line.startswith(marker):
                        return line
                    if line.startswith(("|error|", "|nametaken|")):
                        raise RuntimeError(f"Showdown validation {phase}: {line}")
                    # Do not retain login challenge tokens or the large formats list.
                    if not line.startswith(("|challstr|", "|formats|")):
                        recent.append(line[:500])
                        recent = recent[-5:]
    except TimeoutError as exc:
        raise TimeoutError(
            f"Showdown validation timed out during {phase} after {timeout}s; "
            f"recent messages: {recent!r}"
        ) from exc


async def validated_random_team(builder, server_url, battle_format, player, attempts=20,
                                response_timeout=15, connection_attempts=3):
    """Check the actual server's rules, since poke-env data may differ.

    Showdown /utm and /vtm are documented in server/chat-commands/core.ts.
    Rejected candidate sets never enter a training battle.
    """
    if attempts < 1 or connection_attempts < 1 or response_timeout <= 0:
        raise ValueError("Validation attempts and timeout must be positive")
    recorder = getattr(player, "error_recorder", None)
    for _ in range(attempts):
        packed = builder.yield_team()
        moves = [m for mon in builder.parse_packed_team(packed) for m in mon.moves]
        for retry in range(connection_attempts):
            try:
                async with connect(server_url, open_timeout=10) as websocket:
                    # A WebSocket upgrade alone does not mean Showdown has created
                    # its user/connection yet. Early commands can be dropped.
                    await _validation_message(websocket, "|challstr|", "initialization", response_timeout)
                    await websocket.send("|/utm " + packed)
                    await websocket.send("|/vtm " + battle_format)
                    message = await _validation_message(websocket, "|popup|", "team response", response_timeout)
                break
            except TimeoutError as exc:
                if recorder is not None:
                    recorder.record(player, None, "team_validation_timeout", exc,
                                    order=packed, selected_moves=moves)
                if retry + 1 == connection_attempts:
                    raise
                # Reconnect before retrying the SAME team: a timeout says nothing
                # about its legality. Never admit an unvalidated candidate.
                await asyncio.sleep(retry + 1)
        if "was not found" in message:
            raise ValueError(f"Showdown format unavailable: {message}")
        if "Your team is valid for" in message:
            return PackedTeam(packed)
        if recorder is not None:
            recorder.record(player, None, "team_validation_rejected", message,
                            order=packed, selected_moves=moves)
        if "Your team was rejected" not in message:
            raise RuntimeError(f"Unexpected Showdown validation reply: {message}")
    raise ValueError(f"No valid random team after {attempts} attempts; inspect the error log")
