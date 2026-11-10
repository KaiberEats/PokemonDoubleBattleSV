"""Record suspect moves with context, without claiming they caused an error."""

import json
import logging
import sys
import tempfile
import threading
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from poke_env.data import to_id_str


class TrainingErrorRecorder:
    def __init__(self, path, *, fallback_path=None):
        self.path = Path(path)
        self.run_id = uuid4().hex
        self.active_path = self.path
        self.fallback_path = Path(fallback_path) if fallback_path is not None else (
            Path(tempfile.gettempdir()) / "pokemonshowdown-training-errors" / f"{self.run_id}.jsonl"
        )
        self._lock = threading.Lock()
        self._pending = deque(maxlen=100)
        self._warnings = set()
        self.dropped_records = 0

    def _warn_once(self, key, message):
        if key in self._warnings:
            return
        self._warnings.add(key)
        # Do not use logging: our client handler would treat the logging
        # failure as a battle failure, potentially calling this recorder again.
        try:
            print(f"[WARN] {message}", file=sys.stderr, flush=True)
        except (OSError, ValueError):
            pass  # An unavailable console must not mask the original error.

    def _write(self, line):
        """Best-effort persistence; log I/O must never abort team validation."""
        with self._lock:
            if len(self._pending) == self._pending.maxlen:
                self.dropped_records += 1
                self._warn_once("buffer_full", "Training error log buffer full; oldest records will be dropped.")
            self._pending.append(line)
            for destination in dict.fromkeys((self.active_path, self.fallback_path)):
                try:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("a", encoding="utf-8") as stream:
                        stream.write("".join(self._pending))
                except (OSError, ValueError) as exc:
                    self._warn_once(str(destination), f"Cannot write training error log {destination}: {exc}")
                    continue
                self.active_path = destination
                self._pending.clear()
                if destination != self.path:
                    self._warn_once("fallback", f"Training error log switched to: {destination}")
                return True
            self._warn_once("buffered", "Both training log paths are unavailable; retaining up to 100 records in memory and retrying on the next record.")
            return False

    def record(self, player, battle, kind, error, *, messages=(), selected_moves=(), order=None, related_moves=()):
        protocol_moves = [
            {"actor": msg[2], "move": to_id_str(msg[3])}
            for msg in messages if len(msg) > 3 and msg[1] == "move"
        ]
        teams = {}
        for name in ("team", "opponent_team"):
            teams[name] = [
                {"species": getattr(mon, "species", None),
                 "moves": list((getattr(mon, "moves", {}) or {}).keys())}
                for mon in (getattr(battle, name, {}) or {}).values()
            ]
        row = {
            "event_id": uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat(), "run_id": self.run_id,
            "player": getattr(player, "username", None),
            "battle_tag": getattr(battle, "battle_tag", None),
            "turn": getattr(battle, "turn", None), "kind": kind,
            "error": str(error), "error_type": type(error).__name__,
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__))
            if isinstance(error, BaseException) else None,
            "candidate_moves": sorted(set(selected_moves) | set(related_moves) | {m["move"] for m in protocol_moves}),
            "related_moves": list(related_moves),
            "cause_confirmed": False, "selected_moves": list(selected_moves),
            "selected_order": order, "recent_move_events": protocol_moves,
            "protocol_tail": list(messages), **teams,
        }
        return self._write(json.dumps(row, ensure_ascii=False) + "\n")


class _ClientErrorHandler(logging.Handler):
    """Also catch challenge rejection / websocket errors outside battle parsing."""

    def __init__(self, player):
        super().__init__(logging.WARNING)
        self.player = player

    def emit(self, record):
        if record.levelno < logging.ERROR and "Popup message received:" not in record.getMessage():
            return
        if not self.player.training_error.is_set():
            error = record.exc_info[1] if record.exc_info else record.getMessage()
            self.player.record_training_error(None, "client_error", error)
            self.player.training_error.set()


class TrainingDiagnosticsMixin:
    LATE_TEAM_SHEET_ERRORS = {
        "You cannot reject open team sheets after Team Preview.",
        "You cannot accept open team sheets after Team Preview.",
    }

    def __init__(self, *args, error_recorder=None, **kwargs):
        self.error_recorder = error_recorder
        self.training_error = threading.Event()
        self._protocol_tail = {}
        self._selected_orders = {}
        self._server_error_state = {}
        self._preview_orders_sent = set()
        self._action_requests_sent = {}
        self._parsing_battle_messages = set()
        self._error_handler = None
        super().__init__(*args, **kwargs)
        if error_recorder is not None:
            self._error_handler = _ClientErrorHandler(self)
            self.logger.addHandler(self._error_handler)

    def detach_error_handler(self):
        if self._error_handler is not None:
            self.logger.removeHandler(self._error_handler)

    def record_training_error(self, battle, kind, error, *, related_moves=()):
        recorder = getattr(self, "error_recorder", None)
        if recorder is None:
            return
        tag = getattr(battle, "battle_tag", None)
        selected_moves, order = self._selected_orders.get(tag, ([], None))
        recorder.record(self, battle, kind, error,
                        messages=self._protocol_tail.get(tag, ()),
                        selected_moves=selected_moves, order=order, related_moves=related_moves)

    def choose_move(self, battle):
        try:
            order = self._choose_move(battle)
            if getattr(self, "error_recorder", None) is not None:
                parts = [getattr(order, "first_order", None), getattr(order, "second_order", None)]
                moves = [part.order.id for part in parts
                         if part is not None and hasattr(part.order, "id")]
                self._selected_orders[battle.battle_tag] = (moves, order.message)
            return order
        except Exception as exc:
            self.record_training_error(battle, "choose_move_exception", exc)
            if getattr(self, "error_recorder", None) is not None:
                self.training_error.set()
            raise

    async def _handle_ots_request(self, battle_tag):
        battle = self.battles.get(battle_tag)
        if self.error_recorder is not None and battle is not None and (
            battle.finished or getattr(battle, "turn", 0) > 0
        ):
            return
        await super()._handle_ots_request(battle_tag)

    async def _handle_battle_request(self, battle, from_teampreview_request=False,
                                     maybe_default_order=False):
        # Recent Showdown versions announce preview before sending its request.
        # poke-env 0.9 sets battle.teampreview only when parsing that request.
        request = getattr(battle, "last_request", None) or getattr(battle, "_last_request", {}) or {}
        if from_teampreview_request and not request.get("teamPreview"):
            return
        if (battle.battle_tag in getattr(self, "_parsing_battle_messages", ())
                and not from_teampreview_request and not maybe_default_order):
            return
        if request.get("teamPreview"):
            if not hasattr(self, "_preview_orders_sent"):
                self._preview_orders_sent = set()
            key = (battle.battle_tag, request.get("rqid"))
            if key in self._preview_orders_sent:
                return
            await self.ps_client.send_message(self.teampreview(battle), battle.battle_tag)
            self._preview_orders_sent.add(key)
            return
        return await super()._handle_battle_request(
            battle, from_teampreview_request=from_teampreview_request,
            maybe_default_order=maybe_default_order)

    async def _parse_messages_and_preview(self, split_messages):
        tag = split_messages[0][0].lstrip(">")
        if not hasattr(self, "_parsing_battle_messages"):
            self._parsing_battle_messages = set()
        if not hasattr(self, "_action_requests_sent"):
            self._action_requests_sent = {}
        self._parsing_battle_messages.add(tag)
        try:
            result = await super()._handle_battle_message(split_messages)
        finally:
            self._parsing_battle_messages.discard(tag)
        battle = self.battles.get(tag)
        if battle is not None:
            if battle.finished:
                self._action_requests_sent.pop(tag, None)
                if hasattr(self, "_preview_orders_sent"):
                    self._preview_orders_sent = {key for key in self._preview_orders_sent if key[0] != tag}
            elif any(len(msg) > 2 and msg[1] == "request" and msg[2]
                     for msg in split_messages[1:]):
                request = getattr(battle, "last_request", None) or getattr(battle, "_last_request", {}) or {}
                if request.get("teamPreview"):
                    await self._handle_battle_request(battle, from_teampreview_request=True)
                elif (not request.get("wait")
                      and ("active" in request or any(request.get("forceSwitch", [])))):
                    # poke-env 0.9 normally acts on |turn|, not on ordinary
                    # |request| messages. New servers may deliver them later.
                    key = request.get("rqid")
                    if tag not in self._action_requests_sent or self._action_requests_sent[tag] != key:
                        self._action_requests_sent[tag] = key
                        try:
                            await self._handle_battle_request(battle)
                        except Exception:
                            self._action_requests_sent.pop(tag, None)
                            raise
        return result

    async def _handle_battle_message(self, split_messages):
        if self.error_recorder is None:
            return await self._parse_messages_and_preview(split_messages)
        if self.training_error.is_set():
            return  # Stop feeding queued errors into poke-env's retry loop.
        tag = split_messages[0][0].lstrip(">")
        tail = self._protocol_tail.setdefault(tag, deque(maxlen=80))
        # Keep the full received batch, including messages that may not yet have
        # been parsed when an exception occurs. These are candidates, not causes.
        tail.extend(split_messages[1:])
        battle = self.battles.get(tag)
        # The sheet reply may race the last preview choice in flight. This
        # exact notification does not reject a battle action or require retry.
        pending = [split_messages[0]]
        for msg in split_messages[1:]:
            if len(msg) > 2 and msg[1] == "error" and msg[2] in self.LATE_TEAM_SHEET_ERRORS:
                self.record_training_error(battle, "late_team_sheet_reply", msg[2])
            else:
                pending.append(msg)
        if len(pending) == 1:
            return
        split_messages = pending
        if not hasattr(self, "_server_error_state"):
            self._server_error_state = {}
        request = getattr(battle, "last_request", None) or getattr(battle, "_last_request", {}) or {}
        request_key = (getattr(battle, "turn", None), request.get("rqid"))
        state = self._server_error_state.get(tag)
        if state is None or state["request"] != request_key:
            state = {"request": request_key, "count": 0, "seen": set()}
            self._server_error_state[tag] = state
        for msg in split_messages[1:]:
            if len(msg) > 1 and msg[1] in {"error", "bigerror"}:
                error = "|".join(msg)
                state["count"] += 1
                if error not in state["seen"]:
                    self.record_training_error(battle, "server_error", error)
                    state["seen"].add(error)
                if state["count"] >= 3:
                    self.record_training_error(
                        battle, "server_error_retry_limit",
                        f"Aborted after {state['count']} errors for request {request_key}: {error}")
                    self.training_error.set()
                    return
        try:
            return await self._parse_messages_and_preview(split_messages)
        except Exception as exc:
            self.record_training_error(self.battles.get(tag), "battle_message_exception", exc)
            self.training_error.set()
            raise
        finally:
            battle = self.battles.get(tag)
            if battle is not None and battle.finished:
                self._protocol_tail.pop(tag, None)
                self._selected_orders.pop(tag, None)
                self._server_error_state.pop(tag, None)
