from poke_env import Player
from poke_env.data import to_id_str
from collections import Counter
from switch_history import SwitchHistory
from state_encoder import embed_battle
from action_mapper import (
    action_to_order,
    legal_action_mask,
    random_team_preview_order,
    team_preview_action_mask,
    team_preview_order_from_action,
)
from ppo import RolloutBuffer
from training_errors import TrainingDiagnosticsMixin

class PPODoublesPlayer(TrainingDiagnosticsMixin, Player):
    def __init__(self, ppo_agent, training=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ppo_agent = ppo_agent
        self.training = training
        self.buffer = RolloutBuffer()
        self._last_obs = {}
        self._last_action = {}
        self._last_logprob = {}
        self._last_value = {}
        self._last_action_mask = {}
        self._selection_decisions = {}
        self._move_histories = {}
        self._switch_histories = {}

    async def _handle_battle_message(self, split_messages):
        tag = split_messages[0][0].lstrip(">")
        history = self._move_histories.setdefault(tag, Counter())
        if not hasattr(self, "_switch_histories"):
            self._switch_histories = {}
        self._switch_histories.setdefault(tag, SwitchHistory()).record(split_messages[1:])
        # Record server move events, not selected orders or unrevealed move sets.
        for msg in split_messages[1:]:
            if len(msg) > 3 and msg[1] == "move" and msg[2][:2] in ("p1", "p2"):
                history[(msg[2][:2], to_id_str(msg[3]))] += 1
        try:
            return await super()._handle_battle_message(split_messages)
        finally:
            battle = self.battles.get(tag)
            if battle is not None and battle.finished:
                self._move_histories.pop(tag, None)
                self._switch_histories.pop(tag, None)

    def teampreview(self, battle):
        obs = self._observation(battle)
        action_mask = team_preview_action_mask(battle)
        action, logprob, value = self.ppo_agent.act(obs, action_mask)

        if self.training:
            self._selection_decisions[battle.battle_tag] = (obs, action, logprob, value, action_mask)

        return team_preview_order_from_action(battle, action)

    def _choose_move(self, battle):
        #print("choose_move start", flush=True)

        obs = self._observation(battle)
        action_mask = legal_action_mask(battle)
        #print("after embed_battle", flush=True)

        action, logprob, value = self.ppo_agent.act(obs, action_mask)
        #print("after ppo act", flush=True)

        if self.training and battle.battle_tag in self._last_obs:
            self.buffer.add(
                self._last_obs[battle.battle_tag],
                self._last_action[battle.battle_tag],
                self._last_logprob[battle.battle_tag],
                0.0,
                False,
                self._last_value[battle.battle_tag],
                self._last_action_mask[battle.battle_tag],
            )

        self._last_obs[battle.battle_tag] = obs
        self._last_action[battle.battle_tag] = action
        self._last_logprob[battle.battle_tag] = logprob
        self._last_value[battle.battle_tag] = value
        self._last_action_mask[battle.battle_tag] = action_mask

        #print("before action_to_order", flush=)
        order = action_to_order(self, battle, action)
        #print("after action_to_order", flush=True)

        return order

    def _observation(self, battle):
        tag = battle.battle_tag
        return embed_battle(battle, getattr(self, "_move_histories", {}).get(tag),
                            getattr(self, "_switch_histories", {}).get(tag))

    def finalize_battle(self, battle):
        """
        battle終了後に呼ぶ。
        勝敗のみ報酬：
        win = +1
        lose = -1
        tie/不明 = 0
        """
        if not self.training:
            return

        tag = battle.battle_tag
        if tag not in self._last_obs and tag not in self._selection_decisions:
            return

        if getattr(battle, "won", False):
            reward = 1.0
        elif getattr(battle, "lost", False):
            reward = -1.0
        else:
            reward = 0.0

        if tag in self._last_obs:
            self.buffer.add(
                self._last_obs[tag], self._last_action[tag], self._last_logprob[tag],
                reward, True, self._last_value[tag], self._last_action_mask[tag],
            )
        # Selection is a separate one-step episode: its tuple of species receives
        # the match result directly, without discounting over battle move turns.
        selection = self._selection_decisions.pop(tag, None)
        if selection is not None:
            obs, action, logprob, value, mask = selection
            self.buffer.add(obs, action, logprob, reward, True, value, mask)

        self._last_obs.pop(tag, None)
        self._last_action.pop(tag, None)
        self._last_logprob.pop(tag, None)
        self._last_value.pop(tag, None)
        self._last_action_mask.pop(tag, None)


class FixedTeamRandomDoublesPlayer(TrainingDiagnosticsMixin, Player):
    def teampreview(self, battle):
        return random_team_preview_order(battle)

    def _choose_move(self, battle):
        return self.choose_random_doubles_move(battle)
