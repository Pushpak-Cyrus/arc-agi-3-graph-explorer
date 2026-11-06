"""
CyrusAgent - the real agent implementation.

CYRUS System 1:
    Perception -> Graph Memory -> Progress Signal -> Action Selection

This version uses structured action signatures so that different targets
of ACTION6 are remembered as different experiments.

Simple actions:
    (action_id, None, None)

Coordinate-based actions:
    (6, x, y)

System 2 adds hypothesis-driven action selection after System 1 stalls.
"""

import random
import math
import os
from collections import Counter
from typing import Optional, Tuple

import numpy as np

from agents.agent import Agent, GameAction

from frame_processor import FrameProcessor
from graph_memory import GraphMemory
from budget_tracker import BudgetTracker
from experiment_log import ExperimentLog
from goal_tracker import GoalTracker
from system2 import System2
from system2.interaction_model import InteractionModel
from system2.relational_planner import RelationalPlanner


class CyrusAgent(Agent):
    """
    CYRUS System 1: graph-based exploration agent.

    Decision order:

        1. High-value untested experiment
        2. Any untested experiment
        3. High-value tested action
        4. Navigate toward the nearest graph frontier
        5. Random fallback

    Important:
        ACTION6 is represented by its actual coordinate target.

        Example:

            (6, 12, 25)
            (6, 31, 44)
            (6, 50, 18)

        These are three different experiments and are stored separately
        in GraphMemory.
    """

    # The actual stopping condition is controlled by BudgetTracker.
    MAX_ACTIONS: int = 100_000

    # Shared across agent instances created for different games.
    shared_budget_tracker: Optional[BudgetTracker] = None
    shared_interaction_model: Optional[InteractionModel] = None

    # Set to True to print, every step, how many raw grid cells differ from
    # the previous frame. Useful for diagnosing whether states-per-step
    # ratio is near 1.0 because the game genuinely changes every step, or
    # because some small cosmetic element (cursor mark, blinking indicator,
    # unmasked counter) is churning the hash without real progress.
    DEBUG_FRAME_DIFF: bool = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # ------------------------------------------------------------------
        # Core CYRUS components
        # ------------------------------------------------------------------

        self.frame_processor = FrameProcessor(self.game_id)
        self.graph = GraphMemory()
        self.goal_tracker = GoalTracker()
        self.experiment_log = ExperimentLog(
            game_id=self.game_id,
            output_path=os.getenv("CYRUS_EXPERIMENT_LOG"),
        )
        self.system2_takeover_enabled = os.getenv(
            "CYRUS_SYSTEM2_ENABLED", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}
        self.learned_ranking_enabled = os.getenv(
            "CYRUS_LEARNED_RANKING_ENABLED", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        # Per-episode activity counters used to detect inactive ablation arms.
        self.system2_selected_actions = 0
        self.learned_ranking_decisions = 0
        self.learned_ranking_choice_changes = 0
        self.system2 = System2(
            log_path=os.getenv("CYRUS_SYSTEM2_LOG"),
            game_id=self.game_id,
        )
        self.relational_planner = RelationalPlanner()
        # The visual relation planner is retained as a specialty/baseline
        # controller. General evaluations use learned System 1/System 2 by
        # default so a fixed pattern solver cannot mask generalization gaps.
        self.relational_planner_enabled = os.getenv(
            "CYRUS_RELATIONAL_PLANNER_ENABLED", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if CyrusAgent.shared_interaction_model is None:
            CyrusAgent.shared_interaction_model = self.system2.interactions
        else:
            self.system2.interactions = CyrusAgent.shared_interaction_model
        self.system2.interactions.begin_episode()

        # ------------------------------------------------------------------
        # Transition memory
        # ------------------------------------------------------------------

        # State before the action that produced the next frame.
        self.last_state_hash: Optional[str] = None

        # IMPORTANT:
        # This is the complete experiment signature, not just action_id.
        #
        # Example:
        #   (1, None, None)
        #   (6, 23, 41)
        #
        self.last_action_signature: Optional[Tuple] = None

        # Previous structured state, used by the progress detector.
        self.previous_state = None

        self.last_levels_completed: int = 0
        self.last_score: Optional[float] = None
        self.first_level_advance_action: Optional[int] = None
        self.game_over_resets: int = 0
        self.consecutive_reset_stalls: int = 0
        self._just_reset_after_game_over: bool = False
        self._post_reset_action_pending: bool = False
        self.movement_probe_trials = Counter()
        self.movement_probe_progress = set()
        self.click_class_trials = Counter()
        self.click_class_progress = set()
        self.empty_click_trials = 0
        self.empty_click_progress = False
        # Distinct source states where an action produced an exact no-op.
        # Kept across GAME_OVER resets within this game episode.
        self.action_noop_states = {}
        self.noop_deprioritize_after = max(
            1, int(os.getenv("CYRUS_NOOP_DEPRIORITIZE_AFTER", "3"))
        )
        self.frontier_root_hash: Optional[str] = None
        self.frontier_replay_target: Optional[str] = None
        self.frontier_mode_enabled = os.getenv(
            "CYRUS_FRONTIER_MODE", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.search_order_enabled = os.getenv(
            "CYRUS_SEARCH_ORDER", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.frontier_reset_due = False
        self._last_selection_was_replay = False
        self.frontier_depth_schedule = (
            (6, 12, 24, 50) if self.search_order_enabled else (50,)
        )
        self.frontier_depth_index = 0
        self.frontier_depth_limit = self.frontier_depth_schedule[0]
        self.frontier_attempt_cap = max(1, int(os.getenv(
            "CYRUS_FRONTIER_ATTEMPT_CAP", str(self.frontier_depth_limit)
        )))
        self.frontier_attempt_actions = 0
        self.frontier_attempts = []
        self.zero_length_frontier_attempts = 0
        self.frontier_search_exhausted = False
        self.actions_since_level_start = 0
        self.actions_since_reset = 0
        self._attempt_start_hash = None
        self.last_selection_reason = ""
        self.click_class_noop_states = {}
        self.click_class_changed = set()

        # ------------------------------------------------------------------
        # Budget
        # ------------------------------------------------------------------

        # Fallback for local single-game testing.
        if CyrusAgent.shared_budget_tracker is None:
            CyrusAgent.shared_budget_tracker = BudgetTracker(
                total_time_budget_seconds=8.5 * 3600,
                games=[self.game_id],
            )

        self.budget = CyrusAgent.shared_budget_tracker
        self.budget.start_game(self.game_id)
        self.budget.start_level()

    # ======================================================================
    # Required Agent interface
    # ======================================================================

    def is_done(self, frames, latest_frame) -> bool:
        """Stop on WIN or limits; reset after GAME_OVER when budget permits."""

        state_name = latest_frame.state.name
        game_over = state_name == "GAME_OVER"
        won = state_name == "WIN"
        if getattr(self, "frontier_search_exhausted", False):
            return True
        if self._post_reset_action_pending:
            self._post_reset_action_pending = False
            if game_over:
                self.consecutive_reset_stalls += 1
            else:
                self.consecutive_reset_stalls = 0
        budget_exhausted = self.budget.should_stop_run()
        step_limit_reached = self.action_counter > self.MAX_ACTIONS
        reset_stalled = game_over and self.consecutive_reset_stalls >= 2
        should_stop = (
            won or budget_exhausted or step_limit_reached or reset_stalled
        )

        # The framework checks is_done before choose_action. Complete the
        # final action here so terminal outcomes are not omitted from traces.
        if (game_over or won or should_stop) and self.experiment_log.pending is not None:
            if getattr(latest_frame, "levels_completed", 0) > self.last_levels_completed:
                self.frame_processor.reset_target_history()
            final_state = self.frame_processor.process(
                np.array(latest_frame.frame)
            )
            self._record_observation(
                latest_frame,
                final_state,
                terminal=game_over or won,
            )
            self.experiment_log.close_pending()

        if game_over and not should_stop:
            return False

        return should_stop

    @staticmethod
    def _extract_score(latest_frame) -> Optional[float]:
        """Read an explicit numeric score if this FrameData exposes one."""
        for field_name in ("score", "game_score"):
            value = getattr(latest_frame, field_name, None)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            score = float(value)
            if math.isfinite(score):
                return score
        return None

    def _record_observation(self, latest_frame, state, terminal=False):
        """Close the previous action with typed evidence and graph outcome."""
        levels_after = getattr(latest_frame, "levels_completed", None)
        score_after = self._extract_score(latest_frame)
        caused_reset = bool(getattr(latest_frame, "full_reset", False)) or bool(
            self.last_action_signature
            and int(self.last_action_signature[0]) == GameAction.RESET.value
        )
        level_advanced = (
            levels_after is not None
            and levels_after > self.last_levels_completed
        )

        if level_advanced:
            if self.first_level_advance_action is None:
                # action_counter is the number of actions already executed
                # when this observation confirms the transition.
                self.first_level_advance_action = int(self.action_counter)

        # Feed the actual transition back to the relational planner before it
        # chooses another step. This lets it reject stale placements when the
        # observed pieces did not move as predicted.
        if (self.relational_planner_enabled
                and self.last_action_signature is not None
                and not level_advanced and not caused_reset):
            self.relational_planner.observe(
                self.previous_state,
                state,
                self.last_action_signature,
                level_key=self.last_levels_completed,
            )

        report = self.goal_tracker.compare(
            previous_state=self.previous_state,
            current_state=state,
            levels_before=self.last_levels_completed,
            levels_after=levels_after,
            score_before=self.last_score,
            score_after=score_after,
            caused_reset=caused_reset,
        )
        if report.confirmed_progress and self.last_action_signature is not None:
            previous_action_id = int(self.last_action_signature[0])
            if 1 <= previous_action_id <= 5:
                self.movement_probe_progress.add(previous_action_id)
            elif previous_action_id == GameAction.ACTION6.value:
                click_class = self._click_class_for_signature(
                    self.previous_state, self.last_action_signature
                )
                if click_class is None:
                    self.empty_click_progress = True
                else:
                    self.click_class_progress.add(click_class)

        if (
            self.last_state_hash is not None
            and self.last_action_signature is not None
        ):
            if self.last_state_hash == state["hash"]:
                self.action_noop_states.setdefault(
                    self.last_action_signature, set()
                ).add(self.last_state_hash)
                clicked_class = self._click_candidate_class(
                    self.previous_state, self.last_action_signature
                )
                if clicked_class is not None:
                    self.click_class_noop_states.setdefault(
                        clicked_class, set()
                    ).add(self.last_state_hash)
            else:
                clicked_class = self._click_candidate_class(
                    self.previous_state, self.last_action_signature
                )
                if clicked_class is not None:
                    self.click_class_changed.add(clicked_class)
            if report.confirmed_progress:
                outcome_class = "confirmed_progress"
            elif caused_reset or any(
                signal.strength == "negative_evidence"
                for signal in report.signals
            ):
                outcome_class = "negative_evidence"
            elif report.has_visual_observation:
                outcome_class = "visual_observation"
            else:
                outcome_class = "no_confirmed_progress"

            self.graph.record_transition(
                self.last_state_hash,
                self.last_action_signature,
                state["hash"],
                caused_reset=caused_reset,
                level_id=self.last_levels_completed,
                outcome_class=outcome_class,
                replayed=self._last_selection_was_replay,
            )

            # Only environment-confirmed progress makes an action globally
            # high-value. Visual changes stay recorded as observations, but
            # cannot by themselves trigger repeated exploitation.
            if report.confirmed_progress:
                self.graph.tag_high_value_edge(
                    self.last_state_hash,
                    self.last_action_signature,
                    # A level-completing click may be valid for that level,
                    # but the same screen coordinate is not evidence of a
                    # useful target in the next level.
                    globally_reusable=not level_advanced,
                )

            if any(s.kind == "level_advanced" for s in report.signals):
                print("[CYRUS] Progress signal: level advanced")
            elif any(s.kind == "score_increased" for s in report.signals):
                print("[CYRUS] Progress signal: score increased")
            elif any(s.kind == "new_blob_count" for s in report.signals):
                print("[CYRUS] Observation signal: new blob(s) appeared")
            elif any(s.kind == "new_visual_values" for s in report.signals):
                print("[CYRUS] Observation signal: new blob value appeared")

            self.experiment_log.complete(
                state_hash_after=state["hash"],
                levels_after=levels_after,
                score_after=score_after,
                evidence=[
                    {
                        "kind": signal.kind,
                        "strength": signal.strength,
                        "details": signal.details,
                    }
                    for signal in report.signals
                ],
                caused_reset=caused_reset,
                terminal=terminal,
                confirmed_progress=report.confirmed_progress,
            )

            # Learning runs in every condition. CYRUS_SYSTEM2_ENABLED controls
            # takeover only; CYRUS_LEARNED_RANKING_ENABLED controls whether
            # System 1 reads the learned values below.
            self.system2.observe(
                state_before=self.previous_state,
                sig=self.last_action_signature,
                state_after=state,
                confirmed_progress=report.confirmed_progress,
            )

        if levels_after is not None:
            self.last_levels_completed = levels_after
        self.last_score = score_after
        if level_advanced:
            self._reset_level_probe_counters()
        return report, level_advanced

    def _reset_level_probe_counters(self) -> None:
        """Start a fresh, bounded probe budget after an environment level edge."""
        self.movement_probe_trials.clear()
        self.movement_probe_progress.clear()
        self.click_class_trials.clear()
        self.click_class_progress.clear()
        self.empty_click_trials = 0
        self.empty_click_progress = False

    @staticmethod
    def _click_class_for_signature(state, sig):
        if (not sig or int(sig[0]) != GameAction.ACTION6.value
                or sig[1] is None or sig[2] is None):
            return None
        x, y = int(sig[1]), int(sig[2])
        hits = [
            obj for obj in state.get("objects", ())
            if (y, x) in obj.get("cells", ())
            and not InteractionModel.is_scene_like(obj, state)
        ]
        if not hits:
            return None
        target = min(hits, key=lambda obj: int(obj.get("area", 0)))
        return InteractionModel.object_class(target)

    @staticmethod
    def _click_candidate_class(state, sig):
        """Class of the same-color component hit by a click signature."""
        if (not state or not sig or int(sig[0]) != GameAction.ACTION6.value
                or sig[1] is None or sig[2] is None):
            return None
        x, y = int(sig[1]), int(sig[2])
        for blob in state.get("blobs", ()):
            cells = blob.get("coords", ())
            if (y, x) not in cells:
                continue
            min_row = min(r for r, _ in cells)
            min_col = min(c for _, c in cells)
            normalized = tuple(sorted(
                (int(row) - min_row, int(col) - min_col)
                for row, col in cells
            ))
            return (int(blob.get("primary_value", -1)), len(cells), normalized)
        return None

    def _record_probe_attempt(self, state, sig) -> None:
        action_id = int(sig[0])
        if 1 <= action_id <= 5:
            self.movement_probe_trials[action_id] += 1
        elif action_id == GameAction.ACTION6.value:
            click_class = self._click_class_for_signature(state, sig)
            if click_class is None:
                self.empty_click_trials += 1
            else:
                self.click_class_trials[click_class] += 1

    def _find_frontier_with_iterative_deepening(self, root_hash):
        """Find the shallowest frontier below D, widening D only on exhaustion."""
        while self.frontier_depth_index < len(self.frontier_depth_schedule):
            depth_limit = self.frontier_depth_schedule[self.frontier_depth_index]
            frontier = self.graph.get_shallowest_frontier(
                root_hash, max_depth=depth_limit
            )
            if frontier is not None:
                self.frontier_depth_limit = depth_limit
                self.frontier_attempt_cap = depth_limit
                self.frontier_replay_target = frontier[0]
                return frontier
            if self.frontier_depth_index + 1 < len(self.frontier_depth_schedule):
                self.frontier_depth_index += 1
                self.frontier_depth_limit = self.frontier_depth_schedule[
                    self.frontier_depth_index
                ]
                self.frontier_attempt_cap = self.frontier_depth_limit
                print(
                    "[CYRUS] Widening frontier depth to "
                    f"{self.frontier_depth_limit}"
                )
            else:
                break
        self.frontier_replay_target = None
        return None

    def _close_frontier_attempt(self, end_hash, reason):
        """Record only attempts that actually executed a non-replay action."""
        if self.frontier_attempt_actions <= 0:
            self.frontier_attempt_actions = 0
            self._attempt_start_hash = None
            return
        self.frontier_attempts.append({
            "frontier_state": self._attempt_start_hash or end_hash,
            "end_state": end_hash,
            "actions": self.frontier_attempt_actions,
            "depth_limit": self.frontier_depth_limit,
            "reason": reason,
        })
        self.frontier_attempt_actions = 0
        self._attempt_start_hash = None

    def _frontier_reset_allowed(self):
        """Avoid repeated resets and resets before any level action."""
        previous_was_reset = bool(
            self.last_action_signature
            and int(self.last_action_signature[0]) == GameAction.RESET.value
        )
        return (not previous_was_reset
                and self.actions_since_level_start > 0
                and self.actions_since_reset > 0)

    def _replaying_to_frontier(self, state_hash):
        """Keep following the stored route until its frontier is reached."""
        return (self.frontier_replay_target is not None
                and self.frontier_replay_target != state_hash)

    def choose_action(self, frames, latest_frame) -> GameAction:
        """Run one System 1/System 2 decision step."""

        if latest_frame.state.name == "GAME_OVER":
            self.budget.record_step()
            self.last_state_hash = None
            self.last_action_signature = None
            self.previous_state = None
            self.frontier_reset_due = False
            self.actions_since_reset = 0
            self.actions_since_level_start = 0
            self.game_over_resets += 1
            self._just_reset_after_game_over = True
            self.last_selection_reason = "game_over_reset"
            print("[CYRUS] GAME_OVER; resetting current level")
            return GameAction.RESET

        first_action_after_reset = self._just_reset_after_game_over
        self._just_reset_after_game_over = False

        # ------------------------------------------------------------------
        # 1. PERCEPTION
        # ------------------------------------------------------------------

        if getattr(latest_frame, "levels_completed", 0) > self.last_levels_completed:
            self.frame_processor.reset_target_history()
            self._reset_level_probe_counters()

        raw = np.array(latest_frame.frame)

        if self.DEBUG_FRAME_DIFF:
            prev_raw = getattr(self, "_debug_prev_raw", None)
            if prev_raw is not None and prev_raw.shape == raw.shape:
                diff_mask = prev_raw != raw
                changed = int(diff_mask.sum())
                locations = np.argwhere(diff_mask)[:10].tolist()
                print(
                    f"[CYRUS][frame-diff] changed_cells={changed} "
                    f"sample_locations={locations}"
                )
            self._debug_prev_raw = raw.copy()

        state = self.frame_processor.process(raw)
        state_hash = state["hash"]

        # ------------------------------------------------------------------
        # 2. REGISTER AVAILABLE EXPERIMENTS
        # ------------------------------------------------------------------

        self.graph.add_state(state_hash)

        if self.frontier_root_hash is None:
            self.frontier_root_hash = state_hash

        node = self.graph.nodes[state_hash]

        action_signatures = self._get_action_signatures(
            latest_frame.available_actions,
            state,
        )

        for action_signature, tier in action_signatures:
            node.add_action(
                action_signature,
                priority_tier=tier,
            )

        # ------------------------------------------------------------------
        # 3. RECORD THE OUTCOME OF THE PREVIOUS EXPERIMENT
        # ------------------------------------------------------------------

        _, level_advanced = self._record_observation(
            latest_frame,
            state,
            terminal=False,
        )

        if level_advanced:
            # A confirmed level transition starts a separate observed graph
            # search root. The previous level graph remains available only
            # as history and is not treated as reachable in this level.
            self.frontier_root_hash = state_hash
            self.frontier_replay_target = None
            self.frontier_depth_index = 0
            self.frontier_depth_limit = self.frontier_depth_schedule[0]
            self.frontier_attempt_cap = self.frontier_depth_limit
            self.actions_since_level_start = 0

        reset_was_action = bool(
            self.last_action_signature
            and int(self.last_action_signature[0]) == GameAction.RESET.value
        )
        if first_action_after_reset or (
            self.frontier_mode_enabled and reset_was_action
        ):
            frontier = self._find_frontier_with_iterative_deepening(
                self.frontier_root_hash or state_hash
            )
            if frontier:
                print(
                    "[CYRUS] Reset replay target: "
                    f"depth={len(frontier[1]) - 1} state={frontier[0][:10]}"
                )
            # A full reset at action_count zero is unsafe to issue twice. The
            # newly reset root must either expose an untested action or route
            # toward this selected frontier; it never schedules another RESET.
            if reset_was_action:
                self.actions_since_reset = 0

        # ------------------------------------------------------------------
        # 5. UPDATE LEVEL / BUDGET STATE
        # ------------------------------------------------------------------

        if level_advanced:
            self.budget.record_level_result(solved=True)
            self.budget.start_level()

        self.budget.record_step()

        if self.frontier_mode_enabled and not level_advanced:
            current_untested = node.get_untested_actions()
            cap_reached = (
                self._attempt_start_hash is not None
                and self.frontier_attempt_actions >= self.frontier_attempt_cap
            )
            if ((not current_untested or cap_reached)
                    and not self._replaying_to_frontier(state_hash)):
                reason = "state_fully_explored" if not current_untested else "attempt_cap"
                self._close_frontier_attempt(state_hash, reason)
                self.frontier_reset_due = False
                frontier = self._find_frontier_with_iterative_deepening(
                    self.frontier_root_hash or state_hash
                )
                if (frontier is not None and frontier[0] != state_hash
                        and self._frontier_reset_allowed()):
                    reset_sig = (GameAction.RESET.value, None, None)
                    self.last_state_hash = state_hash
                    self.last_action_signature = reset_sig
                    self.previous_state = state
                    self._last_selection_was_replay = False
                    self.last_selection_reason = "system1_frontier_reset"
                    self.experiment_log.begin(
                        step=self.action_counter,
                        state_hash=state_hash,
                        state_features=self.goal_tracker.summarize_state(state),
                        action_signature=reset_sig,
                        action_id=GameAction.RESET.value,
                        levels_before=latest_frame.levels_completed,
                        score_before=self.last_score,
                        selection_reason=self.last_selection_reason,
                    )
                    self.actions_since_reset = 0
                    self.actions_since_level_start = 0
                    print("[CYRUS] RESET to replay selected frontier")
                    return GameAction.RESET
                # No frontier remains under the current depth. The helper has
                # widened D; component classes can now enter once their simple
                # actions are exhausted. Continue locally without resetting.
                action_signatures = self._get_action_signatures(
                    latest_frame.available_actions, state
                )
                for action_signature, tier in action_signatures:
                    node.add_action(action_signature, tier)
                current_untested = node.get_untested_actions()
                if not current_untested:
                    # The graph has no frontier at any permitted depth and no
                    # newly widened candidate at this state. Never repeat a
                    # tested state-action pair; allow the normal completion
                    # / action budget to terminate this run instead.
                    self.frontier_search_exhausted = True
                    self.last_selection_reason = "system1_search_exhausted"
                    self.last_state_hash = state_hash
                    self.last_action_signature = (GameAction.RESET.value, None, None)
                    self.previous_state = state
                    self.actions_since_reset = 0
                    self.actions_since_level_start = 0
                    return GameAction.RESET

        # ------------------------------------------------------------------
        # 6. SELECT NEXT EXPERIMENT
        # ------------------------------------------------------------------

        selection_reason = "system1_evidence_calibrated"
        action_signature = None
        self._last_selection_was_replay = False
        if self.relational_planner_enabled:
            action_signature = self.relational_planner.choose(
                state,
                [signature for signature, _ in action_signatures],
                level_key=getattr(latest_frame, "levels_completed", 0),
            )
            if action_signature is not None:
                selection_reason = "visual_relational_plan"
                print(
                    "[CYRUS] Visual relational planner action: "
                    f"{action_signature}"
                )

        if (action_signature is None and self.system2_takeover_enabled
                and self.system2.wants_control()):
            action_signature = self.system2.choose(
                state,
                [signature for signature, _ in action_signatures],
            )
            if action_signature is not None:
                selection_reason = "system2_hypothesis_control"
                self.system2_selected_actions += 1
                print(
                    "[CYRUS] System 2 selected experiment: "
                    f"{action_signature}"
                )

        if action_signature is None:
            action_signature = self._select_action(
                state_hash,
                node,
                state,
                latest_frame,
            )
            self.system2.watch(state, action_signature)
            if self._last_selection_was_replay:
                selection_reason = "system1_frontier_replay"
        if self.frontier_mode_enabled and not self._last_selection_was_replay:
            if self._attempt_start_hash is None:
                self._attempt_start_hash = state_hash
                self.frontier_attempt_actions = 0
            self.frontier_attempt_actions += 1

        # ------------------------------------------------------------------
        # 7. BUILD GAME ACTION
        # ------------------------------------------------------------------

        action_id, click_xy = self._decode_action_signature(
            action_signature
        )

        game_action = GameAction.from_id(action_id)

        self._record_probe_attempt(state, action_signature)

        if game_action.is_complex():
            if click_xy is None:
                click_xy = (32, 32)

            x, y = click_xy

            game_action.set_data(
                {
                    "x": int(x),
                    "y": int(y),
                }
            )

        # ------------------------------------------------------------------
        # 8. REMEMBER CURRENT EXPERIMENT
        # ------------------------------------------------------------------

        self.last_state_hash = state_hash
        self.last_action_signature = action_signature
        self.previous_state = state
        selected_edge = node.actions.get(action_signature)
        self.frontier_reset_due = False
        self.last_selection_reason = selection_reason
        self.experiment_log.begin(
            step=self.action_counter,
            state_hash=state_hash,
            state_features=self.goal_tracker.summarize_state(state),
            action_signature=action_signature,
            action_id=action_id,
            levels_before=latest_frame.levels_completed,
            score_before=self.last_score,
            selection_reason=selection_reason,
        )

        if action_id == GameAction.RESET.value:
            self.actions_since_reset = 0
            self.actions_since_level_start = 0
        else:
            self.actions_since_reset += 1
            self.actions_since_level_start += 1

        # If the first action after a reset immediately loses again twice in
        # succession, stop instead of burning the remaining action budget in
        # an unproductive RESET/action loop. A run that survives this action
        # clears the stall count in is_done().
        self._post_reset_action_pending = first_action_after_reset

        return game_action

    # ======================================================================
    # Cleanup / diagnostics
    # ======================================================================

    def cleanup(self, scorecard=None) -> None:
        """
        Print CYRUS diagnostics after the game ends.
        """

        self.experiment_log.close_pending()
        super().cleanup(scorecard)

        stats = self.graph.get_stats()

        print("\n--- CYRUS diagnostic ---")

        print(
            f"Total steps taken:     {self.action_counter}"
        )

        if self.first_level_advance_action is None:
            print("Actions to first confirmed level advance: not reached")
        else:
            print(
                "Actions to first confirmed level advance: "
                f"{self.first_level_advance_action}"
            )

        print(
            f"Unique states seen:    {stats['num_states']}"
        )

        print(
            f"Total actions tracked: {stats['total_actions']}"
        )

        print(
            f"Tested actions:        {stats['tested_actions']}"
        )

        print(
            f"High-value actions:    {stats['high_value_actions']}"
        )

        print(
            f"Untested actions:      {stats['untested_actions']}"
        )

        experiment_stats = self.experiment_log.get_summary()
        print(
            "Experiment records:    "
            f"{experiment_stats['records_completed']} completed, "
            f"{experiment_stats['confirmed_progress']} confirmed progress, "
            f"{experiment_stats['visual_observations']} visual observations"
        )
        print(
            "Evidence diagnostics:  "
            f"{experiment_stats['negative_evidence']} negative, "
            f"{experiment_stats['resets']} resets, "
            f"{experiment_stats['dropped_from_memory']} dropped from memory"
        )
        if self.experiment_log.output_path:
            print(
                "Experiment JSONL:     "
                f"{self.experiment_log.output_path}"
            )

        if self.action_counter > 0:
            ratio = (
                stats["num_states"]
                / self.action_counter
            )

            print(
                f"States-per-step ratio: {ratio:.2f}"
            )

            if ratio > 0.9:
                print(
                    "WARNING: almost every step produced "
                    "a brand-new state."
                )

                print(
                    "This may indicate that the state hash "
                    "still contains a changing element."
                )

        print("------------------------\n")

    # ======================================================================
    # Action representation
    # ======================================================================

    def _get_action_signatures(
        self,
        available_actions,
        state,
    ):
        """
        Convert available GameActions into (signature, priority_tier) pairs.

        Simple actions:

            ((1, None, None), tier)

        ACTION6:

            ((6, x, y), tier)

        Every grouped visible target becomes an ACTION6 experiment. Target
        priority comes from palette-independent visual features only.
        """

        signatures = []

        # Keep every exposed action in the candidate set. The former
        # movement-only probe gate suppressed ACTION6 until several movement
        # attempts had already been spent, which made a known interactive
        # edge click unreachable under short level timers.
        movement_probe_active = False

        for action_id in available_actions:

            action_id = int(action_id)

            # ----------------------------------------------------------
            # Coordinate-based ACTION6
            # ----------------------------------------------------------

            if action_id == GameAction.ACTION6.value:

                # Keep action coordinates on real same-color components.
                # Grouped-object features are diagnostic only until their
                # ranking effect is validated against gameplay results.
                blobs = state.get("blobs", [])
                found_any = False
                class_candidates = {}

                for blob in blobs:

                    coords = blob.get("coords", [])

                    if not coords:
                        continue

                    rows = [
                        r
                        for r, c in coords
                    ]

                    cols = [
                        c
                        for r, c in coords
                    ]

                    # Frame coordinates are represented as:
                    #     row, column
                    #
                    # GameAction expects:
                    #     x, y
                    #
                    y = int(
                        sum(rows) / len(rows)
                    )

                    x = int(
                        sum(cols) / len(cols)
                    )

                    min_row = min(row for row, _ in coords)
                    min_col = min(col for _, col in coords)
                    normalized = tuple(sorted(
                        (int(r) - min_row, int(c) - min_col)
                        for r, c in coords
                    ))
                    class_key = (
                        int(blob.get("primary_value", -1)),
                        len(coords), normalized,
                    )
                    class_candidates.setdefault(class_key, []).append(
                        (action_id, x, y, int(blob.get("priority_tier", 2)))
                    )

                for class_key, candidates in class_candidates.items():
                    candidates.sort(key=lambda item: (item[2], item[1]))
                    representative = candidates[0]
                    changed_before = class_key in getattr(
                        self, "click_class_changed", set()
                    )
                    for candidate in (candidates if changed_before
                                      else [representative]):
                        signature = candidate[:3]
                        tier = candidate[3]
                        signatures.append((signature, tier))
                        found_any = True

                # Bounded generic edge sweep. This complements component
                # clicks with probes along the outer rows/columns without
                # enumerating every pixel. Coordinates use (x, y).
                grid_for_edges = np.asarray(
                    state.get("masked_grid", state.get("grid"))
                )
                if grid_for_edges.ndim == 3:
                    grid_for_edges = grid_for_edges[-1]
                # Edge candidates are deferred until every component
                # representative has been tested from this exact state.
                state_node = getattr(self, "graph", None)
                state_node = (state_node.nodes.get(state.get("hash"))
                              if state_node is not None else None)
                reps = [items[0][:3] for items in class_candidates.values()]
                components_exhausted = bool(state_node) and all(
                    (action_id, x, y) in state_node.actions
                    and state_node.actions[(action_id, x, y)].tested
                    for _aid, x, y in reps
                )
                if (components_exhausted and grid_for_edges.ndim == 2
                        and grid_for_edges.size):
                    height, width = grid_for_edges.shape
                    # Bounded edge sweep: sample corners, quarters, and both
                    # central pixels on each axis. This includes positions
                    # between coarse grid points while avoiding hundreds of
                    # nearly identical click branches per state.
                    edge_x = sorted({
                        0, width - 1, width // 4, width // 2 - 1,
                        width // 2, (3 * width) // 4,
                    })
                    edge_y = sorted({
                        0, height - 1, height // 4, height // 2 - 1,
                        height // 2, (3 * height) // 4,
                    })
                    present = {item[0] for item in signatures}
                    for y in (0, height - 1):
                        for x in edge_x:
                            signature = (action_id, x, y)
                            if signature not in present:
                                signatures.append((signature, 2))
                                present.add(signature)
                    for x in (0, width - 1):
                        for y in edge_y:
                            signature = (action_id, x, y)
                            if signature not in present:
                                signatures.append((signature, 2))
                                present.add(signature)

                # Sample a small grid of visually empty destinations. This
                # lets the same controller discover actions such as
                # select-on-object / move-to-empty-space without requiring a
                # game-specific destination script.
                grid = np.asarray(state.get("masked_grid", state.get("grid")))
                if (not movement_probe_active and grid.ndim == 3):
                    grid = grid[-1]
                if (not movement_probe_active and components_exhausted and grid.ndim == 2
                        and grid.size and (self.empty_click_trials == 0
                                           or self.empty_click_progress)):
                    background = int(np.bincount(
                        grid.astype(np.int64).ravel()
                    ).argmax())
                    occupied = {
                        (row, col)
                        for obj in state.get("objects", [])
                        for row, col in obj.get("cells", ())
                    }
                    existing = {item[0] for item in signatures}
                    for y in (8, 24, 40, 56):
                        for x in (8, 24, 40, 56):
                            signature = (action_id, x, y)
                            if (
                                signature in existing
                                or (y, x) in occupied
                                or int(grid[y, x]) != background
                            ):
                                continue
                            signatures.append((signature, 2))
                            existing.add(signature)

                # If ACTION6 exists but no target was found at all,
                # preserve a safe center click.
                if (not found_any and (self.empty_click_trials == 0
                                       or self.empty_click_progress)):
                    signatures.append(
                        (
                            (action_id, 32, 32),
                            1,
                        )
                    )

            # ----------------------------------------------------------
            # Simple action
            # ----------------------------------------------------------

            else:
                signatures.append(
                    (
                        (action_id, None, None),
                        self._tier_for_action(action_id, state),
                    )
                )

        # Iterative widening: movement/simple controls are exposed first.
        # Coordinate clicks enter this state once its simple controls have
        # each been tried, unless a click class has already changed a frame
        # elsewhere. Existing class no-op evidence affects ranking only.
        if (getattr(self, "frontier_mode_enabled", False)
                and getattr(self, "search_order_enabled", False)):
            node = getattr(self, "graph", None)
            node = (node.nodes.get(state.get("hash"))
                    if node is not None else None)
            simple = [sig for sig, _tier in signatures if int(sig[0]) != 6]
            simple_exhausted = bool(node is not None) and all(
                sig in node.actions and node.actions[sig].tested
                for sig in simple
            )
            if simple and not simple_exhausted:
                signatures = [
                    (sig, tier) for sig, tier in signatures
                    if int(sig[0]) != 6
                    or self._click_candidate_class(state, sig)
                    in getattr(self, "click_class_changed", set())
                ]

        return signatures

    def _decode_action_signature(
        self,
        action_signature,
    ):
        """
        Convert an experiment signature into:

            (action_id, click_xy)

        Examples:

            (1, None, None)
                ->
            (1, None)

            (6, 21, 35)
                ->
            (6, (21, 35))
        """

        if action_signature is None:
            return (
                GameAction.ACTION1.value,
                None,
            )

        action_id = int(
            action_signature[0]
        )

        if (
            len(action_signature) >= 3
            and action_signature[1] is not None
            and action_signature[2] is not None
        ):
            return (
                action_id,
                (
                    int(action_signature[1]),
                    int(action_signature[2]),
                ),
            )

        return (
            action_id,
            None,
        )

    # ======================================================================
    # System 1 action selection
    # ======================================================================

    def _select_action(
        self,
        state_hash,
        node,
        state,
        latest_frame,
    ):
        """
        Select the next experiment.

        Priority:

            0. Globally high-value action (this exact signature, or a
               nearby coordinate, has caused progress anywhere in this
               run - not just from this exact state)
            1. High-value untested experiment (local to this state)
            2. Any untested experiment
            3. High-value tested experiment (local to this state)
            4. Follow graph path toward frontier
            5. Random fallback
        """

        # --------------------------------------------------------------
        # 0. Globally high-value action, regardless of which exact state
        #    we're currently in.
        #
        # Why this matters: node.actions[...].high_value is set on one
        # specific (state_hash, action_sig) edge. In a game where almost
        # every frame hashes to a brand-new state, that exact edge is
        # gone the moment we move on - CYRUS would otherwise "forget" a
        # discovery as soon as it leaves the state where it made it.
        # global_action_stats tracks success per action SIGNATURE across
        # every state, so a target that worked once keeps being favored
        # wherever it shows up again (fuzzy-matched by coordinate, since
        # blob centroids drift a few pixels between near-identical
        # frames).
        #
        # COOLDOWN: "worked once" is not "works every time". A click that
        # caused ONE progress signal (e.g. a panel opening) can still be
        # a dead end on every repeat after that - clicking it forever
        # produces no further "Progress signal" lines and burns the whole
        # action budget. So an action is only eligible here if it hasn't
        # already been retried too many times recently without a NEW
        # success. Once it falls out of the recent window, it becomes
        # eligible again - giving periodic re-checks instead of either
        # permanent exploitation or permanently giving up on it.
        # --------------------------------------------------------------

        GLOBAL_REUSE_WINDOW = 8
        GLOBAL_REUSE_CAP = 2
        level_id = int(getattr(latest_frame, "levels_completed", 0))

        def exploration_key(edge):
            outcomes = self.graph.action_outcome_counts(
                edge.action_sig,
                level_id=level_id,
                coordinate_radius=0,
            )
            unsuccessful = (
                outcomes["no_confirmed_progress"]
                + outcomes["negative_evidence"]
            )
            return (
                self.graph.distinct_state_trials(
                    edge.action_sig, level_id=level_id
                ),
                self.graph.recent_action_count(
                    edge.action_sig,
                    level_id=level_id,
                    coordinate_radius=0,
                ),
                unsuccessful,
                outcomes["attempts"],
                # Distinct-state no-op evidence deprioritizes an action but
                # never removes it from the candidate set. Keep this ahead
                # of action-ID priors so repeated no-ops actually affect
                # selection instead of serving as a diagnostic-only tie-break.
                int(
                    len(self.action_noop_states.get(edge.action_sig, ()))
                    >= self.noop_deprioritize_after
                ),
                len(self.action_noop_states.get(edge.action_sig, ())),
                int(
                    (self._click_candidate_class(state, edge.action_sig) in
                     self.click_class_noop_states)
                    and len(self.click_class_noop_states.get(
                        self._click_candidate_class(state, edge.action_sig), ()
                    )) >= self.noop_deprioritize_after
                ),
                # Cross-run experience is a tie-breaker for actions that
                # have the same local exploration cost. The interaction
                # model keys this by coarse visual shape and action context,
                # so it can transfer without reusing screen coordinates.
                *self._learned_ranking_key(edge.action_sig, state),
                edge.priority_tier,
                edge.times_tested,
            )

        # After GAME_OVER/RESET, first replay observed transitions to the
        # shallowest known state with an untested action. This preserves the
        # graph across attempts and avoids restarting with a fresh random
        # walk. The route is cleared as soon as its target is reached.
        if self.frontier_replay_target is not None:
            if state_hash == self.frontier_replay_target:
                self.frontier_replay_target = None
            else:
                replay_path = self.graph.get_shortest_path(
                    state_hash, self.frontier_replay_target
                )
                if replay_path and len(replay_path) > 1:
                    replay_action = self._find_action_to_state(
                        replay_path[0], replay_path[1]
                    )
                    if replay_action is not None:
                        self._last_selection_was_replay = True
                        print(
                            "[CYRUS] Replaying graph path to frontier: "
                            f"{replay_action}"
                        )
                        return replay_action
                # A reset may yield a visually different root (for example,
                # a timer or animation is part of the observed frame). In
                # that case the stored path is not reachable; abandon this
                # replay target and continue with observed local evidence.
                self.frontier_replay_target = None

        globally_valuable = [
            edge
            for edge in node.actions.values()
            if self.graph.is_globally_high_value(
                edge.action_sig,
                level_id=level_id,
            )
            and not edge.tested
            and self.graph.recent_action_count(
                edge.action_sig,
                window_size=GLOBAL_REUSE_WINDOW,
                level_id=level_id,
            ) < GLOBAL_REUSE_CAP
        ]

        if globally_valuable:

            globally_valuable.sort(
                key=lambda edge: (
                    edge.tested,  # try it here first if we haven't yet
                    self.graph.recent_action_count(
                        edge.action_sig,
                        level_id=level_id,
                    ),
                    -self.graph.global_success_count(
                        edge.action_sig,
                        level_id=level_id,
                    ),
                )
            )

            chosen = globally_valuable[0]

            print(
                "[CYRUS] Reusing globally high-value "
                f"experiment: {chosen.action_sig}"
            )

            return chosen.action_sig

        # --------------------------------------------------------------
        # 1. High-value untested experiments
        # --------------------------------------------------------------

        untested = node.get_untested_actions()

        high_value_untested = [
            edge
            for edge in untested
            if edge.high_value
        ]

        if high_value_untested:

            high_value_untested.sort(
                key=lambda edge: (
                    edge.priority_tier,
                    -edge.success_count,
                )
            )

            chosen = high_value_untested[0]

            print(
                "[CYRUS] Choosing high-value "
                f"untested experiment: "
                f"{chosen.action_sig}"
            )

            return chosen.action_sig

        # --------------------------------------------------------------
        # 2. Any untested experiment
        # --------------------------------------------------------------

        if untested:

            # Recency comes FIRST. Priority tier is a prior about what's
            # *likely* interactive, but when states rarely repeat exactly
            # (ratio close to 1.0), a fresh node always re-offers the same
            # "best tier" candidate as untested - that's what caused CYRUS
            # to alternate between only 2 coordinates forever on vc33.
            # Sorting by recent_action_count first means a target clicked
            # a moment ago gets pushed down even in a brand-new state,
            # forcing exploration to actually diversify.
            self.learned_ranking_decisions += int(self.learned_ranking_enabled)
            ranked_choice = min(untested, key=exploration_key)
            if self.learned_ranking_enabled:
                self.learned_ranking_choice_changes += int(
                    ranked_choice.action_sig != min(
                        untested,
                        key=lambda edge: (
                            self.graph.recent_action_count(edge.action_sig, level_id=level_id),
                            self.graph.action_outcome_counts(edge.action_sig, level_id=level_id)["no_confirmed_progress"]
                            + self.graph.action_outcome_counts(edge.action_sig, level_id=level_id)["negative_evidence"],
                            self.graph.action_outcome_counts(edge.action_sig, level_id=level_id)["attempts"],
                            0, 0, edge.priority_tier, edge.times_tested,
                        ),
                    ).action_sig
                )
            untested.sort(key=exploration_key)

            chosen = untested[0]

            print(
                "[CYRUS] Exploring experiment: "
                f"{chosen.action_sig}"
            )

            return chosen.action_sig

        # --------------------------------------------------------------
        # 3. High-value tested experiment
        # --------------------------------------------------------------

        high_value_tested = [
            edge
            for edge in node.actions.values()
            if edge.high_value
        ]

        if high_value_tested:

            high_value_tested.sort(
                key=lambda edge: (
                    -edge.success_count,
                    edge.priority_tier,
                )
            )

            chosen = high_value_tested[0]

            print(
                "[CYRUS] Reusing high-value "
                f"experiment: {chosen.action_sig}"
            )

            return chosen.action_sig

        # --------------------------------------------------------------
        # 4. Navigate toward graph frontier
        # --------------------------------------------------------------

        frontier = self.graph.get_frontier(
            state_hash
        )

        if frontier is not None:

            frontier_state_hash, path = frontier

            # Already at the frontier.
            if len(path) == 1:

                frontier_node = self.graph.nodes.get(
                    frontier_state_hash
                )

                if frontier_node:

                    frontier_untested = (
                        frontier_node.get_untested_actions()
                    )

                    if frontier_untested:

                        # Same recency-first rationale as step 2.
                        frontier_untested.sort(key=exploration_key)

                        chosen = frontier_untested[0]

                        print(
                            "[CYRUS] At frontier, "
                            "exploring: "
                            f"{chosen.action_sig}"
                        )

                        return chosen.action_sig

            # ----------------------------------------------------------
            # Follow the first edge of the shortest known path.
            # ----------------------------------------------------------

            if len(path) >= 2:

                next_state_hash = path[1]

                next_action = (
                    self._find_action_to_state(
                        state_hash,
                        next_state_hash,
                    )
                )

                if next_action is not None:

                    print(
                        "[CYRUS] Navigating toward "
                        f"frontier: "
                        f"{next_action}"
                    )

                    return next_action

        # --------------------------------------------------------------
        # 5. Random fallback
        # --------------------------------------------------------------

        available_signatures = (
            self._get_action_signatures(
                latest_frame.available_actions,
                state,
            )
        )

        if available_signatures:

            chosen, _tier = random.choice(
                available_signatures
            )

            print(
                "[CYRUS] Random fallback: "
                f"{chosen}"
            )

            return chosen

        # Use an action this game actually exposes. ACTION6 requires a
        # coordinate, so use the display center when perception found no
        # candidate target.
        if latest_frame.available_actions:
            fallback_action = int(latest_frame.available_actions[0])
            if fallback_action == GameAction.ACTION6.value:
                return (fallback_action, 32, 32)
            return (fallback_action, None, None)

        return (GameAction.ACTION1.value, None, None)

    def _learned_ranking_key(self, action_signature, state):
        """System 1 ranking terms, independently gated from System 2 takeover."""
        if not self.learned_ranking_enabled:
            return 0.0, 0.0
        return (
            -self.system2.interactions.value(action_signature, state),
            -self.system2.goals.relevance(state, action_signature),
        )

    # ======================================================================
    # Graph path helpers
    # ======================================================================

    def _find_action_to_state(
        self,
        from_state_hash,
        target_state_hash,
    ):
        """
        Find the remembered action from one state directly to another.

        Returns the complete action signature.
        """

        node = self.graph.nodes.get(
            from_state_hash
        )

        if node is None:
            return None

        for edge in node.actions.values():

            if (
                edge.result_hash
                == target_state_hash
            ):
                return edge.action_sig

        return None

    # ======================================================================
    # Action priority
    # ======================================================================

    def _tier_for_action(
        self,
        action_id: int,
        state,
    ) -> int:
        """
        Assign a basic exploration priority.

        Tier 0:
            Safe movement / exploration

        Tier 1:
            Coordinate-based interaction

        Tier 2:
            Other neutral actions

        Tier 3:
            Potentially destructive / rare actions
        """

        action_id = int(action_id)

        # Movement / directional actions.
        if action_id in [1, 2, 3, 4, 5]:
            return 0

        # Coordinate-based interaction.
        if action_id == GameAction.ACTION6.value:
            return 1

        # Potentially destructive / uncommon action.
        if action_id == 7:
            return 3

        return 2
