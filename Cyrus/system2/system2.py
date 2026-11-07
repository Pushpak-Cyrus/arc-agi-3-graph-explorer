"""CYRUS System 2: deliberate hypothesis-driven control.

System 2 runs alongside System 1 and takes over action choice only when
exploration has stalled. Its loop mirrors the design doc:

    observe -> update world model -> infer goal -> generate hypotheses
            -> predict consequences -> choose action -> observe result
            -> compare prediction with reality -> update

Action score (design doc, Section 12 and 14):

    Score(a) = alpha * InfoGain(a) + beta * GoalRelevance(a)
               + LearnedInteraction(a) - gamma * Risk(a)

    InfoGain   mean uncertainty of the hypotheses that predict on a
               (1.0 if no hypothesis has an opinion, i.e. a novel action)
    GoalRelevance  weight of goal hypotheses that a's target color or object
                    class touches
    LearnedInteraction  exploration bonus or progress lift for the action's
                        object/empty-space context
    Risk       probability a is a no-op, plus a penalty for recent repeats

Gate: System 2 takes control after ``stall_threshold`` steps without
confirmed progress, and keeps it for ``engage_steps`` steps.
"""

import json
import os
from copy import deepcopy
from collections import deque
from typing import Dict, List, Optional, Sequence

import numpy as np

from .frame_diff import diff_states, grid_of
from .action_effect_learner import (
    ActionEffectLearner,
    _non_background_blobs,
    _shape_key,
)
from .goal_inference import GoalInference
from .interaction_model import InteractionModel
from .local_llm_advisor import LocalLLMAdvisor
from .world_model import WorldModel


class System2:
    def __init__(
        self,
        stall_threshold: int = 30,
        engage_steps: int = 10,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 1.0,
        log_path: Optional[str] = None,
        game_id: Optional[str] = None,
    ) -> None:
        self.stall_threshold = stall_threshold
        self.engage_steps = engage_steps
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.world = WorldModel()
        # Learns palette-free, repeatable action effects from the same live
        # transitions used by the prediction-error model. It is diagnostic
        # evidence until a goal hypothesis is independently validated.
        self.action_effects = ActionEffectLearner()
        self.goals = GoalInference(
            memory_path=os.getenv("CYRUS_GOAL_MEMORY")
        )
        self.interactions = InteractionModel(
            memory_path=os.getenv("CYRUS_LEARNING_MEMORY")
        )
        self.local_llm = LocalLLMAdvisor()
        try:
            self.local_llm_interval = max(
                1, int(os.getenv("CYRUS_LOCAL_LLM_INTERVAL", "4"))
            )
        except ValueError:
            self.local_llm_interval = 4
        try:
            self.local_llm_tie_epsilon = max(
                0.0, min(0.25, float(
                    os.getenv("CYRUS_LOCAL_LLM_TIE_EPSILON", "0.05")
                ))
            )
        except ValueError:
            self.local_llm_tie_epsilon = 0.05
        try:
            self.model_plan_weight = max(
                0.0, min(5.0, float(os.getenv("CYRUS_MODEL_PLAN_WEIGHT", "1.0")))
            )
        except ValueError:
            self.model_plan_weight = 1.0
        self._last_local_llm_step = -self.local_llm_interval
        self._recent_advisor_history = deque(maxlen=8)

        self.steps = 0
        self.steps_since_progress = 0
        self.engaged_remaining = 0

        self._pending: Optional[Dict] = None
        self.log_path = log_path or os.getenv("CYRUS_SYSTEM2_LOG")
        self.game_id = game_id

    # ------------------------------------------------------------------
    # Gate
    # ------------------------------------------------------------------

    def wants_control(self) -> bool:
        """True when System 2 should choose the next action."""
        if self.engaged_remaining > 0:
            return True
        if self.steps_since_progress >= self.stall_threshold:
            self.engaged_remaining = self.engage_steps
            self.steps_since_progress = 0
            return True
        return False

    # ------------------------------------------------------------------
    # Prediction and action choice
    # ------------------------------------------------------------------

    def choose(self, state, sigs: Sequence) -> Optional[tuple]:
        """Pick the highest-scoring action. Returns None if there are no candidates."""
        sigs = list(sigs)
        if not sigs:
            return None

        self.goals.ensure_goals(state)
        plan = self._plan_sequence(state, sigs)
        planned_action = (
            plan["actions"][0] if plan and plan.get("actions") else None
        )

        advice = None
        if (self.local_llm.enabled
                and self.steps - self._last_local_llm_step >= self.local_llm_interval):
            self._last_local_llm_step = self.steps
            advice = self.local_llm.advise(
                state,
                sigs,
                self.action_effects.summary(),
                list(self._recent_advisor_history),
                self.goals.candidate_details(state, k=5),
            )
            if advice and os.getenv("CYRUS_LOCAL_LLM_DEBUG") == "1":
                print(
                    "[CYRUS][local-llm] hypotheses="
                    f"{advice['hypotheses']} preferred={advice['preferred_action']} "
                    f"confidence={advice['confidence']:.2f}"
                )
            elif self.local_llm.last_error and os.getenv("CYRUS_LOCAL_LLM_DEBUG") == "1":
                print(f"[CYRUS][local-llm] unavailable: {self.local_llm.last_error}")

        preferred_action = advice.get("preferred_action") if advice else None

        scored = []
        for sig in sigs:
            effect_context = self.action_effects.context(
                self.interactions.selected_class,
                self.interactions.selected_center,
                state,
                sig=sig,
            )
            self.world.ensure_hypotheses([sig], state, context=effect_context)
            predictions = self.world.predictions(
                sig, state, context=effect_context
            )
            effect_info = self.action_effects.uncertainty(
                int(sig[0]), effect_context
            )
            info = max(self._information_gain(predictions), effect_info)
            relevance = self.goals.relevance(state, sig)
            risk = self._risk(sig)
            experience = self.interactions.value(sig, state)
            verified_plan_gain = self._verified_goal_gain(
                state, int(sig[0]), effect_context
            )
            score = (
                self.alpha * info
                + self.beta * relevance
                - self.gamma * risk
                + experience
                + self.model_plan_weight * verified_plan_gain
            )
            if planned_action is not None and sig == planned_action:
                score += 1000.0
            components = {
                "score": round(score, 4),
                "info": round(info, 4),
                "contextual_effect_info": round(effect_info, 4),
                "relevance": round(relevance, 4),
                "risk": round(risk, 4),
                "learned_interaction_value": round(experience, 4),
                "verified_plan_gain": round(verified_plan_gain, 4),
                "plan_goal": plan["goal"] if planned_action is not None else None,
                "planned_sequence": (
                    [list(item) for item in plan["actions"]]
                    if planned_action is not None else []
                ),
            }
            scored.append((
                score, -self.world.action_tries[sig], sig, predictions,
                components, effect_context,
            ))

        best = max(scored, key=lambda row: (row[0], row[1]))
        chosen_row = best
        llm_tiebreak_accepted = False
        if advice and preferred_action is not None:
            preferred_row = next(
                (row for row in scored if row[2] == preferred_action), None
            )
            if (preferred_row is not None
                    and best[0] - preferred_row[0] <= self.local_llm_tie_epsilon):
                chosen_row = preferred_row
                llm_tiebreak_accepted = True

        _, _, chosen, predictions, components, effect_context = chosen_row
        components["local_llm_tiebreak_accepted"] = llm_tiebreak_accepted
        self._pending = {
            "sig": chosen,
            "state_hash": state.get("hash"),
            "predictions": predictions,
            "effect_context": effect_context,
            "components": components,
            "controller": "system2",
            "local_llm_advice": advice,
            "local_llm_advice_applied": llm_tiebreak_accepted,
            "local_llm_prediction": (
                advice if advice and llm_tiebreak_accepted else None
            ),
            "plan": plan if planned_action is not None else None,
        }
        return chosen

    def watch(self, state, sig) -> None:
        """Save pre-action predictions for an action chosen by System 1."""
        effect_context = self.action_effects.context(
            self.interactions.selected_class,
            self.interactions.selected_center,
            state,
            sig=sig,
        )
        self.world.ensure_hypotheses([sig], state, context=effect_context)
        self.goals.ensure_goals(state)
        self._pending = {
            "sig": sig,
            "state_hash": state.get("hash"),
            "predictions": self.world.predictions(
                sig, state, context=effect_context
            ),
            "effect_context": effect_context,
            "components": {
                "learned_interaction_value": round(
                    self.interactions.value(sig, state), 4
                ),
            },
            "controller": "system1",
        }

    def _information_gain(self, predictions: Dict) -> float:
        """Largest uncertainty among the hypotheses that predict on this action.

        Max, not mean: an action carrying one untested hypothesis is still
        worth trying even if a well-known hypothesis also applies to it.
        Averaging diluted that signal and left the agent stuck on known actions.
        """
        uncertainties = [
            self.world.library[key].belief.uncertainty for key in predictions
        ]
        if not uncertainties:
            return 1.0
        return max(uncertainties)

    def _risk(self, sig) -> float:
        return (
            self.world.noop_probability(int(sig[0]))
            + self.world.repeat_fraction(sig)
        )

    def _verified_goal_gain(self, state, action_id: int, context) -> float:
        """One-step model-based planning using only verified object effects.

        Each choice is recomputed from the latest observation, so a failed
        prediction cannot leave a stale multi-action plan in control.
        """
        grid = np.asarray(grid_of(state)).copy()
        if grid.ndim != 2 or not 1 <= action_id <= 6:
            return 0.0
        effects = self.action_effects.verified_effects(action_id, context)
        if not effects:
            return 0.0
        values, counts = np.unique(grid, return_counts=True)
        if not len(values):
            return 0.0
        background = int(values[int(np.argmax(counts))])
        best_gain = 0.0
        blobs = _non_background_blobs(state) if isinstance(state, dict) else []
        for effect in effects:
            dr, dc = effect["vector"]
            for blob_index, blob in enumerate(blobs):
                if _shape_key(blob) != effect["shape"]:
                    continue
                if self.action_effects.object_context(
                    state, blob_index
                ) != effect["object_context"]:
                    continue
                color = int(blob.get("primary_value", background))
                old_cells = {(int(row), int(col)) for row, col in blob["coords"]}
                new_cells = {(row + dr, col + dc) for row, col in old_cells}
                if any(
                    row < 0 or col < 0 or row >= grid.shape[0]
                    or col >= grid.shape[1]
                    for row, col in new_cells
                ):
                    continue
                if any(
                    cell not in old_cells and grid[cell] != background
                    for cell in new_cells
                ):
                    continue
                predicted_grid = grid.copy()
                for cell in old_cells:
                    predicted_grid[cell] = background
                for cell in new_cells:
                    predicted_grid[cell] = color
                gain = self.goals.shaping_delta(
                    state, {"grid": predicted_grid}
                )
                best_gain = max(best_gain, gain)
        return round(max(0.0, min(1.0, best_gain)), 4)

    @staticmethod
    def _relations_for_objects(objects):
        relations = []
        for left_idx, left in enumerate(objects):
            lr0, lc0, lr1, lc1 = left["bbox"]
            for right_idx in range(left_idx + 1, len(objects)):
                right = objects[right_idx]
                rr0, rc0, rr1, rc1 = right["bbox"]
                row_gap = max(0, rr0 - lr1 - 1, lr0 - rr1 - 1)
                col_gap = max(0, rc0 - lc1 - 1, lc0 - rc1 - 1)
                intersects = (
                    lr0 <= rr1 and rr0 <= lr1
                    and lc0 <= rc1 and rc0 <= lc1
                )
                kind = None
                if intersects:
                    kind = "overlap"
                elif row_gap <= 1 and col_gap <= 1:
                    kind = "adjacent"
                elif np.hypot(
                    left["center"][0] - right["center"][0],
                    left["center"][1] - right["center"][1],
                ) <= max(8, left["width"], left["height"],
                         right["width"], right["height"]) * 1.5:
                    kind = "near"
                if kind:
                    relations.append({"left": left_idx, "right": right_idx, "kind": kind})
                if abs(left["center"][0] - right["center"][0]) <= 1:
                    relations.append({"left": left_idx, "right": right_idx, "kind": "aligned_row"})
                if abs(left["center"][1] - right["center"][1]) <= 1:
                    relations.append({"left": left_idx, "right": right_idx, "kind": "aligned_column"})
        return relations

    def _plan_sequence(self, state, sigs, max_depth: int = 8):
        """BFS over verified learned displacements toward the best supported goal."""
        weights = self.goals.weights()
        if not weights:
            return None
        goal_name, weight = max(weights.items(), key=lambda row: row[1])
        goal = self.goals.goals.get(goal_name)
        if goal is None or goal.holds(state):
            return None

        available = {
            int(sig[0]) for sig in sigs
            if sig and sig[1] is None and 1 <= int(sig[0]) <= 5
        }
        if not available:
            return None
        grid = np.asarray(grid_of(state))
        if grid.ndim != 2:
            return None
        values, counts = np.unique(grid, return_counts=True)
        if not len(values):
            return None
        background = int(values[int(np.argmax(counts))])
        objects = state.get("objects", [])
        blobs = _non_background_blobs(state)
        controls = []

        for action_id in sorted(available):
            context = self.action_effects.context(
                self.interactions.selected_class,
                self.interactions.selected_center,
                state,
                sig=(action_id, None, None),
            )
            effects = self.action_effects.verified_effects(action_id, context)
            if not effects:
                continue
            verified = {
                (effect["shape"], effect["object_context"], effect["vector"])
                for effect in effects
            }
            groups = self.action_effects.rigid_groups(action_id, context)
            grouped_signatures = set()
            effect_groups = []
            for group in groups:
                signatures = tuple(group["signatures"])
                if signatures and all(sig in verified for sig in signatures):
                    effect_groups.append(signatures)
                    grouped_signatures.update(signatures)
            effect_groups.extend((signature,) for signature in sorted(
                verified - grouped_signatures, key=repr
            ))

            for signatures in effect_groups:
                vector = tuple(signatures[0][2])
                if any(tuple(signature[2]) != vector for signature in signatures):
                    continue
                if any(self.action_effects.is_passive_effect(
                    action_id,
                    {"shape": signature[0], "object_context": signature[1],
                     "vector": signature[2]},
                ) for signature in signatures):
                    continue
                object_indices = set()
                effect_shapes = []
                for shape, object_context, _vector in signatures:
                    matching_blobs = [
                        (index, blob) for index, blob in enumerate(blobs)
                        if _shape_key(blob) == shape
                        and self.action_effects.object_context(state, index)
                        == object_context
                    ]
                    for _blob_index, blob in matching_blobs:
                        blob_cells = set(tuple(map(int, cell)) for cell in blob["coords"])
                        for object_index, obj in enumerate(objects):
                            cells = set(tuple(map(int, cell))
                                        for cell in obj.get("cells", ()))
                            if blob_cells and blob_cells.issubset(cells):
                                object_indices.add(object_index)
                    effect_shapes.append(shape)
                if not object_indices:
                    continue
                blockers = set()
                for shape, _object_context, move_vector in signatures:
                    blockers.update(self.action_effects.blocked_colors(
                        action_id, context, shape, move_vector
                    ))
                control = {
                    "action_id": action_id,
                    "objects": tuple(sorted(object_indices)),
                    "vector": vector,
                    "blocked_colors": tuple(sorted(blockers)),
                    "effects": signatures,
                }
                if control not in controls:
                    controls.append(control)

        if not controls:
            return None

        def move(sim_state, control):
            sim_grid = np.asarray(grid_of(sim_state)).copy()
            sim_objects = deepcopy(sim_state.get("objects", []))
            dr, dc = control["vector"]
            indices = control["objects"]
            moving_cells = set()
            old_values = {}
            for index in indices:
                obj = sim_objects[index]
                for row, col in obj.get("cells", ()):
                    cell = (int(row), int(col))
                    moving_cells.add(cell)
                    old_values[cell] = int(sim_grid[cell])
            destinations = {
                (row + dr, col + dc) for row, col in moving_cells
            }
            height, width = sim_grid.shape
            if any(row < 0 or col < 0 or row >= height or col >= width
                   for row, col in destinations):
                return None
            if any(
                cell not in moving_cells
                and int(sim_grid[cell]) in control["blocked_colors"]
                for cell in destinations
            ):
                return None

            for row, col in moving_cells:
                if not any(
                    (row, col) in other.get("cells", ())
                    for idx, other in enumerate(sim_objects) if idx not in indices
                ):
                    sim_grid[row, col] = background
            for index in indices:
                obj = sim_objects[index]
                old_cells = set(tuple(map(int, cell)) for cell in obj.get("cells", ()))
                new_cells = frozenset((row + dr, col + dc) for row, col in old_cells)
                obj["cells"] = new_cells
                row0, col0, row1, col1 = obj["bbox"]
                obj["bbox"] = (row0 + dr, col0 + dc, row1 + dr, col1 + dc)
                center_row, center_col = obj["center"]
                obj["center"] = (center_row + dr, center_col + dc)
                for cell in new_cells:
                    source = (cell[0] - dr, cell[1] - dc)
                    sim_grid[cell] = old_values.get(source, background)

            return {
                "grid": sim_grid,
                "masked_grid": sim_grid,
                "objects": sim_objects,
                "relations": self._relations_for_objects(sim_objects),
            }

        start = {
            "grid": grid.copy(),
            "masked_grid": grid.copy(),
            "objects": deepcopy(objects),
            "relations": self._relations_for_objects(objects),
        }
        queue = deque([(start, [])])
        seen = {np.asarray(grid).tobytes() + repr([
            tuple(sorted(tuple(map(int, cell)) for cell in obj.get("cells", ())))
            for obj in objects
        ]).encode("utf-8")}
        expanded = 0
        while queue and expanded < 2500:
            current, path = queue.popleft()
            if len(path) >= max_depth:
                continue
            expanded += 1
            for control in controls:
                following = move(current, control)
                if following is None:
                    continue
                action = (control["action_id"], None, None)
                next_path = path + [action]
                if goal.holds(following):
                    return {
                        "goal": goal_name,
                        "weight": weight,
                        "actions": next_path,
                        "expected_effects": control["effects"],
                        "expanded_nodes": expanded,
                    }
                key = np.asarray(following["grid"]).tobytes() + repr([
                    tuple(sorted(tuple(map(int, cell))
                                 for cell in obj.get("cells", ())))
                    for obj in following["objects"]
                ]).encode("utf-8")
                if key not in seen:
                    seen.add(key)
                    queue.append((following, next_path))
        return None

    # ------------------------------------------------------------------
    # Observation and learning
    # ------------------------------------------------------------------

    def observe(self, state_before, sig, state_after, confirmed_progress: bool) -> Dict:
        """Compare the prediction with what happened, then learn from the error."""
        self.steps += 1

        if self._pending is not None and self._pending["sig"] == sig:
            predictions = self._pending["predictions"]
            effect_context = self._pending.get("effect_context")
            components = self._pending["components"]
            controller = self._pending.get("controller", "unknown")
            local_llm_advice = self._pending.get("local_llm_advice")
            local_llm_advice_applied = self._pending.get(
                "local_llm_advice_applied", False
            )
            local_llm_prediction = self._pending.get("local_llm_prediction")
        else:
            effect_context = self.action_effects.context(
                self.interactions.selected_class,
                self.interactions.selected_center,
                state_before,
                sig=sig,
            )
            self.world.ensure_hypotheses(
                [sig], state_before, context=effect_context
            )
            predictions = self.world.predictions(
                sig, state_before, context=effect_context
            )
            components = {}
            controller = "unknown"
            local_llm_advice = None
            local_llm_advice_applied = False
            local_llm_prediction = None

        delta = diff_states(state_before, state_after)
        errors = self.world.record(
            sig, delta, predictions, context=effect_context
        )
        local_llm_prediction_correct = LocalLLMAdvisor.prediction_correct(
            local_llm_prediction, delta, confirmed_progress
        )
        if local_llm_advice is not None and local_llm_advice_applied:
            self._recent_advisor_history.append({
                "action": list(sig),
                "changed_cells": delta.changed_cells,
                "object_shifts": delta.shifts,
                "confirmed_progress": bool(confirmed_progress),
                "prediction_correct": local_llm_prediction_correct,
            })
        action_effect_report = self.action_effects.observe(
            sig,
            state_before,
            state_after,
            selected_class=self.interactions.selected_class,
            selected_center=self.interactions.selected_center,
            learn_motion=True,
        )
        learned_action_effects = self.action_effects.verified_effects(
            int(sig[0]), effect_context
        )
        shaping_reward = 0.0
        if not confirmed_progress:
            shaping_reward = self.goals.shaping_delta(
                state_before, state_after
            )
        relevant_class = self.interactions.relevant_object_class(
            sig, state_before
        )
        interaction = self.interactions.observe(
            sig,
            state_before,
            state_after,
            delta,
            progressed=confirmed_progress,
            shaping_reward=shaping_reward,
        )
        # A completed-level frame may already show the next level. Attribute
        # success against the state that led to the action, then keep the
        # newly reached state available for future candidates.
        goal_state = state_before if confirmed_progress else state_after
        self.goals.ensure_goals(goal_state)
        self.goals.observe(
            goal_state,
            confirmed_progress,
            relevant_class=relevant_class,
        )

        if confirmed_progress:
            self.goals.reset_level_potentials()
            self.action_effects.reset_context()
            self.steps_since_progress = 0
            self.engaged_remaining = 0
        else:
            self.steps_since_progress += 1
        if self.engaged_remaining > 0:
            self.engaged_remaining -= 1

        self._log({
            "step": self.steps,
            "sig": list(sig),
            "controller": controller,
            "components": components,
            "delta": delta.to_dict(),
            "action_effect_context": repr(effect_context),
            "action_effect_delta": action_effect_report["delta"].to_dict(),
            "action_effect_validation": {
                "predictions": action_effect_report["predictions"],
                "frame_change_prediction": action_effect_report[
                    "frame_change_prediction"
                ],
                "counterexamples": action_effect_report["counterexamples"],
                "verified_effects": learned_action_effects,
            },
            "learned_palette_free_action_effects": learned_action_effects,
            "local_llm_advice": local_llm_advice,
            "local_llm_advice_applied": local_llm_advice_applied,
            "local_llm_prediction_correct": local_llm_prediction_correct,
            "predicted": {str(k): v for k, v in predictions.items()},
            "errors": errors,
            "interaction_learning": interaction,
            "provisional_goal_shaping": round(shaping_reward, 5),
            "confirmed_progress": confirmed_progress,
            "top_goals": self.goals.top(),
            "goal_hypotheses": self.goals.candidate_details(goal_state),
        })
        self._pending = None
        return {"delta": delta, "errors": errors}

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        return {
            "steps": self.steps,
            "steps_since_progress": self.steps_since_progress,
            "world": self.world.summary(),
            "action_effects": self.action_effects.summary(),
            "local_llm": {
                "enabled": self.local_llm.enabled,
                "model": self.local_llm.model,
                "calls": self.local_llm.calls,
                "last_error": self.local_llm.last_error,
            },
            "top_goals": self.goals.top(),
            "interactions": self.interactions.summary(),
        }

    def _log(self, record: Dict) -> None:
        if not self.log_path:
            return
        record.setdefault("game_id", self.game_id)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
