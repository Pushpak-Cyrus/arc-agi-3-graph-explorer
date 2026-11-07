"""Palette-independent action-effect learning from visual transitions.

The learner stores trial-level effects by action, observable context, and
silhouette. It does not use game IDs, object names, or task-specific routes.
A displacement is considered repeatable only after it recurs across trials
in the same context and survives prequential prediction checks.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Tuple

import numpy as np

from .frame_diff import FrameDelta, diff_grids, grid_of
from .interaction_model import InteractionModel


ShapeKey = Tuple[int, int, Tuple[Tuple[int, int], ...]]
ObjectContext = Tuple
EffectKey = Tuple[ShapeKey, ObjectContext, Tuple[int, int]]


def _shape_key(blob: Dict) -> ShapeKey | None:
    cells = blob.get("coords", ())
    if not cells:
        return None
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    normalized = tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))
    height = max(r for r, _ in cells) - r0 + 1
    width = max(c for _, c in cells) - c0 + 1
    return int(height), int(width), normalized


def _center(blob: Dict) -> Tuple[float, float]:
    cells = blob.get("coords", ())
    return (
        sum(r for r, _ in cells) / len(cells),
        sum(c for _, c in cells) / len(cells),
    )


def _non_background_blobs(state: Dict) -> list[Dict]:
    """Return raw blobs except those assigned to the detected background."""
    background_indices = {
        int(index)
        for target in state.get("targets", ())
        if target.get("is_background_candidate")
        for index in target.get("member_blob_indices", ())
    }
    return [blob for index, blob in enumerate(state.get("blobs", ()))
            if index not in background_indices and blob.get("coords")]


def _shape_shifts(before: Dict, after: Dict, max_distance: float = 10.0):
    """Match same silhouettes across frames, independent of palette values."""
    previous = []
    for blob in _non_background_blobs(before):
        key = _shape_key(blob)
        if key is not None:
            previous.append((key, _center(blob)))
    used = set()
    shifts = []
    for blob in _non_background_blobs(after):
        key = _shape_key(blob)
        if key is None:
            continue
        center = _center(blob)
        choices = [
            (idx, old_center)
            for idx, (old_key, old_center) in enumerate(previous)
            if idx not in used and old_key == key
            and np.hypot(center[0] - old_center[0], center[1] - old_center[1]) <= max_distance
        ]
        if not choices:
            continue
        idx, old_center = min(
            choices,
            key=lambda item: np.hypot(center[0] - item[1][0], center[1] - item[1][1]),
        )
        used.add(idx)
        dr = int(round(center[0] - old_center[0]))
        dc = int(round(center[1] - old_center[1]))
        if (dr, dc) != (0, 0):
            shifts.append((key, (dr, dc)))
    return shifts


def _object_context(state: Dict, blobs: list[Dict], target_index: int) -> ObjectContext:
    """Describe a blob's boundary and nearby-object context before acting."""
    blob = blobs[target_index]
    coords = blob.get("coords", ())
    grid = grid_of(state)
    if not coords or grid.ndim != 2:
        return ()
    row0 = min(row for row, _ in coords)
    col0 = min(col for _, col in coords)
    row1 = max(row for row, _ in coords)
    col1 = max(col for _, col in coords)
    height, width = grid.shape

    def distance_bin(distance):
        return 0 if distance <= 0 else (1 if distance <= 2 else (2 if distance <= 5 else 3))

    edge_context = (
        distance_bin(row0),
        distance_bin(height - 1 - row1),
        distance_bin(col0),
        distance_bin(width - 1 - col1),
    )
    center = _center(blob)
    neighbors = []
    for index, other in enumerate(blobs):
        if index == target_index:
            continue
        other_center = _center(other)
        dr = other_center[0] - center[0]
        dc = other_center[1] - center[1]
        distance = float(np.hypot(dr, dc))
        if distance <= 10:
            direction = (
                0 if dr == 0 else (1 if dr > 0 else -1),
                0 if dc == 0 else (1 if dc > 0 else -1),
            )
            neighbors.append((distance, direction))
    nearby_bin = min(3, len(neighbors))
    if neighbors:
        distance, direction = min(neighbors, key=lambda item: item[0])
        nearest_bin = 0 if distance <= 2 else (1 if distance <= 5 else 2)
    else:
        nearest_bin, direction = 3, (0, 0)
    return edge_context, nearby_bin, nearest_bin, direction


def _contextual_shape_shifts(before: Dict, after: Dict,
                             max_distance: float = 10.0):
    """Match silhouettes and retain each mover's pre-action local context."""
    previous = _non_background_blobs(before)
    current = _non_background_blobs(after)
    previous_keys = [_shape_key(blob) for blob in previous]
    used = set()
    shifts = []
    for blob in current:
        key = _shape_key(blob)
        if key is None:
            continue
        center = _center(blob)
        choices = [
            index for index, old_blob in enumerate(previous)
            if index not in used and previous_keys[index] == key
            and np.hypot(
                center[0] - _center(old_blob)[0],
                center[1] - _center(old_blob)[1],
            ) <= max_distance
        ]
        if not choices:
            continue
        index = min(
            choices,
            key=lambda item: np.hypot(
                center[0] - _center(previous[item])[0],
                center[1] - _center(previous[item])[1],
            ),
        )
        used.add(index)
        old_center = _center(previous[index])
        vector = (
            int(round(center[0] - old_center[0])),
            int(round(center[1] - old_center[1])),
        )
        if vector != (0, 0):
            shifts.append((
                key,
                _object_context(before, previous, index),
                vector,
            ))
    return shifts


class ActionEffectLearner:
    """Learn action effects conditioned on recent, observable interaction context.

    The context deliberately avoids game IDs. It includes the immediately
    preceding action, whether that action changed the frame, and the currently
    selected visual object class supplied by InteractionModel. Predictions
    are evaluated before updating the model, so the accuracy is prequential.
    """

    def __init__(self, min_trials: int = 2, min_share: float = 0.6) -> None:
        self.min_trials = max(1, int(min_trials))
        self.min_share = float(min_share)
        self.trials = Counter()
        self.changed_trials = Counter()
        self.effect_trials = defaultdict(Counter)
        self.prediction_trials = Counter()
        self.prediction_correct = Counter()
        self.invalidated_effects = set()
        self.effect_correct_streak = Counter()
        self.change_prediction_trials = Counter()
        self.change_prediction_correct = Counter()
        self.rigid_group_trials = Counter()
        self.effect_action_trials = defaultdict(Counter)
        self.blocked_color_trials = Counter()
        self.last_action_id = None
        self.last_action_changed = False

    @staticmethod
    def transition(before: Dict, after: Dict) -> FrameDelta:
        delta = diff_grids(before, after)
        delta.shape_shifts = _shape_shifts(before, after)
        delta.object_context_shifts = _contextual_shape_shifts(before, after)
        return delta

    @staticmethod
    def object_context(state: Dict, blob_index: int):
        """Return local context for a current non-background blob index."""
        blobs = _non_background_blobs(state)
        if not 0 <= int(blob_index) < len(blobs):
            return ()
        return _object_context(state, blobs, int(blob_index))

    @staticmethod
    def _local_context(state: Dict, selected_class, selected_center):
        """Coarse geometry around the selected object, without palette IDs."""
        if selected_class is None or selected_center is None:
            return None
        objects = state.get("objects", [])
        selected = [
            (index, obj) for index, obj in enumerate(objects)
            if (
                not InteractionModel.is_scene_like(obj, state)
                and (min(int(obj.get("width", 1)), int(obj.get("height", 1))),
                max(int(obj.get("width", 1)), int(obj.get("height", 1))),
                min(4, int(round(
                    float(obj.get("area", 0))
                    / max(1, int(obj.get("width", 1))
                          * int(obj.get("height", 1))) * 4
                ))),
                min(3, max(1, int(obj.get("color_count", 1))))) == selected_class
            )
        ]
        if not selected:
            return None
        selected_index, _ = min(
            selected,
            key=lambda pair: np.hypot(
                float(pair[1].get("center", (0, 0))[0]) - selected_center[0],
                float(pair[1].get("center", (0, 0))[1]) - selected_center[1],
            ),
        )
        grid = grid_of(state)
        row, col = map(float, selected_center)
        edge_distance = min(row, col, grid.shape[0] - 1 - row,
                            grid.shape[1] - 1 - col)
        edge_bin = 0 if edge_distance <= 1 else (1 if edge_distance <= 4 else 2)

        others = []
        for index, obj in enumerate(objects):
            if index == selected_index:
                continue
            if InteractionModel.is_scene_like(obj, state):
                continue
            center = obj.get("center", (0, 0))
            dr = float(center[0]) - row
            dc = float(center[1]) - col
            distance = float(np.hypot(dr, dc))
            if distance <= 6:
                others.append((distance, dr, dc))
        nearby_bin = min(3, len(others))
        if others:
            distance, dr, dc = min(others)
            nearest_bin = 0 if distance <= 2 else (1 if distance <= 4 else 2)
            direction = (
                0 if dr == 0 else (1 if dr > 0 else -1),
                0 if dc == 0 else (1 if dc > 0 else -1),
            )
        else:
            nearest_bin, direction = 3, (0, 0)
        return edge_bin, nearby_bin, nearest_bin, direction

    @staticmethod
    def _clicked_object(state: Dict, sig):
        """Get palette-free class and center for an ACTION6 object target."""
        if (not sig or int(sig[0]) != 6 or len(sig) < 3
                or sig[1] is None or sig[2] is None):
            return None, None
        x, y = int(sig[1]), int(sig[2])
        hits = [
            obj for obj in state.get("objects", ())
            if (y, x) in obj.get("cells", ())
            and not InteractionModel.is_scene_like(obj, state)
        ]
        if not hits:
            return None, None
        obj = min(hits, key=lambda item: item.get("area", 0))
        width = max(1, int(obj.get("width", 1)))
        height = max(1, int(obj.get("height", 1)))
        area = float(obj.get("area", 0))
        color_count = max(1, int(obj.get("color_count", 1)))
        object_class = (
            min(width, height),
            max(width, height),
            min(4, int(round(area / max(1, width * height) * 4))),
            min(3, color_count),
        )
        center = tuple(float(value) for value in obj.get("center", (y, x)))
        return object_class, center

    def context(self, selected_class=None, selected_center=None,
                state: Dict | None = None, sig=None) -> Tuple:
        """Return prior-action, selected-object, click-target, and scene context."""
        state = state if isinstance(state, dict) else {}
        local = self._local_context(state, selected_class, selected_center)
        target_class, target_center = self._clicked_object(state, sig)
        target_local = self._local_context(state, target_class, target_center)
        return (self.last_action_id, self.last_action_changed,
                selected_class, local, target_class, target_local)

    def reset_context(self) -> None:
        """Break short action traces at an environment-confirmed level edge."""
        self.last_action_id = None
        self.last_action_changed = False

    def observe(self, sig, before: Dict, after: Dict,
                selected_class=None, selected_center=None,
                learn_motion: bool = True) -> Dict:
        action_id = int(sig[0])
        context = self.context(selected_class, selected_center, before, sig=sig)
        key = (action_id, context)
        delta = self.transition(before, after)
        predictions = self.predict(action_id, context)
        prediction_rows = []
        change_prediction = self.predict_frame_change(action_id, context)
        change_prediction_row = None
        prior_trials = self.trials[key]
        predicted_probability = (
            (self.changed_trials[key] + 1) / (prior_trials + 2)
            if prior_trials >= 2 else None
        )
        if predicted_probability is not None:
            prediction_key = key
            correct = (
                change_prediction == delta.changed
                if change_prediction is not None else None
            )
            if correct is not None:
                self.change_prediction_trials[prediction_key] += 1
                self.change_prediction_correct[prediction_key] += int(correct)
            change_prediction_row = {
                "predicted_changed": change_prediction,
                "predicted_probability": round(float(predicted_probability), 6),
                "observed_changed": bool(delta.changed),
                "correct": bool(correct) if correct is not None else None,
                "prediction_accuracy": round(
                    self.change_prediction_correct[prediction_key]
                    / self.change_prediction_trials[prediction_key], 3
                ) if self.change_prediction_trials[prediction_key] else None,
            }
        for effect in predictions:
            signature = (
                effect["shape"], effect["object_context"], effect["vector"]
            )
            pred_key = (*key, signature)
            correct = signature in set(delta.object_context_shifts)
            self.prediction_trials[pred_key] += 1
            self.prediction_correct[pred_key] += int(correct)
            if correct and pred_key in self.invalidated_effects:
                self.effect_correct_streak[pred_key] += 1
                if self.effect_correct_streak[pred_key] >= self.min_trials:
                    self.invalidated_effects.discard(pred_key)
                    self.effect_correct_streak.pop(pred_key, None)
            elif not correct:
                self.invalidated_effects.add(pred_key)
                self.effect_correct_streak[pred_key] = 0
            prediction_rows.append({
                **effect,
                "observed": correct,
                "prediction_accuracy": round(
                    self.prediction_correct[pred_key]
                    / self.prediction_trials[pred_key], 3
                ),
            })

        observed_effects = set(delta.object_context_shifts)
        if predictions:
            self._record_blocked_colors(
                action_id, context, predictions, before, observed_effects
            )

        self.trials[key] += 1
        if delta.changed:
            self.changed_trials[key] += 1
        if learn_motion:
            effects_by_vector = defaultdict(set)
            for signature in observed_effects:
                self.effect_trials[key][signature] += 1
                self.effect_action_trials[signature][action_id] += 1
                effects_by_vector[signature[2]].add(signature)
            for vector, signatures in effects_by_vector.items():
                if len(signatures) >= 2:
                    group = tuple(sorted(signatures, key=repr))
                    self.rigid_group_trials[(action_id, context, group)] += 1

        # Clicks are not displacement trials, but they update the context for
        # the following action (for example, a move after selecting an object).
        self.last_action_id = action_id
        self.last_action_changed = bool(delta.changed)
        return {
            "context": repr(context),
            "delta": delta,
            "predictions": prediction_rows,
            "frame_change_prediction": change_prediction_row,
            "counterexamples": (
                [row for row in prediction_rows if not row["observed"]]
                + ([change_prediction_row]
                   if change_prediction_row and not change_prediction_row["correct"]
                   else [])
            ),
            "rigid_groups": self.rigid_groups(action_id, context),
            "blocked_colors": self.blocked_colors(action_id, context),
        }

    def _record_blocked_colors(
        self, action_id, context, predictions, before, observed_effects
    ):
        """Record destination colors when a predicted move is blocked."""
        grid = np.asarray(grid_of(before))
        if grid.ndim != 2 or not grid.size:
            return
        values, counts = np.unique(grid, return_counts=True)
        background = int(values[int(np.argmax(counts))])
        blobs = _non_background_blobs(before)
        for effect in predictions:
            shape = effect["shape"]
            object_context = effect["object_context"]
            dr, dc = effect["vector"]
            if (shape, object_context, (dr, dc)) in observed_effects:
                continue
            for index, blob in enumerate(blobs):
                if (_shape_key(blob) != shape
                        or _object_context(before, blobs, index) != object_context):
                    continue
                old_cells = set(tuple(map(int, cell)) for cell in blob["coords"])
                new_cells = {(row + dr, col + dc) for row, col in old_cells}
                colors = {
                    int(grid[row, col]) for row, col in new_cells
                    if (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]
                        and (row, col) not in old_cells
                        and int(grid[row, col]) != background)
                }
                for color in colors:
                    self.blocked_color_trials[
                        (int(action_id), context, shape, (dr, dc), color)
                    ] += 1

    def blocked_colors(self, action_id, context=None, shape=None, vector=None):
        """Return observed blocking colors keyed by count for a learned move."""
        result = Counter()
        for (aid, observed_context, observed_shape, observed_vector, color), count in self.blocked_color_trials.items():
            if (aid == int(action_id) and observed_context == context
                    and (shape is None or observed_shape == shape)
                    and (vector is None or observed_vector == tuple(vector))):
                result[color] += count
        return dict(result)

    def is_passive_effect(self, action_id, effect) -> bool:
        """Flag a displacement repeated under several unrelated action IDs."""
        signature = (effect["shape"], effect["object_context"], effect["vector"])
        actions = self.effect_action_trials.get(signature, {})
        return len(actions) >= 3 and int(action_id) in actions

    def rigid_groups(self, action_id, context=None):
        """Return repeated same-vector object groups with verified members."""
        verified = {
            (effect["shape"], effect["object_context"], effect["vector"])
            for effect in self.verified_effects(action_id, context)
        }
        groups = []
        for (aid, observed_context, signatures), count in self.rigid_group_trials.items():
            if (aid == int(action_id) and observed_context == context
                    and count >= self.min_trials
                    and all(signature in verified for signature in signatures)):
                groups.append({"signatures": signatures, "trials": count})
        return groups

    def predict_frame_change(self, action_id: int, context=None):
        """Predict stable frame-change behavior after two contextual trials."""
        key = (int(action_id), context)
        trials = self.trials[key]
        if trials < 2:
            return None
        rate = self.changed_trials[key] / trials
        if rate >= 0.8:
            return True
        if rate <= 0.2:
            return False
        return None

    def effects(self, action_id: int, context=None) -> list[dict]:
        action_id = int(action_id)
        key = (action_id, context)
        trials = self.trials[key]
        if not trials:
            return []
        result = []
        for (shape, object_context, vector), count in self.effect_trials[key].most_common():
            share = count / trials
            pred_key = (action_id, context, (shape, object_context, vector))
            prediction_count = self.prediction_trials[pred_key]
            accuracy = (
                self.prediction_correct[pred_key] / prediction_count
                if prediction_count else None
            )
            result.append({
                "shape": shape,
                "object_context": object_context,
                "vector": vector,
                "trials_with_effect": count,
                "trials": trials,
                "share": round(share, 3),
                "repeatable": count >= self.min_trials and share >= self.min_share,
                "prequential_predictions": prediction_count,
                "prequential_accuracy": (
                    round(accuracy, 3) if accuracy is not None else None
                ),
                "verified": bool(
                    prediction_count >= 2 and accuracy is not None
                    and accuracy >= 0.75
                    and pred_key not in self.invalidated_effects
                ),
            })
        return result

    def predict(self, action_id: int, context=None) -> list[dict]:
        """Return repeated candidates and mark which have passed prediction checks."""
        return [effect for effect in self.effects(action_id, context)
                if effect["repeatable"]]

    def verified_effects(self, action_id: int, context=None) -> list[dict]:
        """Effects with repeatable support and at least two correct tests."""
        return [effect for effect in self.predict(action_id, context)
                if effect["verified"]]

    def uncertainty(self, action_id: int, context=None) -> float:
        """Estimate how informative another action in this context may be."""
        trials = self.trials[(int(action_id), context)]
        if trials < self.min_trials:
            return 1.0 / (trials + 1) ** 0.5
        changed = self.changed_trials[(int(action_id), context)]
        p = (changed + 1) / (trials + 2)
        bernoulli_uncertainty = 4.0 * p * (1.0 - p)
        effects = self.effect_trials[(int(action_id), context)]
        total_effects = sum(effects.values())
        direction_uncertainty = (
            1.0 - max(effects.values()) / total_effects
            if total_effects else 0.0
        )
        return round(max(0.0, min(1.0,
            0.5 * bernoulli_uncertainty + 0.5 * direction_uncertainty)), 4)

    def summary(self) -> dict:
        return {
            str(action_id): [
                {
                    "context": repr(context),
                    "trials": self.trials[(action_id, context)],
                    "changed_trials": self.changed_trials[(action_id, context)],
                    "changed_rate": round(
                        self.changed_trials[(action_id, context)]
                        / max(1, self.trials[(action_id, context)]), 3
                    ),
                    "effects": self.effects(action_id, context),
                    "uncertainty": self.uncertainty(action_id, context),
                "frame_change_predictions": self.change_prediction_trials[
                    (action_id, context)
                ],
                "frame_change_accuracy": (
                    round(
                        self.change_prediction_correct[(action_id, context)]
                        / self.change_prediction_trials[(action_id, context)], 3
                    )
                    if self.change_prediction_trials[(action_id, context)] else None
                ),
                }
                for aid, context in self.trials if aid == action_id
            ]
            for action_id in sorted({key[0] for key in self.trials})
        }
