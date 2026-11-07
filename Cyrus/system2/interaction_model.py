"""Online, object-centric learning from action and frame outcomes.

The model is intentionally game-agnostic. It learns which visual object
classes an action targeted, what changed afterward, and whether that
experience coincided with environment-confirmed progress. Object classes use
shape and color-frequency profiles rather than game IDs or hard-coded names.
"""

from collections import Counter, deque
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Dict, Optional, Tuple


class InteractionModel:
    """Remember action affordances and progress by visual object class."""

    def __init__(self, memory_path: Optional[str] = None) -> None:
        self.outcomes: Dict[Tuple, Counter] = {}
        self.total_trials = 0
        self.total_progress = 0
        self.selected_class: Optional[Tuple] = None
        self.selected_center: Optional[Tuple[float, float]] = None
        # Short eligibility trace credits useful steps that led to a delayed
        # level/score signal, not only the final click before it appeared.
        self.recent_contexts = deque(maxlen=6)
        configured_path = (
            memory_path
            or os.getenv("CYRUS_LEARNING_MEMORY")
            or str(Path(__file__).with_name("interaction_memory.json"))
        )
        self.memory_path = Path(configured_path)
        self._load_memory()

    @staticmethod
    def _pack(value):
        if isinstance(value, tuple):
            return {"tuple": [InteractionModel._pack(item) for item in value]}
        if isinstance(value, list):
            return [InteractionModel._pack(item) for item in value]
        return value

    @staticmethod
    def _unpack(value):
        if isinstance(value, dict) and set(value) == {"tuple"}:
            return tuple(InteractionModel._unpack(item) for item in value["tuple"])
        if isinstance(value, list):
            return [InteractionModel._unpack(item) for item in value]
        return value

    def _load_memory(self) -> None:
        """Restore learned visual-action statistics from a previous run."""
        try:
            payload = json.loads(self.memory_path.read_text(encoding="utf-8"))
            for row in payload.get("outcomes", []):
                context = self._migrate_context(self._unpack(row["context"]))
                self.outcomes.setdefault(context, Counter()).update(
                    row.get("stats", {})
                )
            self.total_trials = int(payload.get("total_trials", 0))
            self.total_progress = float(payload.get("total_progress", 0))
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f"[CYRUS] Could not load learning memory: {exc}")

    def save_memory(self) -> None:
        """Atomically persist learned outcome statistics when configured."""
        if self.memory_path is None:
            return
        payload = {
            "version": 2,
            "total_trials": self.total_trials,
            "total_progress": self.total_progress,
            "outcomes": [
                {
                    "context": self._pack(context),
                    "stats": dict(stats),
                }
                for context, stats in self.outcomes.items()
            ],
        }
        try:
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.memory_path.parent,
                delete=False,
            ) as handle:
                json.dump(payload, handle, ensure_ascii=False)
                temp_path = Path(handle.name)
            os.replace(temp_path, self.memory_path)
        except OSError as exc:
            print(f"[CYRUS] Could not save learning memory: {exc}")

    def begin_episode(self) -> None:
        """Clear short-lived action state while retaining learned outcomes."""
        self.selected_class = None
        self.selected_center = None
        self.recent_contexts.clear()

    @staticmethod
    def _class_from_features(width, height, area, color_count) -> Tuple:
        """Compress an object to palette-free shape and occupancy features."""
        width, height = max(1, int(width)), max(1, int(height))
        occupancy = min(1.0, max(0.0, float(area) / (width * height)))
        return (
            min(width, height),
            max(width, height),
            min(4, int(round(occupancy * 4))),
            min(3, max(1, int(color_count))),
        )

    @classmethod
    def object_class(cls, obj: Dict) -> Tuple:
        """Return a palette-invariant class based on geometry and density."""
        return cls._class_from_features(
            obj.get('width', 0), obj.get('height', 0), obj.get('area', 0),
            obj.get('color_count', 1),
        )

    @staticmethod
    def is_scene_like(obj: Dict, state: Dict) -> bool:
        """Reject a frame-spanning connected region as a clickable object.

        FrameProcessor can merge maze walls and other static board structure
        into a full-canvas multi-color region. Treating that region as a
        selected piece corrupts both click and relation contexts.
        """
        raw_grid = state.get("masked_grid", state.get("grid"))
        shape = getattr(raw_grid, "shape", ())
        if len(shape) < 2:
            return False
        height, width = int(shape[-2]), int(shape[-1])
        bbox = obj.get("bbox")
        if not bbox or len(bbox) != 4:
            return False
        row0, col0, row1, col1 = (int(value) for value in bbox)
        object_height = row1 - row0 + 1
        object_width = col1 - col0 + 1
        covers_frame = (
            object_height >= 0.85 * height
            and object_width >= 0.85 * width
        )
        dense_enough = int(obj.get("area", 0)) >= 0.25 * height * width
        multicolor = int(obj.get("color_count", 1)) >= 2
        return covers_frame and dense_enough and multicolor

    @classmethod
    def _migrate_context(cls, context):
        """Upgrade saved contexts that stored palette-specific profiles."""
        if not isinstance(context, tuple) or len(context) not in {3, 5}:
            return context
        action, kind, visual_class = context[:3]
        if (kind not in {'object', 'empty_after_select'}
                or not isinstance(visual_class, tuple)
                or len(visual_class) != 5):
            if len(context) == 3 and kind == 'empty_after_select':
                return (*context, None, None)
            return context
        width, height, area, color_count, _profile = visual_class
        migrated_class = cls._class_from_features(
            width, height, area, color_count
        )
        if kind == 'empty_after_select':
            direction, distance = context[3:] if len(context) == 5 else (None, None)
            return (action, kind, migrated_class, direction, distance)
        return (action, kind, migrated_class)

    @classmethod
    def context(cls, sig, state) -> Tuple:
        """Describe whether an action targets an object or empty space."""
        action_id = int(sig[0])
        if action_id != 6 or len(sig) < 3 or sig[1] is None or sig[2] is None:
            return (action_id, 'no_coordinate_target', None)

        x, y = int(sig[1]), int(sig[2])
        hits = [
            obj for obj in state.get('objects', [])
            if (y, x) in obj.get('cells', ())
            and not cls.is_scene_like(obj, state)
        ]
        if not hits:
            return (action_id, 'empty_space', None)

        target = min(hits, key=lambda obj: obj.get('area', 0))
        return (action_id, 'object', cls.object_class(target))

    def _context(self, sig, state) -> Tuple:
        context = self.context(sig, state)
        if context[1] == 'empty_space' and self.selected_class is not None:
            x, y = int(sig[1]), int(sig[2])
            if self.selected_center is None:
                return (context[0], 'empty_after_select', self.selected_class,
                        None, None)
            dr = y - self.selected_center[0]
            dc = x - self.selected_center[1]
            direction = (
                0 if dr == 0 else (1 if dr > 0 else -1),
                0 if dc == 0 else (1 if dc > 0 else -1),
            )
            distance_bin = min(4, int(math.log2(max(1.0, math.hypot(dr, dc)))))
            return (context[0], 'empty_after_select', self.selected_class,
                    direction, distance_bin)
        return context

    def relevant_object_class(self, sig, state):
        """Return the clicked/selected visual class before outcome updates."""
        context = self._context(sig, state)
        if context[1] in {'object', 'empty_after_select'}:
            return context[2]
        return None

    @staticmethod
    def _centers_by_class(state) -> Dict[Tuple, list]:
        grouped = {}
        for obj in state.get('objects', []):
            if InteractionModel.is_scene_like(obj, state):
                continue
            key = InteractionModel.object_class(obj)
            center = obj.get('center', (0.0, 0.0))
            grouped.setdefault(key, []).append((float(center[0]), float(center[1])))
        return grouped

    @staticmethod
    def _relation_counts(state) -> Counter:
        objects = state.get('objects', [])
        counts = Counter()
        for relation in state.get('relations', []):
            left_idx = relation.get('left')
            right_idx = relation.get('right')
            if not (0 <= left_idx < len(objects) and 0 <= right_idx < len(objects)):
                continue
            if (InteractionModel.is_scene_like(objects[left_idx], state)
                    or InteractionModel.is_scene_like(objects[right_idx], state)):
                continue
            left = InteractionModel.object_class(objects[left_idx])
            right = InteractionModel.object_class(objects[right_idx])
            if repr(right) < repr(left):
                left, right = right, left
            counts[(relation.get('kind'), left, right)] += 1
        return counts

    @staticmethod
    def _matched_movement(before, after) -> bool:
        """Whether same-class objects shifted between the two observations."""
        for key, old_centers in before.items():
            new_centers = list(after.get(key, ()))
            for old in old_centers:
                if not new_centers:
                    break
                idx = min(
                    range(len(new_centers)),
                    key=lambda i: math.dist(old, new_centers[i]),
                )
                new = new_centers.pop(idx)
                if math.dist(old, new) >= 1.5:
                    return True
        return False

    def observe(self, sig, state_before, state_after, delta, progressed: bool,
                shaping_reward: float = 0.0) -> Dict:
        """Learn the observed effect of an action in its visual context."""
        context = self._context(sig, state_before)
        stats = self.outcomes.setdefault(context, Counter())
        before = self._centers_by_class(state_before)
        after = self._centers_by_class(state_after)
        relations_before = self._relation_counts(state_before)
        relations_after = self._relation_counts(state_after)

        target_class = context[2] if context[1] in {
            'object', 'empty_after_select'
        } else None
        target_moved = (
            target_class is not None
            and self._matched_movement(
                {target_class: before.get(target_class, [])},
                {target_class: after.get(target_class, [])},
            )
        )
        any_moved = self._matched_movement(before, after)
        count_changed = len(state_before.get('objects', [])) != len(
            state_after.get('objects', [])
        )
        effects = []
        if target_moved:
            effects.append('target_object_moved')
        elif any_moved:
            effects.append('some_object_moved')
        if count_changed:
            effects.append('object_count_changed')
        added_relations = list((relations_after - relations_before).elements())
        removed_relations = list((relations_before - relations_after).elements())
        if added_relations:
            effects.append('relation_added')
        if removed_relations:
            effects.append('relation_removed')
        if not effects and getattr(delta, 'changed', False):
            effects.append('visual_change_without_object_motion')
        if not effects:
            effects.append('no_detected_object_effect')

        stats['trials'] += 1
        stats['progress'] += int(bool(progressed))
        stats['goal_shaping'] += float(shaping_reward)
        for effect in effects:
            stats[f'effect:{effect}'] += 1
        self.total_trials += 1
        self.total_progress += int(bool(progressed))

        if progressed:
            for distance, prior_context in enumerate(
                reversed(self.recent_contexts), start=1
            ):
                prior = self.outcomes.get(prior_context)
                if prior is not None:
                    prior['progress'] += 0.5 ** distance
            self.recent_contexts.clear()
            self.selected_class = None
            self.selected_center = None
        else:
            self.recent_contexts.append(context)
            if context[1] == 'object':
                # Clicking an object is a candidate selection action.
                # Subsequent empty-space actions are evaluated against it.
                self.selected_class = context[2]
                x, y = int(sig[1]), int(sig[2])
                selected = [
                    obj for obj in state_before.get('objects', [])
                    if (y, x) in obj.get('cells', ())
                    and self.object_class(obj) == self.selected_class
                ]
                if selected:
                    self.selected_center = tuple(
                        float(value) for value in min(
                            selected,
                            key=lambda obj: math.dist(
                                obj.get('center', (0, 0)), (y, x)
                            ),
                        ).get('center', (y, x))
                    )
            elif context[1] == 'empty_after_select' and self.selected_center is not None:
                candidates = [
                    obj.get('center', (0, 0))
                    for obj in state_after.get('objects', [])
                    if self.object_class(obj) == self.selected_class
                ]
                if candidates:
                    self.selected_center = tuple(float(value) for value in min(
                        candidates,
                        key=lambda center: math.dist(center, self.selected_center),
                    ))

        # Keep cross-run learning current without writing on every action.
        if progressed or self.total_trials % 25 == 0:
            self.save_memory()

        return {
            'context': repr(context),
            'effects': effects,
            'relations_added': [repr(item) for item in added_relations[:5]],
            'relations_removed': [repr(item) for item in removed_relations[:5]],
            'trials': int(stats['trials']),
            'progresses': round(float(stats['progress']), 3),
            'progress_rate': round(
                (stats['progress'] + 1) / (stats['trials'] + 2), 4
            ),
            'goal_shaping': round(float(stats['goal_shaping']), 4),
        }

    def value(self, sig, state) -> float:
        """Return a small explore/exploit bonus for System 2 action scoring."""
        context = self._context(sig, state)
        stats = self.outcomes.get(context, Counter())
        trials = stats['trials']
        rate = (stats['progress'] + 1) / (trials + 2)
        baseline = (self.total_progress + 1) / (self.total_trials + 2)
        novelty = 1.0 / math.sqrt(trials + 1)
        progress_lift = max(0.0, rate - baseline)
        shaping = float(stats.get('goal_shaping', 0.0)) / max(1, trials)
        return 0.25 * novelty + 2.0 * progress_lift + 5.0 * shaping

    def summary(self) -> Dict:
        """Small JSON-safe diagnostic summary."""
        return {
            'trials': self.total_trials,
            'confirmed_progress': self.total_progress,
            'contexts': len(self.outcomes),
            'learned': [
                {
                    'context': repr(context),
                    'trials': int(stats['trials']),
                    'progress': round(float(stats['progress']), 3),
                    'effects': {
                        key.removeprefix('effect:'): int(value)
                        for key, value in stats.items()
                        if key.startswith('effect:')
                    },
                }
                for context, stats in self.outcomes.items()
            ],
        }
