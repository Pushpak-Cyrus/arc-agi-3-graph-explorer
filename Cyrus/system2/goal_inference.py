"""Goal inference: competing hypotheses about what counts as success.

Goal predicates are proposed from colors and object relations on screen:

    cleared(c)   no pixel of colour c remains
    contact(a,b) some pixel of colour a is 4-adjacent to some pixel of colour b

A goal is scored by how often it holds when confirmed progress happens,
compared with the base rate of progress. Only environment-confirmed progress
(level advance, score increase) counts as evidence. Visual change does not.

Caveat: progress is rare (one level in 66 steps on r11l, none elsewhere on
Oct 21), so these estimates are very weak early on. Treat goal weights as
priors to be tested, not conclusions.
"""

from dataclasses import dataclass
from collections import deque
from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .frame_diff import grid_of
from .hypotheses import CLICK_ACTION_ID
from .interaction_model import InteractionModel


MIN_GOAL_SUPPORT = 3
GOAL_RATE_PRIOR_STRENGTH = 2.0


@dataclass
class GoalHypothesis:
    name: str
    kind: str  # "cleared", "contact", or a visual object relation
    colours: Tuple[int, ...]
    relation_key: Optional[Tuple] = None
    mover_class: Optional[Tuple] = None
    marker_class: Optional[Tuple] = None
    marker_groups: Tuple = ()
    static_marker_centers: Tuple = ()
    transform: str = "identity"
    true_obs: int = 0       # steps where the predicate held
    true_progress: int = 0  # of those, steps with confirmed progress
    progress_when_false: int = 0  # counterevidence to a necessary-goal claim
    contrastive_credits: int = 0
    retest_failures: int = 0
    pending_retest: bool = False
    initial_count: Optional[int] = None
    initial_distance: Optional[int] = None

    def holds(self, state_or_grid) -> bool:
        state = state_or_grid if isinstance(state_or_grid, dict) else {}
        grid = grid_of(state_or_grid)
        if self.kind in {"cover_markers", "touch_markers"}:
            if not self.marker_groups:
                return False
            height, width = grid.shape
            movers = [
                obj for obj in state.get("objects", [])
                if GoalInference.object_class(obj) == self.mover_class
                and not InteractionModel.is_scene_like(obj, state)
                and tuple(round(float(v), 1) for v in obj.get("center", (0, 0)))
                not in self.static_marker_centers
            ]
            for mover in movers:
                cells = {
                    tuple(map(int, cell))
                    for cell in mover.get("cells", ())
                }
                projected = set()
                for row, col in cells:
                    if self.transform in {"mirror_x", "mirror_xy"}:
                        col = width - 1 - col
                    if self.transform in {"mirror_y", "mirror_xy"}:
                        row = height - 1 - row
                    projected.add((row, col))
                if self.kind == "cover_markers":
                    if all(set(group).issubset(projected)
                           for group in self.marker_groups):
                        return True
                elif all(
                    any(
                        max(abs(marker_row - row), abs(marker_col - col)) <= 1
                        for row, col in projected
                    )
                    for group in self.marker_groups
                    for marker_row, marker_col in group
                ):
                    return True
            return False
        if self.kind == "relation":
            objects = state.get("objects", [])
            for relation in state.get("relations", []):
                left_idx = relation.get("left")
                right_idx = relation.get("right")
                if not (0 <= left_idx < len(objects) and 0 <= right_idx < len(objects)):
                    continue
                if (InteractionModel.is_scene_like(objects[left_idx], state)
                        or InteractionModel.is_scene_like(objects[right_idx], state)):
                    continue
                left = GoalInference.object_class(objects[left_idx])
                right = GoalInference.object_class(objects[right_idx])
                if repr(right) < repr(left):
                    left, right = right, left
                if (relation.get("kind"), left, right) == self.relation_key:
                    return True
            return False

        if self.kind == "cleared":
            return not bool(np.any(grid == self.colours[0]))

        a, b = self.colours
        mask_a = grid == a
        mask_b = grid == b
        if not mask_a.any() or not mask_b.any():
            return False
        near = np.zeros_like(mask_a)
        near[1:, :] |= mask_a[:-1, :]
        near[:-1, :] |= mask_a[1:, :]
        near[:, 1:] |= mask_a[:, :-1]
        near[:, :-1] |= mask_a[:, 1:]
        return bool((near & mask_b).any())

    @property
    def p_progress_given_true(self) -> float:
        """Laplace-smoothed P(progress | predicate holds)."""
        return (self.true_progress + 1) / (self.true_obs + 2)


class GoalInference:
    def __init__(
        self,
        max_colours: int = 6,
        max_goals: int = 512,
        memory_path: Optional[str] = None,
    ) -> None:
        self.max_colours = max_colours
        self.max_goals = max_goals
        self.goals: Dict[str, GoalHypothesis] = {}
        self.total_obs = 0
        self.total_progress = 0
        self.relation_memory = {}
        self._potential_cache = {}
        self._ensured_states = set()
        # Full within-level negative examples allow newly proposed predicates
        # to be evaluated against earlier observations from the same episode.
        self._negative_states = []
        configured_path = (
            memory_path
            or os.getenv("CYRUS_GOAL_MEMORY")
            or str(Path(__file__).with_name("goal_memory.json"))
        )
        self.memory_path = Path(configured_path)
        self._load_memory()

    @staticmethod
    def _freeze(value):
        if isinstance(value, list):
            return tuple(GoalInference._freeze(item) for item in value)
        return value

    def _load_memory(self) -> None:
        try:
            payload = json.loads(self.memory_path.read_text(encoding="utf-8"))
            if payload.get("version") not in {2, 3}:
                return
            self.total_obs = int(payload.get("total_obs", 0))
            self.total_progress = int(payload.get("total_progress", 0))
            for row in payload.get("relations", []):
                key = self._migrate_relation_key(self._freeze(row["key"]))
                self.relation_memory[key] = (
                    int(row.get("true_obs", 0)),
                    int(row.get("true_progress", 0)),
                )
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f"[CYRUS] Could not load goal memory: {exc}")

    def _save_memory(self) -> None:
        payload = {
            "version": 3,
            "total_obs": self.total_obs,
            "total_progress": self.total_progress,
            "relations": [
                {
                    "key": key,
                    "true_obs": counts[0],
                    "true_progress": counts[1],
                }
                for key, counts in self.relation_memory.items()
            ],
        }
        try:
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.memory_path.with_suffix(self.memory_path.suffix + ".tmp")
            temp_path.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(temp_path, self.memory_path)
        except OSError as exc:
            print(f"[CYRUS] Could not save goal memory: {exc}")

    @staticmethod
    def object_class(obj: Dict) -> Tuple:
        """Palette-invariant class matching the interaction learner."""
        return InteractionModel.object_class(obj)

    @classmethod
    def _migrate_relation_key(cls, key):
        """Upgrade persisted relation classes that included palette values."""
        if not isinstance(key, tuple) or len(key) != 3:
            return key
        relation, left, right = key

        def normalize(value):
            if not isinstance(value, tuple) or len(value) != 5:
                return value
            width, height, area, color_count, _profile = value
            return InteractionModel._class_from_features(
                width, height, area, color_count
            )

        return (relation, normalize(left), normalize(right))

    def ensure_goals(self, state_or_grid) -> None:
        """Create color and object-relation goals from the current scene."""
        state = state_or_grid if isinstance(state_or_grid, dict) else {}
        state_key = state.get("hash")
        if state_key is not None and state_key in self._ensured_states:
            return
        grid = grid_of(state_or_grid)
        values, counts = np.unique(grid, return_counts=True)
        if len(values) > 1:
            background = int(values[np.argmax(counts)])
            order = np.argsort(-counts)
            colours = [
                int(values[i]) for i in order if int(values[i]) != background
            ][: self.max_colours]

            for c in colours:
                self._add(GoalHypothesis(f"cleared({c})", "cleared", (c,)))
            for i, a in enumerate(colours):
                for b in colours[i + 1:]:
                    self._add(GoalHypothesis(f"contact({a},{b})", "contact", (a, b)))
                    self._add(GoalHypothesis(f"contact({b},{a})", "contact", (b, a)))

        objects = state.get("objects", [])
        for relation in state.get("relations", []):
            left_idx = relation.get("left")
            right_idx = relation.get("right")
            if not (0 <= left_idx < len(objects) and 0 <= right_idx < len(objects)):
                continue
            if (InteractionModel.is_scene_like(objects[left_idx], state)
                    or InteractionModel.is_scene_like(objects[right_idx], state)):
                continue
            left = self.object_class(objects[left_idx])
            right = self.object_class(objects[right_idx])
            if repr(right) < repr(left):
                left, right = right, left
            key = (relation.get("kind"), left, right)
            self._add(GoalHypothesis(
                name=f"relation{key!r}",
                kind="relation",
                colours=(),
                relation_key=key,
            ))

        if isinstance(state_or_grid, dict):
            self._ensure_marker_goals(state_or_grid)

        self._initialize_missing_baselines(state_or_grid)
        if state_key is not None:
            if len(self._ensured_states) >= 16:
                self._ensured_states.clear()
            self._ensured_states.add(state_key)

    def _ensure_marker_goals(self, state) -> None:
        """Propose cover/touch candidates from observed movers and static sets."""
        history = self._negative_states[-15:] + [self._snapshot(state)]
        if len(history) < 3:
            return

        def group_objects(snapshot):
            grouped = {}
            view = {"grid": snapshot["grid"], "objects": snapshot["objects"]}
            for obj in snapshot["objects"]:
                if InteractionModel.is_scene_like(obj, view):
                    continue
                key = self.object_class(obj)
                center = tuple(
                    round(float(value), 1)
                    for value in obj.get("center", (0, 0))
                )
                grouped.setdefault(key, []).append((center, obj))
            return grouped

        frames = [group_objects(snapshot) for snapshot in history]
        class_keys = set().union(*(set(frame) for frame in frames))
        movers = set()
        static_centers = {}
        for class_key in class_keys:
            center_frames = [
                [center for center, _obj in frame.get(class_key, ())]
                for frame in frames
            ]
            moved = False
            for old_centers, new_centers in zip(center_frames, center_frames[1:]):
                unused = list(new_centers)
                for old in old_centers:
                    if not unused:
                        break
                    nearest = min(
                        range(len(unused)),
                        key=lambda index: float(np.linalg.norm(
                            np.asarray(old) - np.asarray(unused[index])
                        )),
                    )
                    new = unused.pop(nearest)
                    if np.linalg.norm(np.asarray(old) - np.asarray(new)) >= 1.0:
                        moved = True
                        break
                if moved:
                    break
            if moved:
                movers.add(class_key)

            persistent = set(center_frames[0])
            for centers in center_frames[1:]:
                persistent.intersection_update(centers)
            if persistent:
                static_centers[class_key] = persistent

        if not movers or not static_centers:
            return
        current = group_objects(history[-1])
        for mover_class in sorted(movers, key=repr):
            for marker_class, centers in sorted(
                static_centers.items(), key=lambda item: repr(item[0])
            ):
                marker_objects = [
                    obj for center, obj in current.get(marker_class, ())
                    if center in centers and obj.get("cells")
                ]
                if len(marker_objects) < 2:
                    continue
                groups = tuple(sorted(
                    tuple(sorted(tuple(map(int, cell))
                                 for cell in obj.get("cells", ())))
                    for obj in marker_objects
                ))
                if len(groups) < 2:
                    continue
                for transform in ("identity", "mirror_x", "mirror_y", "mirror_xy"):
                    for kind in ("cover_markers", "touch_markers"):
                        name = (
                            f"{kind}({mover_class!r},{marker_class!r},"
                            f"{transform},{len(groups)})"
                        )
                        self._add(GoalHypothesis(
                            name=name,
                            kind=kind,
                            colours=(),
                            mover_class=mover_class,
                            marker_class=marker_class,
                            marker_groups=groups,
                            static_marker_centers=tuple(sorted(centers)),
                            transform=transform,
                        ))

    @staticmethod
    def _contact_distance(grid, first, second) -> int:
        """Exact Manhattan distance between two colors on the finite grid.

        A sampled Cartesian estimate was unstable: a tiny frame edit could
        change which pixels were sampled and report a large false goal gain.
        Multi-source breadth-first search is exact and bounded by the 64x64
        grid size, independent of how many pixels either color occupies.
        """
        mask_a = np.asarray(grid) == first
        mask_b = np.asarray(grid) == second
        if not mask_a.any() or not mask_b.any():
            return max(grid.shape)

        distances = np.full(mask_b.shape, -1, dtype=np.int16)
        queue = deque()
        for row, col in np.argwhere(mask_b):
            row, col = int(row), int(col)
            distances[row, col] = 0
            queue.append((row, col))

        height, width = mask_b.shape
        while queue:
            row, col = queue.popleft()
            next_distance = int(distances[row, col]) + 1
            for next_row, next_col in (
                (row - 1, col), (row + 1, col),
                (row, col - 1), (row, col + 1),
            ):
                if (0 <= next_row < height and 0 <= next_col < width
                        and distances[next_row, next_col] < 0):
                    distances[next_row, next_col] = next_distance
                    queue.append((next_row, next_col))
        return int(distances[mask_a].min())

    def _initialize_missing_baselines(self, state_or_grid) -> None:
        grid = grid_of(state_or_grid)
        for goal in self.goals.values():
            if goal.kind == "cleared" and goal.initial_count is None:
                goal.initial_count = int(np.count_nonzero(grid == goal.colours[0]))
            elif goal.kind == "contact" and goal.initial_distance is None:
                goal.initial_distance = self._contact_distance(
                    grid, goal.colours[0], goal.colours[1]
                )

    def potential(self, state_or_grid) -> Dict[str, float]:
        """Return provisional [0,1] progress estimates for visual goal ideas."""
        state = state_or_grid if isinstance(state_or_grid, dict) else {}
        grid = grid_of(state_or_grid)
        visual_before = {
            name for name, goal in self.goals.items()
            if goal.kind != "relation"
        }
        self.ensure_goals(state_or_grid)
        visual_after = {
            name for name, goal in self.goals.items()
            if goal.kind != "relation"
        }
        if visual_before != visual_after:
            self._potential_cache.clear()
        cache_key = state.get("hash")
        if cache_key is not None and cache_key in self._potential_cache:
            return dict(self._potential_cache[cache_key])
        values = {}
        contact_distances = {}
        for name, goal in self.goals.items():
            if goal.kind == "relation":
                continue
            if goal.kind == "cleared":
                initial = max(1, goal.initial_count or 0)
                remaining = int(np.count_nonzero(grid == goal.colours[0]))
                values[name] = max(0.0, min(1.0, (initial - remaining) / initial))
            elif goal.kind == "contact":
                pair = tuple(sorted(goal.colours))
                if pair not in contact_distances:
                    contact_distances[pair] = self._contact_distance(grid, *pair)
                initial = max(1, goal.initial_distance or 0)
                current = contact_distances[pair]
                values[name] = max(0.0, min(1.0, 1.0 - current / initial))
            elif goal.kind in {"cover_markers", "touch_markers"}:
                marker_cells = {
                    tuple(cell) for group in goal.marker_groups for cell in group
                }
                if not marker_cells:
                    values[name] = 0.0
                    continue
                height, width = grid.shape
                best = 0
                for obj in state.get("objects", []):
                    if (self.object_class(obj) != goal.mover_class
                            or InteractionModel.is_scene_like(obj, state)):
                        continue
                    projected = set()
                    for row, col in obj.get("cells", ()):
                        row, col = int(row), int(col)
                        if goal.transform in {"mirror_x", "mirror_xy"}:
                            col = width - 1 - col
                        if goal.transform in {"mirror_y", "mirror_xy"}:
                            row = height - 1 - row
                        projected.add((row, col))
                    distance = 0 if goal.kind == "cover_markers" else 1
                    covered = sum(
                        any(
                            max(abs(marker_row - row), abs(marker_col - col))
                            <= distance
                            for row, col in projected
                        )
                        for marker_row, marker_col in marker_cells
                    )
                    best = max(best, covered)
                values[name] = best / len(marker_cells)
        if cache_key is not None:
            if len(self._potential_cache) >= 4:
                self._potential_cache.pop(next(iter(self._potential_cache)))
            self._potential_cache[cache_key] = dict(values)
        return values

    def shaping_delta(self, before, after) -> float:
        """Mean change in provisional goal potentials, never confirmed reward."""
        before_values = self.potential(before)
        after_values = self.potential(after)
        common = before_values.keys() & after_values.keys()
        if not common:
            return 0.0
        changes = [after_values[name] - before_values[name] for name in common]
        # Preserve directional evidence while preventing many untouched goals
        # from diluting the few predicates affected by an action.
        positive = [value for value in changes if value > 0]
        negative = [value for value in changes if value < 0]
        gain = sum(positive) / len(positive) if positive else 0.0
        loss = sum(negative) / len(negative) if negative else 0.0
        return max(-1.0, min(1.0, gain + loss))

    def reset_level_potentials(self) -> None:
        """Rebase provisional potentials when the environment advances level."""
        for goal in self.goals.values():
            goal.initial_count = None
            goal.initial_distance = None
        self._potential_cache.clear()
        self._ensured_states.clear()
        self._negative_states.clear()

    def _add(self, goal: GoalHypothesis) -> None:
        if goal.name not in self.goals and len(self.goals) < self.max_goals:
            if goal.kind == "relation" and goal.relation_key in self.relation_memory:
                goal.true_obs, goal.true_progress = self.relation_memory[
                    goal.relation_key
                ]
            self.goals[goal.name] = goal

    @staticmethod
    def _minimal_contrastive_set(success_truth, negative_truths):
        """Find the smallest true-predicate set separating success from negatives.

        A predicate covers a negative example when it is false there but true
        in the success state. Exact branch-and-bound set cover is used after
        duplicate and dominated coverage masks are removed. If any negative
        example is indistinguishable, no predicate set is credited.
        """
        if not negative_truths:
            return ()
        positive_names = sorted(
            name for name, holds in success_truth.items() if holds
        )
        if not positive_names:
            return ()

        masks_to_name = {}
        for name in positive_names:
            mask = 0
            for index, negative in enumerate(negative_truths):
                if not negative.get(name, False):
                    mask |= 1 << index
            if mask:
                masks_to_name.setdefault(mask, name)

        target = (1 << len(negative_truths)) - 1
        if not masks_to_name:
            return ()
        masks = list(masks_to_name)
        union = 0
        for mask in masks:
            union |= mask
        if union != target:
            return ()

        # Discard a predicate when another predicate separates every example
        # it separates and at least one more. Keeping the superset cannot make
        # a minimum solution worse.
        masks = [
            mask for mask in masks
            if not any(mask != other and mask & other == mask for other in masks)
        ]
        names = {mask: masks_to_name[mask] for mask in masks}
        ordered = sorted(masks, key=lambda mask: (-mask.bit_count(), names[mask]))

        greedy = []
        covered = 0
        while covered != target:
            available = [mask for mask in ordered if mask & ~covered]
            if not available:
                return ()
            choice = max(
                available,
                key=lambda mask: ((mask & ~covered).bit_count(), names[mask]),
            )
            greedy.append(choice)
            covered |= choice
        best = list(greedy)
        nodes = 0
        exhausted = False
        memo = {}

        def search(covered_now, chosen):
            nonlocal best, nodes, exhausted
            nodes += 1
            if nodes > 100_000:
                exhausted = True
                return
            if covered_now == target:
                if len(chosen) < len(best):
                    best = list(chosen)
                return
            if len(chosen) + 1 >= len(best):
                return
            previous = memo.get(covered_now)
            if previous is not None and previous <= len(chosen):
                return
            memo[covered_now] = len(chosen)

            uncovered = target & ~covered_now
            gains = [(mask & uncovered).bit_count() for mask in ordered]
            max_gain = max(gains, default=0)
            if not max_gain:
                return
            lower_bound = (uncovered.bit_count() + max_gain - 1) // max_gain
            if len(chosen) + lower_bound >= len(best):
                return

            # Branch on the uncovered negative with the fewest candidate
            # predicates, which sharply reduces the search tree.
            bit_positions = [i for i in range(len(negative_truths))
                             if uncovered & (1 << i)]
            pivot = min(
                bit_positions,
                key=lambda i: sum(bool(mask & (1 << i)) for mask in ordered),
            )
            branches = [mask for mask in ordered if mask & (1 << pivot)]
            branches.sort(
                key=lambda mask: (-(mask & uncovered).bit_count(), names[mask])
            )
            for mask in branches:
                search(covered_now | mask, chosen + [mask])
                if exhausted:
                    return

        search(0, [])
        if exhausted:
            # Do not attach a possibly non-minimal explanation when the
            # bounded exact search cannot certify the smallest set.
            return ()
        return tuple(sorted(names[mask] for mask in best))

    @staticmethod
    def _snapshot(state):
        grid = np.asarray(grid_of(state)).copy()
        return {
            "grid": grid,
            "objects": deepcopy(state.get("objects", [])),
            "relations": deepcopy(state.get("relations", [])),
        }

    def observe(self, state_after, progressed: bool, relevant_class=None) -> None:
        """Credit only predicates that contrast success with prior negatives.

        Credited predicates remain provisional until they hold again without
        an environment-confirmed advance; that counterexample retires them.
        """
        self.total_obs += 1
        if progressed:
            self.total_progress += 1
        truth = {
            name: goal.holds(state_after)
            for name, goal in self.goals.items()
        }
        credited = set()
        if progressed:
            negative_truths = [
                {name: goal.holds(snapshot) for name, goal in self.goals.items()}
                for snapshot in self._negative_states
            ]
            credited = set(self._minimal_contrastive_set(truth, negative_truths))

        for goal in self.goals.values():
            holds = truth.get(goal.name, False)
            if holds:
                goal.true_obs += 1
                if progressed and goal.name in credited:
                    goal.true_progress += 1
                    goal.contrastive_credits += 1
                    goal.pending_retest = True
                elif not progressed and goal.pending_retest:
                    goal.retest_failures += 1
                    goal.pending_retest = False
                if goal.kind == "relation":
                    self.relation_memory[goal.relation_key] = (
                        goal.true_obs,
                        goal.true_progress,
                    )
            elif progressed:
                # This does not prove the predicate is useless (it may be a
                # subgoal), but it is evidence against treating it as a
                # necessary completion condition.
                goal.progress_when_false += 1
                if goal.pending_retest:
                    goal.retest_failures += 1
                    goal.pending_retest = False

        if progressed:
            self._negative_states.clear()
        else:
            self._negative_states.append(self._snapshot(state_after))
            if len(self._negative_states) > 256:
                self._negative_states.pop(0)
        if progressed or self.total_obs % 25 == 0:
            self._save_memory()

    def weights(self) -> Dict[str, float]:
        """Return evidence-backed lift over the episode's progress rate.

        The prior is centred on the observed base rate. This matters when
        predicates have different support: a zero-success predicate with
        fewer observations must not get a larger smoothed rate simply because
        its denominator is smaller. Candidates also need minimum positive
        support before they can affect action ranking.
        """
        base = (self.total_progress + 1) / (self.total_obs + 2)
        lifts = {}
        for name, goal in self.goals.items():
            if (goal.true_obs < MIN_GOAL_SUPPORT or goal.true_progress <= 0
                    or goal.retest_failures > 0):
                continue
            posterior_rate = (
                goal.true_progress + GOAL_RATE_PRIOR_STRENGTH * base
            ) / (goal.true_obs + GOAL_RATE_PRIOR_STRENGTH)
            lift = posterior_rate - base
            if lift > 0:
                lifts[name] = lift
        total = sum(lifts.values())
        if total <= 0:
            return {}
        return {name: value / total for name, value in lifts.items() if value > 0}

    def relevance(self, state, sig) -> float:
        """Score actions that touch colors or objects in learned goal relations."""
        if sig[0] != CLICK_ACTION_ID or sig[1] is None or sig[2] is None:
            return 0.0
        grid = grid_of(state)
        x, y = int(sig[1]), int(sig[2])
        if not (0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]):
            return 0.0
        colour = int(grid[y, x])
        target_class = None
        objects = state.get("objects", []) if isinstance(state, dict) else []
        for obj in objects:
            if InteractionModel.is_scene_like(obj, state):
                continue
            if (y, x) in obj.get("cells", ()):
                target_class = self.object_class(obj)
                break
        weights = self.weights()
        relevance = 0.0
        for name, weight in weights.items():
            goal = self.goals[name]
            if colour in goal.colours:
                relevance += weight
            if goal.kind == "relation" and target_class in goal.relation_key[1:]:
                relevance += weight
            if (goal.kind in {"cover_markers", "touch_markers"}
                    and target_class == goal.mover_class):
                relevance += weight
        return relevance

    def top(self, k: int = 3):
        """Most-weighted goals, for logging and inspection."""
        weights = self.weights()
        ranked = sorted(weights.items(), key=lambda kv: -kv[1])
        return [(name, round(w, 3)) for name, w in ranked[:k]]

    def candidate_details(self, state, k: int = 12):
        """Expose evidence and a falsifier for generated goal candidates."""
        weights = self.weights()
        ranked = sorted(
            self.goals.values(),
            key=lambda goal: (
                -goal.true_progress,
                -goal.progress_when_false,
                -goal.true_obs,
                -weights.get(goal.name, 0.0),
            ),
        )
        rows = []
        for goal in ranked[:max(0, int(k))]:
            if goal.retest_failures:
                status = "retest_failed"
            elif goal.true_progress and goal.progress_when_false:
                status = "mixed_evidence"
            elif goal.true_progress:
                status = "needs_retest" if goal.pending_retest else "contrastive_support"
            elif goal.progress_when_false:
                status = "counterexample_observed"
            elif goal.true_obs:
                status = "observed_unconfirmed"
            else:
                status = "untested"
            rows.append({
                "name": goal.name,
                "kind": goal.kind,
                "weight": round(weights.get(goal.name, 0.0), 4),
                "holds_now": bool(goal.holds(state)),
                "observations_holding": goal.true_obs,
                "confirmed_progress_while_holding": goal.true_progress,
                "confirmed_progress_while_false": goal.progress_when_false,
                "contrastive_credits": goal.contrastive_credits,
                "retest_failures": goal.retest_failures,
                "pending_retest": goal.pending_retest,
                "status": status,
                "test": (
                    "Try a valid action that changes this predicate, then check "
                    "for environment-confirmed progress."
                ),
                "falsifier": (
                    "Environment-confirmed progress while this predicate is "
                    "false counts against it being a necessary goal condition."
                ),
            })
        return rows
