"""Small, perception-driven planners for repeated visual relations.

This planner handles levels where matching movable shapes control a marker at
their centroid and one or more sparse ring targets are visible. It infers the
groups and target mapping from the current frame, then searches safe placements
on the dominant free-space color. It uses no game IDs, sprite names, or fixed
screen coordinates.
"""

from __future__ import annotations

from collections import defaultdict
import itertools
import heapq
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Signature = Tuple[int, Optional[int], Optional[int]]


class RelationalPlanner:
    """Infer and execute short action plans from visual geometric relations."""

    def __init__(self) -> None:
        self.plan: List[Signature] = []
        self.index = 0
        self.level_key = None
        self.attempted_level = None
        self.last_action: Optional[Signature] = None
        self.last_action_index: Optional[int] = None
        self.failed_destinations = set()
        self.action_outcomes = defaultdict(lambda: {"success": 0, "no_motion": 0})

    def reset(self) -> None:
        self.plan.clear()
        self.index = 0
        self.level_key = None
        self.attempted_level = None
        self.last_action = None
        self.last_action_index = None
        self.failed_destinations.clear()
        self.action_outcomes.clear()

    @staticmethod
    def _shape_centers(state):
        """Summarize repeated movable shapes by geometry, not color IDs."""
        grouped = defaultdict(list)
        for blob in state.get("blobs", ()):
            key = RelationalPlanner._shape_key(blob)
            if key is not None:
                grouped[key].append(RelationalPlanner._center(blob))
        return {key: sorted(points) for key, points in grouped.items()}

    def observe(self, state_before, state_after, action, level_key=None):
        """Learn whether the last planned placement caused visible motion.

        A selection click is not expected to move a piece. A placement click is
        checked against the next frame; failed actions invalidate the stale
        remainder so the planner can infer a fresh plan from the observed board.
        """
        if self.level_key != level_key or self.last_action_index is None:
            return
        if self.last_action_index % 2 == 0:
            return
        before = self._shape_centers(state_before or {})
        after = self._shape_centers(state_after or {})
        moved = any(
            len(before.get(key, ())) == len(points)
            and any(math.dist(a, b) >= 1.0
                    for a, b in zip(before[key], points))
            for key, points in after.items()
        )
        outcome = self.action_outcomes[tuple(action or ())]
        outcome["success" if moved else "no_motion"] += 1
        if os.getenv("CYRUS_RELATIONAL_DEBUG") == "1":
            print(
                f"[CYRUS][relational-feedback] index={self.last_action_index} "
                f"action={action} moved={moved}"
            )
        if not moved:
            self.failed_destinations.add(tuple(action or ()))
            self.plan = []
            self.index = 0
            self.attempted_level = None
            self.last_action = None
            self.last_action_index = None

    @staticmethod
    def _bbox(cells: Iterable[Tuple[int, int]]):
        cells = list(cells)
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        return min(rows), min(cols), max(rows), max(cols)

    @staticmethod
    def _shape_key(blob: Dict):
        cells = blob.get("coords", ())
        if not cells:
            return None
        r0, c0, _, _ = RelationalPlanner._bbox(cells)
        return (max(r for r, _ in cells) - r0 + 1,
                max(c for _, c in cells) - c0 + 1,
                tuple(sorted((r - r0, c - c0) for r, c in cells)))

    @staticmethod
    def _center(blob: Dict) -> Tuple[float, float]:
        cells = blob.get("coords", ())
        if not cells:
            return (0.0, 0.0)
        # Movable sprite centers follow their rectangular footprint, not the
        # mean of occupied pixels. This matches the carrier update rule, which
        # averages each sprite's top-left plus integer half-width/half-height;
        # asymmetric shapes can have a very different pixel centroid.
        r0, c0, r1, c1 = RelationalPlanner._bbox(cells)
        return (
            r0 + (r1 - r0 + 1) // 2,
            c0 + (c1 - c0 + 1) // 2,
        )

    @staticmethod
    def _ring_candidates(blobs: Sequence[Dict]):
        """Collect compact clusters of sparse same-color marks shaped like rings."""
        by_color = defaultdict(set)
        for blob in blobs:
            if int(blob.get("size", 0)) > 4:
                continue
            color = int(blob.get("primary_value", -1))
            by_color[color].update(tuple(point) for point in blob.get("coords", ()))

        found = []
        for color, points in by_color.items():
            points = sorted(points)
            for seed_r, seed_c in points:
                group = [(r, c) for r, c in points
                         if abs(r - seed_r) <= 4 and abs(c - seed_c) <= 4]
                if len(group) < 8:
                    continue
                r0, c0, r1, c1 = RelationalPlanner._bbox(group)
                h, w = r1 - r0 + 1, c1 - c0 + 1
                if not (5 <= h <= 9 and 5 <= w <= 9):
                    continue
                center = (sum(r for r, _ in group) / len(group),
                          sum(c for _, c in group) / len(group))
                # Keep sparse marks near the bounding-box perimeter and reject
                # filled clusters such as sprites or counters.
                perimeter = [
                    (r, c) for r, c in group
                    if min(r - r0, r1 - r, c - c0, c1 - c) <= 1
                ]
                if len(perimeter) / len(group) < 0.8:
                    continue
                found.append((center, set(group), color))
        # Sliding windows overlap around one sparse ring. Merge those windows
        # so its center is calculated from the full visual pattern.
        remaining = list(found)
        merged = []
        while remaining:
            center, cells, color = remaining.pop()
            changed = True
            while changed:
                changed = False
                for idx in range(len(remaining) - 1, -1, -1):
                    other_center, other_cells, other_color = remaining[idx]
                    if other_color == color and cells.intersection(other_cells):
                        cells.update(other_cells)
                        remaining.pop(idx)
                        changed = True
            center = (sum(r for r, _ in cells) / len(cells),
                      sum(c for _, c in cells) / len(cells))
            merged.append((center, cells, color))
        return merged

    @staticmethod
    def _matching_pairs(blobs: Sequence[Dict]):
        grouped = defaultdict(list)
        for blob in blobs:
            size = int(blob.get("size", 0))
            key = RelationalPlanner._shape_key(blob)
            if key is None:
                continue
            h, w, _ = key
            if 4 <= h <= 7 and 4 <= w <= 7 and 8 <= size <= 32:
                grouped[key].append(blob)
        pairs = []
        for key, matches in grouped.items():
            if len(matches) < 2:
                continue
            for first, second in itertools.combinations(matches, 2):
                a, b = RelationalPlanner._center(first), RelationalPlanner._center(second)
                if math.dist(a, b) < max(key[0], key[1]) + 2:
                    continue
                pairs.append((first, second, key))
        return pairs

    @staticmethod
    def _marker_matches(blobs: Sequence[Dict], pair):
        first, second, _ = pair
        a, b = RelationalPlanner._center(first), RelationalPlanner._center(second)
        midpoint = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        markers = []
        for blob in blobs:
            cells = blob.get("coords", ())
            size = len(cells)
            if not 8 <= size <= 40 or blob in (first, second):
                continue
            r0, c0, r1, c1 = RelationalPlanner._bbox(cells)
            if r1 - r0 + 1 > 7 or c1 - c0 + 1 > 7:
                continue
            center = RelationalPlanner._center(blob)
            distance = math.dist(center, midpoint)
            if distance <= 2.0:
                markers.append((distance, blob))
        return sorted(markers, key=lambda item: item[0])

    @staticmethod
    def _placement_is_free(
        grid, blob: Dict, center_rc: Tuple[int, int], background: int,
        forbidden: set,
    ) -> bool:
        r0, c0, r1, c1 = RelationalPlanner._bbox(blob.get("coords", ()))
        old_center = tuple(round(v) for v in RelationalPlanner._center(blob))
        dr, dc = center_rc[0] - old_center[0], center_rc[1] - old_center[1]
        # Use the whole visual footprint as a conservative collision envelope,
        # including transparent gaps inside a sprite's bounding box.
        for r in range(r0 + dr, r1 + dr + 1):
            for c in range(c0 + dc, c1 + dc + 1):
                if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
                    return False
                if (r, c) in forbidden or int(grid[r, c]) != background:
                    return False
        return True

    @staticmethod
    def _destination_options(state, pieces, goal_center, goal_cells, background):
        """Find balanced, collision-free destination layouts for 2 or 3 pieces."""
        grid = state["grid"]
        import numpy as np
        grid = np.asarray(grid)
        if grid.ndim == 3:
            grid = grid[-1]
        target = (round(goal_center[0]), round(goal_center[1]))
        layouts = []
        if len(pieces) == 2:
            offsets = [
                (dr, dc)
                for radius in (8, 10, 12, 14)
                for dr in range(-radius, radius + 1, 2)
                for dc in range(-radius, radius + 1, 2)
                if 8 <= math.hypot(dr, dc) <= 14
            ]
            for dr, dc in offsets:
                destination_offsets = [(dr, dc), (-dr, -dc)]
                destinations = [(target[0] + r, target[1] + c)
                                for r, c in destination_offsets]
                if not all(RelationalPlanner._placement_is_free(
                    grid, piece, dest, background, goal_cells
                ) for piece, dest in zip(pieces, destinations)):
                    continue
                footprints = []
                for piece, destination in zip(pieces, destinations):
                    r0, c0, r1, c1 = RelationalPlanner._bbox(piece["coords"])
                    old = tuple(round(v) for v in RelationalPlanner._center(piece))
                    shift = (destination[0] - old[0], destination[1] - old[1])
                    footprints.append({
                        (r, c)
                        for r in range(r0 + shift[0], r1 + shift[0] + 1)
                        for c in range(c0 + shift[1], c1 + shift[1] + 1)
                    })
                if footprints[0] & footprints[1]:
                    continue
                cost = sum(math.dist(RelationalPlanner._center(piece), dest)
                           for piece, dest in zip(pieces, destinations))
                layouts.append((cost, destinations, footprints))
        elif len(pieces) == 3:
            for radius in (8, 10, 12, 14):
                base = ((0, -radius),
                        (round(radius * 0.866), round(radius * 0.5)),
                        (-round(radius * 0.866), round(radius * 0.5)))
                rotations = []
                for quarter_turns in range(4):
                    rotated = []
                    for dr, dc in base:
                        for _ in range(quarter_turns):
                            dr, dc = dc, -dr
                        rotated.append((dr, dc))
                    rotations.append(tuple(rotated))
                for destination_offsets in set(rotations):
                    destinations = [(target[0] + r, target[1] + c)
                                    for r, c in destination_offsets]
                    if not all(RelationalPlanner._placement_is_free(
                        grid, piece, dest, background, goal_cells
                    ) for piece, dest in zip(pieces, destinations)):
                        continue
                    footprints = []
                    for piece, destination in zip(pieces, destinations):
                        r0, c0, r1, c1 = RelationalPlanner._bbox(piece["coords"])
                        old = tuple(round(v) for v in RelationalPlanner._center(piece))
                        shift = (destination[0] - old[0], destination[1] - old[1])
                        footprints.append({
                            (r, c)
                            for r in range(r0 + shift[0], r1 + shift[0] + 1)
                            for c in range(c0 + shift[1], c1 + shift[1] + 1)
                        })
                    if any(footprints[i] & footprints[j]
                           for i in range(3) for j in range(i + 1, 3)):
                        continue
                    cost = sum(math.dist(RelationalPlanner._center(piece), dest)
                               for piece, dest in zip(pieces, destinations))
                    layouts.append((cost, destinations, footprints))
        return sorted(layouts, key=lambda item: item[0])

    @staticmethod
    def _footprint(blob, center_rc):
        r0, c0, r1, c1 = RelationalPlanner._bbox(blob.get("coords", ()))
        old_r, old_c = (round(v) for v in RelationalPlanner._center(blob))
        new_r, new_c = (round(value) for value in center_rc)
        dr, dc = new_r - old_r, new_c - old_c
        return {
            (r, c)
            for r in range(r0 + dr, r1 + dr + 1)
            for c in range(c0 + dc, c1 + dc + 1)
        }

    @classmethod
    def _click_hits_piece(cls, point_rc, piece, center_rc):
        """Return whether ACTION6 would select this piece at the click point."""
        r0, c0, r1, c1 = cls._bbox(piece.get("coords", ()))
        old_r, old_c = (round(value) for value in cls._center(piece))
        new_r, new_c = (round(value) for value in center_rc)
        dr, dc = new_r - old_r, new_c - old_c
        row, col = point_rc
        return r0 + dr <= row <= r1 + dr and c0 + dc <= col <= c1 + dc

    @staticmethod
    def _path_is_clear(grid, background, blocked, start, end, footprint):
        """Check a moving visual footprint along its straight-line trajectory."""
        r0, c0, r1, c1 = footprint
        steps = max(1, int(math.ceil(math.dist(start, end) * 2)))
        for step in range(steps + 1):
            t = step / steps
            cr = round(start[0] + (end[0] - start[0]) * t)
            cc = round(start[1] + (end[1] - start[1]) * t)
            for r in range(cr + r0, cr + r1 + 1):
                for c in range(cc + c0, cc + c1 + 1):
                    if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
                        return False
                    if (r, c) in blocked:
                        return False
        return True

    @classmethod
    def _safe_group_routes(cls, state, pieces, destinations, marker, goal,
                           background, blocked, failed_destinations=()):
        """Try destination assignments and action orders; reject obstacle paths."""
        import numpy as np
        grid = np.asarray(state["grid"])
        if grid.ndim == 3:
            grid = grid[-1]
        count = len(pieces)
        routes = []
        unique_destinations = set(itertools.permutations(destinations))
        marker_center = cls._center(marker)
        marker_box = cls._bbox(marker.get("coords", ()))
        marker_footprint = (
            marker_box[0] - round(marker_center[0]),
            marker_box[1] - round(marker_center[1]),
            marker_box[2] - round(marker_center[0]),
            marker_box[3] - round(marker_center[1]),
        )
        goal_center = goal[0]

        for assigned_destinations in unique_destinations:
            if any((6, int(round(destination[1])), int(round(destination[0])))
                   in failed_destinations for destination in assigned_destinations):
                continue
            final_cells = [cls._footprint(piece, target)
                           for piece, target in zip(pieces, assigned_destinations)]
            if any(final_cells[i] & final_cells[j]
                   for i in range(count) for j in range(i + 1, count)):
                continue
            for order in itertools.permutations(range(count)):
                centers = [cls._center(piece) for piece in pieces]
                carrier = tuple(sum(center[axis] for center in centers) / count
                                for axis in (0, 1))
                safe = True
                for idx in order:
                    start = centers[idx]
                    end = assigned_destinations[idx]
                    if any(cls._click_hits_piece(end, piece, centers[j])
                           for j, piece in enumerate(pieces)):
                        safe = False
                        break
                    # The selected piece is validated at its destination by
                    # _destination_options. Only the aggregate carrier has a
                    # path collision constraint in the observed transition.
                    next_carrier = tuple(
                        carrier[axis] + (end[axis] - start[axis]) / count
                        for axis in (0, 1)
                    )
                    if not cls._path_is_clear(
                        grid, background, blocked, carrier, next_carrier,
                        marker_footprint,
                    ):
                        safe = False
                        break
                    centers[idx] = end
                    carrier = next_carrier
                if not safe or math.dist(carrier, goal_center) > 2.0:
                    continue
                cost = sum(math.dist(cls._center(piece), destination)
                           for piece, destination in zip(pieces, assigned_destinations))
                routes.append((cost, order, assigned_destinations, final_cells))
        return sorted(routes, key=lambda item: item[0])

    @classmethod
    def _waypoint_group_route(cls, state, pieces, marker, goal, background,
                              blocked, reserved, failed_destinations=(), max_moves=20):
        """Route a group marker around obstacles using incremental piece moves."""
        import numpy as np
        grid = np.asarray(state["grid"])
        if grid.ndim == 3:
            grid = grid[-1]
        count = len(pieces)
        centers = [cls._center(piece) for piece in pieces]
        carriers = tuple(sum(center[axis] for center in centers) / count
                         for axis in (0, 1))
        target = tuple(round(value) for value in goal[0])

        marker_center = cls._center(marker)
        marker_box = cls._bbox(marker.get("coords", ()))
        carrier_shape = (
            marker_box[0] - round(marker_center[0]),
            marker_box[1] - round(marker_center[1]),
            marker_box[2] - round(marker_center[0]),
            marker_box[3] - round(marker_center[1]),
        )

        def carrier_free(point):
            r0, c0, r1, c1 = carrier_shape
            cr, cc = point
            return all(
                0 <= r < grid.shape[0]
                and 0 <= c < grid.shape[1]
                and (r, c) not in blocked
                for r in range(cr + r0, cr + r1 + 1)
                for c in range(cc + c0, cc + c1 + 1)
            )

        def astar(start):
            start = (round(start[0]), round(start[1]))
            if not carrier_free(target):
                return None
            frontier = [(math.dist(start, target), 0.0, start)]
            previous = {start: None}
            costs = {start: 0.0}
            while frontier:
                _, cost, current = heapq.heappop(frontier)
                if current == target:
                    path = []
                    node = current
                    while node is not None:
                        path.append(node)
                        node = previous[node]
                    return list(reversed(path))
                if cost != costs[current]:
                    continue
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nxt = (current[0] + dr, current[1] + dc)
                    if not carrier_free(nxt):
                        continue
                    new_cost = cost + 1
                    if new_cost >= costs.get(nxt, float("inf")):
                        continue
                    costs[nxt] = new_cost
                    previous[nxt] = current
                    heapq.heappush(
                        frontier,
                        (new_cost + math.dist(nxt, target), new_cost, nxt),
                    )
            return None

        piece_cells = [set(tuple(point) for point in piece.get("coords", ()))
                       for piece in pieces]
        piece_boxes = [cls._bbox(piece.get("coords", ())) for piece in pieces]
        sequence = []
        total_cost = 0.0
        for _ in range(max_moves):
            carrier = tuple(sum(center[axis] for center in centers) / count
                            for axis in (0, 1))
            if math.dist(carrier, target) <= 1.5:
                final = [cls._footprint(piece, centers[idx])
                         for idx, piece in enumerate(pieces)]
                return total_cost, sequence, final
            path = astar(carrier)
            if not path or len(path) < 2:
                if os.getenv("CYRUS_RELATIONAL_DEBUG") == "1":
                    print(
                        f"[CYRUS][relational-route] no_carrier_path "
                        f"start={carrier} target={target}"
                    )
                if sequence:
                    final = [cls._footprint(piece, centers[idx])
                             for idx, piece in enumerate(pieces)]
                    return total_cost, sequence, final
                return None

            chosen = None
            furthest = min(7, len(path) - 1)
            for path_index in range(furthest, 0, -1):
                waypoint = path[path_index]
                for idx, piece in enumerate(pieces):
                    old_center = centers[idx]
                    destination = tuple(
                        round(old_center[axis] + count * (waypoint[axis] - carrier[axis]))
                        for axis in (0, 1)
                    )
                    if (6, destination[1], destination[0]) in failed_destinations:
                        continue
                    if any(cls._click_hits_piece(destination, piece, centers[j])
                           for j, piece in enumerate(pieces)):
                        continue
                    footprint = cls._footprint(piece, destination)
                    if any(
                        not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1])
                        for r, c in footprint
                    ):
                        continue
                    if footprint & (blocked | reserved | goal[1]):
                        continue
                    if any(footprint & cls._footprint(other, centers[j])
                           for j, other in enumerate(pieces) if j != idx):
                        continue
                    # As in the direct planner, validate the piece's final
                    # footprint, not its interpolated path. The carrier path
                    # below remains collision-checked at every step.
                    updated = list(centers)
                    updated[idx] = destination
                    next_carrier = tuple(
                        sum(center[axis] for center in updated) / count
                        for axis in (0, 1)
                    )
                    if not cls._path_is_clear(
                        grid, background, blocked, carrier, next_carrier,
                        carrier_shape,
                    ):
                        continue
                    chosen = (idx, destination, footprint, next_carrier)
                    break
                if chosen is not None:
                    break
            if chosen is None:
                if os.getenv("CYRUS_RELATIONAL_DEBUG") == "1":
                    print(
                        f"[CYRUS][relational-route] no_piece_step "
                        f"carrier={carrier} path_len={len(path)} "
                        f"group_size={count}"
                    )
                if sequence:
                    final = [cls._footprint(piece, centers[idx])
                             for idx, piece in enumerate(pieces)]
                    return total_cost, sequence, final
                return None

            idx, destination, footprint, _ = chosen
            old = centers[idx]
            dr = destination[0] - round(old[0])
            dc = destination[1] - round(old[1])
            centers[idx] = destination
            piece_cells[idx] = {
                (r + dr, c + dc) for r, c in piece_cells[idx]
            }
            r0, c0, r1, c1 = piece_boxes[idx]
            piece_boxes[idx] = (r0 + dr, c0 + dc, r1 + dr, c1 + dc)
            pieces[idx] = dict(pieces[idx], coords=piece_cells[idx])
            sequence.append((idx, old, destination))
            total_cost += math.dist(old, destination)

        if sequence:
            final = [cls._footprint(piece, centers[idx])
                     for idx, piece in enumerate(pieces)]
            return total_cost, sequence, final
        return None

    def infer_plan(self, state: Dict, level_key=None,
                   failed_destinations=()) -> Optional[List[Signature]]:
        """Return a high-confidence pair-to-ring plan, otherwise no plan."""
        blobs = state.get("blobs", ())
        grid = state.get("grid")
        if grid is None:
            return None
        try:
            import numpy as np
            grid = np.asarray(grid)
            if grid.ndim == 3:
                grid = grid[-1]
            if grid.ndim != 2 or not grid.size:
                return None
            colors, counts = np.unique(grid, return_counts=True)
            background = int(colors[int(counts.argmax())])
        except (TypeError, ValueError):
            return None

        rings = self._ring_candidates(blobs)
        if not rings:
            return None

        shape_groups = defaultdict(list)
        for blob in blobs:
            key = self._shape_key(blob)
            if key is None:
                continue
            h, w, _ = key
            if 4 <= h <= 7 and 4 <= w <= 7 and 8 <= len(blob.get("coords", ())) <= 32:
                shape_groups[key].append(blob)

        # This first relational family supports two- and three-piece groups.
        # Require a complete visual account of all repeated pieces before use.
        marker_blobs = []
        for blob in blobs:
            cells = blob.get("coords", ())
            if not 8 <= len(cells) <= 40:
                continue
            r0, c0, r1, c1 = self._bbox(cells)
            if r1 - r0 + 1 <= 7 and c1 - c0 + 1 <= 7:
                marker_blobs.append(blob)

        candidate_plans = []
        for shape_key, pieces in shape_groups.items():
            if len(pieces) not in (2, 5):
                continue
            options_by_marker = []
            for marker in marker_blobs:
                color = int(marker.get("primary_value", -1))
                marker_center = self._center(marker)
                matching_rings = [ring for ring in rings if ring[2] == color]
                if len(matching_rings) != 1:
                    continue
                goal = matching_rings[0]
                if math.dist(marker_center, goal[0]) < 5:
                    continue
                group_options = []
                for count in (2, 3):
                    if count > len(pieces):
                        continue
                    for group in itertools.combinations(pieces, count):
                        center = tuple(
                            sum(self._center(piece)[axis] for piece in group) / count
                            for axis in (0, 1)
                        )
                        distance = math.dist(center, marker_center)
                        if distance <= 1.25:
                            group_options.append((distance, group, goal, marker))
                if group_options:
                    group_options.sort(key=lambda item: item[0])
                    options_by_marker.append(group_options)

            if not options_by_marker:
                continue
            for assignment in itertools.product(*options_by_marker):
                assigned = [piece for _, group, _, _ in assignment for piece in group]
                if len({id(piece) for piece in assigned}) != len(assigned):
                    continue
                if {id(piece) for piece in assigned} != {id(piece) for piece in pieces}:
                    continue
                goal_keys = [tuple(sorted(goal[1])) for _, _, goal, _ in assignment]
                if len(set(goal_keys)) != len(goal_keys):
                    continue
                candidate_plans.append((
                    sum(distance for distance, _, _, _ in assignment),
                    assignment,
                    pieces,
                ))

        if len(candidate_plans) != 1:
            return None
        _, assignment, _ = candidate_plans[0]
        # Treat every non-background pixel outside known movers, markers, and
        # targets as a fixed obstacle for both piece and centroid trajectories.
        dynamic_cells = set()
        known_dynamic = []
        for blob in blobs:
            if (blob in [piece for _, group, _, _ in assignment for piece in group]
                    or blob in [marker for _, _, _, marker in assignment]):
                cells = {tuple(point) for point in blob.get("coords", ())}
                dynamic_cells.update(cells)
                known_dynamic.append(cells)
        for _, _, goal, _ in assignment:
            dynamic_cells.update(goal[1])
        for obj in state.get("objects", ()):
            cells = {tuple(point) for point in obj.get("cells", ())}
            if any(cells.intersection(known) for known in known_dynamic):
                # The target merger may group a tether/trail with the movable
                # objects. Treat that connected visual structure as dynamic.
                dynamic_cells.update(cells)
        static_blocked = set()
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if int(grid[r, c]) != background and (r, c) not in dynamic_cells:
                    static_blocked.add((r, c))

        group_layout_options = []
        reserved = set()
        for _, group, goal, marker in assignment:
            ordered_group = sorted(group, key=self._center)
            layouts = self._destination_options(
                state, ordered_group, goal[0], goal[1], background
            )
            routes = []
            for _, destinations, _ in layouts:
                for route in self._safe_group_routes(
                    state, ordered_group, destinations, marker, goal,
                    background, static_blocked, failed_destinations,
                ):
                    cost, order, assigned_destinations, footprints = route
                    routes.append((
                        cost,
                        [(idx, self._center(ordered_group[idx]),
                          assigned_destinations[idx]) for idx in order],
                        footprints,
                    ))
            if not routes:
                route = self._waypoint_group_route(
                    state, ordered_group, marker, goal, background,
                    static_blocked, reserved, failed_destinations,
                )
                if route is None:
                    if os.getenv("CYRUS_RELATIONAL_DEBUG") == "1":
                        print(
                            f"[CYRUS][relational-route] group_unroutable "
                            f"size={len(ordered_group)} "
                            f"centers={[self._center(p) for p in ordered_group]} "
                            f"marker={self._center(marker)} target={goal[0]}"
                        )
                    return None
                cost, sequence, footprints = route
                routes.append((cost, sequence, footprints))
            routes.sort(key=lambda item: item[0])
            group_layout_options.append((ordered_group, routes))
            # Keep future group routes clear of the planned final positions.
            best_route = routes[0]
            reserved.update(set().union(*best_route[2]))

        complete_layouts = []
        for selection in itertools.product(*[items[1] for items in group_layout_options]):
            footprints = [footprint for _, _, cells in selection for footprint in cells]
            if any(footprints[i] & footprints[j]
                   for i in range(len(footprints))
                   for j in range(i + 1, len(footprints))):
                continue
            cost = sum(item[0] for item in selection)
            complete_layouts.append((cost, selection))
        if not complete_layouts:
            return None
        _, selected_layout = min(complete_layouts, key=lambda item: item[0])

        def action(x, y):
            return (6, int(round(x)), int(round(y)))

        plan = []
        for (pieces, _), (_, sequence, _) in zip(
            group_layout_options, selected_layout
        ):
            for idx, select_center, destination in sequence:
                piece_r, piece_c = select_center
                plan.append(action(piece_c, piece_r))
                plan.append(action(destination[1], destination[0]))
        return plan

    def choose(self, state: Dict, available: Sequence[Signature], level_key=None):
        """Continue a verified plan or infer one when the relation is visible."""
        if self.level_key != level_key:
            self.plan = []
            self.index = 0
            self.level_key = level_key
            self.attempted_level = None
            self.last_action = None
            self.last_action_index = None
            self.failed_destinations.clear()

        # A new observation means the previous action has completed. Advance
        # only if it was the plan action we actually issued.
        if self.last_action is not None:
            self.index += 1
            self.last_action = None
        if self.plan and self.index >= len(self.plan):
            self.plan = []
            self.index = 0
            # A waypoint segment completed; re-infer from the observed board.
            self.attempted_level = None

        if not self.plan and self.attempted_level != level_key:
            self.attempted_level = level_key
            self.plan = self.infer_plan(
                state, level_key, self.failed_destinations
            ) or []
            if os.getenv("CYRUS_RELATIONAL_DEBUG") == "1":
                print(
                    f"[CYRUS][relational-plan] level={level_key} "
                    f"actions={self.plan}"
                )

        if not self.plan or self.index >= len(self.plan):
            return None
        candidate = self.plan[self.index]
        is_valid_coordinate = (
            candidate[0] == 6
            and candidate[1] is not None
            and candidate[2] is not None
            and 0 <= candidate[1] < 64
            and 0 <= candidate[2] < 64
        )
        if candidate not in available and not is_valid_coordinate:
            self.plan = []
            self.index = 0
            return None
        self.last_action = candidate
        self.last_action_index = self.index
        return candidate
