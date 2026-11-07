"""Frame-difference detection for CYRUS System 2.

Compares two consecutive observations and reports what changed:

* the number of changed cells,
* the connected regions of changed cells,
* object shifts: same-coloured blobs whose centroid moved.

Everything here is pure numpy and works on the dicts produced by
``FrameProcessor.process`` (keys: ``masked_grid``, ``blobs``) or on raw grids.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

Shift = Tuple[int, int]  # (d_row, d_col)


@dataclass(frozen=True)
class ChangedRegion:
    size: int
    bbox: Tuple[int, int, int, int]  # (r0, c0, r1, c1), inclusive
    centroid: Tuple[float, float]


@dataclass
class FrameDelta:
    changed_cells: int
    regions: List[ChangedRegion] = field(default_factory=list)
    # (colour, (d_row, d_col)) for every blob that moved between frames.
    shifts: List[Tuple[int, Shift]] = field(default_factory=list)
    # (palette-free silhouette, pre-action local context, displacement).
    # Used by ActionEffectLearner to distinguish identical objects in
    # different edge/neighbor configurations.
    object_context_shifts: List[Tuple] = field(default_factory=list)
    # (palette-free silhouette, displacement), for effect diagnostics.
    shape_shifts: List[Tuple] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.changed_cells > 0

    def to_dict(self) -> Dict:
        return {
            "changed_cells": self.changed_cells,
            "regions": [
                {"size": r.size, "bbox": list(r.bbox), "centroid": list(r.centroid)}
                for r in self.regions
            ],
            "shifts": [[int(colour), list(shift)] for colour, shift in self.shifts],
            "object_context_shifts": [
                [shape, context, list(shift)]
                for shape, context, shift in self.object_context_shifts
            ],
            "shape_shifts": [
                [shape, list(shift)] for shape, shift in self.shape_shifts
            ],
        }


def grid_of(state_or_grid) -> np.ndarray:
    """Return the 2D grid from a processed state dict, or from a raw array."""
    if isinstance(state_or_grid, dict):
        grid = state_or_grid.get("masked_grid", state_or_grid.get("grid"))
    else:
        grid = state_or_grid
    grid = np.asarray(grid)
    if grid.ndim == 3:
        grid = grid[-1]
    return grid


def centroid_of(coords: Sequence[Tuple[int, int]]) -> Tuple[float, float]:
    n = len(coords)
    if n == 0:
        return (0.0, 0.0)
    return (sum(r for r, _ in coords) / n, sum(c for _, c in coords) / n)


def _components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    """4-connected components of True cells (iterative, no recursion limit)."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components = []
    for r0 in range(height):
        for c0 in range(width):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            seen[r0, c0] = True
            stack = [(r0, c0)]
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(cells)
    return components


def diff_grids(before, after) -> FrameDelta:
    """Cell-level and region-level difference between two grids."""
    b = grid_of(before)
    a = grid_of(after)
    if b.shape != a.shape:
        raise ValueError(f"grid shapes differ: {b.shape} vs {a.shape}")

    mask = b != a
    regions = []
    for cells in _components(mask):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        regions.append(
            ChangedRegion(
                size=len(cells),
                bbox=(min(rows), min(cols), max(rows), max(cols)),
                centroid=centroid_of(cells),
            )
        )
    return FrameDelta(changed_cells=int(mask.sum()), regions=regions)


def match_blob_shifts(
    prev_blobs: Sequence[Dict],
    curr_blobs: Sequence[Dict],
    max_distance: float = 8.0,
) -> List[Tuple[int, Shift]]:
    """Match blobs of the same colour across frames and report their shifts.

    A blob matches a previous blob of the same colour whose size is within
    1.5x and whose centroid is within ``max_distance`` cells. Each previous
    blob can be matched at most once; nearest match wins.
    """
    prev = [
        (int(b["primary_value"]), centroid_of(b["coords"]), int(b["size"]))
        for b in prev_blobs
        if b.get("coords")
    ]
    used = set()
    shifts: List[Tuple[int, Shift]] = []

    for blob in curr_blobs:
        if not blob.get("coords"):
            continue
        colour = int(blob["primary_value"])
        size = int(blob["size"])
        cen = centroid_of(blob["coords"])

        best_idx, best_dist = None, None
        for idx, (pv, pc, ps) in enumerate(prev):
            if idx in used or pv != colour:
                continue
            if max(size, ps) / max(1, min(size, ps)) > 1.5:
                continue
            dist = math.dist(cen, pc)
            if dist <= max_distance and (best_dist is None or dist < best_dist):
                best_idx, best_dist = idx, dist

        if best_idx is None:
            continue
        used.add(best_idx)
        pc = prev[best_idx][1]
        shift = (int(round(cen[0] - pc[0])), int(round(cen[1] - pc[1])))
        if shift != (0, 0):
            shifts.append((colour, shift))

    return shifts


def diff_states(before_state: Dict, after_state: Dict) -> FrameDelta:
    """Full delta between two processed states: cells, regions, and shifts."""
    delta = diff_grids(before_state, after_state)
    delta.shifts = match_blob_shifts(
        before_state.get("blobs", []),
        after_state.get("blobs", []),
    )
    return delta
