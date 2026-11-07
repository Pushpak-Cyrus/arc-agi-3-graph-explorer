"""Audit status-bar masking against a fresh, unmasked processor per frame.

The audit uses one generic action cycle on each locally cached environment.
It records the learned row band, cells removed relative to a fresh processor,
changes inside that band, and palette-free object motion intersecting it.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402

from Cyrus.frame_processor import FrameProcessor  # noqa: E402
from Cyrus.system2.action_effect_learner import (  # noqa: E402
    _center,
    _non_background_blobs,
    _shape_key,
)


def _frame(raw):
    frame = np.asarray(raw.frame)
    return frame[-1] if frame.ndim == 3 else frame


def _frame_objects(state):
    return [blob for blob in _non_background_blobs(state)
            if _shape_key(blob) is not None]


def run_game(arcade, game_id: str, scorecard_id: str, frames: int) -> dict:
    env = arcade.make(game_id, seed=0, scorecard_id=scorecard_id)
    if env is None:
        return {"game_id": game_id, "error": "environment unavailable"}
    raw = env.reset()
    if raw is None:
        return {"game_id": game_id, "error": "reset failed"}

    persistent = FrameProcessor(game_id)
    previous_raw = None
    previous_fresh = None
    result = {
        "game_id": game_id,
        "frames": 0,
        "mask_rows": None,
        "frames_masked": 0,
        "masked_nonzero_cells": 0,
        "dynamic_cells_erased": 0,
        "object_motion_overlapping_mask": 0,
        "max_frame_band_cells_erased": 0,
        "states": [],
    }
    action_cursor = 0

    for index in range(frames):
        raw_grid = _frame(raw).copy()
        persistent_state = persistent.process(np.asarray(raw.frame))
        fresh_state = FrameProcessor(game_id).process(np.asarray(raw.frame))
        persistent_grid = np.asarray(persistent_state["masked_grid"])[-1]
        fresh_grid = np.asarray(fresh_state["masked_grid"])[-1]
        mask = persistent.status_bar_mask

        if mask:
            start, end = map(int, mask)
            rows = tuple(range(start, end + 1))
            erased = int(np.count_nonzero(fresh_grid[start:end + 1]
                                          != persistent_grid[start:end + 1]))
            result["frames_masked"] += 1
            result["masked_nonzero_cells"] += erased
            result["max_frame_band_cells_erased"] = max(
                result["max_frame_band_cells_erased"], erased
            )
            result["mask_rows"] = [start, end]
            if previous_raw is not None:
                result["dynamic_cells_erased"] += int(np.count_nonzero(
                    raw_grid[start:end + 1] != previous_raw[start:end + 1]
                ))
            if previous_fresh is not None:
                old_by_shape = {}
                for blob in _frame_objects(previous_fresh):
                    old_by_shape.setdefault(_shape_key(blob), []).append(blob)
                for blob in _frame_objects(fresh_state):
                    shape = _shape_key(blob)
                    candidates = old_by_shape.get(shape, [])
                    if not candidates:
                        continue
                    now = _center(blob)
                    old = min(candidates, key=lambda item: np.hypot(
                        _center(item)[0] - now[0],
                        _center(item)[1] - now[1],
                    ))
                    before = _center(old)
                    if (before != now and
                            (start <= before[0] <= end
                             or start <= now[0] <= end)):
                        result["object_motion_overlapping_mask"] += 1
                        candidates.remove(old)

        result["frames"] += 1
        result["states"].append(getattr(raw.state, "name", str(raw.state)))
        previous_raw = raw_grid
        previous_fresh = fresh_state

        if index + 1 >= frames:
            break
        available = sorted({int(a) for a in raw.available_actions})
        if not available:
            break
        movement = [aid for aid in available if 1 <= aid <= 5]
        action_id = movement[action_cursor % len(movement)] if movement else available[0]
        action_cursor += 1
        try:
            data = {"x": 32, "y": 32} if action_id == 6 else None
            following = env.step(GameAction.from_id(action_id), data=data)
        except Exception as exc:  # Keep auditing other cached games if one action fails.
            result.setdefault("errors", []).append(f"action {action_id}: {exc}")
            break
        if following is None:
            break
        if getattr(following.state, "name", "") in {"GAME_OVER", "WIN"}:
            following = env.reset()
            if following is None:
                break
        raw = following

    result["states"] = dict(Counter(result["states"]))
    result["mask_detected"] = result["mask_rows"] is not None
    result["dynamic_band_change"] = result["dynamic_cells_erased"] > 0
    result["moving_object_in_band"] = result["object_motion_overlapping_mask"] > 0
    return result


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    games = [entry.game_id.split("-")[0] for entry in arcade.get_environments()]
    scorecard_id = f"status-mask-audit-{date.today().isoformat()}"
    results = [run_game(arcade, game, scorecard_id, frames=35)
               for game in games]

    out_dir = ROOT / "research" / "system2"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"status_bar_mask_audit_{date.today().isoformat()}.json"
    csv_path = json_path.with_suffix(".csv")
    json_path.write_text(json.dumps({"games": results}, indent=2),
                         encoding="utf-8")
    columns = [
        "game_id", "frames", "mask_detected", "mask_rows", "frames_masked",
        "masked_nonzero_cells", "dynamic_cells_erased",
        "object_motion_overlapping_mask", "dynamic_band_change",
        "moving_object_in_band", "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow({key: json.dumps(result.get(key))
                             if isinstance(result.get(key), (list, dict))
                             else result.get(key, "") for key in columns})

    detected = sum(bool(row.get("mask_detected")) for row in results)
    dynamic = sum(bool(row.get("dynamic_band_change")) for row in results)
    moving = sum(bool(row.get("moving_object_in_band")) for row in results)
    print(f"games={len(results)} masks_detected={detected} "
          f"dynamic_mask_bands={dynamic} moving_objects_in_mask={moving}")
    print(f"json={json_path}")
    print(f"csv={csv_path}")


if __name__ == "__main__":
    main()
