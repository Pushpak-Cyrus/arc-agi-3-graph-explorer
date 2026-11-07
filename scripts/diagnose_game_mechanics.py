"""Probe available actions and visible mechanics in locally cached ARC games.

The diagnostic resets the same seeded level before every probe. It does not
make decisions or search for solutions; it records one-step visual effects and
repeatable object shifts. Output is written under research/system2/.

Usage:
    .venv/Scripts/python.exe scripts/diagnose_game_mechanics.py
    .venv/Scripts/python.exe scripts/diagnose_game_mechanics.py --games ls20,vc33
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402

from Cyrus.frame_processor import FrameProcessor  # noqa: E402
from Cyrus.system2.frame_diff import diff_states  # noqa: E402


DEFAULT_STALLED_GAMES = [
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59",
    "lf52", "lp85", "ls20", "m0r0", "re86", "s5i5", "sb26", "sc25",
    "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30",
]


def as_grid(raw: Any) -> np.ndarray:
    frame = np.asarray(raw.frame)
    if frame.ndim == 3:
        frame = frame[-1]
    if frame.ndim != 2:
        raise ValueError(f"Expected a 2D game grid, got {frame.shape}")
    return frame


def state_for(raw: Any, game_id: str) -> dict:
    state = FrameProcessor(game_id).process(np.asarray(raw.frame))
    # Keep color components for displacement measurement, but omit components
    # assigned to the largest background target. Grouped targets can merge a
    # moving sprite with the board, hiding real shifts; counting every raw
    # fragment previously overcounted repeated colors across a single trial.
    background_indices = {
        int(index)
        for target in state.get("targets", [])
        if target.get("is_background_candidate")
        for index in target.get("member_blob_indices", [])
    }
    state["blobs"] = [blob for index, blob in enumerate(state.get("blobs", []))
                      if index not in background_indices]
    return state


def movement_trial_counts(trials: list[dict]) -> Counter:
    """Count each movement signature once per reset-and-action trial."""
    counts = Counter()
    for trial in trials:
        signatures = {
            (int(color), int(shift[0]), int(shift[1]))
            for color, shift in trial["delta"]["shifts"]
        }
        counts.update(signatures)
    return counts


def state_name(raw: Any) -> str:
    return getattr(raw.state, "name", str(raw.state))


def action_label(action_id: int) -> str:
    try:
        return GameAction.from_id(action_id).name
    except ValueError:
        return f"ACTION{action_id}"


def summarize_delta(before: Any, after: Any, game_id: str) -> dict:
    delta = diff_states(state_for(before, game_id), state_for(after, game_id))
    return {
        "changed_cells": delta.changed_cells,
        "shifts": [[color, list(shift)] for color, shift in delta.shifts],
        "state_after": state_name(after),
        "levels_after": int(after.levels_completed),
    }


def target_points(raw: Any, game_id: str, limit: int = 3) -> list[dict]:
    state = state_for(raw, game_id)
    targets = [t for t in state.get("targets", [])
               if t.get("coords") and not t.get("is_background_candidate")]
    targets.sort(key=lambda t: (-int(t.get("size", 0)),
                                int(t.get("priority_tier", 4))))
    points = []
    seen = set()
    for idx, target in enumerate(targets, 1):
        row, col = target["centroid"]
        xy = (int(round(col)), int(round(row)))
        if xy in seen:
            continue
        seen.add(xy)
        points.append({
            "label": f"object{idx}", "x": xy[0], "y": xy[1],
            "color": int(target.get("primary_value", -1)),
            "area": int(target.get("size", 0)),
        })
        if len(points) >= limit:
            break
    return points


def free_points(raw: Any, occupied: tuple[int, int] | None = None,
                limit: int = 4) -> list[tuple[int, int]]:
    grid = as_grid(raw)
    background = int(Counter(grid.reshape(-1).tolist()).most_common(1)[0][0])
    candidates = []
    # Coarse, deterministic probes distributed over the board.
    for y in (8, 20, 32, 44, 56):
        for x in (8, 20, 32, 44, 56):
            if int(grid[y, x]) == background and (x, y) != occupied:
                candidates.append((x, y))
    if occupied is not None:
        ox, oy = occupied
        candidates.sort(key=lambda p: (p[0] - ox) ** 2 + (p[1] - oy) ** 2)
    else:
        candidates.sort(key=lambda p: (p[0] - 32) ** 2 + (p[1] - 32) ** 2)
    return candidates[:limit]


def run_action(env: Any, game_id: str, action_id: int,
               xy: tuple[int, int] | None = None) -> tuple[Any, Any, dict]:
    before = env.reset()
    if before is None:
        raise RuntimeError(f"reset failed for {game_id}")
    data = {"x": int(xy[0]), "y": int(xy[1])} if xy is not None else None
    after = env.step(GameAction.from_id(action_id), data=data)
    if after is None:
        raise RuntimeError(f"{action_label(action_id)} failed for {game_id}")
    return before, after, summarize_delta(before, after, game_id)


def format_shift(shift: list) -> str:
    color, (dr, dc) = shift
    return f"color{color}({dr:+d},{dc:+d})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", default=",".join(DEFAULT_STALLED_GAMES),
                        help="Comma-separated game IDs; defaults to the 24 stalled games")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Identical reset-and-action trials for simple actions")
    parser.add_argument("--click-repeats", type=int, default=2,
                        help="Trials for each object-click probe")
    args = parser.parse_args()
    wanted = [g.strip() for g in args.games.split(",") if g.strip()]

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    available = {e.game_id.split("-")[0] for e in arcade.get_environments()}
    missing = [g for g in wanted if g not in available]
    if missing:
        raise SystemExit(f"Games are not cached locally: {', '.join(missing)}")

    out_dir = ROOT / "research" / "system2"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    csv_path = out_dir / f"mechanics_diagnostic_{stamp}.csv"
    jsonl_path = out_dir / f"mechanics_probes_{stamp}.jsonl"
    scorecard_id = f"mechanics-diagnostic-{stamp}"

    summaries: list[dict] = []
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as detail:
        for game_index, game_id in enumerate(wanted, 1):
            print(f"[{game_index}/{len(wanted)}] {game_id}: probing", flush=True)
            env = arcade.make(game_id, seed=0, scorecard_id=scorecard_id)
            if env is None:
                summaries.append({"game_id": game_id, "error": "environment unavailable"})
                continue
            initial = env.reset()
            if initial is None:
                summaries.append({"game_id": game_id, "error": "initial reset failed"})
                continue

            action_ids = sorted({int(a) for a in initial.available_actions})
            points = target_points(initial, game_id)
            row: dict[str, Any] = {
                "game_id": game_id,
                "available_action_count": len(action_ids),
                "available_actions": ",".join(action_label(a) for a in action_ids),
                "initial_state": state_name(initial),
                "initial_levels_completed": int(initial.levels_completed),
                "object_click_targets": ";".join(
                    f"{p['label']}@({p['x']},{p['y']})/c{p['color']}/n{p['area']}" for p in points
                ),
                "action_effects": [],
                "click_effects": [],
                "consistent_movement": [],
                "errors": [],
            }

            # Test every available non-click action from the same seeded reset.
            for action_id in action_ids:
                if action_id in (0, 6):
                    continue
                trials = []
                for repeat in range(max(1, args.repeats)):
                    try:
                        before, after, delta = run_action(env, game_id, action_id)
                        trial = {"repeat": repeat + 1, "delta": delta,
                                 "available_actions_before": list(before.available_actions)}
                        trials.append(trial)
                        detail.write(json.dumps({
                            "game_id": game_id, "probe": "single_action",
                            "action": action_label(action_id), **trial,
                        }, ensure_ascii=True) + "\n")
                    except Exception as exc:  # keep other games diagnostic if one probe fails
                        row["errors"].append(f"{action_label(action_id)}: {exc}")
                movement_counts = movement_trial_counts(trials)
                repeated_moves = [
                    (color, dr, dc, count)
                    for (color, dr, dc), count in movement_counts.items()
                    if count >= 2
                ]
                changes = [t["delta"]["changed_cells"] for t in trials]
                effect = (f"{action_label(action_id)}:changed={sum(c > 0 for c in changes)}/"
                          f"{len(changes)},cells_avg={mean(changes):.1f}") if changes else \
                         f"{action_label(action_id)}:no successful probes"
                if repeated_moves:
                    effect += ",moves=" + ";".join(
                        f"c{c}({dr:+d},{dc:+d})x{n}/{len(trials)}"
                        for c, dr, dc, n in repeated_moves
                    )
                    row["consistent_movement"].append(effect)
                state_events = sorted({
                    t["delta"]["state_after"] for t in trials
                    if t["delta"]["state_after"] not in ("NOT_FINISHED", "NOT_PLAYED")
                })
                if state_events:
                    effect += ",state=" + "/".join(state_events)
                row["action_effects"].append(effect)

            # Click visible non-background objects, independently reset each time.
            if 6 in action_ids:
                for point in points:
                    trials = []
                    for repeat in range(max(1, args.click_repeats)):
                        try:
                            before, after, delta = run_action(
                                env, game_id, 6, (point["x"], point["y"])
                            )
                            trial = {"repeat": repeat + 1, "target": point, "delta": delta}
                            trials.append(trial)
                            detail.write(json.dumps({
                                "game_id": game_id, "probe": "object_click",
                                "action": "ACTION6", **trial,
                            }, ensure_ascii=True) + "\n")
                        except Exception as exc:
                            row["errors"].append(f"click {point['label']}: {exc}")
                    cells = [t["delta"]["changed_cells"] for t in trials]
                    moved = movement_trial_counts(trials)
                    note = (f"{point['label']}@({point['x']},{point['y']}):"
                            f"changed={sum(c > 0 for c in cells)}/{len(cells)}")
                    if moved:
                        note += ",moves=" + ";".join(
                            f"c{c}({dr:+d},{dc:+d})x{n}/{len(trials)}"
                            for (c, dr, dc), n in moved.items() if n >= 2
                        )
                    row["click_effects"].append(note)

                # One generic select-then-place probe: click the largest visible
                # object and follow with nearby empty cells from a fresh reset.
                if points:
                    source = points[0]
                    for destination in free_points(initial, (source["x"], source["y"])):
                        trials = []
                        for repeat in range(max(1, args.click_repeats)):
                            try:
                                start = env.reset()
                                selected = env.step(GameAction.ACTION6,
                                                    data={"x": source["x"], "y": source["y"]})
                                if start is None or selected is None:
                                    raise RuntimeError("reset or selection failed")
                                placed = env.step(GameAction.ACTION6,
                                                  data={"x": destination[0], "y": destination[1]})
                                if placed is None:
                                    raise RuntimeError("destination click failed")
                                delta = summarize_delta(selected, placed, game_id)
                                trial = {"repeat": repeat + 1, "source": source,
                                         "destination": list(destination), "delta": delta}
                                trials.append(trial)
                                detail.write(json.dumps({
                                    "game_id": game_id, "probe": "select_then_click",
                                    "action": "ACTION6,ACTION6", **trial,
                                }, ensure_ascii=True) + "\n")
                            except Exception as exc:
                                row["errors"].append(f"select/place {destination}: {exc}")
                        cells = [t["delta"]["changed_cells"] for t in trials]
                        shifts = movement_trial_counts(trials)
                        note = (f"select@({source['x']},{source['y']})->"
                                f"click@({destination[0]},{destination[1]}):"
                                f"changed={sum(c > 0 for c in cells)}/{len(cells)}")
                        consistent = [
                            (c, dr, dc, n) for (c, dr, dc), n in shifts.items()
                            if n >= 2
                        ]
                        if consistent:
                            note += ",moves=" + ";".join(
                                f"c{c}({dr:+d},{dc:+d})x{n}/{len(trials)}"
                                for c, dr, dc, n in consistent
                            )
                            row["consistent_movement"].append(note)
                        row["click_effects"].append(note)

            row["action_effects"] = " | ".join(row["action_effects"])
            row["click_effects"] = " | ".join(row["click_effects"])
            row["consistent_movement"] = " | ".join(row["consistent_movement"])
            row["errors"] = " | ".join(row["errors"])
            summaries.append(row)
            detail.flush()

    fields = [
        "game_id", "available_action_count", "available_actions", "initial_state",
        "initial_levels_completed", "object_click_targets", "action_effects",
        "click_effects", "consistent_movement", "errors",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)

    print(f"\nWrote summary: {csv_path}")
    print(f"Wrote probe details: {jsonl_path}")
    print("\nGame | action count | actions | repeatable movement / visible click effects")
    for row in summaries:
        print(f"{row.get('game_id')} | {row.get('available_action_count', '?')} | "
              f"{row.get('available_actions', '')} | "
              f"{row.get('consistent_movement') or row.get('click_effects') or 'no repeatable movement detected'}")


if __name__ == "__main__":
    main()
