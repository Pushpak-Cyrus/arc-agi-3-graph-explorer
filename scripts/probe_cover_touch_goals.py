"""Probe generic object-cover and object-touch goal hypotheses.

This is an evaluation-only tool. It learns which palette-free object classes
move under ACTION1-4 reset probes, treats other visible objects as tentative
landmarks, then tracks overlap/touch predicates during a fresh CYRUS rollout.
Only an environment-reported level advance counts as success evidence.

    .venv/Scripts/python.exe scripts/probe_cover_touch_goals.py
"""
from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
import logging
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "vendor" / "ARC-AGI-3-Agents"), str(ROOT / "Cyrus")]

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402

from Cyrus.frame_processor import FrameProcessor  # noqa: E402

GAMES = ("ls20", "re86", "dc22")
ACTION_CAP = 200
OUT = ROOT / "research" / "system2" / "cover_touch_goal_probe_2026-10-29"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_agent():
    spec = importlib.util.spec_from_file_location("cover_touch_probe_agent", ROOT / "agent" / "my_agent.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.MyAgent


def object_rows(raw: Any, processor: FrameProcessor) -> list[dict]:
    state = processor.process(np.asarray(raw.frame))
    rows = []
    background = {
        int(index)
        for target in state.get("targets", [])
        if target.get("is_background_candidate")
        for index in target.get("member_blob_indices", [])
    }
    for index, blob in enumerate(state.get("blobs", [])):
        if index in background:
            continue
        cells = frozenset((int(r), int(c)) for r, c in blob.get("coords", ()))
        if not cells:
            continue
        r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
        r1, c1 = max(r for r, _ in cells), max(c for _, c in cells)
        h, w = r1-r0+1, c1-c0+1
        silhouette = tuple(sorted((r-r0, c-c0) for r, c in cells))
        rows.append({
            "class": (h, w, silhouette),
            "cells": cells,
            "center": (sum(r for r, _ in cells)/len(cells),
                       sum(c for _, c in cells)/len(cells)),
            "area": len(cells),
            "bbox": (r0, c0, r1, c1),
        })
    return rows


def target_rows(raw: Any, processor: FrameProcessor, mobile_classes: set[tuple]) -> list[dict]:
    """Merge touching palette fragments into object/landmark candidates."""
    state = processor.process(np.asarray(raw.frame))
    blobs = state.get("blobs", [])
    rows = []
    for target in state.get("targets", []):
        if target.get("is_background_candidate"):
            continue
        member_indices = [int(i) for i in target.get("member_blob_indices", [])
                          if 0 <= int(i) < len(blobs)]
        members = [blobs[i] for i in member_indices]
        cells = frozenset((int(r), int(c)) for blob in members
                          for r, c in blob.get("coords", ()))
        if not cells:
            continue
        if len(cells) > 300:
            # Some scenes merge a controllable piece into a board-spanning
            # target. Split those groups back into palette-free components so
            # a large scene component cannot swallow every landmark.
            for blob in members:
                blob_cells = frozenset((int(r), int(c)) for r, c in blob.get("coords", ()))
                if not blob_cells:
                    continue
                shape = blob_shape(blob)
                is_mobile = shape in mobile_classes
                if not is_mobile and len(blob_cells) > 300:
                    continue
                r0, c0 = min(r for r, _ in blob_cells), min(c for _, c in blob_cells)
                r1, c1 = max(r for r, _ in blob_cells), max(c for _, c in blob_cells)
                silhouette = tuple(sorted((r-r0, c-c0) for r, c in blob_cells))
                rows.append({
                    "class": (r1-r0+1, c1-c0+1, silhouette),
                    "cells": blob_cells,
                    "center": (sum(r for r, _ in blob_cells)/len(blob_cells),
                               sum(c for _, c in blob_cells)/len(blob_cells)),
                    "area": len(blob_cells), "mobile": is_mobile,
                })
            continue
        mobile = any(blob_shape(blob) in mobile_classes for blob in members)
        if mobile or len(cells) <= 300:
            r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
            r1, c1 = max(r for r, _ in cells), max(c for _, c in cells)
            h, w = r1-r0+1, c1-c0+1
            silhouette = tuple(sorted((r-r0, c-c0) for r, c in cells))
            rows.append({
                "class": (h, w, silhouette), "cells": cells,
                "center": (sum(r for r, _ in cells)/len(cells),
                           sum(c for _, c in cells)/len(cells)),
                "area": len(cells), "mobile": mobile,
            })
    return rows


def blob_shape(blob: dict) -> tuple:
    cells = blob.get("coords", ())
    if not cells:
        return ()
    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
    h = max(r for r, _ in cells) - r0 + 1
    w = max(c for _, c in cells) - c0 + 1
    return (h, w, tuple(sorted((int(r-r0), int(c-c0)) for r, c in cells)))


def moving_classes(arcade, game_id: str) -> tuple[set[tuple], list[dict]]:
    """Infer palette-free object classes that shift under a movement action."""
    probe_env = arcade.make(game_id, seed=0, scorecard_id=f"cover-touch-probe-{game_id}")
    if probe_env is None:
        raise RuntimeError(f"Could not create environment {game_id}")
    initial = probe_env.reset()
    if initial is None:
        raise RuntimeError(f"Could not reset {game_id}")
    action_ids = sorted(int(a) for a in initial.available_actions if int(a) in (1, 2, 3, 4))
    classes: set[tuple] = set()
    probes = []
    for action_id in action_ids:
        before_raw = probe_env.reset()
        before = object_rows(before_raw, FrameProcessor(game_id))
        after_raw = probe_env.step(GameAction.from_id(action_id))
        after = object_rows(after_raw, FrameProcessor(game_id))
        shifts = []
        used = set()
        for old in before:
            matches = [(math.dist(old["center"], new["center"]), j, new)
                       for j, new in enumerate(after)
                       if j not in used and new["class"] == old["class"]]
            if not matches:
                continue
            distance, j, new = min(matches, key=lambda item: item[0])
            if 1.0 <= distance <= 10.0:
                used.add(j)
                classes.add(old["class"])
                shifts.append({"class": old["class"], "from": old["center"],
                               "to": new["center"], "distance": round(distance, 3)})
        probes.append({"action_id": action_id, "shifted_objects": shifts})
    return classes, probes


def object_distance(first: dict, second: dict) -> tuple[int, bool]:
    """Return minimum Manhattan distance and whether the masks overlap."""
    overlap = not first["cells"].isdisjoint(second["cells"])
    if overlap:
        return 0, True
    # Bounded 64x64 object footprints; this pairwise form is small for the
    # capped candidate set and keeps the relation independent of palette.
    dist = min(abs(r1-r2) + abs(c1-c2)
               for r1, c1 in first["cells"]
               for r2, c2 in second["cells"])
    return int(dist), False


def predicate_snapshot(rows: list[dict], landmarks: list[dict]) -> dict:
    movers = [obj for obj in rows if obj["mobile"]]
    pairs = []
    for mover_index, mover in enumerate(movers):
        for landmark_index, landmark in enumerate(landmarks):
            distance, overlap = object_distance(mover, landmark)
            pairs.append({
                "mover_index": mover_index,
                "landmark_index": landmark_index,
                "mover_class": mover["class"],
                "landmark_class": landmark["class"],
                "mover_center": mover["center"],
                "landmark_center": landmark["center"],
                "distance": distance,
                "touch": distance <= 1,
                "cover": overlap,
            })
    return {
        "mover_count": len(movers),
        "landmark_count": len(landmarks),
        "touching_pairs": [p for p in pairs if p["touch"]],
        "covering_pairs": [p for p in pairs if p["cover"]],
        "nearest_pairs": sorted(pairs, key=lambda p: p["distance"])[:10],
    }


def run_game(arcade, AgentClass, game_id: str) -> dict:
    mobile, probes = moving_classes(arcade, game_id)
    memory = OUT / RUN_ID / "fresh_memory" / game_id
    memory.mkdir(parents=True, exist_ok=True)
    os.environ["CYRUS_SYSTEM2_ENABLED"] = "1"
    os.environ["CYRUS_RELATIONAL_PLANNER_ENABLED"] = "0"
    os.environ["CYRUS_LOCAL_LLM_ENABLED"] = "0"
    os.environ["CYRUS_LEARNING_MEMORY"] = str(memory / "learning.json")
    os.environ["CYRUS_GOAL_MEMORY"] = str(memory / "goals.json")
    os.environ["CYRUS_EXPERIMENT_LOG"] = str(memory / "experiments.jsonl")
    os.environ["CYRUS_SYSTEM2_LOG"] = str(memory / "system2.jsonl")
    AgentClass.shared_budget_tracker = None
    AgentClass.shared_interaction_model = None
    AgentClass.MAX_ACTIONS = ACTION_CAP - 1

    env = arcade.make(game_id, seed=0, scorecard_id=f"cover-touch-rollout-{game_id}")
    initial = env.reset() if env is not None else None
    if initial is None:
        raise RuntimeError(f"Could not create/reset {game_id}")
    agent = AgentClass(card_id="cover-touch-probe", game_id=game_id,
                       agent_name=f"CYRUS.cover-touch.{game_id}",
                       ROOT_URL="http://localhost", record=False,
                       arc_env=env, tags=["cover-touch-probe"])
    processor = FrameProcessor(game_id)
    initial_targets = target_rows(initial, processor, mobile)
    landmarks = [obj for obj in initial_targets if not obj["mobile"]]
    initial_snapshot = predicate_snapshot(initial_targets, landmarks)
    trace_path = memory / "predicates.jsonl"
    original_request = agent.do_action_request
    records = []
    previous_raw = {"value": initial}

    def observe_action(action):
        before_raw = previous_raw["value"]
        before = predicate_snapshot(target_rows(before_raw, processor, mobile), landmarks)
        raw_after = original_request(action)
        after = predicate_snapshot(target_rows(raw_after, processor, mobile), landmarks)
        before_levels = int(before_raw.levels_completed)
        after_levels = int(raw_after.levels_completed)
        sig = getattr(agent, "last_action_signature", None)
        row = {
            "game_id": game_id,
            "step": int(agent.action_counter + 1),
            "action_signature": list(sig) if sig is not None else None,
            "levels_before": before_levels,
            "levels_after": after_levels,
            "confirmed_progress": after_levels > before_levels,
            "mobile_class_count": len(mobile),
            "before": before,
            "after": after,
        }
        records.append(row)
        previous_raw["value"] = raw_after
        return raw_after

    agent.do_action_request = observe_action
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        agent.main()
    trace_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
                          encoding="utf-8")

    progress_rows = [r for r in records if r["confirmed_progress"]]
    summary = {
        "game_id": game_id,
        "seed": 0,
        "action_budget": ACTION_CAP,
        "levels_completed": int(agent.frames[-1].levels_completed),
        "actions": int(agent.action_counter),
        "mobile_classes": [list(x) for x in sorted(mobile, key=repr)],
        "mobile_class_count": len(mobile),
        "movement_probes": probes,
        "initial_predicates": initial_snapshot,
        "trajectory_steps": len(records),
        "steps_with_any_touch": sum(bool(r["after"]["touching_pairs"]) for r in records),
        "steps_with_any_cover": sum(bool(r["after"]["covering_pairs"]) for r in records),
        "confirmed_progress_steps": [r["step"] for r in progress_rows],
        "touch_at_confirmed_progress": [bool(r["after"]["touching_pairs"]) for r in progress_rows],
        "cover_at_confirmed_progress": [bool(r["after"]["covering_pairs"]) for r in progress_rows],
        "trace": str(trace_path.relative_to(ROOT)),
    }
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING)
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    available = {e.game_id.split("-")[0] for e in arcade.get_environments()}
    missing = sorted(set(GAMES) - available)
    if missing:
        raise SystemExit(f"Games not cached: {missing}")
    AgentClass = load_agent()
    summaries = []
    for index, game_id in enumerate(GAMES, 1):
        print(f"[{index}/{len(GAMES)}] probing {game_id}", flush=True)
        summaries.append(run_game(arcade, AgentClass, game_id))
        print(f"  levels={summaries[-1]['levels_completed']} actions={summaries[-1]['actions']} "
              f"movers={len(summaries[-1]['mobile_classes'])} "
              f"touch_states={summaries[-1]['steps_with_any_touch']} "
              f"cover_states={summaries[-1]['steps_with_any_cover']}", flush=True)
    payload = {
        "run_id": RUN_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "evaluation_only_unconfirmed_without_environment_progress",
        "goal_family": ["movable object touches a static landmark", "movable object covers a static landmark"],
        "landmark_selection": "non-scene object classes not observed to move under ACTION1-4 reset probes, area <= 300",
        "games": summaries,
    }
    run_out = OUT / RUN_ID
    (run_out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (run_out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["game_id", "seed", "levels_completed", "actions", "mobile_class_count",
                  "steps_with_any_touch", "steps_with_any_cover", "confirmed_progress_steps"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row[key] for key in fields})
    print(f"Wrote {run_out}", flush=True)


if __name__ == "__main__":
    main()
