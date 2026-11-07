"""Learn reset-repeatable action effects and compare generic goal guesses.

This is a controlled mechanics experiment, not a solver. Every tested action
starts from the same seeded scene; the learner itself has no game-ID rules.

Usage:
    .venv/Scripts/python.exe scripts/learn_action_effects.py --game ls20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from collections import defaultdict
from datetime import date
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402

from Cyrus.frame_processor import FrameProcessor  # noqa: E402
from Cyrus.system2 import ActionEffectLearner, GoalInference  # noqa: E402
from Cyrus.system2.action_effect_learner import (  # noqa: E402
    _center,
    _non_background_blobs,
    _shape_key,
)


def process(raw, game_id: str) -> dict:
    return FrameProcessor(game_id).process(np.asarray(raw.frame))


def contact_shape_metrics(state: dict) -> dict:
    """Generic visual indicators for contact, overlap, and matching shapes."""
    overlap_pairs = {
        tuple(sorted((int(rel["left"]), int(rel["right"]))))
        for rel in state.get("relations", ()) if rel.get("kind") == "overlap"
    }
    by_shape = defaultdict(list)
    for blob in _non_background_blobs(state):
        key = _shape_key(blob)
        if key is not None:
            by_shape[key].append(_center(blob))
    shape_distances = {}
    for key, centers in by_shape.items():
        if len(centers) < 2:
            continue
        distance = min(math.dist(a, b) for a, b in combinations(centers, 2))
        height, width, cells = key
        shape_id = f"{height}x{width}:{len(cells)}cells"
        shape_distances[shape_id] = round(distance, 3)
    return {
        "overlap_pair_count": len(overlap_pairs),
        "repeated_shape_pair_distances": shape_distances,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    if args.game.split("-")[0] not in {
        e.game_id.split("-")[0] for e in arcade.get_environments()
    }:
        raise SystemExit(f"{args.game!r} is not cached locally")
    env = arcade.make(args.game, seed=args.seed,
                      scorecard_id=f"action-effect-learning-{date.today().isoformat()}")
    if env is None:
        raise SystemExit(f"Could not open {args.game}")

    initial_raw = env.reset()
    if initial_raw is None:
        raise SystemExit("Initial reset failed")
    initial_state = process(initial_raw, args.game)
    action_ids = sorted({int(a) for a in initial_raw.available_actions if int(a) != 6})
    learner = ActionEffectLearner(min_trials=2, min_share=0.6)
    goals = GoalInference(memory_path=ROOT / "research" / "system2" /
                          f"temporary_goal_memory_{args.game}.json")
    goals.ensure_goals(initial_state)
    baseline_potential = goals.potential(initial_state)
    baseline_relations = contact_shape_metrics(initial_state)

    observations = []
    relation_deltas = defaultdict(list)
    changed_masks = defaultdict(list)
    after_states = defaultdict(list)
    for action_id in action_ids:
        for repeat in range(1, max(1, args.repeats) + 1):
            before_raw = env.reset()
            if before_raw is None:
                raise SystemExit(f"Reset failed before action {action_id}")
            before = process(before_raw, args.game)
            action = GameAction.from_id(action_id)
            after_raw = env.step(action)
            if after_raw is None:
                raise SystemExit(f"Action {action_id} failed")
            after = process(after_raw, args.game)
            delta = learner.observe(action_id, before, after)

            before_grid = np.asarray(before["grid"])[-1]
            after_grid = np.asarray(after["grid"])[-1]
            changed_masks[action_id].append(set(map(
                tuple, np.argwhere(before_grid != after_grid).tolist()
            )))
            after_states[action_id].append(after)
            after_relations = contact_shape_metrics(after)
            before_distances = baseline_relations["repeated_shape_pair_distances"]
            after_distances = after_relations["repeated_shape_pair_distances"]
            common_shapes = before_distances.keys() & after_distances.keys()
            relation_deltas[action_id].append({
                "overlap_count_change": (
                    after_relations["overlap_pair_count"]
                    - baseline_relations["overlap_pair_count"]
                ),
                "same_shape_distance_changes": {
                    shape: round(after_distances[shape] - before_distances[shape], 3)
                    for shape in sorted(common_shapes)
                },
            })
            observations.append({
                "action_id": action_id,
                "repeat": repeat,
                "changed_cells": delta.changed_cells,
                "object_effects": [
                    {"shape": shape, "vector": vector}
                    for shape, vector in delta.shifts
                ],
                "state": getattr(after_raw.state, "name", str(after_raw.state)),
                "levels_completed": int(after_raw.levels_completed),
            })

    # A cell changed by every distinct action from the same initial scene is
    # a possible shared UI/tick response. Keep both raw and neutralized goal
    # deltas so it cannot silently become evidence for a task objective.
    common_cells_by_action = [
        set.intersection(*trials) for trials in changed_masks.values() if trials
    ]
    shared_cells = (
        set.intersection(*common_cells_by_action) if common_cells_by_action else set()
    )
    potential_deltas = defaultdict(lambda: defaultdict(list))
    raw_potential_deltas = defaultdict(lambda: defaultdict(list))
    baseline_grid = np.asarray(initial_state["masked_grid"]).copy()
    for action_id, states in after_states.items():
        for state in states:
            raw_potential = goals.potential(state)
            for name in baseline_potential.keys() & raw_potential.keys():
                raw_potential_deltas[action_id][name].append(
                    raw_potential[name] - baseline_potential[name]
                )

            adjusted = dict(state)
            adjusted_grid = np.asarray(state["masked_grid"]).copy()
            for row, col in shared_cells:
                adjusted_grid[..., row, col] = baseline_grid[..., row, col]
            adjusted["masked_grid"] = adjusted_grid
            adjusted["grid"] = adjusted_grid
            adjusted["hash"] = hashlib.md5(
                adjusted_grid.astype(np.uint8).tobytes()
            ).hexdigest()
            adjusted_potential = goals.potential(adjusted)
            for name in baseline_potential.keys() & adjusted_potential.keys():
                potential_deltas[action_id][name].append(
                    adjusted_potential[name] - baseline_potential[name]
                )

    goal_results = {}
    for action_id in action_ids:
        def ranked_changes(source):
            changes = []
            for name, values in source[action_id].items():
                avg = float(np.mean(values))
                if abs(avg) > 1e-9:
                    changes.append({"goal": name, "mean_potential_change": round(avg, 6)})
            return sorted(changes,
                          key=lambda item: abs(item["mean_potential_change"]),
                          reverse=True)

        changes = ranked_changes(potential_deltas)
        raw_changes = ranked_changes(raw_potential_deltas)
        rel_trials = relation_deltas[action_id]
        overlap_changes = [r["overlap_count_change"] for r in rel_trials]
        shared_shape_deltas = defaultdict(list)
        for result in rel_trials:
            for shape, change in result["same_shape_distance_changes"].items():
                shared_shape_deltas[shape].append(change)
        goal_results[str(action_id)] = {
            "top_color_contact_or_clear_potential_changes": changes[:8],
            "unadjusted_top_color_contact_or_clear_potential_changes": raw_changes[:8],
            "overlap_count_changes_by_trial": overlap_changes,
            "mean_overlap_count_change": round(float(np.mean(overlap_changes)), 3)
            if overlap_changes else 0.0,
            "same_shape_distance_change_by_trial_mean": {
                shape: round(float(np.mean(values)), 3)
                for shape, values in shared_shape_deltas.items()
            },
        }

    payload = {
        "game_id": args.game,
        "seed": args.seed,
        "repeats_per_action": max(1, args.repeats),
        "available_actions": action_ids,
        "initial_levels_completed": int(initial_raw.levels_completed),
        "initial_visual_metrics": baseline_relations,
        "shared_changed_cells_across_actions": [list(point) for point in sorted(shared_cells)],
        "learned_action_effects": learner.summary(),
        "generic_goal_probe_results": goal_results,
        "observations": observations,
        "interpretation_note": (
            "Visual goal metrics are hypotheses only. No confirmed level advance "
            "means none of these metrics is validated as the task objective."
        ),
    }

    out_dir = ROOT / "research" / "system2"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    json_path = out_dir / f"action_effect_learning_{args.game}_{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved controlled probe results: {json_path}")
    print(f"Game {args.game}; available actions: {action_ids}")
    for action_id, result in learner.summary().items():
        print(f"ACTION{action_id}: {result['changed_trials']}/{result['trials']} "
              f"changed; learned effects={len([e for e in result['effects'] if e['repeatable']])}")
        for effect in learner.predict(int(action_id)):
            h, w, cells = effect["shape"]
            print(f"  shape={h}x{w}/{len(cells)} cells; vector={effect['vector']}; "
                  f"repeated={effect['trials_with_effect']}/{effect['trials']}")
    print("Generic goal-probe summaries (potential changes are unvalidated):")
    print(f"Shared cells changed by every tested action: {sorted(shared_cells)}")
    for action_id, result in goal_results.items():
        top = result["top_color_contact_or_clear_potential_changes"][:3]
        print(f"  ACTION{action_id}: contact/clear={top}; "
              f"overlap_delta={result['mean_overlap_count_change']}; "
              f"same-shape-distance={result['same_shape_distance_change_by_trial_mean']}")
    completed = max((item["levels_completed"] for item in observations), default=0)
    print(f"Confirmed levels completed during one-step probes: {completed}")


if __name__ == "__main__":
    main()
