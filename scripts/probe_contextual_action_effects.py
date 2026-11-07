"""Probe action effects both at reset and after one generic preparatory move.

Some actions can appear ineffective at a boundary while moving the same
objects from an interior state. This controlled probe checks that possibility
without using game IDs, color meanings, or task-specific routes.

Example:
    .venv/Scripts/python.exe scripts/probe_contextual_action_effects.py --game ls20
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
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
from Cyrus.system2.action_effect_learner import ActionEffectLearner  # noqa: E402


def process(raw, game_id: str) -> dict:
    return FrameProcessor(game_id).process(np.asarray(raw.frame))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    known_games = {env.game_id.split("-")[0] for env in arcade.get_environments()}
    if args.game.split("-")[0] not in known_games:
        raise SystemExit(f"{args.game!r} is not cached locally")
    env = arcade.make(
        args.game, seed=args.seed,
        scorecard_id=f"context-action-probe-{date.today().isoformat()}",
    )
    if env is None:
        raise SystemExit(f"Could not open {args.game}")

    initial = env.reset()
    if initial is None:
        raise SystemExit("Initial reset failed")
    action_ids = sorted(int(action) for action in initial.available_actions)
    repeats = max(1, args.repeats)
    results = defaultdict(lambda: {
        "trials": 0, "changed": 0, "shifts": defaultdict(int),
        "states": [], "levels_completed": [],
    })

    # A None preparatory action measures the reset-state baseline. Every other
    # preparation is one available action from reset, applied afresh per trial.
    for target_id in action_ids:
        for prep_id in [None] + [aid for aid in action_ids if aid != target_id]:
            key = (target_id, prep_id)
            bucket = results[key]
            for _ in range(repeats):
                before_prep = env.reset()
                if before_prep is None:
                    raise SystemExit("Reset failed during contextual probe")
                current = before_prep
                if prep_id is not None:
                    current = env.step(GameAction.from_id(prep_id))
                    if current is None:
                        raise SystemExit(f"Preparatory action {prep_id} failed")
                before = process(current, args.game)
                after_raw = env.step(GameAction.from_id(target_id))
                if after_raw is None:
                    raise SystemExit(f"Action {target_id} failed")
                after = process(after_raw, args.game)
                delta = ActionEffectLearner.transition(before, after)
                bucket["trials"] += 1
                bucket["changed"] += int(delta.changed)
                for shape, _object_context, vector in set(
                    delta.object_context_shifts
                ):
                    bucket["shifts"][f"{shape[0]}x{shape[1]}/{len(shape[2])}:{vector}"] += 1
                bucket["states"].append(str(getattr(after_raw.state, "name", after_raw.state)))
                bucket["levels_completed"].append(int(after_raw.levels_completed))

    serial = {}
    for (target_id, prep_id), bucket in sorted(
        results.items(), key=lambda item: (item[0][0], -1 if item[0][1] is None else item[0][1])
    ):
        key = f"action_{target_id}_after_{'reset' if prep_id is None else f'action_{prep_id}'}"
        serial[key] = {
            "action_id": target_id,
            "preparatory_action": prep_id,
            "trials": bucket["trials"],
            "changed_trials": bucket["changed"],
            "repeatable_shifts": {
                shift: count for shift, count in sorted(bucket["shifts"].items())
                if count >= 2 and count / bucket["trials"] >= 0.6
            },
            "all_shift_trial_counts": dict(bucket["shifts"]),
            "levels_completed_max": max(bucket["levels_completed"], default=0),
            "states_seen": sorted(set(bucket["states"])),
        }

    output = {
        "game_id": args.game,
        "seed": args.seed,
        "repeats_per_context": repeats,
        "available_actions": action_ids,
        "contextual_action_effects": serial,
        "interpretation": (
            "Repeated movement after a preparation shows that the action's "
            "effect depends on state. Visual changes and movement are not "
            "level-completion evidence."
        ),
    }
    out_dir = ROOT / "research" / "system2"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"contextual_action_effects_{args.game}_{date.today().isoformat()}"
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    lines = [
        f"# Contextual action-effect probe: `{args.game}` — {date.today().isoformat()}",
        "",
        f"Each action was tested {repeats} times from reset and after each other available action.",
        "A reset-state no-op is not treated as a global no-op if it moves from a different state.",
        "",
        "| Action | Preparation | Changed trials | Repeatable shifts | Max levels completed |",
        "|---|---|---:|---|---:|",
    ]
    for result in serial.values():
        prep = result["preparatory_action"]
        lines.append(
            f"| ACTION{result['action_id']} | "
            f"{'reset' if prep is None else f'ACTION{prep}'} | "
            f"{result['changed_trials']}/{result['trials']} | "
            f"{result['repeatable_shifts'] or 'none'} | "
            f"{result['levels_completed_max']} |"
        )
    lines.extend([
        "",
        "No level completion means these are mechanics observations only.",
        f"Raw records: `{json_path.name}`.",
    ])
    md_path = out_dir / f"{stem}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")
    for result in serial.values():
        prep = result["preparatory_action"]
        print(
            f"ACTION{result['action_id']} after "
            f"{'reset' if prep is None else f'ACTION{prep}'}: "
            f"{result['changed_trials']}/{result['trials']} changed; "
            f"shifts={result['repeatable_shifts'] or 'none'}"
        )
    print("Confirmed levels completed:", max(
        (item["levels_completed_max"] for item in serial.values()), default=0
    ))


if __name__ == "__main__":
    main()
