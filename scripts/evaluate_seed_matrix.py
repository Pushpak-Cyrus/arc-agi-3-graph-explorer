"""Paired five-seed System 2 ablation and same-state random-action baseline.

Runs every cached game with System 2 on and off, a fixed action budget, and
fresh on-disk and in-memory learning state for every run. A disposable deep
copy of the live environment receives one random valid candidate action at
each pre-action state. Its frame-change outcome is used as a same-state
random-action prediction of whether CYRUS's chosen action will change the
frame.

Usage:
    .venv/Scripts/python.exe scripts/evaluate_seed_matrix.py
"""
from __future__ import annotations

import contextlib
import copy
import csv
import importlib.util
import io
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "vendor" / "ARC-AGI-3-Agents"), str(ROOT / "Cyrus")]

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402


SEEDS = (0, 1, 2, 3, 4)
ACTION_BUDGET = 200
OUT = ROOT / "research" / "system2" / "seed_matrix_2026-10-29"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_agent():
    spec = importlib.util.spec_from_file_location("seed_matrix_agent", ROOT / "agent" / "my_agent.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.MyAgent


def branch_action(signature):
    action = GameAction.from_id(int(signature[0]))
    if len(signature) > 2 and signature[1] is not None and signature[2] is not None:
        action.set_data({"x": int(signature[1]), "y": int(signature[2])})
    return action


def run_one(arcade, AgentClass, game_id, seed, system2_on):
    mode = "system2_on" if system2_on else "system2_off"
    run_key = f"{game_id}_seed{seed}_{mode}"
    memory = OUT / RUN_ID / "fresh_memory" / run_key
    memory.mkdir(parents=True, exist_ok=True)
    os.environ["CYRUS_SYSTEM2_ENABLED"] = "1" if system2_on else "0"
    os.environ["CYRUS_RELATIONAL_PLANNER_ENABLED"] = "0"
    os.environ["CYRUS_LOCAL_LLM_ENABLED"] = "0"
    os.environ["CYRUS_LEARNING_MEMORY"] = str(memory / "learning.json")
    os.environ["CYRUS_GOAL_MEMORY"] = str(memory / "goals.json")
    os.environ["CYRUS_EXPERIMENT_LOG"] = str(memory / "experiments.jsonl")
    os.environ["CYRUS_SYSTEM2_LOG"] = str(memory / "system2.jsonl") if system2_on else ""

    AgentClass.shared_budget_tracker = None
    AgentClass.shared_interaction_model = None
    env = arcade.make(game_id, seed=seed, scorecard_id=f"cyrus-matrix-{seed}-{mode}")
    if env is None:
        raise RuntimeError(f"Could not create {game_id}, seed {seed}")
    initial = env.reset()
    if initial is None:
        raise RuntimeError(f"Reset failed: {game_id}, seed {seed}")

    AgentClass.MAX_ACTIONS = ACTION_BUDGET - 1  # framework loop is inclusive
    agent = AgentClass(
        card_id="local-seed-matrix", game_id=game_id,
        agent_name=f"CYRUS.matrix.{run_key}", ROOT_URL="http://localhost",
        record=False, arc_env=env, tags=["seed-matrix"],
    )

    # Capture precisely the candidate signatures available to CYRUS at each
    # state, then evaluate one uniformly sampled candidate in an env clone.
    candidate_box = {"signatures": []}
    original_get_sigs = agent._get_action_signatures
    def capture_candidates(available, state):
        result = original_get_sigs(available, state)
        candidate_box["signatures"] = [tuple(sig) for sig, _tier in result]
        return result
    agent._get_action_signatures = capture_candidates

    rng = random.Random(f"{run_key}:same-state-random")
    branch_agreement = 0
    branch_count = 0
    branch_changed = 0
    actual_changed = 0
    branch_rows = []
    original_request = agent.do_action_request
    def with_random_branch(chosen):
        nonlocal branch_agreement, branch_count, branch_changed, actual_changed
        previous = np.asarray(agent.frames[-1].frame)
        candidates = candidate_box["signatures"]
        # The candidate generator should always include a legal action. Keep
        # a safe action-1 fallback for malformed or empty candidate lists.
        signature = rng.choice(candidates) if candidates else (1, None, None)
        clone = copy.deepcopy(env)
        random_action = branch_action(signature)
        random_raw = clone.step(random_action, data=random_action.action_data.model_dump())
        random_changed = bool(random_raw is not None and not np.array_equal(previous, np.asarray(random_raw.frame)))
        actual_raw = original_request(chosen)
        chosen_changed = not np.array_equal(previous, np.asarray(actual_raw.frame))
        branch_count += 1
        branch_changed += int(random_changed)
        actual_changed += int(chosen_changed)
        branch_agreement += int(random_changed == chosen_changed)
        branch_rows.append({
            "step": int(agent.action_counter + 1),
            "random_changed": random_changed,
            "actual_changed": chosen_changed,
            "random_signature": list(signature),
        })
        return actual_raw
    agent.do_action_request = with_random_branch

    # Quiet per-action logs; the driver emits one progress line per run.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        agent.main()
    with (memory / "random_branches.jsonl").open("w", encoding="utf-8") as stream:
        for branch in branch_rows:
            stream.write(json.dumps(branch, sort_keys=True) + "\n")
    final = agent.frames[-1]
    result = {
        "game_id": game_id,
        "seed": seed,
        "condition": mode,
        "levels_completed": int(final.levels_completed),
        "actions": int(agent.action_counter),
        "final_state": str(final.state),
        "random_branch_states": branch_count,
        "random_branch_changed": branch_changed,
        "actual_changed": actual_changed,
        "random_predictor_correct": branch_agreement,
        "fresh_memory_dir": str(memory.relative_to(ROOT)),
    }
    return result


def summarize(results):
    grouped = defaultdict(list)
    for row in results:
        grouped[(row["game_id"], row["condition"])].append(row)
    per_game = []
    for game_id in sorted({r["game_id"] for r in results}):
        entry = {"game_id": game_id}
        for condition in ("system2_on", "system2_off"):
            rows = grouped[(game_id, condition)]
            entry[f"{condition}_mean_levels"] = sum(r["levels_completed"] for r in rows) / len(rows)
            entry[f"{condition}_total_levels"] = sum(r["levels_completed"] for r in rows)
            entry[f"{condition}_mean_actions"] = sum(r["actions"] for r in rows) / len(rows)
        per_game.append(entry)

    # Model and trivial baselines share the exact System 2-on transitions.
    model_rows = []
    for row in results:
        if row["condition"] != "system2_on":
            continue
        path = ROOT / row["fresh_memory_dir"] / "system2.jsonl"
        if path.exists():
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    validation = record.get("action_effect_validation", {})
                    prediction = validation.get("frame_change_prediction")
                    delta = record.get("delta", {})
                    if prediction is not None and isinstance(delta.get("changed_cells"), (int, float)):
                        model_rows.append({
                            "game_id": row["game_id"], "seed": row["seed"],
                            "step": int(record["step"]),
                            "correct": bool(prediction.get("correct")),
                            "actual_changed": delta["changed_cells"] > 0,
                        })
    random_by_run = {}
    for row in results:
        if row["condition"] != "system2_on":
            continue
        path = ROOT / row["fresh_memory_dir"] / "random_branches.jsonl"
        if path.exists():
            random_by_run[(row["game_id"], row["seed"])] = {
                json.loads(line)["step"]: json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
            }
    matched = [
        (m, random_by_run.get((m["game_id"], m["seed"]), {}).get(m["step"]))
        for m in model_rows
    ]
    matched = [(m, b) for m, b in matched if b is not None]
    n = len(matched)
    model_acc = sum(m["correct"] for m, _b in matched) / n if n else None
    actual_changed_rate = sum(m["actual_changed"] for m, _b in matched) / n if n else None
    random_correct = sum(b["random_changed"] == m["actual_changed"] for m, b in matched)
    random_changed = sum(b["random_changed"] for _m, b in matched)
    all_on_runs = [r for r in results if r["condition"] == "system2_on"]
    random_n_all = sum(r["random_branch_states"] for r in all_on_runs)
    random_correct_all = sum(r["random_predictor_correct"] for r in all_on_runs)
    random_changed_all = sum(r["random_branch_changed"] for r in all_on_runs)
    baseline = {
        "target": "predict whether CYRUS's chosen action changes the observed frame",
        "matched_prediction_rows": n,
        "system2_frame_change_predictor_accuracy": model_acc,
        "always_predict_changed_accuracy": actual_changed_rate,
        "same_state_random_action_accuracy": random_correct / n if n else None,
        "same_state_random_action_frame_change_rate": random_changed / n if n else None,
        "same_state_random_action_branches_on_prediction_rows": n,
        "same_state_random_action_accuracy_all_system2_on_transitions": random_correct_all / random_n_all if random_n_all else None,
        "same_state_random_action_frame_change_rate_all_system2_on_transitions": random_changed_all / random_n_all if random_n_all else None,
        "same_state_random_action_branches_all_system2_on_transitions": random_n_all,
    }
    totals = {}
    for condition in ("system2_on", "system2_off"):
        rows = [r for r in results if r["condition"] == condition]
        totals[condition] = {
            "runs": len(rows),
            "total_levels": sum(r["levels_completed"] for r in rows),
            "mean_levels_per_game_seed": sum(r["levels_completed"] for r in rows) / len(rows),
            "mean_actions_per_run": sum(r["actions"] for r in rows) / len(rows),
        }
    return {"per_game": per_game, "totals": totals, "prediction_baselines": baseline}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING)
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    games = sorted({e.game_id.split("-")[0] for e in arcade.get_environments()})
    AgentClass = load_agent()
    results = []
    total = len(games) * len(SEEDS) * 2
    for seed in SEEDS:
        for game_id in games:
            # Alternate order by seed to avoid always running one condition first.
            modes = (True, False) if seed % 2 == 0 else (False, True)
            for system2_on in modes:
                started = time.perf_counter()
                row = run_one(arcade, AgentClass, game_id, seed, system2_on)
                results.append(row)
                label = "on " if system2_on else "off"
                print(f"[{len(results):3}/{total}] {game_id} seed={seed} S2={label} levels={row['levels_completed']} actions={row['actions']} random={row['random_branch_states']} ({time.perf_counter()-started:.1f}s)", flush=True)
                # Persist after each run so an interruption does not lose work.
                with (OUT / RUN_ID / "runs.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "games": games,
        "seeds": list(SEEDS),
        "action_budget": ACTION_BUDGET,
        "fresh_memory_per_run": True,
        "summary": summarize(results),
    }
    run_out = OUT / RUN_ID
    (run_out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (run_out / "per_game_mean_levels.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["summary"]["per_game"][0]))
        writer.writeheader()
        writer.writerows(summary["summary"]["per_game"])
    print("SUMMARY", json.dumps(summary["summary"], separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
