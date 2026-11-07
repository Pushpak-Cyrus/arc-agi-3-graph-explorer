"""One-layout paired CYRUS evaluation with orthogonal controller flags.

Runs each locally cached game once per condition: System 2 takeover on/off
crossed with System 1 learned-ranking reads on/off. The relational planner
and local LLM are disabled in every headline condition. Every run has fresh
memory files and the same action budget.
"""
from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
import logging
import os
import sys
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "vendor" / "ARC-AGI-3-Agents"),
                str(ROOT / "Cyrus")]

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arc_agi.scorecard import EnvironmentScoreCalculator  # noqa: E402
from arcengine import GameAction  # noqa: E402


ACTION_BUDGET = 200
SEED = 0
DEV_GAMES = {"ls20", "re86", "dc22", "cn04", "ar25"}
CONDITIONS = (
    (False, False, "takeover_off_ranking_off"),
    (True, False, "takeover_on_ranking_off"),
    (False, True, "takeover_off_ranking_on"),
    (True, True, "takeover_on_ranking_on"),
)
OUT = ROOT / "research" / "system2" / "clean_evaluation_2026-10-30"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = OUT / RUN_ID


def load_agent():
    spec = importlib.util.spec_from_file_location(
        "clean_matrix_agent", ROOT / "agent" / "my_agent.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.MyAgent


def layout_hashes(arcade, games):
    hashes = {}
    for game_id in games:
        hashes[game_id] = {}
        for seed in range(5):
            env = arcade.make(
                game_id,
                seed=seed,
                scorecard_id=f"layout-check-{game_id}-{seed}",
            )
            raw = env.reset() if env is not None else None
            if raw is None:
                hashes[game_id][str(seed)] = None
                continue
            frame = np.asarray(raw.frame)
            hashes[game_id][str(seed)] = __import__("hashlib").sha256(
                frame.tobytes()
            ).hexdigest()
    return hashes


def run_one(arcade, agent_class, info_by_short_id, game_id, takeover, ranking,
            label):
    run_key = f"{game_id}_{label}_seed{SEED}"
    memory = RUN_DIR / "fresh_memory" / run_key
    memory.mkdir(parents=True, exist_ok=True)
    os.environ["CYRUS_SYSTEM2_ENABLED"] = "1" if takeover else "0"
    os.environ["CYRUS_LEARNED_RANKING_ENABLED"] = "1" if ranking else "0"
    os.environ["CYRUS_RELATIONAL_PLANNER_ENABLED"] = "0"
    os.environ["CYRUS_LOCAL_LLM_ENABLED"] = "0"
    os.environ["CYRUS_LEARNING_MEMORY"] = str(memory / "learning.json")
    os.environ["CYRUS_GOAL_MEMORY"] = str(memory / "goals.json")
    os.environ["CYRUS_EXPERIMENT_LOG"] = str(memory / "experiments.jsonl")
    os.environ["CYRUS_SYSTEM2_LOG"] = str(memory / "system2.jsonl")

    agent_class.shared_budget_tracker = None
    agent_class.shared_interaction_model = None
    env = arcade.make(
        game_id,
        seed=SEED,
        scorecard_id=f"cyrus-clean-{label}-{SEED}",
    )
    if env is None:
        raise RuntimeError(f"Could not create {game_id}")

    agent_class.MAX_ACTIONS = ACTION_BUDGET - 1
    agent = agent_class(
        card_id=f"clean-{label}", game_id=game_id,
        agent_name=f"CYRUS.clean.{run_key}", ROOT_URL="http://localhost",
        record=False, arc_env=env, tags=["clean-evaluation"],
    )
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        agent.main()

    final = agent.frames[-1]
    records_path = memory / "experiments.jsonl"
    records = [json.loads(line) for line in records_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    completions = []
    for row in records:
        before = row.get("levels_before")
        after = row.get("levels_after")
        if before is not None and after is not None and after > before:
            for level_number in range(int(before) + 1, int(after) + 1):
                completions.append({
                    "level": level_number,
                    "cumulative_actions": int(row["step"]) + 1,
                })
    info = info_by_short_id[game_id]
    baselines = list(info.baseline_actions or [])
    calculator = EnvironmentScoreCalculator(id=game_id)
    prior_actions = 0
    completed = int(final.levels_completed)
    completion_totals = {
        int(row["level"]): int(row["cumulative_actions"])
        for row in completions
    }
    for index, baseline in enumerate(baselines):
        level_number = index + 1
        if level_number <= completed:
            cumulative = completion_totals.get(level_number, prior_actions)
            level_actions = max(0, cumulative - prior_actions)
            level_completed = True
            prior_actions = cumulative
        elif level_number == completed + 1:
            level_actions = max(0, int(agent.action_counter) - prior_actions)
            level_completed = False
        else:
            level_actions = 0
            level_completed = False
        calculator.add_level(
            level_index=level_number,
            completed=level_completed,
            actions_taken=level_actions,
            baseline_actions=int(baseline),
            game_id=game_id,
        )

    click_count = sum(row["action_id"] == 6 for row in records)
    move_count = sum(row["action_id"] in {1, 2, 3, 4, 5} for row in records)
    result = {
        "game_id": game_id,
        "split": "dev" if game_id in DEV_GAMES else "held_out",
        "seed": SEED,
        "condition": label,
        "system2_takeover": bool(takeover),
        "learned_ranking": bool(ranking),
        "relational_planner": False,
        "levels_completed": completed,
        "actions": int(agent.action_counter),
        "actions_to_first_level": agent.first_level_advance_action,
        "click_actions": click_count,
        "movement_actions": move_count,
        "system2_selected_actions": int(agent.system2_selected_actions),
        "learned_ranking_decisions": int(agent.learned_ranking_decisions),
        "learned_ranking_choice_changes": int(agent.learned_ranking_choice_changes),
        "condition_inactive": bool(
            (takeover and agent.system2_selected_actions == 0)
            or (ranking and agent.learned_ranking_choice_changes == 0)
        ),
        "final_state": str(final.state),
        "scorecard_score": float(calculator.to_score().score),
        "baseline_levels": len(baselines),
        "completed_level_actions": completion_totals,
        "fresh_memory_dir": str(memory.relative_to(ROOT)),
    }
    return result


def _accuracy(predictions, labels):
    if not labels:
        return None
    return sum(bool(a == b) for a, b in zip(predictions, labels)) / len(labels)


def prediction_metrics(log_path):
    rows = [json.loads(line) for line in log_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    history_by_signature = {}
    previous_labels = []
    data = []
    for row in rows:
        delta = row.get("action_effect_delta") or {}
        if not isinstance(delta.get("changed_cells"), (int, float)):
            continue
        actual = bool(delta["changed_cells"] > 0)
        sig = tuple(row.get("sig", ()))
        aid = int(sig[0]) if sig else -1
        change_row = (row.get("action_effect_validation") or {}).get(
            "frame_change_prediction"
        ) or {}
        last_outcome = history_by_signature.get(sig)
        majority = (
            sum(previous_labels) / len(previous_labels)
            if previous_labels else None
        )
        model_probability = change_row.get("predicted_probability")
        model_class = change_row.get("predicted_changed")
        data.append({
            "actual": actual,
            "action_id": aid,
            "signature": sig,
            "last_outcome": last_outcome,
            "majority_probability": majority,
            "model_probability": model_probability,
            "model_class": model_class,
        })
        previous_labels.append(actual)
        history_by_signature[sig] = float(actual)

    def brier(name, probability_fn, subset=None):
        selected = [row for row in data if probability_fn(row) is not None
                    and (subset is None or subset(row))]
        if not selected:
            return {"n": 0, "brier": None}
        return {
            "n": len(selected),
            "brier": sum((probability_fn(row) - float(row["actual"])) ** 2
                          for row in selected) / len(selected),
        }

    all_labels = [row["actual"] for row in data]
    majority_class = (
        sum(all_labels) / len(all_labels) >= 0.5 if all_labels else None
    )
    move_click = [row["action_id"] != 6 for row in data]
    model_covered = lambda row: row["model_class"] is not None
    model_rows = [row for row in data if row["model_probability"] is not None]
    calibration_counts = {}
    for row in model_rows:
        probability = min(1.0, max(0.0, float(row["model_probability"])))
        bin_index = min(int(probability * 10), 9)
        bucket = calibration_counts.setdefault(bin_index, [0, 0.0, 0])
        bucket[0] += 1
        bucket[1] += probability
        bucket[2] += int(row["actual"])
    calibration = [
        {
            "lower": round(bin_index / 10, 1),
            "upper": round((bin_index + 1) / 10, 1),
            "n": values[0],
            "mean_probability": values[1] / values[0],
            "observed_rate": values[2] / values[0],
        }
        for bin_index, values in sorted(calibration_counts.items())
    ]
    covered = [row for row in data if model_covered(row)]
    majority_covered = [r for r in covered if r["majority_probability"] is not None]
    majority_data = [r for r in data if r["majority_probability"] is not None]
    def accuracy_entry(selected, predictor):
        return {
            "n": len(selected),
            "accuracy": _accuracy(
                [bool(predictor(row)) for row in selected],
                [row["actual"] for row in selected],
            ),
        }
    return {
        "rows": len(data),
        "model_classification": {
            "n": len(covered),
            "accuracy": _accuracy(
                [bool(row["model_class"]) for row in covered],
                [row["actual"] for row in covered],
            ),
        },
        "accuracy_on_model_covered_rows": {
            "always_changed": accuracy_entry(covered, lambda _r: True),
            "online_majority": accuracy_entry(
                majority_covered, lambda r: r["majority_probability"] >= 0.5,
            ),
            "last_outcome_same_signature": accuracy_entry(
                [r for r in covered if r["last_outcome"] is not None],
                lambda r: r["last_outcome"],
            ),
            "move_vs_click": accuracy_entry(
                covered, lambda r: r["action_id"] != 6,
            ),
        },
        "accuracy_all_available_rows": {
            "always_changed": accuracy_entry(data, lambda _r: True),
            # Descriptive in-sample reference. Kept separate from online
            # majority, which uses only outcomes observed before each row.
            "full_run_majority_class_in_sample": accuracy_entry(
                data, lambda _r: bool(majority_class),
            ),
            "online_majority": accuracy_entry(
                majority_data, lambda r: r["majority_probability"] >= 0.5,
            ),
            "last_outcome_same_signature": accuracy_entry(
                [r for r in data if r["last_outcome"] is not None],
                lambda r: r["last_outcome"],
            ),
            "move_vs_click": accuracy_entry(
                data, lambda r: r["action_id"] != 6,
            ),
        },
        "brier": {
            "learned_probability": brier(
                "model", lambda r: r["model_probability"]
            ),
            "always_changed": brier("always", lambda _r: 1.0),
            "full_run_majority_class_in_sample": brier(
                "majority_class", lambda _r: float(bool(majority_class))
                if majority_class is not None else None,
            ),
            "online_majority": brier("majority", lambda r: r["majority_probability"]),
            "last_outcome_same_signature": brier("last", lambda r: r["last_outcome"]),
            "move_vs_click": brier(
                "heuristic", lambda r: float(r["action_id"] != 6)
            ),
        },
        "brier_on_model_probability_rows": {
            "learned_probability": brier(
                "model", lambda r: r["model_probability"],
                subset=lambda r: r["model_probability"] is not None,
            ),
            "always_changed": brier(
                "always", lambda _r: 1.0,
                subset=lambda r: r["model_probability"] is not None,
            ),
            "online_majority": brier(
                "majority", lambda r: r["majority_probability"],
                subset=lambda r: r["model_probability"] is not None,
            ),
            "last_outcome_same_signature": brier(
                "last", lambda r: r["last_outcome"],
                subset=lambda r: r["model_probability"] is not None,
            ),
            "move_vs_click": brier(
                "heuristic", lambda r: float(r["action_id"] != 6),
                subset=lambda r: r["model_probability"] is not None,
            ),
        },
        "calibration_bins": calibration,
    }


def summarize(results):
    condition_summary = {}
    per_game = []
    game_ids = sorted({row["game_id"] for row in results})
    for _, _, label in CONDITIONS:
        rows = [row for row in results if row["condition"] == label]
        def mean_score(group_rows):
            return (sum(row["scorecard_score"] for row in group_rows)
                    / len(group_rows)) if group_rows else None
        condition_summary[label] = {
            "games": len(rows),
            "total_levels": sum(row["levels_completed"] for row in rows),
            "mean_levels_per_game": (
                sum(row["levels_completed"] for row in rows) / len(rows)
                if rows else None
            ),
            "scorecard_percentage": mean_score(rows),
            "scorecard_percentage_without_ar25": mean_score(
                [row for row in rows if row["game_id"] != "ar25"]
            ),
            "dev_mean_levels": mean_score_levels(
                [row for row in rows if row["split"] == "dev"]
            ),
            "held_out_mean_levels": mean_score_levels(
                [row for row in rows if row["split"] == "held_out"]
            ),
            "dev_scorecard_percentage": mean_score(
                [row for row in rows if row["split"] == "dev"]
            ),
            "held_out_scorecard_percentage": mean_score(
                [row for row in rows if row["split"] == "held_out"]
            ),
        }
    row_index = {(row["game_id"], row["condition"]): row for row in results}
    for game_id in game_ids:
        entry = {"game_id": game_id,
                 "split": "dev" if game_id in DEV_GAMES else "held_out"}
        for _, _, label in CONDITIONS:
            result = row_index[(game_id, label)]
            entry[f"{label}_levels"] = result["levels_completed"]
            entry[f"{label}_scorecard"] = result["scorecard_score"]
            entry[f"{label}_actions"] = result["actions"]
            entry[f"{label}_first_level_action"] = result["actions_to_first_level"]
        per_game.append(entry)
    return {"conditions": condition_summary, "per_game": per_game}


def mean_score_levels(rows):
    return (sum(row["levels_completed"] for row in rows) / len(rows)
            if rows else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games", default=None,
        help="Optional comma-separated subset for a smoke check; default is all cached games.",
    )
    parser.add_argument(
        "--metrics-only", type=Path, default=None,
        help="Recompute frame-change baselines from an existing run directory.",
    )
    args = parser.parse_args()
    if args.metrics_only is not None:
        run_dir = args.metrics_only.resolve()
        results = [json.loads(line) for line in
                   (run_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        baseline_summary = {}
        for _, _, label in CONDITIONS:
            metrics = []
            for result in results:
                if result["condition"] != label:
                    continue
                log_path = ROOT / result["fresh_memory_dir"] / "system2.jsonl"
                metrics.append({
                    "game_id": result["game_id"],
                    **prediction_metrics(log_path),
                })
            baseline_summary[label] = metrics
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["frame_change_baselines"] = baseline_summary
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Updated frame-change baselines: {summary_path}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.WARNING)
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    infos = arcade.get_environments()
    games = sorted({entry.game_id.split("-")[0] for entry in infos})
    if args.games:
        requested = {name.strip().split("-")[0] for name in args.games.split(",")}
        missing = requested - set(games)
        if missing:
            raise SystemExit(f"Games not cached: {sorted(missing)}")
        games = [game for game in games if game in requested]
    info_by_short_id = {entry.game_id.split("-")[0]: entry for entry in infos}
    seeds_by_game = layout_hashes(arcade, games)
    unique_layout_counts = {
        game: len({value for value in seeds.values() if value is not None})
        for game, seeds in seeds_by_game.items()
    }

    agent_class = load_agent()
    results = []
    total = len(games) * len(CONDITIONS)
    for index, game_id in enumerate(games):
        for takeover, ranking, label in CONDITIONS:
            row = run_one(
                arcade, agent_class, info_by_short_id, game_id,
                takeover, ranking, label,
            )
            results.append(row)
            print(
                f"[{len(results):3}/{total}] {game_id} {label}: "
                f"levels={row['levels_completed']} actions={row['actions']} "
                f"score={row['scorecard_score']:.6f}", flush=True,
            )
            with (RUN_DIR / "runs.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")

    baseline_summary = {}
    for _, _, label in CONDITIONS:
        metrics = []
        for result in results:
            if result["condition"] != label:
                continue
            log_path = ROOT / result["fresh_memory_dir"] / "system2.jsonl"
            metrics.append({
                "game_id": result["game_id"],
                **prediction_metrics(log_path),
            })
        baseline_summary[label] = metrics

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "games": games,
        "development_games": sorted(DEV_GAMES),
        "held_out_games": sorted(set(games) - DEV_GAMES),
        "seeds_evaluated": [SEED],
        "layout_hashes_seeds_0_to_4": seeds_by_game,
        "unique_layouts_per_game": unique_layout_counts,
        "effective_layout_n": {
            "min": min(unique_layout_counts.values()) if unique_layout_counts else 0,
            "max": max(unique_layout_counts.values()) if unique_layout_counts else 0,
        },
        "action_budget": ACTION_BUDGET,
        "fresh_memory_per_run": True,
        "relational_planner_enabled": False,
        "local_llm_enabled": False,
        "summary": summarize(results),
        "frame_change_baselines": baseline_summary,
    }
    (RUN_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (RUN_DIR / "per_game.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(summary["summary"]["per_game"][0])
        )
        writer.writeheader()
        writer.writerows(summary["summary"]["per_game"])
    print(f"RUN_DIR={RUN_DIR}", flush=True)
    print(json.dumps(summary["summary"]["conditions"], separators=(",", ":")),
          flush=True)


if __name__ == "__main__":
    main()
