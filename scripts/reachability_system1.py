"""Measure plain CYRUS System 1 reachability at 2k and 5k action budgets.

Each game/budget pair gets seed 0 and fresh memory. System 2 takeover,
learned ranking, relational planning, and the local LLM are disabled. RESET
calls count toward the scorecard and run budget, matching the ARC wrapper.
"""
from __future__ import annotations

import contextlib
import argparse
import csv
import importlib.util
import io
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "vendor" / "ARC-AGI-3-Agents"),
                str(ROOT / "Cyrus")]

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arc_agi.scorecard import EnvironmentScoreCalculator  # noqa: E402

SEED = 0
BUDGETS = (2000, 5000)
OUT = Path(os.getenv(
    "CYRUS_REACHABILITY_OUT",
    str(ROOT / "research" / "system2" / "2026-11-02" / "phase1b"),
))
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = OUT / RUN_ID
SEARCH_ORDER = os.getenv("CYRUS_SEARCH_ORDER", "0").strip().lower() in {
    "1", "true", "yes", "on"
}


def load_agent():
    spec = importlib.util.spec_from_file_location(
        "reachability_agent", ROOT / "agent" / "my_agent.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.MyAgent


def reconstructed_score(baselines, levels_completed, trace, total_actions, game_id):
    """Reconstruct the scorecard from confirmed level edges and counted actions."""
    calculator = EnvironmentScoreCalculator(id=game_id)
    completion_actions = {}
    previous_level = 0
    for row in trace:
        reached = int(row.get("levels_completed") or 0)
        if reached > previous_level:
            for level in range(previous_level + 1, reached + 1):
                completion_actions[level] = int(row["index"])
            previous_level = reached
    prior_actions = 0
    for index, baseline in enumerate(baselines, start=1):
        if index <= levels_completed:
            cumulative = completion_actions.get(index, prior_actions)
            actions = max(0, cumulative - prior_actions)
            completed = True
            prior_actions = cumulative
        elif index == levels_completed + 1:
            actions = max(0, total_actions - prior_actions)
            completed = False
        else:
            actions = 0
            completed = False
        calculator.add_level(
            level_index=index,
            completed=completed,
            actions_taken=actions,
            baseline_actions=int(baseline),
            game_id=game_id,
        )
    return float(calculator.to_score().score)


def reset_trace_metrics(trace):
    """Derive reset categories from captured actions and resulting frames."""
    game_over_resets = sum(
        row.get("action_id") == 0
        and row.get("selection_reason") == "game_over_reset"
        for row in trace
    )
    frontier_resets = sum(
        row.get("action_id") == 0
        and row.get("selection_reason") == "system1_frontier_reset"
        for row in trace
    )
    zero_length_attempts = 0
    full_game_resets = 0
    previous_action = None
    previous_state = None
    previous_level = 0
    actions_since_level_start = 0
    for row in trace:
        action_id = int(row.get("action_id", -1))
        levels_completed = int(row.get("levels_completed") or 0)
        if action_id == 0:
            if (previous_action is not None
                    and previous_state != "GAME_OVER"
                    and (previous_action == 0 or actions_since_level_start == 0)):
                zero_length_attempts += 1
            if previous_level > 0 and levels_completed == 0:
                full_game_resets += 1
            actions_since_level_start = 0
        elif levels_completed > previous_level:
            actions_since_level_start = 0
        else:
            actions_since_level_start += 1
        previous_action = action_id
        previous_state = row.get("state")
        previous_level = levels_completed
    return {
        "game_over_resets": int(game_over_resets),
        "frontier_resets": int(frontier_resets),
        "zero_length_attempt_count": int(zero_length_attempts),
        "full_game_resets": int(full_game_resets),
    }


def run_one(arcade, infos, agent_class, game_id, budget):
    run_key = f"{game_id}_system1_seed{SEED}_cap{budget}"
    # Use absolute paths for trace capture; ROOT is resolved and pathlib will
    # reject relative_to(ROOT) when RUN_DIR itself came from a relative env var.
    memory = (RUN_DIR / "fresh_memory" / run_key).resolve()
    memory.mkdir(parents=True, exist_ok=True)
    scorecard_id = arcade.create_scorecard()
    os.environ.update({
        "CYRUS_SYSTEM2_ENABLED": "0",
        "CYRUS_FRONTIER_MODE": "1",
        "CYRUS_LEARNED_RANKING_ENABLED": "0",
        "CYRUS_RELATIONAL_PLANNER_ENABLED": "0",
        "CYRUS_LOCAL_LLM_ENABLED": "0",
        "CYRUS_LEARNING_MEMORY": str(memory / "learning.json"),
        "CYRUS_GOAL_MEMORY": str(memory / "goals.json"),
        "CYRUS_EXPERIMENT_LOG": str(memory / "experiments.jsonl"),
        "CYRUS_SYSTEM2_LOG": str(memory / "system2.jsonl"),
        "CYRUS_NOOP_DEPRIORITIZE_AFTER": "3",
        "CYRUS_SEARCH_ORDER": "1" if SEARCH_ORDER else "0",
        # The optional search order starts at D=6 and widens to 12/24/50.
        # Bug-fix-only runs retain the previous D=50 attempt cap.
        "CYRUS_FRONTIER_ATTEMPT_CAP": "6" if SEARCH_ORDER else "50",
    })
    random.seed(SEED)
    agent_class.shared_budget_tracker = None
    agent_class.shared_interaction_model = None
    env = arcade.make(game_id, seed=SEED, scorecard_id=scorecard_id)
    if env is None:
        raise RuntimeError(f"Could not create {game_id}")

    agent_class.MAX_ACTIONS = budget - 1  # the vendored loop is inclusive
    agent = agent_class(
        card_id=f"reach-{run_key}", game_id=game_id,
        agent_name=f"CYRUS.reach.{run_key}", ROOT_URL="http://localhost",
        record=False, arc_env=env, tags=["system1-reachability"],
    )
    # The System 1-only condition must not spend time building latent System 2
    # models. These observers cannot alter the graph policy, but disabling them
    # makes the reachability controller an actual plain graph explorer.
    agent.system2.observe = lambda **_kwargs: None
    agent.system2.watch = lambda *_args, **_kwargs: None

    trace = []
    original_take_action = agent.take_action

    def capture_action(action):
        frame = original_take_action(action)
        data = action.action_data.model_dump() if action.is_complex() else {}
        trace.append({
            "index": len(trace) + 1,
            "action": action.name,
            "action_id": int(action.value),
            "data": data,
            "state": frame.state.name if frame else "EXCEPTION",
            "levels_completed": frame.levels_completed if frame else None,
            "full_reset": bool(getattr(frame, "full_reset", False)) if frame else False,
            "selection_reason": getattr(agent, "last_selection_reason", ""),
        })
        return frame

    agent.take_action = capture_action
    exception = None
    started = time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        try:
            agent.main()
        except Exception as exc:  # preserve run evidence and continue the matrix
            exception = f"{type(exc).__name__}: {exc}"

    runtime_seconds = time.monotonic() - started
    final = agent.frames[-1]
    with (memory / "action_trace.jsonl").open("w", encoding="utf-8") as stream:
        for row in trace:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    with (memory / "frontier_attempts.jsonl").open("w", encoding="utf-8") as stream:
        for row in agent.frontier_attempts:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    if agent.levels_completed > 0:
        with (memory / "winning_trace.jsonl").open("w", encoding="utf-8") as stream:
            for row in trace:
                stream.write(json.dumps(row, sort_keys=True) + "\n")

    end_reason = "exception" if exception else None
    if end_reason is None:
        if final.state.name == "WIN":
            end_reason = "WIN"
        elif agent.action_counter >= budget:
            end_reason = "action_cap"
        elif agent.consecutive_reset_stalls >= 2:
            end_reason = "reset_stalled_after_repeated_GAME_OVER"
        elif final.state.name == "GAME_OVER":
            end_reason = "GAME_OVER"
        else:
            end_reason = "harness_or_agent_limit"

    info = next(item for item in infos
                if item.game_id.split("-")[0] == game_id)
    first_level = agent.first_level_advance_action
    official_card = arcade.get_scorecard(scorecard_id)
    if official_card is None:
        raise RuntimeError(f"Offline scorecard unavailable for {game_id}")
    official_environment = next(
        (entry for entry in official_card.environments
         if entry.id.split("-")[0] == game_id), None
    )
    if official_environment is None:
        raise RuntimeError(f"No scorecard environment entry for {game_id}")
    score = float(official_environment.score)
    reset_count = sum(1 for row in trace if row.get("action_id") == 0)
    reset_metrics = reset_trace_metrics(trace)
    peak_levels_completed = max(
        (int(row.get("levels_completed") or 0) for row in trace), default=0
    )
    replay_actions = sum(
        row.get("selection_reason") == "system1_frontier_replay"
        for row in trace
    )
    new_experiments = sum(
        row.get("action_id") != 0
        and row.get("selection_reason") != "system1_frontier_replay"
        for row in trace
    )
    attempts_by_reset = []
    replay_lengths_by_reset = []
    segment = []
    segment_replay_actions = 0
    for row in trace:
        if row.get("action_id") == 0:
            if segment:
                attempts_by_reset.append(len(segment))
                replay_lengths_by_reset.append(segment_replay_actions)
            segment = []
            segment_replay_actions = 0
        else:
            segment.append(row)
            if row.get("selection_reason") == "system1_frontier_replay":
                segment_replay_actions += 1
    if segment:
        attempts_by_reset.append(len(segment))
        replay_lengths_by_reset.append(segment_replay_actions)
    result = {
        "game_id": game_id,
        "split": "dev" if game_id in {"ar25", "cn04", "dc22", "ls20", "re86"}
        else "held_out",
        "seed": SEED,
        "budget": budget,
        "system2_takeover": False,
        "frontier_mode": True,
        "search_order": SEARCH_ORDER,
        "learned_ranking": False,
        "relational_planner": False,
        "levels_completed": int(final.levels_completed),
        "peak_levels_completed": peak_levels_completed,
        "official_scorecard_levels_completed": int(official_environment.levels_completed),
        "first_level_action": first_level,
        "scorecard_score": score,
        "actions_including_resets": int(agent.action_counter),
        "resets": reset_count,
        **reset_metrics,
        "average_replay_length_per_attempt": (
            sum(replay_lengths_by_reset) / len(replay_lengths_by_reset)
            if replay_lengths_by_reset else 0.0
        ),
        "replay_attempt_intervals": len(replay_lengths_by_reset),
        "final_state": final.state.name,
        "end_reason": end_reason,
        "exception": exception,
        "trace_path": str((memory / "winning_trace.jsonl").relative_to(ROOT))
        if agent.levels_completed > 0 else None,
        "action_trace_path": str((memory / "action_trace.jsonl").relative_to(ROOT)),
        "memory_dir": str(memory.relative_to(ROOT)),
        "environment_baseline_actions": list(info.baseline_actions or []),
        "runtime_seconds": runtime_seconds,
        "unique_states": len(agent.graph.nodes),
        "state_action_pairs_tried": sum(
            edge.times_tested
            for node in agent.graph.nodes.values()
            for edge in node.actions.values()
        ),
        "frontier_replay_actions": sum(
            edge.replay_count
            for node in agent.graph.nodes.values()
            for edge in node.actions.values()
        ),
        "unique_states_per_1000_actions": (
            1000.0 * len(agent.graph.nodes) / max(1, len(trace))
        ),
        "new_experiments": new_experiments,
        "experiments_per_reset": new_experiments / max(1, reset_count),
        "replay_actions": replay_actions,
        "replay_fraction": replay_actions / max(1, len(trace)),
        "attempt_lengths": attempts_by_reset,
        "attempt_count": len(agent.frontier_attempts),
        "zero_length_attempt_count": reset_metrics["zero_length_attempt_count"],
        "internal_zero_length_frontier_attempt_count": int(
            agent.zero_length_frontier_attempts
        ),
        "no_op_action_signatures": len(agent.action_noop_states),
        "frontier_root_hash": agent.frontier_root_hash,
    }
    return result


def write_csv(rows, path):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", default=None,
                        help="Optional comma-separated cached game subset.")
    parser.add_argument("--budgets", default="2000,5000",
                        help="Comma-separated action budgets (default: 2000,5000).")
    parser.add_argument("--search-order", action="store_true",
                        help="Enable CYRUS_SEARCH_ORDER=1 for iterative deepening/widening.")
    args = parser.parse_args()
    global SEARCH_ORDER
    SEARCH_ORDER = bool(args.search_order)
    budgets = tuple(int(value.strip()) for value in args.budgets.split(",")
                    if value.strip())
    if not budgets or any(value < 1 for value in budgets):
        raise SystemExit("Budgets must be positive integers")
    OUT.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.WARNING)
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    infos = arcade.get_environments()
    games = sorted({item.game_id.split("-")[0] for item in infos})
    if args.games:
        requested = {value.strip().split("-")[0]
                     for value in args.games.split(",") if value.strip()}
        missing = requested - set(games)
        if missing:
            raise SystemExit(f"Games not cached: {sorted(missing)}")
        games = [game for game in games if game in requested]
    agent_class = load_agent()
    results = []
    for budget in budgets:
        for index, game_id in enumerate(games, start=1):
            try:
                row = run_one(arcade, infos, agent_class, game_id, budget)
            except Exception as exc:
                row = {
                    "game_id": game_id,
                    "split": "dev" if game_id in {"ar25", "cn04", "dc22", "ls20", "re86"}
                    else "held_out",
                    "seed": SEED,
                    "budget": budget,
                    "levels_completed": None,
                    "first_level_action": None,
                    "scorecard_score": None,
                    "actions_including_resets": None,
                    "resets": None,
                    "final_state": None,
                    "end_reason": "exception",
                    "exception": f"{type(exc).__name__}: {exc}",
                    "trace_path": None,
                    "memory_dir": None,
                }
            results.append(row)
            with (RUN_DIR / "runs.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            write_csv(results, RUN_DIR / "runs.csv")
            print(
                f"[{budget} {index:02}/{len(games)}] {game_id}: "
                f"levels={row['levels_completed']} "
                f"first={row['first_level_action']} "
                f"actions={row['actions_including_resets']} "
                f"resets={row['resets']} "
                f"full_game_resets={row.get('full_game_resets', 'NA')} "
                f"zero_length_attempts="
                f"{row.get('zero_length_attempt_count', 'NA')} "
                f"end={row['end_reason']}",
                flush=True,
            )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "search_order": SEARCH_ORDER,
        "budgets": list(budgets),
        "games": games,
        "development_games": ["ar25", "cn04", "dc22", "ls20", "re86"],
        "headline_metric": "environment-confirmed levels_completed and scorecard_score",
        "resets_count_toward_scorecard_actions": True,
        "system2_learning_observers_disabled": True,
        "results": results,
    }
    (RUN_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    commit = "unknown"
    try:
        import subprocess
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        pass
    (RUN_DIR / "README.md").write_text(
        "# System 1 frontier acceptance run\n\n"
        f"- Created UTC: {summary['created_utc']}\n"
        f"- Seed: {SEED}\n- Budgets: {budgets}\n"
        f"- Git commit: {commit}\n"
        "- System 2 takeover, learned ranking, relational planner, and local LLM: off.\n"
        "- Fresh memory per game; RESET actions count toward budget and scorecard.\n"
        "- `CYRUS_FRONTIER_MODE=1` uses multi-action attempts and resets only "
        "at a local dead end or the per-attempt cap.\n"
        f"- Command: `python scripts/reachability_system1.py --games "
        f"{','.join(games)} --budgets {','.join(map(str, budgets))}`\n"
        "- Acceptance fixed before evaluation: first-level completion within "
        "50% of historical actions (cd82 <=862, sp80 <=714, tu93 <=1751).\n"
        "- Full data: `runs.csv`, `runs.jsonl`, `summary.json`; complete "
        "action traces are under each `fresh_memory/` directory.\n",
        encoding="utf-8",
    )
    print(f"REPORT_DIR={RUN_DIR}", flush=True)


if __name__ == "__main__":
    main()
