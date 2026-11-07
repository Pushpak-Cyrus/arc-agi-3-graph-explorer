"""Bounded development-only environment oracle search.

This tool deep-copies cached ARC environments and searches valid action
signatures for a level-1 transition. It is diagnostic only and is not
imported by CYRUS or used in an evaluation score.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "Cyrus")]

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import ActionInput, GameAction  # noqa: E402
from frame_processor import FrameProcessor  # noqa: E402
from system2.goal_inference import GoalInference  # noqa: E402

DEV_GAMES = ("ar25", "cn04", "dc22", "ls20", "re86")
BENCHMARK_GAMES = ("cd82", "sp80", "tu93")
OUT = ROOT / "research" / "system2" / "oracle_dev_2026-10-30"


def last_frame(raw):
    if not raw.frame:
        return np.empty((0, 0), dtype=np.int16)
    return np.asarray(raw.frame[-1])


def state_key(game, raw):
    frame = last_frame(raw)
    return hashlib.sha256(frame.tobytes()).hexdigest()


def click_signatures(raw, game_id):
    """Use observed object-component centers as bounded click signatures."""
    frame = last_frame(raw)
    if frame.size == 0:
        return []
    state = FrameProcessor(game_id).process(frame)
    signatures = set()
    occupied = set()
    grid = np.asarray(state.get("grid"))
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[int(np.argmax(counts))]) if len(values) else None
    for blob in state.get("blobs", ()):
        if blob.get("is_background_candidate"):
            continue
        coords = blob.get("coords", ())
        if not coords:
            continue
        occupied.update((int(row), int(col)) for row, col in coords)
        y = int(round(sum(int(row) for row, _ in coords) / len(coords)))
        x = int(round(sum(int(col) for _, col in coords) / len(coords)))
        signatures.add((GameAction.ACTION6.value, x, y))
        if len(coords) <= 9:
            signatures.update(
                (GameAction.ACTION6.value, int(col), int(row))
                for row, col in coords
            )
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            row, col = y + dy, x + dx
            if (grid.ndim == 2 and 0 <= row < grid.shape[0]
                    and 0 <= col < grid.shape[1]
                    and (row, col) not in occupied
                    and (background is None or int(grid[row, col]) == background)):
                signatures.add((GameAction.ACTION6.value, col, row))
    if grid.ndim == 2 and grid.shape[0] >= 64 and grid.shape[1] >= 64:
        for y in (8, 24, 40, 56):
            for x in (8, 24, 40, 56):
                if ((y, x) not in occupied
                        and (background is None or int(grid[y, x]) == background)):
                    signatures.add((GameAction.ACTION6.value, x, y))
        # Generic edge-row/edge-column probes, bounded to center and corners.
        ys = sorted({0, grid.shape[0] // 2, grid.shape[0] - 1})
        xs = sorted({0, grid.shape[1] // 2, grid.shape[1] - 1})
        sample_x = sorted({round(i * (grid.shape[1] - 1) / 7) for i in range(8)})
        sample_y = sorted({round(i * (grid.shape[0] - 1) / 7) for i in range(8)})
        for y in ys:
            for x in sample_x:
                signatures.add((GameAction.ACTION6.value, x, y))
        for x in xs:
            for y in sample_y:
                signatures.add((GameAction.ACTION6.value, x, y))
    return sorted(signatures)


def available_signatures(game, raw, game_id):
    signatures = set()
    for value in raw.available_actions or ():
        action_id = int(value)
        if action_id == GameAction.RESET.value:
            continue
        if action_id == GameAction.ACTION6.value:
            signatures.update(click_signatures(raw, game_id))
        else:
            signatures.add((action_id, None, None))
    return sorted(signatures, key=lambda sig: (sig[0], -1 if sig[1] is None else sig[1],
                                                -1 if sig[2] is None else sig[2]))


def apply_signature(game, signature, advance_capture=None):
    action_id, x, y = signature
    data = {"x": int(x), "y": int(y)} if action_id == GameAction.ACTION6.value else {}
    action = ActionInput(id=GameAction.from_id(int(action_id)), data=data)
    original_next_level = None
    if advance_capture is not None:
        original_next_level = game.next_level

        def capture_before_level_change():
            frame = game.camera.render(game.current_level.get_sprites())
            advance_capture.append(np.asarray(frame).copy())
            return original_next_level()

        game.next_level = capture_before_level_change
    try:
        return game.perform_action(action, raw=True)
    finally:
        if original_next_level is not None:
            game.next_level = original_next_level


def object_components(frame, game_id):
    """Compact connected-object view for describing verified oracle effects."""
    if frame.size == 0:
        return []
    state = FrameProcessor(game_id).process(frame)
    out = []
    for blob in state.get("blobs", ()):
        if blob.get("is_background_candidate"):
            continue
        coords = blob.get("coords", ())
        if not coords:
            continue
        r0 = min(int(row) for row, _ in coords)
        c0 = min(int(col) for _, col in coords)
        shape = [
            int(max(row for row, _ in coords) - r0 + 1),
            int(max(col for _, col in coords) - c0 + 1),
            sorted([[int(row - r0), int(col - c0)] for row, col in coords]),
        ]
        shape_hash = hashlib.sha256(
            json.dumps(shape, separators=(",", ":")).encode()
        ).hexdigest()
        out.append({
            "color": int(blob.get("primary_value", -1)),
            "size": int(blob.get("size", len(coords))),
            "shape_hash": shape_hash,
            "center_rc": [
                round(sum(int(row) for row, _ in coords) / len(coords), 2),
                round(sum(int(col) for _, col in coords) / len(coords), 2),
            ],
        })
    return out


def moved_objects(before, after, game_id):
    old = object_components(before, game_id)
    new = object_components(after, game_id)
    unused = set(range(len(new)))
    moves = []
    for item in old:
        candidates = [i for i in unused
                      if new[i]["shape_hash"] == item["shape_hash"]]
        if not candidates:
            candidates = [i for i in unused if new[i]["color"] == item["color"]]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda i: (
            abs(new[i]["center_rc"][0] - item["center_rc"][0])
            + abs(new[i]["center_rc"][1] - item["center_rc"][1])
            + 0.05 * abs(new[i]["size"] - item["size"])
        ))
        target = new[nearest]
        unused.remove(nearest)
        dr = round(target["center_rc"][0] - item["center_rc"][0], 2)
        dc = round(target["center_rc"][1] - item["center_rc"][1], 2)
        if abs(dr) + abs(dc) >= 0.5:
            moves.append({
                "color": item["color"],
                "shape_hash_before": item["shape_hash"],
                "shape_hash_after": target["shape_hash"],
                "before_center_rc": item["center_rc"],
                "after_center_rc": target["center_rc"],
                "delta_rc": [dr, dc],
            })
    return moves


def replay_solution(root_game, root_raw, sequence, game_id):
    game = copy.deepcopy(root_game)
    raw = root_raw
    trace = []
    processor = FrameProcessor(game_id)
    structured_states = [processor.process(last_frame(raw))]
    winning_state = None
    for index, signature in enumerate(sequence, start=1):
        levels_before = int(raw.levels_completed)
        before = last_frame(raw)
        advance_capture = []
        after_raw = apply_signature(game, signature, advance_capture)
        after = last_frame(after_raw)
        effect_after = advance_capture[-1] if advance_capture else after
        after_state = processor.process(effect_after)
        trace.append({
            "step": index,
            "action_signature": list(signature),
            "levels_completed": int(after_raw.levels_completed),
            "state": after_raw.state.name,
            "moved_objects": moved_objects(before, after, game_id),
            "changed_pixels": int(np.count_nonzero(before != after))
            if before.shape == after.shape else None,
            "level_advance_frame_captured": bool(advance_capture),
        })
        raw = after_raw
        structured_states.append(after_state)
        if advance_capture:
            trace[-1]["moved_objects"] = moved_objects(
                before, effect_after, game_id
            )
            trace[-1]["changed_pixels"] = (
                int(np.count_nonzero(before != effect_after))
                if before.shape == effect_after.shape else None
            )
            winning_state = after_state
        elif int(after_raw.levels_completed) > levels_before:
            winning_state = after_state
    return trace, structured_states, winning_state


def shape_digest(value):
    normalized = json.loads(json.dumps(value))
    return hashlib.sha256(
        json.dumps(normalized, separators=(",", ":")).encode()
    ).hexdigest()


def cyrus_log_analysis(game_id, trace, structured_states, winning_state):
    log_path = (
        ROOT / "research" / "system2" / "clean_evaluation_2026-10-30"
        / "20261030T061632Z" / "fresh_memory"
        / f"{game_id}_takeover_off_ranking_off_seed0" / "system2.jsonl"
    )
    if not log_path.exists():
        return {"available": False, "reason": "clean-matrix System 2 log missing"}
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]

    matched_effects = []
    all_verified_effects = []
    moves = [
        (int(step["action_signature"][0]), movement)
        for step in trace for movement in step.get("moved_objects", ())
    ]
    for row in rows:
        sig = row.get("sig") or []
        if not sig:
            continue
        action_id = int(sig[0])
        validation = row.get("action_effect_validation") or {}
        for effect in validation.get("verified_effects", ()):
            if not effect.get("verified"):
                continue
            all_verified_effects.append((action_id, effect))
            shape = shape_digest(effect.get("shape"))
            vector = [int(value) for value in effect.get("vector", ())]
            for move_action, movement in moves:
                if (move_action == action_id
                        and movement.get("shape_hash_before") == shape
                        and movement.get("delta_rc") == vector):
                    matched_effects.append({
                        "action_id": action_id,
                        "shape_hash": shape,
                        "vector_rc": vector,
                        "model_trials": effect.get("trials"),
                        "model_prequential_accuracy": effect.get("prequential_accuracy"),
                    })

    logged_goal_names = set()
    logged_goal_rows = {}
    for row in rows:
        for candidate in row.get("goal_hypotheses", ()):
            name = candidate.get("name")
            if name:
                logged_goal_names.add(name)
                logged_goal_rows[name] = candidate

    goals = GoalInference(memory_path=str(
        ROOT / "research" / "system2" / "oracle_dev_2026-10-30" /
        f"{game_id}_oracle_ephemeral_goals.json"
    ))
    for state in structured_states[:-1]:
        goals.ensure_goals(state)
        goals.observe(state, progressed=False)
    goals.ensure_goals(winning_state)
    logged_hypotheses_true_at_win = []
    for name in sorted(logged_goal_names & set(goals.goals)):
        hypothesis = goals.goals[name]
        if hypothesis.holds(winning_state):
            logged_hypotheses_true_at_win.append({
                "name": name,
                "kind": hypothesis.kind,
                "logged_weight": logged_goal_rows[name].get("weight"),
                "logged_status": logged_goal_rows[name].get("status"),
            })
    return {
        "available": True,
        "cyrus_system2_log": str(log_path.relative_to(ROOT)),
        "logged_candidate_name_count": len(logged_goal_names),
        "logged_hypotheses_true_at_winning_state": logged_hypotheses_true_at_win,
        "proposed_goal_family_true_at_winning_state": bool(logged_hypotheses_true_at_win),
        "verified_effects_in_clean_log": len(all_verified_effects),
        "verified_solution_mover_matches": matched_effects,
        "control_learner_matched_solution_mover": bool(matched_effects),
    }


def search_one(game_id, arcade, max_depth, max_nodes, time_limit_s):
    env = arcade.make(game_id, seed=0, scorecard_id=f"oracle-{game_id}")
    if env is None:
        raise RuntimeError(f"Could not create {game_id}")
    root_raw = env.reset()
    root_game = copy.deepcopy(env._game)
    deadline = time.monotonic() + time_limit_s
    expanded = 0
    depth_reached = 0
    exhausted = "depth_limit"
    solution = None
    start_levels = int(root_raw.levels_completed)
    root_key = state_key(root_game, root_raw)
    frontier = deque([(root_game, root_raw, [])])
    visited = {root_key: 0}
    states_by_depth = {0: 1}
    duplicate_transitions = 0
    while frontier:
        if expanded >= max_nodes:
            exhausted = "node_budget"
            break
        if time.monotonic() >= deadline:
            exhausted = "time_budget"
            break
        game, raw, path = frontier.popleft()
        depth_reached = max(depth_reached, len(path))
        if len(path) >= max_depth or raw.state.name in {"GAME_OVER", "WIN"}:
            continue
        for signature in available_signatures(game, raw, game_id):
            if expanded >= max_nodes or time.monotonic() >= deadline:
                exhausted = "node_budget" if expanded >= max_nodes else "time_budget"
                break
            expanded += 1
            child = copy.deepcopy(game)
            try:
                child_raw = apply_signature(child, signature)
            except Exception:
                continue
            child_path = path + [signature]
            if int(child_raw.levels_completed) > start_levels:
                solution = child_path
                exhausted = "solution_found"
                break
            if child_raw.state.name in {"GAME_OVER", "WIN"}:
                continue
            key = state_key(child, child_raw)
            if key not in visited:
                visited[key] = len(child_path)
                states_by_depth[len(child_path)] = states_by_depth.get(len(child_path), 0) + 1
                frontier.append((child, child_raw, child_path))
            else:
                duplicate_transitions += 1
        if solution is not None or exhausted in {"node_budget", "time_budget"}:
            break

    trace, structured_states, winning_state = (
        replay_solution(root_game, root_raw, solution, game_id)
        if solution else ([], [], None)
    )
    log_analysis = cyrus_log_analysis(
        game_id, trace, structured_states, winning_state
    ) if solution else {
        "available": True,
        "control_learner_matched_solution_mover": None,
        "proposed_goal_family_true_at_winning_state": None,
        "reason": "No level-1 oracle sequence found within the configured bounds.",
    }
    return {
        "game_id": game_id,
        "seed": 0,
        "solved_level_1": solution is not None,
        "solution_length": len(solution) if solution is not None else None,
        "solution": [list(item) for item in solution] if solution else None,
        "expanded_nodes": expanded,
        "unique_states": len(visited),
        "unique_states_by_depth": states_by_depth,
        "duplicate_transitions": duplicate_transitions,
        "depth_reached": depth_reached,
        "search_stop": exhausted,
        "time_budget_seconds": time_limit_s,
        "trace": trace,
        "cyrus_log_analysis": log_analysis,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", default=",".join(DEV_GAMES))
    parser.add_argument("--max-depth", type=int, default=14)
    parser.add_argument("--max-nodes", type=int, default=10000)
    parser.add_argument("--time-limit", type=float, default=120.0)
    args = parser.parse_args()
    games = tuple(value.strip() for value in args.games.split(",") if value.strip())
    unknown = set(games) - (set(DEV_GAMES) | set(BENCHMARK_GAMES))
    if unknown:
        raise SystemExit(f"Oracle restricted to development games: {sorted(unknown)}")
    logging.basicConfig(level=logging.WARNING)
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    run_dir = OUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, game in enumerate(games, start=1):
        try:
            row = search_one(game, arcade, args.max_depth,
                             args.max_nodes, args.time_limit)
        except Exception as exc:
            row = {
                "game_id": game,
                "seed": 0,
                "solved_level_1": False,
                "solution_length": None,
                "solution": None,
                "expanded_nodes": 0,
                "depth_reached": 0,
                "search_stop": f"exception: {type(exc).__name__}: {exc}",
                "time_budget_seconds": args.time_limit,
                "trace": [],
            }
        rows.append(row)
        if row["solved_level_1"]:
            (run_dir / f"{game}_level1_trace.json").write_text(
                json.dumps(row["trace"], indent=2), encoding="utf-8"
            )
        with (run_dir / "oracle_results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            f"[{index}/{len(games)}] {game}: solved={row['solved_level_1']} "
            f"length={row['solution_length']} nodes={row['expanded_nodes']} "
            f"depth={row['depth_reached']} stop={row['search_stop']}",
            flush=True,
        )
    (run_dir / "summary.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dev_games_only": list(games),
        "max_depth": args.max_depth,
        "max_nodes_per_game": args.max_nodes,
        "time_limit_seconds_per_game": args.time_limit,
        "diagnostic_only_not_agent_code_or_eval": True,
        "results": rows,
    }, indent=2), encoding="utf-8")
    print(f"REPORT_DIR={run_dir}", flush=True)


if __name__ == "__main__":
    main()
