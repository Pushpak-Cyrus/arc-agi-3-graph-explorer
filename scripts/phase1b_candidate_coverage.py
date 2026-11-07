"""Check whether System 1's observed-frame candidate sets contain oracle moves."""
from __future__ import annotations
import copy
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "vendor" / "ARC-AGI-3-Agents"),
                str(ROOT / "Cyrus")]
import arc_agi
from arc_agi import OperationMode
from Cyrus.cyrus_agent import CyrusAgent
from Cyrus.frame_processor import FrameProcessor
from Cyrus.graph_memory import GraphMemory
from tools.oracle_dev_search import last_frame, apply_signature

OUT = ROOT / "research" / "system2" / "2026-11-02" / "phase1b"
ORACLE_PATHS = {
    "cd82": ROOT / "research/system2/oracle_dev_2026-10-30/20261031T080059Z/cd82_level1_trace.json",
    "sp80": ROOT / "research/system2/oracle_dev_2026-10-30/20261031T080059Z/sp80_level1_trace.json",
    "tu93": ROOT / "research/system2/oracle_dev_2026-10-30/20261031T080059Z/tu93_level1_trace.json",
    "ls20": ROOT / "research/system2/oracle_dev_2026-10-30/20261031T080138Z/ls20_level1_trace.json",
}
CONFIRMED_TRACE_SIGNATURES = {
    "cd82": [(1, None, None), (6, 31, 63), (3, None, None),
             (3, None, None), (2, None, None), (2, None, None),
             (4, None, None), (5, None, None)],
    "sp80": [(4, None, None), (4, None, None), (5, None, None),
             (1, None, None), (1, None, None), (4, None, None),
             (4, None, None), (2, None, None), (2, None, None),
             (3, None, None), (2, None, None), (5, None, None)],
    "tu93": [(a, None, None) for a in
             [4, 2, 2, 4, 1, 4, 2, 2, 3, 3, 3, 1, 1, 2, 1, 3, 1, 3, 3, 2, 4, 4, 4, 1, 2, 2, 3, 4, 1, 4, 3, 1, 3, 4, 2]],
}


def main():
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    results = []
    for game_id, path in ORACLE_PATHS.items():
        trace = json.loads(path.read_text(encoding="utf-8"))
        env = arcade.make(game_id, seed=0,
                          scorecard_id=f"phase1b-candidate-{game_id}")
        raw = env.reset()
        game = copy.deepcopy(env._game)
        processor = FrameProcessor(game_id)
        agent = CyrusAgent.__new__(CyrusAgent)
        agent.click_class_changed = set()
        agent.click_class_noop_states = {}
        agent.graph = GraphMemory()
        step_rows = []
        for oracle_step, row in enumerate(trace, start=1):
            frame = last_frame(raw)
            state = processor.process(np.asarray(frame))
            candidates = agent._get_action_signatures(
                raw.available_actions, state
            )
            signatures = {tuple(sig) for sig, _tier in candidates}
            oracle_sig = tuple(row["action_signature"])
            step_rows.append({
                "step": oracle_step,
                "oracle_action": list(oracle_sig),
                "candidate_included": oracle_sig in signatures,
                "candidate_count": len(signatures),
                "levels_before": int(raw.levels_completed),
            })
            raw = apply_signature(game, oracle_sig)
            if raw.state.name == "WIN" or int(raw.levels_completed) > 0:
                break
        results.append({
            "game_id": game_id,
            "oracle_solution_length": len(trace),
            "steps_checked": len(step_rows),
            "candidate_coverage": sum(r["candidate_included"] for r in step_rows),
            "all_actions_included": all(r["candidate_included"] for r in step_rows),
            "replay_levels_completed": int(raw.levels_completed),
            "steps": step_rows,
        })
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "candidate_coverage.json"
    for result in results:
        print(result["game_id"], result["candidate_coverage"], "/",
              result["steps_checked"], "all=", result["all_actions_included"],
              "levels=", result["replay_levels_completed"])
        for step in result["steps"]:
            print(" ", step["step"], step["oracle_action"],
                  "candidate=", step["candidate_included"])
    # Also check the separately recorded historical winning traces. These
    # include the cd82 click, which is absent from the shorter oracle path.
    historical = []
    for game_id, seq in CONFIRMED_TRACE_SIGNATURES.items():
        env = arcade.make(game_id, seed=0,
                          scorecard_id=f"phase1b-historical-{game_id}")
        raw = env.reset()
        game = copy.deepcopy(env._game)
        processor = FrameProcessor(game_id)
        agent = CyrusAgent.__new__(CyrusAgent)
        agent.click_class_changed = set()
        agent.click_class_noop_states = {}
        agent.graph = GraphMemory()
        checks = []
        for step, sig in enumerate(seq, start=1):
            frame_grid = last_frame(raw)
            state = processor.process(np.asarray(frame_grid))
            candidates = agent._get_action_signatures(
                raw.available_actions, state
            )
            candidate_set = {tuple(item[0]) for item in candidates}
            checks.append({"step": step, "oracle_action": list(sig),
                           "candidate_included": sig in candidate_set})
            raw = apply_signature(game, sig)
            if int(raw.levels_completed) > 0 or raw.state.name == "WIN":
                break
        historical.append({"game_id": game_id, "steps": checks,
                           "candidate_coverage": sum(
                               row["candidate_included"] for row in checks),
                           "steps_checked": len(checks),
                           "levels_completed": int(raw.levels_completed)})
    for result in historical:
        print("historical", result["game_id"],
              result["candidate_coverage"], "/", result["steps_checked"],
              "levels=", result["levels_completed"])
        for step in result["steps"]:
            print(" ", step["step"], step["oracle_action"],
                  "candidate=", step["candidate_included"])
    results.append({"historical_confirmed_traces": historical})
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"OUTPUT={output}")


if __name__ == "__main__":
    main()
