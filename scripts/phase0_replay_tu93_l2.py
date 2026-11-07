"""Replay tu93 Level 1 and the recorded Level 2 winning segment in sequence."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "Cyrus")]

import arc_agi
from arc_agi import OperationMode
from arcengine import ActionInput, GameAction
from tools.oracle_dev_search import last_frame, apply_signature
from scripts.phase0_replay_sequences import describe

SOURCE = ROOT / "research/system2/reachability_system1_2026-10-30/20261030T092714Z/fresh_memory/tu93_system1_seed0_cap5000/winning_trace.jsonl"
OUT = ROOT / "research/system2/2026-11-02/phase1b/tu93_l2_replay.jsonl"

L1 = [4, 2, 2, 4, 1, 4, 2, 2, 3, 3, 3, 1, 1, 2, 1, 3, 1, 3, 3, 2,
      4, 4, 4, 1, 2, 2, 3, 4, 1, 4, 3, 1, 3, 4, 2]


def digest(frame: np.ndarray) -> str:
    return hashlib.sha256(frame.tobytes()).hexdigest()


def perform(game, sig):
    action_id, x, y = sig
    data = {"x": int(x), "y": int(y)} if action_id == GameAction.ACTION6.value else {}
    return game.perform_action(
        ActionInput(id=GameAction.from_id(int(action_id)), data=data), raw=True
    )


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    l1_win = next(row for row in rows if int(row["levels_completed"]) >= 1)
    l2_win = next(row for row in rows if int(row["levels_completed"]) >= 2)
    # Start with the first source action after the Level 1 completion and keep
    # all later source actions, including any RESET that occurred during the
    # Level 2 attempt. No RESET is inserted between the two levels.
    segment = [row for row in rows if int(l1_win["index"]) < int(row["index"]) <= int(l2_win["index"])]

    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make("tu93", seed=0, scorecard_id="phase1b-tu93-l2-replay")
    if env is None:
        raise RuntimeError("could not create tu93")
    raw = env.reset()
    game = copy.deepcopy(env._game)
    for action_id in L1:
        raw = perform(game, (action_id, None, None))
    if int(raw.levels_completed) != 1:
        raise RuntimeError(f"Level 1 setup failed: levels_completed={raw.levels_completed}")

    output = []
    for step, row in enumerate(segment, start=1):
        signature = (int(row["action_id"]), None, None)
        before = last_frame(raw)
        raw = perform(game, signature)
        after = last_frame(raw)
        output.append({
            "step_after_l1": step,
            "source_trace_index": int(row["index"]),
            "action_id": int(row["action_id"]),
            "action": row.get("action"),
            "state_hash_before": digest(before),
            "state_hash_after": digest(after),
            "frame_diff": describe(before, after),
            "no_op": bool(np.array_equal(before, after)),
            "levels_completed": int(raw.levels_completed),
            "state": raw.state.name,
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in output), encoding="utf-8")
    print(json.dumps({
        "level1_setup_actions": len(L1),
        "source_level1_win_index": int(l1_win["index"]),
        "source_level2_win_index": int(l2_win["index"]),
        "segment_actions_including_resets": len(segment),
        "resets_in_segment": sum(int(row["action_id"]) == GameAction.RESET.value for row in segment),
        "levels_completed_after_replay": int(raw.levels_completed),
        "final_state": raw.state.name,
        "output": str(OUT.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
