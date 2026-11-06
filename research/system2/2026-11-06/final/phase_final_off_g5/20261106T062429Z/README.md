# System 1 frontier acceptance run

- Created UTC: 2026-11-06T06:40:16.665852+00:00
- Seed: 0
- Budgets: (5000,)
- Git commit: 5a70a692278e11b90afc836430c934ae1263f3b1
- System 2 takeover, learned ranking, relational planner, and local LLM: off.
- Fresh memory per game; RESET actions count toward budget and scorecard.
- `CYRUS_FRONTIER_MODE=1` uses multi-action attempts and resets only at a local dead end or the per-attempt cap.
- Command: `python scripts/reachability_system1.py --games tn36,tr87,tu93,vc33,wa30 --budgets 5000`
- Acceptance fixed before evaluation: first-level completion within 50% of historical actions (cd82 <=862, sp80 <=714, tu93 <=1751).
- Full data: `runs.csv`, `runs.jsonl`, `summary.json`; complete action traces are under each `fresh_memory/` directory.
