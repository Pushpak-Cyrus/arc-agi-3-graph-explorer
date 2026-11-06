# Final paired evaluation — 2026-11-06

Frozen code: `5a70a692278e11b90afc836430c934ae1263f3b1`.

Two conditions were run across all 25 cached games, seed 0, 5,000 actions per game, fresh memory per game:

- **Flag off:** reset-control fix only; `CYRUS_SEARCH_ORDER=0`.
- **Flag on:** same commit and fix, iterative search order enabled; `CYRUS_SEARCH_ORDER=1`.

System 2 takeover, learned ranking, relational planner, and local LLM were disabled. RESETs count as actions. Runtime: Python 3.12.10, `arc-agi` 0.9.9. Each of five disjoint groups has its own batch directory and README, with per-run raw rows, traces, scorecard data, and logs. The `on` batch used the runner's `--search-order` switch, which sets `CYRUS_SEARCH_ORDER=1` for each game; the `off` batch omitted it and sets the variable to 0.

- Full report: `FINAL_REPORT.md`
- Wide per-game comparison: `all25_comparison.csv` and `.json`
- Aggregate counts and scorecard averages: `condition_summary.csv`
- Paper evidence inventory: `paper_citation_manifest.csv`

The final source build is fixed at the commit above. No agent source was edited after the evaluation began; these files aggregate and document the completed runs.

