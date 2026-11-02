# Phase 1c run index

- Frozen code commit: `db03b91c464b23d25f5bbf1dc215f1d5b5b48267`
- Agent: System 1 only; System 2 takeover, learned ranking, relational planner, and local LLM disabled.
- All evaluation games use seed 0 and independent fresh memory.
- Acceptance run: [`acceptance_final/20261102T070046Z/README.md`](acceptance_final/20261102T070046Z/README.md)
- Full suite, split into five disjoint batches to keep each runner session bounded:
  - [`all25_g1/20261102T160709Z/README.md`](all25_g1/20261102T160709Z/README.md)
  - [`all25_g2/20261102T160709Z/README.md`](all25_g2/20261102T160709Z/README.md)
  - [`all25_g3/20261102T160709Z/README.md`](all25_g3/20261102T160709Z/README.md)
  - [`all25_g4/20261102T160709Z/README.md`](all25_g4/20261102T160709Z/README.md)
  - [`all25_g5/20261102T160709Z/README.md`](all25_g5/20261102T160709Z/README.md)
- Combined per-game data: [`all25_combined.csv`](all25_combined.csv) and [`all25_combined.json`](all25_combined.json)
- Full factual report: [`REPORT.md`](REPORT.md)
- Code build and run history: [`../BUILDS.md`](../BUILDS.md)

The failed attempts under `acceptance/` and the interrupted run under `all25/` are retained for audit but are excluded from the final metrics.
