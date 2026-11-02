# CYRUS Phase 1c: reset/replay repair and all-game evaluation

**Date:** 2026-11-02  
**Frozen code commit:** `db03b91c464b23d25f5bbf1dc215f1d5b5b48267`  
**Agent condition:** System 1 graph explorer only. System 2 takeover, learned ranking, relational planner, and local LLM disabled. No goal-inference action policy. Oracle remains isolated under `tools/` and was not imported by the agent.  
**Protocol:** seed 0, fresh memory per game, 5,000 wrapper-action budget per game; reset actions are included in the action budget. One cached environment layout per game. The full suite ran as five disjoint batches; combined results are in `all25_combined.csv` and `all25_combined.json`.

## Facts

### Changes frozen in Phase 1c

- Frontier replay is protected for the entire route: the explorer continues replaying until the selected frontier is reached, rather than evaluating a replay intermediate state as a fresh dead end.
- A reset is allowed only after at least one action since level start and since the prior reset; the previous action cannot itself be RESET. GAME_OVER remains the reset exception.
- Iterative deepening uses fixed attempt limits `D = 6, 12, 24, 50`, preferring the shallowest reachable frontier under the current D and widening only when none exists within it.
- Iterative widening exposes simple actions first. A click class with a demonstrated frame change may be considered earlier; otherwise click classes enter after the state’s simple actions are tested. No-op evidence deprioritizes after three distinct states and never bans an action.
- The exhausted-root unit test, iterative-widening test, focused frame/graph unit tests, and Python syntax compilation passed: 7 unit tests passed.

### Four-game acceptance

The fixed thresholds were not changed. The three threshold games all **failed** acceptance: `cd82` and `tu93` did not reach Level 1; `sp80` reached it only at action 4,187, later than its 714-action limit. `ls20` also did not complete a level.

| Game | Fixed threshold | Phase 1b: final / first L1 / resets / exp-reset / replay / unique per 1k | Phase 1c: final / peak / first L1 / resets / exp-reset / replay / zero length / unique per 1k | Verdict |
|---|---:|---|---|---|
| cd82 | <=862 | 0 / — / 104 / 45.29 / 3.72% / 217.0 | 0 / 0 / — / 679 / 4.25 / 28.70% / 0 / 44.6 | FAIL |
| sp80 | <=714 | 0 / — / 189 / 22.40 / 11.56% / 140.4 | 1 / 1 / 4187 / 591 / 5.84 / 19.20% / 0 / 73.4 | FAIL |
| tu93 | <=1,751 | 0 / — / 4798 / 0.04 / 0.04% / 29.2 | 0 / 0 / — / 257 / 10.04 / 43.24% / 0 / 202.4 | FAIL |
| ls20 | dev only | 0 / — / 4412 / 0.13 / 0.24% / 68.8 | 0 / 0 / — / 289 / 9.91 / 36.96% / 0 / 204.4 | FAIL |

### Full 25-game paired result

The “prior” columns are the all-25 Phase 2 run `research/system2/2026-11-02/phase2/20261101T185937Z`; current columns are this Phase 1c build. The Phase 1b acceptance run is shown separately above because it was a different four-game evaluation. “Experiments/reset” is new non-replay experiments divided by RESET actions. Zero-length attempts were independently counted from action traces: a frontier RESET with no non-replay experiment since the prior RESET. The run’s stored zero-length counter is not used as the source for this column.

| Split | Game | Final levels: prior -> current (peak) | First L1 action: prior -> current | Actions | Resets: prior -> current | Experiments/reset: prior -> current | Replay: prior -> current | Unique states/1k: prior -> current | Zero-length attempts | Scorecard %: prior -> current | End reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dev | ar25 | 0 -> 0 (0) | — -> — | 5000 | 1948 -> 399 | 0.56 -> 8.11 | 39.0% -> 27.3% | 20.2 -> 105.0 | 0 | 0.0000% -> 0.0000% | action_cap |
| held-out | bp35 | 0 -> 0 (0) | — -> — | 5000 | 98 -> 651 | 50.02 -> 6.01 | 0.0% -> 8.8% | 220.2 -> 87.6 | 0 | 0.0000% -> 0.0000% | action_cap |
| design-exposed | cd82 | 0 -> 0 (0) | — -> — | 5000 | 104 -> 679 | 45.29 -> 4.25 | 3.7% -> 28.7% | 217.0 -> 44.6 | 0 | 0.0000% -> 0.0000% | action_cap |
| dev | cn04 | 0 -> 0 (0) | — -> — | 5000 | 93 -> 635 | 50.26 -> 5.10 | 4.7% -> 22.5% | 265.0 -> 130.4 | 0 | 0.0000% -> 0.0000% | action_cap |
| dev | dc22 | 0 -> 0 (0) | — -> — | 5000 | 96 -> 657 | 50.20 -> 5.11 | 1.7% -> 19.8% | 150.8 -> 31.0 | 0 | 0.0000% -> 0.0000% | action_cap |
| held-out | ft09 | 0 -> 0 (0) | — -> — | 37 | 4964 -> 1 | 0.01 -> 36.00 | 0.0% -> 0.0% | 0.2 -> 27.0 | 0 | 0.0000% -> 0.0000% | harness_or_agent_limit |
| held-out | g50t | 0 -> 0 (0) | — -> — | 5000 | 1563 -> 374 | 0.20 -> 4.11 | 62.3% -> 61.8% | 31.6 -> 88.6 | 0 | 0.0000% -> 0.0000% | action_cap |
| held-out | ka59 | 0 -> 0 (0) | — -> — | 5000 | 94 -> 648 | 50.06 -> 4.57 | 4.0% -> 27.8% | 288.8 -> 86.6 | 0 | 0.0000% -> 0.0000% | action_cap |
| held-out | lf52 | 1 -> 1 (1) | 96 -> 159 | 5000 | 97 -> 763 | 50.01 -> 4.17 | 1.0% -> 21.1% | 197.4 -> 55.6 | 0 | 0.2020% -> 0.0736% | action_cap |
| held-out | lp85 | 0 -> 1 (1) | — -> 679 | 5000 | 4693 -> 474 | 0.06 -> 7.20 | 0.1% -> 22.3% | 4.2 -> 83.8 | 0 | 0.0000% -> 0.0017% | action_cap |
| dev | ls20 | 0 -> 0 (0) | — -> — | 5000 | 4412 -> 289 | 0.13 -> 9.91 | 0.2% -> 37.0% | 68.8 -> 204.4 | 0 | 0.0000% -> 0.0000% | action_cap |
| held-out | m0r0 | 0 -> 0 (0) | — -> — | 5000 | 149 -> 615 | 29.06 -> 4.26 | 10.4% -> 35.3% | 114.6 -> 37.6 | 0 | 0.0000% -> 0.0000% | action_cap |
| held-out | r11l | 1 -> 1 (1) | 48 -> 845 | 5000 | 121 -> 643 | 39.11 -> 5.98 | 2.9% -> 10.2% | 857.0 -> 420.6 | 0 | 1.0003% -> 0.0032% | action_cap |
| dev | re86 | 0 -> 0 (0) | — -> — | 5000 | 2666 -> 489 | 0.84 -> 6.00 | 1.7% -> 31.5% | 371.4 -> 353.8 | 0 | 0.0000% -> 0.0000% | action_cap |
| held-out | s5i5 | 0 -> 0 (0) | — -> — | 5000 | 182 -> 578 | 25.22 -> 6.00 | 4.6% -> 19.0% | 181.4 -> 34.4 | 0 | 0.0000% -> 0.0000% | action_cap |
| held-out | sb26 | 0 -> 0 (0) | — -> — | 5000 | 1999 -> 465 | 0.50 -> 7.23 | 40.0% -> 23.5% | 51.0 -> 151.4 | 0 | 0.0000% -> 0.0000% | action_cap |
| held-out | sc25 | 0 -> 0 (0) | — -> — | 124 | 4446 -> 15 | 0.12 -> 6.80 | 0.1% -> 5.6% | 29.2 -> 500.0 | 0 | 0.0000% -> 0.0000% | harness_or_agent_limit |
| held-out | sk48 | 0 -> 0 (0) | — -> — | 5000 | 4880 -> 293 | 0.02 -> 10.62 | 0.0% -> 31.9% | 2.0 -> 79.0 | 0 | 0.0000% -> 0.0000% | action_cap |
| design-exposed | sp80 | 0 -> 1 (1) | — -> 4187 | 5000 | 189 -> 591 | 22.40 -> 5.84 | 11.6% -> 19.2% | 140.4 -> 73.4 | 0 | 0.0000% -> 0.0004% | action_cap |
| held-out | su15 | 0 -> 0 (0) | — -> — | 5000 | 171 -> 173 | 26.72 -> 26.82 | 5.2% -> 3.8% | 71.8 -> 67.8 | 0 | 0.0000% -> 0.0000% | action_cap |
| held-out | tn36 | 1 -> 0 (0) | 140 -> — | 5000 | 97 -> 641 | 50.15 -> 5.97 | 0.8% -> 10.7% | 568.8 -> 85.8 | 0 | 0.1866% -> 0.0000% | action_cap |
| held-out | tr87 | 0 -> 0 (0) | — -> — | 5000 | 4182 -> 471 | 0.19 -> 5.93 | 0.4% -> 34.8% | 124.8 -> 368.6 | 0 | 0.0000% -> 0.0000% | action_cap |
| design-exposed | tu93 | 0 -> 0 (0) | — -> — | 5000 | 4798 -> 257 | 0.04 -> 10.04 | 0.0% -> 43.2% | 29.2 -> 202.4 | 0 | 0.0000% -> 0.0000% | action_cap |
| held-out | vc33 | 0 (peak 1) -> 1 (1) | 2793 -> 961 | 5000 | 143 -> 600 | 25.17 -> 5.77 | 25.1% -> 18.8% | 127.0 -> 42.2 | 0 | 0.0000% -> 0.0002% | action_cap |
| held-out | wa30 | 0 -> 0 (0) | — -> — | 5000 | 991 -> 513 | 2.05 -> 4.63 | 39.6% -> 42.3% | 230.4 -> 190.6 | 0 | 0.0000% -> 0.0000% | action_cap |

**Totals and split summary:** Phase 1c reached 5 final levels (5 peak) versus 3 final levels (4 peak observed) in Phase 2. The current per-game official scorecard percentages sum to 0.079217; their equal-weight mean across 25 games is **0.003169%**, versus the previous report’s **0.055558%**. Current RESET count fell from 43,176 to 11,914 across the suite. Five games reached a level: `lf52` (159), `lp85` (679), `r11l` (845), `sp80` (4,187), and `vc33` (961). The current runner ended early because the graph was exhausted in `ft09` at 37 actions and `sc25` at 124 actions; these were not 5,000-action-cap completions, and no exception was recorded. All other games used the full 5,000-action budget.

- **Development (5 games):** 0 final levels; mean scorecard 0.000000%.
- **Design-exposed (cd82, sp80, tu93; 3 games):** 1 final level; mean scorecard 0.000138%.
- **Remaining held-out (17 games):** 4 final levels; mean scorecard 0.004636%.

All 25 action traces show zero adjacent RESET actions, zero full-game resets, and zero trace-derived zero-length frontier attempts. No game’s final level count fell below its peak within the Phase 1c run. Compared with the previous Phase 2 final counts, `tn36` regressed from 1 to 0; `vc33` improved from final 0 / peak 1 to final 1. `lf52` and `r11l` remained at 1, while `lp85` and `sp80` added one each.

### Reset behavior in the installed engine

The installed [`arcengine/base_game.py`](</C:/Users/ASUS/OneDrive/Documents/ARC-AGI-3-Kaggle-Starter/.venv/Lib/site-packages/arcengine/base_game.py:189>) implements RESET based on `_action_count`: when it is zero, [`handle_reset()`](</C:/Users/ASUS/OneDrive/Documents/ARC-AGI-3-Kaggle-Starter/.venv/Lib/site-packages/arcengine/base_game.py:305>) calls [`full_reset()`](</C:/Users/ASUS/OneDrive/Documents/ARC-AGI-3-Kaggle-Starter/.venv/Lib/site-packages/arcengine/base_game.py:318>) (unless `ONLY_RESET_LEVELS=true`); after a non-RESET action, it calls [`level_reset()`](</C:/Users/ASUS/OneDrive/Documents/ARC-AGI-3-Kaggle-Starter/.venv/Lib/site-packages/arcengine/base_game.py:326>). `full_reset()` clears score and restores level 0. [`set_level()`](</C:/Users/ASUS/OneDrive/Documents/ARC-AGI-3-Kaggle-Starter/.venv/Lib/site-packages/arcengine/base_game.py:148>) resets the action counter, and a level transition invokes it via [`_really_set_next_level()`](</C:/Users/ASUS/OneDrive/Documents/ARC-AGI-3-Kaggle-Starter/.venv/Lib/site-packages/arcengine/base_game.py:420>). Therefore a RESET at the beginning of a newly advanced level can be interpreted as a full-game reset and return `levels_completed` to 0. This explains the earlier `vc33` 1-to-0 trace. In Phase 1c, the 25 traces contained zero `full_reset=true` events and no within-run level decreases.

The installed [`ScorecardManager.update_scorecard()`](</C:/Users/ASUS/OneDrive/Documents/ARC-AGI-3-Kaggle-Starter/.venv/Lib/site-packages/arc_agi/scorecard.py:834>) treats a full RESET as `new_play`; [`Card.inc_reset_count()`](</C:/Users/ASUS/OneDrive/Documents/ARC-AGI-3-Kaggle-Starter/.venv/Lib/site-packages/arc_agi/scorecard.py:701>) increments both reset count and action count for a level RESET. [`EnvironmentScorecard.from_scorecard()`](</C:/Users/ASUS/OneDrive/Documents/ARC-AGI-3-Kaggle-Starter/.venv/Lib/site-packages/arc_agi/scorecard.py:540>) chooses the play with the highest `levels_completed`, so the official environment result is based on the best play, not necessarily the final play. In these Phase 1c runs each game had one play and no full resets; all recorded level resets count as actions.

### Scorecard reconciliation

The installed [`EnvironmentScorecard._calculate_score()`](</C:/Users/ASUS/OneDrive/Documents/ARC-AGI-3-Kaggle-Starter/.venv/Lib/site-packages/arc_agi/scorecard.py:380>) computes each level’s raw score as `min((baseline_actions / actions_taken)^2 * 100, 115)`, weights each level by its one-based level index, averages across the game’s levels, then caps the result by the percentage of level-weight available from completed levels. RESET actions count in the level action total for a level reset.

The historical `4.7619%` and Phase 2 `1.0003%` are from different `r11l` runs and are not contradictory formula outputs for the same trajectory:

- The October 23 artifact `r11l_failed_destination_2026-10-23.txt` reports aggregate `4.761904...%`; its trace advances after the fourth action (`ACTION6` count 3) and then continues to 64 total actions. In a six-level game, `4.7619%` is the maximum one-level contribution `100 / (1+2+3+4+5+6)`. The October 22 comparison table’s 66-action entry is a different run and reports aggregate `0.190476%`; pairing its 66 actions with 4.7619% conflates runs.
- The Phase 2 `r11l` run reports `1.0003307%` for first completion at action 48. With a 22-action baseline, the level-1 raw score is about `21.0069`; weighting it by level 1 over total weight 21 gives `1.00033%`.
- The current Phase 1c `r11l` completion is much later, at action 845, and its official scorecard value is `0.003228%`.

Every Phase 1c score in this report is read from the installed local scorecard manager. Historical Phase 2 scores are the environment-score reconstruction recorded in that run’s CSV/report, using the same installed scoring formula; they were not retrieved from a persisted official scorecard object. The aggregate is the arithmetic mean of the 25 per-game scorecard percentages, not their sum.

### Build and run ledger

The complete dated history and run IDs are in [`BUILDS.md`](../../BUILDS.md). Phase 1c run directories each contain a README with command, seed, budget, and code commit. The all-game batch IDs are `20261102T160709Z` under `all25_g1` through `all25_g5`; the accepted four-game run is `20261102T070046Z`. The interrupted first all-game attempt (`20261102T070805Z`) and the two failed runner attempts (`20261102T065556Z`, `20261102T065741Z`) are retained and marked excluded from results.

## Interpretation

The specific reset/replay failure is materially reduced: the old all-25 Phase 2 run used 43,176 RESETs and had repeated zero-length attempts in the known failure cases; the Phase 1c traces have 11,914 RESETs total and no replay-only frontier resets. The fixed acceptance targets still did not reach their action limits. That is a direct failure of the predeclared acceptance criteria, despite one very late `sp80` solve.

The all-game result is mixed: final completions rose from 3 to 5, but aggregate score fell from 0.055558% to 0.003169%. The new completions required hundreds to thousands of actions, and `tn36` was lost. The agent is less trapped in the RESET/replay loop, but it still does not choose short paths to known solutions. The remaining bottleneck is action selection/solution efficiency, not evidence that the environments are unsolvable.

## Limits

- One seed and one cached layout per game; no claims about layout generalization.
- The three threshold games were design-exposed and remain separate from the 17-game held-out set.
- The all-25 results are before/after runs, not multiple paired stochastic trials. The environment is deterministic for these cached layouts, but no layout-seed variation was tested.
- The generic report label `harness_or_agent_limit` for `ft09` and `sc25` corresponds to `system1_search_exhausted` in their traces, not a Python exception.
- The agent’s `zero_length_frontier_attempts` field is not incremented on the early-return path in `_close_frontier_attempt`; the reported value 0 is therefore trace-reconstructed rather than trusted from that internal counter. The test and trace scan found no zero-length frontier RESETs.
- Code is frozen at `db03b91c464b23d25f5bbf1dc215f1d5b5b48267`. No source changes or retuning were made after that commit.
