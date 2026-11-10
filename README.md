# ARC-AGI-3 Graph Explorer (CYRUS)

A research prototype for [ARC-AGI-3](https://arxiv.org/abs/2603.24621): an agent that learns to play unseen interactive grid games by building a graph of the states it has observed and searching it systematically, with no per-game or per-level strategies.

**Status: finished research prototype, with negative results.** The agent solves the first level in 5 of 25 local games, usually inefficiently. It does not generalise, and this repository is mainly an honest record of what was built, what the diagnostics showed, and what went wrong.

> These results come from 25 cached game layouts, one seed each, run locally. They are not online competition results and give no evidence of generalisation to unseen layouts.

## Contents

- [Results at a glance](#results-at-a-glance)
- [What we learned](#what-we-learned)
- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Reproducing the results](#reproducing-the-results)
- [Repository layout](#repository-layout)
- [Evaluation protocol and scoring](#evaluation-protocol-and-scoring)
- [Limitations](#limitations)
- [References](#references)
- [License](#license)

## Results at a glance

Final paired runs: seed 0, fresh memory for every game, 5,000 actions per game (RESET actions count). Both runs used the same source snapshot (HEAD `c5fe794`, working tree not clean; source hashes are recorded in `research/system2/BUILDS.md`).

| Setting | Levels solved (of 25 games) | Mean scorecard |
|---|---:|---:|
| Frontier search (default) | 5 | 0.0556% |
| Frontier search + iterative deepening (`--search-order`) | 5 | 0.0032% |

Games where a level was confirmed (first-level action index in brackets; `-` means none):

| Game | Default | With `--search-order` |
|---|---|---|
| lf52 | 1 level (96) | 1 level (159) |
| r11l | 1 level (48) | 1 level (845) |
| tn36 | 1 level (140) | - |
| lp85 | 1 level (672) | 1 level (679) |
| vc33 | 1 level (2,099) | 1 level (961) |
| sp80 | - | 1 level (4,187) |

The other 19 games completed no level in either setting. Full per-game tables, including scores, are in [`research/system2/BUILDS.md`](research/system2/BUILDS.md).

The scorecard rewards efficiency: each level's score is `(baseline_actions / actions_taken)²`. Levels found by wandering at hundreds or thousands of actions score close to zero, which is why five solved games produce a mean under 0.06%.

Three acceptance checks were set in advance (first level in at most 862 actions for cd82, 714 for sp80, 1,751 for tu93). **All three failed.**

## What we learned

1. **Short solutions exist, and the agent doesn't find them efficiently.** A breadth-first oracle (using game-state cloning, which a real agent can't do) found first-level solutions of 5 actions (cd82), 4 (sp80), 18 (tu93) and 13 (ls20). An earlier random-walk build needed 1,429 to 3,503 actions to stumble on solutions in three of these games.
2. **A control-flow bug silently broke 8 of 25 games.** After a RESET, an exhausted search could issue another RESET before replaying to a frontier. In ft09, lp85, ls20, re86, sc25, sk48, tr87 and tu93, nearly every RESET was a duplicate, so those runs measured the bug, not the method. Fixing it raised the count of solved games from 3 to 5.
3. **The engine treats a RESET at action count 0 as a full-game restart.** In the installed `arcengine`, RESET only restores the current level when at least one action has been taken since the level began. Sending RESET right after another RESET, or as the first action after a level advance, returns the game to level 0 with score 0. This explains a level count that dropped from 1 to 0 in vc33. The behaviour depends on the engine's `ONLY_RESET_LEVELS` setting, and the official evaluation's setting is unverified.
4. **The search-order change did not help.** Iterative deepening and simple-actions-first solved sp80 (late) but lost tn36 and made r11l 17 times slower. Short attempts raise replay cost. With only five solved games, read this as "didn't help here", not as a general result.
5. **Candidate generation was not the failure.** Every action in the oracle's winning paths was in the agent's candidate set at the right state. The failing link is selection order and frontier control.
6. **An unexplained gap in tu93.** The agent visited about 1,700 unique states, while the oracle's search through depth 14 found only 349 on the same game. Either the agent wanders far deeper than the solution, or the frame hash changes between visits to the same position. This is unresolved.

## How it works

**System 1 (graph explorer).** The method follows the graph-exploration approach of [arXiv:2512.24156](https://arxiv.org/abs/2512.24156): frames are turned into state representations, each (state, action) pair is a graph edge, and the agent prefers untested actions and navigates to the shallowest observed frontier by replaying known action paths after a RESET. Coordinate (click) actions are stored as distinct experiments, grouped by object class (colour, size, normalised shape) so one representative is tried first.

**Optional search order (`CYRUS_SEARCH_ORDER=1`).** Iterative deepening (attempt depth 6, 12, 24, then 50) and iterative widening (simple actions before clicks at each state). Off by default.

**System 2 (optional, disabled in all reported results).** `Cyrus/system2/` contains hypothesis, interaction-model and goal-inference components. System 2 takeover, learned ranking, the relational planner and the local LLM were switched off for the results above.

**Oracle (diagnostic only).** `tools/oracle_dev_search.py` runs a breadth-first search over state hashes using game-state cloning. It exists to measure how short the solutions are. The agent never imports or calls it, and it contributes nothing to the evaluation.

## Quick start

Requires Python 3.12 and the ARC-AGI-3 framework.

```
make setup
make play-local GAME=ls20
```

Run the policy tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s Cyrus -p "test_*.py"
```

## Reproducing the results

PowerShell (Windows). Each run writes to a timestamped folder under `CYRUS_REACHABILITY_OUT`.

```powershell
$games = "ar25,bp35,cd82,cn04,dc22,ft09,g50t,ka59,lf52,lp85,ls20,m0r0,r11l,re86,s5i5,sb26,sc25,sk48,sp80,su15,tn36,tr87,tu93,vc33,wa30"

# Default setting
$env:CYRUS_REACHABILITY_OUT = "research\system2\local_runs\flag_off"
.\.venv\Scripts\python.exe scripts\reachability_system1.py --games $games --budgets 5000

# With iterative deepening / widening
$env:CYRUS_REACHABILITY_OUT = "research\system2\local_runs\flag_on"
.\.venv\Scripts\python.exe scripts\reachability_system1.py --games $games --budgets 5000 --search-order
```

Runs are deterministic: two runs of the same code at seed 0 gave matching results. Per-run output includes final and peak levels, first-level action, resets, experiments per reset, replay fraction, zero-length attempts and full-game resets (the last two were zero in all 50 game-runs).

## Repository layout

| Path | Purpose |
|---|---|
| `agent/my_agent.py` | Entry point exposing the agent to the ARC agent framework |
| `Cyrus/frame_processor.py` | Observation to state representation and progress signals |
| `Cyrus/graph_memory.py` | State graph, action signatures, transitions, frontier search |
| `Cyrus/cyrus_agent.py` | Attempt and reset control, action selection |
| `Cyrus/system2/` | Optional hypothesis and goal-inference components (disabled in reported results) |
| `scripts/reachability_system1.py` | The 25-game evaluation runner |
| `tools/oracle_dev_search.py` | Offline oracle for diagnostics (never used by the agent) |
| `research/system2/BUILDS.md` | Build and run ledger with per-game tables and source hashes |
| `vendor/ARC-AGI-3-Agents/` | Vendored ARC agent framework (MIT licence) |

## Evaluation protocol and scoring

- **Setup:** 25 cached game layouts, one layout per game, seed 0, fresh memory per game, 5,000 actions per game including RESET.
- **Score:** the locally installed ARC environment scorecard, shown as a percentage. For each level the raw score is `min((baseline_actions / actions_taken)² × 100, 115)`, weighted by the one-based level number, averaged over the game's total level weights, and capped by the completed-level weight fraction. It is not the percentage of games solved.
- **Per game versus mean:** a per-game percentage and a 25-game mean are different numbers. Earlier project notes mixed them up (for example, 4.76% for r11l was one run of one game that finished level 1 in 4 actions). All figures in this README are per-game scorecards or, where stated, the 25-game mean.
- **Levels within a run:** the scorecard uses the final level count of a run, so progress lost to a full-game reset is lost for scoring.
- **Dev and acceptance games:** ar25, cn04, dc22, ls20 and re86 were used for development, and cd82, sp80 and tu93 for design and acceptance tests. No game is a true held-out set.

## Limitations

- One cached layout and one seed per game. Seeds gave identical layouts, so there are no independent trials.
- All 25 games were visible during development, so there is no clean held-out evaluation.
- Results are local. They are not online competition results.
- Five levels solved across 25 games is too few to conclude that one search setting is better than another.
- ft09 and sc25 exhausted the search before 5,000 actions. This does not show that those games can't be solved.
- The engine's full-reset behaviour is from the locally installed `arcengine`. The official evaluation's setting is unverified.
- The tu93 state-count discrepancy (point 6 above) is unexplained.

## References

- ARC-AGI-3 benchmark: [arXiv:2603.24621](https://arxiv.org/abs/2603.24621)
- Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks: [arXiv:2512.24156](https://arxiv.org/abs/2512.24156). That paper reports a median of 30 solved levels out of 52 across six games on the ARC-AGI-3 Preview Challenge. Its game set and protocol differ from this repository's, so the results are not comparable.

## License

There is no project-wide license yet. The vendored `vendor/ARC-AGI-3-Agents/` is MIT-licensed and keeps its own `LICENSE` file. Add a license before reusing this code.

---

Built by Pushpak as an independent research project. Detailed run history is in [`research/system2/BUILDS.md`](research/system2/BUILDS.md).
