# CYRUS final build and evaluation report

**Evaluation date:** 2026-11-06  
**Frozen code commit:** `5a70a692278e11b90afc836430c934ae1263f3b1`  
**Scope:** reset-control bug fix plus one optional, domain-agnostic search-order change; no goal inference or planner; oracle isolated under `tools/` and not imported by the agent.

## What was built

The frozen code commit fixes the frontier/reset control flow so an exhausted root does not trigger a zero-action attempt and a repeated RESET is not issued without intervening progress. It also adds `CYRUS_SEARCH_ORDER=1` as an opt-in search order. With the flag off, the bug fix runs with the pre-existing attempt depth of 50. With the flag on, the explorer uses iterative frontier-depth limits of 6, 12, 24, and 50 and simple-action-first widening before expanding click candidates. Both settings use the same code commit, System 2 takeover off, learned ranking off, relational planner off, local LLM off, seed 0, fresh per-game memory, and a 5,000-action budget. Before the code freeze, the focused unittest set passed (7 tests) and the changed Python files passed `py_compile`. RESETs count toward that budget and the scorecard. The runtime was Python 3.12.10 with `arc-agi` 0.9.9.

## Facts: paired 25-game results

Each game was run once per setting at commit `5a70a692278e11b90afc836430c934ae1263f3b1`. The official scorecard value is shown as percent. “First L1 action” is the action index when the environment first confirmed a level advance; blank means no confirmed level advance.

| Game | Split | Levels off | Levels on | First L1 off | First L1 on | Scorecard off (%) | Scorecard on (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| ar25 | dev | 0 | 0 | — | — | 0.000000 | 0.000000 |
| bp35 | held_out | 0 | 0 | — | — | 0.000000 | 0.000000 |
| cd82 | held_out | 0 | 0 | — | — | 0.000000 | 0.000000 |
| cn04 | dev | 0 | 0 | — | — | 0.000000 | 0.000000 |
| dc22 | dev | 0 | 0 | — | — | 0.000000 | 0.000000 |
| ft09 | held_out | 0 | 0 | — | — | 0.000000 | 0.000000 |
| g50t | held_out | 0 | 0 | — | — | 0.000000 | 0.000000 |
| ka59 | held_out | 0 | 0 | — | — | 0.000000 | 0.000000 |
| lf52 | held_out | 1 | 1 | 96 | 159 | 0.202020 | 0.073645 |
| lp85 | held_out | 1 | 1 | 672 | 679 | 0.001778 | 0.001741 |
| ls20 | dev | 0 | 0 | — | — | 0.000000 | 0.000000 |
| m0r0 | held_out | 0 | 0 | — | — | 0.000000 | 0.000000 |
| r11l | held_out | 1 | 1 | 48 | 845 | 1.000331 | 0.003228 |
| re86 | dev | 0 | 0 | — | — | 0.000000 | 0.000000 |
| s5i5 | held_out | 0 | 0 | — | — | 0.000000 | 0.000000 |
| sb26 | held_out | 0 | 0 | — | — | 0.000000 | 0.000000 |
| sc25 | held_out | 0 | 0 | — | — | 0.000000 | 0.000000 |
| sk48 | held_out | 0 | 0 | — | — | 0.000000 | 0.000000 |
| sp80 | held_out | 0 | 1 | — | 4187 | 0.000000 | 0.000413 |
| su15 | held_out | 0 | 0 | — | — | 0.000000 | 0.000000 |
| tn36 | held_out | 1 | 0 | 140 | — | 0.186589 | 0.000000 |
| tr87 | held_out | 0 | 0 | — | — | 0.000000 | 0.000000 |
| tu93 | held_out | 0 | 0 | — | — | 0.000000 | 0.000000 |
| vc33 | held_out | 1 | 1 | 2099 | 961 | 0.000040 | 0.000189 |

| wa30 | held_out | 0 | 0 | — | — | 0.000000 | 0.000000 |
Per-run diagnostics are in [`all25_comparison.csv`](all25_comparison.csv) and [`all25_comparison.json`](all25_comparison.json), including peak/final levels, actions, resets, experiments per reset, replay fraction, zero-length attempt count, unique states per 1,000 actions, end reason, and exceptions. Batch-level raw outputs and traces are retained under the ten `phase_final_{off,on}_g{1..5}` directories.

### Aggregate scorecard

| Evaluation group | Games | Levels off / on | Mean scorecard off (%) | Mean scorecard on (%) |
|---|---:|---:|---:|---:|
| All games | 25 | 5 / 5 | 0.055630 | 0.003169 |
| Development set (ar25, cn04, dc22, ls20, re86) | 5 | 0 / 0 | 0.000000 | 0.000000 |
| Acceptance/design-exposed (cd82, sp80, tu93) | 3 | 0 / 1 | 0.000000 | 0.000138 |

| Remaining held-out (17 games) | 17 | 5 / 4 | 0.081809 | 0.004636 |
The evaluation harness labels the 20 non-development games as `held_out`. For interpretation, cd82, sp80, and tu93 are separated above because their winning traces and acceptance thresholds were used during design; they are not untouched held-out evidence. The 17 remaining games are not claimed to establish generalization from one layout each.

### Acceptance games

The predeclared first-level action thresholds were cd82 ≤862, sp80 ≤714, and tu93 ≤1,751, with RESETs included in action counts. ls20 was an additional development check without a predeclared action threshold.

| Game | Off: levels / first L1 / score (%) | On: levels / first L1 / score (%) | Outcome |
|---|---|---|---|
| cd82 | 0 / — / 0.000000 | 0 / — / 0.000000 | Threshold not met in either setting |
| sp80 | 0 / — / 0.000000 | 1 / 4,187 / 0.000413 | Level reached only with search order; exceeds 714-action threshold |
| tu93 | 0 / — / 0.000000 | 0 / — / 0.000000 | Threshold not met in either setting |
| ls20 | 0 / — / 0.000000 | 0 / — / 0.000000 | No confirmed level in either setting |


## Interpretation

Both settings confirmed five levels across the full suite. They reached different sets of games: the bug-fix-only setting progressed on lf52, lp85, r11l, tn36, and vc33; search order progressed on lf52, lp85, r11l, sp80, and vc33. Search order gained sp80 and lost tn36, without changing the total level count. Its aggregate scorecard was lower because action efficiency matters: in particular, r11l advanced at action 48 with the flag off and at action 845 with it on. Thus this run does not support a claim that the new search order improved overall performance.
The reset-control fix is supported by the measured zero zero-length frontier attempts in all 50 runs. This addresses the specific reset-loop failure; it does not by itself produce reliable task solving. Average scorecard fell from 0.055630% off to 0.003169% on, despite an equal count of confirmed levels.

## Remaining limitations and limits of evidence

- Each game had one cached layout, seed 0, and one run per setting. The 25 games are not multiple independent layouts of the same task.
- The search-order change is not accepted on the predeclared cd82, sp80, and tu93 thresholds. sp80 is a late completion, not a pass.
- The dev set had no confirmed level in either condition. Confirmed progress remains sparse and concentrated in a small number of games.
- The action cap caused most runs to stop at 5,000. ft09 stopped at 37 actions in both settings; sc25 stopped at 879 off and 124 on. The runner recorded these as `harness_or_agent_limit`, with no exception. Their behavior before termination is preserved in the traces.
- Offline cached environments and the local scorecard calculation are the basis of these results. No submission score, online evaluation, or multi-layout generalization was measured.
- “Unique states per 1,000 actions” is a search-activity diagnostic, not task-solving performance.
- No further agent-code changes are made after this commit. The report and result artifacts are documentation/evidence only.

## Paper evidence inventory

[`paper_citation_manifest.csv`](paper_citation_manifest.csv) is the exact file-by-file inventory of the final evaluation artifacts and the project files needed to describe or audit this result. It lists all 50 run records and their action/frontier traces, the 10 batch logs and run summaries, consolidated tables, relevant source and test files, the Phase 0 oracle/diagnostic artifacts, and the Phase 1c comparison report/data. The agent does not import or use the oracle in the evaluation.


