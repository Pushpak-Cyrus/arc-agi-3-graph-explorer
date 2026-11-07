# CYRUS: a graph-based ARC-AGI-3 explorer

CYRUS is a research prototype for ARC-AGI-3. Its System 1 explorer is based on
the graph-exploration approach described in
[arXiv:2512.24156](https://arxiv.org/abs/2512.24156). An optional System 2 layer
adds hypothesis-driven action selection when exploration stalls. These local
results are evidence about the tested cached layouts only; they do not
demonstrate generalization.

## Architecture

- `agent/my_agent.py` exposes `CyrusAgent` to the ARC agent framework.
- `Cyrus/frame_processor.py` converts observations into state representations
  and progress signals.
- `Cyrus/graph_memory.py` records observed states, action signatures, and
  transitions. Coordinate-target actions are stored as distinct experiments.
- System 1 prioritizes untested actions and navigates to observed graph
  frontiers, replaying known action paths where available.
- `Cyrus/system2/` contains the optional hypothesis, interaction, and
  goal-related components. System 2 takeover, learned ranking, the relational
  planner, and the local LLM were disabled for the paired results below.

`tools/oracle_dev_search.py` is an offline diagnostic tool. The agent does not
import or call it, and it is not part of the evaluation policy.

## Run locally

Use Python 3.12 and the repository setup to install the ARC-AGI-3 framework and
dependencies:

```powershell
make setup
make play-local GAME=ls20
```

The focused policy tests and the 25-game System 1 evaluation can be run with:

```powershell
.\.venv\Scripts\python.exe -m unittest Cyrus.test_action_probe_policy

$env:CYRUS_REACHABILITY_OUT = "research\system2\local_runs\flag_off"
.\.venv\Scripts\python.exe scripts\reachability_system1.py --games ar25,bp35,cd82,cn04,dc22,ft09,g50t,ka59,lf52,lp85,ls20,m0r0,r11l,re86,s5i5,sb26,sc25,sk48,sp80,su15,tn36,tr87,tu93,vc33,wa30 --budgets 5000

$env:CYRUS_REACHABILITY_OUT = "research\system2\local_runs\flag_on"
.\.venv\Scripts\python.exe scripts\reachability_system1.py --games ar25,bp35,cd82,cn04,dc22,ft09,g50t,ka59,lf52,lp85,ls20,m0r0,r11l,re86,s5i5,sb26,sc25,sk48,sp80,su15,tn36,tr87,tu93,vc33,wa30 --budgets 5000 --search-order
```

Each run uses seed 0, fresh memory for each game, and a 5,000-action budget
including RESET actions. `--search-order` enables the optional search order;
when `CYRUS_SEARCH_ORDER` is unset, it is off. Results are written to a
timestamped directory under `CYRUS_REACHABILITY_OUT`.

## Paired 25-game results

The two final runs used the same local source snapshot and HEAD
`c5fe794c21a032e7f22a3982e36c621e75c97e3b`, with the flag-off run completed
first. The working tree was not clean, so the commit hash alone does not
identify the complete source snapshot; the run directories contain the
artifacts and the build ledger records source hashes and this caveat.

The table reports final/peak confirmed levels, the first action at which a
level advance was confirmed, and the local environment scorecard percentage.
`-` means no confirmed level advance.

| Game | Off final/peak | Off first level action | Off score % | On final/peak | On first level action | On score % |
|---|---:|---:|---:|---:|---:|---:|
| ar25 | 0/0 | - | 0.000000 | 0/0 | - | 0.000000 |
| bp35 | 0/0 | - | 0.000000 | 0/0 | - | 0.000000 |
| cd82 | 0/0 | - | 0.000000 | 0/0 | - | 0.000000 |
| cn04 | 0/0 | - | 0.000000 | 0/0 | - | 0.000000 |
| dc22 | 0/0 | - | 0.000000 | 0/0 | - | 0.000000 |
| ft09 | 0/0 | - | 0.000000 | 0/0 | - | 0.000000 |
| g50t | 0/0 | - | 0.000000 | 0/0 | - | 0.000000 |
| ka59 | 0/0 | - | 0.000000 | 0/0 | - | 0.000000 |
| lf52 | 1/1 | 96 | 0.202020 | 1/1 | 159 | 0.073645 |
| lp85 | 1/1 | 672 | 0.001778 | 1/1 | 679 | 0.001741 |
| ls20 | 0/0 | - | 0.000000 | 0/0 | - | 0.000000 |
| m0r0 | 0/0 | - | 0.000000 | 0/0 | - | 0.000000 |
| r11l | 1/1 | 48 | 1.000331 | 1/1 | 845 | 0.003228 |
| re86 | 0/0 | - | 0.000000 | 0/0 | - | 0.000000 |
| s5i5 | 0/0 | - | 0.000000 | 0/0 | - | 0.000000 |
| sb26 | 0/0 | - | 0.000000 | 0/0 | - | 0.000000 |
| sc25 | 0/0 | - | 0.000000 | 0/0 | - | 0.000000 |
| sk48 | 0/0 | - | 0.000000 | 0/0 | - | 0.000000 |
| sp80 | 0/0 | - | 0.000000 | 1/1 | 4187 | 0.000413 |
| su15 | 0/0 | - | 0.000000 | 0/0 | - | 0.000000 |
| tn36 | 1/1 | 140 | 0.186589 | 0/0 | - | 0.000000 |
| tr87 | 0/0 | - | 0.000000 | 0/0 | - | 0.000000 |
| tu93 | 0/0 | - | 0.000000 | 0/0 | - | 0.000000 |
| vc33 | 1/1 | 2099 | 0.000040 | 1/1 | 961 | 0.000189 |
| wa30 | 0/0 | - | 0.000000 | 0/0 | - | 0.000000 |

Each setting completed five levels across the suite. The mean scorecard was
0.0556303% off and 0.00316867% on. Search order gained `sp80` at action 4,187
and lost `tn36`; no game had fewer levels than its matching setting in the
previous paired build. `ft09` and `sc25` exhausted the search before 5,000
actions, without raising exceptions. The development game `ls20` completed no
levels in either setting.

The predeclared acceptance checks all failed: `cd82` did not reach Level 1,
`sp80` reached it only with search order at action 4,187 (threshold 714), and
`tu93` did not reach Level 1 (threshold 1,751).

### Score convention

The score is the local ARC environment scorecard expressed as a percentage,
not the percentage of games solved. For each level, the raw score is
`min((baseline_actions / actions_taken)^2 * 100, 115)`. It is weighted by the
one-based level number, averaged over the game’s total level weights, and
capped by the completed-level weight fraction. RESET actions count toward
actions taken.

The often-quoted `4.7619%` and the `66`-action entry are from different r11l
runs. The October 23 run’s score was `4.761904...%`; the October 22 comparison
entry at 66 actions reported `0.190476%`. The October 23 trace advanced after
four actions and then continued to 64 total actions. In the six-level game,
`4.7619%` is a Level 1 raw score of 100 weighted by `1 / (1+2+3+4+5+6)`.
The 66-action entry must not be paired with that score.

The paired run’s r11l flag-off result is a separate trajectory: Level 1 was
confirmed at action 48. With its 22-action baseline, the raw level score is
about 21.0069, and the Level 1 weight gives `1.000331%`. The flag-on run
confirmed Level 1 at action 845 and scored `0.003228%`.

## Reset/replay fix and known defect

An earlier frontier/reset defect allowed an exhausted root to trigger empty
attempts and unsafe repeated RESETs. The reset-safety changes prevent empty
frontier attempts and avoid issuing a repeated RESET without intervening
progress. With search order off, an unreachable frontier is cleared using the
previous behavior; the reset-safety changes are the only intended behavior
change in that setting.

With search order enabled, a frontier remains pending when no observed replay
path exists, so another opportunity to reach it can occur. If it has no replay
path for three consecutive decisions, it is dropped and another frontier is
selected. This path is opt-in; the environment variable is false when unset.
The final paired runs recorded zero zero-length attempts and zero full-game
resets in all 50 game-runs. A full-game reset is a RESET after which
`levels_completed` drops to zero. These fixes address reset control, not the
broader weakness in finding efficient solutions.

## Limitations

- Each game was tested on one cached layout with one seed (seed 0).
- There is no true held-out set: all 25 games were used during development.
- `ft09` and `sc25` ended on search exhaustion before the nominal action
  budget; this does not establish that the games are unsolvable.
- These are local results, not online competition results, and provide no
  evidence of generalization to unseen layouts.

## License

There is no root-level project license. The vendored
`vendor/ARC-AGI-3-Agents/LICENSE` is MIT; it permits use, modification, and
redistribution subject to retaining the copyright and permission notice, and
includes warranty and liability disclaimers. No new project license is
asserted here.

Detailed runs, previous-build comparisons, and implementation history are in
[`research/system2/BUILDS.md`](research/system2/BUILDS.md).
