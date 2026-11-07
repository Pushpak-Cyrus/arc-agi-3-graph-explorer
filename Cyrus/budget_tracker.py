"""
Budget Tracker for CYRUS
Tracks time and step budgets across games and levels, so CYRUS doesn't
burn its entire 9-hour Kaggle runtime cap stuck on one hard level.

Key design point: the scoring formula is (human_actions / ai_actions)^2,
which punishes excessive actions quadratically. So giving up efficiently
on a stuck level can score better than eventually solving it the slow way.
"""

import time
from typing import Optional, Dict, List


class BudgetTracker:
    """Tracks wall-clock time and step counts across the whole run."""

    def __init__(self, total_time_budget_seconds: float, games: List[str]):
        """
        Args:
            total_time_budget_seconds: total wall-clock time available
                                        (e.g., 9 hours = 32400 seconds;
                                        leave a safety margin below the
                                        actual cap, e.g. use 8.5 hours)
            games: list of game identifiers that will be played this run
        """
        self.total_time_budget = total_time_budget_seconds
        self.games = games
        self.num_games = len(games)

        self.run_start_time = time.time()
        self.current_game = None
        self.current_game_start_time = None
        self.current_level_start_time = None
        self.current_level_steps = 0
        self.current_game_steps = 0

        # Per-game and per-level history, useful for the paper's results section
        self.game_stats = {g: {"steps": 0, "levels_solved": 0, "time_used": 0.0} for g in games}
        self.level_history = []  # list of dicts: game, level, steps, time, solved

    # ---------- Time accounting ----------

    def elapsed_total(self) -> float:
        """Seconds elapsed since the run started."""
        return time.time() - self.run_start_time

    def remaining_total(self) -> float:
        """Seconds remaining in the total budget."""
        return max(0.0, self.total_time_budget - self.elapsed_total())

    def remaining_games(self) -> int:
        """How many games (including the current one) haven't been finished yet."""
        finished = sum(1 for g in self.games if self.game_stats[g]["time_used"] > 0
                       and g != self.current_game)
        return self.num_games - finished

    def fair_share_for_current_game(self) -> float:
        """
        A simple even split of the remaining time across remaining games.
        This is a starting point — a smarter version could weight by
        observed difficulty, but even splitting is a safe default.
        """
        remaining = max(1, self.remaining_games())
        return self.remaining_total() / remaining

    # ---------- Game/level lifecycle ----------

    def start_game(self, game_id: str) -> None:
        """Call when starting a new game."""
        # The local/Kaggle runner creates one agent per game, while the
        # tracker may be shared across those instances. Close out the prior
        # game's timing and lazily register any game not listed at creation.
        if self.current_game is not None and self.current_game != game_id:
            self.finish_game()

        if game_id not in self.game_stats:
            self.game_stats[game_id] = {
                "steps": 0,
                "levels_solved": 0,
                "time_used": 0.0,
            }
        if game_id not in self.games:
            self.games.append(game_id)
            self.num_games = len(self.games)

        self.current_game = game_id
        self.current_game_start_time = time.time()
        self.current_game_steps = 0

    def start_level(self) -> None:
        """Call when starting a new level (including level 1 of a game)."""
        self.current_level_start_time = time.time()
        self.current_level_steps = 0

    def record_step(self) -> None:
        """Call once per environment step."""
        self.current_level_steps += 1
        self.current_game_steps += 1

    def record_level_result(self, solved: bool) -> None:
        """
        Call when a level ends (solved, or abandoned/reset).
        Records the outcome and resets level-level counters.
        """
        elapsed = time.time() - self.current_level_start_time if self.current_level_start_time else 0.0
        self.level_history.append({
            "game": self.current_game,
            "steps": self.current_level_steps,
            "time": elapsed,
            "solved": solved,
        })
        if solved:
            self.game_stats[self.current_game]["levels_solved"] += 1

    def finish_game(self) -> None:
        """Call when done with a game (all levels attempted or time ran out for it)."""
        if self.current_game:
            elapsed = time.time() - self.current_game_start_time
            self.game_stats[self.current_game]["steps"] = self.current_game_steps
            self.game_stats[self.current_game]["time_used"] = elapsed

    # ---------- Decision helpers (used by the Gate) ----------

    def should_abandon_level(self,
                              max_steps_per_level: Optional[int] = None,
                              max_fraction_of_game_budget: float = 0.3) -> bool:
        """
        Decide whether to give up on the current level and move on,
        rather than keep burning the shared time budget.

        Args:
            max_steps_per_level: hard cap on steps for one level, if set
            max_fraction_of_game_budget: give up if this level alone has
                                          eaten more than this fraction of
                                          the current game's fair time share

        Returns:
            True if CYRUS should stop trying this level and move on
        """
        if max_steps_per_level and self.current_level_steps >= max_steps_per_level:
            return True

        level_elapsed = time.time() - self.current_level_start_time if self.current_level_start_time else 0.0
        game_budget = self.fair_share_for_current_game()
        if game_budget > 0 and level_elapsed >= game_budget * max_fraction_of_game_budget:
            return True

        return False

    def should_stop_run(self, safety_margin_seconds: float = 300.0) -> bool:
        """
        Decide whether to stop the entire run to avoid exceeding the
        Kaggle runtime cap. Leaves a safety margin (default 5 minutes)
        to account for save/submission overhead.
        """
        return self.remaining_total() <= safety_margin_seconds

    # ---------- Reporting (useful for your paper's Results section) ----------

    def get_summary(self) -> Dict:
        """Return a summary suitable for logging or the paper's results table."""
        return {
            "total_elapsed_seconds": self.elapsed_total(),
            "total_remaining_seconds": self.remaining_total(),
            "games": self.game_stats,
            "levels_attempted": len(self.level_history),
            "levels_solved": sum(1 for lv in self.level_history if lv["solved"]),
        }


# Example usage
if __name__ == "__main__":
    # Simulate a short run with a tiny budget for testing
    tracker = BudgetTracker(total_time_budget_seconds=60, games=["ls20", "vc33"])

    tracker.start_game("ls20")
    tracker.start_level()

    for i in range(10):
        tracker.record_step()
        time.sleep(0.01)  # simulate step time

    tracker.record_level_result(solved=True)
    tracker.finish_game()

    print("Summary:", tracker.get_summary())
    print("Should abandon level (max 5 steps)?",
          tracker.should_abandon_level(max_steps_per_level=5))
