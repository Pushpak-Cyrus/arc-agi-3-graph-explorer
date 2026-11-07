"""Testable explanations about how actions affect the frame.

Each hypothesis makes a prediction *before* an action is taken and is judged
*after* the outcome is observed. Its belief is a Beta distribution, so one
surprise lowers confidence without erasing everything it has learned.

Hypotheses in this file:

    NoOpAction(a)          action a changes nothing
    ActionChangesFrame(a)  action a changes something
    ClickChangesColour(c)  clicking a pixel of colour c changes something
    ActionDisplacement(a)  action a moves objects by a consistent vector
"""

from collections import Counter
from dataclasses import dataclass
from typing import Optional, Tuple

from .frame_diff import FrameDelta, Shift, grid_of

ActionSig = Tuple  # (action_id, x|None, y|None); ACTION6 carries (x, y)

CLICK_ACTION_ID = 6


@dataclass
class Belief:
    alpha: float = 1.0
    beta: float = 1.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def uncertainty(self) -> float:
        """1.0 at p=0.5 (maximally unsure), 0.0 when fully certain."""
        p = self.mean
        return 4.0 * p * (1.0 - p)

    @property
    def trials(self) -> int:
        return int(self.alpha + self.beta - 2)

    def update(self, correct: bool) -> None:
        if correct:
            self.alpha += 1.0
        else:
            self.beta += 1.0


class Hypothesis:
    """Base class. Subclasses override ``predict`` and optionally ``learn``."""

    name = "hypothesis"

    def __init__(self) -> None:
        self.belief = Belief()

    def predict(self, sig: ActionSig, state, context=None) -> Optional[bool]:
        """True/False = expected to change the frame; None = no opinion."""
        return None

    def outcome_correct(self, prediction, sig: ActionSig, delta: FrameDelta,
                        context=None) -> Optional[bool]:
        if prediction is None:
            return None
        return prediction == delta.changed

    def learn(self, sig: ActionSig, delta: FrameDelta, context=None) -> None:
        """Update non-Bayesian parameters (e.g. an expected displacement)."""


class NoOpAction(Hypothesis):
    def __init__(self, action_id: int) -> None:
        super().__init__()
        self.action_id = action_id
        self.name = f"noop(a{action_id})"

    def predict(self, sig, state, context=None):
        return False if sig[0] == self.action_id else None


class ActionChangesFrame(Hypothesis):
    def __init__(self, action_id: int) -> None:
        super().__init__()
        self.action_id = action_id
        self.name = f"changes(a{action_id})"

    def predict(self, sig, state, context=None):
        return True if sig[0] == self.action_id else None


class ClickChangesColour(Hypothesis):
    def __init__(self, colour: int) -> None:
        super().__init__()
        self.colour = colour
        self.name = f"click_changes(colour={colour})"

    def predict(self, sig, state, context=None):
        if sig[0] != CLICK_ACTION_ID or sig[1] is None or sig[2] is None:
            return None
        grid = grid_of(state)
        x, y = int(sig[1]), int(sig[2])  # x = column, y = row
        if not (0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]):
            return None
        return True if int(grid[y, x]) == self.colour else None


class ActionDisplacement(Hypothesis):
    """A contextual action moves objects by one dominant vector.

    The vector is learned from observed shifts: it becomes active after two
    or more observations and once it accounts for at least 60% of all shifts
    seen after this action. A prediction is correct when some object moves by
    exactly that vector.
    """

    MIN_OBSERVATIONS = 2
    MIN_SHARE = 0.6

    def __init__(self, action_id: int, context=None) -> None:
        super().__init__()
        self.action_id = action_id
        self.context = context
        self.name = f"displace(a{action_id}, context={context!r})"
        self.counts: Counter = Counter()
        self.trials = 0
        self.vector: Optional[Shift] = None

    @property
    def verified(self) -> bool:
        """Require repeated correct predictions, not only a fitted vector."""
        return self.belief.trials >= 2 and self.belief.mean >= 0.75

    def predict(self, sig, state, context=None):
        if (sig[0] != self.action_id or self.vector is None
                or context != self.context):
            return None
        return True

    def outcome_correct(self, prediction, sig, delta, context=None):
        if prediction is None:
            return None
        return any(shift == self.vector for _, shift in delta.shifts)

    def learn(self, sig, delta, context=None):
        if sig[0] != self.action_id or context != self.context:
            return
        self.trials += 1
        # Count each vector once per transition, even if several objects move
        # together, so repeated sprites do not inflate confidence.
        for shift in {shift for _, shift in delta.shifts}:
            self.counts[shift] += 1
        if not self.counts:
            return
        best, n = self.counts.most_common(1)[0]
        if n >= self.MIN_OBSERVATIONS and n / self.trials >= self.MIN_SHARE:
            self.vector = best


class HypothesisLibrary:
    """Keyed, size-capped collection of hypotheses."""

    def __init__(self, max_size: int = 300) -> None:
        self.max_size = max_size
        self.items = {}

    def add(self, key, factory) -> None:
        if key in self.items or len(self.items) >= self.max_size:
            return
        self.items[key] = factory()

    def __contains__(self, key) -> bool:
        return key in self.items

    def __getitem__(self, key) -> Hypothesis:
        return self.items[key]

    def __iter__(self):
        return iter(self.items.items())

    def __len__(self) -> int:
        return len(self.items)
