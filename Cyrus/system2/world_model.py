"""World model: beliefs about how actions affect this environment.

Holds the hypothesis library and the running action statistics. Each step it:

1. creates hypotheses for newly seen actions and colours,
2. collects their predictions before the action,
3. scores those predictions after the outcome (prediction-error learning),
4. lets parameterised hypotheses (displacement) learn from the outcome.
"""

from collections import Counter, deque
from typing import Dict, List, Optional

from .frame_diff import FrameDelta, grid_of
from .hypotheses import (
    CLICK_ACTION_ID,
    ActionChangesFrame,
    ActionDisplacement,
    ClickChangesColour,
    Hypothesis,
    HypothesisLibrary,
    NoOpAction,
)


class WorldModel:
    RECENT_WINDOW = 20

    def __init__(self, max_hypotheses: int = 300) -> None:
        self.library = HypothesisLibrary(max_size=max_hypotheses)
        self.action_tries: Counter = Counter()
        self.action_changes: Counter = Counter()
        self.recent: deque = deque(maxlen=self.RECENT_WINDOW)
        self.validation_history: deque = deque(maxlen=500)
        self.counterexamples: deque = deque(maxlen=100)

    # ------------------------------------------------------------------
    # Hypothesis creation
    # ------------------------------------------------------------------

    def ensure_hypotheses(self, sigs, state, context=None) -> None:
        grid = grid_of(state)
        for sig in sigs:
            action_id = int(sig[0])
            self.library.add(("noop", action_id), lambda a=action_id: NoOpAction(a))
            self.library.add(("changes", action_id), lambda a=action_id: ActionChangesFrame(a))
            if 1 <= action_id <= 6:
                self.library.add(
                    ("displace", action_id, context),
                    lambda a=action_id, c=context: ActionDisplacement(a, c),
                )

            if action_id == CLICK_ACTION_ID and sig[1] is not None and sig[2] is not None:
                x, y = int(sig[1]), int(sig[2])
                if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
                    colour = int(grid[y, x])
                    self.library.add(
                        ("click", colour),
                        lambda c=colour: ClickChangesColour(c),
                    )

    # ------------------------------------------------------------------
    # Prediction and learning
    # ------------------------------------------------------------------

    def predictions(self, sig, state, context=None) -> Dict:
        """Each hypothesis's prediction for this action, before acting."""
        out = {}
        for key, hypothesis in self.library:
            prediction = hypothesis.predict(sig, state, context=context)
            if prediction is not None:
                out[key] = prediction
        return out

    def record(self, sig, delta: FrameDelta, predictions: Dict,
               context=None) -> List[Dict]:
        """Score predictions against the outcome, then update learned parameters.

        Returns the prediction errors (mismatches) for logging.
        """
        errors = []
        for key, prediction in predictions.items():
            hypothesis: Hypothesis = self.library[key]
            correct = hypothesis.outcome_correct(
                prediction, sig, delta, context=context
            )
            if correct is None:
                continue
            hypothesis.belief.update(correct)
            if not correct:
                errors.append({
                    "hypothesis": hypothesis.name,
                    "predicted": prediction,
                    "observed_changed": delta.changed,
                    "observed_shifts": delta.shifts,
                    "context": repr(context),
                })
                self.counterexamples.append(errors[-1])

        for _, hypothesis in self.library:
            hypothesis.learn(sig, delta, context=context)

        self.action_tries[sig] += 1
        if delta.changed:
            self.action_changes[sig] += 1
        self.recent.append(sig)
        self.validation_history.append({
            "sig": tuple(sig),
            "context": context,
            "delta": delta,
            "predicted": dict(predictions),
            "errors": len(errors),
        })
        return errors

    # ------------------------------------------------------------------
    # Queries used by action scoring
    # ------------------------------------------------------------------

    def noop_probability(self, action_id: int) -> float:
        hypothesis = self.library.items.get(("noop", action_id))
        return hypothesis.belief.mean if hypothesis is not None else 0.0

    def repeat_fraction(self, sig) -> float:
        if not self.recent:
            return 0.0
        return self.recent.count(sig) / self.RECENT_WINDOW

    def displacement_vector(self, action_id: int, context=None,
                            verified_only: bool = False):
        hypothesis = self.library.items.get(("displace", action_id, context))
        if verified_only and hypothesis is not None and not hypothesis.verified:
            return None
        return hypothesis.vector if hypothesis is not None else None

    def replay_validation(self, action_id: int, context=None) -> Dict:
        """Recheck a fitted displacement rule against its recorded outcomes.

        This is an explicit consistency check over the observed transition
        ledger. The pre-action predictions in ``validation_history`` remain
        the unbiased, online measure; replay consistency alone is not treated
        as generalization evidence.
        """
        hypothesis = self.library.items.get(("displace", action_id, context))
        if hypothesis is None or hypothesis.vector is None:
            return {"predictions": 0, "correct": 0, "accuracy": None,
                    "verified": False}
        relevant = [
            row for row in self.validation_history
            if row["sig"][0] == action_id and row["context"] == context
        ]
        correct = sum(
            any(shift == hypothesis.vector for _, shift in row["delta"].shifts)
            for row in relevant
        )
        total = len(relevant)
        accuracy = correct / total if total else None
        return {
            "predictions": total,
            "correct": correct,
            "accuracy": round(accuracy, 3) if accuracy is not None else None,
            "online_predictions": hypothesis.belief.trials,
            "online_accuracy": round(hypothesis.belief.mean, 3),
            "verified": bool(
                hypothesis.verified and accuracy is not None and accuracy >= 0.75
            ),
        }

    def summary(self) -> Dict:
        return {
            "hypotheses": len(self.library),
            "actions_tried": int(sum(self.action_tries.values())),
            "displacements": [
                {
                    "action_id": key[1],
                    "context": repr(key[2]),
                    "vector": self.displacement_vector(key[1], key[2]),
                    "verified_vector": self.displacement_vector(
                        key[1], key[2], verified_only=True
                    ),
                    "replay": self.replay_validation(key[1], key[2]),
                }
                for key, _ in self.library if key[0] == "displace"
            ],
            "prediction_ledger": len(self.validation_history),
            "counterexamples": list(self.counterexamples),
        }
