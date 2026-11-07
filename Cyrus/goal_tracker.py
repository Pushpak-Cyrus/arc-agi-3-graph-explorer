"""Classify environment progress separately from frame-derived clues."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceSignal:
    """One observed result, labeled by its evidential strength."""

    kind: str
    strength: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressReport:
    """Evidence produced by comparing two consecutive observations."""

    signals: List[EvidenceSignal] = field(default_factory=list)
    confirmed_progress: bool = False
    legacy_selection_signal: bool = False

    @property
    def has_visual_observation(self) -> bool:
        return any(signal.strength == "observation" for signal in self.signals)

    @property
    def caused_reset(self) -> bool:
        return any(signal.kind == "reset" for signal in self.signals)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confirmed_progress": self.confirmed_progress,
            "legacy_selection_signal": self.legacy_selection_signal,
            "signals": [asdict(signal) for signal in self.signals],
        }


class GoalTracker:
    """Produce evidence labels without treating visual change as reward.

    ``legacy_selection_signal`` retains the former heuristic (level advance,
    increased blob count, or a new blob value) for diagnostics and comparison.
    The selector should use ``confirmed_progress`` when deciding which
    experiments qualify for high-value reuse.
    """

    @staticmethod
    def _blob_values(state: Optional[Dict[str, Any]]) -> set:
        values = set()
        for blob in (state or {}).get("blobs", []):
            value = blob.get("primary_value")
            if value is None:
                value = blob.get("value")
            if value is not None:
                values.add(value)
        return values

    def compare(
        self,
        previous_state: Optional[Dict[str, Any]],
        current_state: Optional[Dict[str, Any]],
        levels_before: Optional[int],
        levels_after: Optional[int],
        score_before: Optional[float] = None,
        score_after: Optional[float] = None,
        caused_reset: bool = False,
    ) -> ProgressReport:
        report = ProgressReport()

        level_advanced = (
            levels_before is not None
            and levels_after is not None
            and levels_after > levels_before
        )
        if level_advanced:
            report.signals.append(EvidenceSignal(
                kind="level_advanced",
                strength="confirmed_progress",
                details={"before": levels_before, "after": levels_after},
            ))
            report.confirmed_progress = True
            report.legacy_selection_signal = True

        if score_before is not None and score_after is not None:
            if score_after > score_before:
                report.signals.append(EvidenceSignal(
                    kind="score_increased",
                    strength="confirmed_progress",
                    details={"before": score_before, "after": score_after},
                ))
                report.confirmed_progress = True
            elif score_after < score_before:
                report.signals.append(EvidenceSignal(
                    kind="score_decreased",
                    strength="negative_evidence",
                    details={"before": score_before, "after": score_after},
                ))

        if previous_state is not None and current_state is not None:
            previous_blobs = previous_state.get("blobs", [])
            current_blobs = current_state.get("blobs", [])

            if len(current_blobs) > len(previous_blobs):
                report.signals.append(EvidenceSignal(
                    kind="new_blob_count",
                    strength="observation",
                    details={
                        "before": len(previous_blobs),
                        "after": len(current_blobs),
                    },
                ))
                report.legacy_selection_signal = True

            new_values = self._blob_values(current_state) - self._blob_values(previous_state)
            if new_values:
                report.signals.append(EvidenceSignal(
                    kind="new_visual_values",
                    strength="observation",
                    details={"values": sorted(new_values)},
                ))
                report.legacy_selection_signal = True

        if caused_reset:
            report.signals.append(EvidenceSignal(
                kind="reset",
                strength="negative_evidence",
                details={},
            ))

        return report

    @staticmethod
    def summarize_state(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Return compact, JSON-friendly visual context for an experiment."""
        blobs = (state or {}).get("blobs", [])
        values = GoalTracker._blob_values(state)
        return {
            "blob_count": len(blobs),
            "target_count": len((state or {}).get("targets", blobs)),
            "primary_values": sorted(values),
            "state_hash": (state or {}).get("hash"),
        }
