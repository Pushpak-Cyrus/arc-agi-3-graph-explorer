"""Structured, bounded experiment records for CYRUS action outcomes."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ExperimentRecord:
    experiment_id: int
    game_id: str
    step: int
    started_at_utc: str
    state_hash_before: str
    state_features_before: Dict[str, Any]
    action_signature: Tuple[Any, ...]
    action_id: int
    policy: str = "system1"
    selection_reason: str = "system1_evidence_calibrated"
    prediction: Optional[Dict[str, Any]] = None
    state_hash_after: Optional[str] = None
    levels_before: Optional[int] = None
    levels_after: Optional[int] = None
    score_before: Optional[float] = None
    score_after: Optional[float] = None
    caused_reset: bool = False
    terminal: bool = False
    outcome_class: Optional[str] = None
    confirmed_progress: bool = False
    evidence: List[Dict[str, Any]] = field(default_factory=list)


class ExperimentLog:
    """Keep recent records in memory and optionally stream completed JSONL.

    Set ``CYRUS_EXPERIMENT_LOG`` to a JSONL filepath to persist every
    completed experiment. Without it, records remain in a bounded in-memory
    buffer for diagnostics and are not written to disk.
    """

    def __init__(self, game_id: str, max_records: int = 5000,
                 output_path: Optional[str] = None):
        self.game_id = game_id
        self.max_records = max(1, int(max_records))
        self.output_path = output_path or os.getenv("CYRUS_EXPERIMENT_LOG")
        self.records: List[ExperimentRecord] = []
        self.pending: Optional[ExperimentRecord] = None
        self.next_id = 1
        self.dropped_records = 0

    def begin(
        self,
        step: int,
        state_hash: str,
        state_features: Dict[str, Any],
        action_signature: Tuple[Any, ...],
        action_id: int,
        levels_before: Optional[int],
        score_before: Optional[float],
        selection_reason: str = "system1_evidence_calibrated",
    ) -> ExperimentRecord:
        if self.pending is not None:
            self.complete(
                state_hash_after=None,
                levels_after=None,
                score_after=None,
                evidence=[],
                caused_reset=False,
                terminal=False,
                outcome_class="missing_observation",
            )

        record = ExperimentRecord(
            experiment_id=self.next_id,
            game_id=self.game_id,
            step=int(step),
            started_at_utc=datetime.now(timezone.utc).isoformat(),
            state_hash_before=state_hash,
            state_features_before=dict(state_features),
            action_signature=tuple(action_signature),
            action_id=int(action_id),
            selection_reason=selection_reason,
            levels_before=levels_before,
            score_before=score_before,
        )
        self.next_id += 1
        self.records.append(record)
        self.pending = record
        self._trim()
        return record

    def complete(
        self,
        state_hash_after: Optional[str],
        levels_after: Optional[int],
        score_after: Optional[float],
        evidence: List[Dict[str, Any]],
        caused_reset: bool,
        terminal: bool,
        outcome_class: Optional[str] = None,
        confirmed_progress: bool = False,
    ) -> Optional[ExperimentRecord]:
        record = self.pending
        if record is None:
            return None

        record.state_hash_after = state_hash_after
        record.levels_after = levels_after
        record.score_after = score_after
        record.evidence = list(evidence)
        record.caused_reset = bool(caused_reset)
        record.terminal = bool(terminal)
        record.confirmed_progress = bool(confirmed_progress)
        record.outcome_class = outcome_class or self._classify_outcome(record)
        self.pending = None
        self._append_jsonl(record)
        return record

    @staticmethod
    def _classify_outcome(record: ExperimentRecord) -> str:
        if record.confirmed_progress:
            return "confirmed_progress"
        if record.caused_reset:
            return "reset"
        strengths = {item.get("strength") for item in record.evidence}
        if "negative_evidence" in strengths:
            return "negative_evidence"
        if "observation" in strengths:
            return "visual_observation"
        return "no_confirmed_progress"

    def close_pending(self) -> Optional[ExperimentRecord]:
        """Close an outstanding choice if the run exits without an outcome."""
        if self.pending is None:
            return None
        return self.complete(
            state_hash_after=None,
            levels_after=None,
            score_after=None,
            evidence=[],
            caused_reset=False,
            terminal=False,
            outcome_class="incomplete",
        )

    def _append_jsonl(self, record: ExperimentRecord) -> None:
        if not self.output_path:
            return
        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    def _trim(self) -> None:
        overflow = len(self.records) - self.max_records
        if overflow > 0:
            del self.records[:overflow]
            self.dropped_records += overflow

    def get_summary(self) -> Dict[str, int]:
        completed = [
            record for record in self.records
            if record.outcome_class not in (
                None,
                "incomplete",
                "missing_observation",
            )
        ]
        return {
            "records_retained": len(self.records),
            "records_completed": len(completed),
            "confirmed_progress": sum(r.confirmed_progress for r in completed),
            "visual_observations": sum(
                any(item.get("strength") == "observation" for item in r.evidence)
                for r in completed
            ),
            "negative_evidence": sum(
                any(item.get("strength") == "negative_evidence" for item in r.evidence)
                for r in completed
            ),
            "resets": sum(r.caused_reset for r in completed),
            "incomplete": sum(
                r.outcome_class in ("incomplete", "missing_observation")
                for r in self.records
            ),
            "dropped_from_memory": self.dropped_records,
        }
