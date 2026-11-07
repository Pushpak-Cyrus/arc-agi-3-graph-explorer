"""Optional local-LLM hypothesis adviser for System 2.

The adviser can suggest testable explanations and rank one already-available
action. It cannot invent or execute actions. System 2 remains the controller,
and every prediction is checked against the next observed frame.

Development backend: an Ollama-compatible chat endpoint on loopback only.
Enable with CYRUS_LOCAL_LLM_ENABLED=1 and set CYRUS_LOCAL_LLM_MODEL.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Sequence, Tuple
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np

from .frame_diff import grid_of


Signature = Tuple[int, Optional[int], Optional[int]]
_OUTCOME_KINDS = {
    "object_moves", "frame_changes", "no_frame_change",
    "confirmed_progress", "unknown",
}
_DIRECTIONS = {"up", "down", "left", "right", "any", "none", "unknown"}


class LocalLLMAdvisor:
    """Ask a local model for bounded, verifiable advice; fail closed on errors."""

    def __init__(self) -> None:
        self.enabled = os.getenv("CYRUS_LOCAL_LLM_ENABLED", "0").strip().lower() in {
            "1", "true", "yes", "on",
        }
        self.model = os.getenv("CYRUS_LOCAL_LLM_MODEL", "qwen3.5:4b").strip()
        self.base_url = os.getenv(
            "CYRUS_LOCAL_LLM_URL", "http://127.0.0.1:11434/api/chat"
        ).strip()
        try:
            self.timeout = max(1.0, min(60.0, float(
                os.getenv("CYRUS_LOCAL_LLM_TIMEOUT", "15")
            )))
        except ValueError:
            self.timeout = 15.0
        self.last_error: Optional[str] = None
        self.calls = 0

    @staticmethod
    def _validate_local_url(url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme != "http" or parsed.hostname not in {
            "127.0.0.1", "localhost", "::1",
        }:
            raise ValueError("Local LLM URL must use HTTP on loopback")
        if parsed.username or parsed.password:
            raise ValueError("Credentials are not accepted in the local LLM URL")
        return url

    @staticmethod
    def _encode_grid(state: Dict) -> str:
        grid = np.asarray(grid_of(state), dtype=np.int16)
        # One character per cell keeps the visual board compact. A-C represent
        # palette values 10-12; row labels preserve spatial orientation.
        alphabet = "0123456789ABCDEF"
        rows = []
        for index, row in enumerate(grid):
            cells = "".join(
                alphabet[int(value)] if 0 <= int(value) < len(alphabet) else "?"
                for value in row
            )
            rows.append(f"{index:02d}:{cells}")
        return "\n".join(rows)

    @staticmethod
    def _normalize_signature(value) -> Optional[Signature]:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return None
        try:
            action = int(value[0])
            x = None if value[1] is None else int(value[1])
            y = None if value[2] is None else int(value[2])
        except (TypeError, ValueError, OverflowError):
            return None
        return action, x, y

    @classmethod
    def _validate_advice(cls, value, candidates: Sequence[Signature]) -> Optional[Dict]:
        if not isinstance(value, dict):
            return None
        try:
            confidence = float(value.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        preferred = cls._normalize_signature(value.get("preferred_action"))
        if preferred not in set(candidates) or confidence < 0.55:
            preferred = None

        hypotheses = []
        raw_hypotheses = value.get("hypotheses", [])
        if isinstance(raw_hypotheses, list):
            for item in raw_hypotheses[:3]:
                if not isinstance(item, dict):
                    continue
                claim = item.get("claim")
                evidence = item.get("evidence")
                falsifier = item.get("falsifier")
                if not all(isinstance(part, str) for part in (claim, evidence, falsifier)):
                    continue
                hypotheses.append({
                    "claim": claim[:180],
                    "evidence": evidence[:220],
                    "falsifier": falsifier[:180],
                })

        expected = value.get("expected_outcome", {})
        if not isinstance(expected, dict):
            expected = {}
        kind = expected.get("kind", "unknown")
        direction = expected.get("direction", "unknown")
        if kind not in _OUTCOME_KINDS:
            kind = "unknown"
        if direction not in _DIRECTIONS:
            direction = "unknown"

        return {
            "hypotheses": hypotheses,
            "preferred_action": preferred,
            "expected_outcome": {"kind": kind, "direction": direction},
            "confidence": confidence,
        }

    def _prompt(
        self,
        state: Dict,
        candidates: Sequence[Signature],
        learned_effects: Dict,
        history: Sequence[Dict],
        top_goals: Sequence,
    ) -> str:
        context = {
            "grid_encoding": "Each row is index:64 cells; 0-9 are palette values 0-9, A=10, B=11, C=12.",
            "grid": self._encode_grid(state),
            "available_action_signatures": [list(sig) for sig in candidates],
            "learned_action_effects": learned_effects,
            "recent_action_outcomes": list(history)[-6:],
            "system2_candidate_goals": list(top_goals)[:5],
        }
        return (
            "You advise a visual puzzle agent. Analyze only the supplied grid and "
            "interaction evidence; do not assume a game identity or invent unseen rules. "
            "Return one JSON object only. Give at most three concise, falsifiable "
            "hypotheses. Select a preferred_action only from the exact available action "
            "signatures; use null if evidence is weak. expected_outcome.kind must be "
            "one of object_moves, frame_changes, no_frame_change, confirmed_progress, "
            "unknown. expected_outcome.direction must be up, down, left, right, any, "
            "none, or unknown. Do not provide hidden chain-of-thought; give only short "
            "evidence and a falsifier. JSON schema: {\"hypotheses\":[{\"claim\":str,"
            "\"evidence\":str,\"falsifier\":str}],\"preferred_action\":[id,x,y]|null,"
            "\"expected_outcome\":{\"kind\":str,\"direction\":str},"
            "\"confidence\":number}.\n\nOBSERVATION:\n"
            + json.dumps(context, separators=(",", ":"), default=str)
        )

    def advise(
        self,
        state: Dict,
        candidates: Sequence[Signature],
        learned_effects: Dict,
        history: Sequence[Dict],
        top_goals: Sequence,
    ) -> Optional[Dict]:
        if not self.enabled:
            return None
        if not candidates or not self.model:
            return None
        try:
            url = self._validate_local_url(self.base_url)
            payload = {
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": self._prompt(
                        state, candidates, learned_effects, history, top_goals
                    ),
                }],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0, "num_predict": 500},
            }
            request = Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body.get("message", {}).get("content", "")
            parsed = json.loads(content) if isinstance(content, str) else content
            advice = self._validate_advice(parsed, candidates)
            if advice is None:
                raise ValueError("Model response did not match the advice schema")
            self.calls += 1
            self.last_error = None
            return advice
        except (OSError, URLError, TimeoutError, ValueError, TypeError,
                KeyError, json.JSONDecodeError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    @staticmethod
    def prediction_correct(advice: Dict, delta, confirmed_progress: bool) -> Optional[bool]:
        """Check the advised action's declared outcome against observed evidence."""
        if not advice or not isinstance(advice, dict):
            return None
        expected = advice.get("expected_outcome", {})
        kind = expected.get("kind", "unknown")
        direction = expected.get("direction", "unknown")
        if kind == "unknown":
            return None
        moved = bool(delta.shifts)
        changed = bool(delta.changed)
        if kind == "object_moves" and not moved:
            return False
        if kind == "frame_changes" and not changed:
            return False
        if kind == "no_frame_change" and changed:
            return False
        if kind == "confirmed_progress" and not confirmed_progress:
            return False
        if kind == "object_moves" and moved and direction not in {"unknown", "any", "none"}:
            vector = {
                "up": (-1, 0), "down": (1, 0),
                "left": (0, -1), "right": (0, 1),
            }[direction]
            if not any(
                (int(np.sign(shift[0])), int(np.sign(shift[1]))) == vector
                for _, shift in delta.shifts
            ):
                return False
        return True
