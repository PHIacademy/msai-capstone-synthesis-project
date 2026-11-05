"""Per-session state of the tutor. Only Tutor methods write to it (Capstone 6, Section 3.3)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionState:
    session_id: str = "session"
    seen_units: set = field(default_factory=set)       # units already served this session
    pending_unit: str | None = None                    # the unit waiting for an answer
    pending: list = field(default_factory=list)        # item_ids of that unit not yet answered
    mastery_tag: dict = field(default_factory=dict)    # sub-topic -> {"attempts", "correct"}
    mastery_module: dict = field(default_factory=dict) # module letter (or "general") -> same
    history: list = field(default_factory=list)        # one record per answered item
    served: list = field(default_factory=list)         # {"unit", "source", "provisional"} per unit served
    turn: int = 0

    @staticmethod
    def _row(table: dict, key: str) -> dict:
        return table.setdefault(key, {"attempts": 0, "correct": 0})
