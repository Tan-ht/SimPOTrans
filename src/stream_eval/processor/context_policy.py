from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ContextRetentionPolicy(Protocol):
    """Policy that decides how much streaming context should be retained."""

    def should_reset_before_append(
        self,
        *,
        current_seconds: float,
        next_chunk_seconds: float,
    ) -> bool:
        """Return True when the current context should be reset before appending."""

    def is_full(self, *, current_seconds: float) -> bool:
        """Return True when the current context has reached its retention limit."""


@dataclass(frozen=True)
class FixedDurationContextPolicy:
    """Retain a streaming context until it reaches a fixed duration."""

    max_seconds: float

    def __post_init__(self) -> None:
        if self.max_seconds <= 0:
            raise ValueError(
                f"max_seconds must be positive for FixedDurationContextPolicy, got {self.max_seconds}."
            )

    def should_reset_before_append(
        self,
        *,
        current_seconds: float,
        next_chunk_seconds: float,
    ) -> bool:
        return current_seconds > 0.0 and current_seconds + next_chunk_seconds >= self.max_seconds

    def is_full(self, *, current_seconds: float) -> bool:
        return current_seconds >= self.max_seconds


def make_context_retention_policy(config: object) -> ContextRetentionPolicy:
    policy_name = str(getattr(config, "context_retention_policy", "fixed_duration"))
    if policy_name in {"fixed_duration", "fixed_seconds", "fixed_window"}:
        return FixedDurationContextPolicy(
            max_seconds=float(getattr(config, "max_speech_context_seconds", 30.0))
        )

    raise ValueError(
        "Unsupported context_retention_policy "
        f"{policy_name!r}. Supported policies: fixed_duration."
    )
