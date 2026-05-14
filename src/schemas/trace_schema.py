from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from src.schemas.agent_output_schema import StepType


@dataclass
class TraceStep:
    """
    Full trajectory record for a single step in an episode.

    step_type is one of:
        "action"  — model took an action (environment updated)
        "finish"  — model ended the episode

    finish steps support two mutually exclusive outcome fields:
        final_equation   – used by formula_discovery tasks
        finish_reason    – used by student_simulation tasks
    """

    step_id: int
    step_type: StepType
    raw_model_output: str
    reasoning: Optional[str] = None
    parsed_action: Optional[Dict[str, Any]] = None
    observation_before: Optional[Dict[str, Any]] = None
    observation_after: Optional[Dict[str, Any]] = None
    state_before: Optional[Dict[str, Any]] = None
    state_after: Optional[Dict[str, Any]] = None
    final_equation: Optional[str] = None
    finish_reason: Optional[str] = None
    prompt: Optional[str] = None
    done: bool = False

    def __post_init__(self) -> None:
        if self.step_id < 0:
            raise ValueError("step_id must be >= 0")

        if not isinstance(self.raw_model_output, str) or not self.raw_model_output.strip():
            raise ValueError("raw_model_output must be a non-empty string")

        if self.step_type == "finish":
            has_equation = bool(self.final_equation and self.final_equation.strip())
            has_reason   = bool(self.finish_reason  and self.finish_reason.strip())
            if not has_equation and not has_reason:
                raise ValueError(
                    "finish step must provide either final_equation "
                    "(formula_discovery) or finish_reason (student_simulation)"
                )
            if not self.done:
                raise ValueError("done must be True when step_type='finish'")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)