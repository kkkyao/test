from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Optional

from src.schemas.action_schema import ActionSpec

# Two step types only.
StepType = Literal["action", "finish"]


@dataclass
class AgentStep:
    """
    Structured single-step output parsed from the model.

    finish steps support two mutually exclusive outcome fields:
        final_equation   – used by formula_discovery tasks
        finish_reason    – used by student_simulation tasks
    Exactly one must be provided when step_type='finish'.
    """

    step_type: StepType
    reasoning: Optional[str] = None
    action: Optional[ActionSpec] = None
    final_equation: Optional[str] = None
    finish_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.reasoning is not None and not self.reasoning.strip():
            raise ValueError("reasoning must be a non-empty string if provided")

        if self.step_type == "action" and self.action is None:
            raise ValueError("action must be provided when step_type='action'")

        if self.step_type == "finish":
            has_equation = bool(self.final_equation and self.final_equation.strip())
            has_reason   = bool(self.finish_reason  and self.finish_reason.strip())
            if not has_equation and not has_reason:
                raise ValueError(
                    "finish step must provide either final_equation "
                    "(formula_discovery) or finish_reason (student_simulation)"
                )
            if self.action is not None:
                raise ValueError("action must be null when step_type='finish'")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)