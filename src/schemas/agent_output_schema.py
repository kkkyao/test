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
    """

    step_type: StepType
    reasoning: Optional[str] = None
    action: Optional[ActionSpec] = None
    final_equation: Optional[str] = None

    def __post_init__(self) -> None:
        if self.reasoning is not None and not self.reasoning.strip():
            raise ValueError("reasoning must be a non-empty string if provided")

        if self.step_type == "action" and self.action is None:
            raise ValueError("action must be provided when step_type='action'")

        if self.step_type == "finish":
            if self.final_equation is None or not self.final_equation.strip():
                raise ValueError(
                    "final_equation must be provided when step_type='finish'"
                )
            if self.action is not None:
                raise ValueError("action must be null when step_type='finish'")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)