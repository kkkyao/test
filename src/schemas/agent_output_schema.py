from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Optional

from src.schemas.action_schema import ActionSpec

# "thought" has been removed as a standalone step type.
# Reasoning is now a required field present on every step, so a dedicated
# thought step is redundant and wastes the exploration budget.
StepType = Literal["hypothesis", "action", "finish"]


@dataclass
class AgentStep:
    """
    Structured single-step output parsed from the model.

    Every step should carry a reasoning field explaining current thinking.
    hypothesis is optional but can appear on any step, including action steps —
    the model does not need a separate step just to state a hypothesis.
    """

    step_type: StepType
    reasoning: Optional[str] = None
    hypothesis: Optional[str] = None
    action: Optional[ActionSpec] = None
    final_equation: Optional[str] = None

    def __post_init__(self) -> None:
        if self.reasoning is not None and not self.reasoning.strip():
            raise ValueError("reasoning must be a non-empty string if provided")

        if self.step_type == "hypothesis":
            if self.hypothesis is None or not self.hypothesis.strip():
                raise ValueError(
                    "hypothesis must be provided when step_type='hypothesis'"
                )

        if self.step_type == "action" and self.action is None:
            raise ValueError("action must be provided when step_type='action'")

        if self.step_type == "finish":
            if self.final_equation is None or not self.final_equation.strip():
                raise ValueError(
                    "final_equation must be provided when step_type='finish'"
                )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)