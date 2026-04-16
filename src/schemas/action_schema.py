from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Optional

ActionType = Literal["increase", "decrease", "set"]


@dataclass
class ActionSpec:
    """
    Structured action produced by the agent.
    """

    action_type: ActionType
    variable: str
    value: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.variable:
            raise ValueError("variable must be a non-empty string")

        if self.action_type == "set" and self.value is None:
            raise ValueError("value must be provided when action_type='set'")

        if self.action_type in {"increase", "decrease"} and self.value is not None:
            raise ValueError(
                "value must be None when action_type is 'increase' or 'decrease'"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)