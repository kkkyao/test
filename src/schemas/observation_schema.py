from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ObservationMode(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TEXT_IMAGE = "text_image"


@dataclass
class Observation:
    """
    Structured observation passed from the environment to the agent.

    visible_state should only contain JSON-serializable values.
    """

    mode: ObservationMode
    visible_state: Dict[str, Any]
    available_actions: List[Dict[str, Any]] = field(default_factory=list)
    text: Optional[str] = None
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            self.mode = ObservationMode(self.mode)

        if not isinstance(self.visible_state, dict):
            raise ValueError("visible_state must be a dictionary")

        if self.mode in {ObservationMode.TEXT, ObservationMode.TEXT_IMAGE} and not self.text:
            raise ValueError(f"text must be provided when mode='{self.mode.value}'")

        if self.mode in {ObservationMode.IMAGE, ObservationMode.TEXT_IMAGE} and not self.image_path:
            raise ValueError(f"image_path must be provided when mode='{self.mode.value}'")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["mode"] = self.mode.value  # ensure JSON-friendly
        return data