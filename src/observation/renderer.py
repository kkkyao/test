from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from src.schemas.observation_schema import Observation, ObservationMode


class TextRenderer:
    """
    Render environment state into a text-based Observation.
    """

    def __init__(
        self,
        variables: Dict[str, Dict[str, Any]],
        action_mode: str,
        target_variable: str,
        naming_mode: str = "concrete",
        metadata_level: str = "minimal",
        name_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        if not isinstance(variables, dict) or not variables:
            raise ValueError("variables must be a non-empty dictionary")

        if action_mode not in {"increase_decrease", "set_value"}:
            raise ValueError(
                "action_mode must be either 'increase_decrease' or 'set_value'"
            )

        if naming_mode not in {"concrete", "abstract"}:
            raise ValueError("naming_mode must be either 'concrete' or 'abstract'")

        if metadata_level not in {"rich", "minimal", "none"}:
            raise ValueError(
                "metadata_level must be one of: 'rich', 'minimal', 'none'"
            )

        if naming_mode == "abstract" and not name_mapping:
            raise ValueError("name_mapping must be provided when naming_mode='abstract'")

        for variable, cfg in variables.items():
            if not isinstance(cfg, dict):
                raise ValueError(f"variable config for '{variable}' must be a dictionary")

        self.variables = deepcopy(variables)
        self.action_mode = action_mode
        self.target_variable = target_variable
        self.naming_mode = naming_mode
        self.metadata_level = metadata_level
        self.name_mapping = dict(name_mapping or {})

        self._reverse_name_mapping = {v: k for k, v in self.name_mapping.items()}

    def render(self, state: Dict[str, Any]) -> Observation:
        """
        Render the current state into a text Observation.
        """
        if not isinstance(state, dict):
            raise ValueError("state must be a dictionary")

        visible_state = self._build_visible_state(state)
        available_actions = self._build_available_actions()
        text = self._build_text(visible_state, available_actions)

        metadata: Dict[str, Any] = {
            "target_variable": self._display_name(self.target_variable),
            "target_variable_internal": self.target_variable,
            "naming_mode": self.naming_mode,
            "metadata_level": self.metadata_level,
            "action_mode": self.action_mode,
        }

        if self.name_mapping:
            metadata["name_mapping"] = dict(self.name_mapping)

        return Observation(
            mode=ObservationMode.TEXT,
            visible_state=visible_state,
            available_actions=available_actions,
            text=text,
            metadata=metadata,
        )

    def to_internal_variable(self, display_name: str) -> str:
        """
        Translate a display variable name back to the internal (env) name.

        In concrete mode this is an identity operation.
        In abstract mode it maps e.g. "A" -> "concentration".
        Returns the input unchanged if no mapping is found.
        """
        return self._reverse_name_mapping.get(display_name, display_name)

    def _build_visible_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        visible_state: Dict[str, Any] = {}

        for variable, value in state.items():
            display_name = self._display_name(variable)
            visible_state[display_name] = value

        return visible_state

    def _build_available_actions(self) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []

        for variable, cfg in self.variables.items():
            if not cfg.get("manipulable", False):
                continue

            display_name = self._display_name(variable)

            if self.action_mode == "increase_decrease":
                actions.append({"action_type": "increase", "variable": display_name})
                actions.append({"action_type": "decrease", "variable": display_name})
            else:
                actions.append({"action_type": "set", "variable": display_name})

        return actions

    def _build_text(
        self,
        visible_state: Dict[str, Any],
        available_actions: List[Dict[str, Any]],
    ) -> str:
        lines: List[str] = ["Current state:"]

        for variable, value in visible_state.items():
            lines.append(f"{variable} = {value}")

        if self.metadata_level == "rich":
            lines.append("")
            lines.append("Variable descriptions:")
            for display_name in visible_state.keys():
                internal_name = self._internal_name(display_name)
                cfg = self.variables.get(internal_name, {})
                description = cfg.get("description", "No description available.")
                lines.append(f"{display_name}: {description}")

        elif self.metadata_level == "minimal":
            lines.append("")
            lines.append("Variables:")
            lines.append(", ".join(visible_state.keys()))

        lines.append("")
        lines.append("Available actions:")

        for action in available_actions:
            if action["action_type"] == "set":
                lines.append(f"- set {action['variable']} to <value>")
            else:
                lines.append(f"- {action['action_type']} {action['variable']}")

        return "\n".join(lines)

    def _display_name(self, variable: str) -> str:
        if self.naming_mode == "abstract":
            return self.name_mapping.get(variable, variable)
        return variable

    def _internal_name(self, display_name: str) -> str:
        if self.naming_mode == "abstract":
            return self._reverse_name_mapping.get(display_name, display_name)
        return display_name