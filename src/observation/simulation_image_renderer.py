from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from src.observation.html_simulation_renderer import HtmlSimulationRenderer
from src.schemas.observation_schema import Observation, ObservationMode


class SimulationImageRenderer:
    """
    Renderer for the simulation-only condition.

    It returns a TEXT_IMAGE observation:
    - text: minimal non-numeric control text
    - image_path: current simulation screenshot
    - available_actions: generated from manipulable variables in config
    - visible_state: deliberately non-numeric placeholders, not real state values

    Why visible_state contains no real values
    ----------------------------------------
    In simulation-only, the model must read current and historical state values
    from screenshots. Therefore this renderer avoids putting numeric state in
    Observation.visible_state. This prevents accidental leakage through prompt
    history templates that might include visible_state_before/after.

    The observation text is safe to include in the prompt because it only gives
    generic task-control information, variable names, the target variable, and
    valid actions. It does not include current state values or history values.

    The actual numeric state is still stored separately by EpisodeRunner in
    state_before/state_after for evaluation and logging.
    """

    def __init__(
        self,
        variables: Dict[str, Dict[str, Any]],
        action_mode: str,
        target_variable: str,
        naming_mode: str = "concrete",
        metadata_level: str = "minimal",
        name_mapping: Optional[Dict[str, str]] = None,
        image_renderer: Optional[HtmlSimulationRenderer] = None,
    ) -> None:
        if not isinstance(variables, dict) or not variables:
            raise ValueError("variables must be a non-empty dictionary")

        if action_mode not in {"increase_decrease", "set_value"}:
            raise ValueError(
                "action_mode must be either 'increase_decrease' or 'set_value'"
            )

        if not isinstance(target_variable, str) or not target_variable.strip():
            raise ValueError("target_variable must be a non-empty string")

        if naming_mode not in {"concrete", "abstract"}:
            raise ValueError("naming_mode must be either 'concrete' or 'abstract'")

        if metadata_level not in {"rich", "minimal", "none"}:
            raise ValueError(
                "metadata_level must be one of: 'rich', 'minimal', 'none'"
            )

        if naming_mode == "abstract" and not name_mapping:
            raise ValueError("name_mapping must be provided when naming_mode='abstract'")

        if not isinstance(image_renderer, HtmlSimulationRenderer):
            raise TypeError("image_renderer must be a HtmlSimulationRenderer instance")

        for variable, cfg in variables.items():
            if not isinstance(cfg, dict):
                raise ValueError(f"variable config for '{variable}' must be a dictionary")

        self.variables = deepcopy(variables)
        self.action_mode = action_mode
        self.target_variable = target_variable
        self.naming_mode = naming_mode
        self.metadata_level = metadata_level
        self.name_mapping = dict(name_mapping or {})
        self.image_renderer = image_renderer

        self._reverse_name_mapping = {
            display: internal for internal, display in self.name_mapping.items()
        }

    # -------------------------------------------------------------------------
    # RendererProtocol
    # -------------------------------------------------------------------------

    def render(
        self,
        state: Dict[str, Any],
        history: Optional[List[Dict[str, Any]]] = None,
        step_id: Optional[int] = None,
    ) -> Observation:
        """
        Render current state as a simulation screenshot observation.

        The numeric state is rendered into the image, but not exposed in text.
        """
        if not isinstance(state, dict):
            raise ValueError("state must be a dictionary")

        image_path = self.image_renderer.render(
            state=state,
            history=history,
            step_id=step_id,
        )

        visible_state = self._build_non_numeric_visible_state(state)
        available_actions = self._build_available_actions()
        text = self._build_control_text(available_actions)

        metadata: Dict[str, Any] = {
            "target_variable": self._display_name(self.target_variable),
            "target_variable_internal": self.target_variable,
            "naming_mode": self.naming_mode,
            "metadata_level": self.metadata_level,
            "action_mode": self.action_mode,
            "observation_kind": "simulation_only",
            "simulation_type": self.image_renderer.simulation_type,
            "state_values_in_text": False,
            "state_values_in_image": True,
        }

        if self.name_mapping:
            metadata["name_mapping"] = dict(self.name_mapping)

        return Observation(
            mode=ObservationMode.TEXT_IMAGE,
            visible_state=visible_state,
            available_actions=available_actions,
            text=text,
            image_path=image_path,
            metadata=metadata,
        )

    def to_internal_variable(self, display_name: str) -> str:
        """
        Translate display variable name back to the internal environment name.
        """
        return self._reverse_name_mapping.get(display_name, display_name)

    # -------------------------------------------------------------------------
    # Observation content
    # -------------------------------------------------------------------------

    def _build_non_numeric_visible_state(
        self,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Return placeholders instead of numeric state values.

        This keeps Observation JSON structurally useful while preventing state
        leakage into prompts if the prompt history accidentally includes
        visible_state_before/after.
        """
        visible_state: Dict[str, Any] = {}

        for variable in state.keys():
            display_name = self._display_name(variable)
            visible_state[display_name] = "<read from simulation screenshot>"

        return visible_state

    def _build_available_actions(self) -> List[Dict[str, Any]]:
        """
        Build available actions from config-defined manipulable variables only.
        """
        actions: List[Dict[str, Any]] = []

        for variable, cfg in self.variables.items():
            if not cfg.get("manipulable", False):
                continue

            display_name = self._display_name(variable)

            if self.action_mode == "increase_decrease":
                actions.append(
                    {
                        "action_type": "increase",
                        "variable": display_name,
                    }
                )
                actions.append(
                    {
                        "action_type": "decrease",
                        "variable": display_name,
                    }
                )

            else:  # set_value
                actions.append(
                    {
                        "action_type": "set",
                        "variable": display_name,
                        "value": "<number>",
                    }
                )

        return actions

    def _build_control_text(
        self,
        available_actions: List[Dict[str, Any]],
    ) -> str:
        """
        Build generic non-numeric control text for simulation-only observations.

        This text is safe to include in the prompt. It lists the variables,
        target variable, and valid actions, but never includes current state
        values or historical state values.
        """
        lines: List[str] = []

        lines.append("Visual simulation observation:")
        lines.append("The current experimental state is shown in the simulation screenshot.")
        lines.append("Read all variable values and instrument readings from the image, not from text.")

        if self.metadata_level != "none":
            lines.append("")
            lines.append("Variables:")
            lines.append(", ".join(self._display_name(v) for v in self._state_variable_order()))

            lines.append("")
            lines.append("Target variable:")
            lines.append(self._display_name(self.target_variable))

            lines.append("")
            lines.append("Available actions:")
            for action in available_actions:
                action_type = action["action_type"]
                variable = action["variable"]

                if action_type == "set":
                    lines.append(f"- set {variable} to <number>")
                else:
                    lines.append(f"- {action_type} {variable}")

        return "\n".join(lines)

    def _state_variable_order(self) -> List[str]:
        """
        Return manipulable variables first, then target variable.

        This only controls display order in non-numeric control text.
        """
        ordered: List[str] = []

        for variable, cfg in self.variables.items():
            if cfg.get("manipulable", False):
                ordered.append(variable)

        if self.target_variable not in ordered:
            ordered.append(self.target_variable)

        return ordered

    # -------------------------------------------------------------------------
    # Naming
    # -------------------------------------------------------------------------

    def _display_name(self, variable: str) -> str:
        if self.naming_mode == "abstract":
            return self.name_mapping.get(variable, variable)
        return variable