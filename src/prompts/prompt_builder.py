from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.schemas.observation_schema import Observation


class PromptBuilder:
    """
    Assemble the model prompt from config-supplied text and runtime data.

    Responsibilities:
        - Read all prompt text (system intro, rules, step-type descriptions,
          output format, section headers) from a prompt config dict.
        - Substitute runtime values ({target_variable}, {max_steps}) via
          str.format().
        - Insert the current observation text and formatted history.
        - Show only the output format template matching the current action_mode.

    The builder owns NO hardcoded strings. Every piece of text the model
    sees must come from the config passed at init time.
    """

    def __init__(
        self,
        prompt_config: Dict[str, Any],
        target_variable: str,
        max_steps: int,
        action_mode: str,
        include_history: bool = True,
        history_window: Optional[int] = None,
    ) -> None:
        if not isinstance(prompt_config, dict) or not prompt_config:
            raise ValueError("prompt_config must be a non-empty dictionary")

        if not isinstance(target_variable, str) or not target_variable.strip():
            raise ValueError("target_variable must be a non-empty string")

        if not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")

        if action_mode not in {"increase_decrease", "set_value"}:
            raise ValueError("action_mode must be 'increase_decrease' or 'set_value'")

        if history_window is not None:
            if not isinstance(history_window, int) or history_window <= 0:
                raise ValueError("history_window must be a positive integer or None")

        self._cfg = prompt_config
        self.target_variable = target_variable
        self.max_steps = max_steps
        self.action_mode = action_mode
        self.include_history = include_history
        self.history_window = history_window

        self._require_keys(
            self._cfg,
            [
                "system_intro",
                "task_template",
                "exploration_lines",
                "step_type_descriptions",
                "rules",
                "output_format_header",
                f"output_format_action_{action_mode}",
                "output_format_finish",
                "section_headers",
                "history_labels",
                "forced_finish_template",          # required for forced-finish flow
            ],
        )
        self._require_keys(
            self._cfg["section_headers"],
            ["step_types", "rules", "observation", "history", "output_format"],
        )
        self._require_keys(
            self._cfg["history_labels"],
            ["empty", "step_prefix", "step_type"],
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def build_prompt(
        self,
        observation: Observation,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build and return the full prompt string for the current step.
        """
        if not isinstance(observation, Observation):
            raise ValueError("observation must be an Observation instance")

        history = history or []
        if not isinstance(history, list):
            raise ValueError("history must be a list")

        display_target = observation.metadata.get(
            "target_variable", self.target_variable
        )

        sections: List[str] = []

        # 1. System intro
        sections.append(self._cfg["system_intro"])

        # 2. Task + exploration instructions
        sections.append("")
        sections.append(
            self._cfg["task_template"].format(target_variable=display_target)
        )
        for line in self._cfg["exploration_lines"]:
            sections.append(line.format(max_steps=self.max_steps))

        # 3. Allowed step types
        sections.append("")
        sections.append(self._cfg["section_headers"]["step_types"])
        for step_type, description in self._cfg["step_type_descriptions"].items():
            sections.append(f"- {step_type}: {description}")

        # 4. Rules
        sections.append("")
        sections.append(self._cfg["section_headers"]["rules"])
        for rule in self._cfg["rules"]:
            sections.append(f"- {rule}")

        # 5. Current observation
        sections.append("")
        sections.append(self._cfg["section_headers"]["observation"])
        sections.append(observation.text or "")

        # 6. History
        sections.append("")
        sections.append(self._cfg["section_headers"]["history"])
        if self.include_history:
            sections.append(self._format_history(history))
        else:
            sections.append(self._cfg["history_labels"]["empty"])

        # 7. Output format
        sections.append("")
        sections.append(self._cfg["section_headers"]["output_format"])
        sections.append(self._cfg["output_format_header"])
        sections.append(self._cfg[f"output_format_action_{self.action_mode}"].rstrip())
        sections.append(self._cfg["output_format_finish"].rstrip())

        return "\n".join(sections)

    def build_final_prompt(
        self,
        observation: Observation,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build a plain-text prompt used when the episode ends without a
        voluntary finish (max_steps reached or parse_error mid-episode).

        The model is asked to write a single equation line — NO JSON.
        This is intentionally simpler than build_prompt() so that even
        weaker models can produce a parseable result.

        The prompt includes:
          - the condensed history so the model has its observations
          - the forced_finish_template from config (plain-text instruction)
        """
        if not isinstance(observation, Observation):
            raise ValueError("observation must be an Observation instance")

        history = history or []
        if not isinstance(history, list):
            raise ValueError("history must be a list")

        display_target = observation.metadata.get(
            "target_variable", self.target_variable
        )

        # Variable names the model has been shown (display names)
        variable_list = ", ".join(observation.visible_state.keys())

        history_text = self._format_history(history)

        template: str = self._cfg["forced_finish_template"]

        sections: List[str] = [
            self._cfg["section_headers"]["history"],
            history_text,
            "",
            template.format(
                target_variable=display_target,
                variable_list=variable_list,
            ).strip(),
        ]

        return "\n".join(sections)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        """
        Convert history dicts into a compact, labelled text block.
        """
        if not history:
            return self._cfg["history_labels"]["empty"]

        if self.history_window is not None:
            history = history[-self.history_window:]

        labels = self._cfg["history_labels"]
        lines: List[str] = []

        for item in history:
            if not isinstance(item, dict):
                raise ValueError("each history item must be a dictionary")

            step_id   = item.get("step_id", "?")
            step_type = item.get("step_type", "?")

            lines.append(labels["step_prefix"].format(step_id=step_id))
            lines.append(labels["step_type"].format(value=step_type))

            reasoning = item.get("reasoning")
            if reasoning and "reasoning" in labels:
                lines.append(labels["reasoning"].format(value=reasoning))

            hypothesis_text = item.get("hypothesis_text") or item.get("hypothesis")
            if hypothesis_text and "hypothesis" in labels:
                lines.append(labels["hypothesis"].format(value=hypothesis_text))

            if step_type == "action":
                parsed_action = item.get("parsed_action") or item.get("action")
                if parsed_action and "action" in labels:
                    lines.append(
                        labels["action"].format(
                            value=self._format_action(parsed_action)
                        )
                    )

                observation_before = item.get("observation_before")
                if isinstance(observation_before, dict) and "visible_state_before" in labels:
                    visible_state = observation_before.get("visible_state")
                    if visible_state:
                        lines.append(
                            labels["visible_state_before"].format(value=visible_state)
                        )

                observation_after = item.get("observation_after")
                if isinstance(observation_after, dict) and "visible_state_after" in labels:
                    visible_state = observation_after.get("visible_state")
                    if visible_state:
                        lines.append(
                            labels["visible_state_after"].format(value=visible_state)
                        )

            lines.append("")

        return "\n".join(lines).strip()

    @staticmethod
    def _format_action(action: Any) -> str:
        if not isinstance(action, dict):
            return str(action)
        action_type = action.get("action_type", "?")
        variable    = action.get("variable", "?")
        value       = action.get("value")
        if action_type == "set" and value is not None:
            return f"set {variable} to {value}"
        if action_type in {"increase", "decrease"}:
            return f"{action_type} {variable}"
        return str(action)

    @staticmethod
    def _require_keys(d: Dict[str, Any], keys: List[str]) -> None:
        missing = [k for k in keys if k not in d]
        if missing:
            raise ValueError(
                f"prompt_config is missing required keys: {missing}"
            )