from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.schemas.observation_schema import Observation


class PromptBuilder:
    """
    Build the model prompt from task instructions, observation, history,
    and output format requirements.
    """

    def __init__(
        self,
        target_variable: str,
        max_steps: int,
        include_history: bool = True,
        history_window: Optional[int] = None,
    ) -> None:
        if not isinstance(target_variable, str) or not target_variable.strip():
            raise ValueError("target_variable must be a non-empty string")

        if not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")

        if history_window is not None:
            if not isinstance(history_window, int) or history_window <= 0:
                raise ValueError("history_window must be a positive integer or None")

        self.target_variable = target_variable
        self.max_steps = max_steps
        self.include_history = include_history
        self.history_window = history_window

    def build_prompt(
        self,
        observation: Observation,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build the current prompt for the model.
        """
        if not isinstance(observation, Observation):
            raise ValueError("observation must be an Observation instance")

        history = history or []
        if not isinstance(history, list):
            raise ValueError("history must be a list")

        display_target = observation.metadata.get(
            "target_variable", self.target_variable
        )

        history_text = self._format_history(history) if self.include_history else "None"

        sections = [
            "You are interacting with a scientific environment.",
            "",
            f"Your goal is to discover the relationship governing {display_target}.",
            "Explore the environment step by step.",
            "Stop only when you are confident enough to output the final equation.",
            f"You may take up to {self.max_steps} steps in total.",
            "",
            "Allowed step types:",
            "- thought: explain what you are considering next",
            "- hypothesis: state a current hypothesis about the variable relationship",
            "- action: choose exactly one valid action from the available actions",
            "- finish: stop exploring and output the final equation",
            "",
            "Rules:",
            "- At each step, return exactly one step_type.",
            "- If step_type is action, choose exactly one valid action.",
            "- If step_type is finish, you must provide final_equation.",
            "- finish is a step_type, not an action.action_type.",
            "- Return only valid JSON.",
            "- Do not include any extra commentary outside the JSON object.",
            "",
            "Current observation:",
            observation.text or "",
            "",
            "History:",
            history_text,
            "",
            "Return a JSON object with this format:",
            "{",
            '  "step_type": "thought | hypothesis | action | finish",',
            '  "reasoning": "string or null",',
            '  "hypothesis": "string or null",',
            '  "action": {',
            '    "action_type": "increase | decrease | set",',
            '    "variable": "string",',
            '    "value": "number or null"',
            '  } or null,',
            '  "final_equation": "string or null"',
            "}",
        ]

        return "\n".join(sections)

    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        """
        Convert structured history into a compact text block.
        """
        if not history:
            return "None"

        if self.history_window is not None:
            history = history[-self.history_window :]

        lines: List[str] = []

        for item in history:
            if not isinstance(item, dict):
                raise ValueError("each history item must be a dictionary")

            step_id = item.get("step_id", "?")
            step_type = item.get("step_type", "?")
            reasoning = item.get("reasoning")
            parsed_action = item.get("parsed_action") or item.get("action")
            hypothesis_text = item.get("hypothesis_text") or item.get("hypothesis")
            final_equation = item.get("final_equation")
            observation_after = item.get("observation_after")

            lines.append(f"Step {step_id}:")
            lines.append(f"- step_type: {step_type}")

            if reasoning:
                lines.append(f"- reasoning: {reasoning}")

            if hypothesis_text:
                lines.append(f"- hypothesis: {hypothesis_text}")

            if parsed_action:
                lines.append(f"- action: {self._format_action(parsed_action)}")

            if final_equation:
                lines.append(f"- final_equation: {final_equation}")

            if isinstance(observation_after, dict):
                visible_state_after = observation_after.get("visible_state")
                if visible_state_after:
                    lines.append(f"- visible_state_after: {visible_state_after}")

            lines.append("")

        return "\n".join(lines).strip()

    @staticmethod
    def _format_action(action: Any) -> str:
        """
        Format a parsed action dict into a human-readable string.
        Falls back to str(action) if the structure is unexpected.
        """
        if not isinstance(action, dict):
            return str(action)

        action_type = action.get("action_type", "?")
        variable = action.get("variable", "?")
        value = action.get("value")

        if action_type == "set" and value is not None:
            return f"set {variable} to {value}"
        if action_type in {"increase", "decrease"}:
            return f"{action_type} {variable}"

        return str(action)