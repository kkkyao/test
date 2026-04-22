from __future__ import annotations

import json
import re
from typing import Callable

from src.schemas.action_schema import ActionSpec
from src.schemas.agent_output_schema import AgentStep

_VALID_STEP_TYPES = {"action", "finish"}


class TextLLMAgent:
    """
    Lightweight text-only agent that:
    1. sends a prompt to a model callable
    2. parses the returned JSON into an AgentStep
    """

    def __init__(
        self,
        model_callable: Callable[[str], str],
        strip_markdown_fences: bool = True,
    ) -> None:
        if not callable(model_callable):
            raise ValueError("model_callable must be callable")

        self.model_callable = model_callable
        self.strip_markdown_fences = strip_markdown_fences

    def act(self, prompt: str) -> tuple[AgentStep, str]:
        """
        Generate one model step from a prompt.

        Returns:
            (parsed_agent_step, raw_model_output)
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        raw_output = self._generate(prompt)
        parsed_step = self._parse_output(raw_output)
        return parsed_step, raw_output

    def _generate(self, prompt: str) -> str:
        raw_output = self.model_callable(prompt)
        if not isinstance(raw_output, str):
            raise ValueError("model_callable must return a string")
        return raw_output

    def _parse_output(self, raw_text: str) -> AgentStep:
        cleaned = self._clean_output(raw_text)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"model output is not valid JSON: {cleaned}") from e

        if not isinstance(data, dict):
            raise ValueError("model output JSON must be an object")

        if "step_type" not in data:
            raise ValueError("model output JSON must contain 'step_type'")

        step_type = data["step_type"]
        if not isinstance(step_type, str):
            raise ValueError("'step_type' must be a string")

        if step_type not in _VALID_STEP_TYPES:
            raise ValueError(
                f"invalid step_type: '{step_type}'. "
                f"Must be one of: {sorted(_VALID_STEP_TYPES)}"
            )

        # reasoning is required on every step
        reasoning = data.get("reasoning")
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError("'reasoning' must be a non-empty string on every step")

        # finish: must have final_equation, must not have action
        if step_type == "finish":
            if not data.get("final_equation"):
                raise ValueError("'final_equation' must be non-empty when step_type='finish'")
            if data.get("action") is not None:
                raise ValueError("'action' must be null when step_type='finish'")

        # action: must have action object
        if step_type == "action" and data.get("action") is None:
            raise ValueError("'action' must be provided when step_type='action'")

        # parse action
        action = None
        if data.get("action") is not None:
            if not isinstance(data["action"], dict):
                raise ValueError("'action' must be an object or null")

            action_data = data["action"]

            if "action_type" not in action_data:
                raise ValueError("'action.action_type' is required when action is provided")
            if "variable" not in action_data:
                raise ValueError("'action.variable' is required when action is provided")

            action = ActionSpec(
                action_type=action_data["action_type"],
                variable=action_data["variable"],
                value=action_data.get("value"),
            )

        # hypothesis is optional on any step — light validation only
        hypothesis = data.get("hypothesis")
        if hypothesis is not None and not isinstance(hypothesis, str):
            raise ValueError("'hypothesis' must be a string or null")

        return AgentStep(
            step_type=step_type,
            reasoning=reasoning,
            hypothesis=hypothesis if hypothesis else None,
            action=action,
            final_equation=data.get("final_equation"),
        )

    def _clean_output(self, raw_text: str) -> str:
        """
        Extract a JSON object from model output.

        Handles:
        - Pure JSON
        - JSON wrapped in ```json ... ``` fences
        - Prose before/after a fenced or bare JSON block
        """
        text = raw_text.strip()

        if not self.strip_markdown_fences:
            return text

        # 1. Try fenced block first
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            candidate = fence_match.group(1).strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        # 2. Find first '{' and walk to matching '}'
        start = text.find("{")
        if start == -1:
            return text

        depth = 0
        in_string = False
        escape_next = False

        for i, ch in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start: i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break

        return text