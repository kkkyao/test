from __future__ import annotations

import json
import re
from typing import Callable

from src.schemas.action_schema import ActionSpec
from src.schemas.agent_output_schema import AgentStep

_VALID_STEP_TYPES = {"hypothesis", "action", "finish"}


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
            print(f"[PARSE ERROR] raw output was:\n{raw_text[:500]}", flush=True)
            raise ValueError(f"model output is not valid JSON: {cleaned}") from e

        if not isinstance(data, dict):
            raise ValueError("model output JSON must be an object")

        if "step_type" not in data:
            raise ValueError("model output JSON must contain 'step_type'")

        if not isinstance(data["step_type"], str):
            raise ValueError("'step_type' must be a string")

        step_type = data["step_type"]

        if step_type not in _VALID_STEP_TYPES:
            raise ValueError(
                f"invalid step_type: '{step_type}'. "
                f"Must be one of: {sorted(_VALID_STEP_TYPES)}"
            )

        # reasoning is required on every step
        reasoning = data.get("reasoning")
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError("'reasoning' must be a non-empty string on every step")

        # step_type-specific constraints
        if step_type == "action" and data.get("action") is None:
            raise ValueError("'action' must be provided when step_type='action'")

        if step_type in {"hypothesis", "finish"} and data.get("action") is not None:
            raise ValueError(
                f"'action' must be null when step_type='{step_type}'"
            )

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
                action_type=action_data.get("action_type"),
                variable=action_data.get("variable"),
                value=action_data.get("value"),
            )

        return AgentStep(
            step_type=step_type,
            reasoning=reasoning,
            hypothesis=data.get("hypothesis"),
            action=action,
            final_equation=data.get("final_equation"),
        )

    def _clean_output(self, raw_text: str) -> str:
        """
        Extract a JSON object from model output.

        Handles common real-model output patterns:
        - Pure JSON
        - JSON wrapped in ```json ... ``` fences
        - Prose before/after a fenced JSON block
        - Prose before/after a bare JSON object

        Strategy: find the first '{' and the matching closing '}', then
        verify the extracted substring parses as valid JSON. Falls back
        to returning the stripped text if no braces are found.
        """
        text = raw_text.strip()

        if not self.strip_markdown_fences:
            return text

        # 1. Try to extract from a fenced block first (```json ... ```)
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            candidate = fence_match.group(1).strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        # 2. Find the first '{' and walk to the matching '}'
        start = text.find("{")
        if start == -1:
            return text  # no JSON object found; let caller handle the error

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
                    candidate = text[start : i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break  # malformed; fall through to returning stripped text

        return text