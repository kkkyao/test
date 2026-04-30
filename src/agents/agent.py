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
            # Include a short, readable excerpt of what we tried to parse.
            # The full raw output is already stored in TraceStep.raw_model_output,
            # so there is no need to dump the entire string into the error message.
            excerpt = cleaned[:300].replace("\n", "\\n")
            raise ValueError(
                f"[json_decode] model output is not valid JSON "
                f"(error: {e.msg} at pos {e.pos}): {excerpt!r}"
            ) from e

        if not isinstance(data, dict):
            raise ValueError("[schema] model output JSON must be an object")

        if "hypothesis" in data:
            raise ValueError("[schema] model output JSON must not contain 'hypothesis'")

        if "step_type" not in data:
            raise ValueError("[schema] model output JSON must contain 'step_type'")

        if not isinstance(data["step_type"], str):
            raise ValueError("[schema] 'step_type' must be a string")

        step_type = data["step_type"]

        if step_type not in _VALID_STEP_TYPES:
            raise ValueError(
                f"[schema] invalid step_type: '{step_type}'. "
                f"Must be one of: {sorted(_VALID_STEP_TYPES)}"
            )

        # reasoning is required on every step
        reasoning = data.get("reasoning")
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError("[schema] 'reasoning' must be a non-empty string on every step")

        # step_type-specific constraints
        if step_type == "action" and data.get("action") is None:
            raise ValueError("[schema] 'action' must be provided when step_type='action'")

        if step_type == "finish" and data.get("action") is not None:
            raise ValueError("[schema] 'action' must be null when step_type='finish'")

        action = None
        if data.get("action") is not None:
            if not isinstance(data["action"], dict):
                raise ValueError("[schema] 'action' must be an object or null")

            action_data = data["action"]

            if "action_type" not in action_data:
                raise ValueError("[schema] 'action.action_type' is required when action is provided")

            if "variable" not in action_data:
                raise ValueError("[schema] 'action.variable' is required when action is provided")

            action_type = action_data.get("action_type")

            if action_type in {"increase", "decrease"} and "value" in action_data:
                raise ValueError(
                    "[schema] 'action.value' must not be present when action_type is "
                    "'increase' or 'decrease'"
                )

            if action_type == "set":
                if "value" not in action_data:
                    raise ValueError(
                        "[schema] 'action.value' is required when action_type='set'"
                    )
                if not isinstance(action_data.get("value"), (int, float)):
                    raise ValueError(
                        "[schema] 'action.value' must be numeric when action_type='set'"
                    )

            action = ActionSpec(
                action_type=action_type,
                variable=action_data.get("variable"),
                value=action_data.get("value"),
            )

        return AgentStep(
            step_type=step_type,
            reasoning=reasoning,
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
        - Literal (unescaped) newlines / tabs inside JSON string values,
          which many 7B instruction-tuned models (e.g. Mistral-7B) produce.

        Strategy:
        1. Strip thinking blocks (<think>...</think>).
        2. Try fenced block extraction.
        3. Walk braces to find the JSON object boundary.
        4. If json.loads still fails, apply _repair() and retry once.
        5. Fall back to returning the stripped text so the caller can
           produce a useful error message.
        """
        text = raw_text.strip()

        if not self.strip_markdown_fences:
            return text

        # 0. Strip Qwen3/Qwen3.5 thinking block if present.
        if "</think>" in text:
            text = text[text.index("</think>") + len("</think>"):].strip()

        # 1. Try fenced block (```json ... ```)
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            candidate = fence_match.group(1).strip()
            result = self._try_loads_with_repair(candidate)
            if result is not None:
                return result

        # 2. Walk braces to find the JSON object boundary
        candidate = self._extract_by_brace_walk(text)
        if candidate is not None:
            result = self._try_loads_with_repair(candidate)
            if result is not None:
                return result
            # Brace-walk found a boundary but repair could not fix it.
            # Return the candidate so the error message shows the JSON fragment
            # rather than the entire raw output (makes W&B run table readable).
            return candidate

        return text

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_by_brace_walk(text: str) -> str | None:
        """
        Find the first '{' and walk characters to locate the matching '}'.

        Returns the extracted substring, or None if no balanced pair is found.
        """
        start = text.find("{")
        if start == -1:
            return None

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
                    return text[start : i + 1]

        return None

    @staticmethod
    def _repair(text: str) -> str:
        """
        Apply lightweight fixes for JSON formatting mistakes that
        7B instruction-tuned models (e.g. Mistral-7B) commonly produce.

        Fixes applied (in order):
        1. Escape literal (unescaped) newlines, carriage returns, and tabs
           that appear inside quoted string values.  This is the most common
           cause of JSONDecodeError when the extracted candidate looks valid
           in a W&B table but still fails json.loads.
        2. Remove trailing commas before } or ]  (e.g.  {"a": 1,} ).
        3. Replace Python-style None / True / False with JSON null / true / false.
        """
        # 1. Escape literal control characters inside quoted strings.
        in_string = False
        escape_next = False
        repaired: list[str] = []

        for ch in text:
            if escape_next:
                escape_next = False
                repaired.append(ch)
                continue
            if ch == "\\" and in_string:
                escape_next = True
                repaired.append(ch)
                continue
            if ch == '"':
                in_string = not in_string
                repaired.append(ch)
                continue
            if in_string:
                if ch == "\n":
                    repaired.append("\\n")
                    continue
                if ch == "\r":
                    repaired.append("\\r")
                    continue
                if ch == "\t":
                    repaired.append("\\t")
                    continue
            repaired.append(ch)

        fixed = "".join(repaired)

        # 2. Trailing commas
        fixed = re.sub(r",(\s*[}\]])", r"\1", fixed)

        # 3. Python literals
        fixed = re.sub(r"\bNone\b", "null", fixed)
        fixed = re.sub(r"\bTrue\b", "true", fixed)
        fixed = re.sub(r"\bFalse\b", "false", fixed)

        return fixed

    @classmethod
    def _try_loads_with_repair(cls, candidate: str) -> str | None:
        """
        Try json.loads on candidate, then on _repair(candidate).

        Returns the (possibly repaired) string if parsing succeeds,
        or None if both attempts fail.
        """
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

        repaired = cls._repair(candidate)
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            return None