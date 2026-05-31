from __future__ import annotations

import json
import re
from typing import Callable, List, Optional

from src.schemas.action_schema import ActionSpec
from src.schemas.agent_output_schema import AgentStep

_VALID_STEP_TYPES = {"action", "finish"}

# Characters that are valid after a backslash in a JSON string literal.
# JSON spec: " \ / b f n r t u  (u must be followed by 4 hex digits)
_VALID_JSON_ESCAPE_CHARS = frozenset('"\\\/bfnrtu')


class ParseError(ValueError):
    """
    ValueError subclass raised when model output cannot be parsed or
    validated against the agent output schema.

    Carries the raw model output and cleaned (JSON-extracted) output so
    callers can log the full text without truncation.
    """

    def __init__(
        self,
        message: str,
        raw_output: str = "",
        cleaned_output: str = "",
    ) -> None:
        super().__init__(message)
        self.raw_output = raw_output
        self.cleaned_output = cleaned_output


class TextLLMAgent:
    """
    Lightweight text-only agent that:
    1. sends a prompt to a model callable
    2. parses the returned JSON into an AgentStep

    The JSON parsing helpers (_parse_output, _clean_output, etc.) are all
    @staticmethod so that VisionLanguageAgent can call them directly without
    inheriting from this class.
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

    def act(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,  # accepted but ignored by text-only agent
    ) -> tuple[AgentStep, str]:
        """
        Generate one model step from a prompt.

        Parameters
        ----------
        prompt:
            The full prompt string to send to the model.
        image_paths:
            Ignored by TextLLMAgent. Accepted so the interface is compatible
            with AgentProtocol and VisionLanguageAgent.

        Returns
        -------
        (parsed_agent_step, raw_model_output)
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        raw_output = self._generate(prompt)
        parsed_step = TextLLMAgent._parse_output(
            raw_output, self.strip_markdown_fences
        )
        return parsed_step, raw_output

    def _generate(self, prompt: str) -> str:
        raw_output = self.model_callable(prompt)

        if not isinstance(raw_output, str):
            raise ValueError("model_callable must return a string")

        return raw_output

    # -------------------------------------------------------------------------
    # Static parsing helpers
    # All methods below are @staticmethod so VisionLanguageAgent can call
    # TextLLMAgent._parse_output(raw_text) without instantiating this class.
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_output(
        raw_text: str,
        strip_markdown_fences: bool = True,
    ) -> AgentStep:
        """
        Parse raw model output into an AgentStep.

        Raises ParseError (a ValueError subclass) on any parsing or schema
        failure.  The exception carries:
          .raw_output     – the original model output string (full, untruncated)
          .cleaned_output – the JSON candidate extracted by _clean_output
        """
        cleaned = TextLLMAgent._clean_output(raw_text, strip_markdown_fences)

        def _fail(msg: str) -> None:
            raise ParseError(msg, raw_output=raw_text, cleaned_output=cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            excerpt = cleaned[:300].replace("\n", "\\n")
            raise ParseError(
                f"[json_decode] model output is not valid JSON "
                f"(error: {e.msg} at pos {e.pos}): {excerpt!r}",
                raw_output=raw_text,
                cleaned_output=cleaned,
            ) from e

        if not isinstance(data, dict):
            _fail("[schema] model output JSON must be an object")

        if "hypothesis" in data:
            _fail("[schema] model output JSON must not contain 'hypothesis'")

        if "step_type" not in data:
            _fail("[schema] model output JSON must contain 'step_type'")

        if not isinstance(data["step_type"], str):
            _fail("[schema] 'step_type' must be a string")

        step_type = data["step_type"]

        if step_type not in _VALID_STEP_TYPES:
            _fail(
                f"[schema] invalid step_type: '{step_type}'. "
                f"Must be one of: {sorted(_VALID_STEP_TYPES)}"
            )

        # reasoning is required on every step
        reasoning = data.get("reasoning")
        if not isinstance(reasoning, str) or not reasoning.strip():
            _fail("[schema] 'reasoning' must be a non-empty string on every step")

        # step_type-specific constraints
        if step_type == "action" and data.get("action") is None:
            _fail("[schema] 'action' must be provided when step_type='action'")

        if step_type == "finish" and data.get("action") is not None:
            _fail("[schema] 'action' must be null when step_type='finish'")

        action = None
        if data.get("action") is not None:
            if not isinstance(data["action"], dict):
                _fail("[schema] 'action' must be an object or null")

            action_data = data["action"]

            if "action_type" not in action_data:
                _fail("[schema] 'action.action_type' is required when action is provided")

            if "variable" not in action_data:
                _fail("[schema] 'action.variable' is required when action is provided")

            action_type = action_data.get("action_type")

            if action_type in {"increase", "decrease"} and "value" in action_data:
                _fail(
                    "[schema] 'action.value' must not be present when action_type is "
                    "'increase' or 'decrease'"
                )

            if action_type == "set":
                if "value" not in action_data:
                    _fail("[schema] 'action.value' is required when action_type='set'")
                if not isinstance(action_data.get("value"), (int, float)):
                    _fail("[schema] 'action.value' must be numeric when action_type='set'")

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
            finish_reason=data.get("finish_reason"),
        )

    @staticmethod
    def _clean_output(
        raw_text: str,
        strip_markdown_fences: bool = True,
    ) -> str:
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

        if not strip_markdown_fences:
            return text

        # 0. Strip Qwen3/Qwen3.5 thinking block if present.
        if "</think>" in text:
            text = text[text.index("</think>") + len("</think>"):].strip()

        # 1. Try fenced block (```json ... ```)
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            candidate = fence_match.group(1).strip()
            result = TextLLMAgent._try_loads_with_repair(candidate)
            if result is not None:
                return result

        # 2. Walk braces to find the JSON object boundary
        candidate = TextLLMAgent._extract_by_brace_walk(text)
        if candidate is not None:
            result = TextLLMAgent._try_loads_with_repair(candidate)
            if result is not None:
                return result
            # Brace-walk found a boundary but repair could not fix it.
            # Return the candidate so the error message shows the JSON fragment
            # rather than the entire raw output (makes W&B run table readable).
            return candidate

        return text

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
        r"""
        Apply lightweight fixes for JSON formatting mistakes that
        instruction-tuned models commonly produce.

        Fixes applied (in order):
        1. Escape literal (unescaped) newlines, carriage returns, and tabs
           inside quoted string values.
        2. Fix invalid JSON backslash escape sequences inside string values.
           JSON only allows: \" \\ \/ \b \f \n \r \t \uXXXX.
           Any other \X (e.g. \epsilon, \frac, \cdot from LaTeX) is made safe
           by doubling the backslash: \epsilon -> \\epsilon.
        3. Remove trailing commas before } or ].
        4. Replace Python-style None / True / False with JSON null / true / false.
        """
        # Steps 1 + 2: character-by-character pass over string values.
        in_string = False
        escape_next = False
        repaired: list[str] = []

        for ch in text:
            if escape_next:
                escape_next = False
                if in_string and ch not in _VALID_JSON_ESCAPE_CHARS:
                    # The preceding backslash (already in repaired) starts an
                    # invalid escape sequence.  Double it so that e.g.
                    # \epsilon becomes \\epsilon (valid JSON -> value \epsilon).
                    repaired[-1] = "\\\\"
                repaired.append(ch)
                continue
            if ch == "\\" and in_string:
                escape_next = True
                repaired.append(ch)  # may be replaced above if next char is invalid
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

        # Step 3. Trailing commas
        fixed = re.sub(r",(\s*[}\]])", r"\1", fixed)

        # Step 4. Python literals
        fixed = re.sub(r"\bNone\b", "null", fixed)
        fixed = re.sub(r"\bTrue\b", "true", fixed)
        fixed = re.sub(r"\bFalse\b", "false", fixed)

        return fixed

    @staticmethod
    def _try_loads_with_repair(candidate: str) -> str | None:
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

        repaired = TextLLMAgent._repair(candidate)
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            return None