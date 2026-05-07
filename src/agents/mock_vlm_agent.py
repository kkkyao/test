from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.schemas.action_schema import ActionSpec
from src.schemas.agent_output_schema import AgentStep


class MockVLMAgent:
    """
    A deterministic agent that replays a fixed action sequence.

    Does NOT call any model — ignores both prompt and image_paths.
    Used to debug the full text+image pipeline without needing a real VLM:
    the sequence of actions is predictable, so you can verify that
    - screenshots are generated at every step
    - image_paths_history grows correctly
    - trajectory / logger output is well-formed

    action_sequence format
    ----------------------
    Each element is a dict with one of the following shapes:

        # action step
        {"step_type": "action", "action_type": "increase", "variable": "A"}
        {"step_type": "action", "action_type": "decrease", "variable": "B"}
        {"step_type": "action", "action_type": "set",      "variable": "A", "value": 3.0}

        # finish step
        {"step_type": "finish", "final_equation": "Y = A * B"}

    When the sequence is exhausted:
    - loop=False (default): returns a finish step with equation "unknown"
    - loop=True:            cycles back to the first element

    The agent also produces a raw_output JSON string that mirrors the
    AgentStep it returns, so EpisodeLogger stores a valid record.
    """

    _DEFAULT_FINISH_EQUATION = "unknown"
    _DEFAULT_REASONING = "Mock agent step."

    def __init__(
        self,
        action_sequence: List[Dict[str, Any]],
        loop: bool = False,
    ) -> None:
        if not isinstance(action_sequence, list) or not action_sequence:
            raise ValueError("action_sequence must be a non-empty list")

        for i, item in enumerate(action_sequence):
            if not isinstance(item, dict):
                raise ValueError(f"action_sequence[{i}] must be a dict")
            if "step_type" not in item:
                raise ValueError(f"action_sequence[{i}] must contain 'step_type'")
            if item["step_type"] not in {"action", "finish"}:
                raise ValueError(
                    f"action_sequence[{i}]['step_type'] must be 'action' or 'finish', "
                    f"got '{item['step_type']}'"
                )
            if item["step_type"] == "action":
                if "action_type" not in item:
                    raise ValueError(
                        f"action_sequence[{i}] with step_type='action' "
                        "must contain 'action_type'"
                    )
                if "variable" not in item:
                    raise ValueError(
                        f"action_sequence[{i}] with step_type='action' "
                        "must contain 'variable'"
                    )
            if item["step_type"] == "finish":
                if "final_equation" not in item:
                    raise ValueError(
                        f"action_sequence[{i}] with step_type='finish' "
                        "must contain 'final_equation'"
                    )

        self.action_sequence = list(action_sequence)
        self.loop = loop
        self._index = 0

    # -------------------------------------------------------------------------
    # Public API (AgentProtocol)
    # -------------------------------------------------------------------------

    def act(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
    ) -> tuple[AgentStep, str]:
        """
        Return the next pre-set action.

        Parameters
        ----------
        prompt:
            Ignored.
        image_paths:
            Ignored.  Accepted so MockVLMAgent satisfies AgentProtocol.

        Returns
        -------
        (AgentStep, raw_output_json_str)
        """
        spec = self._next_spec()
        agent_step = self._build_agent_step(spec)
        raw_output = self._build_raw_output(spec)
        return agent_step, raw_output

    def reset(self) -> None:
        """Reset the replay pointer to the start of the sequence."""
        self._index = 0

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _next_spec(self) -> Dict[str, Any]:
        """
        Pop the next action spec from the sequence.

        If the sequence is exhausted:
        - loop=True  → wrap around
        - loop=False → return a synthetic finish step
        """
        if self._index >= len(self.action_sequence):
            if self.loop:
                self._index = 0
            else:
                # Sequence exhausted: force a finish
                return {
                    "step_type": "finish",
                    "final_equation": self._DEFAULT_FINISH_EQUATION,
                }

        spec = self.action_sequence[self._index]
        self._index += 1
        return spec

    def _build_agent_step(self, spec: Dict[str, Any]) -> AgentStep:
        """Convert a spec dict into a validated AgentStep."""
        step_type = spec["step_type"]
        reasoning = spec.get("reasoning", self._DEFAULT_REASONING)

        if step_type == "finish":
            return AgentStep(
                step_type="finish",
                reasoning=reasoning,
                action=None,
                final_equation=spec["final_equation"],
            )

        # step_type == "action"
        action_type = spec["action_type"]
        variable = spec["variable"]
        value = spec.get("value", None)

        action = ActionSpec(
            action_type=action_type,
            variable=variable,
            value=value,
        )

        return AgentStep(
            step_type="action",
            reasoning=reasoning,
            action=action,
            final_equation=None,
        )

    def _build_raw_output(self, spec: Dict[str, Any]) -> str:
        """
        Produce a JSON string that mirrors the AgentStep.

        This is what gets stored in TraceStep.raw_model_output, so the
        logger and interaction_log look the same as with a real agent.
        """
        step_type = spec["step_type"]
        reasoning = spec.get("reasoning", self._DEFAULT_REASONING)

        if step_type == "finish":
            payload: Dict[str, Any] = {
                "step_type": "finish",
                "reasoning": reasoning,
                "action": None,
                "final_equation": spec["final_equation"],
            }
        else:
            action_type = spec["action_type"]
            variable = spec["variable"]
            value = spec.get("value", None)

            action_payload: Dict[str, Any] = {
                "action_type": action_type,
                "variable": variable,
            }
            if action_type == "set" and value is not None:
                action_payload["value"] = value

            payload = {
                "step_type": "action",
                "reasoning": reasoning,
                "action": action_payload,
                "final_equation": None,
            }

        return json.dumps(payload, ensure_ascii=False)