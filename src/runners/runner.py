from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.agents.agent import TextLLMAgent
from src.envs.env import EquationEnv
from src.observation.renderer import TextRenderer
from src.prompts.prompt_builder import PromptBuilder
from src.schemas.action_schema import ActionSpec
from src.schemas.observation_schema import Observation, ObservationMode
from src.schemas.trace_schema import TraceStep


class EpisodeRunner:
    """
    Run one full exploration episode:
    environment -> renderer -> prompt builder -> agent -> trace

    Termination guarantee
    ---------------------
    The episode always ends with a finish step, so termination_reason
    is always "finish_success" or "finish_wrong" — never "max_steps".

    If the model does not voluntarily call finish (max_steps exhausted
    or a parse_error interrupted the exploration), one additional call
    is made with build_final_prompt(), which asks for a plain-text
    equation.  If that also fails, the episode is recorded with
    finish_reached=False and the original parse_error is preserved.
    """

    def __init__(
        self,
        env: EquationEnv,
        renderer: TextRenderer,
        prompt_builder: PromptBuilder,
        agent: TextLLMAgent,
        max_steps: int,
    ) -> None:
        if not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")

        self.env = env
        self.renderer = renderer
        self.prompt_builder = prompt_builder
        self.agent = agent
        self.max_steps = max_steps

    def run_episode(self) -> Dict[str, Any]:
        """
        Run a single episode and return structured results.

        Returns
        -------
        dict with keys:
            steps              : lightweight per-step view
            trajectory         : full TraceStep dicts
            final_equation     : equation submitted by model, or None
            finish_reached     : whether a finish step was recorded
            finish_step_id     : step index of finish, or None
            num_steps          : total steps executed (excl. forced-finish)
            parse_error        : error message if agent.act() raised ValueError
            forced_finish      : True when the equation came from the forced
                                 final prompt rather than a voluntary finish
        """
        initial_state = self.env.reset()
        observation = self.renderer.render(initial_state)

        history_for_prompt: List[Dict[str, Any]] = []
        trajectory_steps: List[TraceStep] = []

        finish_reached = False
        finish_step_id: Optional[int] = None
        final_equation: Optional[str] = None
        parse_error: Optional[str] = None

        # ── Main exploration loop ─────────────────────────────────────────────
        for step_id in range(self.max_steps):
            state_before = self.env.get_state()
            observation_before = observation.to_dict()

            prompt = self.prompt_builder.build_prompt(
                observation=observation,
                history=history_for_prompt,
            )

            try:
                agent_step, raw_output = self.agent.act(prompt)
            except ValueError as exc:
                parse_error = str(exc)
                break

            if agent_step.step_type == "finish":
                state_after       = state_before
                observation_after = observation_before
                done              = True
                finish_reached    = True
                finish_step_id    = step_id
                final_equation    = agent_step.final_equation

            elif agent_step.step_type == "action":
                env_action = self._translate_action(agent_step.action)
                try:
                    state_after  = self.env.step(env_action)
                    observation  = self.renderer.render(state_after)
                    observation_after = observation.to_dict()

                except (ValueError, KeyError) as exc:
                    error_text = (
                        f"[ERROR] Invalid action: '{env_action.variable}' cannot be manipulated. "
                        f"Only the variables shown in 'Available actions' can be changed. "
                        f"Please choose a valid action.\n\n{observation.text}"
                    )
                    observation = Observation(
                        mode=ObservationMode.TEXT,
                        visible_state=observation.visible_state,
                        available_actions=observation.available_actions,
                        text=error_text,
                        metadata=observation.metadata,
                    )
                    state_after       = state_before
                    observation_after = observation.to_dict()

                done = False

            else:
                state_after       = state_before
                observation_after = observation_before
                done              = False

            trace_step = TraceStep(
                step_id=step_id,
                step_type=agent_step.step_type,
                raw_model_output=raw_output,
                reasoning=agent_step.reasoning,
                parsed_action=agent_step.action.to_dict() if agent_step.action else None,
                observation_before=observation_before,
                observation_after=observation_after,
                state_before=state_before,
                state_after=state_after,
                final_equation=agent_step.final_equation,
                prompt=prompt,
                done=done,
            )

            trajectory_steps.append(trace_step)
            history_for_prompt.append(trace_step.to_dict())

            if done:
                break

        # ── Forced-finish step ────────────────────────────────────────────────
        # If the episode ended without a voluntary finish (max_steps reached
        # or a parse_error broke the loop early), ask the model one more time
        # with a simpler plain-text prompt that does not require JSON.
        forced_finish = False

        if not finish_reached:
            final_equation, forced_finish = self._run_forced_finish(
                observation=observation,
                history=history_for_prompt,
            )
            if final_equation is not None:
                finish_reached = True
                finish_step_id = len(trajectory_steps)  # virtual step index
                # Clear parse_error only if we got something useful back.
                # Keep it as an annotation in the result for diagnostics.

        steps = [self._to_step_view(step) for step in trajectory_steps]

        return {
            "steps":          steps,
            "trajectory":     [step.to_dict() for step in trajectory_steps],
            "final_equation": final_equation,
            "finish_reached": finish_reached,
            "finish_step_id": finish_step_id,
            "num_steps":      len(trajectory_steps),
            "parse_error":    parse_error,
            "forced_finish":  forced_finish,
        }

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _run_forced_finish(
        self,
        observation: Observation,
        history: List[Dict[str, Any]],
    ) -> tuple[Optional[str], bool]:
        """
        Make one additional model call to collect a final equation.

        Uses build_final_prompt() which asks for a plain equation line
        (no JSON), so this succeeds even when the model cannot produce
        well-formed JSON.

        Returns (equation_string, True) on success,
                (None, False) if no usable equation could be extracted.
        """
        try:
            final_prompt = self.prompt_builder.build_final_prompt(
                observation=observation,
                history=history,
            )
            # Bypass full schema validation — we only need a raw string.
            raw_response = self.agent._generate(final_prompt)
            equation = self._extract_equation_line(raw_response)
            if equation:
                return equation, True
        except Exception:
            pass

        return None, False

    @staticmethod
    def _extract_equation_line(raw_text: str) -> Optional[str]:
        """
        Extract an equation from a free-text model response.

        Strategy (in order of preference):
        1. Last non-empty line that contains '='.
        2. Last non-empty line that contains at least one operator (+,-,*,/).
        3. Last non-empty line (last-resort fallback).

        Strips common markdown artefacts (backticks, bold markers).
        Returns None if the response is empty.
        """
        # Strip markdown artefacts (backticks, triple-backtick fences, bold/italic
        # underscores) but NOT '*' which is also the multiplication operator.
        cleaned = re.sub(r"```[a-z]*", "", raw_text)   # opening fence tags
        cleaned = re.sub(r"```", "", cleaned)            # closing fences
        cleaned = re.sub(r"`", "", cleaned)              # inline backticks
        cleaned = cleaned.strip()
        lines = [l.strip() for l in cleaned.splitlines() if l.strip()]

        if not lines:
            return None

        # Preference 1: line with an '=' sign
        for line in reversed(lines):
            if "=" in line:
                # Strip any leading prose before the equation proper.
                # E.g. "The equation is: absorbance = ..." → "absorbance = ..."
                eq_match = re.search(r"[A-Za-z_]\w*\s*=", line)
                if eq_match:
                    return line[eq_match.start():].strip()
                return line

        # Preference 2: line with an arithmetic operator
        for line in reversed(lines):
            if any(op in line for op in ("+", "-", "*", "/")):
                return line

        # Preference 3: last line
        return lines[-1]

    def _translate_action(self, action: ActionSpec) -> ActionSpec:
        """Translate display variable name to internal env name."""
        internal_variable = self.renderer.to_internal_variable(action.variable)
        if internal_variable == action.variable:
            return action
        return ActionSpec(
            action_type=action.action_type,
            variable=internal_variable,
            value=action.value,
        )

    def _to_step_view(self, step: TraceStep) -> Dict[str, Any]:
        return {
            "step_id":        step.step_id,
            "step_type":      step.step_type,
            "reasoning":      step.reasoning,
            "parsed_action":  step.parsed_action,
            "final_equation": step.final_equation,
            "done":           step.done,
        }