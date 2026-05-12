from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.agents.protocols import AgentProtocol, RendererProtocol
from src.envs.env import EquationEnv
from src.prompts.prompt_builder import PromptBuilder
from src.schemas.action_schema import ActionSpec
from src.schemas.observation_schema import Observation, ObservationMode
from src.schemas.trace_schema import TraceStep


class EpisodeRunner:
    """
    Run one full exploration episode:
    environment -> renderer -> prompt builder -> agent -> trace

    Supports both text-only and text+image pipelines via RendererProtocol
    and AgentProtocol.  The runner is agnostic to the concrete renderer and
    agent types: TextRenderer + TextLLMAgent work exactly as before;
    TextImageRenderer + VisionLanguageAgent (or MockVLMAgent) add screenshot
    collection and multi-image VLM calls with no changes to this class.

    Image history
    -------------
    Every time a renderer returns an Observation with a non-None image_path,
    that path is appended to image_paths_history.  Before each agent.act()
    call the list is sliced to the last image_history_window entries (None
    means keep all).  TextLLMAgent ignores image_paths; VisionLanguageAgent
    uses them to pass all historical bar-chart screenshots to the VLM.

    Termination guarantee
    ---------------------
    The episode always ends with a finish step.  If the model does not
    voluntarily call finish (max_steps exhausted or a parse_error broke the
    loop), one additional call is made with build_final_prompt().  If that
    also fails the episode is recorded with finish_reached=False.
    """

    def __init__(
        self,
        env: EquationEnv,
        renderer: RendererProtocol,
        prompt_builder: PromptBuilder,
        agent: AgentProtocol,
        max_steps: int,
        image_history_window: Optional[int] = None,
    ) -> None:
        if not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")

        if image_history_window is not None:
            if not isinstance(image_history_window, int) or image_history_window <= 0:
                raise ValueError(
                    "image_history_window must be a positive integer or None"
                )

        self.env = env
        self.renderer = renderer
        self.prompt_builder = prompt_builder
        self.agent = agent
        self.max_steps = max_steps
        self.image_history_window = image_history_window

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
            image_paths        : ordered list of PNG paths generated this episode
                                 (empty list for text-only runs)
        """
        initial_state = self.env.reset()

        # step_id=0 for the initial observation (before any action is taken)
        observation = self.renderer.render(
            state=initial_state,
            history=[],
            step_id=0,
        )

        # Collect screenshot path produced by the initial render (if any)
        image_paths_history: List[str] = []
        if observation.image_path:
            image_paths_history.append(observation.image_path)

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

            # Apply image_history_window before passing to agent
            images_to_pass = self._window(image_paths_history)

            try:
                agent_step, raw_output = self.agent.act(
                    prompt=prompt,
                    image_paths=images_to_pass if images_to_pass else None,
                )
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
                    state_after = self.env.step(env_action)

                    # Render next observation; step_id+1 because this image
                    # shows the state AFTER the current action.
                    observation = self.renderer.render(
                        state=state_after,
                        history=history_for_prompt,
                        step_id=step_id + 1,
                    )
                    observation_after = observation.to_dict()

                    # Collect screenshot if renderer produced one
                    if observation.image_path:
                        image_paths_history.append(observation.image_path)

                except (ValueError, KeyError, OverflowError) as exc:
                    # Invalid action: keep current observation, inform the model
                    error_text = (
                        f"[ERROR] Invalid action: '{env_action.variable}' cannot be "
                        f"manipulated. Only the variables shown in 'Available actions' "
                        f"can be changed. Please choose a valid action.\n\n"
                        f"{observation.text}"
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
                    # No new screenshot for an invalid step

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
        forced_finish = False

        if not finish_reached:
            final_equation, forced_finish = self._run_forced_finish(
                observation=observation,
                history=history_for_prompt,
                image_paths=image_paths_history,
            )
            if final_equation is not None:
                finish_reached = True
                finish_step_id = len(trajectory_steps)

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
            "image_paths":    image_paths_history,
        }

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _window(self, image_paths: List[str]) -> List[str]:
        """
        Return the last image_history_window entries of image_paths.
        Returns the full list when image_history_window is None.
        """
        if self.image_history_window is None:
            return image_paths
        return image_paths[-self.image_history_window:]

    def _run_forced_finish(
        self,
        observation: Observation,
        history: List[Dict[str, Any]],
        image_paths: Optional[List[str]] = None,
    ) -> tuple[Optional[str], bool]:
        """
        Make one additional model call to collect a final equation.

        For TextLLMAgent: uses build_final_prompt() (plain-text, no JSON)
        via the internal _generate() method.

        For VisionLanguageAgent / MockVLMAgent: falls back to act() with
        image history when available, then tries to extract an equation from
        the response.

        Returns (equation_string, True) on success,
                (None, False) if no usable equation could be extracted.
        """
        try:
            final_prompt = self.prompt_builder.build_final_prompt(
                observation=observation,
                history=history,
            )

            # Path A: agent exposes _generate() (TextLLMAgent)
            # Ask for a plain equation line — no JSON schema required.
            if hasattr(self.agent, "_generate"):
                raw_response = self.agent._generate(final_prompt)  # type: ignore[attr-defined]
                equation = self._extract_equation_line(raw_response)
                if equation:
                    return equation, True

            # Path B: VisionLanguageAgent / MockVLMAgent — call act() with image history
            else:
                images_to_pass = self._window(image_paths or [])

                agent_step, _ = self.agent.act(
                    prompt=final_prompt,
                    image_paths=images_to_pass if images_to_pass else None,
                )
                if (
                    agent_step.step_type == "finish"
                    and agent_step.final_equation
                ):
                    return agent_step.final_equation, True

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
        cleaned = re.sub(r"```[a-z]*", "", raw_text)
        cleaned = re.sub(r"```", "", cleaned)
        cleaned = re.sub(r"`", "", cleaned)
        cleaned = cleaned.strip()
        lines = [l.strip() for l in cleaned.splitlines() if l.strip()]

        if not lines:
            return None

        for line in reversed(lines):
            if "=" in line:
                eq_match = re.search(r"[A-Za-z_]\w*\s*=", line)
                if eq_match:
                    return line[eq_match.start():].strip()
                return line

        for line in reversed(lines):
            if any(op in line for op in ("+", "-", "*", "/")):
                return line

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