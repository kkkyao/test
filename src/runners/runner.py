from __future__ import annotations

import hashlib
import inspect
import re
from typing import Any, Dict, List, Optional, Tuple

from src.agents.agent import ParseError
from src.agents.protocols import AgentProtocol, RendererProtocol
from src.envs.env import EquationEnv
from src.prompts.prompt_builder import PromptBuilder
from src.schemas.action_schema import ActionSpec
from src.schemas.observation_schema import Observation, ObservationMode
from src.schemas.trace_schema import TraceStep
from src.utils.image_path_utils import validate_image_paths


class EpisodeRunner:
    """
    Run one full exploration episode:
    environment -> renderer -> prompt builder -> agent -> trace

    task_mode
    ---------
    "formula_discovery"  (default)
        The model is expected to submit a final_equation.
        Forced-finish attempts to extract an equation from the model.

    "student_simulation"
        The model explores via screenshots and submits a finish_reason
        (e.g. "I observed that doubling mass halves acceleration").
        No equation is required or extracted.
        The evaluator is not called — pass auto_evaluate=False in config.

    Image history
    -------------
    Every time a renderer returns an Observation with a non-None image_path,
    that path is appended to image_paths_history.  Before each agent.act()
    call the list is sliced to the last image_history_window entries (None
    means keep all).

    Parse error retry
    -----------------
    When agent.act() raises a ValueError (parse error), the runner retries
    up to max_parse_retries times.  On each retry the exact schema error
    message is appended to the prompt so the model can self-correct.
    Only the final failure is recorded in parse_error and breaks the loop.

    Termination guarantee
    ---------------------
    The episode always ends with a finish step.  If the model does not
    voluntarily call finish (max_steps exhausted or a parse_error broke the
    loop), _run_forced_finish() is called once.  If that also fails the
    episode is recorded with finish_reached=False.
    """

    def __init__(
        self,
        env: EquationEnv,
        renderer: RendererProtocol,
        prompt_builder: PromptBuilder,
        agent: AgentProtocol,
        max_steps: int,
        image_history_window: Optional[int] = None,
        task_mode: str = "formula_discovery",
        max_parse_retries: int = 1,
    ) -> None:
        if not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")

        if image_history_window is not None:
            if not isinstance(image_history_window, int) or image_history_window <= 0:
                raise ValueError(
                    "image_history_window must be a positive integer or None"
                )

        if task_mode not in {"formula_discovery", "student_simulation"}:
            raise ValueError(
                "task_mode must be 'formula_discovery' or 'student_simulation'"
            )

        if not isinstance(max_parse_retries, int) or max_parse_retries < 0:
            raise ValueError("max_parse_retries must be a non-negative integer")

        self.env = env
        self.renderer = renderer
        self.prompt_builder = prompt_builder
        self.agent = agent
        self.max_steps = max_steps
        self.image_history_window = image_history_window
        self.task_mode = task_mode
        self.max_parse_retries = max_parse_retries

    def run_episode(self) -> Dict[str, Any]:
        """
        Run a single episode and return structured results.

        Returns
        -------
        dict with keys:
            steps              : lightweight per-step view
            trajectory         : full TraceStep dicts
            final_equation     : equation submitted by model (formula_discovery),
                                 or None (student_simulation)
            finish_reason      : reason string submitted by model
                                 (student_simulation), or None (formula_discovery)
            finish_reached     : whether a finish step was recorded
            finish_step_id     : step index of finish, or None
            num_steps          : total steps executed (excl. forced-finish)
            parse_error        : error message if all retries failed
            forced_finish      : True when finish came from the forced final
                                 prompt rather than a voluntary finish
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

        # Does this episode expect visual observations?
        # Determined once from the initial observation mode so that the flag
        # remains stable even during steps where the observation is temporarily
        # replaced by a plain TEXT observation (e.g. after an invalid action).
        episode_uses_images: bool = observation.mode in {
            ObservationMode.IMAGE, ObservationMode.TEXT_IMAGE
        }

        history_for_prompt: List[Dict[str, Any]] = []
        trajectory_steps: List[TraceStep] = []

        finish_reached = False
        finish_step_id: Optional[int] = None
        final_equation: Optional[str] = None
        finish_reason: Optional[str] = None
        parse_error: Optional[str] = None
        all_failed_parse_attempts: List[Dict[str, Any]] = []
        image_validation_log: List[Dict[str, Any]] = []
        image_error_count: int = 0

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

            # ── Image existence validation ────────────────────────────────────
            # Validate BEFORE calling the agent so missing-image errors are
            # logged separately from parse errors and do not silently masquerade
            # as model failures.  Validation never prevents the call from
            # proceeding — the episode continues regardless, so existing
            # successful runs are unaffected.
            img_validation = validate_image_paths(
                image_paths=images_to_pass if images_to_pass else None,
                requires_images=episode_uses_images,
                step_id=step_id,
            )
            image_validation_log.append(img_validation)
            if img_validation["has_image_error"]:
                image_error_count += 1

            agent_step, raw_output, step_parse_error, step_failed_attempts = (
                self._act_with_retry(
                    prompt=prompt,
                    image_paths=images_to_pass if images_to_pass else None,
                    step_id=step_id,
                )
            )
            all_failed_parse_attempts.extend(step_failed_attempts)

            if step_parse_error:
                parse_error = step_parse_error
                break

            if agent_step.step_type == "finish":
                state_after       = state_before
                observation_after = observation_before
                done              = True
                finish_reached    = True
                finish_step_id    = step_id
                final_equation    = agent_step.final_equation
                finish_reason     = agent_step.finish_reason

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
                finish_reason=agent_step.finish_reason,
                prompt=prompt,
                done=done,
            )

            trajectory_steps.append(trace_step)
            history_for_prompt.append(trace_step.to_dict())

            if done:
                break

        # ── Forced-finish step ────────────────────────────────────────────────
        forced_finish = False
        # Diagnostic fields — populated only when forced-finish is attempted.
        forced_finish_trigger:       Optional[str] = None
        forced_finish_used_images:   bool          = False
        forced_finish_error_type:    Optional[str] = None
        forced_finish_error_message: Optional[str] = None

        if not finish_reached:
            # Record what caused us to skip voluntary finish before calling.
            forced_finish_trigger = "parse_error" if parse_error else "max_steps"

            (
                forced_eq, forced_reason, forced_finish,
                _ff_diags,
            ) = self._run_forced_finish(
                observation=observation,
                history=history_for_prompt,
                image_paths=image_paths_history,
            )
            forced_finish_used_images   = _ff_diags["forced_finish_used_images"]
            forced_finish_error_type    = _ff_diags["forced_finish_error_type"]
            forced_finish_error_message = _ff_diags["forced_finish_error_message"]

            if forced_eq is not None or forced_reason is not None:
                finish_reached = True
                finish_step_id = len(trajectory_steps)
                final_equation = forced_eq
                finish_reason  = forced_reason

        steps = [self._to_step_view(step) for step in trajectory_steps]

        return {
            "steps":                       steps,
            "trajectory":                  [step.to_dict() for step in trajectory_steps],
            "final_equation":              final_equation,
            "finish_reason":               finish_reason,
            "finish_reached":              finish_reached,
            "finish_step_id":              finish_step_id,
            "num_steps":                   len(trajectory_steps),
            "parse_error":                 parse_error,
            "forced_finish":               forced_finish,
            "forced_finish_trigger":       forced_finish_trigger,
            "forced_finish_used_images":   forced_finish_used_images,
            "forced_finish_error_type":    forced_finish_error_type,
            "forced_finish_error_message": forced_finish_error_message,
            "image_paths":                 image_paths_history,
            "parse_error_attempts":        all_failed_parse_attempts,
            "image_validation_log":        image_validation_log,
            "image_error_count":           image_error_count,
            "has_image_error":             image_error_count > 0,
            "max_parse_retries":           self.max_parse_retries,
        }

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _act_with_retry(
        self,
        prompt: str,
        image_paths: Optional[List[str]],
        step_id: int = -1,
    ) -> Tuple[Optional[Any], str, Optional[str], List[Dict[str, Any]]]:
        """
        Call agent.act(), retrying up to max_parse_retries times on ValueError.

        On each retry the exact schema error is appended to the prompt so the
        model can see what it did wrong and self-correct.

        Returns
        -------
        (agent_step, raw_output, parse_error, failed_attempts)
            parse_error is None on success; failed_attempts is [] on success.
            agent_step is None only when all attempts failed.
            failed_attempts holds one dict per failed call with full raw output.
        """
        last_error: Optional[str] = None
        failed_attempts: List[Dict[str, Any]] = []
        prompt_hash = hashlib.sha256(prompt.encode("utf-8", errors="replace")).hexdigest()[:16]

        for attempt in range(self.max_parse_retries + 1):
            current_prompt = prompt
            if attempt > 0 and last_error:
                current_prompt = (
                    prompt
                    + f"\n\n[CORRECTION NEEDED] Your previous response caused a "
                    f"JSON schema error: {last_error}\n"
                    f"Please output a valid JSON object that follows the format exactly. "
                    f"Make sure every required field is present."
                )

            try:
                agent_step, raw_output = self.agent.act(
                    prompt=current_prompt,
                    image_paths=image_paths,
                )
                return agent_step, raw_output, None, failed_attempts
            except ValueError as exc:
                last_error = str(exc)
                failed_attempts.append({
                    "step_id":             step_id,
                    "attempt_index":       attempt,
                    "prompt_hash":         prompt_hash,
                    "image_paths":         list(image_paths) if image_paths else [],
                    "raw_model_output":    getattr(exc, "raw_output", ""),
                    "cleaned_model_output": getattr(exc, "cleaned_output", ""),
                    "parse_error_message": last_error,
                })

        return None, "", last_error, failed_attempts

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
    ) -> tuple[Optional[str], Optional[str], bool, Dict[str, Any]]:
        """
        Make one additional call to collect a terminal output when max_steps
        is exhausted or a parse error broke the main loop.

        Returns
        -------
        (final_equation, finish_reason, forced, diagnostics)

        diagnostics keys:
            forced_finish_used_images   bool         True iff images were passed to
                                                     the VLM generate call.
            forced_finish_error_type    str | None   Exception class name on failure.
            forced_finish_error_message str | None   Exception message (≤400 chars).

        formula_discovery
            Detects whether the agent is text-only (TextLLMAgent) or VLM
            (VisionLanguageAgent) by inspecting the _generate() signature:
            - text-only: _generate(prompt)           → no image_paths parameter
            - VLM:       _generate(prompt, image_paths) → has image_paths parameter

            Previously, VisionLanguageAgent silently fell through because
            _generate(final_prompt) raised TypeError (missing image_paths), which
            was caught by a broad except clause.  This left VLM agents unable to
            recover from parse errors via forced_finish.

        student_simulation
            Returns a synthetic finish_reason without calling the model.
        """
        diags: Dict[str, Any] = {
            "forced_finish_used_images":   False,
            "forced_finish_error_type":    None,
            "forced_finish_error_message": None,
        }

        if self.task_mode == "student_simulation":
            return None, "maximum exploration steps reached", True, diags

        # formula_discovery path
        try:
            final_prompt = self.prompt_builder.build_final_prompt(
                observation=observation,
                history=history,
            )

            generate_fn = getattr(self.agent, "_generate", None)

            if generate_fn is not None:
                # Inspect signature to distinguish TextLLMAgent from VisionLanguageAgent.
                # TextLLMAgent._generate(prompt)                 → 1 param
                # VisionLanguageAgent._generate(prompt, image_paths) → 2 params
                sig_params = list(inspect.signature(generate_fn).parameters)
                needs_images = "image_paths" in sig_params

                if needs_images:
                    # VLM path: pass image history so the model has visual context.
                    images_to_pass = self._window(image_paths or [])
                    diags["forced_finish_used_images"] = bool(images_to_pass)
                    raw_response = generate_fn(final_prompt, images_to_pass)
                else:
                    # Text-only path: unchanged from original behaviour.
                    raw_response = generate_fn(final_prompt)

                equation = self._extract_equation_line(raw_response)
                if equation:
                    return equation, None, True, diags

            else:
                # MockVLMAgent and any agent without _generate: use act().
                images_to_pass = self._window(image_paths or [])
                diags["forced_finish_used_images"] = bool(images_to_pass)

                agent_step, _ = self.agent.act(
                    prompt=final_prompt,
                    image_paths=images_to_pass if images_to_pass else None,
                )
                if (
                    agent_step.step_type == "finish"
                    and agent_step.final_equation
                ):
                    return agent_step.final_equation, None, True, diags

        except Exception as exc:
            diags["forced_finish_error_type"]    = type(exc).__name__
            diags["forced_finish_error_message"] = str(exc)[:400]

        return None, None, False, diags

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
            "finish_reason":  step.finish_reason,
            "done":           step.done,
        }