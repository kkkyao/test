from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.agents.agent import TextLLMAgent
from src.envs.env import EquationEnv
from src.observation.renderer import TextRenderer
from src.prompts.prompt_builder import PromptBuilder
from src.schemas.action_schema import ActionSpec
from src.schemas.trace_schema import TraceStep


class EpisodeRunner:
    """
    Run one full exploration episode:
    environment -> renderer -> prompt builder -> agent -> trace
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
            steps            : lightweight per-step view
            trajectory       : full TraceStep dicts
            final_equation   : equation submitted by model, or None
            finish_reached   : whether model emitted finish
            finish_step_id   : step index of finish, or None
            num_steps        : total steps executed
            parse_error      : error message if agent.act() raised ValueError
        """
        initial_state = self.env.reset()
        observation = self.renderer.render(initial_state)

        history_for_prompt: List[Dict[str, Any]] = []
        trajectory_steps: List[TraceStep] = []

        finish_reached = False
        finish_step_id: Optional[int] = None
        final_equation: Optional[str] = None
        parse_error: Optional[str] = None

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

            if agent_step.step_type == "action":
                env_action = self._translate_action(agent_step.action)
                state_after = self.env.step(env_action)
                observation = self.renderer.render(state_after)
                observation_after = observation.to_dict()
                done = False

            else:  # finish
                state_after = state_before
                observation_after = observation_before
                done = True
                finish_reached = True
                finish_step_id = step_id
                final_equation = agent_step.final_equation

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
                hypothesis_text=agent_step.hypothesis,
                final_equation=agent_step.final_equation,
                prompt=prompt,
                done=done,
            )

            trajectory_steps.append(trace_step)
            history_for_prompt.append(trace_step.to_dict())

            if done:
                break

        steps = [self._to_step_view(step) for step in trajectory_steps]

        return {
            "steps": steps,
            "trajectory": [step.to_dict() for step in trajectory_steps],
            "final_equation": final_equation,
            "finish_reached": finish_reached,
            "finish_step_id": finish_step_id,
            "num_steps": len(trajectory_steps),
            "parse_error": parse_error,
        }

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
            "step_id": step.step_id,
            "step_type": step.step_type,
            "reasoning": step.reasoning,
            "hypothesis_text": step.hypothesis_text,
            "parsed_action": step.parsed_action,
            "final_equation": step.final_equation,
            "done": step.done,
        }