from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from src.schemas.observation_schema import Observation


@runtime_checkable
class RendererProtocol(Protocol):
    """
    Interface that all renderer implementations must satisfy.

    Both TextRenderer and TextImageRenderer implement this protocol.
    The runner depends only on this interface, not on concrete renderer classes.
    """

    def render(
        self,
        state: Dict[str, Any],
        history: Optional[List[Dict[str, Any]]] = None,
        step_id: Optional[int] = None,
    ) -> Observation:
        """
        Render the current environment state into an Observation.

        Parameters
        ----------
        state:
            Current environment state dict (variable name -> value).
        history:
            List of past TraceStep dicts. May be ignored by text-only renderers.
        step_id:
            Current step index. Used by visual renderers for file naming.
            May be ignored by text-only renderers.

        Returns
        -------
        Observation with mode TEXT, IMAGE, or TEXT_IMAGE depending on implementation.
        """
        ...

    def to_internal_variable(self, display_name: str) -> str:
        """
        Translate a display variable name back to the internal environment name.

        Used by the runner to translate the agent's action (which uses display
        names) into the variable name the environment understands.

        Parameters
        ----------
        display_name:
            The variable name as shown to the model (may be abstract, e.g. "A").

        Returns
        -------
        The internal variable name used by EquationEnv (e.g. "concentration").
        If no mapping exists, returns display_name unchanged.
        """
        ...


@runtime_checkable
class AgentProtocol(Protocol):
    """
    Interface that all agent implementations must satisfy.

    TextLLMAgent, VisionLanguageAgent, and MockVLMAgent all implement
    this protocol. The runner depends only on this interface.
    """

    def act(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
    ) -> tuple:
        """
        Generate one agent step from a prompt and optional image context.

        Parameters
        ----------
        prompt:
            The full prompt string built by PromptBuilder.
        image_paths:
            Ordered list of absolute paths to PNG screenshots, one per past step
            (oldest first, current step last). Text-only agents ignore this.
            None and [] are treated equivalently.

        Returns
        -------
        (AgentStep, raw_model_output_str)
        """
        ...