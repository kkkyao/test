from __future__ import annotations

from typing import Callable, List, Optional

from src.agents.agent import TextLLMAgent
from src.schemas.agent_output_schema import AgentStep


class VisionLanguageAgent:
    """
    Agent that sends a text prompt together with a list of images to a VLM
    and parses the JSON response into an AgentStep.

    JSON parsing is fully delegated to TextLLMAgent's static helpers, so
    there is no duplicated logic between the two agent types.

    model_callable contract
    -----------------------
    The callable must accept (prompt: str, image_paths: List[str]) and return
    the raw model output as a string.  image_paths is an ordered list of
    absolute PNG paths (oldest first, current step last).  An empty list means
    no images are available for this call.

    Example for Qwen2.5-VL (pseudocode):

        def build_qwen_vl_callable(model, processor):
            def call(prompt, image_paths):
                content = []
                for p in image_paths:
                    content.append({"type": "image", "image": f"file://{p}"})
                content.append({"type": "text", "text": prompt})
                messages = [{"role": "user", "content": content}]
                inputs = processor.apply_chat_template(messages, ...)
                output = model.generate(**inputs)
                return processor.decode(output[0])
            return call
    """

    def __init__(
        self,
        model_callable: Callable[[str, List[str]], str],
        strip_markdown_fences: bool = True,
    ) -> None:
        if not callable(model_callable):
            raise ValueError("model_callable must be callable")

        self.model_callable = model_callable
        self.strip_markdown_fences = strip_markdown_fences

    def act(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
    ) -> tuple[AgentStep, str]:
        """
        Generate one agent step from a prompt and optional image context.

        Parameters
        ----------
        prompt:
            Full prompt string built by PromptBuilder.
        image_paths:
            Ordered list of absolute PNG paths to pass to the VLM.
            Oldest image first, current step last.
            None and [] are treated equivalently (text-only call).

        Returns
        -------
        (parsed_agent_step, raw_model_output)
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        images = image_paths or []

        raw_output = self._generate(prompt, images)
        parsed_step = TextLLMAgent._parse_output(
            raw_output, self.strip_markdown_fences
        )
        return parsed_step, raw_output

    def _generate(self, prompt: str, image_paths: List[str]) -> str:
        """Call the model and validate the response is a string."""
        raw_output = self.model_callable(prompt, image_paths)

        if not isinstance(raw_output, str):
            raise ValueError(
                f"model_callable must return a string, got {type(raw_output).__name__}"
            )

        return raw_output