from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.observation.html_chart_renderer import HtmlChartRenderer
from src.observation.renderer import TextRenderer
from src.schemas.observation_schema import Observation, ObservationMode


class TextImageRenderer:
    """
    Combine TextRenderer and HtmlChartRenderer into a single renderer
    that satisfies RendererProtocol and returns TEXT_IMAGE observations.

    Responsibilities
    ----------------
    - Delegate text generation to TextRenderer
    - Delegate screenshot generation to HtmlChartRenderer
    - Assemble both outputs into one Observation(mode=TEXT_IMAGE)
    - Forward to_internal_variable() calls to TextRenderer

    The caller (EpisodeRunner) receives a single Observation that contains
    both the text prompt section and the path to the current-step PNG,
    which the runner then adds to image_paths_history before calling the VLM.
    """

    def __init__(
        self,
        text_renderer: TextRenderer,
        chart_renderer: HtmlChartRenderer,
    ) -> None:
        if not isinstance(text_renderer, TextRenderer):
            raise TypeError("text_renderer must be a TextRenderer instance")

        if not isinstance(chart_renderer, HtmlChartRenderer):
            raise TypeError("chart_renderer must be an HtmlChartRenderer instance")

        self.text_renderer = text_renderer
        self.chart_renderer = chart_renderer

    # -------------------------------------------------------------------------
    # RendererProtocol
    # -------------------------------------------------------------------------

    def render(
        self,
        state: Dict[str, Any],
        history: Optional[List[Dict[str, Any]]] = None,
        step_id: Optional[int] = None,
    ) -> Observation:
        """
        Render the current state into a TEXT_IMAGE Observation.

        Parameters
        ----------
        state:
            Current environment state dict (internal variable names → values).
        history:
            Passed through to both sub-renderers.
            TextRenderer ignores it; HtmlChartRenderer also currently ignores
            it (bar chart shows current state only).
        step_id:
            Used by HtmlChartRenderer for PNG/HTML file naming
            (step_0000.png, step_0001.png, …).
            TextRenderer ignores it.

        Returns
        -------
        Observation with mode=TEXT_IMAGE, containing both text and image_path.

        Raises
        ------
        ValueError
            If TextRenderer or HtmlChartRenderer raises during rendering.
        """
        # 1. Generate text observation (never raises on valid state)
        text_obs = self.text_renderer.render(
            state=state,
            history=history,
            step_id=step_id,
        )

        # 2. Generate screenshot and get the absolute PNG path
        image_path = self.chart_renderer.render(
            state=state,
            history=history,
            step_id=step_id,
        )

        # 3. Assemble TEXT_IMAGE observation
        #    visible_state, available_actions, metadata all come from the
        #    text renderer (single source of truth for naming / action lists).
        return Observation(
            mode=ObservationMode.TEXT_IMAGE,
            visible_state=text_obs.visible_state,
            available_actions=text_obs.available_actions,
            text=text_obs.text,
            image_path=image_path,
            metadata=text_obs.metadata,
        )

    def to_internal_variable(self, display_name: str) -> str:
        """
        Translate a display variable name back to the internal env name.

        Delegates to TextRenderer which holds the authoritative name mapping.
        """
        return self.text_renderer.to_internal_variable(display_name)