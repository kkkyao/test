from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader, select_autoescape
from playwright.sync_api import sync_playwright

from src.benchmarks.schemas import BenchmarkCase
from src.observation.html_chart_renderer import HtmlChartRenderer
from src.observation.html_simulation_renderer import HtmlSimulationRenderer


class BenchmarkVisualRenderer:
    """
    Render benchmark cases into visual inputs.

    Modalities (original):
    - text        No images.
    - bar         HtmlChartRenderer (existing simulation chart).
    - line        HtmlStepPlotRenderer — parallel-coordinates per-step.
    - scatter     HtmlStepPlotRenderer — parallel-coordinates per-step.
    - simulation  HtmlSimulationRenderer.

    Modalities (abstract visual encoding benchmark):
    - bar_ocr / bar_nocr       Standalone bar chart (shared scale 1–10).
    - slider_ocr / slider_nocr Horizontal slider visualization.
    - dot_ocr / dot_nocr       Dot-position visualization.
    - line_ocr / line_nocr     Time-series line chart (growing history).
    - scatter_ocr / scatter_nocr Time-series scatter (growing history).

    All step-wise modalities produce one image per step in case.states.
    History-aware modalities (line/scatter) produce growing-history images:
    the image at step t shows data for steps 0..t.
    """

    def __init__(self, config: Dict[str, Any], output_dir: str) -> None:
        self.config = config
        self.output_dir = Path(output_dir)

        benchmark_cfg = config["benchmark"]
        variables_cfg = benchmark_cfg["variables"]

        self.target_variable = variables_cfg.get("target_variable")
        self.variable_specs = deepcopy(variables_cfg["specs"])
        self.variable_order = (
            list(variables_cfg["input_variables"]) + [self.target_variable]
        )

        self.visual_cfg = config.get("visual", {})
        self.playwright_cfg = self.visual_cfg.get("playwright", {})

    def render_case(self, case: BenchmarkCase) -> List[str]:
        """
        Render a case into image paths.

        Returns an ordered list of PNG paths.
        Text modality returns an empty list.
        """
        if case.modality == "text":
            return []

        # ── Original modalities ───────────────────────────────────────
        if case.modality == "bar":
            return self._render_bar_sequence(case)

        if case.modality == "line":
            return self._render_step_plot_sequence(case, plot_type="line")

        if case.modality == "scatter":
            return self._render_step_plot_sequence(case, plot_type="scatter")

        if case.modality == "simulation":
            return self._render_simulation_sequence(case)

        # ── Abstract visual encoding modalities ───────────────────────
        if case.modality in {"bar_ocr", "bar_nocr"}:
            return self._render_bar_standalone_sequence(case)

        if case.modality in {"slider_ocr", "slider_nocr"}:
            return self._render_slider_sequence(case)

        if case.modality in {"dot_ocr", "dot_nocr"}:
            return self._render_dot_sequence(case)

        if case.modality in {"grid_ocr", "grid_nocr"}:
            return self._render_grid_sequence(case)

        if case.modality in {"line_ocr", "line_nocr"}:
            return self._render_history_sequence(case, plot_type="line")

        if case.modality in {"scatter_ocr", "scatter_nocr"}:
            return self._render_history_sequence(case, plot_type="scatter")

        raise ValueError(f"Unsupported modality: {case.modality}")

    # ------------------------------------------------------------------
    # Existing bar renderer: one image per step
    # ------------------------------------------------------------------

    def _render_bar_sequence(self, case: BenchmarkCase) -> List[str]:
        bar_cfg = self.visual_cfg.get("bar", {})
        output_dir = self.output_dir / "images" / case.base_id / "bar"

        cached = self._cached_paths(output_dir, len(case.states))
        if cached:
            return cached

        renderer = HtmlChartRenderer(
            variables=self._renderer_variables(),
            target_variable=self.target_variable,
            naming_mode="concrete",
            name_mapping=None,
            output_dir=str(output_dir),
            template_path=bar_cfg.get(
                "template_path",
                "templates/chart_state_view.html",
            ),
            headless=self.playwright_cfg.get("headless", True),
            slow_mo=self.playwright_cfg.get("slow_mo", 0),
            normalize_bars=bar_cfg.get("normalize_bars", True),
            exclude_variables=bar_cfg.get("exclude_variables", []),
        )

        image_paths: List[str] = []

        for step_id, state in enumerate(case.states):
            image_paths.append(
                renderer.render(
                    state=state,
                    history=None,
                    step_id=step_id,
                )
            )

        return image_paths

    # ------------------------------------------------------------------
    # New line/scatter renderer: one image per step
    # ------------------------------------------------------------------

    def _render_step_plot_sequence(
        self,
        case: BenchmarkCase,
        plot_type: str,
    ) -> List[str]:
        """
        Render line/scatter as one image per step.

        Each image shows all variable values for exactly one state.
        This matches the information structure of bar and simulation modalities.
        """
        plot_cfg = self.visual_cfg.get(plot_type, {})
        output_dir = self.output_dir / "images" / case.base_id / plot_type

        cached = self._cached_paths(output_dir, len(case.states))
        if cached:
            return cached

        renderer = HtmlStepPlotRenderer(
            output_dir=str(output_dir),
            template_path=plot_cfg.get(
                "template_path",
                "templates/benchmarks/step_plot.html",
            ),
            plot_type=plot_type,
            variable_specs=self.variable_specs,
            variable_order=self.variable_order,
            target_variable=self.target_variable,
            width=plot_cfg.get("width", 900),
            height=plot_cfg.get("height", 520),
            headless=self.playwright_cfg.get("headless", True),
            slow_mo=self.playwright_cfg.get("slow_mo", 0),
            normalize_y=plot_cfg.get("normalize_y", True),
        )

        image_paths: List[str] = []

        for step_id, state in enumerate(case.states):
            image_paths.append(
                renderer.render(
                    state=state,
                    step_id=step_id,
                )
            )

        return image_paths

    # ------------------------------------------------------------------
    # Existing simulation renderer: one image per step
    # ------------------------------------------------------------------

    def _render_simulation_sequence(self, case: BenchmarkCase) -> List[str]:
        sim_cfg = self.visual_cfg.get("simulation", {})
        output_dir = self.output_dir / "images" / case.base_id / "simulation"

        cached = self._cached_paths(output_dir, len(case.states))
        if cached:
            return cached

        simulation_type = sim_cfg.get("simulation_type")
        if not simulation_type:
            raise ValueError(
                "visual.simulation.simulation_type is required for simulation modality"
            )

        renderer = HtmlSimulationRenderer(
            variables=self._renderer_variables(),
            target_variable=self.target_variable,
            naming_mode="concrete",
            name_mapping=None,
            output_dir=str(output_dir),
            template_dir=sim_cfg.get("template_dir", "templates/simulations"),
            simulation_type=simulation_type,
            headless=self.playwright_cfg.get("headless", True),
            slow_mo=self.playwright_cfg.get("slow_mo", 0),
            exclude_variables=sim_cfg.get("exclude_variables", []),
            viewport_width=sim_cfg.get("viewport", {}).get("width", 900),
            viewport_height=sim_cfg.get("viewport", {}).get("height", 500),
        )

        image_paths: List[str] = []

        for step_id, state in enumerate(case.states):
            image_paths.append(
                renderer.render(
                    state=state,
                    history=None,
                    step_id=step_id,
                )
            )

        return image_paths

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cached_paths(output_dir: Path, n_steps: int) -> List[str]:
        """
        Return resolved PNG paths for step_0000 … step_{n-1} if every file
        already exists, otherwise return an empty list.

        Used to skip re-rendering when a (base_id, modality) directory was
        already rendered by a previous case in the same run.
        """
        paths = [output_dir / f"step_{i:04d}.png" for i in range(n_steps)]
        if all(p.exists() for p in paths):
            return [str(p.resolve()) for p in paths]
        return []

    def _renderer_variables(self) -> Dict[str, Dict[str, Any]]:
        """
        Convert benchmark variable specs into the format expected by existing renderers.
        """
        variables: Dict[str, Dict[str, Any]] = {}

        for var, spec in self.variable_specs.items():
            variables[var] = {
                "min_value": spec.get("min_value"),
                "max_value": spec.get("max_value"),
                "initial_value": spec.get(
                    "initial_value",
                    spec.get("min_value", 0),
                ),
                "manipulable": var != self.target_variable,
            }

        return variables

    # ------------------------------------------------------------------
    # Abstract visual encoding: bar standalone
    # ------------------------------------------------------------------

    def _render_bar_standalone_sequence(self, case: BenchmarkCase) -> List[str]:
        cfg = self.visual_cfg.get("bar_standalone", {})
        show_values = case.modality.endswith("_ocr")
        output_dir = self.output_dir / "images" / case.base_id / case.modality

        cached = self._cached_paths(output_dir, len(case.states))
        if cached:
            return cached

        input_variables = case.metadata.get("input_variables", list(self.variable_specs.keys()))
        renderer = HtmlBarStandaloneRenderer(
            output_dir=str(output_dir),
            template_path=cfg.get(
                "template_path", "templates/benchmarks/bar_standalone.html"
            ),
            input_variables=input_variables,
            variable_specs=self.variable_specs,
            width=cfg.get("width", 900),
            height=cfg.get("height", 500),
            headless=self.playwright_cfg.get("headless", True),
            slow_mo=self.playwright_cfg.get("slow_mo", 0),
        )

        image_paths: List[str] = []
        for step_id, state in enumerate(case.states):
            image_paths.append(
                renderer.render(state=state, step_id=step_id, show_values=show_values)
            )
        return image_paths

    # ------------------------------------------------------------------
    # Abstract visual encoding: slider
    # ------------------------------------------------------------------

    def _render_slider_sequence(self, case: BenchmarkCase) -> List[str]:
        cfg = self.visual_cfg.get("slider", {})
        show_values = case.modality.endswith("_ocr")
        output_dir = self.output_dir / "images" / case.base_id / case.modality

        cached = self._cached_paths(output_dir, len(case.states))
        if cached:
            return cached

        input_variables = case.metadata.get("input_variables", list(self.variable_specs.keys()))
        renderer = HtmlSliderRenderer(
            output_dir=str(output_dir),
            template_path=cfg.get(
                "template_path", "templates/benchmarks/slider_view.html"
            ),
            input_variables=input_variables,
            variable_specs=self.variable_specs,
            width=cfg.get("width", 900),
            height=cfg.get("height", 500),
            headless=self.playwright_cfg.get("headless", True),
            slow_mo=self.playwright_cfg.get("slow_mo", 0),
        )

        image_paths: List[str] = []
        for step_id, state in enumerate(case.states):
            image_paths.append(
                renderer.render(state=state, step_id=step_id, show_values=show_values)
            )
        return image_paths

    # ------------------------------------------------------------------
    # Abstract visual encoding: dot plot (vertical dot chart)
    # ------------------------------------------------------------------

    def _render_dot_sequence(self, case: BenchmarkCase) -> List[str]:
        cfg = self.visual_cfg.get("dot", {})
        show_values = case.modality.endswith("_ocr")
        output_dir = self.output_dir / "images" / case.base_id / case.modality

        cached = self._cached_paths(output_dir, len(case.states))
        if cached:
            return cached

        input_variables = case.metadata.get("input_variables", list(self.variable_specs.keys()))
        renderer = HtmlDotPlotRenderer(
            output_dir=str(output_dir),
            template_path=cfg.get(
                "template_path", "templates/benchmarks/dot_plot_view.html"
            ),
            input_variables=input_variables,
            variable_specs=self.variable_specs,
            width=cfg.get("width", 900),
            height=cfg.get("height", 500),
            headless=self.playwright_cfg.get("headless", True),
            slow_mo=self.playwright_cfg.get("slow_mo", 0),
        )

        image_paths: List[str] = []
        for step_id, state in enumerate(case.states):
            image_paths.append(
                renderer.render(state=state, step_id=step_id, show_values=show_values)
            )
        return image_paths

    # ------------------------------------------------------------------
    # Abstract visual encoding: filled grid
    # ------------------------------------------------------------------

    def _render_grid_sequence(self, case: BenchmarkCase) -> List[str]:
        cfg = self.visual_cfg.get("grid", {})
        show_values = case.modality.endswith("_ocr")
        output_dir = self.output_dir / "images" / case.base_id / case.modality

        cached = self._cached_paths(output_dir, len(case.states))
        if cached:
            return cached

        input_variables = case.metadata.get("input_variables", list(self.variable_specs.keys()))
        renderer = HtmlGridRenderer(
            output_dir=str(output_dir),
            template_path=cfg.get(
                "template_path", "templates/benchmarks/grid_view.html"
            ),
            input_variables=input_variables,
            variable_specs=self.variable_specs,
            width=cfg.get("width", 900),
            height=cfg.get("height", 500),
            headless=self.playwright_cfg.get("headless", True),
            slow_mo=self.playwright_cfg.get("slow_mo", 0),
        )

        image_paths: List[str] = []
        for step_id, state in enumerate(case.states):
            image_paths.append(
                renderer.render(state=state, step_id=step_id, show_values=show_values)
            )
        return image_paths

    # ------------------------------------------------------------------
    # Abstract visual encoding: history chart (line / scatter)
    # ------------------------------------------------------------------

    def _render_history_sequence(
        self, case: BenchmarkCase, plot_type: str
    ) -> List[str]:
        cfg = self.visual_cfg.get("history_chart", {})
        show_values = case.modality.endswith("_ocr")
        output_dir = self.output_dir / "images" / case.base_id / case.modality

        cached = self._cached_paths(output_dir, len(case.states))
        if cached:
            return cached

        input_variables = case.metadata.get("input_variables", list(self.variable_specs.keys()))
        renderer = HtmlHistoryChartRenderer(
            output_dir=str(output_dir),
            template_path=cfg.get(
                "template_path", "templates/benchmarks/history_chart.html"
            ),
            plot_type=plot_type,
            input_variables=input_variables,
            variable_specs=self.variable_specs,
            width=cfg.get("width", 900),
            height=cfg.get("height", 520),
            headless=self.playwright_cfg.get("headless", True),
            slow_mo=self.playwright_cfg.get("slow_mo", 0),
        )

        image_paths: List[str] = []
        for step_id in range(len(case.states)):
            # Pass growing history: states 0 .. step_id (inclusive).
            history_states = case.states[: step_id + 1]
            image_paths.append(
                renderer.render(
                    states=history_states,
                    current_step_id=step_id,
                    show_values=show_values,
                )
            )
        return image_paths


class HtmlStepPlotRenderer:
    """
    Render one line/scatter image per step.

    Each image shows the variable values for a single state:

        x-axis = variable names
        y-axis = normalized values
        point labels = raw numeric values

    This makes line/scatter comparable to bar and simulation modalities:
    all visual modalities provide an ordered sequence of step images.
    """

    COLORS = {
        "normal": "#2563eb",
        "target": "#dc2626",
    }

    def __init__(
        self,
        output_dir: str,
        template_path: str,
        plot_type: str,
        variable_specs: Dict[str, Dict[str, Any]],
        variable_order: List[str],
        target_variable: str,
        width: int = 900,
        height: int = 520,
        headless: bool = True,
        slow_mo: int = 0,
        normalize_y: bool = True,
    ) -> None:
        if plot_type not in {"line", "scatter"}:
            raise ValueError("plot_type must be 'line' or 'scatter'")

        if not os.path.isfile(template_path):
            raise FileNotFoundError(f"template not found: {template_path}")

        self.output_dir = Path(output_dir).resolve()
        self.html_dir = self.output_dir / "html"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.html_dir.mkdir(parents=True, exist_ok=True)

        self.template_path = Path(template_path).resolve()
        self.plot_type = plot_type
        self.variable_specs = variable_specs
        self.variable_order = variable_order
        self.target_variable = target_variable
        self.width = int(width)
        self.height = int(height)
        self.headless = headless
        self.slow_mo = slow_mo
        self.normalize_y = normalize_y

        env = Environment(
            loader=FileSystemLoader(str(self.template_path.parent)),
            autoescape=select_autoescape(["html"]),
        )
        self.template = env.get_template(self.template_path.name)

    def render(
        self,
        state: Dict[str, float],
        step_id: int,
    ) -> str:
        render_data = self._build_render_data(
            state=state,
            step_id=step_id,
        )
        html_path = self._render_html(
            render_data=render_data,
            step_id=step_id,
        )
        image_path = self._take_screenshot(
            html_path=html_path,
            step_id=step_id,
        )
        return image_path

    def _build_render_data(
        self,
        state: Dict[str, float],
        step_id: int,
    ) -> Dict[str, Any]:
        margin_left = 80
        margin_right = 40
        margin_top = 55
        margin_bottom = 95

        plot_width = self.width - margin_left - margin_right
        plot_height = self.height - margin_top - margin_bottom

        n_vars = len(self.variable_order)
        if n_vars <= 1:
            raise ValueError("step plot requires at least two variables")

        points: List[Dict[str, Any]] = []

        for idx, var in enumerate(self.variable_order):
            if var not in state:
                raise KeyError(f"state is missing variable '{var}'")

            raw_value = float(state[var])

            x = margin_left + (idx / (n_vars - 1)) * plot_width
            norm = self._normalize_value(var, raw_value)
            y = margin_top + (1.0 - norm) * plot_height

            is_target = var == self.target_variable

            points.append(
                {
                    "variable": var,
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "value": raw_value,
                    "value_label": self._format_value(raw_value),
                    "is_target": is_target,
                    "color": (
                        self.COLORS["target"]
                        if is_target
                        else self.COLORS["normal"]
                    ),
                }
            )

        path = " ".join(
            [
                ("M" if i == 0 else "L") + f" {p['x']} {p['y']}"
                for i, p in enumerate(points)
            ]
        )

        x_ticks = [
            {
                "x": p["x"],
                "label": p["variable"],
                "is_target": p["is_target"],
            }
            for p in points
        ]

        y_ticks = []
        for value in [0, 0.25, 0.5, 0.75, 1.0]:
            y = margin_top + (1.0 - value) * plot_height
            y_ticks.append(
                {
                    "y": round(y, 2),
                    "label": f"{value:.2f}",
                }
            )

        return {
            "case_id": case_id,
            "step_id": step_id,
            "plot_type": self.plot_type,
            "title": f"{self.plot_type.capitalize()} plot of variable values",
            "width": self.width,
            "height": self.height,
            "margin_left": margin_left,
            "margin_right": margin_right,
            "margin_top": margin_top,
            "margin_bottom": margin_bottom,
            "plot_width": plot_width,
            "plot_height": plot_height,
            "points": points,
            "path": path,
            "x_ticks": x_ticks,
            "y_ticks": y_ticks,
            "x_axis_y": margin_top + plot_height,
            "y_axis_x": margin_left,
            "target_variable": self.target_variable,
            "normalize_y": self.normalize_y,
            "y_axis_label": "Normalized value; point labels show raw values",
        }

    def _normalize_value(self, var: str, value: float) -> float:
        spec = self.variable_specs.get(var, {})

        if self.normalize_y:
            lo = float(spec.get("min_value", 0.0))
            hi = float(spec.get("max_value", max(value, 1.0)))
        else:
            lows = [
                float(self.variable_specs[v].get("min_value", 0.0))
                for v in self.variable_order
            ]
            highs = [
                float(self.variable_specs[v].get("max_value", 1.0))
                for v in self.variable_order
            ]
            lo = min(lows)
            hi = max(highs)

        if hi <= lo:
            return 0.5

        norm = (value - lo) / (hi - lo)
        return max(0.0, min(1.0, norm))

    def _render_html(
        self,
        render_data: Dict[str, Any],
        step_id: int,
    ) -> str:
        html_content = self.template.render(**render_data)
        html_path = self.html_dir / f"step_{step_id:04d}.html"
        html_path.write_text(html_content, encoding="utf-8")
        return str(html_path.resolve())

    def _take_screenshot(
        self,
        html_path: str,
        step_id: int,
        _max_retries: int = 3,
    ) -> str:
        image_path = self.output_dir / f"step_{step_id:04d}.png"
        file_uri = Path(html_path).resolve().as_uri()

        last_exc: Exception | None = None

        for attempt in range(_max_retries):
            try:
                with sync_playwright() as pw:
                    browser = pw.chromium.launch(
                        headless=self.headless,
                        slow_mo=self.slow_mo,
                    )
                    page = browser.new_page(
                        viewport={
                            "width": self.width,
                            "height": self.height,
                        }
                    )
                    try:
                        page.goto(file_uri, wait_until="networkidle")
                        page.screenshot(path=str(image_path), full_page=False)
                    finally:
                        page.close()
                        browser.close()

                return str(image_path.resolve())

            except Exception as exc:
                last_exc = exc
                if attempt < _max_retries - 1:
                    import time
                    time.sleep(1)

        raise RuntimeError(
            f"Screenshot failed after {_max_retries} attempts: {last_exc}"
        ) from last_exc

    @staticmethod
    def _format_value(value: float) -> str:
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.2f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# Shared screenshot mixin
# ---------------------------------------------------------------------------

class _ScreenshotMixin:
    """Playwright screenshot helper shared by the new abstract renderers."""

    _width: int
    _height: int
    _headless: bool
    _slow_mo: int

    def _take_screenshot(
        self,
        html_path: str,
        image_path: "Path",
        _max_retries: int = 3,
    ) -> str:
        file_uri = Path(html_path).resolve().as_uri()
        last_exc: Exception | None = None

        for attempt in range(_max_retries):
            try:
                with sync_playwright() as pw:
                    browser = pw.chromium.launch(
                        headless=self._headless,
                        slow_mo=self._slow_mo,
                    )
                    page = browser.new_page(
                        viewport={"width": self._width, "height": self._height}
                    )
                    try:
                        page.goto(file_uri, wait_until="networkidle")
                        page.screenshot(path=str(image_path), full_page=False)
                    finally:
                        page.close()
                        browser.close()
                return str(image_path.resolve())
            except Exception as exc:
                last_exc = exc
                if attempt < _max_retries - 1:
                    import time
                    time.sleep(1)

        raise RuntimeError(
            f"Screenshot failed after {_max_retries} attempts: {last_exc}"
        ) from last_exc

    @staticmethod
    def _fmt(value: float) -> str:
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.2f}".rstrip("0").rstrip(".")

    @staticmethod
    def _make_ticks(lo: float, hi: float, step: float) -> List[Dict[str, Any]]:
        """Return tick list with pct positions for a track from lo to hi."""
        ticks: List[Dict[str, Any]] = []
        span = hi - lo
        if span <= 0:
            return ticks
        v = lo
        while v <= hi + 1e-9:
            pct = (v - lo) / span * 100.0
            ticks.append({"pct": round(pct, 2)})
            v += step
        return ticks


# ---------------------------------------------------------------------------
# HtmlBarStandaloneRenderer
# ---------------------------------------------------------------------------

class HtmlBarStandaloneRenderer(_ScreenshotMixin):
    """
    Benchmark-specific single-panel bar chart.
    Shared scale across all input variables.
    Supports show_values flag for OCR/non-OCR conditions.
    """

    def __init__(
        self,
        output_dir: str,
        template_path: str,
        input_variables: List[str],
        variable_specs: Dict[str, Dict[str, Any]],
        width: int = 900,
        height: int = 500,
        headless: bool = True,
        slow_mo: int = 0,
    ) -> None:
        if not os.path.isfile(template_path):
            raise FileNotFoundError(f"Template not found: {template_path}")

        self._output_dir = Path(output_dir).resolve()
        self._html_dir = self._output_dir / "html"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._html_dir.mkdir(parents=True, exist_ok=True)

        self._input_variables = input_variables
        self._specs = variable_specs
        self._width = width
        self._height = height
        self._headless = headless
        self._slow_mo = slow_mo

        tp = Path(template_path).resolve()
        env = Environment(
            loader=FileSystemLoader(str(tp.parent)),
            autoescape=select_autoescape(["html"]),
        )
        self._template = env.get_template(tp.name)

    def render(
        self,
        state: Dict[str, float],
        step_id: int,
        show_values: bool,
    ) -> str:
        # Shared scale: use the union of all variable ranges.
        all_lo = min(
            float(self._specs[v].get("min_value", 1)) for v in self._input_variables
        )
        all_hi = max(
            float(self._specs[v].get("max_value", 10)) for v in self._input_variables
        )
        span = all_hi - all_lo if all_hi > all_lo else 1.0

        bars = []
        for var in self._input_variables:
            value = float(state.get(var, all_lo))
            norm = max(0.0, min(1.0, (value - all_lo) / span))
            bars.append({
                "name": var,
                "value": value,
                "norm_height": norm,
            })

        html = self._template.render(
            step_id=step_id,
            bars=bars,
            show_values=show_values,
            width=self._width,
            height=self._height,
        )
        html_path = self._html_dir / f"step_{step_id:04d}.html"
        html_path.write_text(html, encoding="utf-8")

        img_path = self._output_dir / f"step_{step_id:04d}.png"
        return self._take_screenshot(str(html_path), img_path)


# ---------------------------------------------------------------------------
# HtmlSliderRenderer
# ---------------------------------------------------------------------------

class HtmlSliderRenderer(_ScreenshotMixin):
    """
    Horizontal slider visualization for abstract input variables.
    Knob position encodes the variable value.
    Supports show_values flag for OCR/non-OCR conditions.
    """

    def __init__(
        self,
        output_dir: str,
        template_path: str,
        input_variables: List[str],
        variable_specs: Dict[str, Dict[str, Any]],
        width: int = 900,
        height: int = 500,
        headless: bool = True,
        slow_mo: int = 0,
    ) -> None:
        if not os.path.isfile(template_path):
            raise FileNotFoundError(f"Template not found: {template_path}")

        self._output_dir = Path(output_dir).resolve()
        self._html_dir = self._output_dir / "html"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._html_dir.mkdir(parents=True, exist_ok=True)

        self._input_variables = input_variables
        self._specs = variable_specs
        self._width = width
        self._height = height
        self._headless = headless
        self._slow_mo = slow_mo

        tp = Path(template_path).resolve()
        env = Environment(
            loader=FileSystemLoader(str(tp.parent)),
            autoescape=select_autoescape(["html"]),
        )
        self._template = env.get_template(tp.name)

    def render(
        self,
        state: Dict[str, float],
        step_id: int,
        show_values: bool,
    ) -> str:
        sliders = []
        for var in self._input_variables:
            spec = self._specs[var]
            lo = float(spec.get("min_value", 1))
            hi = float(spec.get("max_value", 10))
            step = float(spec.get("step", 1.0))
            value = float(state.get(var, lo))

            span = hi - lo if hi > lo else 1.0
            knob_pct = max(0.0, min(100.0, (value - lo) / span * 100.0))
            fill_pct = knob_pct

            ticks = self._make_ticks(lo, hi, step)

            sliders.append({
                "name": var,
                "value": value,
                "value_label": self._fmt(value),
                "knob_pct": round(knob_pct, 2),
                "fill_pct": round(fill_pct, 2),
                "ticks": ticks,
            })

        html = self._template.render(
            step_id=step_id,
            sliders=sliders,
            show_values=show_values,
            width=self._width,
            height=self._height,
        )
        html_path = self._html_dir / f"step_{step_id:04d}.html"
        html_path.write_text(html, encoding="utf-8")

        img_path = self._output_dir / f"step_{step_id:04d}.png"
        return self._take_screenshot(str(html_path), img_path)


# ---------------------------------------------------------------------------
# HtmlDotPlotRenderer  (vertical dot chart: x = variable, y = value)
# ---------------------------------------------------------------------------

class HtmlDotPlotRenderer(_ScreenshotMixin):
    """
    Vertical dot chart for abstract input variables.

    x-axis: variable names (categorical)
    y-axis: variable value (1–10)

    Each variable is represented by one dot at its current value height.
    This is a pure position encoding, comparable to bar (length) and
    slider/grid (position or count).
    """

    HEADER_HEIGHT = 72   # page-header + panel-label combined

    def __init__(
        self,
        output_dir: str,
        template_path: str,
        input_variables: List[str],
        variable_specs: Dict[str, Dict[str, Any]],
        width: int = 900,
        height: int = 500,
        headless: bool = True,
        slow_mo: int = 0,
    ) -> None:
        if not os.path.isfile(template_path):
            raise FileNotFoundError(f"Template not found: {template_path}")

        self._output_dir = Path(output_dir).resolve()
        self._html_dir = self._output_dir / "html"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._html_dir.mkdir(parents=True, exist_ok=True)

        self._input_variables = input_variables
        self._specs = variable_specs
        self._width = width
        self._height = height
        self._headless = headless
        self._slow_mo = slow_mo

        tp = Path(template_path).resolve()
        env = Environment(
            loader=FileSystemLoader(str(tp.parent)),
            autoescape=select_autoescape(["html"]),
        )
        self._template = env.get_template(tp.name)

    def render(
        self,
        state: Dict[str, float],
        step_id: int,
        show_values: bool,
    ) -> str:
        render_data = self._build_render_data(state, step_id, show_values)
        html = self._template.render(**render_data)

        html_path = self._html_dir / f"step_{step_id:04d}.html"
        html_path.write_text(html, encoding="utf-8")

        img_path = self._output_dir / f"step_{step_id:04d}.png"
        return self._take_screenshot(str(html_path), img_path)

    def _build_render_data(
        self,
        state: Dict[str, float],
        step_id: int,
        show_values: bool,
    ) -> Dict[str, Any]:
        # SVG dimensions (body minus header/panel-label area)
        svg_w = self._width
        svg_h = self._height - self.HEADER_HEIGHT

        margin_l = 52
        margin_r = 24
        margin_t = 24
        margin_b = 52
        plot_x = margin_l
        plot_y = margin_t
        plot_w = svg_w - margin_l - margin_r
        plot_h = svg_h - margin_t - margin_b

        # Y scale: use the shared range of all input variables
        all_lo = min(float(self._specs[v].get("min_value", 1)) for v in self._input_variables)
        all_hi = max(float(self._specs[v].get("max_value", 10)) for v in self._input_variables)
        y_span = all_hi - all_lo if all_hi > all_lo else 1.0

        # Y-axis ticks at each integer
        y_ticks = []
        for v in range(int(all_lo), int(all_hi) + 1):
            norm = (v - all_lo) / y_span
            y = plot_y + (1.0 - norm) * plot_h
            y_ticks.append({"y": round(y, 2), "label": str(v)})

        # X positions: evenly spaced with padding
        n_vars = len(self._input_variables)
        x_pad = plot_w / (n_vars + 1)
        x_positions = [plot_x + x_pad * (i + 1) for i in range(n_vars)]

        points = []
        for i, var in enumerate(self._input_variables):
            value = float(state.get(var, all_lo))
            norm = max(0.0, min(1.0, (value - all_lo) / y_span))
            x = x_positions[i]
            y = plot_y + (1.0 - norm) * plot_h
            points.append({
                "name": var,
                "value": value,
                "value_label": self._fmt(value),
                "x": round(x, 2),
                "y": round(y, 2),
            })

        return {
            "step_id": step_id,
            "width": self._width,
            "height": self._height,
            "svg_w": svg_w,
            "svg_h": svg_h,
            "plot_x": plot_x,
            "plot_y": plot_y,
            "plot_w": plot_w,
            "plot_h": plot_h,
            "y_ticks": y_ticks,
            "points": points,
            "show_values": show_values,
        }


# ---------------------------------------------------------------------------
# HtmlHistoryChartRenderer
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# HtmlGridRenderer  (filled-unit grid: count encoding)
# ---------------------------------------------------------------------------

class HtmlGridRenderer(_ScreenshotMixin):
    """
    Filled grid visualization for abstract input variables.

    Each variable is represented by a row of N cells.
    The number of filled cells encodes the variable value.
    Remaining cells are shown unfilled (empty boxes).

    This provides a discrete unit-count encoding, distinct from:
      Bar   → length
      Slider/Dot → position
      Line/Scatter → trajectory
    """

    def __init__(
        self,
        output_dir: str,
        template_path: str,
        input_variables: List[str],
        variable_specs: Dict[str, Dict[str, Any]],
        width: int = 900,
        height: int = 500,
        headless: bool = True,
        slow_mo: int = 0,
    ) -> None:
        if not os.path.isfile(template_path):
            raise FileNotFoundError(f"Template not found: {template_path}")

        self._output_dir = Path(output_dir).resolve()
        self._html_dir = self._output_dir / "html"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._html_dir.mkdir(parents=True, exist_ok=True)

        self._input_variables = input_variables
        self._specs = variable_specs
        self._width = width
        self._height = height
        self._headless = headless
        self._slow_mo = slow_mo

        tp = Path(template_path).resolve()
        env = Environment(
            loader=FileSystemLoader(str(tp.parent)),
            autoescape=select_autoescape(["html"]),
        )
        self._template = env.get_template(tp.name)

    def render(
        self,
        state: Dict[str, float],
        step_id: int,
        show_values: bool,
    ) -> str:
        # Determine the total number of cells = max_value of the shared range.
        total_cells = int(
            max(float(self._specs[v].get("max_value", 10)) for v in self._input_variables)
        )

        rows = []
        for var in self._input_variables:
            value = float(state.get(var, 1.0))
            filled_count = max(0, min(total_cells, int(round(value))))
            cells = [{"filled": i < filled_count} for i in range(total_cells)]
            rows.append({
                "name": var,
                "value": value,
                "value_label": self._fmt(value),
                "filled_count": filled_count,
                "cells": cells,
            })

        html = self._template.render(
            step_id=step_id,
            rows=rows,
            total_cells=total_cells,
            show_values=show_values,
            width=self._width,
            height=self._height,
        )
        html_path = self._html_dir / f"step_{step_id:04d}.html"
        html_path.write_text(html, encoding="utf-8")

        img_path = self._output_dir / f"step_{step_id:04d}.png"
        return self._take_screenshot(str(html_path), img_path)


SERIES_COLORS = [
    "#2563eb",  # blue
    "#dc2626",  # red
    "#16a34a",  # green
    "#d97706",  # amber
    "#7c3aed",  # purple
]


class HtmlHistoryChartRenderer(_ScreenshotMixin):
    """
    Time-series history chart for abstract input variables.

    Each render() call receives the history of states up to the current step
    and draws all of them on a single chart (x = step index, y = value).

    For line_ocr / scatter_ocr: value labels are shown on the CURRENT
    (rightmost) step's points only.  Historical points stay unlabelled.
    """

    HEADER_HEIGHT = 46

    def __init__(
        self,
        output_dir: str,
        template_path: str,
        plot_type: str,
        input_variables: List[str],
        variable_specs: Dict[str, Dict[str, Any]],
        width: int = 900,
        height: int = 520,
        headless: bool = True,
        slow_mo: int = 0,
    ) -> None:
        if plot_type not in {"line", "scatter"}:
            raise ValueError("plot_type must be 'line' or 'scatter'")
        if not os.path.isfile(template_path):
            raise FileNotFoundError(f"Template not found: {template_path}")

        self._output_dir = Path(output_dir).resolve()
        self._html_dir = self._output_dir / "html"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._html_dir.mkdir(parents=True, exist_ok=True)

        self._plot_type = plot_type
        self._input_variables = input_variables
        self._specs = variable_specs
        self._width = width
        self._height = height
        self._headless = headless
        self._slow_mo = slow_mo

        tp = Path(template_path).resolve()
        env = Environment(
            loader=FileSystemLoader(str(tp.parent)),
            autoescape=select_autoescape(["html"]),
        )
        self._template = env.get_template(tp.name)

    def render(
        self,
        states: List[Dict[str, float]],
        current_step_id: int,
        show_values: bool,
    ) -> str:
        render_data = self._build_render_data(states, current_step_id, show_values)

        html = self._template.render(**render_data)
        html_path = self._html_dir / f"step_{current_step_id:04d}.html"
        html_path.write_text(html, encoding="utf-8")

        img_path = self._output_dir / f"step_{current_step_id:04d}.png"
        return self._take_screenshot(str(html_path), img_path)

    def _build_render_data(
        self,
        states: List[Dict[str, float]],
        current_step_id: int,
        show_values: bool,
    ) -> Dict[str, Any]:
        # Layout constants
        margin_l = 55
        margin_r = 30
        margin_t = 20
        margin_b = 55
        header_h = self.HEADER_HEIGHT

        svg_h = self._height - header_h
        plot_x = margin_l
        plot_y = margin_t
        plot_w = self._width - margin_l - margin_r
        plot_h = svg_h - margin_t - margin_b

        # Y axis: values are in [1, 10]; show ticks at 1, 3, 5, 7, 9, 10
        all_lo = min(
            float(self._specs[v].get("min_value", 1)) for v in self._input_variables
        )
        all_hi = max(
            float(self._specs[v].get("max_value", 10)) for v in self._input_variables
        )
        y_span = all_hi - all_lo if all_hi > all_lo else 1.0

        y_tick_values = []
        cur = all_lo
        while cur <= all_hi + 1e-9:
            y_tick_values.append(cur)
            cur += max(1.0, (all_hi - all_lo) / 5)
        if all_hi not in y_tick_values:
            y_tick_values.append(all_hi)

        y_ticks = []
        for v in y_tick_values:
            norm = (v - all_lo) / y_span
            y_coord = plot_y + (1.0 - norm) * plot_h
            y_ticks.append({"y": round(y_coord, 2), "label": self._fmt(v)})

        # X axis ticks: one per step in history
        n_steps = len(states)
        x_step = plot_w / max(n_steps - 1, 1)
        x_ticks = []
        for i in range(n_steps):
            x = plot_x + i * x_step if n_steps > 1 else plot_x + plot_w / 2
            x_ticks.append({"x": round(x, 2), "label": str(i)})

        # Build series
        series = []
        for var_idx, var in enumerate(self._input_variables):
            color = SERIES_COLORS[var_idx % len(SERIES_COLORS)]
            points = []
            for step_i, state in enumerate(states):
                value = float(state.get(var, all_lo))
                norm = max(0.0, min(1.0, (value - all_lo) / y_span))
                x = plot_x + step_i * x_step if n_steps > 1 else plot_x + plot_w / 2
                y = plot_y + (1.0 - norm) * plot_h
                is_current = step_i == len(states) - 1
                points.append({
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "value": value,
                    "value_label": self._fmt(value),
                    "is_current": is_current,
                })

            # SVG path for line chart
            path = ""
            if len(points) >= 2:
                path_parts = [f"M {points[0]['x']} {points[0]['y']}"]
                for p in points[1:]:
                    path_parts.append(f"L {p['x']} {p['y']}")
                path = " ".join(path_parts)

            series.append({
                "name": var,
                "color": color,
                "points": points,
                "path": path,
            })

        return {
            "plot_type": self._plot_type,
            "current_step_id": current_step_id,
            "n_vars": len(self._input_variables),
            "width": self._width,
            "height": self._height,
            "header_height": header_h,
            "plot_x": plot_x,
            "plot_y": plot_y,
            "plot_w": plot_w,
            "plot_h": plot_h,
            "y_ticks": y_ticks,
            "x_ticks": x_ticks,
            "series": series,
            "show_values": show_values,
        }