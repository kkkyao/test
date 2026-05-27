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

    Modalities:
    - text:
        No images. The prompt contains the numeric table.

    - bar:
        Reuse existing HtmlChartRenderer.
        One image per step.

    - line:
        Use benchmark-specific HtmlStepPlotRenderer.
        One image per step.
        Each image shows the current state's variable values as connected points.

    - scatter:
        Use benchmark-specific HtmlStepPlotRenderer.
        One image per step.
        Each image shows the current state's variable values as independent points.

    - simulation:
        Reuse existing HtmlSimulationRenderer.
        One screenshot per step.

    Important:
    All visual modalities are now step-wise image sequences.
    This keeps bar, line, scatter, and simulation comparable.
    """

    def __init__(self, config: Dict[str, Any], output_dir: str) -> None:
        self.config = config
        self.output_dir = Path(output_dir)

        benchmark_cfg = config["benchmark"]
        variables_cfg = benchmark_cfg["variables"]

        self.target_variable = variables_cfg["target_variable"]
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

        if case.modality == "bar":
            return self._render_bar_sequence(case)

        if case.modality == "line":
            return self._render_step_plot_sequence(case, plot_type="line")

        if case.modality == "scatter":
            return self._render_step_plot_sequence(case, plot_type="scatter")

        if case.modality == "simulation":
            return self._render_simulation_sequence(case)

        raise ValueError(f"Unsupported modality: {case.modality}")

    # ------------------------------------------------------------------
    # Existing bar renderer: one image per step
    # ------------------------------------------------------------------

    def _render_bar_sequence(self, case: BenchmarkCase) -> List[str]:
        bar_cfg = self.visual_cfg.get("bar", {})
        output_dir = self.output_dir / "images" / case.case_id / "bar"

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
        output_dir = self.output_dir / "images" / case.case_id / plot_type

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
                    case_id=case.case_id,
                    step_id=step_id,
                )
            )

        return image_paths

    # ------------------------------------------------------------------
    # Existing simulation renderer: one image per step
    # ------------------------------------------------------------------

    def _render_simulation_sequence(self, case: BenchmarkCase) -> List[str]:
        sim_cfg = self.visual_cfg.get("simulation", {})
        output_dir = self.output_dir / "images" / case.case_id / "simulation"

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
    # Shared config adapter
    # ------------------------------------------------------------------

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
        case_id: str,
        step_id: int,
    ) -> str:
        render_data = self._build_render_data(
            state=state,
            case_id=case_id,
            step_id=step_id,
        )
        html_path = self._render_html(
            render_data=render_data,
            case_id=case_id,
            step_id=step_id,
        )
        image_path = self._take_screenshot(
            html_path=html_path,
            case_id=case_id,
            step_id=step_id,
        )
        return image_path

    def _build_render_data(
        self,
        state: Dict[str, float],
        case_id: str,
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
        case_id: str,
        step_id: int,
    ) -> str:
        html_content = self.template.render(**render_data)
        html_path = self.html_dir / f"{case_id}_step_{step_id:04d}.html"
        html_path.write_text(html_content, encoding="utf-8")
        return str(html_path.resolve())

    def _take_screenshot(
        self,
        html_path: str,
        case_id: str,
        step_id: int,
        _max_retries: int = 3,
    ) -> str:
        image_path = self.output_dir / f"{case_id}_step_{step_id:04d}.png"
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