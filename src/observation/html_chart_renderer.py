from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape
from playwright.sync_api import sync_playwright


class HtmlChartRenderer:
    """
    Render the current environment state as a bar-chart HTML page,
    take a Playwright screenshot, and return the absolute PNG path.

    Directory layout inside output_dir
    -----------------------------------
    {output_dir}/
        html/
            step_0000.html
            step_0001.html
            ...
        step_0000.png          ← returned image_path
        step_0001.png
        ...

    Normalisation
    -------------
    Each bar's display height (0–1) is computed as:
        norm_height = (value − lo) / (hi − lo)

    Priority for (lo, hi):
    1. bar_value_range[var]              if explicitly supplied
    2. variables[var]["min_value"] /
       variables[var]["max_value"]       if present in env config
    3. dynamic min / max across all
       values in the current state       fallback (always works)

    When normalize_bars=False the dynamic range is always used,
    ignoring any configured min/max.  Both branches still produce
    a proportional chart; the difference is the reference frame.

    exclude_variables
    -----------------
    Internal variable names listed here are silently skipped when
    building bar data.  Useful for derived variables that are not
    relevant to the agent's task (e.g. transmittance in Beer's law).
    """

    def __init__(
        self,
        variables: Dict[str, Dict[str, Any]],
        target_variable: str,
        naming_mode: str,
        name_mapping: Optional[Dict[str, str]],
        output_dir: str,
        template_path: str,
        headless: bool = True,
        slow_mo: int = 0,
        normalize_bars: bool = True,
        bar_value_range: Optional[Dict[str, tuple]] = None,
        exclude_variables: Optional[List[str]] = None,
        output_bar_lo: float = 0.0,
        output_bar_hi: float = 600.0,
    ) -> None:
        if not isinstance(variables, dict) or not variables:
            raise ValueError("variables must be a non-empty dictionary")

        if naming_mode not in {"concrete", "abstract"}:
            raise ValueError("naming_mode must be 'concrete' or 'abstract'")

        if naming_mode == "abstract" and not name_mapping:
            raise ValueError("name_mapping must be provided when naming_mode='abstract'")

        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ValueError("output_dir must be a non-empty string")

        if not os.path.isfile(template_path):
            raise FileNotFoundError(f"template not found: {template_path}")

        self.variables = deepcopy(variables)
        self.target_variable = target_variable
        self.naming_mode = naming_mode
        self.name_mapping: Dict[str, str] = dict(name_mapping or {})
        self.headless = headless
        self.slow_mo = slow_mo
        self.normalize_bars = normalize_bars
        self.bar_value_range: Dict[str, tuple] = dict(bar_value_range or {})
        self.exclude_variables: set = set(exclude_variables or [])

        # Viewport dimensions used by Playwright (previously hardcoded in
        # _take_screenshot). 650px height accommodates two stacked panels.
        self._viewport_width: int = 900
        self._viewport_height: int = 650

        # Visual-only display range for the Outputs panel bar chart.
        # Used solely for bar height normalization — does NOT affect env
        # dynamics, action space, or target computation.
        # A bar at 100% (full height) means value >= output_bar_hi; it does NOT
        # mean the value equals the true theoretical maximum of the target.
        # Values above output_bar_hi are clipped to 100% by _normalise(); the
        # raw numeric value label above the bar always shows the true value.
        self._output_bar_lo: float = float(output_bar_lo)
        self._output_bar_hi: float = float(output_bar_hi)

        # Output directories
        self._base_dir = Path(output_dir).resolve()
        self._html_dir = self._base_dir / "html"
        self._html_dir.mkdir(parents=True, exist_ok=True)
        self._base_dir.mkdir(parents=True, exist_ok=True)

        # Jinja2 environment
        template_path_abs = Path(template_path).resolve()
        self._jinja_env = Environment(
            loader=FileSystemLoader(str(template_path_abs.parent)),
            autoescape=select_autoescape(["html"]),
        )
        self._template = self._jinja_env.get_template(template_path_abs.name)



    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def render(
        self,
        state: Dict[str, Any],
        history: Optional[List[Dict[str, Any]]] = None,
        step_id: Optional[int] = None,
    ) -> str:
        """
        Generate a bar-chart screenshot for the current state.

        Parameters
        ----------
        state:
            Current environment state dict (variable name → value).
        history:
            Unused in this renderer. Accepted for RendererProtocol compatibility.
        step_id:
            Used for output file naming.  None → files named "step_none.*".

        Returns
        -------
        Absolute path to the saved PNG file.
        """
        if not isinstance(state, dict):
            raise ValueError("state must be a dictionary")

        render_data = self._build_render_data(state, step_id)
        html_path = self._render_html(render_data, step_id)
        image_path = self._take_screenshot(html_path, step_id)
        return image_path

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _build_render_data(
        self,
        state: Dict[str, Any],
        step_id: Optional[int],
    ) -> Dict[str, Any]:
        """
        Split visible state into two panels:

        input_bars  (Controls panel, top)
            Variables with manipulable=True in the env config.
            Normalized per-variable using config min_value/max_value when
            normalize_bars=True, or using input-only dyn_min/dyn_max when
            normalize_bars=False.  The target variable never participates in
            the input-panel dyn range calculation.

        output_bars  (Outputs panel, bottom)
            Only self.target_variable.
            Normalized against self._output_bar_lo / self._output_bar_hi,
            which is a visual-only display range configured via
            display.output_bar_range in the env config.  This range does NOT
            affect env dynamics, action space, or target computation.
            Values above output_bar_hi are clipped to 100% height by
            _normalise(); the raw label above the bar shows the true value.

        Non-manipulable, non-target variables (e.g. transmittance in Beer's
        law) are silently omitted from both panels even when not listed in
        exclude_variables.  bar_value_range overrides apply only to input
        variables.
        """
        # 1. Remove excluded variables.
        visible_state = {
            k: v for k, v in state.items()
            if k not in self.exclude_variables
        }

        # 2. Split: manipulable → Controls; target → Outputs; others → ignored.
        input_state: Dict[str, Any] = {}
        output_state: Dict[str, Any] = {}
        for var_name, raw_value in visible_state.items():
            cfg = self.variables.get(var_name, {})
            if cfg.get("manipulable", False):
                input_state[var_name] = raw_value
            elif var_name == self.target_variable:
                output_state[var_name] = raw_value

        # 3. Compute dyn range for input panel (input variables only —
        #    target is never included here).
        input_numeric = [
            float(v) for v in input_state.values() if isinstance(v, (int, float))
        ]
        if input_numeric:
            input_dyn_min = min(input_numeric)
            input_dyn_max = max(input_numeric)
        else:
            input_dyn_min, input_dyn_max = 0.0, 1.0

        # 4. Build input_bars (Controls panel).
        input_bars: List[Dict[str, Any]] = []
        for var_name, raw_value in input_state.items():
            if not isinstance(raw_value, (int, float)):
                continue
            value = float(raw_value)
            lo, hi = self._get_range(var_name, input_dyn_min, input_dyn_max)
            input_bars.append({
                "name":        self._display_name(var_name),
                "value":       value,
                "norm_height": self._normalise(value, lo, hi),
            })

        # 5. Build output_bars (Outputs panel) using visual-only fixed range.
        #    _normalise() clamps values above output_bar_hi to 1.0 (100%),
        #    so a full-height bar means value >= visual max, not true maximum.
        output_bars: List[Dict[str, Any]] = []
        for var_name, raw_value in output_state.items():
            if not isinstance(raw_value, (int, float)):
                continue
            value = float(raw_value)
            output_bars.append({
                "name":        self._display_name(var_name),
                "value":       value,
                "norm_height": self._normalise(value, self._output_bar_lo, self._output_bar_hi),
                "vis_lo":      self._output_bar_lo,
                "vis_hi":      self._output_bar_hi,
            })

        return {
            "step_id":        step_id,
            "target_display": self._display_name(self.target_variable),
            "input_bars":     input_bars,
            "output_bars":    output_bars,
        }

    def _get_range(
        self,
        var_name: str,
        dyn_min: float,
        dyn_max: float,
    ) -> tuple[float, float]:
        """
        Return (lo, hi) for normalising var_name.

        Priority:
          1. Explicit bar_value_range entry
          2. min_value / max_value from variable config  (only if normalize_bars=True)
          3. Dynamic range across all current values
        """
        if var_name in self.bar_value_range:
            lo, hi = self.bar_value_range[var_name]
            return float(lo), float(hi)

        if self.normalize_bars and var_name in self.variables:
            cfg = self.variables[var_name]
            lo_raw = cfg.get("min_value")
            hi_raw = cfg.get("max_value")
            lo = float(lo_raw) if lo_raw is not None else dyn_min
            hi = float(hi_raw) if hi_raw is not None else dyn_max
            return lo, hi

        return dyn_min, dyn_max

    @staticmethod
    def _normalise(value: float, lo: float, hi: float) -> float:
        """
        Map value into [0, 1] given [lo, hi].

        Edge cases:
        - lo == hi  → 0.5 (all bars same height, avoids division by zero)
        - value out of range → clamped to [0, 1]
        """
        if hi <= lo:
            return 0.5
        norm = (value - lo) / (hi - lo)
        return max(0.0, min(1.0, norm))

    def _render_html(
        self,
        render_data: Dict[str, Any],
        step_id: Optional[int],
    ) -> str:
        """
        Render the Jinja2 template and write the HTML file.
        """
        html_content = self._template.render(**render_data)

        filename = (
            f"step_{step_id:04d}.html" if step_id is not None else "step_none.html"
        )
        html_path = self._html_dir / filename
        html_path.write_text(html_content, encoding="utf-8")
        return str(html_path.resolve())

    def _take_screenshot(
        self,
        html_path: str,
        step_id: Optional[int],
        _max_retries: int = 3,
    ) -> str:
        """
        Open the HTML file in a headless Chromium browser and save a screenshot.

        A fresh sync_playwright context is used for each call so that this
        renderer is safe to instantiate inside an asyncio loop (e.g. between
        runs managed by wandb).  The browser is launched and closed within
        each call, keeping resource usage bounded.

        Retries up to _max_retries times on playwright errors (e.g. the
        intermittent "Unable to capture screenshot" protocol error).
        """
        filename = (
            f"step_{step_id:04d}.png" if step_id is not None else "step_none.png"
        )
        image_path = self._base_dir / filename
        file_uri = Path(html_path).as_uri()

        last_exc: Optional[Exception] = None
        for attempt in range(_max_retries):
            try:
                with sync_playwright() as pw:
                    browser = pw.chromium.launch(
                        headless=self.headless,
                        slow_mo=self.slow_mo,
                    )
                    page = browser.new_page(viewport={"width": self._viewport_width, "height": self._viewport_height})
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

    # -------------------------------------------------------------------------
    # Naming helpers
    # -------------------------------------------------------------------------

    def _display_name(self, variable: str) -> str:
        if self.naming_mode == "abstract":
            return self.name_mapping.get(variable, variable)
        return variable