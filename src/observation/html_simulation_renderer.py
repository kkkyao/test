from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape
from playwright.sync_api import sync_playwright


class HtmlSimulationRenderer:
    """
    Render the current environment state as a PhET-like simulation page,
    take a Playwright screenshot, and return the absolute PNG path.

    This renderer only creates visual observations. It does not decide which
    text enters the prompt. For simulation-only, PromptBuilder should use
    include_observation_text=False so current state values are read from images,
    not from text.

    Supported simulation types:
    - mass
    - ohm
    - resistors
    - distance
    - kinematics
    - beers
    - beers_wavelength
    - concentration

    Directory layout inside output_dir
    -----------------------------------
    {output_dir}/
        html/
            step_0000.html
            step_0001.html
            ...
        step_0000.png
        step_0001.png
        ...
    """

    _SUPPORTED_SIMULATIONS = {
        "mass",
        "ohm",
        "resistors",
        "distance",
        "kinematics",
        "beers",
        "beers_wavelength",
        "concentration",
    }

    _REQUIRED_STATE_KEYS = {
        "mass": [
            "density",
            "volume",
            "mass",
        ],
        "ohm": [
            "current",
            "resistance",
            "voltage",
        ],
        "resistors": [
            "resistance_1",
            "resistance_2",
            "total_resistance",
        ],
        "distance": [
            "distance_1",
            "distance_2",
            "distance_3",
            "total_distance",
        ],
        "kinematics": [
            "initial_position",
            "velocity",
            "time",
            "position",
        ],
        "beers": [
            "molar_absorptivity",
            "concentration",
            "path_length",
            "absorbance",
        ],
        "beers_wavelength": [
            "wavelength",
            "concentration",
            "path_length",
            "absorbance",
        ],
        "concentration": [
            "solute_amount",
            "solution_volume",
            "concentration",
        ],
    }

    def __init__(
        self,
        variables: Dict[str, Dict[str, Any]],
        target_variable: str,
        naming_mode: str,
        name_mapping: Optional[Dict[str, str]],
        output_dir: str,
        template_dir: str,
        simulation_type: str,
        headless: bool = True,
        slow_mo: int = 0,
        exclude_variables: Optional[List[str]] = None,
        viewport_width: int = 900,
        viewport_height: int = 500,
    ) -> None:
        if not isinstance(variables, dict) or not variables:
            raise ValueError("variables must be a non-empty dictionary")

        if not isinstance(target_variable, str) or not target_variable.strip():
            raise ValueError("target_variable must be a non-empty string")

        if naming_mode not in {"concrete", "abstract"}:
            raise ValueError("naming_mode must be 'concrete' or 'abstract'")

        if naming_mode == "abstract" and not name_mapping:
            raise ValueError("name_mapping must be provided when naming_mode='abstract'")

        if simulation_type not in self._SUPPORTED_SIMULATIONS:
            raise ValueError(
                f"Unsupported simulation_type='{simulation_type}'. "
                f"Supported: {sorted(self._SUPPORTED_SIMULATIONS)}"
            )

        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ValueError("output_dir must be a non-empty string")

        if not isinstance(template_dir, str) or not template_dir.strip():
            raise ValueError("template_dir must be a non-empty string")

        template_dir_path = Path(template_dir).resolve()
        if not template_dir_path.is_dir():
            raise FileNotFoundError(f"template_dir not found: {template_dir}")

        template_name = f"{simulation_type}_sim.html"
        template_path = template_dir_path / template_name
        if not template_path.is_file():
            raise FileNotFoundError(f"simulation template not found: {template_path}")

        self.variables = deepcopy(variables)
        self.target_variable = target_variable
        self.naming_mode = naming_mode
        self.name_mapping: Dict[str, str] = dict(name_mapping or {})
        self.simulation_type = simulation_type
        self.headless = headless
        self.slow_mo = slow_mo
        self.exclude_variables: set[str] = set(exclude_variables or [])
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height

        self._base_dir = Path(output_dir).resolve()
        self._html_dir = self._base_dir / "html"
        self._html_dir.mkdir(parents=True, exist_ok=True)
        self._base_dir.mkdir(parents=True, exist_ok=True)

        self._jinja_env = Environment(
            loader=FileSystemLoader(str(template_dir_path)),
            autoescape=select_autoescape(["html"]),
        )
        self._template = self._jinja_env.get_template(template_name)

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
        Generate a simulation screenshot for the current state.

        Parameters
        ----------
        state:
            Current environment state dict using internal variable names.
        history:
            Accepted for renderer protocol compatibility. Not used here.
        step_id:
            Used for output file naming.

        Returns
        -------
        Absolute path to the saved PNG screenshot.
        """
        if not isinstance(state, dict):
            raise ValueError("state must be a dictionary")

        render_data = self._build_render_data(state=state, step_id=step_id)
        html_path = self._render_html(render_data=render_data, step_id=step_id)
        image_path = self._take_screenshot(html_path=html_path, step_id=step_id)
        return image_path

    # -------------------------------------------------------------------------
    # Render-data dispatcher
    # -------------------------------------------------------------------------

    def _build_render_data(
        self,
        state: Dict[str, Any],
        step_id: Optional[int],
    ) -> Dict[str, Any]:
        self._require_state_keys(state)

        if self.simulation_type == "mass":
            return self._build_mass_render_data(state, step_id)

        if self.simulation_type == "ohm":
            return self._build_ohm_render_data(state, step_id)

        if self.simulation_type == "resistors":
            return self._build_resistors_render_data(state, step_id)

        if self.simulation_type == "distance":
            return self._build_distance_render_data(state, step_id)

        if self.simulation_type == "kinematics":
            return self._build_kinematics_render_data(state, step_id)

        if self.simulation_type == "beers":
            return self._build_beers_render_data(state, step_id)

        if self.simulation_type == "beers_wavelength":
            return self._build_beers_wavelength_render_data(state, step_id)

        if self.simulation_type == "concentration":
            return self._build_concentration_render_data(state, step_id)

        raise ValueError(f"Unsupported simulation_type='{self.simulation_type}'")

    # -------------------------------------------------------------------------
    # Domain-specific render data
    # -------------------------------------------------------------------------

    def _build_mass_render_data(
        self,
        state: Dict[str, Any],
        step_id: Optional[int],
    ) -> Dict[str, Any]:
        density = self._value(state, "density")
        volume = self._value(state, "volume")
        mass = self._value(state, "mass")

        density_norm = self._norm_var("density", density)
        volume_norm = self._norm_var("volume", volume)
        mass_norm = self._norm_target("mass", mass)

        return self._base_render_data(
            step_id=step_id,
            title="Material Mass Experiment",
            variables={
                "density": self._payload("density", density),
                "volume": self._payload("volume", volume),
            },
            target=self._payload("mass", mass, is_target=True),
            simulation={
                "density_norm": density_norm,
                "volume_norm": volume_norm,
                "mass_norm": mass_norm,
                "block_width": int(95 + volume_norm * 115),
                "block_height": int(65 + volume_norm * 85),
                "density_alpha": round(0.34 + density_norm * 0.48, 3),
                "density_needle_angle": round(-120 + density_norm * 240, 2),
                "volume_fill_percent": round(18 + volume_norm * 74, 2),
                "scale_bar_width": round(14 + mass_norm * 78, 2),
            },
        )

    def _build_ohm_render_data(
        self,
        state: Dict[str, Any],
        step_id: Optional[int],
    ) -> Dict[str, Any]:
        current = self._value(state, "current")
        resistance = self._value(state, "resistance")
        voltage = self._value(state, "voltage")

        current_norm = self._norm_var("current", current)
        resistance_norm = self._norm_var("resistance", resistance)
        voltage_norm = self._norm_target("voltage", voltage)

        return self._base_render_data(
            step_id=step_id,
            title="Circuit Voltage Experiment",
            variables={
                "current": self._payload("current", current),
                "resistance": self._payload("resistance", resistance),
            },
            target=self._payload("voltage", voltage, is_target=True),
            simulation={
                "current_norm": current_norm,
                "resistance_norm": resistance_norm,
                "voltage_norm": voltage_norm,
                "current_needle_angle": round(-120 + current_norm * 240, 2),
                "resistance_knob_angle": round(-135 + resistance_norm * 270, 2),
                "voltage_needle_angle": round(-115 + voltage_norm * 230, 2),
                "wire_glow_opacity": round(0.22 + current_norm * 0.58, 3),
                "resistor_width": int(78 + resistance_norm * 80),
            },
        )

    def _build_resistors_render_data(
        self,
        state: Dict[str, Any],
        step_id: Optional[int],
    ) -> Dict[str, Any]:
        r1 = self._value(state, "resistance_1")
        r2 = self._value(state, "resistance_2")
        total = self._value(state, "total_resistance")

        r1_norm = self._norm_var("resistance_1", r1)
        r2_norm = self._norm_var("resistance_2", r2)
        total_norm = self._norm_target("total_resistance", total)

        return self._base_render_data(
            step_id=step_id,
            title="Series Resistance Experiment",
            variables={
                "resistance_1": self._payload("resistance_1", r1),
                "resistance_2": self._payload("resistance_2", r2),
            },
            target=self._payload("total_resistance", total, is_target=True),
            simulation={
                "r1_norm": r1_norm,
                "r2_norm": r2_norm,
                "total_norm": total_norm,
                "r1_width": int(70 + r1_norm * 80),
                "r2_width": int(70 + r2_norm * 80),
                "ohmmeter_needle_angle": round(-115 + total_norm * 230, 2),
                "r1_band_offset": round(18 + r1_norm * 36, 2),
                "r2_band_offset": round(18 + r2_norm * 36, 2),
            },
        )

    def _build_distance_render_data(
        self,
        state: Dict[str, Any],
        step_id: Optional[int],
    ) -> Dict[str, Any]:
        d1 = self._value(state, "distance_1")
        d2 = self._value(state, "distance_2")
        d3 = self._value(state, "distance_3")
        total = self._value(state, "total_distance")

        d1_norm = self._norm_var("distance_1", d1)
        d2_norm = self._norm_var("distance_2", d2)
        d3_norm = self._norm_var("distance_3", d3)
        total_norm = self._norm_target("total_distance", total)

        return self._base_render_data(
            step_id=step_id,
            title="Route Distance Measurement",
            variables={
                "distance_1": self._payload("distance_1", d1),
                "distance_2": self._payload("distance_2", d2),
                "distance_3": self._payload("distance_3", d3),
            },
            target=self._payload("total_distance", total, is_target=True),
            simulation={
                "d1_norm": d1_norm,
                "d2_norm": d2_norm,
                "d3_norm": d3_norm,
                "total_norm": total_norm,
                "seg1_len": int(80 + d1_norm * 100),
                "seg2_len": int(80 + d2_norm * 100),
                "seg3_len": int(80 + d3_norm * 100),
                "trip_meter_width": round(14 + total_norm * 78, 2),
            },
        )

    def _build_kinematics_render_data(
        self,
        state: Dict[str, Any],
        step_id: Optional[int],
    ) -> Dict[str, Any]:
        x0 = self._value(state, "initial_position")
        velocity = self._value(state, "velocity")
        time = self._value(state, "time")
        position = self._value(state, "position")

        x0_norm = self._norm_var("initial_position", x0)
        velocity_norm = self._norm_var("velocity", velocity)
        time_norm = self._norm_var("time", time)
        position_norm = self._norm_target("position", position)

        track_left = 70
        track_width = 460

        return self._base_render_data(
            step_id=step_id,
            title="Motion Track Experiment",
            variables={
                "initial_position": self._payload("initial_position", x0),
                "velocity": self._payload("velocity", velocity),
                "time": self._payload("time", time),
            },
            target=self._payload("position", position, is_target=True),
            simulation={
                "x0_norm": x0_norm,
                "velocity_norm": velocity_norm,
                "time_norm": time_norm,
                "position_norm": position_norm,
                "track_left": track_left,
                "track_width": track_width,
                "start_x": int(track_left + x0_norm * track_width),
                "car_x": int(track_left + position_norm * track_width),
                "velocity_needle_angle": round(-120 + velocity_norm * 240, 2),
                "time_fill_percent": round(8 + time_norm * 84, 2),
            },
        )

    def _build_beers_render_data(
        self,
        state: Dict[str, Any],
        step_id: Optional[int],
    ) -> Dict[str, Any]:
        epsilon = self._value(state, "molar_absorptivity")
        concentration = self._value(state, "concentration")
        path_length = self._value(state, "path_length")
        absorbance = self._value(state, "absorbance")

        epsilon_norm = self._norm_var("molar_absorptivity", epsilon)
        concentration_norm = self._norm_var("concentration", concentration)
        path_norm = self._norm_var("path_length", path_length)
        absorbance_norm = self._norm_target("absorbance", absorbance)

        return self._base_render_data(
            step_id=step_id,
            title="Light Absorption Experiment",
            variables={
                "molar_absorptivity": self._payload("molar_absorptivity", epsilon),
                "concentration": self._payload("concentration", concentration),
                "path_length": self._payload("path_length", path_length),
            },
            target=self._payload("absorbance", absorbance, is_target=True),
            simulation={
                "epsilon_norm": epsilon_norm,
                "concentration_norm": concentration_norm,
                "path_norm": path_norm,
                "absorbance_norm": absorbance_norm,
                "solution_alpha": round(0.18 + concentration_norm * 0.62, 3),
                "cuvette_width": int(72 + path_norm * 92),
                "beam_after_opacity": round(0.82 - absorbance_norm * 0.64, 3),
                "detector_needle_angle": round(-115 + absorbance_norm * 230, 2),
                "solute_dial_angle": round(-135 + epsilon_norm * 270, 2),
            },
        )


    def _build_beers_wavelength_render_data(
        self,
        state: Dict[str, Any],
        step_id: Optional[int],
    ) -> Dict[str, Any]:
        wavelength = self._value(state, "wavelength")
        concentration = self._value(state, "concentration")
        path_length = self._value(state, "path_length")
        absorbance = self._value(state, "absorbance")

        wavelength_norm = self._norm_var("wavelength", wavelength)
        concentration_norm = self._norm_var("concentration", concentration)
        path_norm = self._norm_var("path_length", path_length)
        absorbance_norm = self._norm_target("absorbance", absorbance)

        return self._base_render_data(
            step_id=step_id,
            title="Light Absorption Experiment",
            variables={
                "wavelength": self._payload("wavelength", wavelength),
                "concentration": self._payload("concentration", concentration),
                "path_length": self._payload("path_length", path_length),
            },
            target=self._payload("absorbance", absorbance, is_target=True),
            simulation={
                "wavelength_norm": wavelength_norm,
                "concentration_norm": concentration_norm,
                "path_norm": path_norm,
                "absorbance_norm": absorbance_norm,
                "solution_alpha": round(0.18 + concentration_norm * 0.62, 3),
                "cuvette_width": int(70 + path_norm * 190),
                "beam_after_opacity": round(0.82 - absorbance_norm * 0.64, 3),
                "detector_needle_angle": round(-115 + absorbance_norm * 230, 2),
                "wavelength_nm": int(round(wavelength)),
            },
        )


    def _build_concentration_render_data(
        self,
        state: Dict[str, Any],
        step_id: Optional[int],
    ) -> Dict[str, Any]:
        solute_amount = self._value(state, "solute_amount")
        solution_volume = self._value(state, "solution_volume")
        concentration = self._value(state, "concentration")

        amount_norm = self._norm_var("solute_amount", solute_amount)
        volume_norm = self._norm_var("solution_volume", solution_volume)
        concentration_norm = self._norm_target("concentration", concentration)

        return self._base_render_data(
            step_id=step_id,
            title="Concentration Experiment",
            variables={
                "solute_amount": self._payload("solute_amount", solute_amount),
                "solution_volume": self._payload("solution_volume", solution_volume),
            },
            target=self._payload("concentration", concentration, is_target=True),
            simulation={
                "amount_norm": amount_norm,
                "volume_norm": volume_norm,
                "concentration_norm": concentration_norm,
                "solution_alpha": round(0.12 + concentration_norm * 0.68, 3),
                "water_fill_percent": round(18 + volume_norm * 70, 2),
                "solute_stream_opacity": round(0.28 + amount_norm * 0.64, 3),
                "meter_bar_width": round(8 + concentration_norm * 84, 2),
            },
        )


    # -------------------------------------------------------------------------
    # Shared render-data helpers
    # -------------------------------------------------------------------------

    def _base_render_data(
        self,
        step_id: Optional[int],
        title: str,
        variables: Dict[str, Dict[str, Any]],
        target: Dict[str, Any],
        simulation: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "step_id": step_id,
            "title": title,
            "simulation_type": self.simulation_type,
            "target_display": self._display_name(self.target_variable),
            "variables": variables,
            "target": target,
            "simulation": simulation,
            "viewport": {
                "width": self.viewport_width,
                "height": self.viewport_height,
            },
        }

    def _payload(
        self,
        var_name: str,
        value: float,
        is_target: bool = False,
    ) -> Dict[str, Any]:
        lo, hi = self._range_for_payload(var_name, value, is_target=is_target)

        return {
            "internal": var_name,
            "display": self._display_name(var_name),
            "value": self._format_value(value),
            "raw_value": value,
            "min": lo,
            "max": hi,
            "norm": self._normalise(value, lo, hi),
            "step": float(self.variables.get(var_name, {}).get("step", 1)),
            "is_target": is_target,
        }

    def _range_for_payload(
        self,
        var_name: str,
        value: float,
        is_target: bool,
    ) -> tuple[float, float]:
        if is_target:
            return self._target_range(var_name, value)
        return self._variable_range(var_name, value)

    # -------------------------------------------------------------------------
    # HTML + screenshot
    # -------------------------------------------------------------------------

    def _render_html(
        self,
        render_data: Dict[str, Any],
        step_id: Optional[int],
    ) -> str:
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
    ) -> str:
        filename = (
            f"step_{step_id:04d}.png" if step_id is not None else "step_none.png"
        )
        image_path = self._base_dir / filename
        file_uri = Path(html_path).as_uri()

        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=self.headless,
                slow_mo=self.slow_mo,
            )
            page = browser.new_page(
                viewport={
                    "width": self.viewport_width,
                    "height": self.viewport_height,
                }
            )
            page.goto(file_uri, wait_until="networkidle")
            page.screenshot(path=str(image_path), full_page=False)
            browser.close()

        return str(image_path.resolve())

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def _require_state_keys(self, state: Dict[str, Any]) -> None:
        required = self._REQUIRED_STATE_KEYS[self.simulation_type]
        missing = [key for key in required if key not in state]

        if missing:
            raise KeyError(
                f"simulation_type='{self.simulation_type}' requires missing "
                f"state keys: {missing}"
            )

    # -------------------------------------------------------------------------
    # Ranges and normalisation
    # -------------------------------------------------------------------------

    def _variable_range(
        self,
        var_name: str,
        current_value: float,
    ) -> tuple[float, float]:
        cfg = self.variables.get(var_name, {})
        lo_raw = cfg.get("min_value")
        hi_raw = cfg.get("max_value")

        if lo_raw is not None and hi_raw is not None:
            lo = float(lo_raw)
            hi = float(hi_raw)
            if hi > lo:
                return lo, hi

        hi = max(1.0, current_value)
        return 0.0, hi

    def _target_range(
        self,
        var_name: str,
        current_value: float,
    ) -> tuple[float, float]:
        """
        Stable target ranges derived from config max values.

        These ranges prevent visual elements from changing scale between steps,
        which is important for comparing ordered screenshots.
        """
        if self.simulation_type == "mass":
            hi = self._max("density") * self._max("volume")
            return 0.0, max(hi, current_value, 1.0)

        if self.simulation_type == "ohm":
            hi = self._max("current") * self._max("resistance")
            return 0.0, max(hi, current_value, 1.0)

        if self.simulation_type == "resistors":
            hi = self._max("resistance_1") + self._max("resistance_2")
            return 0.0, max(hi, current_value, 1.0)

        if self.simulation_type == "distance":
            hi = self._max("distance_1") + self._max("distance_2") + self._max("distance_3")
            return 0.0, max(hi, current_value, 1.0)

        if self.simulation_type == "kinematics":
            hi = self._max("initial_position") + self._max("velocity") * self._max("time")
            return 0.0, max(hi, current_value, 1.0)

        if self.simulation_type == "beers":
            hi = (
                self._max("molar_absorptivity")
                * self._max("concentration")
                * self._max("path_length")
            )
            return 0.0, max(hi, current_value, 1.0)

        if self.simulation_type == "beers_wavelength":
            hi = (
                self._max("wavelength")
                * self._max("concentration")
                * self._max("path_length")
                / 50.0
            )
            return 0.0, max(hi, current_value, 1.0)

        if self.simulation_type == "concentration":
            lo_volume, _ = self._variable_range("solution_volume", current_value)
            min_volume = max(lo_volume, 1e-9)
            hi = self._max("solute_amount") / min_volume
            return 0.0, max(hi, current_value, 1.0)

        return 0.0, max(1.0, current_value)

    def _norm_var(self, var_name: str, value: float) -> float:
        lo, hi = self._variable_range(var_name, value)
        return self._normalise(value, lo, hi)

    def _norm_target(self, var_name: str, value: float) -> float:
        lo, hi = self._target_range(var_name, value)
        return self._normalise(value, lo, hi)

    def _max(self, var_name: str) -> float:
        cfg = self.variables.get(var_name, {})
        max_value = cfg.get("max_value")

        if max_value is not None:
            return float(max_value)

        initial_value = cfg.get("initial_value")
        if initial_value is not None:
            return float(initial_value)

        return 1.0

    @staticmethod
    def _normalise(value: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.5
        norm = (value - lo) / (hi - lo)
        return max(0.0, min(1.0, norm))

    # -------------------------------------------------------------------------
    # Basic helpers
    # -------------------------------------------------------------------------

    def _value(self, state: Dict[str, Any], var_name: str) -> float:
        raw = state[var_name]
        if not isinstance(raw, (int, float)):
            raise ValueError(
                f"state value for '{var_name}' must be numeric, "
                f"got {type(raw).__name__}"
            )
        return float(raw)

    def _display_name(self, variable: str) -> str:
        if self.naming_mode == "abstract":
            return self.name_mapping.get(variable, variable)
        return variable

    @staticmethod
    def _format_value(value: float) -> str:
        """
        Keep instrument readouts compact so numbers do not overflow panels.
        """
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.2f}".rstrip("0").rstrip(".")