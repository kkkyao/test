from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from src.envs.equation_engine import EquationEngine
from src.schemas.action_schema import ActionSpec


class EquationEnv:
    """
    Environment that maintains a manipulable state and recomputes outputs
    from config-defined equations after each action.
    """

    def __init__(
        self,
        variables: Dict[str, Dict[str, Any]],
        equations: Dict[str, str],
        action_mode: str,
    ) -> None:
        if not isinstance(variables, dict) or not variables:
            raise ValueError("variables must be a non-empty dictionary")

        if action_mode not in {"increase_decrease", "set_value"}:
            raise ValueError(
                "action_mode must be either 'increase_decrease' or 'set_value'"
            )

        self.variables = deepcopy(variables)
        self.action_mode = action_mode

        for variable, cfg in self.variables.items():
            if not isinstance(cfg, dict):
                raise ValueError(f"variable config for '{variable}' must be a dictionary")

        equation_targets = set(equations.keys())
        manipulable_names = {
            k for k, v in self.variables.items() if v.get("manipulable", False)
        }
        conflicts = equation_targets & manipulable_names
        if conflicts:
            raise ValueError(
                f"equation targets must not overlap with manipulable variable definitions: {sorted(conflicts)}"
            )

        for variable, cfg in self.variables.items():
            if cfg.get("manipulable", False) and "initial_value" not in cfg:
                raise ValueError(
                    f"manipulable variable '{variable}' must define initial_value"
                )

        if self.action_mode == "increase_decrease":
            for variable, cfg in self.variables.items():
                if cfg.get("manipulable", False):
                    if "step_size" not in cfg or cfg["step_size"] is None:
                        raise ValueError(
                            f"manipulable variable '{variable}' must define step_size "
                            "in increase_decrease mode"
                        )

        self.engine = EquationEngine(equations)

        self.initial_state = self._build_initial_state()
        self.state = deepcopy(self.initial_state)
        self._recompute_outputs()

    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment to its initial state and recompute outputs.
        """
        self.state = deepcopy(self.initial_state)
        self._recompute_outputs()
        return self.get_state()

    def step(self, action: ActionSpec) -> Dict[str, Any]:
        """
        Apply one action to the environment and recompute outputs.
        """
        if not isinstance(action, ActionSpec):
            raise ValueError("action must be an ActionSpec instance")

        variable = action.variable
        if variable not in self.variables:
            raise KeyError(f"unknown variable: '{variable}'")

        var_cfg = self.variables[variable]
        if not var_cfg.get("manipulable", False):
            raise ValueError(f"variable '{variable}' is not manipulable")

        if self.action_mode == "increase_decrease":
            if action.action_type not in {"increase", "decrease"}:
                raise ValueError(
                    f"action_type '{action.action_type}' is not allowed in "
                    f"action_mode='increase_decrease'"
                )

            current_value = self.state[variable]
            step_size = var_cfg["step_size"]

            if action.action_type == "increase":
                new_value = current_value + step_size
            else:
                new_value = current_value - step_size

        else:  # set_value
            if action.action_type != "set":
                raise ValueError(
                    f"action_type '{action.action_type}' is not allowed in "
                    f"action_mode='set_value'"
                )
            if action.value is None:
                raise ValueError(
                    f"set action for variable '{variable}' must provide a numeric value"
                )

            new_value = action.value

        new_value = self._apply_bounds(variable, new_value)
        self.state[variable] = new_value
        self._recompute_outputs()

        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        """
        Return a copy of the current environment state.
        """
        return deepcopy(self.state)

    def _build_initial_state(self) -> Dict[str, Any]:
        """
        Build the initial state from variable definitions.
        """
        state: Dict[str, Any] = {}

        for variable, cfg in self.variables.items():
            if "initial_value" in cfg:
                value = cfg["initial_value"]
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"initial_value for variable '{variable}' must be numeric, "
                        f"got {type(value).__name__}"
                    )
                state[variable] = float(value)

        return state

    def _recompute_outputs(self) -> None:
        """
        Recompute all equation-defined outputs and write them into state.
        """
        outputs = self.engine.compute_all(self.state)
        self.state.update(outputs)

    def _apply_bounds(self, variable: str, value: Any) -> float:
        """
        Apply optional min/max bounds to a variable value.
        """
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"value for variable '{variable}' must be numeric, got {type(value).__name__}"
            )

        value = float(value)
        cfg = self.variables[variable]

        min_value = cfg.get("min_value", None)
        max_value = cfg.get("max_value", None)

        if min_value is not None:
            value = max(value, float(min_value))

        if max_value is not None:
            value = min(value, float(max_value))

        return value