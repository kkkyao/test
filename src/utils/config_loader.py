from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml


PathLike = Union[str, Path]


def load_yaml(path: PathLike) -> Dict[str, Any]:
    """
    Load one YAML file and return a dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"YAML file is empty: {path}")

    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a dictionary at top level: {path}")

    return data


def load_config(main_config_path: PathLike) -> Dict[str, Any]:
    """
    Load the main config and merge referenced sub-configs into one final config.
    """
    main_config_path = Path(main_config_path)
    main_config = load_yaml(main_config_path)
    base_dir = main_config_path.parent

    experiment_path = _resolve_subconfig(base_dir, main_config.get("experiment_config"))
    env_path = _resolve_subconfig(base_dir, main_config.get("env_config"))
    model_path = _resolve_subconfig(base_dir, main_config.get("model_config"))

    experiment_config = load_yaml(experiment_path)
    env_config = load_yaml(env_path)
    model_config = load_yaml(model_path)

    # First-stage loader uses top-level block merge only.
    final_config: Dict[str, Any] = {}
    final_config.update(experiment_config)
    final_config.update(env_config)
    final_config.update(model_config)

    final_config["_config_sources"] = {
        "main": str(main_config_path),
        "experiment": str(experiment_path),
        "env": str(env_path),
        "model": str(model_path),
    }

    _validate_config(final_config)
    return final_config


def _resolve_subconfig(base_dir: Path, filename: Any) -> Path:
    """
    Resolve a sub-config path relative to the main config directory.
    """
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("sub-config filename must be a non-empty string")

    path = Path(filename)
    if not path.is_absolute():
        path = base_dir / path

    if not path.exists():
        raise FileNotFoundError(f"sub-config file not found: {path}")

    return path


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Perform minimal validation on the merged final config.
    """
    required_top_level = ["experiment", "environment", "actions", "agent"]
    for key in required_top_level:
        if key not in config:
            raise ValueError(f"final config must contain top-level key: '{key}'")

    experiment = config["experiment"]
    environment = config["environment"]
    actions = config["actions"]
    agent = config["agent"]

    if not isinstance(experiment, dict):
        raise ValueError("'experiment' must be a dictionary")
    if not isinstance(environment, dict):
        raise ValueError("'environment' must be a dictionary")
    if not isinstance(actions, dict):
        raise ValueError("'actions' must be a dictionary")
    if not isinstance(agent, dict):
        raise ValueError("'agent' must be a dictionary")

    if "max_steps" not in experiment:
        raise ValueError("'experiment.max_steps' must be provided")

    if "variables" not in environment:
        raise ValueError("'environment.variables' must be provided")

    if "equations" not in environment:
        raise ValueError("'environment.equations' must be provided")

    if "target_variable" not in environment:
        raise ValueError("'environment.target_variable' must be provided")

    if "action_mode" not in actions:
        raise ValueError("'actions.action_mode' must be provided")