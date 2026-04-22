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

    Sub-configs loaded:
        experiment_config  →  experiment + logging blocks
        env_config         →  environment + representation + actions + evaluation + noise
        model_config       →  agent block
        prompt_config      →  prompt block  (NEW)

    First-stage loader uses top-level block merge only.
    """
    main_config_path = Path(main_config_path)
    main_config = load_yaml(main_config_path)
    base_dir = main_config_path.parent

    experiment_path = _resolve_subconfig(base_dir, main_config.get("experiment_config"))
    env_path        = _resolve_subconfig(base_dir, main_config.get("env_config"))
    model_path      = _resolve_subconfig(base_dir, main_config.get("model_config"))
    prompt_path     = _resolve_subconfig(base_dir, main_config.get("prompt_config"))

    experiment_config = load_yaml(experiment_path)
    env_config        = load_yaml(env_path)
    model_config      = load_yaml(model_path)
    prompt_config     = load_yaml(prompt_path)

    final_config: Dict[str, Any] = {}
    final_config.update(experiment_config)
    final_config.update(env_config)
    final_config.update(model_config)
    final_config.update(prompt_config)   # adds the "prompt" top-level block

    final_config["_config_sources"] = {
        "main":       str(main_config_path),
        "experiment": str(experiment_path),
        "env":        str(env_path),
        "model":      str(model_path),
        "prompt":     str(prompt_path),
    }

    _validate_config(final_config)
    return final_config


def _resolve_subconfig(base_dir: Path, filename: Any) -> Path:
    """
    Resolve a sub-config path relative to the main config directory.
    Accepts both relative and absolute paths.
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
    Minimal validation on the merged final config.
    """
    required_top_level = ["experiment", "environment", "actions", "agent", "prompt"]
    for key in required_top_level:
        if key not in config:
            raise ValueError(f"final config must contain top-level key: '{key}'")

    for key in required_top_level:
        if not isinstance(config[key], dict):
            raise ValueError(f"'{key}' must be a dictionary")

    experiment  = config["experiment"]
    environment = config["environment"]
    actions     = config["actions"]
    prompt      = config["prompt"]

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

    # Prompt config must carry the core text blocks.
    required_prompt_keys = [
        "system_intro",
        "task_template",
        "exploration_lines",
        "step_type_descriptions",
        "rules",
        "output_format_header",
        "output_format_schema",
        "section_headers",
        "history_labels",
    ]
    missing = [k for k in required_prompt_keys if k not in prompt]
    if missing:
        raise ValueError(
            f"'prompt' config is missing required keys: {missing}"
        )