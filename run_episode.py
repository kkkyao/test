from __future__ import annotations

import argparse
import json
from typing import Any, Callable, Dict

from src.agents.agent import TextLLMAgent
from src.envs.env import EquationEnv
from src.observation.renderer import TextRenderer
from src.prompts.prompt_builder import PromptBuilder
from src.runners.runner import EpisodeRunner
from src.tracing.logger import EpisodeLogger
from src.utils.config_loader import load_config


def build_model_callable(agent_config: Dict[str, Any]) -> Callable[[str], str]:
    """
    Build a model callable from agent config.

    Supported backends:
    - mock: multi-step placeholder for pipeline testing
    - others: raise NotImplementedError
    """
    backend = agent_config.get("backend", "mock")

    if backend == "mock":
        call_count = 0

        def mock_model(prompt: str) -> str:
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return json.dumps(
                    {
                        "step_type": "action",
                        "reasoning": "I will increase concentration to observe how absorbance changes.",
                        "hypothesis": None,
                        "action": {
                            "action_type": "increase",
                            "variable": "concentration",
                            "value": None,
                        },
                        "final_equation": None,
                    }
                )

            if call_count == 2:
                return json.dumps(
                    {
                        "step_type": "hypothesis",
                        "reasoning": "Absorbance seems to increase when concentration increases.",
                        "hypothesis": "Absorbance increases with concentration.",
                        "action": None,
                        "final_equation": None,
                    }
                )

            return json.dumps(
                {
                    "step_type": "finish",
                    "reasoning": "I am confident enough to provide the final equation.",
                    "hypothesis": None,
                    "action": None,
                    "final_equation": "absorbance = concentration * path_length / 200",
                }
            )

        return mock_model

    raise NotImplementedError(
        f"Unsupported backend '{backend}'. "
        "Replace build_model_callable() with a real model backend."
    )


def main(config_path: str) -> None:
    config = load_config(config_path)

    experiment_cfg = config["experiment"]
    environment_cfg = config["environment"]
    actions_cfg = config["actions"]
    agent_cfg = config["agent"]
    representation_cfg = config.get("representation", {})
    logging_cfg = config.get("logging", {})

    target_variable = environment_cfg["target_variable"]
    variables = environment_cfg["variables"]
    equations = environment_cfg["equations"]
    action_mode = actions_cfg["action_mode"]
    max_steps = experiment_cfg["max_steps"]

    if target_variable not in equations:
        raise ValueError(
            f"target_variable '{target_variable}' must exist in environment.equations"
        )

    naming_mode = representation_cfg.get("naming_mode", "concrete")
    metadata_level = representation_cfg.get("metadata_level", "minimal")
    name_mapping = representation_cfg.get("name_mapping", {})

    display_target_variable = (
        name_mapping.get(target_variable, target_variable)
        if naming_mode == "abstract"
        else target_variable
    )

    output_dir = logging_cfg.get("output_dir", "outputs/default_run")
    save_steps = logging_cfg.get("save_steps", True)
    save_trajectory = logging_cfg.get("save_trajectory", True)
    save_interaction_log = logging_cfg.get("save_interaction_log", True)

    env = EquationEnv(
        variables=variables,
        equations=equations,
        action_mode=action_mode,
    )

    renderer = TextRenderer(
        variables=variables,
        action_mode=action_mode,
        target_variable=target_variable,
        naming_mode=naming_mode,
        metadata_level=metadata_level,
        name_mapping=name_mapping,
    )

    prompt_builder = PromptBuilder(
        target_variable=display_target_variable,
        max_steps=max_steps,
        include_history=True,
        history_window=experiment_cfg.get("history_window"),
    )

    model_callable = build_model_callable(agent_cfg)
    agent = TextLLMAgent(model_callable=model_callable)

    runner = EpisodeRunner(
        env=env,
        renderer=renderer,
        prompt_builder=prompt_builder,
        agent=agent,
        max_steps=max_steps,
    )

    logger = EpisodeLogger(
        output_dir=output_dir,
        save_steps=save_steps,
        save_trajectory=save_trajectory,
        save_interaction_log=save_interaction_log,
    )

    result = runner.run_episode()
    saved_paths = logger.save_episode(result)

    print("\n=== Episode finished ===")
    print(f"Config: {config_path}")
    print(f"Target variable: {target_variable}")
    print(f"Finish reached: {result.get('finish_reached')}")
    print(f"Total steps: {result.get('num_steps')}")
    print(f"Final equation: {result.get('final_equation')}")

    print("\nSaved files:")
    for name, path in saved_paths.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run one scientific exploration episode."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the main config YAML file.",
    )
    args = parser.parse_args()

    main(args.config)