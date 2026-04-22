from __future__ import annotations

import argparse
import json
from typing import Any, Callable, Dict

from src.agents.agent import TextLLMAgent
from src.envs.env import EquationEnv
from src.evaluation.evaluator import EpisodeEvaluator
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
    - hf_qwen: local Hugging Face Qwen inference
    """
    backend = agent_config.get("backend", "mock")

    if backend == "mock":
        call_count = 0

        def mock_model(prompt: str) -> str:
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return json.dumps({
                    "step_type": "action",
                    "reasoning": "I will increase concentration to observe how absorbance changes.",
                    "hypothesis": "Absorbance may be proportional to concentration.",
                    "action": {"action_type": "increase", "variable": "concentration", "value": None},
                    "final_equation": None,
                })

            if call_count == 2:
                return json.dumps({
                    "step_type": "action",
                    "reasoning": "Now I will vary path_length while keeping concentration fixed.",
                    "hypothesis": None,
                    "action": {"action_type": "increase", "variable": "path_length", "value": None},
                    "final_equation": None,
                })

            return json.dumps({
                "step_type": "finish",
                "reasoning": "I am confident enough to provide the final equation.",
                "hypothesis": None,
                "action": None,
                "final_equation": "absorbance = concentration * path_length / 200",
            })

        return mock_model

    if backend == "hf_qwen":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = agent_config.get("model_name")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                "agent.model_name must be a non-empty string when backend='hf_qwen'"
            )

        generation_cfg = agent_config.get("generation", {})
        max_new_tokens = int(generation_cfg.get("max_new_tokens", 512))
        temperature    = float(generation_cfg.get("temperature", 0.7))
        top_p          = float(generation_cfg.get("top_p", 0.8))
        top_k          = int(generation_cfg.get("top_k", 20))
        do_sample      = bool(generation_cfg.get("do_sample", temperature > 0.0))

        device_map        = agent_config.get("device_map", "auto")
        torch_dtype_cfg   = str(agent_config.get("torch_dtype", "auto")).lower()
        trust_remote_code = bool(agent_config.get("trust_remote_code", True))

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if torch_dtype_cfg not in dtype_map:
            raise ValueError(f"agent.torch_dtype must be one of: {list(dtype_map)}")
        torch_dtype: Any = dtype_map[torch_dtype_cfg]

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        model.eval()

        def hf_qwen_model(prompt: str) -> str:
            messages = [{"role": "user", "content": prompt}]
            rendered_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(rendered_text, return_tensors="pt")
            if hasattr(model, "device"):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.eos_token_id,
                )

            prompt_length = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][prompt_length:]
            return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return hf_qwen_model

    raise NotImplementedError(
        f"Unsupported backend '{backend}'. Supported: 'mock', 'hf_qwen'."
    )


def main(
    config_path: str,
    env_config: str | None = None,
    model_config: str | None = None,
) -> None:
    config = load_config(config_path, env_config_override=env_config, model_config_override=model_config)

    experiment_cfg     = config["experiment"]
    environment_cfg    = config["environment"]
    actions_cfg        = config["actions"]
    agent_cfg          = config["agent"]
    representation_cfg = config.get("representation", {})
    logging_cfg        = config.get("logging", {})
    evaluation_cfg     = config.get("evaluation", {})

    target_variable = environment_cfg["target_variable"]
    variables       = environment_cfg["variables"]
    equations       = environment_cfg["equations"]
    action_mode     = actions_cfg["action_mode"]
    max_steps       = experiment_cfg["max_steps"]
    auto_evaluate   = experiment_cfg.get("auto_evaluate", False)
    naming_mode     = representation_cfg.get("naming_mode", "concrete")
    metadata_level  = representation_cfg.get("metadata_level", "minimal")
    name_mapping    = representation_cfg.get("name_mapping", {})

    output_dir           = logging_cfg.get("output_dir", "outputs/default_run")
    save_steps           = logging_cfg.get("save_steps", True)
    save_trajectory      = logging_cfg.get("save_trajectory", True)
    save_interaction_log = logging_cfg.get("save_interaction_log", True)

    if target_variable not in equations:
        raise ValueError(
            f"target_variable '{target_variable}' must exist in environment.equations"
        )

    env            = EquationEnv(variables=variables, equations=equations, action_mode=action_mode)
    renderer       = TextRenderer(
        variables=variables, action_mode=action_mode, target_variable=target_variable,
        naming_mode=naming_mode, metadata_level=metadata_level, name_mapping=name_mapping,
    )
    prompt_builder = PromptBuilder(
        prompt_config=config["prompt"], target_variable=target_variable,
        max_steps=max_steps, include_history=True,
        history_window=experiment_cfg.get("history_window"),
    )
    model_callable = build_model_callable(agent_cfg)
    agent          = TextLLMAgent(model_callable=model_callable)
    runner         = EpisodeRunner(
        env=env, renderer=renderer, prompt_builder=prompt_builder,
        agent=agent, max_steps=max_steps,
    )
    logger         = EpisodeLogger(
        output_dir=output_dir, save_steps=save_steps,
        save_trajectory=save_trajectory, save_interaction_log=save_interaction_log,
    )

    result = runner.run_episode()

    evaluation = None
    if auto_evaluate:
        ground_truth     = equations[target_variable]
        variable_mapping = evaluation_cfg.get("variable_mapping")
        evaluator = EpisodeEvaluator(
            ground_truth_equation=ground_truth,
            variable_mapping=variable_mapping,
        )
        evaluation = evaluator.evaluate(result)

    saved_paths = logger.save_episode(result, evaluation=evaluation)

    print("\n=== Episode finished ===")
    print(f"Config:          {config_path}")
    if env_config:
        print(f"Env override:    {env_config}")
    if model_config:
        print(f"Model override:  {model_config}")
    print(f"Target variable: {target_variable}")
    print(f"Finish reached:  {result.get('finish_reached')}")
    print(f"Total steps:     {result.get('num_steps')}")
    print(f"Final equation:  {result.get('final_equation')}")

    if evaluation is not None:
        print("\n=== Evaluation ===")
        print(f"Success:            {evaluation.get('success')}")
        print(f"Equation correct:   {evaluation.get('equation_correct')}")
        print(f"Termination reason: {evaluation.get('termination_reason')}")

    print("\nSaved files:")
    for name, path in saved_paths.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run one scientific exploration episode."
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to the main config YAML file.",
    )
    parser.add_argument(
        "--env_config", type=str, default=None,
        help="Override the env_config from main config. "
             "E.g. configs/env_beers_abstract.yaml",
    )
    parser.add_argument(
        "--model_config", type=str, default=None,
        help="Override the model_config from main config. "
             "E.g. configs/model_qwen3_4b.yaml",
    )
    args = parser.parse_args()
    main(args.config, env_config=args.env_config, model_config=args.model_config)