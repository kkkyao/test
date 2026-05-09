from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable, Dict, List, Optional

from src.agents.agent import TextLLMAgent
from src.envs.env import EquationEnv
from src.evaluation.evaluator import EpisodeEvaluator
from src.observation.renderer import TextRenderer
from src.prompts.prompt_builder import PromptBuilder
from src.runners.runner import EpisodeRunner
from src.tracing.logger import EpisodeLogger
from src.utils.config_loader import load_config


# ── Default mock action sequence used when none is supplied in config ─────────
_DEFAULT_MOCK_VLM_SEQUENCE: List[Dict[str, Any]] = [
    {"step_type": "action", "action_type": "increase", "variable": "A",
     "reasoning": "Mock: increase first variable."},
    {"step_type": "action", "action_type": "increase", "variable": "B",
     "reasoning": "Mock: increase second variable."},
    {"step_type": "action", "action_type": "increase", "variable": "A",
     "reasoning": "Mock: increase first variable again."},
    {"step_type": "finish", "final_equation": "Y = A * B",
     "reasoning": "Mock: submit placeholder equation."},
]


def build_model_callable(agent_config: Dict[str, Any]) -> Callable[[str], str]:
    """
    Build a text-only model callable from agent config.

    Supported backends:
    - mock:    multi-step placeholder for pipeline testing
    - hf_qwen: local Hugging Face inference (Qwen, Llama, Mistral, Gemma, …)

    NOTE: run_experiment.py imports this function directly, so its name and
    signature must not change.
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
                    "action": {"action_type": "increase", "variable": "concentration"},
                    "final_equation": None,
                })

            if call_count == 2:
                return json.dumps({
                    "step_type": "action",
                    "reasoning": "Now I will vary path_length while keeping concentration fixed.",
                    "action": {"action_type": "increase", "variable": "path_length"},
                    "final_equation": None,
                })

            return json.dumps({
                "step_type": "finish",
                "reasoning": "I am confident enough to provide the final equation.",
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
        max_new_tokens = int(generation_cfg.get("max_new_tokens", 1024))
        temperature    = float(generation_cfg.get("temperature", 0.7))
        top_p          = float(generation_cfg.get("top_p", 0.8))
        top_k          = int(generation_cfg.get("top_k", 20))
        do_sample      = bool(generation_cfg.get("do_sample", temperature > 0.0))

        device_map        = agent_config.get("device_map", "auto")
        trust_remote_code = bool(agent_config.get("trust_remote_code", True))
        disable_thinking  = bool(agent_config.get("disable_thinking", False))

        torch_dtype_cfg = str(
            agent_config.get("dtype", agent_config.get("torch_dtype", "auto"))
        ).lower()

        dtype_map = {
            "auto":     "auto",
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
            "float32":  torch.float32,
        }
        if torch_dtype_cfg not in dtype_map:
            raise ValueError(f"agent.dtype must be one of: {list(dtype_map)}")
        torch_dtype: Any = dtype_map[torch_dtype_cfg]

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        model.eval()

        def hf_qwen_model(prompt: str) -> str:
            messages = [{"role": "user", "content": prompt}]

            template_kwargs: Dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if disable_thinking:
                template_kwargs["enable_thinking"] = False

            try:
                rendered_text = tokenizer.apply_chat_template(messages, **template_kwargs)
            except TypeError:
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


def build_vlm_callable(
    agent_config: Dict[str, Any],
) -> Callable[[str, List[str]], str]:
    """
    Build a VLM callable from agent config.

    The callable signature is (prompt: str, image_paths: List[str]) -> str.
    image_paths is an ordered list of absolute PNG paths (oldest first).
    An empty list means no images are available for this call.

    Supported backends:
    - mock_vlm:   dummy callable; images are accepted but ignored.
                  Use MockVLMAgent directly instead (see build_agent_text_image);
                  this stub exists only as a callable fallback.
    - hf_qwen_vl: Qwen2.5-VL loaded locally via Hugging Face transformers.
    """
    backend = agent_config.get("backend", "mock_vlm")

    if backend == "mock_vlm":
        # Dummy callable: ignores images, returns a fixed finish JSON.
        # In practice, build_agent_text_image() uses MockVLMAgent directly
        # and never calls this function.
        def mock_vlm_callable(prompt: str, image_paths: List[str]) -> str:
            return json.dumps({
                "step_type": "finish",
                "reasoning": "Mock VLM callable stub.",
                "action": None,
                "final_equation": "Y = A * B",
            })

        return mock_vlm_callable

    if backend == "hf_qwen_vl":
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as e:
            raise ImportError(
                "qwen_vl_utils is required for backend='hf_qwen_vl'. "
                "Install with: pip install qwen-vl-utils"
            ) from e

        model_name = agent_config.get("model_name")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                "agent.model_name must be a non-empty string when backend='hf_qwen_vl'"
            )

        torch_dtype_cfg = str(
            agent_config.get("dtype", agent_config.get("torch_dtype", "bfloat16"))
        ).lower()
        dtype_map = {
            "auto":     "auto",
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
            "float32":  torch.float32,
        }
        if torch_dtype_cfg not in dtype_map:
            raise ValueError(f"agent.dtype must be one of: {list(dtype_map)}")
        torch_dtype = dtype_map[torch_dtype_cfg]

        device_map        = agent_config.get("device_map", "auto")
        trust_remote_code = bool(agent_config.get("trust_remote_code", True))
        generation_cfg    = agent_config.get("generation", {})
        max_new_tokens    = int(generation_cfg.get("max_new_tokens", 1024))
        temperature       = float(generation_cfg.get("temperature", 0.0))
        do_sample         = bool(generation_cfg.get("do_sample", False))
        top_p             = float(generation_cfg.get("top_p", 1.0))

        print(f"Loading VLM: {model_name} ...")
        processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        model.eval()
        print("VLM loaded.")

        def hf_qwen_vl_callable(prompt: str, image_paths: List[str]) -> str:
            # Build multimodal message: images first, then text
            content: List[Dict[str, Any]] = []
            for p in image_paths:
                content.append({"type": "image", "image": f"file://{p}"})
            content.append({"type": "text", "text": prompt})

            messages = [{"role": "user", "content": content}]

            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text_input],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                )

            # Trim prompt tokens, decode only generated tokens
            trimmed = output_ids[0][inputs["input_ids"].shape[1]:]
            return processor.decode(trimmed, skip_special_tokens=True).strip()

        return hf_qwen_vl_callable

    raise NotImplementedError(
        f"Unsupported VLM backend '{backend}'. "
        f"Supported: 'mock_vlm', 'hf_qwen_vl'."
    )


def build_agent_text_image(
    agent_config: Dict[str, Any],
    name_mapping: Optional[Dict[str, str]] = None,
) -> Any:
    """
    Build the agent for text+image mode.

    - mock_vlm   → MockVLMAgent with action_sequence from config (or default)
    - hf_qwen_vl → VisionLanguageAgent wrapping a real Qwen2.5-VL callable
    """
    from src.agents.mock_vlm_agent import MockVLMAgent
    from src.agents.vlm_agent import VisionLanguageAgent

    backend = agent_config.get("backend", "mock_vlm")

    if backend == "mock_vlm":
        # Action sequence: use config value if provided, else default
        raw_seq = agent_config.get("mock_action_sequence", _DEFAULT_MOCK_VLM_SEQUENCE)

        # If name_mapping is active (abstract naming), translate variable names
        # in the sequence from display names to whatever the config provides.
        # The sequence in config should already use display names (e.g. A, B, Y).
        action_sequence = list(raw_seq)

        return MockVLMAgent(
            action_sequence=action_sequence,
            loop=bool(agent_config.get("loop", False)),
        )

    if backend == "hf_qwen_vl":
        vlm_callable = build_vlm_callable(agent_config)
        return VisionLanguageAgent(
            model_callable=vlm_callable,
            strip_markdown_fences=True,
        )

    raise NotImplementedError(
        f"Unsupported text+image backend '{backend}'. "
        f"Supported: 'mock_vlm', 'hf_qwen_vl'."
    )


def main(
    config_path: str,
    env_config: str | None = None,
    model_config: str | None = None,
) -> None:
    config = load_config(
        config_path,
        env_config_override=env_config,
        model_config_override=model_config,
    )

    experiment_cfg     = config["experiment"]
    environment_cfg    = config["environment"]
    actions_cfg        = config["actions"]
    agent_cfg          = config["agent"]
    representation_cfg = config.get("representation", {})
    visual_cfg         = config.get("visual", {})
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

    # New: observation mode switch (default = "text" to preserve existing behaviour)
    observation_mode = (
    visual_cfg.get("observation_mode")
    or representation_cfg.get("observation_mode", "text")
    )
    image_history_window = visual_cfg.get("image_history_window", None)

    output_dir           = logging_cfg.get("output_dir", "outputs/default_run")
    save_steps           = logging_cfg.get("save_steps", True)
    save_trajectory      = logging_cfg.get("save_trajectory", True)
    save_interaction_log = logging_cfg.get("save_interaction_log", True)

    if target_variable not in equations:
        raise ValueError(
            f"target_variable '{target_variable}' must exist in environment.equations"
        )

    # ── Environment (shared by both modes) ───────────────────────────────────
    env = EquationEnv(
        variables=variables, equations=equations, action_mode=action_mode
    )

    # ── Prompt builder (shared by both modes) ─────────────────────────────────
    prompt_builder = PromptBuilder(
        prompt_config=config["prompt"],
        target_variable=target_variable,
        max_steps=max_steps,
        action_mode=action_mode,
        equation_variables=evaluation_cfg.get("equation_variables", []),
        include_history=True,
        history_window=experiment_cfg.get("history_window"),
    )

    # ── Renderer + Agent: branch on observation_mode ──────────────────────────
    if observation_mode == "text":
        # ── Original text-only path (unchanged) ──────────────────────────────
        renderer = TextRenderer(
            variables=variables,
            action_mode=action_mode,
            target_variable=target_variable,
            naming_mode=naming_mode,
            metadata_level=metadata_level,
            name_mapping=name_mapping,
        )
        model_callable = build_model_callable(agent_cfg)
        agent          = TextLLMAgent(model_callable=model_callable)

    elif observation_mode == "text_image":
        # ── New text+image path ───────────────────────────────────────────────
        from src.observation.html_chart_renderer import HtmlChartRenderer
        from src.observation.text_image_renderer import TextImageRenderer

        template_path = visual_cfg.get(
            "template_path", "templates/chart_state_view.html"
        )
        images_output_dir = os.path.join(
            output_dir, visual_cfg.get("output_subdir", "images")
        )

        text_renderer = TextRenderer(
            variables=variables,
            action_mode=action_mode,
            target_variable=target_variable,
            naming_mode=naming_mode,
            metadata_level=metadata_level,
            name_mapping=name_mapping,
        )
        chart_renderer = HtmlChartRenderer(
            variables=variables,
            target_variable=target_variable,
            naming_mode=naming_mode,
            name_mapping=name_mapping if name_mapping else None,
            output_dir=images_output_dir,
            template_path=template_path,
            headless=visual_cfg.get("playwright", {}).get("headless", True),
            slow_mo=visual_cfg.get("playwright", {}).get("slow_mo", 0),
            normalize_bars=visual_cfg.get("normalize_bars", True),
        )
        renderer = TextImageRenderer(
            text_renderer=text_renderer,
            chart_renderer=chart_renderer,
        )
        agent = build_agent_text_image(agent_cfg, name_mapping=name_mapping)

    else:
        raise ValueError(
            f"Unknown observation_mode: '{observation_mode}'. "
            f"Must be 'text' or 'text_image'."
        )

    # ── Runner ────────────────────────────────────────────────────────────────
    runner = EpisodeRunner(
        env=env,
        renderer=renderer,
        prompt_builder=prompt_builder,
        agent=agent,
        max_steps=max_steps,
        image_history_window=image_history_window,
    )

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = EpisodeLogger(
        output_dir=output_dir,
        save_steps=save_steps,
        save_trajectory=save_trajectory,
        save_interaction_log=save_interaction_log,
    )

    # ── Run ───────────────────────────────────────────────────────────────────
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

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Episode finished ===")
    print(f"Config:           {config_path}")
    if env_config:
        print(f"Env override:     {env_config}")
    if model_config:
        print(f"Model override:   {model_config}")
    print(f"Observation mode: {observation_mode}")
    print(f"Target variable:  {target_variable}")
    print(f"Finish reached:   {result.get('finish_reached')}")
    print(f"Total steps:      {result.get('num_steps')}")
    print(f"Final equation:   {result.get('final_equation')}")

    image_paths = result.get("image_paths", [])
    if image_paths:
        print(f"Screenshots:      {len(image_paths)} PNG(s) saved")
        print(f"  First: {image_paths[0]}")
        print(f"  Last:  {image_paths[-1]}")

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
             "E.g. configs/model_qwen35_4b.yaml",
    )
    args = parser.parse_args()
    main(args.config, env_config=args.env_config, model_config=args.model_config)