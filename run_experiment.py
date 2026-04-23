from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

import wandb

from run_episode import build_model_callable
from src.agents.agent import TextLLMAgent
from src.envs.env import EquationEnv
from src.evaluation.evaluator import EpisodeEvaluator
from src.observation.renderer import TextRenderer
from src.prompts.prompt_builder import PromptBuilder
from src.runners.runner import EpisodeRunner
from src.tracing.logger import EpisodeLogger
from src.utils.config_loader import load_config


def compute_aggregate(all_evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate metrics across all runs."""
    n = len(all_evaluations)
    if n == 0:
        return {}

    termination_reasons = [e["termination_reason"] for e in all_evaluations]
    steps_to_success = [
        e["steps_to_success"] for e in all_evaluations
        if e["steps_to_success"] is not None
    ]

    return {
        "n_runs":                    n,
        "success_rate":              sum(e["success"] for e in all_evaluations) / n,
        "finish_rate":               sum(e["finish_called"] for e in all_evaluations) / n,
        "parse_error_rate":          sum(1 for r in termination_reasons if r == "parse_error") / n,
        "mean_total_steps":          statistics.mean(e["total_steps"] for e in all_evaluations),
        "mean_steps_to_success":     statistics.mean(steps_to_success) if steps_to_success else None,
        "mean_state_coverage_ratio": statistics.mean(e["state_coverage_ratio"] for e in all_evaluations),
        "mean_redundancy_penalty":   statistics.mean(e["redundancy_penalty"] for e in all_evaluations),
        "mean_variable_isolation_score": statistics.mean(
            e["variable_isolation_score"] for e in all_evaluations
        ),
        "mean_discovery_efficiency": statistics.mean(e["discovery_efficiency"] for e in all_evaluations),
        "termination_breakdown": {
            reason: sum(1 for r in termination_reasons if r == reason)
            for reason in ("finish_success", "finish_wrong", "max_steps", "parse_error")
        },
    }


def main(
    config_path: str,
    n_runs: int,
    wandb_project: str,
    wandb_entity: Optional[str],
    run_name: Optional[str],
) -> None:
    config = load_config(config_path)

    experiment_cfg    = config["experiment"]
    environment_cfg   = config["environment"]
    actions_cfg       = config["actions"]
    agent_cfg         = config["agent"]
    representation_cfg = config.get("representation", {})
    logging_cfg       = config.get("logging", {})
    evaluation_cfg    = config.get("evaluation", {})

    target_variable  = environment_cfg["target_variable"]
    variables        = environment_cfg["variables"]
    equations        = environment_cfg["equations"]
    action_mode      = actions_cfg["action_mode"]
    max_steps        = experiment_cfg["max_steps"]
    naming_mode      = representation_cfg.get("naming_mode", "concrete")
    metadata_level   = representation_cfg.get("metadata_level", "minimal")
    name_mapping     = representation_cfg.get("name_mapping", {})
    base_output_dir  = logging_cfg.get("output_dir", "outputs/default_run")

    if target_variable not in equations:
        raise ValueError(
            f"target_variable '{target_variable}' must exist in environment.equations"
        )

    # ── W&B init ──────────────────────────────────────────────────────────────
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name or experiment_cfg.get("name", "experiment"),
        config={
            "n_runs":          n_runs,
            "model_name":      agent_cfg.get("model_name"),
            "backend":         agent_cfg.get("backend"),
            "naming_mode":     naming_mode,
            "metadata_level":  metadata_level,
            "max_steps":       max_steps,
            "target_variable": target_variable,
            "action_mode":     action_mode,
            "experiment_name": experiment_cfg.get("name"),
            "temperature":     agent_cfg.get("generation", {}).get("temperature"),
            "top_p":           agent_cfg.get("generation", {}).get("top_p"),
            "top_k":           agent_cfg.get("generation", {}).get("top_k"),
            "_config_sources": config.get("_config_sources", {}),
        },
        tags=[
            agent_cfg.get("model_name", "unknown"),
            naming_mode,
            experiment_cfg.get("name", "experiment"),
        ],
    )

    # ── Build components once — model is loaded into GPU once only ────────────
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
        prompt_config=config["prompt"],
        target_variable=target_variable,
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

    ground_truth     = equations[target_variable]
    variable_mapping = evaluation_cfg.get("variable_mapping")
    evaluator = EpisodeEvaluator(
        ground_truth_equation=ground_truth,
        variable_mapping=variable_mapping,
    )

    # ── W&B table — one row per episode ───────────────────────────────────────
    episode_table = wandb.Table(columns=[
        "run_id", "success", "equation_correct", "finish_called",
        "termination_reason", "total_steps", "steps_to_success",
        "final_equation", "state_coverage_ratio", "redundancy_penalty",
        "variable_isolation_score", "discovery_efficiency", "error_message",
    ])

    all_evaluations: List[Dict[str, Any]] = []

    # ── Main loop ─────────────────────────────────────────────────────────────
    for run_id in range(n_runs):
        print(f"\n--- Run {run_id + 1}/{n_runs} ---")

        run_output_dir = str(Path(base_output_dir) / f"run_{run_id:02d}")
        logger = EpisodeLogger(
            output_dir=run_output_dir,
            save_steps=logging_cfg.get("save_steps", True),
            save_trajectory=logging_cfg.get("save_trajectory", True),
            save_interaction_log=logging_cfg.get("save_interaction_log", True),
        )

        result     = runner.run_episode()
        evaluation = evaluator.evaluate(result)
        saved_paths = logger.save_episode(result, evaluation=evaluation)
        all_evaluations.append(evaluation)

        # Upload trajectory files as W&B Artifact for this run
        artifact = wandb.Artifact(
            name=f"episode-{run_name or experiment_cfg.get('name', 'run')}-{run_id:02d}",
            type="episode_data",
            metadata={
                "run_id":           run_id,
                "model":            agent_cfg.get("model_name"),
                "naming_mode":      naming_mode,
                "termination":      evaluation["termination_reason"],
                "success":          evaluation["success"],
                "final_equation":   evaluation["final_equation"],
            },
        )
        for artifact_name, file_path in saved_paths.items():
            artifact.add_file(file_path, name=f"{artifact_name}.json")
        wandb.log_artifact(artifact)

        # Per-episode W&B log
        wandb.log(
            {
                "success":                   int(evaluation["success"]),
                "equation_correct":          int(evaluation["equation_correct"]),
                "finish_called":             int(evaluation["finish_called"]),
                "total_steps":               evaluation["total_steps"],
                "steps_to_success":          evaluation["steps_to_success"] or 0,
                "state_coverage_ratio":      evaluation["state_coverage_ratio"],
                "redundancy_penalty":        evaluation["redundancy_penalty"],
                "variable_isolation_score":  evaluation["variable_isolation_score"],
                "discovery_efficiency":      evaluation["discovery_efficiency"],
            },
            step=run_id,
        )

        episode_table.add_data(
            run_id,
            evaluation["success"],
            evaluation["equation_correct"],
            evaluation["finish_called"],
            evaluation["termination_reason"],
            evaluation["total_steps"],
            evaluation["steps_to_success"],
            evaluation["final_equation"],
            evaluation["state_coverage_ratio"],
            evaluation["redundancy_penalty"],
            evaluation["variable_isolation_score"],
            evaluation["discovery_efficiency"],
            evaluation["error_message"],
        )

        print(
            f"  success={evaluation['success']}  "
            f"steps={evaluation['total_steps']}  "
            f"reason={evaluation['termination_reason']}"
        )

    # ── Aggregate ─────────────────────────────────────────────────────────────
    aggregate = compute_aggregate(all_evaluations)

    aggregate_path = Path(base_output_dir) / "aggregate.json"
    with aggregate_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)

    # Log table and aggregate to W&B
    wandb.log({"episodes": episode_table})
    for key, value in aggregate.items():
        if isinstance(value, (int, float)) and value is not None:
            wandb.summary[key] = value
    wandb.summary["termination_breakdown"] = aggregate["termination_breakdown"]

    print("\n=== Experiment complete ===")
    print(f"  success_rate: {aggregate['success_rate']:.2%}")
    print(f"  finish_rate:  {aggregate['finish_rate']:.2%}")
    print(f"  mean_steps:   {aggregate['mean_total_steps']:.1f}")
    if aggregate["mean_steps_to_success"] is not None:
        print(f"  mean_steps_to_success: {aggregate['mean_steps_to_success']:.1f}")
    print(f"  aggregate saved to: {aggregate_path}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run N episodes and log results to W&B."
    )
    parser.add_argument("--config",        type=str, default="configs/config.yaml")
    parser.add_argument("--n_runs",        type=int, default=10)
    parser.add_argument("--wandb_project", type=str, default="scientific-exploration")
    parser.add_argument("--wandb_entity",  type=str, default=None)
    parser.add_argument("--run_name",      type=str, default=None,
                        help="W&B run display name. Defaults to experiment.name in config.")
    args = parser.parse_args()

    main(
        config_path=args.config,
        n_runs=args.n_runs,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        run_name=args.run_name,
    )