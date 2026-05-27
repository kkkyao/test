from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.benchmarks.runner import VisualBenchmarkRunner


def load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML dictionary")

    return data


def apply_overrides(
    config: Dict[str, Any],
    n_sequences: Optional[int],
    modalities: Optional[str],
    backend: Optional[str],
    model_name: Optional[str],
) -> Dict[str, Any]:
    if n_sequences is not None:
        config["benchmark"]["n_sequences"] = int(n_sequences)

    if modalities:
        config["benchmark"]["modalities"] = [
            item.strip() for item in modalities.split(",") if item.strip()
        ]

    if backend:
        config.setdefault("agent", {})["backend"] = backend

    if model_name:
        config.setdefault("agent", {})["model_name"] = model_name

    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run static text/visual chart QA benchmark."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmark_visual_beers_wavelength.yaml",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--n_sequences",
        type=int,
        default=None,
        help="Override benchmark.n_sequences.",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default=None,
        help="Comma-separated override, e.g. text,bar,line,scatter,simulation",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Override agent.backend, e.g. mock_benchmark or hf_qwen_vl.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Override agent.model_name.",
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=None,
        help="Debug option: only run first N cases.",
    )

    # W&B options
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name. If omitted, W&B logging is disabled.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Optional W&B entity/team name.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional W&B run name. Defaults to --run_name.",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="Optional W&B mode.",
    )

    args = parser.parse_args()

    config = load_yaml(args.config)
    config = apply_overrides(
        config=config,
        n_sequences=args.n_sequences,
        modalities=args.modalities,
        backend=args.backend,
        model_name=args.model_name,
    )

    benchmark_name = config["benchmark"].get("name", "visual_benchmark")
    run_name = args.run_name or benchmark_name

    output_root = Path(
        config.get("logging", {}).get("output_dir", "outputs/visual_benchmark")
    )
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = VisualBenchmarkRunner(
        config=config,
        output_dir=str(output_dir),
        max_cases=args.max_cases,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name or run_name,
        wandb_mode=args.wandb_mode,
    )
    runner.run()


if __name__ == "__main__":
    main()