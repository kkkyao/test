from pathlib import Path
import wandb

ENTITY = "leyao-li-epfl"
PROJECT = "visual-benchmarks-new"

# artifact name = f"{run_name}_results:latest"
RUNS = {
    "abstract_visual_qwen25vl_3b": "abstract_visual_qwen25vl_3b_results:latest",
    "abstract_visual_qwen25vl_7b": "abstract_visual_qwen25vl_7b_results:latest",
}

OUT_ROOT = Path("wandb_visual_benchmark_downloads") / "abstract_visual"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

api = wandb.Api()

for run_name, artifact_name in RUNS.items():
    full_name = f"{ENTITY}/{PROJECT}/{artifact_name}"
    out_dir = OUT_ROOT / run_name

    print(f"Downloading {full_name} -> {out_dir}")
    artifact = api.artifact(full_name, type="visual_benchmark_results")
    artifact.download(root=str(out_dir))
    downloaded = [f.name for f in out_dir.iterdir()]
    print(f"  Files: {downloaded}")

print("\nAll downloads complete.")
