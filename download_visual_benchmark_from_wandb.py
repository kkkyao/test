from pathlib import Path
import wandb

ENTITY = "leyao-li-epfl"
PROJECT = "visual-benchmarks"

ARTIFACTS = {
    "qwen25vl_3b_all_modalities_rerun": "qwen25vl_3b_all_modalities_rerun_results:latest",
    "qwen25vl_7b_all_modalities": "qwen25vl_7b_all_modalities_results:latest",
}

# 如果你 3B 最后是 rerun，就把上面 3B 这一行改成：
# "qwen25vl_3b_all_modalities_rerun": "qwen25vl_3b_all_modalities_rerun_results:latest",

OUT_ROOT = Path("wandb_visual_benchmark_downloads")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

api = wandb.Api()

for run_name, artifact_name in ARTIFACTS.items():
    full_name = f"{ENTITY}/{PROJECT}/{artifact_name}"
    out_dir = OUT_ROOT / run_name

    print(f"Downloading {full_name} -> {out_dir}")
    artifact = api.artifact(full_name, type="visual_benchmark_results")
    artifact.download(root=str(out_dir))

print("Done.")
