import wandb
import os

ENTITY = "leyao-li-epfl"
PROJECT = "scientific-exploration"

api = wandb.Api()

runs = api.runs(f"{ENTITY}/{PROJECT}")

print(f"Found {len(runs)} runs")

for run in runs:
    run_name = run.name.replace("/", "_")
    save_dir = f"wandb_downloads/{run_name}"

    os.makedirs(save_dir, exist_ok=True)

    print(f"Downloading {run.name}")

    for file in run.files():
        file.download(root=save_dir, replace=True)

print("Done.")