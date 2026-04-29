import wandb
from pathlib import Path

api = wandb.Api()

ENTITY = "leyao-li-epfl"
PROJECT = "scientific-exploration"

PREFIX = "episode-resistors_concrete_qwen35_4b"
VERSION = "v3"

# 下载 00 到 32，共 33 个
START = 0
END = 32

FILES = {
    "interaction_log.json",
    "steps.json",
    "summary.json",
    "trajectory.json",
}

out_root = Path("./wandb_episode_jsons")
out_root.mkdir(parents=True, exist_ok=True)

for i in range(START, END + 1):
    artifact_name = f"{PREFIX}-{i:02d}:{VERSION}"
    artifact_path = f"{ENTITY}/{PROJECT}/{artifact_name}"

    print(f"\n[Artifact] {artifact_path}")

    try:
        artifact = api.artifact(artifact_path, type="episode_data")
    except Exception as e:
        print(f"  [ERROR] Cannot load artifact: {artifact_path}")
        print(f"  {e}")
        continue

    out_dir = out_root / artifact_name.replace(":", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    found = set()

    for f in artifact.files():
        print("  found:", f.name)

        if f.name in FILES:
            print("  downloading:", f.name)
            f.download(root=str(out_dir), replace=True)
            found.add(f.name)

    missing = FILES - found
    if missing:
        print("  [WARNING] missing:", missing)

print("\nDone.")
print(f"Saved under: {out_root.resolve()}")