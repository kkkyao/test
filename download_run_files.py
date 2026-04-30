import wandb
from pathlib import Path

api = wandb.Api()

ENTITY = "leyao-li-epfl"
PROJECT = "scientific-exploration"
PROJECT_PATH = f"{ENTITY}/{PROJECT}"

FILES_TO_DOWNLOAD = {
    "output.log",
    "wandb-summary.json",
}

OUT_ROOT = Path("./wandb_run_files")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# 第一次建议 True，只预览有哪些 run 和文件。
# 确认没问题后改成 False 正式下载。
DRY_RUN = False

# 如果已经下载过，就跳过
SKIP_IF_EXISTS = True


def safe_name(name: str) -> str:
    return (
        name.replace("/", "__")
        .replace(":", "_")
        .replace(" ", "_")
    )


runs = api.runs(PROJECT_PATH)

selected = []

for run in runs:
    run_id = run.id
    run_name = run.name or run_id

    try:
        files = list(run.files())
    except Exception as e:
        print(f"[ERROR] Cannot list files for run {run_id}: {e}")
        continue

    matched_files = []

    for f in files:
        if f.name in FILES_TO_DOWNLOAD:
            matched_files.append(f)

    if matched_files:
        selected.append((run, matched_files))

print(f"Matched runs: {len(selected)}")

if DRY_RUN:
    print("\nPreview first 50 runs:")
    for run, matched_files in selected[:50]:
        print(f"\nRun: {run.name} / {run.id}")
        for f in matched_files:
            print(f"  file: {f.name}")

    if len(selected) > 50:
        print(f"\n... and {len(selected) - 50} more runs")

    print("\nDRY_RUN=True, so nothing was downloaded.")
    print("Set DRY_RUN=False to download.")
    raise SystemExit


for idx, (run, matched_files) in enumerate(selected, start=1):
    run_id = run.id
    run_name = run.name or run_id

    out_dir = OUT_ROOT / f"{safe_name(run_name)}__{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{idx}/{len(selected)}] Run: {run_name} / {run_id}")

    for f in matched_files:
        out_path = out_dir / f.name

        if SKIP_IF_EXISTS and out_path.exists():
            print(f"  already exists, skip: {f.name}")
            continue

        try:
            print(f"  downloading: {f.name}")
            f.download(root=str(out_dir), replace=True)
        except Exception as e:
            print(f"  [ERROR] failed to download {f.name}: {e}")

print("\nDone.")
print(f"Saved under: {OUT_ROOT.resolve()}")