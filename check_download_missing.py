import csv
from pathlib import Path

SELECTED_CSV = "selected_artifacts_with_time.csv"

# 这里要和下载脚本里的 out_root 一致
OUT_ROOT = Path("./wandb_episode_jsons_all")

FILES = {
    "interaction_log.json",
    "steps.json",
    "summary.json",
    "trajectory.json",
    "output.log",
}

missing_rows = []
complete_rows = []

with open(SELECTED_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        collection_name = row["collection_name"]
        version = row["version"]

        out_dir = OUT_ROOT / f"{collection_name}_{version}"

        missing = sorted(
            filename for filename in FILES
            if not (out_dir / filename).exists()
        )

        if missing:
            missing_rows.append({
                **row,
                "out_dir": str(out_dir),
                "missing_files": ",".join(missing),
            })
        else:
            complete_rows.append(row)

print("=" * 80)
print(f"Selected artifacts: {len(complete_rows) + len(missing_rows)}")
print(f"Complete downloads: {len(complete_rows)}")
print(f"Incomplete / missing downloads: {len(missing_rows)}")
print("=" * 80)

with open("not_downloaded_or_incomplete.csv", "w", encoding="utf-8", newline="") as f:
    fieldnames = [
        "type",
        "collection_name",
        "version",
        "aliases",
        "full_name",
        "created_at",
        "out_dir",
        "missing_files",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(missing_rows)

print("Wrote: not_downloaded_or_incomplete.csv")

print("\nFirst 50 incomplete/missing:")
for row in missing_rows[:50]:
    print(row["collection_name"], "missing:", row["missing_files"])