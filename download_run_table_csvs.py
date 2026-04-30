import csv
import json
import wandb
import pandas as pd
from pathlib import Path

api = wandb.Api()

CSV_PATH = "wandb_artifact_list.csv"

TARGET_TYPE = "run_table"

OUT_ROOT = Path("./wandb_run_tables")
RAW_DIR = OUT_ROOT / "raw"
CSV_DIR = OUT_ROOT / "csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

VERSION_MODE = "latest"
# 可选：
# VERSION_MODE = "all"
# VERSION_MODE = "v0"

DRY_RUN = False
# 第一次先 True 预览。
# 确认没问题后改成 False 正式下载和转换。


def safe_name(name: str) -> str:
    return name.replace("/", "__").replace(":", "_")


selected_rows = []

with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["type"] != TARGET_TYPE:
            continue

        version = row["version"]
        aliases = row.get("aliases", "") or ""

        if VERSION_MODE == "latest":
            if "latest" not in aliases:
                continue
        elif VERSION_MODE == "all":
            pass
        elif VERSION_MODE.startswith("v"):
            if version != VERSION_MODE:
                continue
        else:
            raise ValueError(f"Unsupported VERSION_MODE: {VERSION_MODE}")

        selected_rows.append(row)


selected_rows.sort(key=lambda r: r["full_name"])

print(f"Matched run_table artifacts: {len(selected_rows)}")

if DRY_RUN:
    print("\nPreview first 100 run_table artifacts:")
    for row in selected_rows[:100]:
        print(" ", row["full_name"])

    if len(selected_rows) > 100:
        print(f"\n... and {len(selected_rows) - 100} more")

    print("\nDRY_RUN=True, so nothing was downloaded.")
    print("Set DRY_RUN=False to download and convert.")
    raise SystemExit


all_dfs = []

for idx, row in enumerate(selected_rows, start=1):
    artifact_path = row["full_name"]
    collection_name = row["collection_name"]
    version = row["version"]

    print(f"\n[{idx}/{len(selected_rows)}] {artifact_path}")

    artifact_raw_dir = RAW_DIR / f"{collection_name}_{version}"
    artifact_raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        artifact = api.artifact(artifact_path, type=TARGET_TYPE)
    except Exception as e:
        print(f"  [ERROR] Cannot load artifact: {e}")
        continue

    downloaded_files = []

    try:
        for f in artifact.files():
            print("  found:", f.name)
            downloaded = f.download(root=str(artifact_raw_dir), replace=True)
            downloaded_files.append(Path(downloaded.name if hasattr(downloaded, "name") else downloaded))
    except Exception as e:
        print(f"  [ERROR] download failed: {e}")
        continue

    table_json_files = list(artifact_raw_dir.rglob("*.table.json"))

    if not table_json_files:
        # 有些情况下可能就是普通 json
        table_json_files = [
            p for p in artifact_raw_dir.rglob("*.json")
            if p.name != "wandb_manifest.json"
        ]

    if not table_json_files:
        print("  [WARNING] no table json found")
        continue

    for table_json in table_json_files:
        print("  converting:", table_json)

        try:
            with open(table_json, "r", encoding="utf-8") as f:
                obj = json.load(f)

            if "columns" not in obj or "data" not in obj:
                print("  [WARNING] json does not contain columns/data:", table_json)
                continue

            df = pd.DataFrame(obj["data"], columns=obj["columns"])

            df.insert(0, "source_artifact", artifact_path)
            df.insert(1, "source_collection", collection_name)
            df.insert(2, "source_version", version)

            out_csv = CSV_DIR / f"{safe_name(collection_name + ':' + version)}__{table_json.stem}.csv"
            df.to_csv(out_csv, index=False)

            print("  saved csv:", out_csv)

            all_dfs.append(df)

        except Exception as e:
            print(f"  [ERROR] convert failed: {e}")
            continue


if all_dfs:
    merged = pd.concat(all_dfs, ignore_index=True)
    merged_csv = OUT_ROOT / "all_run_tables_merged.csv"
    merged.to_csv(merged_csv, index=False)
    print(f"\nMerged CSV saved to: {merged_csv}")
    print(f"Rows: {len(merged)}")
else:
    print("\nNo tables converted.")

print("\nDone.")
print(f"Raw files: {RAW_DIR.resolve()}")
print(f"CSV files: {CSV_DIR.resolve()}")