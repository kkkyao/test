import csv
import wandb

ENTITY = "leyao-li-epfl"
PROJECT = "scientific-exploration"

TARGET_TYPE = "episode_data"
OUT_CSV = "wandb_artifact_list.csv"

api = wandb.Api()

rows = []

project_path = f"{ENTITY}/{PROJECT}"

print(f"Exporting artifact collections from: {project_path}")
print(f"Target artifact type: {TARGET_TYPE}")

for artifact_type in api.artifact_types(project=project_path):
    print("Artifact type:", artifact_type.name)

    if artifact_type.name != TARGET_TYPE:
        continue

    for collection in artifact_type.collections():
        collection_name = collection.name

        # 保险处理：如果 collection.name 意外带了 :version，就去掉
        if ":" in collection_name:
            collection_name = collection_name.split(":", 1)[0]

        full_name = f"{ENTITY}/{PROJECT}/{collection_name}:latest"

        rows.append({
            "type": TARGET_TYPE,
            "collection_name": collection_name,
            "version": "latest",
            "aliases": "latest",
            "full_name": full_name,
        })

rows.sort(key=lambda r: r["collection_name"])

with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "type",
            "collection_name",
            "version",
            "aliases",
            "full_name",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print("=" * 80)
print(f"Saved {len(rows)} rows to {OUT_CSV}")
print(f"Project: {ENTITY}/{PROJECT}")
print(f"Rows containing 10runs: {sum('10runs' in r['collection_name'] for r in rows)}")
print("=" * 80)

print("\nFirst 30 rows containing 10runs:")
shown = 0
for r in rows:
    if "10runs" in r["collection_name"]:
        print(" ", r["full_name"])
        shown += 1
        if shown >= 30:
            break