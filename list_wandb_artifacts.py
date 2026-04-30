import csv
import wandb

ENTITY = "leyao-li-epfl"
PROJECT = "scientific-exploration"
PROJECT_PATH = f"{ENTITY}/{PROJECT}"

api = wandb.Api()

rows = []

print(f"Listing artifacts from: {PROJECT_PATH}")

for artifact_type in api.artifact_types(project=PROJECT_PATH):
    type_name = artifact_type.name
    print(f"\n[TYPE] {type_name}")

    try:
        collections = artifact_type.collections()
    except Exception as e:
        print(f"  [ERROR] Cannot list collections for type={type_name}: {e}")
        continue

    for collection in collections:
        collection_name = collection.name
        print(f"  [COLLECTION] {collection_name}")

        try:
            versions = list(collection.artifacts())
        except Exception as e:
            print(f"    [ERROR] Cannot list versions: {e}")
            rows.append({
                "type": type_name,
                "collection_name": collection_name,
                "version": "",
                "aliases": "",
                "full_name": f"{ENTITY}/{PROJECT}/{collection_name}",
            })
            continue

        for artifact in versions:
            version = getattr(artifact, "version", "")
            aliases = ",".join(getattr(artifact, "aliases", []) or [])

            full_name = f"{ENTITY}/{PROJECT}/{collection_name}:{version}"
            print(f"    {full_name} aliases=[{aliases}]")

            rows.append({
                "type": type_name,
                "collection_name": collection_name,
                "version": version,
                "aliases": aliases,
                "full_name": full_name,
            })

out_csv = "wandb_artifact_list.csv"

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["type", "collection_name", "version", "aliases", "full_name"],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved artifact list to: {out_csv}")