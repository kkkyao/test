import csv
import re
import wandb
from pathlib import Path

api = wandb.Api()

CSV_PATH = "wandb_artifact_list.csv"

TARGET_TYPE = "episode_data"

FILES = {
    "interaction_log.json",
    "steps.json",
    "summary.json",
    "trajectory.json",
}

out_root = Path("./wandb_episode_jsons_all")
out_root.mkdir(parents=True, exist_ok=True)

# ============================================================
# 下载配置
# ============================================================

# 推荐下载 latest。
# 如果想下载所有历史版本，改成 "all"。
# 如果只想下载某个版本，改成 "v0" / "v1" / "v2" / "v3"。
VERSION_MODE = "latest"

TASKS = {
    "ohm",
    "mass",
    "kinematics",
    "beers",
    "distance",
    "resistors",
    "fma",
}

FORMS = {
    "abbrev",
    "abstract",
    "concrete",
}

MODELS = {
    "gemma3_4b",
    "llama31_8b",
    "mistral_7b",
    "qwen25_3b",
    "qwen25_7b",
    "qwen35_4b",
    "qwen35_9b",
}

# 不下载 episode-abstract_qwen35_4b-00 / episode-concrete_qwen35_4b-00 这种没有 task 的 artifact
INCLUDE_NO_TASK_SPECIAL = False

# 不下载 episode-test_...
INCLUDE_TEST_ARTIFACT = False

# 如果某个 artifact 的四个 JSON 已经完整存在，就跳过
SKIP_IF_COMPLETE = True

# 第一次可以先设 True 预览数量和名字。
# 确认没问题后改成 False 正式下载。
DRY_RUN = False

FAILED_LOG = "failed_artifacts.txt"
MISSING_LOG = "missing_files.txt"


# ============================================================
# 工具函数
# ============================================================

def get_base_name(collection_name: str) -> str:
    """
    去掉最后的 -00 / -01 / -29 / -32 编号。
    例如：
      episode-resistors_concrete_qwen35_4b-24
    变成：
      episode-resistors_concrete_qwen35_4b
    """
    return re.sub(r"-\d+$", "", collection_name)


def get_episode_index(collection_name: str) -> int:
    """
    从 collection_name 末尾提取 episode 编号。
    """
    m = re.search(r"-(\d+)$", collection_name)
    if m:
        return int(m.group(1))
    return -1


def match_base_name(base_name: str) -> bool:
    """
    只匹配标准格式：

      episode-{task}_{form}_{model}

    例如：
      episode-ohm_abbrev_gemma3_4b
      episode-mass_concrete_qwen35_4b
      episode-resistors_abstract_qwen25_7b

    不匹配：
      episode-abstract_qwen35_4b
      episode-concrete_qwen35_4b
    """
    if not base_name.startswith("episode-"):
        return False

    name = base_name[len("episode-"):]

    if name.startswith("test_"):
        return INCLUDE_TEST_ARTIFACT

    parts = name.split("_")

    # 标准格式至少是：
    # task_form_model_part1_model_part2
    # 例如 resistors_concrete_qwen35_4b
    if len(parts) < 4:
        return False

    task = parts[0]
    form = parts[1]
    model = "_".join(parts[2:])

    return (
        task in TASKS
        and form in FORMS
        and model in MODELS
    )


def is_complete(out_dir: Path) -> bool:
    """
    四个目标 JSON 都存在，就认为已经完整下载。
    """
    return all((out_dir / filename).exists() for filename in FILES)


# ============================================================
# 从 CSV 里筛选已有 artifact
# ============================================================

selected_rows = []

with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        artifact_type = row["type"]
        collection_name = row["collection_name"]
        version = row["version"]
        aliases = row.get("aliases", "") or ""

        if artifact_type != TARGET_TYPE:
            continue

        base_name = get_base_name(collection_name)

        if not match_base_name(base_name):
            continue

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


selected_rows.sort(
    key=lambda r: (
        get_base_name(r["collection_name"]),
        get_episode_index(r["collection_name"]),
        r["version"],
    )
)

print(f"Matched artifact versions: {len(selected_rows)}")


# ============================================================
# 预览模式
# ============================================================

if DRY_RUN:
    print("\nPreview first 200 matched artifacts:")
    for row in selected_rows[:200]:
        print(" ", row["full_name"])

    if len(selected_rows) > 200:
        print(f"\n... and {len(selected_rows) - 200} more")

    print("\nDRY_RUN=True, so nothing was downloaded.")
    print("After checking the preview, set DRY_RUN=False to download.")
    raise SystemExit


# ============================================================
# 正式下载
# ============================================================

Path(FAILED_LOG).write_text("", encoding="utf-8")
Path(MISSING_LOG).write_text("", encoding="utf-8")

for idx, row in enumerate(selected_rows, start=1):
    artifact_path = row["full_name"]
    collection_name = row["collection_name"]
    version = row["version"]

    out_dir = out_root / f"{collection_name}_{version}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{idx}/{len(selected_rows)}] [Artifact] {artifact_path}")

    if SKIP_IF_COMPLETE and is_complete(out_dir):
        print("  already complete, skip")
        continue

    try:
        artifact = api.artifact(artifact_path, type=TARGET_TYPE)
    except Exception as e:
        print(f"  [ERROR] Cannot load artifact: {artifact_path}")
        print(f"  {e}")

        with open(FAILED_LOG, "a", encoding="utf-8") as log:
            log.write(f"{artifact_path}\tLOAD_ERROR\t{repr(e)}\n")

        continue

    found = set()

    try:
        for f in artifact.files():
            print("  found:", f.name)

            if f.name in FILES:
                print("  downloading:", f.name)
                f.download(root=str(out_dir), replace=True)
                found.add(f.name)

    except Exception as e:
        print(f"  [ERROR] Download failed: {artifact_path}")
        print(f"  {e}")

        with open(FAILED_LOG, "a", encoding="utf-8") as log:
            log.write(f"{artifact_path}\tDOWNLOAD_ERROR\t{repr(e)}\n")

        continue

    missing = FILES - found

    if missing:
        print("  [WARNING] missing:", missing)

        with open(MISSING_LOG, "a", encoding="utf-8") as log:
            log.write(f"{artifact_path}\tmissing={sorted(missing)}\n")

print("\nDone.")
print(f"Saved under: {out_root.resolve()}")
print(f"Failed log: {Path(FAILED_LOG).resolve()}")
print(f"Missing files log: {Path(MISSING_LOG).resolve()}")