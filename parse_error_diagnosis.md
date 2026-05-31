# Parse Error Diagnosis Report

**Target episodes:**
- index=900 · Chart-only · beers · abbrev · model_qwen25_vl_3b · failed at step 6
- index=1086 · Chart-only · beers · abbrev · model_qwen25_vl_3b · failed at step 2

---

## 是否能从 W&B 下载文件里直接定位原因

**能，但原始 raw output 只保留了截断版本。**

`summary.json` 里的 `parse_error` 字段保存了 300 字符的截断 raw output 和精确的 JSONDecodeError 信息（错误类型 + 字节位置）。这已经足够判断根因。

缺失的信息：**完整的 raw output**（失败步的完整模型输出）没有保存到任何文件中。`trajectory.json` 和 `interaction_log.json` 只记录**成功解析**的步骤，失败的那次调用没有日志。

---

## Episode 1 详细分析（index=900）

| 字段 | 值 |
|---|---|
| episode_path | `.../episode-chart_only_no_action_beers_abbrev_model_qwen25_vl_3b-03_latest` |
| failed_step | 6（trajectory 共有步骤 0-5，step 6 的调用失败） |
| termination_reason | `parse_error` |
| finish_reached | `false` |
| 错误消息 | `[json_decode] model output is not valid JSON (error: Expecting ',' delimiter at pos 252)` |

**直接原因：JSON 字段间缺少逗号（missing comma）**

模型输出的 JSON 结构如下（从 error 消息中截取）：

```json
{
  "reasoning": "Observing the bar chart, I notice that the orange bar, which represents
variable A, is significantly taller than the blue bars representing variables epsilon and
c. This suggests that A is positively correlated with epsilon and c."
  "step_type": "action",
  "action": {
    "action_type": ...
```

问题：`"reasoning"` 字段值结束后，直接跟了 `"step_type"`，**缺少逗号分隔符**。

`json.loads` 在位置 252（`"` 后、`"step_type"` 前）报错：`Expecting ',' delimiter`。

**为什么 `_repair()` 没有修复这个问题：**

`_repair()` 处理的是「多余逗号」（trailing comma），即 `regex: ,(\s*[}\]])`，但它**不能添加缺失的逗号**。少了逗号的 JSON 在语法层面完全不可自动修复，没有明确规则判断哪里该加。

**错误分类：J（其他 — JSON 字段间缺少逗号）**

成功步骤 0-5 的 raw output 均保存在 `trajectory.json` / `interaction_log.json` 中。失败的 step 6 调用仅在 `summary.json.parse_error` 里有截断痕迹。

---

## Episode 2 详细分析（index=1086）

| 字段 | 值 |
|---|---|
| episode_path | `.../episode-chart_only_no_action_beers_abbrev_model_qwen25_vl_3b-06_latest` |
| failed_step | 2（trajectory 共有步骤 0-1，step 2 的调用失败） |
| termination_reason | `parse_error` |
| finish_reached | `false` |
| 错误消息 | `[json_decode] model output is not valid JSON (error: Invalid \escape at pos 285)` |

**直接原因：`final_equation` 字段含 LaTeX 反斜杠转义（`\epsilon`、`\frac`）**

模型输出的 JSON（从 error 消息截取）：

```json
{
  "reasoning": "Observing the bar chart, it appears that as epsilon increases, A also
increases, while both c and l decrease. This suggests a relationship where A depends on
epsilon and inversely on both c and l.",
  "step_type": "finish",
  "action": null,
  "final_equation": "A = \epsilon - \frac{...}"
```

JSON 标准只允许有限的转义序列（`\"`, `\\`, `\/`, `\b`, `\f`, `\n`, `\r`, `\t`, `\uXXXX`）。  
`\e`（`\epsilon` 的前两个字符）**不是合法的 JSON 转义序列**，json.loads 在位置 285 报 `Invalid \escape`。

注：`\f` 本身是合法的 JSON 转义（换页符），但 `\frac` 里 `\f` 被解析为换页符，剩余 `rac` 紧跟其后，语义上是 LaTeX 污染，并非 JSON 语法层面引发此错误（真正的问题是 `\e`）。

**为什么 `_repair()` 没有修复这个问题：**

`_repair()` 只修复字符串内的控制字符（literal `\n`、`\r`、`\t` → 转义为 `\\n`、`\\r`、`\\t`）。它处理的是「字符串里有未转义的控制字符」，而不是「字符串里有无效的 `\X` 序列」。`\epsilon` 里的 `\e` 已经是一个转义尝试，`_repair()` 没有逻辑将 `\e` → `\\e`。

**错误分类：J（其他 — LaTeX 符号污染 `final_equation` 字段，`\epsilon`/`\frac` 为无效 JSON 转义序列）**

步骤 0（```json...``` 格式）和步骤 1 的 raw output 均正常保存。失败的 step 2 仅有截断痕迹。

---

## 缺少的字段

两个 episode 都存在同一个结构性日志空白：

> **失败步的完整 raw output 没有被任何文件保存。**

`interaction_log.json` 和 `trajectory.json` 只记录**成功解析**的步骤。当 `_act_with_retry()` 所有 retry 都失败时，最后的 raw output 只以截断形式（约 300 字符）嵌入到 `parse_error` 错误消息字符串里，存入 `summary.json`。完整的 raw output（可能有 500-2000 字符）永久丢失。

这使得：
- 无法复现完整的 JSON 修复尝试过程
- 无法统计失败 JSON 的完整结构模式
- 无法判断是否有更多 LaTeX 符号或其他字段污染

---

## 下一步最小修复建议

### 1. 修复 EP1 类型（missing comma）

`_repair()` 无法自动补逗号，但可以用 `json5` 或 `demjson3` 库来解析这类"宽松 JSON"，它们容忍缺少逗号的情况：

```python
try:
    import json5
    data = json5.loads(candidate)
except ImportError:
    pass  # 回退到现有逻辑
```

或者用正则在 `"` 后、`"` 前（跨行）插入逗号，但这个正则比较脆弱。

### 2. 修复 EP2 类型（LaTeX `\escape` 污染）

在 `_repair()` 中添加一步：将字符串值内的非法 `\X` 序列（其中 X 不属于 `"\\bfnrtu/`）替换为 `\\X`（双反斜杠转义）：

```python
# 在现有 _repair() 里，step 1（字符串内控制字符处理）之后，添加：
# 将 invalid \X escape 替换为 \\X（仅当 in_string）
# 合法转义: " \ / b f n r t u
VALID_ESCAPES = set('"\\\/bfnrtu')
# 在逐字符遍历中，遇到 \ 且下一字符不在 VALID_ESCAPES 时，输出 \\ 而不是 \
```

### 3. 修复日志空白（最小改动）

在 `EpisodeRunner._act_with_retry()` 里，把**最终失败的 raw output 完整保存**：

```python
# 在返回 (None, "", last_error) 之前，把 last_raw 一并返回
return None, last_raw, last_error  # 不截断
```

并在 `interaction_log.json` 里为失败调用写一条 `{"step_id": X, "attempt": N, "raw_model_output": ..., "parse_error": ...}` 记录，方便事后检查。

---

## 总结

| index | 错误类型 | 直接原因 | `_repair()` 能修复？ |
|---|---|---|---|
| 900 | J — 缺少逗号 | reasoning 字段后缺 `,`，step_type 前没有分隔符 | 否 |
| 1086 | J — LaTeX 转义污染 | `final_equation` 含 `\epsilon`、`\frac`，`\e` 是无效 JSON 转义 | 否 |

两个 parse error 均已从 W&B 下载文件中完整定位原因，但失败步的完整 raw output 已经丢失（只有 300 字符截断）。
