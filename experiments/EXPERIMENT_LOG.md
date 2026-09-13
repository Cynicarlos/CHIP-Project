# PatientPheX 实验记录

## 固定配置

- 训练集：`PatientPheX-train.jsonl`，80 篇文献。
- A 榜：`PatientPheX-A.jsonl`，20 篇文献。
- 本体：`hp.obo`。
- 本地划分：随机种子 42，64 篇训练、16 篇验证。
- 基础模型：`Qwen/Qwen3-8B`。
- 训练：单卡 QLoRA，4-bit NF4，LoRA rank 16，alpha 32，dropout 0.05。

## 当前最优方案：E41

严格 HPO 候选供 E18 adapter 做 Qwen 患者关联；关联生成后，用复数/英美拼写归一化和高置信缩写实体补回同句、距离不超过 125 字符的候选。最终任务一实体使用归一化结果。

固定验证集结果：

```json
{
  "f1_men": 0.7414075286415712,
  "f1_doc": 0.781042654028436,
  "f1_micro": 0.6564885496183206,
  "f1_macro": 0.5621905619572395,
  "score": 0.6852823235613918
}
```

相对此前 E32 的 `0.6636955086091131`，score 提升 `0.0215868159522787`。

## E32 历史方案

```text
HPO 规则匹配和训练集表面别名
→ 括号/斜杠缩写补充与英美拼写归一化
→ 同 passage 候选
→ Qwen3-8B QLoRA 患者归属
```

缩写补充仅使用训练集中同一 HPO ID 的全称-缩写对，或全称首字母与 HPO ID 一致的高置信模式。任务 2 的关联仍由 Qwen 产生；最终输出用增强后的任务 1 实体替换关联结果中的实体字段。

## E18：本地验证任务 2

使用 `splits/train.jsonl` 构造同 passage SFT 数据，得到 428 条样本，其中 384 条包含正标签。E18 使用 5 epoch、batch size 1、梯度累积 16、学习率 `2e-4`，输出 adapter：
`experiments/models/E18-qwen3-8b-local`。

E18 原始验证输出为 `experiments/predictions/E18-local-valid.jsonl`，其完整评测结果为：

```json
{
  "f1_men": 0.7194304857621441,
  "f1_doc": 0.7680461982675649,
  "f1_micro": 0.6426484907497565,
  "f1_macro": 0.519389302936832,
  "score": 0.6623786194290744
}
```

## E32：历史验证结果

在 E18 关联结果上重新生成任务 1 实体，启用训练集缩写映射和英美拼写归一化。输出为 `experiments/predictions/E32-final-valid.jsonl`：

```json
{
  "f1_men": 0.724698042482299,
  "f1_doc": 0.7680461982675649,
  "f1_micro": 0.6426484907497565,
  "f1_macro": 0.519389302936832,
  "score": 0.6636955086091131
}
```

相对 E18，验证集新增 13 个缩写实体，其中 11 个与标注完全一致；score 提升约 `0.00132`。

## E21/E32：全量 A 榜输出

使用完整训练集构造 631 条 SFT 样本，其中 554 条包含正标签，训练完整 adapter：
`experiments/models/E21-qwen3-8b-local-full`。

E21 任务 2 输出：`experiments/predictions/E21-qwen-A-filtered.jsonl`。
E32 最终 A 榜输出：`experiments/predictions/E32-final-A.jsonl`。

最终输出包含 20 篇文献、1176 个实体和 334 个 phenotype 值，并已通过：

```bash
python -m patientphex validate \
  --input PatientPheX-A.jsonl \
  --pred experiments/predictions/E32-final-A.jsonl
```

## 候选增强端到端实验：未入选

将缩写增强实体用于任务 2 候选后，验证集候选实体数由 1080 增加到 1093。

- E33：复用 E18 adapter，仅替换为增强候选。score 为 `0.663900852787868`；任务二 micro F1 为 `0.642578125`，略低于 E18 的 `0.6426484907497565`，macro F1 从 `0.519389302936832` 升至 `0.5202810454016082`。
- E34：使用增强候选重新构造 SFT 数据并训练 adapter。score 为 `0.661498011626328`；任务二 micro F1 降至 `0.6308724832214764`，未超过 E18。

结论：增强实体适合在 Qwen 关联生成后作为任务一后处理；直接改变任务二候选并重新训练会破坏候选分布。该结论后来由 E40/E41 的严格候选+后处理方案延续。

## E35：同句近距离关联后处理

在 E18 关联结果上补回同句且距离不超过 100 字符的候选。合并任务一实体后 score 为 `0.6760483350759804`；新增关联 12 个，其中 9 个正确、3 个错误，且没有删除原有关联。

## E36：扩展否定作用域，未采用

尝试识别 `no A or B`、`not have A` 等更宽的否定结构，验证集 score 降至 `0.6584770695672053`，训练集表现也下降，已回退。

## E38：复数词形归一化

加入常见 `-s/-ies` 归一化并保留原文 span。使用 E18 关联、只替换任务一实体时，`f1_men=0.7414075286415712`、`f1_doc=0.781042654028436`，score 为 `0.671121994089149`。

同时将严格词典与归一化词典拆开，默认任务二 SFT 文件与原 E18 文件逐字一致。

## E40/E41：当前最优组合

E40 使用 E18 adapter、严格候选和同句近距离阈值 125 生成实际 Qwen 结果；E41 再将 Qwen 关联与归一化任务一实体合并，得到当前最优 score `0.6852823235613918`。

## E42/E43、E44/E45：候选块大小消融

- `max_candidates=4`：完整 score `0.6791947249658623`，低于 8。
- `max_candidates=16`：完整 score `0.6504007385906074`，明显低于 8。

因此最终固定 `max_candidates=8`。

## A 榜输出

使用完整训练集训练的 E21 adapter，按 E40/E41 流程生成：

- Qwen 中间结果：`experiments/predictions/E46-full-qwen-enhanced-post-A.jsonl`。
- 最终提交：`experiments/predictions/E47-final-A.jsonl`。
- 20 篇文献、1226 个实体、344 个 phenotype 值。
- 已通过 `patientphex validate`；A 榜没有本地金标准，无法计算 score。
