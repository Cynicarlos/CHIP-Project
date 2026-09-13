# PatientPheX 比赛实验报告

> 当前状态：暂定最终方案
>
> 本报告记录当前仓库中保留的代码、模型、数据、实验过程和可复现步骤。验证集结果用于本地方案选择，A 榜数据没有公开金标准，因此 A 榜只做格式和完整性校验。

## 1. 项目概述

PatientPheX 是一个面向医学文献的表型信息抽取任务。输入是一篇包含多个段落和患者 mention 的医学文献，输出包括两类结果：

1. 从文献中识别表型实体，并链接到 HPO（Human Phenotype Ontology）编号。
2. 判断每个表型是否属于指定患者。

当前方案采用规则和语言模型结合的两阶段流程：

```text
HPO 本体匹配 + 训练集表面别名
        ↓
复数/英美拼写归一化、缩写增强
        ↓
严格候选构造（同 passage）
        ↓
Qwen3-8B QLoRA 患者-表型关联预测
        ↓
同句近距离高置信关联补回
        ↓
生成最终 submission
```

当前固定验证集最优结果：

| 指标 | 数值 |
|---|---:|
| `f1_men` | 0.7414075286415712 |
| `f1_doc` | 0.781042654028436 |
| `f1_micro` | 0.6564885496183206 |
| `f1_macro` | 0.5621905619572395 |
| `score` | **0.6852823235613918** |

相较此前方案的 `0.6636955086091131`，总体 score 提升 `0.0215868159522787`。

## 2. 任务定义

### 2.1 输入数据

每篇文献使用一行 JSONL 表示，主要字段如下：

- `pmc_id`：文献唯一 ID。
- `pmid`：PubMed ID，可能为空。
- `patient`：目标患者列表。每个患者包含 `patient_id` 和若干 `mention`。
- `full_text`：文章段落列表，包含段落类型、章节、全局 offset 和文本。
- `entities`：训练数据中的表型实体答案；测试数据为空。
- `association`：训练数据中的患者-表型关联答案；测试数据为空。

offset 是整篇文章级别的全局字符位置，不是段落内相对位置。

### 2.2 任务一：表型实体识别和 HPO 链接

任务一输出 `entities` 列表，每个实体包含：

```json
{
  "identifier": "HP:0001250",
  "type": "Phenotype",
  "offset": 1234,
  "length": 8,
  "text": "epilepsy",
  "note": null
}
```

其中：

- `identifier` 是 HPO ID；多个 ID 使用分号连接。
- `offset`、`length`、`text` 必须和原文严格对应。
- 否定实体可以使用 `note: "NO"`，当前最终方案保留了原有的保守否定规则。
- 无法链接到 HPO 的实体使用 `identifier: "-1"`，关联任务中使用其原文表型文本。

### 2.3 任务二：患者-表型关联

任务二输出每个患者的 phenotype 列表：

```json
{
  "patient_id": "P1",
  "phenotype": ["HP:0001250", "HP:0000708"]
}
```

任务二不是简单的全局分类，而是从当前文献候选实体中选择属于目标患者的表型。当前实现将候选实体分块提供给 Qwen，模型返回候选块内的正候选编号。

### 2.4 两个子任务的关系

两个子任务在评分上分别计算，但在推理流程上不是完全独立的：

- 任务一决定任务二能看到哪些候选，因此实体漏检会造成任务二候选缺失。
- 任务二只输出 HPO 值，不重新生成实体 span。
- 当前方案将任务二的模型候选和任务一最终输出解耦：Qwen 使用训练时的严格候选，最终提交再用增强后的任务一实体替换实体字段。

这样既利用了增强实体提高任务一召回率，又避免改变 Qwen 已经适应的候选分布。

## 3. 数据和验证方案

### 3.1 数据文件

| 文件 | 用途 |
|---|---|
| `PatientPheX-train.jsonl` | 比赛训练集，共 80 篇文献，包含标注 |
| `PatientPheX-A.jsonl` | A 榜测试集，共 20 篇文献，不含答案 |
| `PatientPheX-B.jsonl` | B 榜测试集，开放后使用，不含答案 |
| `hp.obo` | 比赛指定 HPO 本体 |
| `submit_pred_ex.jsonl` | 提交格式示例 |

HPO 匹配限定在 `HP:0000118`（Phenotypic abnormality）分支内，并忽略 obsolete term。

### 3.2 固定本地划分

由于比赛训练集只有 80 篇文献，使用整篇文献划分，避免同一篇文献的段落同时出现在训练和验证中：

- 随机种子：`42`
- 本地训练集：64 篇
- 本地验证集：16 篇

划分命令：

```bash
python -m patientphex split \
  --input PatientPheX-train.jsonl \
  --train-output splits/train.jsonl \
  --valid-output splits/valid.jsonl \
  --valid-ratio 0.2 \
  --seed 42
```

固定划分文件已保存在 `splits/train.jsonl` 和 `splits/valid.jsonl`，复现实验时不需要重新划分。

## 4. 项目结构

```text
.
├── PatientPheX-train.jsonl       # 比赛训练集
├── PatientPheX-A.jsonl           # A 榜测试集
├── hp.obo                        # HPO 本体
├── README.md                     # 数据字段说明
├── README_BASELINE.md            # 当前方案的命令行复现说明
├── EXPERIMENT_REPORT.md          # 本实验报告
├── requirements-lora.txt         # QLoRA 依赖
├── patientphex/
│   ├── __main__.py               # split/evaluate/validate/build-task2/predict
│   ├── core.py                   # HPO 匹配、实体增强、关联基线、评测
│   ├── data.py                   # 任务二候选和 SFT 数据构造
│   ├── qwen_lora.py               # Qwen QLoRA 训练和推理
│   └── split.py                  # 固定数据划分
├── splits/
│   ├── train.jsonl               # 64 篇本地训练文献
│   ├── valid.jsonl               # 16 篇本地验证文献
│   └── task2-local-sft.jsonl     # 本地 Qwen SFT 数据
├── task2-local-sft.jsonl         # 全量训练 SFT 数据
└── experiments/
    ├── EXPERIMENT_LOG.md         # 实验简表
    ├── models/                   # E18/E21 LoRA adapter
    └── predictions/              # 当前方案的中间结果和最终结果
```

当前保留的预测文件：

- `experiments/predictions/E40-strict-qwen-enhanced-post-valid.jsonl`：验证集 Qwen 中间结果。
- `experiments/predictions/E41-plural-qwen-post-valid.jsonl`：验证集当前最优完整结果。
- `experiments/predictions/E46-full-qwen-enhanced-post-A.jsonl`：A 榜 Qwen 中间结果。
- `experiments/predictions/E47-final-A.jsonl`：A 榜最终提交结果。

失败消融和旧版本预测文件已删除，避免误用。

## 5. 方法实现

### 5.1 HPO 词典构造

`patientphex/core.py` 中的 `HpoLexicon` 负责解析 OBO 文件并构造匹配词典：

1. 解析 HPO term 的 `name` 和 `synonym`。
2. 保留 `HP:0000118` 分支内的非 obsolete term。
3. 使用 token 序列建立索引。
4. 对重叠匹配采用最长 span 优先。
5. 过滤过短或过于通用的单 token 词，例如 `patient`、`disease`、`normal`。

### 5.2 训练集表面别名

训练数据中存在一些 HPO 本体没有覆盖的表面表达。程序从训练标注中统计：

```text
表面 token 序列 → 训练集中最常见的 HPO ID
```

这些别名只从训练数据学习，不读取验证集或测试集答案。训练别名会覆盖本体中有歧义的映射，以适应比赛标注习惯。

### 5.3 词形和拼写归一化

当前任务一词典支持：

- 英美拼写：`centre/center`、`behaviour/behavior`、`tumour/tumor`、`fibre/fiber`。
- 常见复数：`seizures → seizure`、`epilepsies → epilepsy`、`features → feature`。
- `-ies` 结尾的常见变换：`abnormalities → abnormality`。

归一化只用于匹配键，不改变提交中的原文 span。例如原文中的 `epilepsies` 仍然作为 `text` 输出，只有匹配时按 `epilepsy` 查词典。

为了保持 Qwen 训练和推理的一致性，实现中保留两种词典：

- 严格词典：不使用复数归一化，用于任务二 SFT 和 Qwen 模型候选。
- 归一化词典：使用复数和拼写归一化，用于最终任务一实体和任务二后处理。

默认 `build-task2` 生成的 `splits/task2-local-sft.jsonl` 已验证与原 E18 训练文件逐字一致。

### 5.4 缩写增强

`predict_entities_with_acronyms` 使用两类高精度规则：

1. 从训练标注中学习同一 HPO ID 的“全称 + 括号缩写”关系。
2. 当括号中的大写缩写字母与全称 token 首字母一致时，补充该缩写实体。
3. 对紧邻 HPO 实体的高置信大写斜杠缩写进行补充，例如 `SHFM/ectrodactyly` 中的 `SHFM`。

缩写实体的 offset 和 text 始终来自原文，避免提交格式错误。

### 5.5 任务二 SFT 数据

`patientphex/data.py` 构造患者-候选对话样本：

- 候选限定为同一 passage 中出现的表型实体。
- 一个样本最多包含 8 个候选。
- 训练样本的候选池为“规则预测实体 + gold 实体”的并集，使训练阶段的正例不会因规则漏检而完全消失。
- Qwen 推理时只使用规则预测实体，不加入 gold 实体。
- 标签格式为：

```json
{"positive_candidates":[0,3]}
```

当前本地训练集生成 428 条样本，其中 384 条包含正标签；全量训练集生成 631 条样本，其中 554 条包含正标签。

### 5.6 Qwen QLoRA 训练

基础模型为 `Qwen/Qwen3-8B`，采用单卡 QLoRA：

| 配置 | 数值 |
|---|---:|
| 量化 | 4-bit NF4 |
| double quantization | 开启 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| target modules | q/k/v/o projection，gate/up/down projection |
| max length | 2048 |
| epoch | 5 |
| batch size | 1 |
| gradient accumulation | 16 |
| learning rate | `2e-4` |

训练只对 assistant 输出计算 loss，用户提示和系统提示部分使用 `-100` mask。

### 5.7 Qwen 推理后的高置信补回

Qwen 先在严格候选中生成关联结果，然后启用 `--add-same-sentence`：

- 候选实体和目标患者 mention 必须位于同一 passage。
- 二者必须位于同一句。
- 全局 offset 距离不超过 125 个字符。
- 只做集合并集，不删除 Qwen 已经选出的值。

该规则在验证集上主要补回 Qwen 漏选但局部证据很强的候选，同时避免把归一化实体全部加入患者关联。

## 6. 实验过程和结果

所有分数均使用仓库内的 `patientphex evaluate` 计算。最终 score 是四项 F1 的平均值：

```text
score = (f1_men + f1_doc + f1_micro + f1_macro) / 4
```

### 6.1 主要实验对比

| 实验 | 主要变化 | `f1_men` | `f1_doc` | `f1_micro` | `f1_macro` | score | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| E18 | 严格候选 + Qwen 关联 | 0.719430 | 0.768046 | 0.642648 | 0.519389 | 0.662379 | 初始 Qwen 基线 |
| E32 | E18 关联 + 缩写/拼写实体增强 | 0.724698 | 0.768046 | 0.642648 | 0.519389 | 0.663696 | 早期最佳 |
| E33 | 增强实体直接作为 Qwen 候选 | — | — | 0.642578 | 0.520281 | 0.663901 | 候选分布变化，未采用 |
| E34 | 增强候选重新训练 Qwen | — | — | 0.630872 | 0.522375 | 0.661498 | 训练数据过少，未采用 |
| E35 | Qwen 结果补回同句近距离候选 | 0.724698 | 0.768046 | 0.652551 | 0.558899 | 0.676048 | 有效 |
| E36 | 扩展宽松否定作用域 | 0.716228 | 0.755643 | 0.642648 | 0.519389 | 0.658477 | 误伤正例，回退 |
| E38 | 任务一加入复数词形归一化 | 0.741408 | 0.781043 | 0.642648 | 0.519389 | 0.671122 | 有效 |
| E40 | 严格 Qwen 候选 + 归一化后处理 | 0.719430 | 0.768046 | 0.656489 | 0.562191 | 0.676539 | 中间结果 |
| E41 | E40 关联 + E38 任务一实体 | 0.741408 | 0.781043 | 0.656489 | 0.562191 | **0.685282** | 当前最优 |
| E42/E43 | `max_candidates=4` | 0.741408 | 0.781043 | 0.646445 | 0.547883 | 0.679195 | 低于 8 |
| E44/E45 | `max_candidates=16` | 0.741408 | 0.781043 | 0.538895 | 0.540258 | 0.650401 | 明显退化 |

### 6.2 关键结论

#### 任务一：精确词典仍然是最有效的实体基础

任务一的规则方法效果较好，原因是：

- HPO 本体提供了大量医学术语、同义词和层级信息。
- 训练集表面别名可以校正比赛数据中的标注习惯。
- 任务一要求精确 span 和 HPO ID，通用生成模型容易产生边界或链接错误。

复数归一化是本轮最有效的任务一改进。它只修改查词键，不做宽松模糊匹配，因此风险较低。

#### 任务二：候选覆盖和模型选择都重要

固定验证集的 Qwen 关联漏项中，一部分是正确候选已经存在但模型没有选择，另一部分是任务一没有提供候选。直接把增强实体加入 Qwen 输入会改变块划分和候选位置，反而破坏已经学到的候选分布。

最终采用“严格候选供 Qwen 选择、增强实体只做后处理”的折中方式，取得了更稳定的收益。

#### 同句近距离规则优于单纯距离规则

只按距离、同 passage 全量关联或最近患者 mention 都明显弱于 Qwen。有效规则必须同时满足同 passage、同句和较小距离，并且只补回不删除结果。

## 7. 当前最终方案和文件

### 7.1 本地验证方案

- 本地训练 adapter：`experiments/models/E18-qwen3-8b-local`
- Qwen 中间结果：`experiments/predictions/E40-strict-qwen-enhanced-post-valid.jsonl`
- 最终验证结果：`experiments/predictions/E41-plural-qwen-post-valid.jsonl`

E41 已通过：

```bash
python -m patientphex validate \
  --input splits/valid.jsonl \
  --pred experiments/predictions/E41-plural-qwen-post-valid.jsonl
```

### 7.2 A 榜方案

- 全量训练 adapter：`experiments/models/E21-qwen3-8b-local-full`
- Qwen 中间结果：`experiments/predictions/E46-full-qwen-enhanced-post-A.jsonl`
- 最终提交：`experiments/predictions/E47-final-A.jsonl`

A 榜最终文件统计：

- 文献数：20
- 实体数：1226
- phenotype 关联值数：344

已通过：

```bash
python -m patientphex validate \
  --input PatientPheX-A.jsonl \
  --pred experiments/predictions/E47-final-A.jsonl
```

## 8. 完整复现步骤

### 8.1 环境和模型缓存

安装依赖：

```bash
python -m pip install -r requirements-lora.txt
python -m py_compile patientphex/*.py
```

模型下载使用 Hugging Face 镜像，训练和推理使用本地缓存：

```bash
export HF_HOME=/data3/chenxianmin/model_cache/huggingface
export TRANSFORMERS_CACHE=/data3/chenxianmin/model_cache/huggingface/hub
export HF_ENDPOINT=https://hf-mirror.net
export HF_HUB_OFFLINE=1
```

如果 `Qwen/Qwen3-8B` 尚未缓存，执行：

```bash
HF_HUB_OFFLINE=0 HF_ENDPOINT=https://hf-mirror.net \
HF_HOME=/data3/chenxianmin/model_cache/huggingface \
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-8B')"
```

### 8.2 生成本地划分

已有固定划分时跳过此步；如需重新生成：

```bash
python -m patientphex split \
  --input PatientPheX-train.jsonl \
  --train-output splits/train.jsonl \
  --valid-output splits/valid.jsonl \
  --valid-ratio 0.2 \
  --seed 42
```

### 8.3 构造本地 SFT 数据

```bash
python -m patientphex build-task2 \
  --input splits/train.jsonl \
  --output splits/task2-local-sft.jsonl \
  --hpo hp.obo \
  --max-candidates 8 \
  --same-passage-only
```

### 8.4 训练本地 E18 adapter

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
HF_HOME=/data3/chenxianmin/model_cache/huggingface \
TRANSFORMERS_CACHE=/data3/chenxianmin/model_cache/huggingface/hub \
python -m patientphex.qwen_lora train \
  --data splits/task2-local-sft.jsonl \
  --model Qwen/Qwen3-8B \
  --output experiments/models/E18-qwen3-8b-local \
  --qlora \
  --max-length 2048 \
  --epochs 5 \
  --batch-size 1 \
  --grad-accum 16 \
  --learning-rate 2e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05
```

### 8.5 生成验证集 Qwen 关联

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
HF_HOME=/data3/chenxianmin/model_cache/huggingface \
TRANSFORMERS_CACHE=/data3/chenxianmin/model_cache/huggingface/hub \
python -m patientphex.qwen_lora predict \
  --input splits/valid.jsonl \
  --output experiments/predictions/E40-strict-qwen-enhanced-post-valid.jsonl \
  --hpo hp.obo \
  --model Qwen/Qwen3-8B \
  --adapter experiments/models/E18-qwen3-8b-local \
  --train-alias splits/train.jsonl \
  --qlora \
  --max-candidates 8 \
  --max-length 2048 \
  --same-passage-only \
  --filter-same-passage \
  --add-same-sentence \
  --same-sentence-distance 125
```

### 8.6 生成并评测当前最优验证结果

```bash
python -m patientphex predict \
  --input splits/valid.jsonl \
  --output experiments/predictions/E41-plural-qwen-post-valid.jsonl \
  --hpo hp.obo \
  --train-alias splits/train.jsonl \
  --acronym-alias splits/train.jsonl \
  --association experiments/predictions/E40-strict-qwen-enhanced-post-valid.jsonl
```

```bash
python -m patientphex evaluate \
  --gold splits/valid.jsonl \
  --pred experiments/predictions/E41-plural-qwen-post-valid.jsonl
```

### 8.7 构造全量训练 SFT 数据

```bash
python -m patientphex build-task2 \
  --input PatientPheX-train.jsonl \
  --output task2-local-sft.jsonl \
  --hpo hp.obo \
  --max-candidates 8 \
  --same-passage-only
```

### 8.8 训练全量 E21 adapter

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
HF_HOME=/data3/chenxianmin/model_cache/huggingface \
TRANSFORMERS_CACHE=/data3/chenxianmin/model_cache/huggingface/hub \
python -m patientphex.qwen_lora train \
  --data task2-local-sft.jsonl \
  --model Qwen/Qwen3-8B \
  --output experiments/models/E21-qwen3-8b-local-full \
  --qlora \
  --max-length 2048 \
  --epochs 5 \
  --batch-size 1 \
  --grad-accum 16 \
  --learning-rate 2e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05
```

### 8.9 生成 A 榜提交

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
HF_HOME=/data3/chenxianmin/model_cache/huggingface \
TRANSFORMERS_CACHE=/data3/chenxianmin/model_cache/huggingface/hub \
python -m patientphex.qwen_lora predict \
  --input PatientPheX-A.jsonl \
  --output experiments/predictions/E46-full-qwen-enhanced-post-A.jsonl \
  --hpo hp.obo \
  --model Qwen/Qwen3-8B \
  --adapter experiments/models/E21-qwen3-8b-local-full \
  --train-alias PatientPheX-train.jsonl \
  --qlora \
  --max-candidates 8 \
  --max-length 2048 \
  --same-passage-only \
  --filter-same-passage \
  --add-same-sentence \
  --same-sentence-distance 125
```

```bash
python -m patientphex predict \
  --input PatientPheX-A.jsonl \
  --output experiments/predictions/E47-final-A.jsonl \
  --hpo hp.obo \
  --train-alias PatientPheX-train.jsonl \
  --acronym-alias PatientPheX-train.jsonl \
  --association experiments/predictions/E46-full-qwen-enhanced-post-A.jsonl
```

```bash
python -m patientphex validate \
  --input PatientPheX-A.jsonl \
  --pred experiments/predictions/E47-final-A.jsonl
```

### 8.10 B 榜复用方式

B 榜文件开放后，将 A 榜输入替换为 `PatientPheX-B.jsonl`，使用 E21 adapter 和相同参数生成：

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
HF_HOME=/data3/chenxianmin/model_cache/huggingface \
TRANSFORMERS_CACHE=/data3/chenxianmin/model_cache/huggingface/hub \
python -m patientphex.qwen_lora predict \
  --input PatientPheX-B.jsonl \
  --output experiments/predictions/E21-qwen-B-enhanced-post.jsonl \
  --hpo hp.obo \
  --model Qwen/Qwen3-8B \
  --adapter experiments/models/E21-qwen3-8b-local-full \
  --train-alias PatientPheX-train.jsonl \
  --qlora \
  --max-candidates 8 \
  --max-length 2048 \
  --same-passage-only \
  --filter-same-passage \
  --add-same-sentence \
  --same-sentence-distance 125
```

```bash
python -m patientphex predict \
  --input PatientPheX-B.jsonl \
  --output experiments/predictions/E47-final-B.jsonl \
  --hpo hp.obo \
  --train-alias PatientPheX-train.jsonl \
  --acronym-alias PatientPheX-train.jsonl \
  --association experiments/predictions/E21-qwen-B-enhanced-post.jsonl
```

## 9. 评测、校验和有效性检查

评测验证集：

```bash
python -m patientphex evaluate \
  --gold splits/valid.jsonl \
  --pred experiments/predictions/E41-plural-qwen-post-valid.jsonl
```

校验预测文件的文献、患者、实体 span 和字段格式：

```bash
python -m patientphex validate \
  --input splits/valid.jsonl \
  --pred experiments/predictions/E41-plural-qwen-post-valid.jsonl

python -m patientphex validate \
  --input PatientPheX-A.jsonl \
  --pred experiments/predictions/E47-final-A.jsonl
```

代码语法检查：

```bash
python -m py_compile patientphex/*.py
```

## 10. 局限性和风险

1. 本地验证集只有 16 篇文献，阈值和后处理规则仍可能对该划分存在过拟合。
2. HPO 规则匹配对未出现在本体或训练别名中的复杂改写、长距离共指和隐含表达召回有限。
3. HPO 版本中的 obsolete term 被忽略，但比赛标注可能存在旧 ID 或标注习惯差异。
4. Qwen 任务二是候选约束生成，候选池之外的正确关系无法由模型恢复。
5. 同句近距离后处理是高精度启发式，适合当前数据分布，不能保证适用于所有医学文献。
6. A 榜没有公开 gold，因此 A 榜结果只能证明格式正确，不能证明真实榜单收益。

## 11. 后续可选方向

当前版本暂不继续扩展，但后续可以考虑：

- 使用多折文献级交叉验证选择距离阈值，降低对单一验证划分的依赖。
- 训练独立的患者-实体二分类器，与 Qwen 结果进行概率或规则融合。
- 使用医学 NER/linker 生成候选，再保留 Qwen 做患者归属。
- 引入跨句患者共指和章节结构特征。
- 对 HPO 歧义短语增加基于训练频率和上下文的链接重排。

这些方向尚未纳入当前提交方案，不能与本报告中的 E41/E47 结果混用。

## 12. 最终结论

在当前数据规模和资源条件下，最稳定的方案不是让 Qwen 同时承担实体识别、HPO 链接和患者归属，而是：

1. 用 HPO 词典和训练集别名完成高精度实体识别。
2. 用轻量词形、拼写和缩写规则提升任务一召回。
3. 保持 Qwen 训练时的严格候选分布。
4. 用 Qwen 完成主要患者关联判断。
5. 仅对同句近距离的增强候选做保守补回。

该方案对应的验证集最优结果为 `0.6852823235613918`，A 榜最终可提交文件为 `experiments/predictions/E47-final-A.jsonl`。
