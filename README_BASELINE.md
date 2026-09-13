# PatientPheX 当前最优方案

当前方案由两部分组成：

```text
HPO 规则匹配 + 训练集表面别名
→ 高精度缩写补充与英美拼写归一化
→ 同 passage 候选构造
→ Qwen3-8B QLoRA 患者归属
```

任务 1 的实体和 HPO ID 由规则及缩写模块生成；任务 2 使用 Qwen 预测患者归属。任务 2 的 Qwen 候选保持训练时的严格匹配分布，推理后再用归一化实体做同句近距离高置信补回。当前固定验证集 score 为 `0.6852823235613918`，最终 A 榜文件为 `experiments/predictions/E47-final-A.jsonl`。

## 1. 文件和环境

主要代码和数据如下：

- `patientphex/core.py`：实体识别、缩写增强、患者归属、评测和校验。
- `patientphex/data.py`：任务 2 候选和 SFT 数据构造。
- `patientphex/qwen_lora.py`：Qwen3-8B QLoRA 训练和推理。
- `patientphex/split.py`：固定本地验证集划分。
- `PatientPheX-train.jsonl`、`PatientPheX-A.jsonl`、`hp.obo`：比赛数据和本体。
- `splits/`：固定划分及本地 SFT 数据。
- `experiments/models/`：E18 本地 adapter 和 E21 全量 adapter。
- `experiments/predictions/`：当前最优验证集和 A 榜输出，分别为 `E41-plural-qwen-post-valid.jsonl` 和 `E47-final-A.jsonl`。

安装依赖并检查代码：

```bash
python -m pip install -r requirements-lora.txt
python -m py_compile patientphex/*.py
```

模型下载使用镜像，训练和推理使用本地缓存：

```bash
export HF_HOME=/data3/chenxianmin/model_cache/huggingface
export TRANSFORMERS_CACHE=/data3/chenxianmin/model_cache/huggingface/hub
export HF_ENDPOINT=https://hf-mirror.net
export HF_HUB_OFFLINE=1
```

如果 Qwen3-8B 尚未缓存，先执行：

```bash
HF_HUB_OFFLINE=0 HF_ENDPOINT=https://hf-mirror.net \
HF_HOME=/data3/chenxianmin/model_cache/huggingface \
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-8B')"
```

## 2. 固定本地验证集

按整篇文献划分，避免同一篇文献同时出现在训练和验证中：

```bash
python -m patientphex split \
  --input PatientPheX-train.jsonl \
  --train-output splits/train.jsonl \
  --valid-output splits/valid.jsonl \
  --valid-ratio 0.2 \
  --seed 42
```

结果应为 64 篇训练文献和 16 篇验证文献。若已有 `splits/train.jsonl` 和 `splits/valid.jsonl`，无需重复划分。

## 3. 构造任务 2 训练数据

本地验证和最终训练都使用同 passage 候选：

```bash
python -m patientphex build-task2 \
  --input splits/train.jsonl \
  --output splits/task2-local-sft.jsonl \
  --hpo hp.obo \
  --max-candidates 8 \
  --same-passage-only
```

最终 A 榜训练数据：

```bash
python -m patientphex build-task2 \
  --input PatientPheX-train.jsonl \
  --output task2-local-sft.jsonl \
  --hpo hp.obo \
  --max-candidates 8 \
  --same-passage-only
```

默认候选使用严格 HPO token 匹配，以保持 E18/E21 adapter 的训练分布；复数和英美拼写归一化只在最终任务 1 输出及任务 2 的高置信后处理中启用。

## 4. 训练和评估本地 Qwen adapter

训练 E18：

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

生成验证集任务 2 结果。Qwen 使用严格候选，随后补回归一化实体中与患者 mention 同句且距离不超过 125 个字符的候选：

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

将当前最优任务 1 实体替换到 Qwen 的关联结果中，并评测完整方案：

```bash
python -m patientphex predict \
  --input splits/valid.jsonl \
  --output experiments/predictions/E41-plural-qwen-post-valid.jsonl \
  --hpo hp.obo \
  --train-alias splits/train.jsonl \
  --acronym-alias splits/train.jsonl \
  --association experiments/predictions/E40-strict-qwen-enhanced-post-valid.jsonl

python -m patientphex evaluate \
  --gold splits/valid.jsonl \
  --pred experiments/predictions/E41-plural-qwen-post-valid.jsonl
```

参考结果：

```json
{
  "f1_men": 0.7414075286415712,
  "f1_doc": 0.781042654028436,
  "f1_micro": 0.6564885496183206,
  "f1_macro": 0.5621905619572395,
  "score": 0.6852823235613918
}
```

## 5. 全量训练并生成 A 榜文件

训练 E21：

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

先生成全量训练配置下的任务 2 预测，并启用同句近距离高置信补回：

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

最后用全量训练集学习表面别名和缩写，并复用 E21 的患者关联：

```bash
python -m patientphex predict \
  --input PatientPheX-A.jsonl \
  --output experiments/predictions/E47-final-A.jsonl \
  --hpo hp.obo \
  --train-alias PatientPheX-train.jsonl \
  --acronym-alias PatientPheX-train.jsonl \
  --association experiments/predictions/E46-full-qwen-enhanced-post-A.jsonl

python -m patientphex validate \
  --input PatientPheX-A.jsonl \
  --pred experiments/predictions/E47-final-A.jsonl
```

当前 A 榜输出为 20 篇文献、1226 个实体和 344 个 phenotype 值。A 榜没有本地金标准，不能计算 score。

## 6. B 榜

B 榜文件开放后，替换输入文件即可。先用 E21 adapter 生成任务 2 关联，再生成最终提交：

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

python -m patientphex predict \
  --input PatientPheX-B.jsonl \
  --output experiments/predictions/E47-final-B.jsonl \
  --hpo hp.obo \
  --train-alias PatientPheX-train.jsonl \
  --acronym-alias PatientPheX-train.jsonl \
  --association experiments/predictions/E21-qwen-B-enhanced-post.jsonl

python -m patientphex validate \
  --input PatientPheX-B.jsonl \
  --pred experiments/predictions/E47-final-B.jsonl
```

`--acronym-alias` 只从训练标注学习全称和缩写的对应关系；`--association` 只复用 Qwen 的患者关联，不会覆盖新生成的任务 1 实体。
