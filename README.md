# PatientPheX 数据说明

当前方案的完整实验报告见：[EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)。

## 文件

- `PatientPheX-train.jsonl`：训练集，共 80 篇文献，包含完整答案。
- `PatientPheX-A.jsonl`：A榜测试集，共 20 篇文献；`entities` 和 `association` 为空，不包含答案。
- `PatientPheX-B.jsonl`：B榜测试集（B榜时释放），共 100 篇文献；`entities` 和 `association` 为空，不包含答案。
- `submit_pred_ex.jsonl`：提交格式示例。示例取自训练集，不包含开发集或测试集答案。
- `hp.obo`：HPO本体文件。评测指定使用2026-06-23释放的HPO版本下Phenotypic abnormality （HP:0000118）分支。本体信息可参考官网https://hpo.jax.org/

所有文件均使用 UTF-8 编码。JSONL 文件每行是一篇文献，文章按照原始 PMC JSON 文件名的字典序排列。

## 文献字段

字段顺序固定如下：

1. `pmc_id`：PubMed Central 文献 ID，也是文献主键。
2. `pmid`：PubMed ID，可能为 `null`。
3. `patient`：目标患者列表。每项包含 `patient_id` 和按原数据顺序保存的 `mention`。
4. `full_text`：文章段落列表，每段包含 `section_type`、`type`、`offset` 和 `text`。
5. `entities`：表型实体答案；开发集和测试集中为空列表。
6. `association`：患者与表型的关联答案；开发集和测试集中为空列表。

offset 均为原始完整文章中的全局字符位置。

## 表型实体

`entities` 中每项包含：

- `identifier`：HPO ID；多个 ID 可能以分号分隔。
- `type`：实体类型，当前为 `Phenotype`。
- `offset`、`length`、`text`：实体位置和原文。
- `note`：附加标记，默认为 `null`，否定表型实体时应该为`NO`。

## 患者表型关联

`association` 中每项包含：

- `patient_id`：与 `patient[].patient_id` 对应。
- `phenotype`：按原数据顺序保存的字符串列表。正常项为 HPO ID；原始 HPO ID 为 `-1` 时，保存原始表型文本。

## 提交格式

提交文件采用 UTF-8 JSONL 格式，每行一篇文献：

```json
{"pmc_id":"文献ID","pmid":"PubMed ID或null","entities":[...],"association":[{"patient_id":"P1","phenotype":["HP:0000001","无法映射的原文表型"]}]}
```

- `pmc_id` 必须与待预测数据一致且不得重复。
- `entities` 保存表型实体预测。
- `association` 保存患者表型关联预测。
- 实际提交应覆盖对应开发集或测试集的全部文献。
- `submit_pred_ex.jsonl` 只用于展示格式。
