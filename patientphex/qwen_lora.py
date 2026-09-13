"""Optional Qwen LoRA training and candidate-constrained inference.

Dependencies are imported only when the corresponding command is executed.
This keeps the rule baseline usable without peft/bitsandbytes.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .core import (
    HpoLexicon,
    _passage_for_offset,
    _sentence_index,
    learn_acronym_aliases,
    predict_entities,
    predict_entities_with_acronyms,
    read_jsonl,
    validate_prediction,
    write_jsonl,
)
from .data import _candidate_record, make_task2_messages, same_passage_candidates


def _render(tokenizer, messages: list[dict], add_generation_prompt: bool) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def _load_training_rows(path: str | Path) -> list[dict]:
    return read_jsonl(path)


def _train(args: argparse.Namespace) -> None:
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit("训练需要安装 peft、transformers 和 torch；QLoRA 还需要 bitsandbytes。") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True, "device_map": "auto"}
    if args.qlora:
        if not torch.cuda.is_available():
            raise SystemExit("QLoRA 需要 CUDA GPU；当前仅准备训练代码，未启动训练。")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.qlora:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        ),
    )
    model.print_trainable_parameters()

    class ChatDataset(torch.utils.data.Dataset):
        def __init__(self, rows: list[dict]):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index: int):
            messages = self.rows[index]["messages"]
            prompt = _render(tokenizer, messages[:-1], True)
            full = _render(tokenizer, messages, False)
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(
                full,
                add_special_tokens=False,
            )["input_ids"]
            if len(full_ids) > args.max_length or len(prompt_ids) >= len(full_ids):
                raise ValueError(
                    f"sample {index} needs {len(full_ids)} tokens, but --max-length={args.max_length}; "
                    "increase --max-length so the assistant answer is not truncated"
                )
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
            return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}

    def collate(features: list[dict]) -> dict:
        max_len = max(len(item["input_ids"]) for item in features)
        pad_id = tokenizer.pad_token_id
        return {
            "input_ids": torch.tensor([item["input_ids"] + [pad_id] * (max_len - len(item["input_ids"])) for item in features]),
            "attention_mask": torch.tensor([item["attention_mask"] + [0] * (max_len - len(item["attention_mask"])) for item in features]),
            "labels": torch.tensor([item["labels"] + [-100] * (max_len - len(item["labels"])) for item in features]),
        }

    rows = _load_training_rows(args.data)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
            gradient_checkpointing=True,
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        ),
        train_dataset=ChatDataset(rows),
        data_collator=collate,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


def _parse_answer(text: str) -> list[int]:
    """Parse only positive candidate IDs; malformed output becomes empty."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return []
    try:
        answer = json.loads(match.group(0))
        return [int(x) for x in answer.get("positive_candidates", []) if str(x).isdigit()]
    except (TypeError, ValueError, json.JSONDecodeError):
        return []


def _load_inference_model(args):
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise SystemExit("Qwen 推理需要安装 peft、transformers 和 torch；QLoRA 还需要 bitsandbytes。") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    kwargs = {"trust_remote_code": True, "device_map": "auto"}
    if args.qlora:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    base = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    return tokenizer, PeftModel.from_pretrained(base, args.adapter), torch


def _nearby_same_sentence_values(
    doc: dict,
    candidates: list[dict],
    patient_mentions: list[dict],
    max_distance: int,
) -> list[str]:
    """Return high-confidence candidates close to a patient mention in its sentence."""
    values = []
    for entity in candidates:
        if entity.get("note") == "NO":
            continue
        entity_passage = _passage_for_offset(doc, entity["offset"])
        if not entity_passage:
            continue
        entity_sentence = _sentence_index(
            entity_passage["text"], entity["offset"] - entity_passage["offset"]
        )
        for mention in patient_mentions:
            mention_passage = _passage_for_offset(doc, mention["offset"])
            if mention_passage is not entity_passage:
                continue
            mention_sentence = _sentence_index(
                mention_passage["text"], mention["offset"] - mention_passage["offset"]
            )
            if entity_sentence != mention_sentence:
                continue
            if abs(entity["offset"] - mention["offset"]) > max_distance:
                continue
            value = entity["text"] if entity["identifier"] == "-1" else entity["identifier"]
            if value not in values:
                values.append(value)
            break
    return values


def _qwen_predict(args: argparse.Namespace) -> None:
    tokenizer, model, torch = _load_inference_model(args)
    docs = read_jsonl(args.input)
    aliases = None
    strict_aliases = None
    acronym_aliases = {}
    if args.train_alias:
        from .core import learn_surface_aliases

        alias_docs = read_jsonl(args.train_alias)
        aliases = learn_surface_aliases(alias_docs)
        strict_aliases = learn_surface_aliases(alias_docs, normalize=False)
        acronym_aliases = learn_acronym_aliases(alias_docs)
    lexicon = HpoLexicon(args.hpo, aliases)
    strict_lexicon = HpoLexicon(args.hpo, strict_aliases, normalize_tokens=False)
    predictions = []

    for doc in docs:
        entities = (
            predict_entities_with_acronyms(doc, lexicon, acronym_aliases)
            if args.acronym_alias
            else predict_entities(doc, strict_lexicon or lexicon)
        )
        postprocess_entities = entities
        if args.add_same_sentence and not args.acronym_alias:
            postprocess_entities = predict_entities_with_acronyms(doc, lexicon, acronym_aliases)
        raw_candidates = [entity for entity in entities if entity.get("note") != "NO"]
        associations = []
        for patient in doc.get("patient", []):
            candidates = raw_candidates
            if args.same_passage_only:
                candidates = same_passage_candidates(doc, candidates, patient.get("mention", []))
            values = []
            mentions = [mention["text"] for mention in patient.get("mention", [])]
            for start in range(0, len(candidates), args.max_candidates):
                chunk = candidates[start : start + args.max_candidates]
                serialized = [_candidate_record(doc, entity, i) for i, entity in enumerate(chunk)]
                messages = make_task2_messages(patient["patient_id"], mentions, serialized)[:-1]
                prompt = _render(tokenizer, messages, True)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
                inputs = {key: value.to(model.device) for key, value in inputs.items()}
                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
                generated = output[0][inputs["input_ids"].shape[1] :]
                answer = _parse_answer(tokenizer.decode(generated, skip_special_tokens=True))
                for index in answer:
                    if 0 <= index < len(chunk):
                        entity = chunk[index]
                        if args.filter_same_passage and not same_passage_candidates(
                            doc, [entity], patient.get("mention", [])
                        ):
                            continue
                        value = entity["text"] if entity["identifier"] == "-1" else entity["identifier"]
                        if value not in values:
                            values.append(value)
            if args.add_same_sentence:
                for value in _nearby_same_sentence_values(
                    doc,
                    postprocess_entities,
                    patient.get("mention", []),
                    args.same_sentence_distance,
                ):
                    if value not in values:
                        values.append(value)
            associations.append({"patient_id": patient["patient_id"], "phenotype": values})

        prediction = {"pmc_id": doc["pmc_id"], "pmid": doc.get("pmid"), "entities": entities, "association": associations}
        validate_prediction(doc, prediction)
        predictions.append(prediction)
    write_jsonl(predictions, args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional Qwen LoRA tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--qlora", action="store_true")
    train.add_argument("--max-length", type=int, default=4096)
    train.add_argument("--epochs", type=float, default=3)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--grad-accum", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--lora-r", type=int, default=16)
    train.add_argument("--lora-alpha", type=int, default=32)
    train.add_argument("--lora-dropout", type=float, default=0.05)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--hpo", required=True)
    predict.add_argument("--model", required=True)
    predict.add_argument("--adapter", required=True)
    predict.add_argument("--train-alias")
    predict.add_argument("--acronym-alias", action="store_true")
    predict.add_argument("--qlora", action="store_true")
    predict.add_argument("--max-candidates", type=int, default=8)
    predict.add_argument("--max-length", type=int, default=4096)
    predict.add_argument("--same-passage-only", action="store_true")
    predict.add_argument("--filter-same-passage", action="store_true")
    predict.add_argument(
        "--add-same-sentence",
        action="store_true",
        help="补回与患者 mention 同句且距离足够近的候选，不改变 Qwen 已选结果",
    )
    predict.add_argument("--same-sentence-distance", type=int, default=100)
    args = parser.parse_args()
    if args.command == "train":
        _train(args)
    else:
        _qwen_predict(args)


if __name__ == "__main__":
    main()
