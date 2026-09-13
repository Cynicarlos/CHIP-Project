"""Reproducible document-level train/validation split."""

from __future__ import annotations

import random
from pathlib import Path

from .core import read_jsonl, write_jsonl


def _size(doc: dict) -> tuple[int, int, int, int]:
    return (
        1,
        len(doc.get("patient", [])),
        len(doc.get("entities", [])),
        sum(len(a.get("phenotype", [])) for a in doc.get("association", [])),
    )


def split_documents(docs: list[dict], valid_ratio: float = 0.2, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Split whole documents while approximately matching data-size ratios."""
    if not 0 < valid_ratio < 1:
        raise ValueError("valid_ratio must be between 0 and 1")
    if len(docs) < 2:
        raise ValueError("at least two documents are required")

    rng = random.Random(seed)
    indexed = list(enumerate(docs))
    rng.shuffle(indexed)
    indexed.sort(key=lambda item: _size(item[1])[1:], reverse=True)

    target_docs = max(1, round(len(docs) * valid_ratio))
    totals = [sum(_size(doc)[i] for _, doc in indexed) for i in range(1, 4)]
    targets = [total * valid_ratio for total in totals]
    valid_ids: set[int] = set()
    valid_totals = [0, 0, 0, 0]

    def cost(values: list[int]) -> float:
        # Document count has the strongest constraint; the other dimensions
        # keep large multi-patient papers from concentrating in one split.
        doc_cost = ((values[0] - target_docs) / max(1, target_docs)) ** 2
        size_cost = sum(((values[i] - targets[i - 1]) / max(1, targets[i - 1])) ** 2 for i in range(1, 4))
        return 4 * doc_cost + size_cost

    for processed, (index, doc) in enumerate(indexed):
        stats = list(_size(doc))
        if valid_totals[0] >= target_docs:
            put_valid = False
        elif len(docs) - processed <= target_docs - valid_totals[0]:
            put_valid = True
        else:
            valid_cost = cost([valid_totals[i] + stats[i] for i in range(4)])
            train_cost = cost(valid_totals)
            # Always reserve enough documents for the requested train split.
            put_valid = valid_cost <= train_cost
        if put_valid:
            valid_ids.add(index)
            valid_totals = [valid_totals[i] + stats[i] for i in range(4)]

    # Preserve the original JSONL order inside each split.
    train = [doc for index, doc in enumerate(docs) if index not in valid_ids]
    valid = [doc for index, doc in enumerate(docs) if index in valid_ids]
    return train, valid


def split_file(
    input_path: str | Path,
    train_output: str | Path,
    valid_output: str | Path,
    valid_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[int, int]:
    train, valid = split_documents(read_jsonl(input_path), valid_ratio, seed)
    for path in [Path(train_output), Path(valid_output)]:
        path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, train_output)
    write_jsonl(valid, valid_output)
    return len(train), len(valid)
