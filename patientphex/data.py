"""Build supervised datasets for the PatientPheX pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from .core import (
    HpoLexicon,
    _association_map,
    _passage_for_offset,
    learn_acronym_aliases,
    learn_surface_aliases,
    predict_entities,
    predict_entities_with_acronyms,
    read_jsonl,
    write_jsonl,
)


SYSTEM_PROMPT = (
    "You are a biomedical phenotype relation extractor. "
    "Choose only candidate IDs supported by the supplied evidence. "
    "Return JSON only in the form {\"positive_candidates\":[整数列表]}."
)


def _entity_key(entity: dict) -> tuple:
    return (
        entity["offset"],
        entity["length"],
        entity["identifier"],
        entity["text"],
        entity.get("note"),
    )


def _candidate_context(doc: dict, entity: dict, window: int = 160) -> tuple[str, str]:
    passage = _passage_for_offset(doc, entity["offset"])
    if not passage:
        return "", ""
    local = entity["offset"] - passage["offset"]
    start = max(0, local - window)
    end = min(len(passage["text"]), local + entity["length"] + window)
    return passage["section_type"], passage["text"][start:end]


def _candidate_record(doc: dict, entity: dict, candidate_id: int) -> dict:
    section_type, context = _candidate_context(doc, entity)
    return {
        "candidate_id": candidate_id,
        "identifier": entity["identifier"],
        "text": entity["text"],
        "offset": entity["offset"],
        "section_type": section_type,
        "negated": entity.get("note") == "NO",
        "context": context,
    }


def same_passage_candidates(doc: dict, entities: list[dict], patient_mentions: list[dict]) -> list[dict]:
    """Keep evidence from passages that explicitly mention the target patient."""
    owners = {
        id(passage)
        for mention in patient_mentions
        if (passage := _passage_for_offset(doc, mention["offset"])) is not None
    }
    return [
        entity
        for entity in entities
        if (passage := _passage_for_offset(doc, entity["offset"])) is not None
        and id(passage) in owners
    ]


def make_task2_messages(patient_id: str, mentions: list[str], candidates: list[dict], labels: list[int] | None = None) -> list[dict]:
    """Build the same chat format for training and inference."""
    user = {
        "target_patient": patient_id,
        "patient_mentions": mentions,
        "candidates": candidates,
        "instruction": "Select candidates that describe the target patient.",
    }
    answer = {"positive_candidates": labels or []}
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
    ]


def _is_positive(entity: dict, gold_values: set[str]) -> bool:
    if entity.get("note") == "NO":
        return False
    identifiers = [x for x in entity.get("identifier", "").split(";") if x.startswith("HP:")]
    if any(identifier in gold_values for identifier in identifiers):
        return True
    return entity["text"] in gold_values


def build_task2_records(
    docs: list[dict],
    hpo_path: str | Path,
    alias_docs: list[dict] | None = None,
    max_candidates: int = 8,
    same_passage: bool = False,
    enhanced_entities: bool = False,
) -> list[dict]:
    """Create chunked patient-candidate chat samples.

    The candidate pool is the union of baseline predictions and gold entities.
    The latter prevents missing gold mentions from becoming unlearnable during
    the first training pass; inference uses only baseline/model candidates.
    """
    aliases = learn_surface_aliases(alias_docs or docs, normalize=enhanced_entities)
    lexicon = HpoLexicon(hpo_path, aliases, normalize_tokens=enhanced_entities)
    acronym_aliases = learn_acronym_aliases(alias_docs or docs) if enhanced_entities else {}
    records = []
    for doc in docs:
        predicted = (
            predict_entities_with_acronyms(doc, lexicon, acronym_aliases)
            if enhanced_entities else predict_entities(doc, lexicon)
        )
        by_key = {_entity_key(entity): entity for entity in predicted}
        for entity in doc.get("entities", []):
            by_key.setdefault(_entity_key(entity), entity)
        raw_candidates = list(by_key.values())

        gold_associations = _association_map(doc)
        for patient in doc.get("patient", []):
            patient_id = patient["patient_id"]
            mentions = [mention["text"] for mention in patient.get("mention", [])]
            candidates = raw_candidates
            if same_passage:
                candidates = same_passage_candidates(doc, candidates, patient.get("mention", []))
            gold_values = gold_associations.get(patient_id, set())
            for chunk_id, start in enumerate(range(0, len(candidates), max_candidates)):
                chunk = candidates[start : start + max_candidates]
                serialized = [_candidate_record(doc, entity, i) for i, entity in enumerate(chunk)]
                labels = [
                    candidate["candidate_id"]
                    for candidate, entity in zip(serialized, chunk)
                    if _is_positive(entity, gold_values)
                ]
                records.append(
                    {
                        "pmc_id": doc["pmc_id"],
                        "patient_id": patient_id,
                        "chunk_id": chunk_id,
                        "messages": make_task2_messages(patient_id, mentions, serialized, labels),
                    }
                )
    return records


def build_task2_file(
    input_path: str | Path,
    output_path: str | Path,
    hpo_path: str | Path,
    max_candidates: int = 32,
    same_passage: bool = False,
    enhanced_entities: bool = False,
) -> None:
    docs = read_jsonl(input_path)
    write_jsonl(
        build_task2_records(docs, hpo_path, docs, max_candidates, same_passage, enhanced_entities),
        output_path,
    )
