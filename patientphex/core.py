"""Core data, HPO matching, prediction, and evaluation utilities.

This is intentionally dependency-free.  It provides a reproducible baseline
that can later be replaced piece by piece by a neural NER/linker or relation
classifier.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
SENTENCE_RE = re.compile(r"[^.!?]+(?:[.!?]+|$)", re.S)
ACRONYM_RE = re.compile(r"[ \t]*\(([A-Z][A-Z0-9/-]{1,9})(?:[ \t]*[;,:]|[ \t]*\))")
SLASH_ACRONYM_RE = re.compile(r"([A-Z][A-Z0-9-]{2,9})/$")
SPELLING_VARIANTS = {
    "centre": "center",
    "centres": "centers",
    "behaviour": "behavior",
    "behavioural": "behavioral",
    "tumour": "tumor",
    "tumours": "tumors",
    "fibre": "fiber",
    "fibres": "fibers",
}

# Very common one-token words in HPO are too ambiguous for an exact matcher.
# Multi-token terms and distinctive one-token phenotype names are retained.
GENERIC_SINGLE = {
    "adult", "age", "birth", "boy", "case", "child", "children", "clinical",
    "condition", "disease", "female", "finding", "girl", "history", "infant",
    "male", "man", "normal", "patient", "person", "subject", "woman", "year",
}


def read_jsonl(path: str | Path) -> list[dict]:
    """Read one JSON object per line."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    """Write UTF-8 JSONL without changing the source order."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def validate_document(doc: dict) -> None:
    """Check that annotated spans match the source passage text."""
    passages = doc.get("full_text", [])
    for entity in doc.get("entities", []):
        start, end = entity["offset"], entity["offset"] + entity["length"]
        owner = next(
            (p for p in passages if p["offset"] <= start and end <= p["offset"] + len(p["text"])),
            None,
        )
        if owner is None:
            raise ValueError(f"entity outside passages: {doc['pmc_id']} {entity}")
        local_start = start - owner["offset"]
        actual = owner["text"][local_start : local_start + entity["length"]]
        if actual != entity["text"]:
            raise ValueError(f"offset/text mismatch: {doc['pmc_id']} {entity}")


def _token_key(text: str, normalize: bool = True) -> tuple[str, ...]:
    tokens = (m.group(0) for m in TOKEN_RE.finditer(text))
    return tuple(_normalise_token(token) if normalize else token.lower() for token in tokens)


def _normalise_token(token: str) -> str:
    """Normalize spelling and common English plural forms for HPO lookup."""
    token = SPELLING_VARIANTS.get(token.lower(), token.lower())
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 4 and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def _usable_key(key: tuple[str, ...]) -> bool:
    """Reject very short, generic aliases that are not useful as phenotype terms."""
    return bool(key) and not (len(key) == 1 and (len(key[0]) < 5 or key[0] in GENERIC_SINGLE))


def _parse_obo(path: str | Path) -> dict[str, dict]:
    """Parse only the HPO fields needed by the matcher."""
    terms: dict[str, dict] = {}
    current: dict | None = None
    in_term = False

    def flush() -> None:
        nonlocal current
        if current and current.get("id", "").startswith("HP:") and not current.get("obsolete"):
            terms[current["id"]] = current
        current = None

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line == "[Term]":
                flush()
                current = {"synonyms": [], "parents": []}
                in_term = True
                continue
            if line.startswith("[") and line != "[Term]":
                flush()
                in_term = False
                continue
            if not in_term or current is None:
                continue
            if line.startswith("id: "):
                current["id"] = line[4:]
            elif line.startswith("name: "):
                current["name"] = line[6:]
            elif line.startswith("synonym: "):
                match = re.match(r'synonym: "(.*?)"', line)
                if match:
                    current["synonyms"].append(match.group(1))
            elif line.startswith("is_a: "):
                current["parents"].append(line[6:].split(" ! ", 1)[0])
            elif line == "is_obsolete: true":
                current["obsolete"] = True
        flush()
    return terms


def _phenotypic_branch(terms: dict[str, dict], root: str = "HP:0000118") -> set[str]:
    children: dict[str, list[str]] = defaultdict(list)
    for term_id, term in terms.items():
        for parent in term["parents"]:
            children[parent].append(term_id)

    branch = {root}
    stack = [root]
    while stack:
        parent = stack.pop()
        for child in children[parent]:
            if child not in branch:
                branch.add(child)
                stack.append(child)
    return branch


def learn_surface_aliases(
    docs: Iterable[dict], normalize: bool = True
) -> dict[tuple[str, ...], str]:
    """Learn train-only surface-to-label aliases for phrases absent from HPO."""
    counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for doc in docs:
        for entity in doc.get("entities", []):
            if entity.get("note") == "NO":
                continue
            ids = entity.get("identifier", "-1")
            if "-1" in ids:
                continue
            key = _token_key(entity["text"], normalize)
            if _usable_key(key):
                counts[key][ids] += 1
    return {key: labels.most_common(1)[0][0] for key, labels in counts.items()}


def learn_acronym_aliases(docs: Iterable[dict]) -> dict[str, str]:
    """Learn high-precision full-form-to-acronym mappings from annotations."""
    mappings: dict[str, Counter[str]] = defaultdict(Counter)
    for doc in docs:
        for passage in doc.get("full_text", []):
            entities = [
                entity
                for entity in doc.get("entities", [])
                if entity.get("note") != "NO"
                and passage["offset"] <= entity["offset"]
                and entity["offset"] + entity["length"] <= passage["offset"] + len(passage["text"])
            ]
            for entity in entities:
                end = entity["offset"] - passage["offset"] + entity["length"]
                match = ACRONYM_RE.match(passage["text"], end)
                if not match:
                    continue
                start = passage["offset"] + match.start(1)
                acronym = next(
                    (item for item in entities
                     if item["offset"] == start and item["length"] == len(match.group(1))),
                    None,
                )
                if acronym and acronym.get("identifier") == entity.get("identifier"):
                    mappings[match.group(1)][entity["identifier"]] += 1
    return {acronym: ids.most_common(1)[0][0] for acronym, ids in mappings.items()}


class HpoLexicon:
    """HPO name/synonym matcher using longest token-span matching."""

    def __init__(
        self,
        obo_path: str | Path,
        aliases: dict[tuple[str, ...], str] | None = None,
        normalize_tokens: bool = True,
    ):
        self.normalize_tokens = normalize_tokens
        terms = _parse_obo(obo_path)
        branch = _phenotypic_branch(terms)
        forms: dict[tuple[str, ...], str] = {}

        for term_id in branch:
            term = terms.get(term_id)
            if not term:
                continue
            for phrase in [term.get("name", ""), *term.get("synonyms", [])]:
                key = _token_key(phrase, normalize_tokens)
                if not _usable_key(key):
                    continue
                ids = set(forms.get(key, "").split(";")) - {""}
                ids.add(term_id)
                forms[key] = ";".join(sorted(ids))

        # Train aliases are deliberately allowed to override ambiguous HPO synonyms.
        for key, identifier in (aliases or {}).items():
            if key:
                forms[key] = identifier

        self.forms = forms
        self.lengths_by_first: dict[str, list[int]] = defaultdict(list)
        for key in forms:
            self.lengths_by_first[key[0]].append(len(key))
        for first in self.lengths_by_first:
            self.lengths_by_first[first] = sorted(set(self.lengths_by_first[first]), reverse=True)

    def lookup(self, text: str) -> str | None:
        """Return the learned/dictionary ID for one predicted entity span."""
        key = _token_key(text, self.normalize_tokens)
        return self.forms.get(key)

    def match(self, text: str) -> list[tuple[int, int, str]]:
        """Return non-overlapping ``(start, end, HPO-id-string)`` matches."""
        tokens = list(TOKEN_RE.finditer(text))
        raw_matches: list[tuple[int, int, str]] = []
        for i, token in enumerate(tokens):
            first = _normalise_token(token.group(0)) if self.normalize_tokens else token.group(0).lower()
            lengths = self.lengths_by_first.get(first, [])
            for length in lengths:
                if i + length > len(tokens):
                    continue
                key = tuple(
                    _normalise_token(t.group(0)) if self.normalize_tokens else t.group(0).lower()
                    for t in tokens[i : i + length]
                )
                identifier = self.forms.get(key)
                if identifier:
                    raw_matches.append((tokens[i].start(), tokens[i + length - 1].end(), identifier))

        # Longest match wins, preventing e.g. "cognitive impairment" and "impairment".
        raw_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        selected: list[tuple[int, int, str]] = []
        for match in raw_matches:
            if selected and match[0] < selected[-1][1]:
                continue
            selected.append(match)
        return selected


def _is_negated(text: str, start: int, end: int) -> bool:
    """Small high-precision negation heuristic for the baseline."""
    left = text[max(0, start - 100) : start]
    left = re.split(r"[.!?;:\n]", left)[-1]
    right = text[end : min(len(text), end + 60)]
    before = re.search(
        r"(?:\bno\b|\bwithout\b|\babsen(?:ce|t)\b|\bden(?:y|ied|ies)\b|"
        r"\bnegative for\b|\bfree of\b|\black of\b)\W*$",
        left,
        re.I,
    )
    after = re.match(r"\W*(?:was|were|is|are|remained)?\W*(?:absent|negative|denied)", right, re.I)
    return bool(before or after)


def predict_entities(doc: dict, lexicon: HpoLexicon) -> list[dict]:
    """Recognize all HPO-like mentions while preserving global offsets."""
    entities = []
    for passage in doc.get("full_text", []):
        text = passage["text"]
        for start, end, identifier in lexicon.match(text):
            entities.append(
                {
                    "identifier": identifier,
                    "type": "Phenotype",
                    "offset": passage["offset"] + start,
                    "length": end - start,
                    "text": text[start:end],
                    "note": "NO" if _is_negated(text, start, end) else None,
                }
            )
    return sorted(entities, key=lambda e: (e["offset"], -e["length"]))


def predict_entities_with_acronyms(
    doc: dict, lexicon: HpoLexicon, acronym_aliases: dict[str, str]
) -> list[dict]:
    """Add train-seen or unambiguous initialism abbreviations after a full form."""
    entities = predict_entities(doc, lexicon)
    for passage in doc.get("full_text", []):
        text = passage["text"]
        local_entities = [
            entity for entity in entities
            if passage["offset"] <= entity["offset"] < passage["offset"] + len(passage["text"])
        ]
        for entity in list(local_entities):
            end = entity["offset"] - passage["offset"] + entity["length"]
            match = ACRONYM_RE.match(passage["text"], end)
            if not match:
                continue
            acronym = match.group(1)
            identifier = acronym_aliases.get(acronym)
            letters = "".join(char for char in acronym if char.isalpha()).upper()
            initials = "".join(
                token.group(0)[0].upper() for token in TOKEN_RE.finditer(entity["text"])
            )
            if identifier is None and letters == initials:
                identifier = entity["identifier"]
            if identifier != entity["identifier"]:
                continue
            start = passage["offset"] + match.start(1)
            if any(item["offset"] == start and item["length"] == len(match.group(1)) for item in entities):
                continue
            entities.append(
                {
                    "identifier": identifier,
                    "type": "Phenotype",
                    "offset": start,
                    "length": len(acronym),
                    "text": acronym,
                    "note": None,
                }
            )
        for entity in list(local_entities):
            if entity.get("identifier") == "-1":
                continue
            local_start = entity["offset"] - passage["offset"]
            match = SLASH_ACRONYM_RE.search(text[:local_start])
            if not match:
                continue
            acronym = match.group(1)
            start = passage["offset"] + match.start(1)
            if any(item["offset"] == start and item["length"] == len(acronym) for item in entities):
                continue
            entities.append(
                {
                    "identifier": entity["identifier"],
                    "type": "Phenotype",
                    "offset": start,
                    "length": len(acronym),
                    "text": acronym,
                    "note": None,
                }
            )
    return sorted(entities, key=lambda e: (e["offset"], -e["length"]))


def _sentence_index(text: str, position: int) -> int:
    for index, match in enumerate(SENTENCE_RE.finditer(text)):
        if match.start() <= position < match.end():
            return index
    return -1


def _passage_for_offset(doc: dict, offset: int) -> dict | None:
    return next(
        (p for p in doc.get("full_text", []) if p["offset"] <= offset < p["offset"] + len(p["text"])),
        None,
    )


def _association_score(entity: dict, patient_mention: dict, passage: dict, patient_count: int) -> float:
    """Score local evidence between one phenotype and one supplied patient mention."""
    entity_local = entity["offset"] - passage["offset"]
    patient_local = patient_mention["offset"] - passage["offset"]
    distance = abs(entity_local - patient_local)
    score = 2.0  # same passage
    if _sentence_index(passage["text"], entity_local) == _sentence_index(passage["text"], patient_local):
        score += 4.0
    score += max(0.0, 2.0 - distance / 400.0)
    if passage["section_type"] in {"CASE", "RESULTS", "FIG"}:
        score += 1.0
    if patient_count == 1 and passage["section_type"] in {"CASE", "RESULTS", "FIG"}:
        score += 2.0
    return score


def associate_patients(doc: dict, entities: list[dict]) -> list[dict]:
    """Assign candidate entities to patients using supplied local mentions.

    This is intentionally conservative for multi-patient papers.  It is a
    baseline relation module, not a full coreference resolver.
    """
    patients = doc.get("patient", [])
    output = {patient["patient_id"]: [] for patient in patients}
    mentions = []
    for patient in patients:
        for mention in patient.get("mention", []):
            passage = _passage_for_offset(doc, mention["offset"])
            if passage:
                mentions.append((patient["patient_id"], mention, passage))

    for entity in entities:
        if entity.get("note") == "NO":
            continue
        passage = _passage_for_offset(doc, entity["offset"])
        if not passage:
            continue
        candidates = [
            (patient_id, _association_score(entity, mention, owner, len(patients)))
            for patient_id, mention, owner in mentions
            if owner is passage
        ]
        if not candidates:
            continue
        patient_id, score = max(candidates, key=lambda item: item[1])
        threshold = 5.0 if len(patients) == 1 else 6.0
        if score < threshold:
            continue
        value = entity["text"] if entity["identifier"] == "-1" else entity["identifier"]
        if value not in output[patient_id]:
            output[patient_id].append(value)

    return [{"patient_id": patient_id, "phenotype": values} for patient_id, values in output.items()]


def predict_document(
    doc: dict,
    lexicon: HpoLexicon,
    acronym_aliases: dict[str, str] | None = None,
) -> dict:
    entities = predict_entities_with_acronyms(doc, lexicon, acronym_aliases or {})
    return {
        "pmc_id": doc["pmc_id"],
        "pmid": doc.get("pmid"),
        "entities": entities,
        "association": associate_patients(doc, entities),
    }


def validate_prediction(input_doc: dict, prediction: dict) -> None:
    """Validate the fields required by the competition submission format."""
    if prediction.get("pmc_id") != input_doc.get("pmc_id"):
        raise ValueError(f"pmc_id mismatch: {prediction.get('pmc_id')}")
    valid_patients = {patient["patient_id"] for patient in input_doc.get("patient", [])}
    seen_patients = set()
    for entity in prediction.get("entities", []):
        required = {"identifier", "type", "offset", "length", "text", "note"}
        if not required <= entity.keys() or entity["type"] != "Phenotype":
            raise ValueError(f"invalid entity: {entity}")
        if entity["offset"] < 0 or entity["length"] != len(entity["text"]):
            raise ValueError(f"invalid entity span: {entity}")
        start, end = entity["offset"], entity["offset"] + entity["length"]
        passage = next(
            (p for p in input_doc.get("full_text", [])
             if p["offset"] <= start and end <= p["offset"] + len(p["text"])),
            None,
        )
        if passage is None or passage["text"][start - passage["offset"] : end - passage["offset"]] != entity["text"]:
            raise ValueError(f"entity does not match source text: {entity}")
    for association in prediction.get("association", []):
        patient_id = association.get("patient_id")
        if patient_id not in valid_patients or patient_id in seen_patients:
            raise ValueError(f"invalid or duplicate patient_id: {patient_id}")
        seen_patients.add(patient_id)
        if not isinstance(association.get("phenotype"), list):
            raise ValueError(f"invalid phenotype list: {association}")
    if seen_patients != valid_patients:
        missing = valid_patients - seen_patients
        raise ValueError(f"missing patient associations: {sorted(missing)}")


def predict_file(input_path: str | Path, output_path: str | Path, hpo_path: str | Path,
                 train_alias_path: str | Path | None = None,
                 acronym_alias_path: str | Path | None = None,
                 association_path: str | Path | None = None) -> None:
    docs = read_jsonl(input_path)
    alias_docs = read_jsonl(train_alias_path) if train_alias_path else []
    aliases = learn_surface_aliases(alias_docs) if train_alias_path else None
    acronym_aliases = learn_acronym_aliases(read_jsonl(acronym_alias_path)) if acronym_alias_path else {}
    associations = (
        {row["pmc_id"]: row["association"] for row in read_jsonl(association_path)}
        if association_path else {}
    )
    lexicon = HpoLexicon(hpo_path, aliases)
    predictions = []
    for doc in docs:
        prediction = predict_document(doc, lexicon, acronym_aliases)
        if doc["pmc_id"] in associations:
            prediction["association"] = associations[doc["pmc_id"]]
        predictions.append(prediction)
    for doc, prediction in zip(docs, predictions):
        validate_prediction(doc, prediction)
    write_jsonl(predictions, output_path)


def _hpo_units(identifier: str) -> list[str]:
    return [part for part in identifier.split(";") if part]


def _mention_units(entity: dict) -> list[tuple[int, int, str]]:
    span = (entity["offset"], entity["length"])
    return [(*span, identifier) for identifier in _hpo_units(entity["identifier"]) if identifier.startswith("HP:")]


def _mention_metrics(gold: dict, pred: dict) -> tuple[int, int, int]:
    gold_positive: list[tuple[int, int, str]] = []
    gold_noid: list[tuple[int, int, str]] = []
    gold_negative: set[tuple[int, int]] = set()
    for entity in gold.get("entities", []):
        span = (entity["offset"], entity["length"])
        if entity.get("note") == "NO":
            gold_negative.add(span)
        elif "-1" in _hpo_units(entity.get("identifier", "")):
            gold_noid.append((*span, "-1"))
            gold_positive.extend(_mention_units(entity))
        else:
            gold_positive.extend(_mention_units(entity))

    used_positive: set[int] = set()
    used_noid: set[int] = set()
    tp = fp = 0
    for entity in pred.get("entities", []):
        if entity.get("note") == "NO":
            continue
        if "-1" in _hpo_units(entity.get("identifier", "")):
            span = (entity["offset"], entity["length"])
            match = next((i for i, g in enumerate(gold_noid) if i not in used_noid and span == g[:2]), None)
            if match is not None:
                used_noid.add(match)
                tp += 1
            else:
                fp += 1
        for unit in _mention_units(entity):
            match = next((i for i, g in enumerate(gold_positive) if i not in used_positive and unit == g), None)
            if match is not None:
                used_positive.add(match)
                tp += 1
                continue
            # The official rule scores an unlinked entity by position only.
            match = next((i for i, g in enumerate(gold_noid) if i not in used_noid and unit[:2] == g[:2]), None)
            if match is not None:
                used_noid.add(match)
                tp += 1
            else:
                fp += 1

    fn = (len(gold_positive) - len(used_positive)) + (len(gold_noid) - len(used_noid))
    # A positive prediction at a gold negative span is an FP, already counted
    # above.  Negative predictions are intentionally ignored.
    _ = gold_negative
    return tp, fp, fn


def _doc_hpo_set(doc: dict, field: str) -> set[str]:
    values = set()
    if field == "entities":
        for entity in doc.get("entities", []):
            if entity.get("note") != "NO" and entity.get("identifier") != "-1":
                values.update(identifier for identifier in _hpo_units(entity["identifier"]) if identifier.startswith("HP:"))
    else:
        for association in doc.get("association", []):
            for value in association.get("phenotype", []):
                if value.startswith("HP:"):
                    values.update(_hpo_units(value))
    return values


def _association_map(doc: dict) -> dict[str, set[str]]:
    result = {}
    for association in doc.get("association", []):
        values = set()
        for value in association.get("phenotype", []):
            if value.startswith("HP:") and ";" in value:
                values.update(identifier for identifier in _hpo_units(value) if identifier.startswith("HP:"))
            else:
                values.add(value)
        result[association["patient_id"]] = values
    return result


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate(gold_path: str | Path, pred_path: str | Path) -> dict[str, float]:
    """Evaluate predictions with the four competition metrics."""
    gold_rows = {row["pmc_id"]: row for row in read_jsonl(gold_path)}
    pred_rows = {row["pmc_id"]: row for row in read_jsonl(pred_path)}
    mention = [0, 0, 0]
    document = [0, 0, 0]
    association = [0, 0, 0]
    macro_f1 = []

    for pmc_id, gold in gold_rows.items():
        pred = pred_rows.get(pmc_id, {"entities": [], "association": []})
        mtp, mfp, mfn = _mention_metrics(gold, pred)
        mention = [mention[i] + (mtp, mfp, mfn)[i] for i in range(3)]

        gold_set = _doc_hpo_set(gold, "entities")
        pred_set = _doc_hpo_set(pred, "entities")
        document[0] += len(gold_set & pred_set)
        document[1] += len(pred_set - gold_set)
        document[2] += len(gold_set - pred_set)

        gold_assoc = _association_map(gold)
        pred_assoc = _association_map(pred)
        for patient_id in gold_assoc:
            g, p = gold_assoc[patient_id], pred_assoc.get(patient_id, set())
            association[0] += len(g & p)
            association[1] += len(p - g)
            association[2] += len(g - p)
            if not g and not p:
                macro_f1.append(1.0)
            else:
                macro_f1.append(_f1(len(g & p), len(p - g), len(g - p)))

    scores = {
        "f1_men": _f1(*mention),
        "f1_doc": _f1(*document),
        "f1_micro": _f1(*association),
        "f1_macro": sum(macro_f1) / len(macro_f1) if macro_f1 else 0.0,
    }
    scores["score"] = 0.25 * sum(scores.values())
    return scores
