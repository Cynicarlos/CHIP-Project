"""Command line entry point: python -m patientphex ..."""

from __future__ import annotations

import argparse
import json

from .core import evaluate, predict_file, read_jsonl, validate_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="PatientPheX minimal baseline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="generate a submission JSONL")
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--hpo", required=True)
    predict.add_argument("--train-alias", help="optional training JSONL for surface aliases")
    predict.add_argument("--acronym-alias", help="optional training JSONL for full-form acronym pairs")
    predict.add_argument("--association", help="optional JSONL whose association field is reused")

    score = subparsers.add_parser("evaluate", help="evaluate a prediction JSONL")
    score.add_argument("--gold", required=True)
    score.add_argument("--pred", required=True)

    validate = subparsers.add_parser("validate", help="validate a prediction JSONL")
    validate.add_argument("--input", required=True)
    validate.add_argument("--pred", required=True)

    task2 = subparsers.add_parser("build-task2", help="build Qwen patient-candidate SFT data")
    task2.add_argument("--input", required=True)
    task2.add_argument("--output", required=True)
    task2.add_argument("--hpo", required=True)
    task2.add_argument("--max-candidates", type=int, default=8)
    task2.add_argument("--same-passage-only", action="store_true")
    task2.add_argument("--acronym-alias", action="store_true")

    split = subparsers.add_parser("split", help="split whole documents into train/validation")
    split.add_argument("--input", required=True)
    split.add_argument("--train-output", required=True)
    split.add_argument("--valid-output", required=True)
    split.add_argument("--valid-ratio", type=float, default=0.2)
    split.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.command == "predict":
        predict_file(
            args.input,
            args.output,
            args.hpo,
            args.train_alias,
            args.acronym_alias,
            args.association,
        )
    elif args.command == "evaluate":
        print(json.dumps(evaluate(args.gold, args.pred), ensure_ascii=False, indent=2))
    elif args.command == "validate":
        inputs = read_jsonl(args.input)
        predictions = {row["pmc_id"]: row for row in read_jsonl(args.pred)}
        if set(predictions) != {row["pmc_id"] for row in inputs}:
            raise ValueError("prediction documents do not match input documents")
        for doc in inputs:
            validate_prediction(doc, predictions[doc["pmc_id"]])
        print(f"validated_documents={len(inputs)}")
    elif args.command == "build-task2":
        from .data import build_task2_file

        build_task2_file(
            args.input,
            args.output,
            args.hpo,
            args.max_candidates,
            args.same_passage_only,
            args.acronym_alias,
        )
    elif args.command == "split":
        from .split import split_file

        train_count, valid_count = split_file(
            args.input, args.train_output, args.valid_output, args.valid_ratio, args.seed
        )
        print(f"train_documents={train_count} valid_documents={valid_count}")


if __name__ == "__main__":
    main()
