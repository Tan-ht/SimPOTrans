"""Run source-aware MetricX-24 scoring on longform and offline outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


LONGFORM_INSTANCES_NAME = "instances.resegmented.jsonl"
OFFLINE_INSTANCES_NAME = "instances.log"
INSTANCE_NAMES = (LONGFORM_INSTANCES_NAME, OFFLINE_INSTANCES_NAME)
SEGMENT_SCORES_NAME = "metricx_scores.jsonl"
METADATA_NAME = "metricx_metadata.json"
SUMMARY_NAME = "scores.tsv"
METRIC_NAME = "MetricX-24"


@dataclass(frozen=True)
class Example:
    index: int
    docid: Any
    segid: Any
    source: str
    hypothesis: str
    reference: str


@dataclass(frozen=True)
class EvaluationTarget:
    output_dir: Path
    instances_path: Path
    examples: tuple[Example, ...]
    instances_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score longform or offline translation outputs with reference-based "
            "MetricX-24. "
            "Stored scores are negated into [-25, 0], where higher is better."
        )
    )
    parser.add_argument("result_root", type=Path)
    parser.add_argument("source_sentences_file", type=Path, nargs="?")
    parser.add_argument(
        "--evaluation-data-root",
        type=Path,
        help="Root containing <language-pair>/<eval-data>/src_segments.txt.",
    )
    parser.add_argument(
        "--eval-data",
        default="auto",
        help=(
            "Evaluation dataset directory name (for example stream_mcif). "
            "The default 'auto' discovers every dataset and resolves each "
            "output using its language pair and segment count."
        ),
    )
    parser.add_argument("--metricx-repo", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="google/metricx-24-hybrid-xxl-v2p6-bfloat16",
    )
    parser.add_argument("--tokenizer", default="google/mt5-xl")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-length", type=int, default=1536)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def discover_instance_files(result_root: Path) -> list[Path]:
    """Find supported longform and offline instance files."""
    result_root = result_root.resolve()
    if not result_root.is_dir():
        raise FileNotFoundError(f"Result root does not exist: {result_root}")
    direct = [result_root / name for name in INSTANCE_NAMES]
    direct = [path for path in direct if path.is_file()]
    if direct:
        if len(direct) != 1:
            raise ValueError(
                f"Multiple supported instance files found in {result_root}: {direct}"
            )
        return direct

    instance_files = {
        path
        for name in INSTANCE_NAMES
        for path in result_root.rglob(name)
        if name == OFFLINE_INSTANCES_NAME
        or path.parent.name == "omnisteval_longform"
    }
    if not instance_files:
        raise FileNotFoundError(
            f"No supported instance files ({', '.join(INSTANCE_NAMES)}) "
            f"found under: {result_root}"
        )
    return sorted(instance_files, key=str)


def read_source_sentences(path: Path) -> list[str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Source sentences file does not exist: {path}")
    sentences = path.read_text(encoding="utf-8").splitlines()
    if not sentences:
        raise ValueError(f"Source sentences file is empty: {path}")
    empty_lines = [index for index, text in enumerate(sentences) if not text.strip()]
    if empty_lines:
        preview = ", ".join(map(str, empty_lines[:5]))
        raise ValueError(f"Source sentences contain empty lines at indices: {preview}")
    return sentences


def discover_source_files(
    evaluation_data_root: Path, eval_data: str
) -> list[Path]:
    evaluation_data_root = evaluation_data_root.resolve()
    if not evaluation_data_root.is_dir():
        raise FileNotFoundError(
            f"Evaluation data root does not exist: {evaluation_data_root}"
        )
    pattern = (
        "*/*/src_segments.txt"
        if eval_data == "auto"
        else f"*/{eval_data}/src_segments.txt"
    )
    paths = list(evaluation_data_root.glob(pattern))
    if eval_data == "auto":
        # Legacy offline ACL6060 experiments use a shared English source file
        # outside evaluation_data rather than per-language src_segments.txt.
        paths.extend(evaluation_data_root.parent.glob("*_src_text.txt"))
    paths = sorted(set(paths), key=str)
    if not paths:
        raise FileNotFoundError(
            f"No source files matching {pattern} found under "
            f"{evaluation_data_root}"
        )
    return paths


def resolve_source_for_output(
    output_dir: Path,
    source_files: Sequence[Path],
    source_digests: dict[Path, str],
    source_counts: dict[Path, int],
    instance_count: int,
) -> Path:
    """Resolve source by language pair and count, with identical fallback."""
    path_parts = set(output_dir.resolve().parts)
    language_matched = [
        path
        for path in source_files
        if path.parent.parent.name in path_parts
    ]
    count_matched = [
        path
        for path in language_matched
        if source_counts[path] == instance_count
    ]
    if not count_matched:
        # Older offline results may share a source file across target
        # languages and keep it outside the language-pair directories.
        count_matched = [
            path for path in source_files if source_counts[path] == instance_count
        ]
    if len(count_matched) == 1:
        return count_matched[0]
    if not count_matched:
        available = ", ".join(
            f"{path}={source_counts[path]}" for path in source_files
        )
        raise ValueError(
            f"No source file has {instance_count} lines for "
            f"{output_dir}; available sources: {available}"
        )

    digest_groups: dict[str, list[Path]] = {}
    for path in count_matched:
        digest_groups.setdefault(source_digests[path], []).append(path)
    if len(digest_groups) == 1:
        # The same English source segmentation may be shared across target
        # languages. A deterministic representative is safe only after both
        # segment count and full content hash agree.
        return count_matched[0]
    raise ValueError(
        f"Multiple non-identical source files have {instance_count} lines "
        f"for {output_dir}: {count_matched}"
    )


def count_jsonl_records(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_target(
    instances_path: Path, source_sentences: Sequence[str]
) -> EvaluationTarget:
    output_dir = instances_path.parent
    rows: list[dict[str, Any]] = []
    with instances_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {instances_path} at line {line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object in {instances_path} at line {line_number}"
                )
            rows.append(row)

    if len(rows) != len(source_sentences):
        raise ValueError(
            f"Source/instance count mismatch for {output_dir}: "
            f"{len(source_sentences)} source lines versus {len(rows)} instances"
        )

    by_index: dict[int, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        if "index" not in row:
            raise ValueError(
                f"Missing 'index' in {instances_path} at record {row_number}"
            )
        index = row["index"]
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(
                f"Non-integer index in {instances_path} at record "
                f"{row_number}: {index!r}"
            )
        if index in by_index:
            raise ValueError(f"Duplicate index {index} in {instances_path}")
        if index < 0 or index >= len(source_sentences):
            raise ValueError(
                f"Index {index} in {instances_path} is outside source range "
                f"[0, {len(source_sentences) - 1}]"
            )
        by_index[index] = row

    expected_indices = set(range(len(source_sentences)))
    if set(by_index) != expected_indices:
        missing = sorted(expected_indices - set(by_index))
        raise ValueError(f"Missing instance indices in {instances_path}: {missing[:5]}")

    examples: list[Example] = []
    for index in range(len(source_sentences)):
        row = by_index[index]
        hypothesis = row.get("prediction")
        reference = row.get("reference")
        # An empty hypothesis is a valid (very poor) translation and must be
        # scored by MetricX rather than aborting the whole experiment.
        if not isinstance(hypothesis, str):
            raise ValueError(
                f"Invalid non-string prediction at index {index} in {instances_path}"
            )
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(
                f"Empty or invalid reference at index {index} in {instances_path}"
            )
        examples.append(
            Example(
                index=index,
                docid=row.get("docid"),
                segid=row.get("segid"),
                source=source_sentences[index],
                hypothesis=hypothesis,
                reference=reference,
            )
        )
    return EvaluationTarget(
        output_dir=output_dir,
        instances_path=instances_path,
        examples=tuple(examples),
        instances_sha256=sha256(instances_path),
    )


def expected_metadata(
    *,
    target: EvaluationTarget,
    source_path: Path,
    source_sha256: str,
    model_name: str,
    tokenizer_name: str,
    max_input_length: int,
) -> dict[str, Any]:
    return {
        "metric": METRIC_NAME,
        "score_transform": "quality_score = -raw_metricx_score",
        "higher_is_better": True,
        "score_range": [-25.0, 0.0],
        "model": model_name,
        "tokenizer": tokenizer_name,
        "max_input_length": max_input_length,
        "source_sentences_file": str(source_path.resolve()),
        "source_sha256": source_sha256,
        "instances_sha256": target.instances_sha256,
        "num_segments": len(target.examples),
    }


def read_metric_value(output_dir: Path) -> float | None:
    scores_path = output_dir / SUMMARY_NAME
    if scores_path.is_file():
        with scores_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle, delimiter="\t"))
        values: list[str] = []
        if rows and rows[0] == ["metric", "value"]:
            values = [
                row[1]
                for row in rows[1:]
                if len(row) == 2 and row[0] == METRIC_NAME
            ]
        elif len(rows) == 2 and len(rows[0]) == len(rows[1]):
            values = [
                rows[1][index]
                for index, metric in enumerate(rows[0])
                if metric == METRIC_NAME
            ]
        if len(values) == 1:
            try:
                return float(values[0])
            except ValueError:
                pass

    json_path = output_dir / "scores.json"
    try:
        summary = json.loads(json_path.read_text(encoding="utf-8"))
        return float(summary[METRIC_NAME])
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def output_is_current(
    target: EvaluationTarget, metadata_expected: dict[str, Any]
) -> bool:
    try:
        metadata = json.loads(
            (target.output_dir / METADATA_NAME).read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    if metadata != metadata_expected:
        return False

    try:
        score_rows = [
            json.loads(line)
            for line in (target.output_dir / SEGMENT_SCORES_NAME)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    if len(score_rows) != len(target.examples):
        return False
    try:
        aggregate = sum(
            float(row["metricx_score"]) for row in score_rows
        ) / len(score_rows)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    stored = read_metric_value(target.output_dir)
    return stored is not None and math.isclose(
        stored, aggregate, abs_tol=0.000051
    )


def batched(
    items: Sequence[Example], batch_size: int
) -> Iterable[Sequence[Example]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def format_input(example: Example) -> str:
    return (
        f"source: {example.source} candidate: {example.hypothesis} "
        f"reference: {example.reference}"
    )


def tokenize_batch(
    tokenizer: Any, examples: Sequence[Example], max_input_length: int
) -> Any:
    features = []
    for example in examples:
        feature = tokenizer(
            format_input(example),
            max_length=max_input_length,
            truncation=True,
            padding=False,
        )
        # Match the official metricx24/predict.py preprocessing.
        feature["input_ids"] = feature["input_ids"][:-1]
        feature["attention_mask"] = feature["attention_mask"][:-1]
        if not feature["input_ids"]:
            raise ValueError(
                f"Tokenization produced an empty input at index {example.index}"
            )
        features.append(feature)
    return tokenizer.pad(features, padding=True, return_tensors="pt")


def load_model_and_tokenizer(
    metricx_repo: Path,
    model_name: str,
    tokenizer_name: str,
    device: str,
) -> tuple[Any, Any, Any]:
    metricx_repo = metricx_repo.resolve()
    models_path = metricx_repo / "metricx24" / "models.py"
    if not models_path.is_file():
        raise FileNotFoundError(f"Official MetricX models.py not found: {models_path}")
    sys.path.insert(0, str(metricx_repo))

    import torch
    import transformers
    from metricx24 import models

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    model = models.MT5ForRegression.from_pretrained(
        model_name, torch_dtype="auto"
    )
    model.to(resolved_device)
    model.eval()
    return model, tokenizer, resolved_device


def predict_raw_scores(
    model: Any,
    tokenizer: Any,
    device: Any,
    examples: Sequence[Example],
    *,
    batch_size: int,
    max_input_length: int,
) -> list[float]:
    import torch

    predictions: list[float] = []
    with torch.inference_mode():
        for batch_examples in batched(examples, batch_size):
            batch = tokenize_batch(
                tokenizer, batch_examples, max_input_length
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            values = (
                output.predictions.detach().float().cpu().reshape(-1).tolist()
            )
            predictions.extend(float(value) for value in values)

    if len(predictions) != len(examples):
        raise RuntimeError(
            f"MetricX returned {len(predictions)} scores for "
            f"{len(examples)} examples"
        )
    for index, value in enumerate(predictions):
        if not math.isfinite(value) or not 0.0 <= value <= 25.0:
            raise ValueError(f"Invalid raw MetricX score at index {index}: {value}")
    return predictions


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def upsert_summary_score(scores_path: Path, value: float) -> None:
    if not scores_path.is_file():
        raise FileNotFoundError(
            f"OmniSTEval summary does not exist: {scores_path}"
        )
    with scores_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if rows and rows[0] == ["metric", "value"]:
        output_rows = [rows[0]]
        inserted = False
        for row in rows[1:]:
            if len(row) != 2:
                raise ValueError(f"Malformed row in {scores_path}: {row!r}")
            if row[0] == METRIC_NAME:
                if not inserted:
                    output_rows.append([METRIC_NAME, f"{value:.4f}"])
                    inserted = True
            else:
                output_rows.append(row)
        if not inserted:
            output_rows.append([METRIC_NAME, f"{value:.4f}"])
    elif len(rows) == 2 and len(rows[0]) == len(rows[1]):
        header, values = rows
        metric_indices = [
            index for index, metric in enumerate(header) if metric == METRIC_NAME
        ]
        if len(metric_indices) > 1:
            raise ValueError(f"Duplicate {METRIC_NAME} columns in {scores_path}")
        if metric_indices:
            values[metric_indices[0]] = f"{value:.4f}"
        else:
            header.append(METRIC_NAME)
            values.append(f"{value:.4f}")
        output_rows = [header, values]
    else:
        raise ValueError(f"Unexpected scores.tsv layout in {scores_path}")

    temporary = scores_path.with_name(scores_path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(
                handle, delimiter="\t", lineterminator="\n"
            )
            writer.writerows(output_rows)
        os.replace(temporary, scores_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def upsert_json_summary(json_path: Path, value: float) -> None:
    if not json_path.is_file():
        return
    try:
        summary = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON summary in {json_path}: {error}") from error
    if not isinstance(summary, dict):
        raise ValueError(f"Expected a JSON object in {json_path}")
    summary[METRIC_NAME] = round(value, 6)
    atomic_write_text(
        json_path,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )


def write_outputs(
    target: EvaluationTarget,
    raw_scores: Sequence[float],
    metadata: dict[str, Any],
) -> float:
    quality_scores = [
        0.0 if score == 0.0 else -score for score in raw_scores
    ]
    aggregate = sum(quality_scores) / len(quality_scores)
    if not -25.0 <= aggregate <= 0.0:
        raise ValueError(
            f"Aggregated MetricX-24 score is outside [-25, 0]: {aggregate}"
        )

    lines = []
    for example, raw_score, quality_score in zip(
        target.examples, raw_scores, quality_scores, strict=True
    ):
        lines.append(
            json.dumps(
                {
                    "index": example.index,
                    "docid": example.docid,
                    "segid": example.segid,
                    "source": example.source,
                    "hypothesis": example.hypothesis,
                    "reference": example.reference,
                    "raw_metricx_score": raw_score,
                    "metricx_score": quality_score,
                    "model": metadata["model"],
                },
                ensure_ascii=False,
            )
        )
    atomic_write_text(
        target.output_dir / SEGMENT_SCORES_NAME,
        "\n".join(lines) + "\n",
    )
    atomic_write_text(
        target.output_dir / METADATA_NAME,
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    )
    upsert_summary_score(target.output_dir / SUMMARY_NAME, aggregate)
    upsert_json_summary(target.output_dir / "scores.json", aggregate)
    return aggregate


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError(
            f"--batch-size must be positive, got {args.batch_size}"
        )
    if args.max_input_length <= 1:
        raise ValueError(
            "--max-input-length must be greater than 1, got "
            f"{args.max_input_length}"
        )

    instance_files = discover_instance_files(args.result_root)

    explicit_source = (
        args.source_sentences_file.resolve()
        if args.source_sentences_file is not None
        else None
    )
    if explicit_source is not None:
        source_files = [explicit_source]
    else:
        if args.evaluation_data_root is None:
            raise ValueError(
                "--evaluation-data-root is required when source_sentences_file "
                "is omitted"
            )
        source_files = discover_source_files(
            args.evaluation_data_root, args.eval_data
        )
    source_digests = {path: sha256(path) for path in source_files}
    source_cache: dict[Path, list[str]] = {}
    source_counts = {
        path: len(read_source_sentences(path)) for path in source_files
    }

    failures: list[tuple[Path, Exception]] = []
    pending: list[tuple[EvaluationTarget, dict[str, Any]]] = []
    skipped = 0
    for instances_path in instance_files:
        output_dir = instances_path.parent
        try:
            source_path = (
                explicit_source
                if explicit_source is not None
                else resolve_source_for_output(
                    output_dir,
                    source_files,
                    source_digests,
                    source_counts,
                    count_jsonl_records(instances_path),
                )
            )
            if source_path not in source_cache:
                source_cache[source_path] = read_source_sentences(source_path)
            source_sentences = source_cache[source_path]
            target = load_target(instances_path, source_sentences)
            metadata = expected_metadata(
                target=target,
                source_path=source_path,
                source_sha256=source_digests[source_path],
                model_name=args.model,
                tokenizer_name=args.tokenizer,
                max_input_length=args.max_input_length,
            )
            if not args.force and output_is_current(target, metadata):
                print(f"skip (current): {output_dir}")
                skipped += 1
            else:
                pending.append((target, metadata))
        except Exception as error:
            failures.append((output_dir, error))
            print(
                f"validation failed: {output_dir}: {error}",
                file=sys.stderr,
            )

    completed = 0
    if pending:
        model, tokenizer, device = load_model_and_tokenizer(
            args.metricx_repo, args.model, args.tokenizer, args.device
        )
        for target, metadata in pending:
            try:
                print(
                    f"score: {target.output_dir} "
                    f"({len(target.examples)} segments)"
                )
                raw_scores = predict_raw_scores(
                    model,
                    tokenizer,
                    device,
                    target.examples,
                    batch_size=args.batch_size,
                    max_input_length=args.max_input_length,
                )
                aggregate = write_outputs(target, raw_scores, metadata)
                completed += 1
                print(
                    f"{METRIC_NAME}={aggregate:.4f}: {target.output_dir}"
                )
            except Exception as error:
                failures.append((target.output_dir, error))
                print(
                    f"scoring failed: {target.output_dir}: {error}",
                    file=sys.stderr,
                )

    print(
        f"MetricX-24 summary: completed={completed}, skipped={skipped}, "
        f"failed={len(failures)}"
    )
    if failures:
        print("Failed directories:", file=sys.stderr)
        for path, error in failures:
            print(f"  {path}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
