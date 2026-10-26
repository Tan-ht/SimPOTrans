from __future__ import annotations

import argparse
import csv
import json
import numbers
import os
import subprocess
import tempfile
from pathlib import Path
from textwrap import dedent

import pandas as pd


DEFAULT_COMET_MODEL = "Unbabel/XCOMET-XL"


_COMET_SCORE_SCRIPT = """
import json
import sys
from comet import download_model, load_from_checkpoint


def to_float(value):
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


data_path, model_name, batch_size, gpus = sys.argv[1:]
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

model_path = download_model(model_name)
model = load_from_checkpoint(model_path)
output = model.predict(data, batch_size=int(batch_size), gpus=int(gpus))
scores = getattr(output, "scores", None)
if scores is None:
    scores = [getattr(output, "system_score")] * len(data)
score_values = [to_float(score) for score in scores]
system_score = getattr(output, "system_score", None)
if system_score is None:
    system_score = sum(score_values) / max(len(score_values), 1)
print(json.dumps({"system_score": to_float(system_score), "scores": score_values}))
"""


def _round_number(value):
    if isinstance(value, numbers.Number):
        return round(value, 6)
    return value


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _load_instances(output_dir: Path) -> list[dict[str, object]]:
    instances_path = output_dir / "instances.log"
    if not instances_path.exists():
        raise FileNotFoundError(f"Missing instances log: {instances_path}")

    instances: list[dict[str, object]] = []
    with instances_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                instances.append(json.loads(line))
    return sorted(instances, key=lambda row: int(row.get("index", len(instances))))


def _load_tsv_src_text(path: Path) -> list[str] | None:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None or "src_text" not in reader.fieldnames:
            return None
        return [row["src_text"] for row in reader]


def _infer_acl_source_text(root_path: Path, eval_data_path: Path) -> Path | None:
    if eval_data_path.name != "simul_eval_acl":
        return None
    candidate = root_path / "data" / "acl6060_src_text.txt"
    return candidate if candidate.exists() else None


def _find_source_text_path(root_path: Path, eval_data_path: Path) -> Path | None:
    for name in (
        "source_text.txt",
        "src_text.txt",
        "source_segments.txt",
        "src_segments.txt",
    ):
        candidate = eval_data_path / name
        if candidate.exists():
            return candidate

    acl_source_text = _infer_acl_source_text(root_path, eval_data_path)
    if acl_source_text is not None:
        return acl_source_text

    for candidate in sorted(eval_data_path.glob("*.tsv")):
        if _load_tsv_src_text(candidate) is not None:
            return candidate

    return None


def _load_source_texts(path: Path, expected_len: int) -> list[str]:
    if path.suffix == ".tsv":
        source_texts = _load_tsv_src_text(path)
        if source_texts is None:
            raise ValueError(f"Source TSV does not contain a src_text column: {path}")
    else:
        source_texts = _read_lines(path)

    if len(source_texts) != expected_len:
        raise ValueError(
            f"Source text count mismatch for {path}: expected {expected_len}, got {len(source_texts)}"
        )
    return source_texts


def _run_comet(
    data: list[dict[str, str]],
    *,
    python: Path,
    model_name: str,
    batch_size: int,
    gpus: int,
    visible_devices: str | None,
) -> tuple[float, list[float]]:
    if not python.is_absolute():
        python = Path.cwd() / python
    if not python.is_file():
        raise RuntimeError(
            f"COMET Python interpreter not found: {python}. Run scripts/setup_comet_env.sh first, "
            "or set COMET_PYTHON=/path/to/python."
        )

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as f:
        json.dump(data, f, ensure_ascii=False)
        data_path = Path(f.name)

    env = os.environ.copy()
    if visible_devices not in (None, "", "none", "null"):
        env["CUDA_VISIBLE_DEVICES"] = str(visible_devices)

    try:
        result = subprocess.run(
            [str(python), "-c", _COMET_SCORE_SCRIPT, str(data_path), model_name, str(batch_size), str(gpus)],
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
    finally:
        data_path.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(
            "COMET scoring subprocess failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"COMET scoring subprocess produced no output. stderr:\n{result.stderr}")

    payload = json.loads(lines[-1])
    scores = [float(score) for score in payload["scores"]]
    if len(scores) != len(data):
        raise RuntimeError(f"COMET returned {len(scores)} score(s) for {len(data)} example(s).")
    return float(payload["system_score"]), scores


def _merge_scores(scores_path: Path, scores: dict[str, float]) -> None:
    table = pd.read_csv(scores_path, sep="\t") if scores_path.exists() else pd.DataFrame([{}])
    if table.empty:
        table = pd.DataFrame([{}])
    for name, value in scores.items():
        table[name] = [_round_number(value)]
    table.to_csv(scores_path, sep="\t", index=False)


def _merge_metrics(metrics_path: Path, rows: list[dict[str, float | int]]) -> None:
    new_metrics = pd.DataFrame(rows)
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path, sep="\t")
        if "index" in metrics.columns:
            metrics = metrics.drop(
                columns=[column for column in new_metrics.columns if column in metrics.columns and column != "index"]
            )
            merged = metrics.merge(new_metrics, on="index", how="outer")
        else:
            new_without_index = new_metrics.drop(columns=["index"], errors="ignore")
            metrics = metrics.drop(columns=[column for column in new_without_index.columns if column in metrics.columns])
            merged = pd.concat([metrics, new_without_index], axis=1)
    else:
        merged = new_metrics
    merged = merged.round(6)
    merged.to_csv(metrics_path, sep="\t", index=False)


def _json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Number):
        return round(float(value), 6)
    return value


def _write_scores_json(scores_path: Path, json_path: Path) -> dict[str, object]:
    scores = pd.read_csv(scores_path, sep="	")
    if scores.empty:
        payload: dict[str, object] = {}
    else:
        payload = {column: _json_value(scores.iloc[0][column]) for column in scores.columns}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add XCOMET-XL reference and reference-free QE scores to a SimulEval output directory."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--eval-data", required=True, type=Path)
    parser.add_argument("--root-path", required=True, type=Path)
    parser.add_argument("--source-text", type=Path)
    parser.add_argument("--comet-python", default=".comet-venv/bin/python", type=Path)
    parser.add_argument("--comet-model", default=DEFAULT_COMET_MODEL)
    parser.add_argument("--comet-batch-size", default=4, type=int)
    parser.add_argument("--comet-gpus", default=1, type=int)
    parser.add_argument("--comet-visible-devices")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--skip-qe", action="store_true")
    parser.add_argument("--json-output", default="scores.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instances = _load_instances(args.output)
    if not instances:
        raise ValueError(f"No instances found under {args.output}")

    source_text_path = args.source_text or _find_source_text_path(args.root_path, args.eval_data)
    if source_text_path is None:
        raise FileNotFoundError(
            dedent(
                f"""
                Could not find source transcripts for COMET/QE scoring under {args.eval_data}.
                Provide one with COMET_SOURCE_TEXT=/path/to/source_text.txt or pass --source-text.
                """
            ).strip()
        )

    sources = _load_source_texts(source_text_path, len(instances))
    hypotheses = [str(row.get("prediction", "")) for row in instances]
    references = [str(row.get("reference", "")) for row in instances]
    indices = [int(row.get("index", i)) for i, row in enumerate(instances)]

    corpus_scores: dict[str, float] = {}
    metric_rows: list[dict[str, float | int]] = [{"index": index} for index in indices]

    if not args.skip_reference:
        reference_data = [
            {"src": src, "mt": hyp, "ref": ref}
            for src, hyp, ref in zip(sources, hypotheses, references, strict=True)
        ]
        system_score, sent_scores = _run_comet(
            reference_data,
            python=args.comet_python,
            model_name=args.comet_model,
            batch_size=args.comet_batch_size,
            gpus=args.comet_gpus,
            visible_devices=args.comet_visible_devices,
        )
        corpus_scores["COMET"] = system_score
        for row, score in zip(metric_rows, sent_scores, strict=True):
            row["COMET"] = score

    if not args.skip_qe:
        qe_data = [
            {"src": src, "mt": hyp}
            for src, hyp in zip(sources, hypotheses, strict=True)
        ]
        system_score, sent_scores = _run_comet(
            qe_data,
            python=args.comet_python,
            model_name=args.comet_model,
            batch_size=args.comet_batch_size,
            gpus=args.comet_gpus,
            visible_devices=args.comet_visible_devices,
        )
        corpus_scores["COMET_QE"] = system_score
        for row, score in zip(metric_rows, sent_scores, strict=True):
            row["COMET_QE"] = score

    if not corpus_scores:
        raise ValueError("Both reference COMET and QE scoring were skipped.")

    json_output = Path(args.json_output)
    if not json_output.is_absolute():
        json_output = args.output / json_output

    scores_path = args.output / "scores.tsv"
    _merge_scores(scores_path, corpus_scores)
    _merge_metrics(args.output / "metrics.tsv", metric_rows)
    scores_payload = _write_scores_json(scores_path, json_output)
    print(json.dumps(scores_payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
