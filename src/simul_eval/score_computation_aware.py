from __future__ import annotations

import argparse
import numbers
from pathlib import Path
from statistics import mean

import pandas as pd
from simuleval.evaluator.instance import LogInstance
from simuleval.evaluator.scorers import get_scorer_class


DEFAULT_LATENCY_METRICS = ["LAAL", "AL", "AP", "DAL", "ATD"]


def _load_instances(path: Path, latency_unit: str) -> dict[int, LogInstance]:
    instances: dict[int, LogInstance] = {}
    with path.open("r", encoding="utf-8") as f:
        for local_index, line in enumerate(f):
            if not line.strip():
                continue
            instances[local_index] = LogInstance(line, latency_unit)
    return instances


def _score_instance(
    instance: LogInstance,
    metric: str,
    *,
    computation_aware: bool,
    use_ref_len: bool,
) -> float | None:
    scorer = get_scorer_class("latency", metric)(
        computation_aware=computation_aware,
        use_ref_len=use_ref_len,
    )
    try:
        return scorer({0: instance})
    except AssertionError:
        return None


def _compute_latency_scores(
    instances: dict[int, LogInstance],
    metrics: list[str],
    *,
    use_ref_len: bool,
) -> tuple[dict[str, float], list[dict[str, float | int | None]]]:
    corpus_scores: dict[str, float] = {}
    instance_rows: list[dict[str, float | int | None]] = []

    per_metric_values: dict[str, list[float]] = {
        name: [] for metric in metrics for name in (metric, f"{metric}_CA")
    }

    for instance in instances.values():
        row: dict[str, float | int | None] = {"index": instance.index}
        for metric in metrics:
            for suffix, computation_aware in (("", False), ("_CA", True)):
                name = f"{metric}{suffix}"
                score = _score_instance(
                    instance,
                    metric,
                    computation_aware=computation_aware,
                    use_ref_len=use_ref_len,
                )
                row[name] = score
                if score is not None:
                    per_metric_values[name].append(score)
        instance_rows.append(row)

    for name, values in per_metric_values.items():
        if values:
            corpus_scores[name] = mean(values)

    return corpus_scores, instance_rows


def _round_number(value):
    if isinstance(value, numbers.Number):
        return round(value, 3)
    return value


def _merge_scores(scores_path: Path, latency_scores: dict[str, float]) -> None:
    scores = pd.read_csv(scores_path, sep="\t") if scores_path.exists() else pd.DataFrame([{}])
    if scores.empty:
        scores = pd.DataFrame([{}])

    for name, value in latency_scores.items():
        scores[name] = [_round_number(value)]

    scores.to_csv(scores_path, sep="\t", index=False)


def _merge_metrics(metrics_path: Path, metric_rows: list[dict[str, float | int | None]]) -> None:
    new_metrics = pd.DataFrame(metric_rows)
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path, sep="\t")
        if "index" in metrics.columns and "index" in new_metrics.columns:
            metrics = metrics.drop(
                columns=[column for column in new_metrics.columns if column in metrics.columns and column != "index"]
            )
            merged = metrics.merge(new_metrics, on="index", how="outer")
        else:
            new_metrics = new_metrics.drop(columns=["index"], errors="ignore")
            metrics = metrics.drop(
                columns=[column for column in new_metrics.columns if column in metrics.columns]
            )
            merged = pd.concat([metrics, new_metrics], axis=1)
    else:
        merged = new_metrics

    merged = merged.round(3)
    merged.to_csv(metrics_path, sep="\t", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add ideal and computation-aware SimulEval latency metrics to an output directory."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--eval-latency-unit", default="word", choices=["word", "char", "spm"])
    parser.add_argument("--latency-metrics", nargs="+", default=DEFAULT_LATENCY_METRICS)
    parser.add_argument("--no-use-ref-len", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instances_log = args.output / "instances.log"
    if not instances_log.exists():
        raise FileNotFoundError(f"Missing instances log: {instances_log}")

    instances = _load_instances(instances_log, args.eval_latency_unit)
    latency_scores, metric_rows = _compute_latency_scores(
        instances,
        args.latency_metrics,
        use_ref_len=not args.no_use_ref_len,
    )

    _merge_scores(args.output / "scores.tsv", latency_scores)
    _merge_metrics(args.output / "metrics.tsv", metric_rows)


if __name__ == "__main__":
    main()
