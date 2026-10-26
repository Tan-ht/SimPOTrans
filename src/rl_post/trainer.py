from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any, Callable

import torch

from st_backbone.dataset import STTrainDataset
from st_backbone.model import STModel

from .checkpoint import _maybe_resume_training, _maybe_save_best_checkpoint, _save_training_checkpoint
from .common import (
    apply_config_overrides,
    compute_validation_bleu,
    init_wandb,
    load_dataset,
    load_model,
    load_reference_model,
    load_reward_scorer,
    load_validation_dataset,
    load_yaml,
    log_wandb,
    make_progress_bar,
    make_validation_decode_config,
    make_validation_progress_bar,
    mean,
    metric_key,
    normalize_text,
    prefix_metrics,
    rollout_num_tokens,
    sample_num_frames,
    sample_rollout_batch,
    save_config_snapshot,
    save_validation_rows,
    set_policy_train_mode,
    set_progress_position,
    set_progress_postfix,
    should_validate,
    std,
    text_len,
    update_policy,
    update_progress_bar,
    validation_lang_pair_metrics,
    validation_source_metrics,
    resolve_device,
)
from .reward import RewardScorer
from .rollout import (
    _split_chunks,
    _streaming_generate,
    batched_offline_sequence_log_probs,
    batched_sequence_log_probs_full_context,
    collect_offline_rollouts,
    collect_rollouts,
    offline_generate_group,
    offline_rollout_num_tokens,
)

logger = logging.getLogger(__name__)

CollectFn = Callable[[STModel, STModel | None, STTrainDataset, list[dict[str, Any]], RewardScorer, dict[str, Any], int], list[Any]]
ValidateFn = Callable[[STModel, STTrainDataset, RewardScorer, dict[str, Any], int], dict[str, float]]
LogProbFn = Callable[[STModel, list[Any], dict[str, Any]], list[torch.Tensor]]
TokenCountFn = Callable[[Any], int]


def _rollout_metrics(rollouts: list[Any], token_count_fn: TokenCountFn = rollout_num_tokens) -> dict[str, float]:
    rewards = [r.reward for r in rollouts]
    advantages = [r.advantage for r in rollouts]
    hyp_lens = [text_len(r.hypothesis, str(r.sample.get("tgt_lang", ""))) for r in rollouts]
    token_lens = [token_count_fn(r) for r in rollouts]
    metrics = {
        "reward_mean": mean(rewards),
        "reward_max": max(rewards) if rewards else 0.0,
        "reward_min": min(rewards) if rewards else 0.0,
        "reward_std": std(rewards),
        "advantage_mean": mean(advantages),
        "advantage_std": std(advantages),
        "hyp_len_mean": mean(hyp_lens),
        "token_len_mean": mean(token_lens),
        "num_rollouts": float(len(rollouts)),
    }
    component_values: dict[str, list[float]] = {}
    for rollout in rollouts:
        for name, value in getattr(rollout, "reward_components", {}).items():
            component_values.setdefault(str(name), []).append(float(value))
    metrics.update({f"reward/{name}": mean(values) for name, values in component_values.items()})
    return metrics


def _rollout_lang_pair_metrics(rollouts: list[Any]) -> dict[str, float]:
    grouped_rewards: dict[str, list[float]] = {}
    for rollout in rollouts:
        lang_pair = f"{rollout.sample.get('src_lang', '')}-{rollout.sample.get('tgt_lang', '')}"
        grouped_rewards.setdefault(lang_pair, []).append(rollout.reward)

    return {
        f"reward_mean/{metric_key(lang_pair)}": mean(rewards)
        for lang_pair, rewards in grouped_rewards.items()
    }


def _source_key(sample: dict[str, Any]) -> str:
    dataset = str(sample.get("dataset", "") or "unknown")
    lang_pair = str(sample.get("lang_pair", ""))
    if not lang_pair:
        lang_pair = f"{sample.get('src_lang', '')}-{sample.get('tgt_lang', '')}"
    return f"{dataset}:{lang_pair}"


def _limit_validation_samples(samples: list[dict[str, Any]], limit: Any) -> list[dict[str, Any]]:
    if limit is None:
        return samples
    max_samples = int(limit)
    if max_samples <= 0:
        return []
    if len(samples) <= max_samples:
        return samples

    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(_source_key(sample), []).append(sample)

    selected: list[dict[str, Any]] = []
    group_values = list(grouped.values())
    offset = 0
    while len(selected) < max_samples:
        added = False
        for group in group_values:
            if offset < len(group):
                selected.append(group[offset])
                added = True
                if len(selected) >= max_samples:
                    break
        if not added:
            break
        offset += 1

    return selected


@torch.no_grad()
def validate_streaming(
    model: STModel,
    dataset: STTrainDataset,
    reward_scorer: RewardScorer,
    cfg: dict[str, Any],
    step: int,
) -> dict[str, float]:
    valid_cfg = cfg.get("validation", {})
    model.eval()

    samples = _limit_validation_samples(list(dataset._samples), valid_cfg.get("num_samples"))

    decode_cfg = make_validation_decode_config(cfg)
    rows: list[dict[str, Any]] = []
    valid_progress = make_validation_progress_bar(cfg, len(samples), step)

    try:
        for sample in samples:
            prepared = dataset._prepare_example(sample)
            chunks = _split_chunks(
                prepared["source"],
                float(valid_cfg.get("chunk_size_seconds", cfg["rollout"].get("chunk_size_seconds", 1.0))),
                int(cfg["dataset"].get("sample_rate", 16000)),
            )
            _, hypothesis, _ = _streaming_generate(
                model,
                chunks,
                prepared["src_lang"],
                prepared["tgt_lang"],
                decode_cfg,
                cfg,
            )
            rows.append(
                {
                    "id": prepared["id"],
                    "src": prepared.get("source_text", ""),
                    "mt": hypothesis,
                    "ref": prepared["target_text"],
                    "src_lang": prepared["src_lang"],
                    "tgt_lang": prepared["tgt_lang"],
                    "dataset": sample.get("dataset", ""),
                    "manifest_root": sample.get("manifest_root", ""),
                }
            )
            if valid_progress is not None:
                valid_progress.update(1)
    finally:
        if valid_progress is not None:
            valid_progress.close()

    return _score_and_save_validation_rows(model, rows, reward_scorer, cfg, step, label="validation")


@torch.no_grad()
def validate_offline(
    model: STModel,
    dataset: STTrainDataset,
    reward_scorer: RewardScorer,
    cfg: dict[str, Any],
    step: int,
) -> dict[str, float]:
    valid_cfg = cfg.get("validation", {})
    model.eval()

    samples = _limit_validation_samples(list(dataset._samples), valid_cfg.get("num_samples"))

    decode_cfg = make_validation_decode_config(cfg)
    rows: list[dict[str, Any]] = []
    valid_progress = make_validation_progress_bar(cfg, len(samples), step)

    try:
        for sample in samples:
            prepared = dataset._prepare_example(sample)
            generated = offline_generate_group(model, prepared, decode_cfg, group_size=1)
            hypothesis = generated[0][1] if generated else ""
            rows.append(
                {
                    "id": prepared["id"],
                    "src": prepared.get("source_text", ""),
                    "mt": normalize_text(hypothesis, prepared["tgt_lang"]),
                    "ref": prepared["target_text"],
                    "src_lang": prepared["src_lang"],
                    "tgt_lang": prepared["tgt_lang"],
                    "dataset": sample.get("dataset", ""),
                    "manifest_root": sample.get("manifest_root", ""),
                }
            )
            if valid_progress is not None:
                valid_progress.update(1)
    finally:
        if valid_progress is not None:
            valid_progress.close()

    return _score_and_save_validation_rows(model, rows, reward_scorer, cfg, step, label="offline validation")


def _score_and_save_validation_rows(
    model: STModel,
    rows: list[dict[str, Any]],
    reward_scorer: RewardScorer,
    cfg: dict[str, Any],
    step: int,
    *,
    label: str,
) -> dict[str, float]:
    if rows:
        logger.info("%s generated %d hypothesis/hypotheses; scoring with %s.", label, len(rows), reward_scorer.name)
        scores = reward_scorer.score_texts(
            [str(row["src"]) for row in rows],
            [str(row["mt"]) for row in rows],
            [str(row["ref"]) for row in rows],
        )
        for row, score in zip(rows, scores):
            row[reward_scorer.name] = score

    bleu_metrics = compute_validation_bleu(rows)
    out_dir = Path(cfg["output"]["dir"]).expanduser()
    save_validation_rows(rows, out_dir / "validation" / f"step_{step}.jsonl")

    if rows:
        logger.info("%s hypothesis: %s", label, rows[0]["mt"])
        logger.info("%s reference: %s", label, rows[0]["ref"])

    scorer_values = [float(row[reward_scorer.name]) for row in rows if reward_scorer.name in row]
    reward_score = mean(scorer_values)
    metrics = {
        "num_samples": float(len(rows)),
        "reward": reward_score,
        reward_scorer.name: reward_score,
        **bleu_metrics,
        **validation_lang_pair_metrics(rows, reward_scorer.name),
        **validation_source_metrics(rows, reward_scorer.name),
    }

    logger.info(
        "%s step=%d %s=%.4f bleu=%.4f n=%d",
        label,
        step,
        reward_scorer.name,
        metrics[reward_scorer.name],
        metrics.get("bleu", 0.0),
        len(rows),
    )
    set_policy_train_mode(model)
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["streaming", "offline"], default=None, help="Training mode. Defaults to rollout.mode from config, then streaming.")
    parser.add_argument("--set", action="append", default=[], help="Override config with dotted key=value, e.g. rollout.max_steps=100.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_from_args(args, default_mode="streaming")


def train_from_args(args: argparse.Namespace, *, default_mode: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = load_yaml(args.config)
    apply_config_overrides(cfg, args.set)
    mode = str(args.mode or cfg.get("rollout", {}).get("mode", default_mode)).lower()
    cfg.setdefault("rollout", {})["mode"] = mode

    if mode == "offline":
        train_grpo(
            cfg,
            mode="offline",
            dataset_label="rl_post_offline",
            validation_label="rl_post_offline_validation",
            progress_desc="Offline GRPO",
            rollout_phase="offline_rollout",
            collect_fn=collect_offline_rollouts,
            log_prob_fn=batched_offline_sequence_log_probs,
            token_count_fn=offline_rollout_num_tokens,
            validate_fn=validate_offline,
            log_prefix="offline ",
        )
    elif mode in {"streaming", "grpo", "online"}:
        train_grpo(
            cfg,
            mode="streaming",
            dataset_label="rl_post",
            validation_label="rl_post_validation",
            progress_desc="GRPO",
            rollout_phase="rollout",
            collect_fn=collect_rollouts,
            log_prob_fn=batched_sequence_log_probs_full_context,
            token_count_fn=rollout_num_tokens,
            validate_fn=validate_streaming,
            log_prefix="",
        )
    else:
        raise ValueError(f"Unsupported rollout.mode: {mode}")


def train_grpo(
    cfg: dict[str, Any],
    *,
    mode: str,
    dataset_label: str,
    validation_label: str,
    progress_desc: str,
    rollout_phase: str,
    collect_fn: CollectFn,
    log_prob_fn: LogProbFn,
    token_count_fn: TokenCountFn,
    validate_fn: ValidateFn,
    log_prefix: str,
) -> None:
    seed = int(cfg["rollout"].get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)

    device = resolve_device(cfg["model"].get("device", "cuda"))
    model = load_model(cfg, device)
    use_reference_model = bool(cfg["grpo"].get("reference_model"))
    if not use_reference_model and float(cfg["grpo"].get("kl_coef", 0.0)) > 0.0:
        logger.warning("grpo.reference_model=false; forcing grpo.kl_coef=0.0 because KL needs reference log-probs.")
        cfg["grpo"]["kl_coef"] = 0.0
    ref_model = load_reference_model(cfg, device) if use_reference_model else None
    dataset = load_dataset(cfg, label=dataset_label)
    valid_dataset = load_validation_dataset(cfg, label=validation_label) if bool(cfg.get("validation", {}).get("enabled", False)) else None
    reward_scorer = load_reward_scorer(cfg)
    validation_reward_scorer = reward_scorer.validation_scorer()
    wandb_run = init_wandb(cfg)
    set_policy_train_mode(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_cfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(optimizer_cfg.get("lr", 1e-5)),
        betas=tuple(float(x) for x in optimizer_cfg.get("betas", [0.9, 0.95])),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.01)),
    )

    out_dir = Path(cfg["output"]["dir"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(cfg, out_dir)

    samples = dataset._get_reader_samples(is_train=True, seed=seed)
    rng = random.Random(seed)
    step = 0
    rollout_step = 0
    best_valid_score: float | None = None
    best_model_metric = str(cfg.get("reward", {}).get("best_model_metric", "reward"))
    last_valid_metrics: dict[str, float] | None = None

    try:
        resume_state = _maybe_resume_training(model, optimizer, out_dir, cfg, rng, device)
        if resume_state is not None:
            step = int(resume_state.get("step", 0))
            rollout_step = int(resume_state.get("rollout_step", 0))
            best_valid_score = resume_state.get("best_valid_score")
            last_valid_metrics = resume_state.get("last_valid_metrics")

        progress = make_progress_bar(cfg, desc=progress_desc)
        set_progress_position(progress, step)

        if step == 0 and valid_dataset is not None and bool(cfg["validation"].get("at_start", False)):
            valid_metrics = validate_fn(model, valid_dataset, validation_reward_scorer, cfg, step)
            last_valid_metrics = valid_metrics
            best_valid_score = _maybe_save_best_checkpoint(
                model,
                out_dir,
                valid_metrics,
                best_valid_score,
                step,
                metric_name=best_model_metric,
            )
            log_wandb(wandb_run, prefix_metrics("valid", valid_metrics), step)

        max_update_steps = int(cfg["rollout"].get("max_steps", 1000))
        if step >= max_update_steps:
            logger.warning(
                "%straining already at step=%d >= max_steps=%d; exiting without new rollouts. "
                "Set RESUME_MODE=never or use a fresh OUTPUT_DIR to start a new run.",
                log_prefix,
                step,
                max_update_steps,
            )
        while step < max_update_steps:
            rollout_step += 1
            batch_samples = sample_rollout_batch(samples, rng, cfg)
            set_progress_postfix(progress, phase=rollout_phase, samples=len(batch_samples), step=step)
            rollouts = collect_fn(model, ref_model, dataset, batch_samples, reward_scorer, cfg, rollout_step, progress=progress)
            update_metrics = update_policy(
                model,
                optimizer,
                rollouts,
                cfg,
                current_step=step,
                max_steps=max_update_steps,
                log_prob_fn=log_prob_fn,
                token_count_fn=token_count_fn,
                progress=progress,
            )
            num_updates = int(update_metrics.pop("num_updates", 0))
            if num_updates == 0:
                logger.warning("No optimizer update was performed for %s rollout step %d.", mode, rollout_step)
                continue

            step += num_updates
            train_metrics = {**update_metrics, **_rollout_metrics(rollouts, token_count_fn)}
            train_metrics.update(_rollout_lang_pair_metrics(rollouts))
            train_metrics["num_updates"] = float(num_updates)
            train_metrics["rollout_step"] = float(rollout_step)
            train_metrics["batch_num_samples"] = float(len(batch_samples))
            train_metrics["batch_source_frames"] = float(sum(sample_num_frames(sample) for sample in batch_samples))

            if step % int(cfg["output"].get("log_every_steps", 1)) == 0:
                logger.info(
                    "%sstep=%d rollout=%d updates=%d samples=%d frames=%.0f reward=%.4f/%.4f loss=%.4f pg=%.4f kl=%.4f hyp_len=%.1f hyp=%s",
                    log_prefix,
                    step,
                    rollout_step,
                    num_updates,
                    int(train_metrics["batch_num_samples"]),
                    train_metrics["batch_source_frames"],
                    train_metrics["reward_mean"],
                    train_metrics["reward_max"],
                    update_metrics["loss"],
                    update_metrics["pg_loss"],
                    update_metrics["kl"],
                    train_metrics["hyp_len_mean"],
                    rollouts[0].hypothesis[:160] if rollouts else "",
                )
                log_wandb(wandb_run, prefix_metrics("train", train_metrics), step)

            update_progress_bar(progress, train_metrics)

            if step % int(cfg["output"].get("save_every_steps", 100)) == 0:
                _save_training_checkpoint_pair(
                    model,
                    optimizer,
                    out_dir,
                    step=step,
                    rollout_step=rollout_step,
                    rng=rng,
                    train_metrics=train_metrics,
                    last_valid_metrics=last_valid_metrics,
                    best_valid_score=best_valid_score,
                )

            if should_validate(valid_dataset, cfg, step):
                valid_metrics = validate_fn(model, valid_dataset, validation_reward_scorer, cfg, step)
                last_valid_metrics = valid_metrics
                best_valid_score = _maybe_save_best_checkpoint(
                    model,
                    out_dir,
                    valid_metrics,
                    best_valid_score,
                    step,
                    metric_name=best_model_metric,
                )
                log_wandb(wandb_run, prefix_metrics("valid", valid_metrics), step)

        _save_training_checkpoint(
            model,
            optimizer,
            out_dir / "last",
            step=step,
            rollout_step=rollout_step,
            rng=rng,
            train_metrics=None,
            last_valid_metrics=last_valid_metrics,
            best_valid_score=best_valid_score,
        )
    finally:
        if 'progress' in locals() and progress is not None:
            progress.close()
        reward_scorer.close()
        if wandb_run is not None:
            wandb_run.finish()


def _save_training_checkpoint_pair(
    model: STModel,
    optimizer: torch.optim.Optimizer,
    out_dir: Path,
    *,
    step: int,
    rollout_step: int,
    rng: random.Random,
    train_metrics: dict[str, float] | None,
    last_valid_metrics: dict[str, float] | None,
    best_valid_score: float | None,
) -> None:
    for ckpt_dir in (out_dir / f"step_{step}", out_dir / "last"):
        _save_training_checkpoint(
            model,
            optimizer,
            ckpt_dir,
            step=step,
            rollout_step=rollout_step,
            rng=rng,
            train_metrics=train_metrics,
            last_valid_metrics=last_valid_metrics,
            best_valid_score=best_valid_score,
        )
