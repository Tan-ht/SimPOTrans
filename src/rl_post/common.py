from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor
import yaml

from st_backbone.dataset import STTrainDataset, STTrainDatasetConfig
from st_backbone.model import (
    DecodeConfig,
    LoraConfigSection,
    STModel,
    STModelConfig,
    _configure_trainable_parameters,
    create_st_model,
)

from .reward import CompositeRewardScorer, RewardScorer, XCometRewardScorer

logger = logging.getLogger(__name__)


def metric_key(raw: str) -> str:
    return raw.replace("/", "_").replace(" ", "_") or "unknown"


def mean(values: list[float] | list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def std(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    value_mean = mean(values)
    return float((sum((float(value) - value_mean) ** 2 for value in values) / len(values)) ** 0.5)


def prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def normalize_text(text: str, tgt_lang: str) -> str:
    text = text.strip()
    if tgt_lang in {"ja", "ko", "zh", "zh-cn", "zh-tw"}:
        return "".join(text.split())
    return " ".join(text.split())


def text_len(text: str, lang: str) -> int:
    if lang in {"ja", "ko", "zh", "zh-cn", "zh-tw"}:
        return len("".join(text.split()))
    return len(text.split())


def load_yaml(path: Path) -> dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config_snapshot(cfg: dict[str, Any], out_dir: Path) -> None:
    with (out_dir / "config.resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def apply_config_overrides(cfg: dict[str, Any], overrides: list[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Config override must be key=value: {override}")
        key, raw_value = override.split("=", 1)
        target: dict[str, Any] = cfg
        parts = key.split(".")
        if any(not part for part in parts):
            raise ValueError(f"Invalid config override key: {key}")
        for part in parts[:-1]:
            child = target.get(part)
            if child is None:
                child = {}
                target[part] = child
            if not isinstance(child, dict):
                raise ValueError(f"Cannot override nested key under non-dict config value: {part}")
            target = child
        target[parts[-1]] = yaml.safe_load(raw_value)


def make_progress_bar(cfg: dict[str, Any], *, desc: str = "GRPO") -> Any | None:
    output_cfg = cfg.get("output", {})
    if not bool(output_cfg.get("progress_bar", True)):
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        logger.warning("tqdm is not installed; progress bar disabled.")
        return None
    return tqdm(
        total=int(cfg["rollout"].get("max_steps", 1000)),
        desc=desc,
        dynamic_ncols=True,
    )


def make_validation_progress_bar(cfg: dict[str, Any], total: int, step: int) -> Any | None:
    output_cfg = cfg.get("output", {})
    if not bool(output_cfg.get("progress_bar", True)):
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm(
        total=total,
        desc=f"validation@{step}",
        dynamic_ncols=True,
        leave=False,
    )


def set_progress_postfix(progress: Any | None, **values: Any) -> None:
    if progress is not None:
        progress.set_postfix(**values)


def set_progress_position(progress: Any | None, step: int) -> None:
    if progress is None or step <= 0:
        return
    progress.n = step
    progress.refresh()


def update_progress_bar(progress: Any | None, metrics: dict[str, float]) -> None:
    if progress is None:
        return
    progress.set_postfix(
        reward=f"{metrics.get('reward_mean', 0.0):.3f}",
        loss=f"{metrics.get('loss', 0.0):.3f}",
        kl=f"{metrics.get('kl', 0.0):.3f}",
        samples=int(metrics.get("batch_num_samples", 0.0)),
    )


def init_wandb(cfg: dict[str, Any]) -> Any | None:
    wandb_cfg = cfg.get("wandb", {})
    if not bool(wandb_cfg.get("enabled", False)):
        return None
    try:
        import wandb
    except ImportError as ex:
        raise RuntimeError("wandb logging is enabled, but `wandb` is not installed.") from ex

    return wandb.init(
        entity=wandb_cfg.get("entity"),
        project=wandb_cfg.get("project", "SimPOTrans"),
        name=wandb_cfg.get("run_name"),
        group=wandb_cfg.get("group"),
        job_type=wandb_cfg.get("job_type", "rl_post"),
        id=wandb_cfg.get("run_id"),
        resume=wandb_cfg.get("resume", "allow"),
        config=cfg,
    )


def log_wandb(run: Any | None, metrics: dict[str, float], step: int) -> None:
    if run is not None:
        run.log(metrics, step=step)


def load_model(cfg: dict[str, Any], device: torch.device) -> STModel:
    model_cfg = load_model_config(
        Path(cfg["model"]["config_path"]),
        cfg.get("model", {}).get("config_overrides"),
    )
    model_cfg.torch_dtype = str(cfg["model"].get("dtype", model_cfg.torch_dtype))
    model = create_st_model(model_cfg)
    checkpoint = cfg["model"].get("checkpoint")
    if checkpoint:
        state = torch.load(resolve_checkpoint_path(Path(checkpoint)), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        logger.info("Loaded SFT checkpoint: %s", checkpoint)
    _configure_trainable_parameters(model)
    model.to(device)
    return model


def load_reference_model(cfg: dict[str, Any], device: torch.device) -> STModel:
    ref_model = load_model(copy.deepcopy(cfg), device)
    ref_model.requires_grad_(False)
    ref_model.eval()
    return ref_model


def load_reward_scorer(cfg: dict[str, Any]) -> RewardScorer:
    scorer = str(cfg.get("reward", {}).get("scorer", "xcomet")).lower()
    if scorer in {"xcomet", "xcomet-xl", "comet"}:
        logger.info("Using persistent XCOMET reward scorer.")
        return XCometRewardScorer(cfg)
    if scorer in {"composite", "composite_reward"}:
        mode = str(cfg.get("rollout", {}).get("mode", "streaming")).lower()
        if mode == "offline":
            raise ValueError("Composite reward is only supported for streaming RL; offline RL must use reward.scorer: xcomet.")
        logger.info("Using composite reward scorer for streaming training; validation uses XCOMET.")
        return CompositeRewardScorer(cfg)
    raise ValueError("Unsupported rl_post reward scorer: %s. Use xcomet or composite." % scorer)


def set_policy_train_mode(model: STModel) -> None:
    model.train()
    if model.config.freeze_encoder:
        model.audio_encoder.eval()


def load_model_config(
    path: Path, runtime_overrides: dict[str, Any] | None = None
) -> STModelConfig:
    with path.expanduser().open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    overrides = dict(raw.get("model", raw).get("config_overrides", raw.get("model", raw)) or {})
    if runtime_overrides:
        overrides.update(dict(runtime_overrides))
    if overrides.get("lora") is not None:
        overrides["lora"] = LoraConfigSection(**overrides["lora"])
    if overrides.get("decode") is not None:
        overrides["decode"] = DecodeConfig(**overrides["decode"])
    for key in ("prompt_path", "decode_config_path"):
        if overrides.get(key) is not None:
            overrides[key] = Path(overrides[key])
    return STModelConfig(**overrides)


def load_dataset(cfg: dict[str, Any], *, label: str = "rl_post") -> STTrainDataset:
    dataset = STTrainDataset(dataset_config_from(cfg["dataset"]))
    dataset.log_summary(label)
    return dataset


def load_validation_dataset(cfg: dict[str, Any], *, label: str = "rl_post_validation") -> STTrainDataset:
    valid_cfg = cfg.get("validation", {})
    raw = dict(cfg["dataset"])
    raw["splits"] = valid_cfg.get("splits", raw.get("valid_splits", "dev_st"))
    if valid_cfg.get("max_sentences_per_lang_pair") is not None:
        raw["max_sentences_per_lang_pair"] = valid_cfg["max_sentences_per_lang_pair"]
    dataset = STTrainDataset(dataset_config_from(raw))
    dataset.log_summary(label)
    return dataset


def dataset_config_from(raw_cfg: dict[str, Any]) -> STTrainDatasetConfig:
    raw = dict(raw_cfg)
    for key in ("manifest_root", "audio_root", "manifest_list"):
        if raw.get(key) is not None:
            raw[key] = Path(raw[key])
    return STTrainDatasetConfig(**raw)


def make_rollout_decode_config(cfg: dict[str, Any], *, seed: int) -> DecodeConfig:
    rollout = cfg["rollout"]
    return DecodeConfig(
        max_new_tokens=int(rollout.get("max_new_tokens", 64)),
        do_sample=True,
        beam_num=1,
        top_p=rollout.get("top_p"),
        top_k=rollout.get("top_k"),
        temperature=float(rollout.get("temperature", 0.7)),
        min_length=1,
        seed=seed,
    )


def make_validation_decode_config(cfg: dict[str, Any]) -> DecodeConfig:
    rollout = cfg["rollout"]
    valid = cfg.get("validation", {})
    do_sample = bool(valid.get("do_sample", False))
    temperature = float(valid.get("temperature", 0.7 if do_sample else 0.0))
    return DecodeConfig(
        max_new_tokens=int(valid.get("max_new_tokens", rollout.get("max_new_tokens", 64))),
        do_sample=do_sample,
        beam_num=int(valid.get("beam_num", 1)),
        top_p=valid.get("top_p", rollout.get("top_p") if do_sample else None),
        top_k=valid.get("top_k", rollout.get("top_k") if do_sample else None),
        temperature=temperature,
        min_length=1,
        seed=int(valid.get("seed", rollout.get("seed", 0))),
    )


def resolve_checkpoint_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_file():
        return path
    model_pt = path / "model.pt"
    if model_pt.is_file():
        return model_pt
    candidates = sorted(path.glob("model/pp_*/tp_*/sdp_*.pt"))
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise RuntimeError(f"Multiple checkpoint shards under {path}; pass the shard path.")
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def resolve_device(raw: str) -> torch.device:
    if raw == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        raw = "cpu"
    return torch.device(raw)


def sample_rollout_batch(
    samples: list[dict[str, Any]],
    rng: Any,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    if not samples:
        return []

    rollout_cfg = cfg["rollout"]
    max_examples = max(1, int(rollout_cfg.get("batch_size", 1)))
    max_frames = rollout_cfg.get("max_source_frames_per_batch")
    if max_frames is None:
        return rng.sample(samples, k=min(max_examples, len(samples)))

    frame_budget = max(1, int(max_frames))
    shuffled = list(samples)
    rng.shuffle(shuffled)

    batch: list[dict[str, Any]] = []
    total_frames = 0
    for sample in shuffled:
        frames = sample_num_frames(sample)
        if batch and (len(batch) >= max_examples or total_frames + frames > frame_budget):
            continue

        batch.append(sample)
        total_frames += frames
        if len(batch) >= max_examples or total_frames >= frame_budget:
            break

    if not batch:
        batch.append(shuffled[0])

    return batch


def sample_num_frames(sample: dict[str, Any]) -> int:
    for key in ("n_frames", "source_len"):
        value = sample.get(key)
        if value is not None:
            return max(1, int(value))
    return 1


def should_validate(dataset: STTrainDataset | None, cfg: dict[str, Any], step: int) -> bool:
    if dataset is None:
        return False
    every = int(cfg.get("validation", {}).get("every_steps", 0))
    return every > 0 and step % every == 0


def validation_lang_pair_metrics(rows: list[dict[str, Any]], reward_name: str) -> dict[str, float]:
    grouped_rewards: dict[str, list[float]] = {}
    for row in rows:
        if reward_name not in row:
            continue
        lang_pair = f"{row.get('src_lang', '')}-{row.get('tgt_lang', '')}"
        grouped_rewards.setdefault(lang_pair, []).append(float(row[reward_name]))

    return {
        f"reward/{metric_key(lang_pair)}": mean(rewards)
        for lang_pair, rewards in grouped_rewards.items()
    }


def validation_source_metrics(rows: list[dict[str, Any]], reward_name: str) -> dict[str, float]:
    grouped_rewards: dict[str, list[float]] = {}
    for row in rows:
        if reward_name not in row:
            continue
        dataset = str(row.get("dataset", "") or "unknown")
        lang_pair = f"{row.get('src_lang', '')}-{row.get('tgt_lang', '')}"
        grouped_rewards.setdefault(f"{dataset}:{lang_pair}", []).append(float(row[reward_name]))

    return {
        f"reward_source/{metric_key(source)}": mean(rewards)
        for source, rewards in grouped_rewards.items()
    }


def save_validation_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Saved validation predictions: %s", path)


def compute_validation_bleu(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"bleu": 0.0}
    hyps = [str(row["mt"]) for row in rows]
    refs = [str(row["ref"]) for row in rows]
    return {"bleu": compute_bleu_score(hyps, refs)}


def compute_bleu_score(hyps: list[str], refs: list[str]) -> float:
    if not hyps:
        return 0.0
    try:
        import sacrebleu

        return float(sacrebleu.corpus_bleu(hyps, [refs]).score)
    except Exception:
        return fallback_corpus_bleu(hyps, refs)


def fallback_corpus_bleu(hyps: list[str], refs: list[str]) -> float:
    if not hyps:
        return 0.0
    scores = []
    for hyp, ref in zip(hyps, refs):
        hyp_tokens = hyp.split()
        ref_tokens = ref.split()
        if not hyp_tokens or not ref_tokens:
            scores.append(0.0)
            continue
        overlap = len(set(hyp_tokens) & set(ref_tokens))
        scores.append(100.0 * overlap / max(len(hyp_tokens), 1))
    return mean(scores)


# Shared GRPO policy update.

LogProbFn = Callable[[STModel, list[Any], dict[str, Any]], list[Tensor]]
TokenCountFn = Callable[[Any], int]


def rollout_num_tokens(rollout: Any) -> int:
    generated_ids = getattr(rollout, "generated_ids", [])
    if torch.is_tensor(generated_ids):
        return int(generated_ids.numel())
    return sum(int(ids.numel()) for ids in generated_ids)


def update_policy(
    model: STModel,
    optimizer: torch.optim.Optimizer,
    rollouts: list[Any],
    cfg: dict[str, Any],
    *,
    current_step: int,
    max_steps: int,
    log_prob_fn: LogProbFn,
    token_count_fn: TokenCountFn = rollout_num_tokens,
    progress: Any | None = None,
) -> dict[str, float]:
    update_epochs = int(cfg["grpo"].get("update_epochs", 20))
    clip_range = float(cfg["grpo"].get("clip_range", 0.2))
    kl_coef = float(cfg["grpo"].get("kl_coef", 0.0))
    max_grad_norm = float(cfg["optimizer"].get("max_grad_norm", 1.0))
    valid_rollouts = [rollout for rollout in rollouts if token_count_fn(rollout) > 0]
    num_valid_rollouts = len(valid_rollouts)
    logprob_micro_batch_size = max(1, int(cfg["grpo"].get("logprob_micro_batch_size", 4)))

    num_updates = 0
    last_metrics = {"loss": 0.0, "pg_loss": 0.0, "kl": 0.0, "num_updates": 0.0}
    if num_valid_rollouts == 0:
        return last_metrics

    for update_epoch in range(update_epochs):
        if current_step + num_updates >= max_steps:
            break
        set_progress_postfix(
            progress,
            phase="update",
            step=current_step + num_updates + 1,
            update_epoch=update_epoch + 1,
            update_epochs=update_epochs,
        )
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        pg_loss_sum = 0.0
        kl_sum = 0.0
        ratio_sum = 0.0
        ratio_count = 0
        ratio_max = 0.0
        clip_count = 0
        approx_kl_sum = 0.0
        positive_advantage_count = 0

        for batch_start in range(0, num_valid_rollouts, logprob_micro_batch_size):
            batch = valid_rollouts[batch_start : batch_start + logprob_micro_batch_size]
            set_progress_postfix(
                progress,
                phase="update",
                step=current_step + num_updates + 1,
                update_epoch=update_epoch + 1,
                update_epochs=update_epochs,
                rollout=min(batch_start + len(batch), num_valid_rollouts),
                rollouts=num_valid_rollouts,
            )
            batch_log_probs = log_prob_fn(model, batch, cfg)

            batch_losses = []
            batch_pg_losses = []
            batch_kls = []
            for rollout, log_probs in zip(batch, batch_log_probs):
                if log_probs.numel() == 0:
                    continue

                old_log_probs = rollout.old_log_probs.to(log_probs.device)
                log_ratio = log_probs - old_log_probs
                ratio = torch.exp(log_ratio)
                ratio_detached = ratio.detach()
                ratio_sum += float(ratio_detached.sum().cpu())
                ratio_count += int(ratio_detached.numel())
                ratio_max = max(ratio_max, float(ratio_detached.max().cpu()))
                clip_count += int(((ratio_detached < 1.0 - clip_range) | (ratio_detached > 1.0 + clip_range)).sum().cpu())
                approx_kl_sum += float(((ratio_detached - 1.0) - log_ratio.detach()).sum().cpu())
                advantage = torch.tensor(rollout.advantage, device=log_probs.device)
                if rollout.advantage > 0.0:
                    positive_advantage_count += 1
                unclipped = ratio * advantage
                clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantage
                pg_loss = -torch.minimum(unclipped, clipped).mean()

                kl = torch.zeros((), device=log_probs.device)
                if kl_coef > 0.0 and rollout.ref_log_probs is not None:
                    ref_log_probs = rollout.ref_log_probs.to(log_probs.device)
                    kl = 0.5 * (log_probs - ref_log_probs).pow(2).mean()

                loss = pg_loss + kl_coef * kl
                batch_losses.append(loss)
                batch_pg_losses.append(pg_loss)
                batch_kls.append(kl)

            if not batch_losses:
                continue

            micro_loss = torch.stack(batch_losses).sum() / num_valid_rollouts
            micro_loss.backward()

            loss_sum += sum(float(loss.detach().cpu()) for loss in batch_losses)
            pg_loss_sum += sum(float(loss.detach().cpu()) for loss in batch_pg_losses)
            kl_sum += sum(float(loss.detach().cpu()) for loss in batch_kls)

            del batch_log_probs, batch_losses, batch_pg_losses, batch_kls, micro_loss

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_grad_norm,
        )
        optimizer.step()
        num_updates += 1

        last_metrics = {
            "loss": loss_sum / num_valid_rollouts,
            "pg_loss": pg_loss_sum / num_valid_rollouts,
            "kl": kl_sum / num_valid_rollouts,
            "num_updates": float(num_updates),
            "clip_frac": float(clip_count / max(ratio_count, 1)),
            "approx_kl_policy_old": float(approx_kl_sum / max(ratio_count, 1)),
            "ratio_mean": float(ratio_sum / max(ratio_count, 1)),
            "ratio_max": float(ratio_max),
            "positive_advantage_frac": float(positive_advantage_count / max(num_valid_rollouts, 1)),
        }
        if progress is not None:
            progress.update(1)
            progress.set_postfix(
                phase="update",
                step=current_step + num_updates,
                update_epoch=update_epoch + 1,
                update_epochs=update_epochs,
                loss=f"{last_metrics['loss']:.3f}",
                kl=f"{last_metrics['kl']:.3f}",
            )

    return last_metrics
