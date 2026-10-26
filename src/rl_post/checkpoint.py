from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import torch

from st_backbone.model import STModel

logger = logging.getLogger(__name__)


def _set_policy_train_mode_for_resume(model: STModel) -> None:
    model.train()
    if model.config.freeze_encoder:
        model.audio_encoder.eval()


def _validation_score_for_checkpoint(metrics: dict[str, float], metric_name: str = "reward") -> float:
    if metric_name in metrics:
        return float(metrics[metric_name])
    if "reward" in metrics:
        return float(metrics["reward"])
    if "bleu" in metrics:
        return float(metrics["bleu"])
    return 0.0
def _maybe_save_best_checkpoint(
    model: STModel,
    out_dir: Path,
    valid_metrics: dict[str, float],
    best_valid_score: float | None,
    step: int,
    *,
    metric_name: str = "reward",
) -> float:
    score = _validation_score_for_checkpoint(valid_metrics, metric_name)
    if best_valid_score is not None and score <= best_valid_score:
        return best_valid_score
    best_dir = out_dir / "best"
    _save_checkpoint(model, best_dir / "model.pt")
    _save_checkpoint_metrics(best_dir / "metrics.json", step, None, valid_metrics)
    logger.info("Saved new best checkpoint at step=%d validation_%s=%.6f.", step, metric_name, score)
    return score
def _save_checkpoint_metrics(
    path: Path,
    step: int,
    train_metrics: dict[str, float] | None,
    valid_metrics: dict[str, float] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "train": train_metrics or {},
        "valid": valid_metrics or {},
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    logger.info("Saved checkpoint metrics: %s", path)
def _maybe_resume_training(
    model: STModel,
    optimizer: torch.optim.Optimizer,
    out_dir: Path,
    cfg: dict[str, Any],
    rng: random.Random,
    device: torch.device,
) -> dict[str, Any] | None:
    resume_mode = str(cfg.get("output", {}).get("resume_mode", "allow")).lower()
    if resume_mode in {"none", "false", "no", "never", "off"}:
        logger.info("GRPO checkpoint auto-resume disabled by output.resume_mode=%s.", resume_mode)
        return None
    if resume_mode not in {"allow", "auto", "true", "yes", "require", "required", "must"}:
        raise ValueError(f"Unsupported output.resume_mode: {resume_mode}")

    state_path = _find_resume_state_path(out_dir, cfg)
    if state_path is None:
        model_path = _find_resume_model_path(out_dir, cfg)
        if model_path is not None:
            model_state = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(model_state, strict=True)
            model.to(device)
            metrics = _load_checkpoint_metrics(model_path.parent / "metrics.json")
            inferred_step = int(metrics.get("step", _checkpoint_step_from_dir(model_path.parent) or 0))
            inferred_rollout = int(metrics.get("train", {}).get("rollout_step", 0)) if isinstance(metrics.get("train"), dict) else 0
            logger.warning(
                "Resumed GRPO model weights from legacy checkpoint %s at step=%d, but optimizer/RNG state was not available.",
                model_path,
                inferred_step,
            )
            _set_policy_train_mode_for_resume(model)
            return {
                "step": inferred_step,
                "rollout_step": inferred_rollout,
                "best_valid_score": None,
                "last_valid_metrics": metrics.get("valid") if isinstance(metrics.get("valid"), dict) else None,
            }
        if resume_mode in {"require", "required", "must"}:
            raise FileNotFoundError(f"No GRPO training checkpoint found under {out_dir} for required resume.")
        logger.info("No GRPO training checkpoint found under %s; starting from model.checkpoint.", out_dir)
        return None

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    model_state = state.get("model")
    if model_state is None:
        raise RuntimeError(f"GRPO training checkpoint has no model state: {state_path}")
    model.load_state_dict(model_state, strict=True)
    model.to(device)

    optimizer_state = state.get("optimizer")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        _move_optimizer_state_to_device(optimizer, device)
    else:
        logger.warning("GRPO training checkpoint has no optimizer state; optimizer will be reinitialized: %s", state_path)

    if "python_random_state" in state:
        random.setstate(state["python_random_state"])
    if "rollout_rng_state" in state:
        rng.setstate(state["rollout_rng_state"])
    if "torch_rng_state" in state:
        torch.random.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("torch_cuda_rng_state_all") is not None:
        try:
            torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])
        except Exception as ex:
            logger.warning("Could not restore CUDA RNG state from %s: %s", state_path, ex)

    logger.info(
        "Resumed GRPO training from %s at step=%d rollout_step=%d best_valid_score=%s.",
        state_path,
        int(state.get("step", 0)),
        int(state.get("rollout_step", 0)),
        state.get("best_valid_score"),
    )
    _set_policy_train_mode_for_resume(model)
    return state
def _find_resume_state_path(out_dir: Path, cfg: dict[str, Any]) -> Path | None:
    resume_from = cfg.get("output", {}).get("resume_from")
    if resume_from:
        path = Path(str(resume_from)).expanduser()
        if path.is_dir():
            path = path / "training_state.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Requested GRPO resume checkpoint not found: {path}")
        return path

    last_state = out_dir / "last" / "training_state.pt"
    if last_state.is_file():
        return last_state

    candidates: list[tuple[int, Path]] = []
    for path in out_dir.glob("step_*/training_state.pt"):
        step = _checkpoint_step_from_dir(path.parent)
        if step is not None:
            candidates.append((step, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]
def _find_resume_model_path(out_dir: Path, cfg: dict[str, Any]) -> Path | None:
    resume_from = cfg.get("output", {}).get("resume_from")
    if resume_from:
        path = Path(str(resume_from)).expanduser()
        if path.is_dir():
            path = path / "model.pt"
        if path.is_file():
            return path
        return None

    last_model = out_dir / "last" / "model.pt"
    if last_model.is_file():
        return last_model

    candidates: list[tuple[int, Path]] = []
    for path in out_dir.glob("step_*/model.pt"):
        step = _checkpoint_step_from_dir(path.parent)
        if step is not None:
            candidates.append((step, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]
def _load_checkpoint_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as ex:
        logger.warning("Could not read checkpoint metrics from %s: %s", path, ex)
        return {}
def _checkpoint_step_from_dir(path: Path) -> int | None:
    name = path.name
    if not name.startswith("step_"):
        return None
    raw_step = name[len("step_") :]
    if not raw_step.isdigit():
        return None
    return int(raw_step)
def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)
def _save_training_checkpoint(
    model: STModel,
    optimizer: torch.optim.Optimizer,
    ckpt_dir: Path,
    *,
    step: int,
    rollout_step: int,
    rng: random.Random,
    train_metrics: dict[str, float] | None,
    last_valid_metrics: dict[str, float] | None,
    best_valid_score: float | None,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _save_checkpoint(model, ckpt_dir / "model.pt")
    _save_checkpoint_metrics(ckpt_dir / "metrics.json", step, train_metrics, last_valid_metrics)

    payload: dict[str, Any] = {
        "version": 1,
        "step": step,
        "rollout_step": rollout_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "python_random_state": random.getstate(),
        "rollout_rng_state": rng.getstate(),
        "torch_rng_state": torch.random.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "best_valid_score": best_valid_score,
        "last_valid_metrics": last_valid_metrics,
        "train_metrics": train_metrics,
    }
    state_path = ckpt_dir / "training_state.pt"
    torch.save(payload, state_path)
    logger.info("Saved GRPO training state: %s", state_path)
def _save_checkpoint(model: STModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info("Saved checkpoint: %s", path)
