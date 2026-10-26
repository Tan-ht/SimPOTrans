from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import StoppingCriteriaList

from st_backbone.dataset import STTrainDataset
from st_backbone.model import (
    DecodeConfig,
    SpeechEncoderNotReadyError,
    STModel,
    StopOnTokens,
    StreamingSTState,
)

from .reward import RewardScorer

logger = logging.getLogger(__name__)


@dataclass
class Rollout:
    sample: dict[str, Any]
    chunks: list[Tensor]
    generated_ids: list[Tensor]
    old_log_probs: Tensor
    ref_log_probs: Tensor | None
    reward: float
    advantage: float
    hypothesis: str
    chunk_texts: list[str] = field(default_factory=list)
    reward_components: dict[str, float] = field(default_factory=dict)


def _make_decode_config(cfg: dict[str, Any], *, seed: int) -> DecodeConfig:
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


def _set_policy_train_mode(model: STModel) -> None:
    model.train()
    if model.config.freeze_encoder:
        model.audio_encoder.eval()


def _set_progress_postfix(progress: Any | None, **values: Any) -> None:
    if progress is not None:
        progress.set_postfix(**values)


def _split_chunks(waveform: Tensor, chunk_seconds: float, sample_rate: int) -> list[Tensor]:
    chunk_samples = max(1, int(round(chunk_seconds * sample_rate)))
    return [waveform[start : start + chunk_samples] for start in range(0, waveform.numel(), chunk_samples)]


def _normalize_text(text: str, tgt_lang: str) -> str:
    text = text.strip()
    if tgt_lang in {"ja", "ko", "zh", "zh-cn", "zh-tw"}:
        return "".join(text.split())
    return " ".join(text.split())


def _join_streaming_text(texts: list[str], tgt_lang: str) -> str:
    texts = [text for text in texts if text]
    if tgt_lang in {"ja", "ko", "zh", "zh-cn", "zh-tw"}:
        return "".join(texts)
    return " ".join(texts)


def _text_len(text: str, lang: str) -> int:
    if lang in {"ja", "ko", "zh", "zh-cn", "zh-tw"}:
        return len("".join(text.split()))
    return len(text.split())


def _mean(values: list[float] | list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return float((sum((float(value) - mean) ** 2 for value in values) / len(values)) ** 0.5)


def collect_rollouts(
    model: STModel,
    ref_model: STModel | None,
    dataset: STTrainDataset,
    samples: list[dict[str, Any]],
    reward_scorer: RewardScorer,
    cfg: dict[str, Any],
    step: int,
    *,
    progress: Any | None = None,
) -> list[Rollout]:
    model.eval()
    group_size = int(cfg["rollout"].get("group_size", 4))
    log_progress = bool(cfg.get("output", {}).get("log_rollout_progress", False))
    rollout_started_at = time.monotonic()
    all_rollouts: list[Rollout] = []
    groups: list[list[Rollout]] = []

    for sample_idx, sample in enumerate(samples):
        sample_started_at = time.monotonic()
        prepared = dataset._prepare_example(sample)
        chunks = _split_chunks(
            prepared["source"],
            float(cfg["rollout"].get("chunk_size_seconds", 1.0)),
            int(cfg["dataset"].get("sample_rate", 16000)),
        )
        lang_pair = f"{prepared['src_lang']}-{prepared['tgt_lang']}"
        if log_progress:
            logger.info(
                "rollout step=%d sample=%d/%d id=%s lang=%s frames=%d chunks=%d",
                step,
                sample_idx + 1,
                len(samples),
                prepared.get("id", ""),
                lang_pair,
                int(prepared.get("source_len", 0)),
                len(chunks),
            )
        _set_progress_postfix(
            progress,
            phase="rollout_generate",
            sample=sample_idx + 1,
            samples=len(samples),
            lang=lang_pair,
            chunks=len(chunks),
        )
        group: list[Rollout] = []
        if group_size > 1:
            candidate_started_at = time.monotonic()
            _set_progress_postfix(
                progress,
                phase="rollout_generate_group",
                sample=sample_idx + 1,
                candidates=group_size,
                lang=lang_pair,
            )
            decode_cfg = _make_decode_config(cfg, seed=step * 1009)
            generated_group = _streaming_generate_group_batched(
                model,
                chunks,
                prepared["src_lang"],
                prepared["tgt_lang"],
                decode_cfg,
                cfg,
                group_size=group_size,
            )
            group_elapsed = time.monotonic() - candidate_started_at
            for group_idx, (generated_ids, hypothesis, chunk_texts) in enumerate(generated_group):
                if log_progress:
                    logger.info(
                        "rollout step=%d sample=%d/%d candidate=%d/%d generated tokens=%d hyp_len=%d group_elapsed=%.1fs",
                        step,
                        sample_idx + 1,
                        len(samples),
                        group_idx + 1,
                        group_size,
                        sum(int(ids.numel()) for ids in generated_ids),
                        _text_len(hypothesis, prepared["tgt_lang"]),
                        group_elapsed,
                    )
                group.append(
                    Rollout(
                        sample=prepared,
                        chunks=chunks,
                        generated_ids=generated_ids,
                        old_log_probs=torch.empty(0),
                        ref_log_probs=None,
                        reward=0.0,
                        advantage=0.0,
                        hypothesis=hypothesis,
                        chunk_texts=chunk_texts,
                    )
                )
        else:
            for group_idx in range(group_size):
                candidate_started_at = time.monotonic()
                _set_progress_postfix(
                    progress,
                    phase="rollout_generate",
                    sample=sample_idx + 1,
                    candidate=group_idx + 1,
                    candidates=group_size,
                    lang=lang_pair,
                )
                decode_cfg = _make_decode_config(cfg, seed=step * 1009 + group_idx)
                generated_ids, hypothesis, chunk_texts = _streaming_generate(
                    model,
                    chunks,
                    prepared["src_lang"],
                    prepared["tgt_lang"],
                    decode_cfg,
                    cfg,
                )
                if log_progress:
                    logger.info(
                        "rollout step=%d sample=%d/%d candidate=%d/%d generated tokens=%d hyp_len=%d elapsed=%.1fs",
                        step,
                        sample_idx + 1,
                        len(samples),
                        group_idx + 1,
                        group_size,
                        sum(int(ids.numel()) for ids in generated_ids),
                        _text_len(hypothesis, prepared["tgt_lang"]),
                        time.monotonic() - candidate_started_at,
                    )
                group.append(
                    Rollout(
                        sample=prepared,
                        chunks=chunks,
                        generated_ids=generated_ids,
                        old_log_probs=torch.empty(0),
                        ref_log_probs=None,
                        reward=0.0,
                        advantage=0.0,
                        hypothesis=hypothesis,
                        chunk_texts=chunk_texts,
                    )
                )

        groups.append(group)
        all_rollouts.extend(group)
        if log_progress:
            logger.info(
                "rollout step=%d sample=%d/%d done elapsed=%.1fs",
                step,
                sample_idx + 1,
                len(samples),
                time.monotonic() - sample_started_at,
            )

    _set_progress_postfix(progress, phase="rollout_logprob", rollouts=len(all_rollouts))
    if log_progress:
        logger.info("rollout step=%d refreshing policy/ref logprobs for %d rollout(s).", step, len(all_rollouts))
    logprob_started_at = time.monotonic()
    _refresh_rollout_log_probs(model, ref_model, all_rollouts, cfg)
    if log_progress:
        logger.info("rollout step=%d logprob refresh done elapsed=%.1fs", step, time.monotonic() - logprob_started_at)

    _set_progress_postfix(progress, phase="rollout_reward", rollouts=len(all_rollouts))
    if log_progress:
        logger.info("rollout step=%d scoring rewards for %d rollout(s).", step, len(all_rollouts))
    reward_started_at = time.monotonic()
    rewards = reward_scorer.score_rollouts(all_rollouts)
    if log_progress:
        logger.info("rollout step=%d reward scoring done elapsed=%.1fs", step, time.monotonic() - reward_started_at)
    for rollout, reward in zip(all_rollouts, rewards):
        rollout.reward = reward

    for group in groups:
        _assign_group_advantages(group, cfg)

    if log_progress:
        logger.info("rollout step=%d complete rollouts=%d elapsed=%.1fs", step, len(all_rollouts), time.monotonic() - rollout_started_at)
    _set_policy_train_mode(model)
    return all_rollouts
def _refresh_rollout_log_probs(
    model: STModel,
    ref_model: STModel | None,
    rollouts: list[Rollout],
    cfg: dict[str, Any],
) -> None:
    logprob_micro_batch_size = max(1, int(cfg["grpo"].get("logprob_micro_batch_size", 4)))
    with torch.no_grad():
        for batch_start in range(0, len(rollouts), logprob_micro_batch_size):
            batch = rollouts[batch_start : batch_start + logprob_micro_batch_size]
            old_log_probs = batched_sequence_log_probs_full_context(model, batch, cfg)
            for rollout, log_probs in zip(batch, old_log_probs):
                rollout.old_log_probs = log_probs.detach().cpu()

            if ref_model is not None:
                ref_log_probs = batched_sequence_log_probs_full_context(ref_model, batch, cfg)
                for rollout, log_probs in zip(batch, ref_log_probs):
                    rollout.ref_log_probs = log_probs.detach().cpu()
def _assign_group_advantages(group: list[Rollout], cfg: dict[str, Any]) -> None:
    rewards = torch.tensor([r.reward for r in group], dtype=torch.float32)
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    eps = float(cfg["grpo"].get("advantage_eps", 1e-6))
    advantages = (rewards - mean) / (std + eps)
    for rollout, advantage in zip(group, advantages.tolist()):
        rollout.advantage = float(advantage)
def batched_sequence_log_probs_full_context(
    model: STModel,
    rollouts: list[Rollout],
    cfg: dict[str, Any],
) -> list[Tensor]:
    sequences: list[Tensor] = []
    target_positions: list[Tensor] = []
    target_ids: list[Tensor] = []

    for rollout in rollouts:
        sequence, positions, ids = _build_full_context_teacher_forcing_sequence(
            model,
            rollout,
            cfg,
        )
        sequences.append(sequence)
        target_positions.append(positions)
        target_ids.append(ids)

    if not sequences:
        return []

    max_len = max(sequence.size(0) for sequence in sequences)
    hidden_size = sequences[0].size(-1)
    batch_size = len(sequences)
    inputs_embeds = sequences[0].new_zeros((batch_size, max_len, hidden_size))
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=model.device)

    for index, sequence in enumerate(sequences):
        seq_len = sequence.size(0)
        inputs_embeds[index, :seq_len] = sequence
        attention_mask[index, :seq_len] = 1

    outputs = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )

    results: list[Tensor] = []
    for index, (positions, ids) in enumerate(zip(target_positions, target_ids)):
        if ids.numel() == 0:
            results.append(torch.empty(0, device=model.device))
            continue
        logits = outputs.logits[index, positions.to(model.device), :]
        log_probs = F.log_softmax(logits, dim=-1).gather(
            -1,
            ids.to(model.device).unsqueeze(-1),
        ).squeeze(-1)
        results.append(log_probs)

    return results
def _build_full_context_teacher_forcing_sequence(
    model: STModel,
    rollout: Rollout,
    cfg: dict[str, Any],
) -> tuple[Tensor, Tensor, Tensor]:
    src_lang = rollout.sample["src_lang"]
    tgt_lang = rollout.sample["tgt_lang"]
    pieces: list[Tensor] = []
    target_positions: list[Tensor] = []
    target_ids: list[Tensor] = []

    system_embeds = _streaming_system_embeds(model, src_lang, tgt_lang)
    if system_embeds is not None:
        pieces.append(system_embeds.squeeze(0))

    speech_context: Tensor | None = None
    speech_cache = model.init_speech_encoder_cache_state()
    try:
        for chunk_idx, (chunk, turn_ids) in enumerate(zip(rollout.chunks, rollout.generated_ids)):
            chunk = chunk.to(model.device)
            speech_context = chunk if speech_context is None else torch.cat([speech_context, chunk], dim=0)
            speech_embeds = model.forward_adaptor(
                model.encode_speech_context(
                    [speech_context],
                    int(chunk.numel()),
                    encoder_cache_state=speech_cache,
                    is_final=chunk_idx == len(rollout.chunks) - 1,
                )
            )
            turn_embeds = model._streaming_turn_embeds(speech_embeds, src_lang, tgt_lang).squeeze(0)
            pieces.append(turn_embeds)

            token_ids = turn_ids.to(model.device).view(-1)
            if token_ids.numel() > 0:
                start = sum(piece.size(0) for piece in pieces) - 1
                target_positions.append(
                    torch.arange(start, start + token_ids.numel(), dtype=torch.long)
                )
                target_ids.append(token_ids)
                token_embeds = model.llm.get_input_embeddings()(token_ids.view(1, -1)).squeeze(0)
                pieces.append(token_embeds)

            newline_embeds = _text_embeds(model, "\n")
            if newline_embeds is not None:
                pieces.append(newline_embeds.squeeze(0))
    finally:
        model.clear_speech_encoder_cache_state(speech_cache)

    if not pieces:
        hidden_size = model.llm.get_input_embeddings().embedding_dim
        sequence = torch.empty((0, hidden_size), device=model.device)
    else:
        sequence = torch.cat(pieces, dim=0)

    positions = torch.cat(target_positions, dim=0) if target_positions else torch.empty(0, dtype=torch.long)
    ids = torch.cat(target_ids, dim=0) if target_ids else torch.empty(0, dtype=torch.long, device=model.device)
    return sequence, positions, ids
def _streaming_system_embeds(model: STModel, src_lang: str, tgt_lang: str) -> Tensor | None:
    system_messages, _ = model._render_streaming_messages(src_lang, tgt_lang)
    if not system_messages:
        return None
    tokenized = model.llm_tokenizer.apply_chat_template(
        system_messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        return_tensors="pt",
    ).to(model.device)
    if tokenized.numel() == 0:
        return None
    return model.llm.get_input_embeddings()(tokenized)
def _text_embeds(model: STModel, text: str) -> Tensor | None:
    token_ids = model.llm_tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(model.device)
    if token_ids.numel() == 0:
        return None
    return model.llm.get_input_embeddings()(token_ids)
def sequence_log_probs(
    model: STModel,
    chunks: list[Tensor],
    generated_ids: list[Tensor],
    src_lang: str,
    tgt_lang: str,
    cfg: dict[str, Any],
    *,
    require_grad: bool,
) -> Tensor:
    context = torch.enable_grad() if require_grad else torch.no_grad()
    collected: list[Tensor] = []

    with torch.no_grad():
        state = model.init_streaming_state(src_lang=src_lang, tgt_lang=tgt_lang)
        speech_cache = model.init_speech_encoder_cache_state()

    speech_context: Tensor | None = None
    try:
        for chunk_idx, (chunk, turn_ids) in enumerate(zip(chunks, generated_ids)):
            if turn_ids.numel() == 0:
                continue

            chunk = chunk.to(model.device)
            speech_context = chunk if speech_context is None else torch.cat([speech_context, chunk], dim=0)

            with context:
                speech_embeds = model.forward_adaptor(
                    model.encode_speech_context(
                        [speech_context],
                        int(chunk.numel()),
                        encoder_cache_state=speech_cache,
                        is_final=chunk_idx == len(chunks) - 1,
                    )
                )
                turn_embeds = model._streaming_turn_embeds(speech_embeds, src_lang, tgt_lang)
                token_ids = turn_ids.to(model.device).view(1, -1)
                token_embeds = model.llm.get_input_embeddings()(token_ids)
                inputs_embeds = torch.cat([turn_embeds, token_embeds], dim=1)
                outputs = model.llm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=_cache_attention_mask_for(model, state, inputs_embeds.size(1)),
                    cache_position=_cache_position_for(model, state, inputs_embeds.size(1)),
                    past_key_values=state.past_key_values,
                    use_cache=False,
                    return_dict=True,
                )
                start = turn_embeds.size(1) - 1
                logits = outputs.logits[:, start : start + token_ids.size(1), :]
                log_probs = F.log_softmax(logits, dim=-1).gather(
                    -1, token_ids.unsqueeze(-1)
                ).squeeze(0).squeeze(-1)
                collected.append(log_probs)

            with torch.no_grad():
                detached_turn = turn_embeds.detach()
                model._append_embeds_to_streaming_state(state, detached_turn)
                model._append_token_ids_to_streaming_state(state, token_ids.detach())
                model._append_text_to_streaming_state(state, "\n")
    finally:
        model.clear_speech_encoder_cache_state(speech_cache)

    if not collected:
        return torch.empty(0, device=model.device)

    return torch.cat(collected, dim=0)
@torch.no_grad()
def _streaming_generate_group_batched(
    model: STModel,
    chunks: list[Tensor],
    src_lang: str,
    tgt_lang: str,
    decode_cfg: DecodeConfig,
    cfg: dict[str, Any],
    *,
    group_size: int,
) -> list[tuple[list[Tensor], str, list[str]]]:
    if group_size <= 1:
        generated_ids, hypothesis, chunk_texts = _streaming_generate(model, chunks, src_lang, tgt_lang, decode_cfg, cfg)
        return [(generated_ids, hypothesis, chunk_texts)]

    speech_cache = model.init_speech_encoder_cache_state()
    system_embeds = _streaming_system_embeds(model, src_lang, tgt_lang)
    turn_history: list[Tensor] = []
    generated_history: list[list[Tensor]] = [[] for _ in range(group_size)]
    text_history: list[list[str]] = [[] for _ in range(group_size)]
    speech_context: Tensor | None = None

    try:
        for chunk_idx, chunk in enumerate(chunks):
            chunk = chunk.to(model.device)
            speech_context = chunk if speech_context is None else torch.cat([speech_context, chunk], dim=0)
            speech_embeds = model.forward_adaptor(
                model.encode_speech_context(
                    [speech_context],
                    int(chunk.numel()),
                    encoder_cache_state=speech_cache,
                    is_final=chunk_idx == len(chunks) - 1,
                )
            )
            turn_embeds = _streaming_turn_embeds_for_speech(model, speech_embeds, src_lang, tgt_lang)

            sequences = [
                _build_generation_context_embeds(
                    model,
                    system_embeds,
                    turn_history,
                    generated_history[candidate_idx],
                    turn_embeds,
                )
                for candidate_idx in range(group_size)
            ]
            inputs_embeds, attention_mask = _left_pad_embed_sequences(sequences, model.device)

            if decode_cfg.do_sample:
                torch.manual_seed(decode_cfg.seed + len(turn_history))
            outputs = model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                eos_token_id=[model.eos_token_id],
                pad_token_id=model.llm_tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([StopOnTokens([model.eos_token_id])]),
                max_new_tokens=decode_cfg.max_new_tokens,
                num_beams=decode_cfg.beam_num,
                do_sample=decode_cfg.do_sample,
                min_length=decode_cfg.min_length,
                return_dict_in_generate=True,
                output_logits=False,
                use_cache=True,
                **_decode_generation_kwargs_for_rl(decode_cfg),
            )

            sequences_out = outputs.sequences.detach().cpu()
            for candidate_idx in range(group_size):
                ids = sequences_out[candidate_idx].view(-1)
                generated_history[candidate_idx].append(ids)
                text_history[candidate_idx].append(
                    _normalize_text(model.decode(ids.unsqueeze(0))[0], tgt_lang)
                )
            turn_history.append(turn_embeds)
    finally:
        model.clear_speech_encoder_cache_state(speech_cache)

    return [
        (
            generated_history[candidate_idx],
            _join_streaming_text(text_history[candidate_idx], tgt_lang),
            text_history[candidate_idx],
        )
        for candidate_idx in range(group_size)
    ]
def _build_generation_context_embeds(
    model: STModel,
    system_embeds: Tensor | None,
    turn_history: list[Tensor],
    generated_history: list[Tensor],
    current_turn_embeds: Tensor,
) -> Tensor:
    pieces: list[Tensor] = []
    if system_embeds is not None:
        pieces.append(system_embeds.squeeze(0))

    for turn_embeds, token_ids in zip(turn_history, generated_history):
        pieces.append(turn_embeds.squeeze(0))
        token_embeds = _ids_embeds(model, token_ids)
        if token_embeds is not None:
            pieces.append(token_embeds.squeeze(0))
        newline_embeds = _text_embeds(model, "\n")
        if newline_embeds is not None:
            pieces.append(newline_embeds.squeeze(0))

    pieces.append(current_turn_embeds.squeeze(0))
    return torch.cat(pieces, dim=0)
def _ids_embeds(model: STModel, token_ids: Tensor) -> Tensor | None:
    if token_ids.numel() == 0:
        return None
    return model.llm.get_input_embeddings()(token_ids.to(model.device).view(1, -1))
def _left_pad_embed_sequences(sequences: list[Tensor], device: torch.device) -> tuple[Tensor, Tensor]:
    max_len = max(sequence.size(0) for sequence in sequences)
    hidden_size = sequences[0].size(-1)
    batch_size = len(sequences)
    inputs_embeds = sequences[0].new_zeros((batch_size, max_len, hidden_size), device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    for index, sequence in enumerate(sequences):
        seq_len = sequence.size(0)
        start = max_len - seq_len
        inputs_embeds[index, start:] = sequence
        attention_mask[index, start:] = 1
    return inputs_embeds, attention_mask
def _streaming_turn_embeds_for_speech(
    model: STModel,
    speech_embeds: Tensor,
    src_lang: str,
    tgt_lang: str,
) -> Tensor:
    system_messages, user_message = model._render_streaming_messages(src_lang, tgt_lang)
    if user_message is None:
        raise ValueError("Streaming prompt must contain one user message with the audio placeholder.")

    full_ids = model.llm_tokenizer.apply_chat_template(
        [*system_messages, user_message],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to(model.device)
    if full_ids.ndim == 2:
        full_ids = full_ids[0]

    if system_messages:
        system_ids = model.llm_tokenizer.apply_chat_template(
            system_messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
            return_tensors="pt",
        ).to(model.device)
        if system_ids.ndim == 2:
            system_ids = system_ids[0]
        if full_ids.numel() >= system_ids.numel() and torch.equal(full_ids[: system_ids.numel()], system_ids):
            full_ids = full_ids[system_ids.numel() :]
        else:
            logger.warning("Could not strip cached system prompt from batched streaming turn.")

    prompt_ids = full_ids.unsqueeze(0)
    tokenized_prompt = type("TokenizedPrompt", (), {})()
    tokenized_prompt.input_ids = prompt_ids
    tokenized_prompt.attention_mask = torch.ones_like(prompt_ids)
    return model.wrap_prompt(speech_embeds, tokenized_prompt)
def _decode_generation_kwargs_for_rl(cfg: DecodeConfig) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if cfg.do_sample:
        kwargs["temperature"] = cfg.temperature
        if cfg.top_p is not None:
            kwargs["top_p"] = cfg.top_p
        if cfg.top_k is not None:
            kwargs["top_k"] = cfg.top_k
    return kwargs
@torch.no_grad()
def _streaming_generate(
    model: STModel,
    chunks: list[Tensor],
    src_lang: str,
    tgt_lang: str,
    decode_cfg: DecodeConfig,
    cfg: dict[str, Any],
) -> tuple[list[Tensor], str, list[str]]:
    state = model.init_streaming_state(src_lang=src_lang, tgt_lang=tgt_lang)
    speech_cache = model.init_speech_encoder_cache_state()
    speech_context: Tensor | None = None
    generated_ids: list[Tensor] = []
    texts: list[str] = []

    try:
        for chunk_idx, chunk in enumerate(chunks):
            chunk = chunk.to(model.device)
            speech_context = chunk if speech_context is None else torch.cat([speech_context, chunk], dim=0)
            try:
                ids, state = model.generate_streaming_turn(
                    speech_context,
                    state,
                    current_num_samples=int(chunk.numel()),
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    decode_config=decode_cfg,
                    speech_encoder_cache_state=speech_cache,
                    is_final=chunk_idx == len(chunks) - 1,
                )
            except SpeechEncoderNotReadyError:
                continue
            ids = ids.detach().view(-1).cpu()
            generated_ids.append(ids)
            texts.append(_normalize_text(model.decode(ids.unsqueeze(0))[0], tgt_lang))
    finally:
        model.clear_speech_encoder_cache_state(speech_cache)

    return generated_ids, _join_streaming_text(texts, tgt_lang), texts
def _cache_attention_mask_for(model: STModel, state: StreamingSTState, seq_len: int) -> Tensor:
    past_len = _past_key_values_len(state.past_key_values)
    return torch.ones((1, past_len + seq_len), dtype=torch.long, device=model.device)
def _cache_position_for(model: STModel, state: StreamingSTState, seq_len: int) -> Tensor:
    past_len = _past_key_values_len(state.past_key_values)
    return torch.arange(past_len, past_len + seq_len, dtype=torch.long, device=model.device)
def _past_key_values_len(past_key_values: Any | None) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    first_layer = past_key_values[0]
    first_key = first_layer[0] if isinstance(first_layer, (tuple, list)) else first_layer.key_cache
    return int(first_key.shape[-2])
def _sample_rollout_batch(
    samples: list[dict[str, Any]],
    rng: random.Random,
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
        frames = _sample_num_frames(sample)
        if batch and (len(batch) >= max_examples or total_frames + frames > frame_budget):
            continue

        batch.append(sample)
        total_frames += frames
        if len(batch) >= max_examples or total_frames >= frame_budget:
            break

    if not batch:
        batch.append(shuffled[0])

    return batch
def _sample_num_frames(sample: dict[str, Any]) -> int:
    for key in ("n_frames", "source_len"):
        value = sample.get(key)
        if value is not None:
            return max(1, int(value))
    return 1
def _should_validate(dataset: STTrainDataset | None, cfg: dict[str, Any], step: int) -> bool:
    if dataset is None:
        return False
    every = int(cfg.get("validation", {}).get("every_steps", 0))
    return every > 0 and step % every == 0


# Offline full-utterance GRPO rollout utilities.
@dataclass
class OfflineRollout:
    sample: dict[str, Any]
    generated_ids: list[Tensor]
    old_log_probs: Tensor
    ref_log_probs: Tensor | None
    reward: float
    advantage: float
    hypothesis: str
    reward_components: dict[str, float] = field(default_factory=dict)


def collect_offline_rollouts(
    model: STModel,
    ref_model: STModel | None,
    dataset: STTrainDataset,
    samples: list[dict[str, Any]],
    reward_scorer: RewardScorer,
    cfg: dict[str, Any],
    step: int,
    *,
    progress: Any | None = None,
) -> list[OfflineRollout]:
    model.eval()
    group_size = int(cfg["rollout"].get("group_size", 4))
    log_progress = bool(cfg.get("output", {}).get("log_rollout_progress", False))
    rollout_started_at = time.monotonic()
    all_rollouts: list[OfflineRollout] = []
    groups: list[list[OfflineRollout]] = []

    for sample_idx, sample in enumerate(samples):
        sample_started_at = time.monotonic()
        prepared = dataset._prepare_example(sample)
        lang_pair = f"{prepared['src_lang']}-{prepared['tgt_lang']}"
        _set_progress_postfix(
            progress,
            phase="offline_generate",
            sample=sample_idx + 1,
            samples=len(samples),
            lang=lang_pair,
            candidates=group_size,
        )
        if log_progress:
            logger.info(
                "offline rollout step=%d sample=%d/%d id=%s lang=%s frames=%d candidates=%d",
                step,
                sample_idx + 1,
                len(samples),
                prepared.get("id", ""),
                lang_pair,
                int(prepared.get("source_len", 0)),
                group_size,
            )

        decode_cfg = _make_decode_config(cfg, seed=step * 1009 + sample_idx)
        generated_group = offline_generate_group(model, prepared, decode_cfg, group_size=group_size)
        group: list[OfflineRollout] = []
        for group_idx, (token_ids, hypothesis) in enumerate(generated_group):
            if log_progress:
                logger.info(
                    "offline rollout step=%d sample=%d/%d candidate=%d/%d tokens=%d hyp_len=%d",
                    step,
                    sample_idx + 1,
                    len(samples),
                    group_idx + 1,
                    group_size,
                    int(token_ids.numel()),
                    _text_len(hypothesis, prepared["tgt_lang"]),
                )
            group.append(
                OfflineRollout(
                    sample=prepared,
                    generated_ids=[token_ids],
                    old_log_probs=torch.empty(0),
                    ref_log_probs=None,
                    reward=0.0,
                    advantage=0.0,
                    hypothesis=hypothesis,
                )
            )

        groups.append(group)
        all_rollouts.extend(group)
        if log_progress:
            logger.info(
                "offline rollout step=%d sample=%d/%d done elapsed=%.1fs",
                step,
                sample_idx + 1,
                len(samples),
                time.monotonic() - sample_started_at,
            )

    _set_progress_postfix(progress, phase="offline_logprob", rollouts=len(all_rollouts))
    _refresh_offline_rollout_log_probs(model, ref_model, all_rollouts, cfg)

    _set_progress_postfix(progress, phase="offline_reward", rollouts=len(all_rollouts))
    rewards = reward_scorer.score_rollouts(all_rollouts)
    for rollout, reward in zip(all_rollouts, rewards):
        rollout.reward = reward

    for group in groups:
        assign_group_advantages(group, cfg)

    if log_progress:
        logger.info("offline rollout step=%d complete rollouts=%d elapsed=%.1fs", step, len(all_rollouts), time.monotonic() - rollout_started_at)
    _set_policy_train_mode(model)
    return all_rollouts


@torch.no_grad()
def offline_generate_group(
    model: STModel,
    prepared: dict[str, Any],
    decode_cfg: DecodeConfig,
    *,
    group_size: int,
) -> list[tuple[Tensor, str]]:
    group_size = max(1, int(group_size))
    source = prepared["source"]
    sources = [source for _ in range(group_size)]
    src_langs = [prepared["src_lang"] for _ in range(group_size)]
    tgt_langs = [prepared["tgt_lang"] for _ in range(group_size)]

    tokens = model.generate(
        sources,
        src_langs=src_langs,
        tgt_langs=tgt_langs,
        decode_config=decode_cfg,
    ).detach().cpu()

    results: list[tuple[Tensor, str]] = []
    for row in tokens:
        ids = _trim_generated_ids(model, row.view(-1))
        hypothesis = _normalize_text(model.decode(ids.unsqueeze(0))[0], prepared["tgt_lang"])
        results.append((ids, hypothesis))
    return results


def _trim_generated_ids(model: STModel, token_ids: Tensor) -> Tensor:
    token_ids = token_ids.detach().cpu().view(-1)
    pad_id = model.llm_tokenizer.pad_token_id
    eos_id = model.eos_token_id
    kept: list[int] = []
    for token_id in token_ids.tolist():
        token = int(token_id)
        if token == eos_id:
            kept.append(token)
            break
        if pad_id is not None and token == int(pad_id):
            continue
        kept.append(token)
    return torch.tensor(kept, dtype=torch.long)


def _refresh_offline_rollout_log_probs(
    model: STModel,
    ref_model: STModel | None,
    rollouts: list[OfflineRollout],
    cfg: dict[str, Any],
) -> None:
    logprob_micro_batch_size = max(1, int(cfg["grpo"].get("logprob_micro_batch_size", 4)))
    with torch.no_grad():
        for batch_start in range(0, len(rollouts), logprob_micro_batch_size):
            batch = rollouts[batch_start : batch_start + logprob_micro_batch_size]
            old_log_probs = batched_offline_sequence_log_probs(model, batch, cfg)
            for rollout, log_probs in zip(batch, old_log_probs):
                rollout.old_log_probs = log_probs.detach().cpu()

            if ref_model is not None:
                ref_log_probs = batched_offline_sequence_log_probs(ref_model, batch, cfg)
                for rollout, log_probs in zip(batch, ref_log_probs):
                    rollout.ref_log_probs = log_probs.detach().cpu()


def assign_group_advantages(group: list[OfflineRollout], cfg: dict[str, Any]) -> None:
    rewards = torch.tensor([r.reward for r in group], dtype=torch.float32)
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    eps = float(cfg["grpo"].get("advantage_eps", 1e-6))
    advantages = (rewards - mean) / (std + eps)
    for rollout, advantage in zip(group, advantages.tolist()):
        rollout.advantage = float(advantage)


def batched_offline_sequence_log_probs(
    model: STModel,
    rollouts: list[OfflineRollout],
    cfg: dict[str, Any],
) -> list[Tensor]:
    sequences: list[Tensor] = []
    target_positions: list[Tensor] = []
    target_ids: list[Tensor] = []

    for rollout in rollouts:
        sequence, positions, ids = _build_offline_teacher_forcing_sequence(model, rollout)
        sequences.append(sequence)
        target_positions.append(positions)
        target_ids.append(ids)

    if not sequences:
        return []

    max_len = max(sequence.size(0) for sequence in sequences)
    hidden_size = sequences[0].size(-1)
    batch_size = len(sequences)
    inputs_embeds = sequences[0].new_zeros((batch_size, max_len, hidden_size))
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=model.device)

    for index, sequence in enumerate(sequences):
        seq_len = sequence.size(0)
        inputs_embeds[index, :seq_len] = sequence
        attention_mask[index, :seq_len] = 1

    outputs = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )

    results: list[Tensor] = []
    for index, (positions, ids) in enumerate(zip(target_positions, target_ids)):
        if ids.numel() == 0:
            results.append(torch.empty(0, device=model.device))
            continue
        logits = outputs.logits[index, positions.to(model.device), :]
        log_probs = F.log_softmax(logits, dim=-1).gather(
            -1,
            ids.to(model.device).unsqueeze(-1),
        ).squeeze(-1)
        results.append(log_probs)

    return results


def _build_offline_teacher_forcing_sequence(
    model: STModel,
    rollout: OfflineRollout,
) -> tuple[Tensor, Tensor, Tensor]:
    sample = rollout.sample
    token_ids = rollout.generated_ids[0].to(model.device).view(-1)
    if token_ids.numel() == 0:
        hidden_size = model.llm.get_input_embeddings().embedding_dim
        return (
            torch.empty((0, hidden_size), device=model.device),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long, device=model.device),
        )

    speech_embeds = model.forward_adaptor(model.encode_speech([sample["source"].to(model.device)]))
    prompts = model._render_prompts([sample["src_lang"]], [sample["tgt_lang"]])
    tokenized = model._tokenize_prompts(prompts)
    wrapped_embeds = model.wrap_prompt(speech_embeds, tokenized).squeeze(0)
    token_embeds = model.llm.get_input_embeddings()(token_ids.view(1, -1)).squeeze(0)

    sequence = torch.cat([wrapped_embeds, token_embeds], dim=0)
    start = wrapped_embeds.size(0) - 1
    positions = torch.arange(start, start + token_ids.numel(), dtype=torch.long)
    return sequence, positions, token_ids


def offline_rollout_num_tokens(rollout: OfflineRollout) -> int:
    return sum(int(ids.numel()) for ids in rollout.generated_ids)
