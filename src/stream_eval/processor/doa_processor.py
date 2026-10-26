from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from transformers import StoppingCriteriaList

from simulstream.server.speech_processors import SAMPLE_RATE
from simulstream.server.speech_processors.incremental_output import (
    IncrementalOutput,
)

from st_backbone.model import (
    AUDIO_PLACEHOLDER_TOKEN,
    StopOnTokens,
    _decode_generation_kwargs,
)
from stream_eval.processor.base_processor import (
    _empty_output,
    _normalize_incremental_text,
)
from stream_eval.processor.fixed_processor_prefix import (
    SimPOTransChunkProcessor as _FixedPrefixChunkProcessor,
)

logger = logging.getLogger(__name__)
_DEFAULT_AUDIO_FRAME_SAMPLES = 320


class DOAChunkProcessor(_FixedPrefixChunkProcessor):
    """Fixed-prefix SimPOTrans processor gated by decoder-only attention alignments."""

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__(config)
        self.doa_cutoff_frames = int(getattr(config, "doa_cutoff_frames", 15))
        if self.doa_cutoff_frames < 0:
            raise ValueError("doa_cutoff_frames must be non-negative.")
        self.doa_layers = _normalize_index_selection(getattr(config, "doa_layers", 15))
        self.doa_heads = _normalize_index_selection(getattr(config, "doa_heads", None))
        self.soft_speech_context_reset_seconds = float(
            getattr(config, "soft_speech_context_reset_seconds", 20.0)
        )
        self._history_audio_end_frames: list[int | None] = []

    @classmethod
    def load_model(cls, config: SimpleNamespace) -> None:
        super().load_model(config)
        if not bool(getattr(config, "doa_force_eager_attention", True)):
            return
        if cls._model is None:
            return

        llm_config = getattr(cls._model.llm, "config", None)
        if llm_config is None:
            return
        if hasattr(llm_config, "_attn_implementation"):
            llm_config._attn_implementation = "eager"
        if hasattr(llm_config, "attn_implementation"):
            llm_config.attn_implementation = "eager"
        logger.info("Configured LLM attention implementation for DOA attention reads.")

    @torch.no_grad()
    def process_chunk(self, waveform: np.float32) -> IncrementalOutput:
        if self._model is None or self._device is None:
            raise RuntimeError("SimPOTrans model is not loaded.")

        if waveform.size == 0:
            return _empty_output()

        audio = torch.from_numpy(
            np.asarray(waveform, dtype=np.float32),
        ).to(self._device)
        chunk_seconds = float(audio.numel()) / SAMPLE_RATE

        forced_output = self._force_write_before_hard_reset_if_needed(
            chunk_seconds=chunk_seconds,
        )
        if forced_output is not None:
            self._reset_prompt_context()

        self._append_prefix_speech_chunk(audio, chunk_seconds)
        if forced_output is not None and forced_output.new_tokens:
            return forced_output

        return self._generate_doa_turn(cutoff_frames=self.doa_cutoff_frames)

    @torch.no_grad()
    def end_of_stream(self) -> IncrementalOutput:
        if self._model is None or self._speech_context_audio is None:
            return _empty_output()
        return self._generate_doa_turn(cutoff_frames=0)

    def clear(self) -> None:
        super().clear()
        self._history_audio_end_frames.clear()

    def _reset_prompt_context(self) -> None:
        super()._reset_prompt_context()
        self._history_audio_end_frames.clear()

    def _append_prefix_speech_chunk(
        self,
        audio: torch.Tensor,
        chunk_seconds: float,
    ) -> None:
        self._speech_context_audio = (
            audio
            if self._speech_context_audio is None
            else torch.cat([self._speech_context_audio, audio], dim=0)
        )
        self._speech_context_seconds += chunk_seconds

    def _force_write_before_hard_reset_if_needed(
        self,
        *,
        chunk_seconds: float,
    ) -> IncrementalOutput | None:
        if self._speech_context_audio is None:
            return None
        if self._speech_context_seconds + chunk_seconds <= self.max_speech_context_seconds:
            return None

        logger.debug(
            "Force-writing DOA prefix context before hard reset: "
            "speech_context_seconds=%.3f, next_chunk_seconds=%.3f, "
            "max_speech_context_seconds=%.3f, history_turns=%d.",
            self._speech_context_seconds,
            chunk_seconds,
            self.max_speech_context_seconds,
            len(self._history_text),
        )
        return self._generate_doa_turn(cutoff_frames=0, force_write=True)

    def _generate_doa_turn(
        self,
        *,
        cutoff_frames: int,
        force_write: bool = False,
    ) -> IncrementalOutput:
        if self._model is None or self._device is None or self._speech_context_audio is None:
            return _empty_output()

        tokens, turn_embeds, audio_start, audio_end = self._generate_candidate_turn()
        if tokens.numel() == 0:
            return _empty_output()

        alignments = self._token_audio_alignments(
            turn_embeds=turn_embeds,
            tokens=tokens,
            audio_start=audio_start,
            audio_end=audio_end,
        )
        if force_write:
            stable_len = int(tokens.size(1))
            logger.debug("DOA force-write decision: committing %d tokens.", stable_len)
        else:
            stable_len = _stable_prefix_len_from_alignments(
                alignments,
                audio_len=audio_end - audio_start,
                cutoff_frames=cutoff_frames,
            )
        if stable_len <= 0:
            logger.debug(
                "DOA read decision: no stable tokens with cutoff_frames=%d.",
                cutoff_frames,
            )
            return _empty_output()

        committed_tokens = tokens[:, :stable_len]
        text = _normalize_incremental_text(
            self._model.decode(committed_tokens)[0],
            self.target_language,
        )
        if not text:
            return _empty_output()

        committed_alignments = alignments[:stable_len]
        self._commit_prefix_text(text, committed_alignments)

        return IncrementalOutput(
            new_tokens=[text],
            new_string=text,
            deleted_tokens=[],
            deleted_string="",
        )

    def _commit_prefix_text(self, text: str, alignments: list[int]) -> None:
        self._history.append(text)
        self._history_text.append(text)
        self._history_audio_end_frames.append(max(alignments) if alignments else None)
        if self._should_reset_after_punctuated_translation(text):
            self._prune_committed_audio_prefix()

    def _should_reset_after_punctuated_translation(self, text: str) -> bool:
        if self._speech_context_seconds <= self.soft_speech_context_reset_seconds:
            return False
        return _ends_with_terminal_punctuation(text)

    def _prune_committed_audio_prefix(self) -> None:
        audio = self._speech_context_audio
        aligned_frames = [
            frame for frame in self._history_audio_end_frames if frame is not None
        ]
        deleted_turns = len(self._history_text)
        self._history.clear()
        self._history_text.clear()
        self._history_audio_end_frames.clear()

        if audio is None or not aligned_frames:
            logger.debug(
                "Reset DOA prefix text after punctuation without audio pruning: "
                "deleted_turns=%d, has_audio=%s, aligned_frames=%d.",
                deleted_turns,
                audio is not None,
                len(aligned_frames),
            )
            return

        prune_frames = max(aligned_frames) + 1
        prune_samples = min(int(audio.numel()), self._audio_frames_to_samples(prune_frames))
        remaining = audio[prune_samples:].clone()
        self._speech_context_audio = remaining if remaining.numel() > 0 else None
        self._speech_context_seconds = float(remaining.numel()) / SAMPLE_RATE
        logger.debug(
            "Pruned DOA prefix context after punctuation: deleted_turns=%d, "
            "prune_frames=%d, prune_samples=%d, remaining_seconds=%.3f.",
            deleted_turns,
            prune_frames,
            prune_samples,
            self._speech_context_seconds,
        )

    def _audio_frames_to_samples(self, frame_count: int) -> int:
        processor = getattr(self._model, "processor", None) if self._model is not None else None
        hop_length = getattr(processor, "hop_length", None)
        if hop_length is None:
            return int(frame_count) * _DEFAULT_AUDIO_FRAME_SAMPLES
        return int(frame_count) * int(hop_length) * 2

    def _generate_candidate_turn(self) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        if self._model is None or self._speech_context_audio is None:
            raise RuntimeError("SimPOTrans model is not loaded.")

        turn_embeds, audio_start, audio_end = self._prefix_turn_embeds_with_audio_span(
            self._speech_context_audio,
        )

        cfg = self._decode_config or self._model.decode_config
        stopping = StoppingCriteriaList([StopOnTokens([self._model.eos_token_id])])

        if cfg.do_sample:
            torch.manual_seed(cfg.seed)

        outputs = self._model.llm.generate(
            inputs_embeds=turn_embeds,
            eos_token_id=[self._model.eos_token_id],
            pad_token_id=self._model.llm_tokenizer.pad_token_id,
            stopping_criteria=stopping,
            max_new_tokens=cfg.max_new_tokens,
            num_beams=cfg.beam_num,
            do_sample=cfg.do_sample,
            min_length=cfg.min_length,
            return_dict_in_generate=True,
            output_logits=True,
            **_decode_generation_kwargs(cfg),
        )

        sequences = outputs.sequences
        if sequences.ndim == 1:
            sequences = sequences.unsqueeze(0)
        return sequences[:1], turn_embeds, audio_start, audio_end

    def _prefix_turn_embeds_with_audio_span(
        self,
        speech_audio: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        if self._model is None:
            raise RuntimeError("SimPOTrans model is not loaded.")

        speech_embeds = self._model.forward_adaptor(
            self._model.encode_speech([speech_audio])
        )
        prompts = self._model._render_prompts(
            [self.source_language],
            [self.target_language],
        )
        tokenized = self._model._tokenize_prompts(prompts)
        prompt_ids = tokenized.input_ids.to(self._model.device)
        prompt_mask = tokenized.attention_mask.to(self._model.device)
        prompt_embeds = self._model.llm.get_input_embeddings()(prompt_ids)

        prompt_len = int(prompt_mask[0].sum().item())
        audio_positions = (prompt_ids[0, :prompt_len] == self._model.audio_here_id).nonzero(
            as_tuple=True,
        )[0]
        if len(audio_positions) != 1:
            raise ValueError(
                f"Prompt must contain exactly one {AUDIO_PLACEHOLDER_TOKEN} token."
            )

        audio_start = int(audio_positions.item())
        audio_len = int(speech_embeds.size(1))
        before = prompt_embeds[:, :audio_start]
        after = prompt_embeds[:, audio_start + 1 : prompt_len]
        turn_embeds = torch.cat([before, speech_embeds, after], dim=1)

        prefix = self._assistant_prefix()
        if prefix:
            prefix_ids = self._model.llm_tokenizer(
                prefix,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.to(self._model.device)
            if prefix_ids.numel() > 0:
                prefix_embeds = self._model.llm.get_input_embeddings()(prefix_ids)
                turn_embeds = torch.cat([turn_embeds, prefix_embeds], dim=1)

        return turn_embeds, audio_start, audio_start + audio_len

    def _token_audio_alignments(
        self,
        *,
        turn_embeds: torch.Tensor,
        tokens: torch.Tensor,
        audio_start: int,
        audio_end: int,
    ) -> list[int]:
        if self._model is None or self._device is None:
            raise RuntimeError("SimPOTrans model is not loaded.")

        token_count = int(tokens.size(1))
        if token_count == 0:
            return []

        token_embeds = self._model.llm.get_input_embeddings()(tokens.to(self._device))
        replay_embeds = torch.cat([turn_embeds, token_embeds], dim=1)
        outputs = self._model.llm(
            inputs_embeds=replay_embeds,
            use_cache=False,
            return_dict=True,
            output_attentions=True,
        )
        attentions = getattr(outputs, "attentions", None)
        if not attentions:
            raise RuntimeError(
                "DOA requires decoder attention weights, but the LLM did not return "
                "attentions. Try setting doa_force_eager_attention=true."
            )

        return _audio_alignments_from_attentions(
            attentions,
            token_count=token_count,
            turn_len=int(turn_embeds.size(1)),
            audio_start=audio_start,
            audio_end=audio_end,
            past_len=0,
            layers=self.doa_layers,
            heads=self.doa_heads,
        )


def _stable_prefix_len_from_attentions(
    attentions: Any,
    *,
    token_count: int,
    turn_len: int,
    audio_start: int,
    audio_end: int,
    cutoff_frames: int,
    past_len: int = 0,
    layers: list[int] | None = None,
    heads: list[int] | None = None,
) -> int:
    alignments = _audio_alignments_from_attentions(
        attentions,
        token_count=token_count,
        turn_len=turn_len,
        audio_start=audio_start,
        audio_end=audio_end,
        past_len=past_len,
        layers=layers,
        heads=heads,
    )
    return _stable_prefix_len_from_alignments(
        alignments,
        audio_len=audio_end - audio_start,
        cutoff_frames=cutoff_frames,
    )


def _audio_alignments_from_attentions(
    attentions: Any,
    *,
    token_count: int,
    turn_len: int,
    audio_start: int,
    audio_end: int,
    past_len: int = 0,
    layers: list[int] | None = None,
    heads: list[int] | None = None,
) -> list[int]:
    audio_len = audio_end - audio_start
    if token_count <= 0 or audio_len <= 0:
        return []

    selected_layers = _select_indices(layers, len(attentions), "layer")
    matrices = []
    token_start = max(0, turn_len - 1)
    token_end = token_start + token_count
    for layer_index in selected_layers:
        layer_attention = attentions[layer_index]
        if layer_attention is None:
            continue
        if layer_attention.ndim != 4:
            raise ValueError(
                "Expected attention tensor with shape "
                f"(batch, heads, query, key), got {tuple(layer_attention.shape)}."
            )
        selected_heads = _select_indices(heads, int(layer_attention.size(1)), "head")
        query_len = int(layer_attention.size(-2))
        key_len = int(layer_attention.size(-1))
        query_start = token_start if token_end <= query_len else max(0, query_len - token_count)
        query_end = min(query_start + token_count, query_len)
        key_audio_start, key_audio_end = _resolve_attention_audio_span(
            local_audio_start=audio_start,
            local_audio_end=audio_end,
            past_len=past_len,
            key_len=key_len,
        )
        if query_end <= query_start or key_audio_end <= key_audio_start:
            continue
        audio_attention = layer_attention[
            0,
            selected_heads,
            query_start:query_end,
            key_audio_start:key_audio_end,
        ]
        if audio_attention.size(-1) == 0 or audio_attention.size(-2) == 0:
            continue
        matrices.append(audio_attention.float().mean(dim=0))

    if not matrices:
        return []

    proxy_attention = torch.stack(matrices, dim=0).mean(dim=0)
    return [int(value) for value in proxy_attention.argmax(dim=-1).tolist()]


def _stable_prefix_len_from_alignments(
    alignments: list[int],
    *,
    audio_len: int,
    cutoff_frames: int,
) -> int:
    stable_limit = max(0, audio_len - int(cutoff_frames))
    prefix_len = 0
    for alignment in alignments:
        if alignment >= stable_limit:
            break
        prefix_len += 1
    return prefix_len


def _resolve_attention_audio_span(
    *,
    local_audio_start: int,
    local_audio_end: int,
    past_len: int,
    key_len: int,
) -> tuple[int, int]:
    absolute_start = past_len + local_audio_start
    absolute_end = past_len + local_audio_end
    if absolute_end <= key_len:
        return absolute_start, absolute_end
    if local_audio_end <= key_len:
        return local_audio_start, local_audio_end
    return 0, 0


def _normalize_index_selection(raw: object) -> list[int] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped or stripped.lower() in {"none", "null", "all"}:
            return None
        return [int(part.strip()) for part in stripped.split(",") if part.strip()]
    if isinstance(raw, (list, tuple)):
        if not raw:
            return None
        return [int(item) for item in raw]
    return [int(raw)]


def _select_indices(indices: list[int] | None, total: int, label: str) -> list[int]:
    if total <= 0:
        return []
    if indices is None:
        return list(range(total))

    selected = []
    for index in indices:
        resolved = index + total if index < 0 else index
        if resolved < 0 or resolved >= total:
            raise IndexError(
                f"DOA {label} index {index} is out of range for {total} {label}s."
            )
        selected.append(resolved)
    return selected


def _ends_with_terminal_punctuation(text: str) -> bool:
    stripped = text.strip().rstrip('\"\'”’）)]】》')
    return stripped.endswith((".", "!", "?", "。", "！", "？"))


SimPOTransChunkProcessor = DOAChunkProcessor

__all__ = [
    "DOAChunkProcessor",
    "SimPOTransChunkProcessor",
    "_audio_alignments_from_attentions",
    "_stable_prefix_len_from_alignments",
    "_stable_prefix_len_from_attentions",
]
