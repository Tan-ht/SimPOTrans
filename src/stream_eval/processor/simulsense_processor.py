from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import torch
import torch.nn as nn
from transformers import StoppingCriteriaList

from simulstream.metrics.logger import METRICS_LOGGER
from simulstream.server.speech_processors import SAMPLE_RATE
from simulstream.server.speech_processors.incremental_output import (
    IncrementalOutput,
)

from stream_eval.processor.base_processor import (
    BaseSimPOTransChunkProcessor,
    _empty_output,
    _normalize_incremental_text,
)
from stream_eval.processor.context_policy import (
    ContextRetentionPolicy,
    make_context_retention_policy,
)
from st_backbone.model import (
    StopOnTokens,
    StreamingSTState,
    _cache_attention_mask,
    _cache_position,
    _decode_generation_kwargs,
    _expand_past_key_values_for_beams,
)

logger = logging.getLogger(__name__)

LANGUAGE_TO_SIMULSENSE_ID = {
    "en": 0,
    "de": 1,
    "fr": 2,
    "es": 3,
}


class SimulSenseWeightPredictor(nn.Module):
    """Sense-unit weight predictor used by the legacy SimulSense checkpoint."""

    def __init__(
        self,
        *,
        encoder_dim: int,
        hidden_dim: int,
        num_langs: int,
        num_layers: int,
        vocab_size: int | None,
    ) -> None:
        super().__init__()
        self.linear_proj = nn.Parameter(torch.empty(num_langs, encoder_dim, hidden_dim))
        self.mlp_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
                for _ in range(num_layers)
            ]
        )
        if vocab_size is not None:
            self.ctc_head = nn.Linear(hidden_dim, vocab_size)
        self.weight_proj = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        encoder_states: torch.Tensor,
        lang_ids: list[int] | torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if not torch.is_tensor(lang_ids):
            lang_ids = torch.tensor(lang_ids, device=encoder_states.device)
        lang_ids = lang_ids.to(device=encoder_states.device, dtype=torch.long)

        expert_linear_proj = self.linear_proj[lang_ids]
        hidden_states = torch.einsum(
            "btc,bcd->btd",
            encoder_states,
            expert_linear_proj,
        )
        for mlp_layer in self.mlp_layers:
            hidden_states = mlp_layer(hidden_states)

        ctc_logits = (
            self.ctc_head(hidden_states)
            if hasattr(self, "ctc_head")
            else None
        )
        logits = self.weight_proj(hidden_states).squeeze(-1)
        if padding_mask is not None:
            logits = logits.masked_fill(~padding_mask.bool(), -1e4)

        weights = torch.sigmoid(logits)
        return {
            "weights": weights,
            "quantity": weights.sum(dim=-1),
            "ctc_logits": ctc_logits,
        }

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        load_ctc_head: bool = False,
    ) -> "SimulSenseWeightPredictor":
        state_dict = normalize_simulsense_state_dict(state_dict)
        linear_proj = state_dict.get("linear_proj")
        if linear_proj is None:
            raise ValueError("SimulSense checkpoint is missing linear_proj.")
        if linear_proj.ndim != 3:
            raise ValueError(
                "SimulSense linear_proj must have shape "
                f"(num_langs, encoder_dim, hidden_dim), got {tuple(linear_proj.shape)}."
            )

        layer_indices = {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("mlp_layers.") and key.split(".")[1].isdigit()
        }
        num_layers = max(layer_indices) + 1 if layer_indices else 0
        vocab_size = (
            int(state_dict["ctc_head.weight"].shape[0])
            if load_ctc_head and "ctc_head.weight" in state_dict
            else None
        )
        if not load_ctc_head:
            state_dict = {
                key: value
                for key, value in state_dict.items()
                if not key.startswith("ctc_head.")
            }
        num_langs, encoder_dim, hidden_dim = linear_proj.shape
        predictor = cls(
            encoder_dim=int(encoder_dim),
            hidden_dim=int(hidden_dim),
            num_langs=int(num_langs),
            num_layers=num_layers,
            vocab_size=vocab_size,
        )
        loading_details = predictor.load_state_dict(state_dict, strict=True)
        if loading_details.missing_keys or loading_details.unexpected_keys:
            raise ValueError(
                "Invalid SimulSense checkpoint. "
                f"missing={loading_details.missing_keys}, "
                f"unexpected={loading_details.unexpected_keys}"
            )
        return predictor


def _ends_with_terminal_punctuation(text: str) -> bool:
    stripped = text.strip().rstrip('\"\'”’）)]】》')
    return stripped.endswith((".", "!", "?", "。", "！", "？"))


def normalize_simulsense_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix("weight_predictor."): value
        for key, value in state_dict.items()
    }


def load_simulsense_weight_predictor(
    simulsense_path: str | Path,
) -> SimulSenseWeightPredictor:
    path = Path(simulsense_path).expanduser()
    checkpoint_path = path / "sense_units_detector.pt" if path.is_dir() else path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"SimulSense detector checkpoint not found: {checkpoint_path}"
        )

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"SimulSense detector checkpoint must contain a state dict: {checkpoint_path}"
        )
    return SimulSenseWeightPredictor.from_state_dict(state_dict)


class SimulSenseChunkProcessor(BaseSimPOTransChunkProcessor):
    """SimPOTrans dialogue streaming gated by a SimulSense sense-unit policy."""

    _weight_predictor: ClassVar[SimulSenseWeightPredictor | None] = None

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__(config)
        self.threshold = float(getattr(config, "threshold", 1.45))
        self.max_wait_duration = float(getattr(config, "max_wait_duration", 6.0))
        self.soft_speech_context_reset_seconds = float(
            getattr(config, "soft_speech_context_reset_seconds", 15.0)
        )
        self._context_retention_policy: ContextRetentionPolicy = (
            make_context_retention_policy(config)
        )
        self._num_sense_units = 0
        self._unprocessed_num_samples = 0
        self._has_pending_empty_turn = False
        self._latest_encoder_states: torch.Tensor | None = None
        self._latest_encoder_audio_len = 0

    @classmethod
    def load_model(cls, config: SimpleNamespace) -> None:
        super().load_model(config)
        if (
            cls._model is not None
            and cls._model.config.speech_encoder_backend.lower().replace("-", "_")
            in {"whisper_rt", "whisperrt"}
        ):
            raise RuntimeError(
                "SimulSense streaming evaluation is not compatible with the "
                "WhisperRT speech encoder. Train a separate SimulSense "
                "detector for WhisperRT features before using this processor."
            )
        if cls._weight_predictor is not None:
            return

        simulsense_path = getattr(config, "simulsense_path", "checkpoints/simulsense")
        logger.info("Loading SimulSense detector from %s.", simulsense_path)
        predictor = load_simulsense_weight_predictor(simulsense_path)
        if cls._model is None:
            raise RuntimeError("SimPOTrans model is not loaded.")

        predictor.to(device=cls._model.device, dtype=cls._model.dtype)
        predictor.eval()
        cls._weight_predictor = predictor

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

        if (
            self._stream_state is not None
            and self._context_retention_policy.should_reset_before_append(
                current_seconds=self._speech_context_seconds,
                next_chunk_seconds=chunk_seconds,
            )
        ):
            logger.debug(
                "Resetting SimPOTrans SimulSense state before appending %.3f seconds "
                "to %.3f seconds of speech context.",
                chunk_seconds,
                self._speech_context_seconds,
            )
            self._reset_simulsense_context(
                reason="hard_limit_before_append",
                next_chunk_seconds=chunk_seconds,
            )

        if self._stream_state is None:
            self._reset_streaming_state()

        self._speech_context_audio = (
            audio
            if self._speech_context_audio is None
            else torch.cat([self._speech_context_audio, audio], dim=0)
        )
        self._speech_context_seconds += chunk_seconds
        self._unprocessed_num_samples += int(audio.numel())

        should_write = self._should_write()
        wait_too_long = (
            self._unprocessed_num_samples
            >= int(round(self.max_wait_duration * SAMPLE_RATE))
        )
        context_full = self._context_retention_policy.is_full(
            current_seconds=self._speech_context_seconds,
        )
        if (
            should_write
            or wait_too_long
            or context_full
            or self._has_pending_empty_turn
        ):
            return self._generate_dialogue_turn()

        return _empty_output()

    @torch.no_grad()
    def end_of_stream(self) -> IncrementalOutput:
        if self._unprocessed_num_samples > 0 and self._speech_context_audio is not None:
            return self._generate_dialogue_turn()
        return _empty_output()

    def clear(self) -> None:
        super().clear()
        self._num_sense_units = 0
        self._unprocessed_num_samples = 0
        self._has_pending_empty_turn = False
        self._clear_encoder_buffer()

    def _reset_simulsense_context(self, *, reason: str, **extra: float | int | str) -> None:
        speech_context_seconds = self._speech_context_seconds
        unprocessed_seconds = float(self._unprocessed_num_samples) / SAMPLE_RATE
        kv_cache_len = self._current_kv_cache_len()
        history_turns = len(self._history)
        self.clear()
        payload = {
            "event": "speech_context_reset",
            "reason": reason,
            "speech_context_seconds": speech_context_seconds,
            "unprocessed_seconds": unprocessed_seconds,
            "kv_cache_len": kv_cache_len,
            "history_turns": history_turns,
            "soft_limit_seconds": self.soft_speech_context_reset_seconds,
            "hard_limit_seconds": self.max_speech_context_seconds,
            "timestamp": time.time(),
        }
        payload.update(extra)
        METRICS_LOGGER.info(json.dumps({"st": payload}, ensure_ascii=False))

    def _reset_simulsense_speech_context(self) -> None:
        self._reset_simulsense_context(reason="speech_context_reset")

    def _clear_encoder_buffer(self) -> None:
        self._latest_encoder_states = None
        self._latest_encoder_audio_len = 0

    def _should_reset_after_punctuated_translation(self, text: str) -> bool:
        if self._speech_context_seconds <= self.soft_speech_context_reset_seconds:
            return False
        return _ends_with_terminal_punctuation(text)

    def _source_language_id(self) -> int:
        try:
            return LANGUAGE_TO_SIMULSENSE_ID[self.source_language]
        except KeyError as ex:
            supported = ", ".join(sorted(LANGUAGE_TO_SIMULSENSE_ID))
            raise ValueError(
                "Unsupported SimulSense source language "
                f"{self.source_language!r}. Supported languages: {supported}."
            ) from ex

    @torch.no_grad()
    def _should_write(self) -> bool:
        if self._model is None or self._weight_predictor is None:
            raise RuntimeError("SimPOTrans model or SimulSense detector is not loaded.")
        if self._speech_context_audio is None:
            return False

        encoder_states = self._model.encode_speech(
            [self._speech_context_audio],
        )
        self._latest_encoder_states = encoder_states
        self._latest_encoder_audio_len = int(
            self._speech_context_audio.numel()
        )
        output = self._weight_predictor(
            encoder_states.to(dtype=self._weight_predictor.linear_proj.dtype),
            [self._source_language_id()],
        )
        quantity = output["quantity"]
        if quantity is None:
            return False
        cur_num_sense_units = int(quantity.item() // self.threshold)
        if cur_num_sense_units >= self._num_sense_units + 1:
            self._num_sense_units = cur_num_sense_units
            return True
        return False

    @torch.no_grad()
    def _generate_dialogue_turn(self) -> IncrementalOutput:
        if (
            self._model is None
            or self._stream_state is None
            or self._speech_context_audio is None
        ):
            return _empty_output()

        current_num_samples = max(1, self._unprocessed_num_samples)
        encoder_states = self._current_encoder_states()
        speech_context_states = self._slice_current_encoder_states(
            encoder_states,
            current_num_samples,
        )
        turn_start_len = self._current_kv_cache_len()
        candidate_state = self._copy_streaming_state()
        tokens, candidate_state = self._generate_dialogue_turn_from_encoder_states(
            speech_context_states,
            candidate_state,
        )

        text = _normalize_incremental_text(
            self._model.decode(tokens)[0],
            self.target_language,
        )
        if not text:
            self._has_pending_empty_turn = True
            return _empty_output()

        self._stream_state = candidate_state
        self._record_dialogue_turn_cache(turn_start_len)
        self._unprocessed_num_samples = 0
        self._has_pending_empty_turn = False
        self._clear_encoder_buffer()
        self._history.append(text)
        output = IncrementalOutput(
            new_tokens=[text],
            new_string=text,
            deleted_tokens=[],
            deleted_string="",
        )
        if self._should_reset_after_punctuated_translation(text):
            self._reset_simulsense_context(
                reason="punctuated_translation_after_soft_limit",
                text_preview=text[:80],
            )
        return output

    def _current_encoder_states(self) -> torch.Tensor:
        if self._model is None or self._speech_context_audio is None:
            raise RuntimeError("SimPOTrans model or speech context is not available.")

        speech_context_len = int(self._speech_context_audio.numel())
        if (
            self._latest_encoder_states is not None
            and self._latest_encoder_audio_len == speech_context_len
        ):
            return self._latest_encoder_states

        encoder_states = self._model.encode_speech(
            [self._speech_context_audio],
        )
        self._latest_encoder_states = encoder_states
        self._latest_encoder_audio_len = speech_context_len
        return encoder_states

    def _slice_current_encoder_states(
        self,
        encoder_states: torch.Tensor,
        current_num_samples: int,
    ) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("SimPOTrans model is not loaded.")

        current_len = max(
            1,
            math.ceil(current_num_samples / (self._model.processor.hop_length * 2)),
        )
        current_len = min(current_len, encoder_states.size(1))
        return encoder_states[:, -current_len:, :]

    def _generate_dialogue_turn_from_encoder_states(
        self,
        encoder_states: torch.Tensor,
        state: StreamingSTState,
    ) -> tuple[torch.Tensor, StreamingSTState]:
        if self._model is None or self._device is None:
            raise RuntimeError("SimPOTrans model is not loaded.")

        if state.past_key_values is None:
            state = self._model.init_streaming_state(
                src_lang=self.source_language,
                tgt_lang=self.target_language,
            )

        speech_embeds = self._model.forward_adaptor(encoder_states)
        turn_embeds = self._model._streaming_turn_embeds(
            speech_embeds,
            self.source_language,
            self.target_language,
        )

        cfg = self._decode_config or self._model.decode_config
        stopping = StoppingCriteriaList([StopOnTokens([self._model.eos_token_id])])

        if cfg.do_sample:
            torch.manual_seed(cfg.seed)

        generation_past_key_values = _expand_past_key_values_for_beams(
            state.past_key_values, cfg.beam_num
        )
        outputs = self._model.llm.generate(
            inputs_embeds=turn_embeds,
            attention_mask=_cache_attention_mask(
                generation_past_key_values, turn_embeds.size(1), self._device
            ),
            cache_position=_cache_position(
                generation_past_key_values, turn_embeds.size(1), self._device
            ),
            past_key_values=generation_past_key_values,
            eos_token_id=[self._model.eos_token_id],
            pad_token_id=self._model.llm_tokenizer.pad_token_id,
            stopping_criteria=stopping,
            max_new_tokens=cfg.max_new_tokens,
            num_beams=cfg.beam_num,
            do_sample=cfg.do_sample,
            min_length=cfg.min_length,
            return_dict_in_generate=True,
            output_logits=True,
            use_cache=True,
            **_decode_generation_kwargs(cfg),
        )

        next_state = StreamingSTState(past_key_values=state.past_key_values)
        self._model._append_embeds_to_streaming_state(next_state, turn_embeds)
        self._model._append_token_ids_to_streaming_state(next_state, outputs.sequences)
        self._model._append_text_to_streaming_state(next_state, "\n")

        return outputs.sequences, next_state


SimPOTransChunkProcessor = SimulSenseChunkProcessor

__all__ = [
    "LANGUAGE_TO_SIMULSENSE_ID",
    "SimulSenseChunkProcessor",
    "SimulSenseWeightPredictor",
    "SimPOTransChunkProcessor",
    "load_simulsense_weight_predictor",
    "normalize_simulsense_state_dict",
]
