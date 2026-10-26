from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from simulstream.server.speech_processors import SAMPLE_RATE
from simulstream.server.speech_processors.incremental_output import (
    IncrementalOutput,
)

from st_backbone.model import SpeechEncoderNotReadyError

from stream_eval.processor.base_processor import (
    BaseSimPOTransChunkProcessor,
    _empty_output,
    _load_model_config,
    _make_decode_config,
    _normalize_incremental_text,
    _resolve_checkpoint_path,
)

logger = logging.getLogger(__name__)
_CJK_LANGUAGES = {"ja", "ko", "zh", "zh-cn", "zh-tw"}
_DIALOGUE_REPETITION_NGRAM = 4
_DIALOGUE_REPETITION_RATIO_THRESHOLD = 0.7
_INVALID_TRANSLATION_MARKERS = ("[Music]", "[Speaker]", "（音乐）")
_HISTORY_POLICIES = {"none", "soft", "punctuation", "fixed", "kv"}

__all__ = [
    "SimPOTransChunkProcessor",
    "_empty_output",
    "_load_model_config",
    "_make_decode_config",
    "_normalize_incremental_text",
    "_resolve_checkpoint_path",
]


def _ends_with_terminal_punctuation(text: str) -> bool:
    stripped = text.strip().rstrip('\"\'”’）)]】》')
    return stripped.endswith((".", "!", "?", "。", "！", "？"))


class SimPOTransChunkProcessor(BaseSimPOTransChunkProcessor):
    """Chunk-by-chunk SimPOTrans decoding with dialogue-style KV cache."""

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__(config)
        self.history_selection_policy = str(
            getattr(config, "history_selection_policy", "soft")
        ).lower()
        if self.history_selection_policy not in _HISTORY_POLICIES:
            raise ValueError(
                "Unsupported history_selection_policy: "
                f"{self.history_selection_policy}. Expected one of "
                f"{sorted(_HISTORY_POLICIES)}."
            )
        self.soft_speech_context_reset_seconds = float(
            getattr(config, "soft_speech_context_reset_seconds", 15.0)
        )
        self.fixed_history_words = int(getattr(config, "fixed_history_words", 100))
        self.speech_context_policy = str(
            getattr(config, "speech_context_policy", "fixed_duration")
        )
        llm_context_turns = getattr(config, "llm_context_turns", None)
        self.llm_context_turns = (
            None if llm_context_turns is None else int(llm_context_turns)
        )
        if self.history_selection_policy != "kv":
            self.llm_context_turns = None

        self._speech_context_chunks: list[torch.Tensor] = []
        self._dialogue_context_seconds = 0.0
        self._history_word_count = 0
        self._uncommitted_num_samples = 0
        self._previous_turn_repetition_suspect = False
        self._system_cache_len = 0
        self._turn_cache_boundaries: list[tuple[int, int]] = []

    def clear(self) -> None:
        super().clear()
        self._history_word_count = 0
        self._previous_turn_repetition_suspect = False
        self._system_cache_len = 0
        self._turn_cache_boundaries = []

    def _reset_streaming_state(self) -> None:
        super()._reset_streaming_state()
        self._system_cache_len = self._current_kv_cache_len()
        self._turn_cache_boundaries = []

    def _reset_speech_buffer(self) -> None:
        super()._reset_speech_buffer()
        self._speech_context_chunks = []
        self._dialogue_context_seconds = 0.0
        self._uncommitted_num_samples = 0

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

        if self._stream_state is None:
            self._reset_streaming_state()

        if self.history_selection_policy == "kv":
            self._append_kv_policy_speech_chunk(audio, chunk_seconds)
        else:
            self._append_reset_policy_speech_chunk(audio, chunk_seconds)

        if self._speech_context_audio is None:
            return _empty_output()

        current_num_samples = max(
            1,
            min(self._uncommitted_num_samples, int(self._speech_context_audio.numel())),
        )
        turn_start_len = self._current_kv_cache_len()
        candidate_state = self._copy_streaming_state()
        candidate_speech_cache = self._copy_speech_encoder_cache_state()
        try:
            tokens, candidate_state = self._model.generate_streaming_turn(
                self._speech_context_audio,
                candidate_state,
                current_num_samples=current_num_samples,
                src_lang=self.source_language,
                tgt_lang=self.target_language,
                decode_config=self._decode_config,
                speech_encoder_cache_state=candidate_speech_cache,
            )
        except SpeechEncoderNotReadyError:
            return _empty_output()

        text = _normalize_incremental_text(
            self._model.decode(tokens)[0],
            self.target_language,
        )
        if not text:
            return _empty_output()
        if _has_invalid_translation_marker(text):
            logger.debug("Dropping SimPOTrans dialogue turn due to invalid marker: %r", text[:80])
            return _empty_output()
        if self._has_dialogue_ngram_repetition(text):
            logger.debug(
                "Dropping SimPOTrans dialogue turn due to repeated %d-gram: %r",
                _DIALOGUE_REPETITION_NGRAM,
                text[:80],
            )
            return _empty_output()

        text = _format_committed_translation(text, self.target_language)

        self._stream_state = candidate_state
        self._commit_speech_encoder_cache_state(candidate_speech_cache)
        self._record_dialogue_turn_cache(turn_start_len)
        self._uncommitted_num_samples = 0
        self._update_dialogue_repetition_suspect(text)
        self._history.append(text)
        self._history_word_count += _count_history_units(text, self.target_language)
        self._trim_dialogue_history()
        self._maybe_reset_after_committed_turn(text)

        return IncrementalOutput(
            new_tokens=[text],
            new_string=text,
            deleted_tokens=[],
            deleted_string="",
        )

    @torch.no_grad()
    def end_of_stream(self) -> IncrementalOutput:
        if (
            self._model is None
            or self._stream_state is None
            or self._speech_context_audio is None
            or self._uncommitted_num_samples <= 0
        ):
            return _empty_output()

        current_num_samples = max(
            1,
            min(self._uncommitted_num_samples, int(self._speech_context_audio.numel())),
        )
        turn_start_len = self._current_kv_cache_len()
        candidate_state = self._copy_streaming_state()
        candidate_speech_cache = self._copy_speech_encoder_cache_state()
        try:
            tokens, candidate_state = self._model.generate_streaming_turn(
                self._speech_context_audio,
                candidate_state,
                current_num_samples=current_num_samples,
                src_lang=self.source_language,
                tgt_lang=self.target_language,
                decode_config=self._decode_config,
                speech_encoder_cache_state=candidate_speech_cache,
                is_final=True,
            )
        except SpeechEncoderNotReadyError:
            return _empty_output()
        text = _normalize_incremental_text(
            self._model.decode(tokens)[0],
            self.target_language,
        )
        if (
            not text
            or _has_invalid_translation_marker(text)
            or self._has_dialogue_ngram_repetition(text)
        ):
            return _empty_output()

        text = _format_committed_translation(text, self.target_language)

        self._stream_state = candidate_state
        self._commit_speech_encoder_cache_state(candidate_speech_cache)
        self._record_dialogue_turn_cache(turn_start_len)
        self._uncommitted_num_samples = 0
        self._update_dialogue_repetition_suspect(text)
        self._history.append(text)
        self._history_word_count += _count_history_units(text, self.target_language)
        self._trim_dialogue_history()
        self._maybe_reset_after_committed_turn(text)
        return IncrementalOutput(
            new_tokens=[text],
            new_string=text,
            deleted_tokens=[],
            deleted_string="",
        )

    def _append_reset_policy_speech_chunk(
        self,
        audio: torch.Tensor,
        chunk_seconds: float,
    ) -> None:
        if self._speech_context_seconds + chunk_seconds > self.max_speech_context_seconds:
            logger.debug(
                "Resetting SimPOTrans dialogue context: "
                "speech_context_seconds=%.3f, next_chunk_seconds=%.3f, "
                "max_speech_context_seconds=%.3f, kv_cache_len=%d, "
                "history_turns=%d, uncommitted_samples=%d.",
                self._speech_context_seconds,
                chunk_seconds,
                self.max_speech_context_seconds,
                self._current_kv_cache_len(),
                len(self._history),
                self._uncommitted_num_samples,
            )
            self.clear()
            self._reset_streaming_state()

        self._speech_context_audio = (
            audio
            if self._speech_context_audio is None
            else torch.cat([self._speech_context_audio, audio], dim=0)
        )
        self._speech_context_seconds += chunk_seconds
        self._dialogue_context_seconds += chunk_seconds
        self._uncommitted_num_samples += int(audio.numel())

    def _append_kv_policy_speech_chunk(
        self,
        audio: torch.Tensor,
        chunk_seconds: float,
    ) -> None:
        if self.speech_context_policy == "sliding_window":
            while (
                self._speech_context_chunks
                and self._speech_context_seconds + chunk_seconds
                > self.max_speech_context_seconds
            ):
                self._drop_oldest_speech_chunk()
        elif self._speech_context_seconds + chunk_seconds > self.max_speech_context_seconds:
            logger.debug(
                "Resetting SimPOTrans speech context only: "
                "speech_context_seconds=%.3f, next_chunk_seconds=%.3f, "
                "max_speech_context_seconds=%.3f, kv_cache_len=%d, "
                "history_turns=%d, uncommitted_samples=%d.",
                self._speech_context_seconds,
                chunk_seconds,
                self.max_speech_context_seconds,
                self._current_kv_cache_len(),
                len(self._history),
                self._uncommitted_num_samples,
            )
            self._reset_speech_buffer()

        self._speech_context_chunks.append(audio)
        self._speech_context_seconds += chunk_seconds
        self._dialogue_context_seconds += chunk_seconds
        self._uncommitted_num_samples += int(audio.numel())
        self._refresh_speech_context_audio()

    def _drop_oldest_speech_chunk(self) -> None:
        total_before = sum(int(chunk.numel()) for chunk in self._speech_context_chunks)
        removed = self._speech_context_chunks.pop(0)
        removed_samples = int(removed.numel())
        committed_prefix_samples = max(0, total_before - self._uncommitted_num_samples)
        if removed_samples > committed_prefix_samples:
            removed_uncommitted = removed_samples - committed_prefix_samples
            self._uncommitted_num_samples = max(
                0,
                self._uncommitted_num_samples - removed_uncommitted,
            )
        self._speech_context_seconds = max(
            0.0,
            self._speech_context_seconds - float(removed_samples) / SAMPLE_RATE,
        )

    def _refresh_speech_context_audio(self) -> None:
        self._speech_context_audio = (
            torch.cat(self._speech_context_chunks, dim=0)
            if self._speech_context_chunks
            else None
        )

    def _maybe_reset_after_committed_turn(self, text: str) -> None:
        if self.history_selection_policy in {"none", "kv"}:
            return
        if self.history_selection_policy == "soft":
            if self._should_reset_after_punctuated_translation(text):
                self._reset_dialogue_context(
                    reason="punctuated_translation_after_soft_limit",
                    text_preview=text[:80],
                )
            return
        if self.history_selection_policy == "punctuation":
            if _ends_with_terminal_punctuation(text):
                self._reset_dialogue_context(
                    reason="punctuated_translation",
                    text_preview=text[:80],
                )
            return
        if self.history_selection_policy == "fixed":
            if self._history_word_count >= self.fixed_history_words:
                self._reset_dialogue_context(
                    reason="fixed_history_words",
                    fixed_history_words=self.fixed_history_words,
                    observed_history_words=self._history_word_count,
                    text_preview=text[:80],
                )

    def _record_dialogue_turn_cache(self, start_len: int) -> None:
        if self.llm_context_turns is None:
            return
        end_len = self._current_kv_cache_len()
        if end_len <= start_len:
            return
        self._turn_cache_boundaries.append((start_len, end_len))
        self._trim_dialogue_cache_if_needed()

    def _trim_dialogue_cache_if_needed(self) -> None:
        if self.llm_context_turns is None:
            return
        max_turns = max(0, self.llm_context_turns)
        if len(self._turn_cache_boundaries) <= max_turns:
            return
        if self._stream_state is None or self._stream_state.past_key_values is None:
            return

        latest_turns = self._turn_cache_boundaries[-max_turns:] if max_turns else []
        keep_ranges = []
        if self._system_cache_len > 0:
            keep_ranges.append((0, self._system_cache_len))
        keep_ranges.extend(latest_turns)
        self._stream_state.past_key_values = _slice_past_key_values(
            self._stream_state.past_key_values,
            keep_ranges,
        )

        new_boundaries: list[tuple[int, int]] = []
        cursor = self._system_cache_len
        for start, end in latest_turns:
            turn_len = max(0, end - start)
            new_boundaries.append((cursor, cursor + turn_len))
            cursor += turn_len
        self._turn_cache_boundaries = new_boundaries

    def _trim_dialogue_history(self) -> None:
        if self.llm_context_turns is None:
            return
        max_turns = max(0, self.llm_context_turns)
        if len(self._history) > max_turns:
            del self._history[: len(self._history) - max_turns]

    def _has_dialogue_ngram_repetition(self, text: str) -> bool:
        repeat_ratio = self._previous_turn_repeat_ratio(text)
        return (
            repeat_ratio >= _DIALOGUE_REPETITION_RATIO_THRESHOLD
            and self._previous_turn_repetition_suspect
        )

    def _update_dialogue_repetition_suspect(self, text: str) -> None:
        self._previous_turn_repetition_suspect = (
            self._previous_turn_repeat_ratio(text)
            >= _DIALOGUE_REPETITION_RATIO_THRESHOLD
        )

    def _previous_turn_repeat_ratio(self, text: str) -> float:
        if not self._history:
            return 0.0

        current_ngrams = _dialogue_ngrams(
            text,
            self.target_language,
            _DIALOGUE_REPETITION_NGRAM,
        )
        if not current_ngrams:
            return 0.0

        previous_ngrams = _dialogue_ngrams(
            self._history[-1],
            self.target_language,
            _DIALOGUE_REPETITION_NGRAM,
        )
        if not previous_ngrams:
            return 0.0

        return len(current_ngrams & previous_ngrams) / len(current_ngrams)

    def _should_reset_after_punctuated_translation(self, text: str) -> bool:
        if self._dialogue_context_seconds <= self.soft_speech_context_reset_seconds:
            return False
        return _ends_with_terminal_punctuation(text)

    def _reset_dialogue_context(self, *, reason: str, **extra: float | int | str) -> None:
        speech_context_seconds = self._speech_context_seconds
        kv_cache_len = self._current_kv_cache_len()
        history_turns = len(self._history)
        history_word_count = self._history_word_count
        uncommitted_seconds = float(self._uncommitted_num_samples) / SAMPLE_RATE
        dialogue_context_seconds = self._dialogue_context_seconds
        self.clear()
        logger.debug(
            "Reset SimPOTrans dialogue context: reason=%s, "
            "speech_context_seconds=%.3f, unprocessed_seconds=%.3f, "
            "kv_cache_len=%d, history_turns=%d, history_words=%d, "
            "dialogue_context_seconds=%.3f, extra=%s.",
            reason,
            speech_context_seconds,
            uncommitted_seconds,
            kv_cache_len,
            history_turns,
            history_word_count,
            dialogue_context_seconds,
            extra,
        )


def _dialogue_ngrams(text: str, language: str, n: int) -> set[str | tuple[str, ...]]:
    if n <= 0:
        return set()

    language = language.lower()
    if language in _CJK_LANGUAGES:
        chars = "".join(text.split())
        if len(chars) < n:
            return set()
        return {chars[index : index + n] for index in range(len(chars) - n + 1)}

    words = text.split()
    if len(words) < n:
        return set()
    return {tuple(words[index : index + n]) for index in range(len(words) - n + 1)}


def _has_invalid_translation_marker(text: str) -> bool:
    return any(marker in text for marker in _INVALID_TRANSLATION_MARKERS)


def _format_committed_translation(text: str, target_language: str) -> str:
    language = target_language.lower()
    if language == "en" or language.startswith("en-"):
        return text + " "
    return text


def _count_history_units(text: str, language: str) -> int:
    language = language.lower()
    if language in _CJK_LANGUAGES:
        return len("".join(text.split()))
    return len(text.split())


def _slice_past_key_values(past_key_values: Any, ranges: list[tuple[int, int]]) -> Any:
    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            if getattr(layer, "keys", None) is not None:
                layer.keys = _slice_cache_tensor(layer.keys, ranges)
            if getattr(layer, "values", None) is not None:
                layer.values = _slice_cache_tensor(layer.values, ranges)
        return past_key_values

    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        for index in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[index] = _slice_cache_tensor(
                past_key_values.key_cache[index],
                ranges,
            )
            past_key_values.value_cache[index] = _slice_cache_tensor(
                past_key_values.value_cache[index],
                ranges,
            )
        if hasattr(past_key_values, "_seen_tokens"):
            past_key_values._seen_tokens = _ranges_total_len(ranges)
        return past_key_values

    if isinstance(past_key_values, tuple):
        return tuple(_slice_cache_layer(layer, ranges) for layer in past_key_values)

    raise TypeError(
        "Unsupported past_key_values type for dialogue trimming: "
        f"{type(past_key_values).__name__}"
    )


def _slice_cache_layer(layer: Any, ranges: list[tuple[int, int]]) -> Any:
    if isinstance(layer, tuple) and len(layer) >= 2:
        key = _slice_cache_tensor(layer[0], ranges)
        value = _slice_cache_tensor(layer[1], ranges)
        return (key, value, *layer[2:])
    if isinstance(layer, list) and len(layer) >= 2:
        sliced = list(layer)
        sliced[0] = _slice_cache_tensor(sliced[0], ranges)
        sliced[1] = _slice_cache_tensor(sliced[1], ranges)
        return sliced
    return layer


def _slice_cache_tensor(tensor: torch.Tensor, ranges: list[tuple[int, int]]) -> torch.Tensor:
    pieces = [tensor[..., start:end, :] for start, end in ranges if end > start]
    if not pieces:
        return tensor[..., :0, :]
    return torch.cat(pieces, dim=-2)


def _ranges_total_len(ranges: list[tuple[int, int]]) -> int:
    return sum(max(0, end - start) for start, end in ranges)
