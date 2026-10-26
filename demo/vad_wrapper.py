from __future__ import annotations

import time
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
from silero_vad import VADIterator, load_silero_vad

from simulstream.server.speech_processors import (
    SAMPLE_RATE,
    SpeechProcessor,
    speech_processor_class_load,
)
from simulstream.server.speech_processors.incremental_output import (
    IncrementalOutput,
    merge_incremental_outputs,
)


class FixedDialogueVADProcessor(SpeechProcessor):
    """Silero VAD wrapper for the SimPOTrans fixed dialogue streaming processor."""

    vad_model: ClassVar[object | None] = None

    @classmethod
    def speech_processor_class(cls, config: SimpleNamespace) -> type[SpeechProcessor]:
        class_name = getattr(
            config,
            "base_speech_processor_class",
            "stream_eval.processor.fixed_processor_dialogue.SimPOTransChunkProcessor",
        )
        return speech_processor_class_load(class_name)

    @classmethod
    def load_model(cls, config: SimpleNamespace) -> None:
        super().load_model(config)
        if cls.vad_model is None:
            cls.vad_model = load_silero_vad()
        cls.speech_processor_class(config).load_model(config)

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__(config)
        if SAMPLE_RATE != 16000:
            raise ValueError("Silero VAD requires a 16 kHz processor sample rate.")

        base_cls = self.speech_processor_class(self.config)
        self.speech_processor = base_cls(self.config)
        self.source_language = str(getattr(self.config, "source_language", ""))
        self.target_language = str(getattr(self.config, "target_language", ""))
        self.default_threshold = float(getattr(self.config, "threshold", 1.0))
        self.language_pair_thresholds = self._load_language_pair_thresholds()
        self._apply_language_pair_threshold()
        self.vad_iterator = VADIterator(
            self.vad_model,
            threshold=float(getattr(self.config, "vad_threshold", 0.5)),
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=int(
                getattr(self.config, "vad_min_silence_duration_ms", 100)
            ),
            speech_pad_ms=int(getattr(self.config, "vad_speech_pad_ms", 30)),
        )
        self.window_size_samples = int(
            getattr(self.config, "vad_window_size_samples", 512)
        )
        vad_clear_silence_seconds = float(
            getattr(self.config, "vad_clear_silence_seconds", 10.0)
        )
        self.vad_clear_silence_samples = (
            int(round(vad_clear_silence_seconds * SAMPLE_RATE))
            if vad_clear_silence_seconds > 0.0
            else 0
        )
        self.speech_chunk_samples = int(round(self.speech_chunk_size * SAMPLE_RATE))
        min_flush_seconds = float(getattr(self.config, "vad_min_flush_chunk_seconds", 0.5))
        self.min_flush_chunk_samples = max(0, int(round(min_flush_seconds * SAMPLE_RATE)))
        self.residual_audio: np.ndarray | None = None
        self.previous_audio_chunk: np.ndarray | None = None
        self.speech_buffer: np.ndarray | None = None
        self.audio_cursor_position = 0
        self.in_speech = False
        self._silent_samples_since_speech = 0
        self._silence_auto_cleared = False
        self._simulst_call_sequence = 0
        self.deferred_flush_audio: np.ndarray | None = None

    def clear(self) -> None:
        super().clear()
        self._clear_vad_segment_state()
        self.vad_iterator.reset_states()
        self.speech_processor.clear()

    def warmup(self) -> None:
        vad_audio = np.zeros(max(self.window_size_samples, int(0.25 * SAMPLE_RATE)), dtype=np.float32)
        self.process_chunk(vad_audio)
        model_audio = np.zeros(max(1, self.speech_chunk_samples), dtype=np.float32)
        self.speech_processor.process_chunk(model_audio)
        self.speech_processor.end_of_stream()
        self.clear()

    def process_chunk(self, waveform: np.ndarray) -> IncrementalOutput:
        if waveform.size == 0:
            return self._merge_outputs([])

        waveform = np.asarray(waveform, dtype=np.float32)
        if self.residual_audio is not None:
            waveform = np.concatenate((self.residual_audio, waveform))
            self.residual_audio = None

        outputs: list[IncrementalOutput] = []
        for start in range(0, len(waveform), self.window_size_samples):
            chunk = waveform[start : start + self.window_size_samples]
            if len(chunk) < self.window_size_samples:
                self.residual_audio = chunk
                break

            speech_event = self.vad_iterator(chunk, return_seconds=False)
            if speech_event:
                outputs.extend(self._handle_vad_event(chunk, speech_event))
            elif self.in_speech:
                self._append_to_speech_buffer(chunk)
                outputs.extend(self._emit_ready_speech_chunks())
            else:
                self._handle_vad_silence(len(chunk))

            self.audio_cursor_position += self.window_size_samples
            self.previous_audio_chunk = chunk

        return self._merge_outputs(outputs)

    def end_of_stream(self) -> IncrementalOutput:
        outputs: list[IncrementalOutput] = []
        if self.residual_audio is not None and self.in_speech:
            self._append_to_speech_buffer(self.residual_audio)
        self.residual_audio = None
        if self.deferred_flush_audio is not None:
            self._prepend_to_speech_buffer(self.deferred_flush_audio)
            self.deferred_flush_audio = None

        outputs.extend(self._flush_current_speech_segment(force=True))
        self._clear_vad_segment_state(reset_previous_audio=False)
        return self._merge_outputs(outputs)

    def set_source_language(self, language: str) -> None:
        self.source_language = language
        self.speech_processor.set_source_language(language)
        self._apply_language_pair_threshold()

    def set_target_language(self, language: str) -> None:
        self.target_language = language
        self.speech_processor.set_target_language(language)
        self._apply_language_pair_threshold()

    def tokens_to_string(self, tokens: list[str]) -> str:
        return self.speech_processor.tokens_to_string(tokens)

    def _load_language_pair_thresholds(self) -> dict[tuple[str, str], float]:
        thresholds = getattr(self.config, "language_pair_thresholds", [])
        pair_thresholds: dict[tuple[str, str], float] = {}

        for item in thresholds:
            source = str(getattr(item, "source", "")).strip()
            target = str(getattr(item, "target", "")).strip()
            if not source or not target:
                continue
            pair_thresholds[(source, target)] = float(getattr(item, "threshold"))

        return pair_thresholds

    def _apply_language_pair_threshold(self) -> None:
        threshold = self.language_pair_thresholds.get(
            (self.source_language, self.target_language),
            self.default_threshold,
        )
        if hasattr(self.speech_processor, "threshold"):
            self.speech_processor.threshold = threshold

    def _handle_vad_event(
        self,
        chunk: np.ndarray,
        speech_event: dict[str, int],
    ) -> list[IncrementalOutput]:
        outputs: list[IncrementalOutput] = []
        if "start" in speech_event:
            self._start_speech(chunk, int(speech_event["start"]))
            outputs.append(self._vad_speech_start_output())
            outputs.extend(self._emit_ready_speech_chunks())

        if "end" in speech_event:
            outputs.extend(self._end_speech(chunk, int(speech_event["end"])))

        return outputs

    def _start_speech(self, chunk: np.ndarray, absolute_start: int) -> None:
        self._reset_vad_silence_counter()
        self.in_speech = True
        previous = self.previous_audio_chunk
        if previous is None:
            previous = np.empty(0, dtype=np.float32)

        start_offset = self.audio_cursor_position - len(previous)
        chunk_start_position = max(0, absolute_start - start_offset)
        current = np.concatenate((previous, chunk))[chunk_start_position:]
        if self.deferred_flush_audio is not None:
            current = np.concatenate((self.deferred_flush_audio, current))
            self.deferred_flush_audio = None
        self.speech_buffer = current

    def _end_speech(self, chunk: np.ndarray, absolute_end: int) -> list[IncrementalOutput]:
        if not self.in_speech:
            return []

        speech_buffer_len = len(self.speech_buffer) if self.speech_buffer is not None else 0
        speech_buffer_offset = self.audio_cursor_position - speech_buffer_len
        chunk_end_position = absolute_end - speech_buffer_offset
        if chunk_end_position >= 0:
            self._append_to_speech_buffer(chunk)
            if self.speech_buffer is not None:
                self.speech_buffer = self.speech_buffer[:chunk_end_position]

        outputs = self._flush_current_speech_segment(force=False)
        self.in_speech = False
        if outputs or self.deferred_flush_audio is None:
            outputs.append(self._segment_end_output())
        self._reset_vad_silence_counter()
        return outputs

    def _emit_ready_speech_chunks(self) -> list[IncrementalOutput]:
        outputs: list[IncrementalOutput] = []
        while self.speech_buffer is not None and len(self.speech_buffer) >= self.speech_chunk_samples:
            speech_chunk = self.speech_buffer[: self.speech_chunk_samples]
            remaining = self.speech_buffer[self.speech_chunk_samples :]
            self.speech_buffer = remaining if len(remaining) > 0 else None
            start_time = time.time()
            output = self.speech_processor.process_chunk(speech_chunk)
            computation_time = time.time() - start_time
            outputs.append(self._mark_processor_called(
                output,
                chunk_samples=len(speech_chunk),
                flush=False,
                computation_time=computation_time,
            ))
        return outputs

    def _flush_current_speech_segment(self, *, force: bool = False) -> list[IncrementalOutput]:
        outputs: list[IncrementalOutput] = []
        if self.speech_buffer is not None and len(self.speech_buffer) > 0:
            chunk_samples = len(self.speech_buffer)
            if not force and 0 < chunk_samples < self.min_flush_chunk_samples:
                self._defer_flush_audio(self.speech_buffer)
                self.speech_buffer = None
                return outputs
            start_time = time.time()
            output = self.speech_processor.process_chunk(self.speech_buffer)
            computation_time = time.time() - start_time
            outputs.append(self._mark_processor_called(
                output,
                chunk_samples=chunk_samples,
                flush=True,
                computation_time=computation_time,
            ))
            self.speech_buffer = None
        outputs.append(self.speech_processor.end_of_stream())
        return outputs

    @staticmethod
    def _vad_speech_start_output() -> IncrementalOutput:
        output = IncrementalOutput([], "", [], "")
        output.vad_speech_start = True
        return output

    @staticmethod
    def _segment_end_output() -> IncrementalOutput:
        output = IncrementalOutput([], "", [], "")
        output.segment_end = True
        return output

    def _mark_processor_called(
        self,
        output: IncrementalOutput,
        *,
        chunk_samples: int,
        flush: bool,
        computation_time: float,
    ) -> IncrementalOutput:
        self._simulst_call_sequence += 1
        output.processor_called = True
        output.simulst_call_index = self._simulst_call_sequence
        output.simulst_chunk_seconds = chunk_samples / SAMPLE_RATE
        output.simulst_flush = flush
        output.simulst_computation_time = computation_time
        return output

    def _append_to_speech_buffer(self, audio_chunk: np.ndarray) -> None:
        if self.speech_buffer is None:
            self.speech_buffer = audio_chunk
        else:
            self.speech_buffer = np.concatenate((self.speech_buffer, audio_chunk))

    def _prepend_to_speech_buffer(self, audio_chunk: np.ndarray) -> None:
        if self.speech_buffer is None:
            self.speech_buffer = audio_chunk
        else:
            self.speech_buffer = np.concatenate((audio_chunk, self.speech_buffer))

    def _defer_flush_audio(self, audio_chunk: np.ndarray) -> None:
        if self.deferred_flush_audio is None:
            self.deferred_flush_audio = audio_chunk
        else:
            self.deferred_flush_audio = np.concatenate((self.deferred_flush_audio, audio_chunk))

    def _handle_vad_silence(self, num_samples: int) -> None:
        if self.vad_clear_silence_samples <= 0 or self._silence_auto_cleared:
            return

        self._silent_samples_since_speech += num_samples
        if self._silent_samples_since_speech >= self.vad_clear_silence_samples:
            self.speech_processor.clear()
            self._clear_vad_segment_state()
            self.vad_iterator.reset_states()
            self._silence_auto_cleared = True

    def _reset_vad_silence_counter(self) -> None:
        self._silent_samples_since_speech = 0
        self._silence_auto_cleared = False

    def _clear_vad_segment_state(self, reset_previous_audio: bool = True) -> None:
        self.residual_audio = None
        if reset_previous_audio:
            self.previous_audio_chunk = None
        self.speech_buffer = None
        self.deferred_flush_audio = None
        self.audio_cursor_position = 0
        self.in_speech = False
        self._silent_samples_since_speech = 0
        self._silence_auto_cleared = False
        self._simulst_call_sequence = 0

    def _merge_outputs(self, outputs: list[IncrementalOutput]) -> IncrementalOutput:
        merged = merge_incremental_outputs(outputs, self.tokens_to_string)
        if any(getattr(output, "vad_speech_start", False) for output in outputs):
            merged.vad_speech_start = True
        processor_outputs = [
            output for output in outputs if getattr(output, "processor_called", False)
        ]
        if processor_outputs:
            merged.processor_called = True
            merged.simulst_outputs = [
                {
                    "call_index": getattr(output, "simulst_call_index", None),
                    "chunk_seconds": getattr(output, "simulst_chunk_seconds", None),
                    "computation_time": getattr(output, "simulst_computation_time", None),
                    "flush": bool(getattr(output, "simulst_flush", False)),
                    "new_tokens": output.new_tokens,
                    "new_text": output.new_string or "",
                    "deleted_tokens": output.deleted_tokens,
                    "deleted_text": output.deleted_string or "",
                }
                for output in processor_outputs
            ]
        if any(getattr(output, "segment_end", False) for output in outputs):
            merged.segment_end = True
        return merged


SimulSenseVADProcessor = FixedDialogueVADProcessor
