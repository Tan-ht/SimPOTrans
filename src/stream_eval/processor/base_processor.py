from __future__ import annotations

import copy
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import torch
import yaml

from simulstream.server.speech_processors.base import SpeechProcessor
from simulstream.server.speech_processors.incremental_output import (
    IncrementalOutput,
)

from st_backbone.model import (
    DecodeConfig,
    LoraConfigSection,
    STModelConfig,
    StreamingSTState,
    create_st_model,
    _past_key_values_len,
)

logger = logging.getLogger(__name__)

_CJK_LANGUAGES = {"ja", "ko", "zh", "zh-cn", "zh-tw"}


class BaseSimPOTransChunkProcessor(SpeechProcessor):
    """Shared SimPOTrans SimulStream processor plumbing.

    Concrete processors implement their own ``process_chunk`` strategy while
    this base class owns model loading, runtime config parsing, language
    helpers, and streaming cache lifecycle.
    """

    _model: ClassVar[torch.nn.Module | None] = None
    _device: ClassVar[torch.device | None] = None
    _decode_config: ClassVar[DecodeConfig | None] = None

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__(config)
        self.source_language = (
            getattr(config, "source_language", None)
            or getattr(config, "source_lang", None)
            or "en"
        )
        self.target_language = (
            getattr(config, "target_language", None)
            or getattr(config, "target_lang", None)
            or "zh"
        )
        self.max_speech_context_seconds = float(
            getattr(config, "max_speech_context_seconds", 30.0)
        )
        self._stream_state: StreamingSTState | None = None
        self._speech_encoder_cache_state: object | None = None
        self._speech_context_seconds = 0.0
        self._speech_context_audio: torch.Tensor | None = None
        self._history: list[str] = []

    @classmethod
    def load_model(cls, config: SimpleNamespace) -> None:
        if cls._model is not None:
            return

        if bool(getattr(config, "llm_raw_kv_cache", False)):
            from st_backbone.patch.qwen3_raw_kv import patch_qwen3_raw_kv_cache

            patch_qwen3_raw_kv_cache()

        model_config = _load_model_config(Path(config.model_config), config)
        if hasattr(config, "decode"):
            model_config.decode = _make_decode_config(
                config.decode,
                model_config.decode,
            )

        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(getattr(config, "device", default_device))
        logger.info("Loading SimPOTrans model on %s.", device)
        model = create_st_model(model_config)

        checkpoint = getattr(config, "checkpoint", None) or getattr(
            config,
            "ckpt_path",
            None,
        )
        if checkpoint:
            checkpoint_path = _resolve_checkpoint_path(Path(checkpoint))
            state = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state, strict=True)
            logger.info("Loaded SimPOTrans checkpoint: %s", checkpoint_path)

        model.to(device)
        model.eval()

        cls._model = model
        cls._device = device
        cls._decode_config = model_config.decode

    def set_source_language(self, language: str) -> None:
        self.source_language = language

    def set_target_language(self, language: str) -> None:
        self.target_language = language

    def end_of_stream(self) -> IncrementalOutput:
        return _empty_output()

    def tokens_to_string(self, tokens: list[str]) -> str:
        if self.target_language in _CJK_LANGUAGES:
            return "".join(tokens)

        return " ".join(token for token in tokens if token)

    def clear(self) -> None:
        self._stream_state = None
        self._reset_speech_buffer()
        self._history.clear()

    def _reset_speech_buffer(self) -> None:
        current_cache = getattr(self, "_speech_encoder_cache_state", None)
        if self._model is not None:
            self._model.clear_speech_encoder_cache_state(current_cache)
            self._speech_encoder_cache_state = self._model.init_speech_encoder_cache_state()
        else:
            self._speech_encoder_cache_state = None
        self._speech_context_seconds = 0.0
        self._speech_context_audio = None

    def _reset_streaming_state(self) -> None:
        if self._model is None:
            raise RuntimeError("SimPOTrans model is not loaded.")

        self._stream_state = self._model.init_streaming_state(
            src_lang=self.source_language,
            tgt_lang=self.target_language,
        )
        self._reset_speech_buffer()

    def _record_dialogue_turn_cache(self, start_len: int) -> None:
        return

    def _current_kv_cache_len(self) -> int:
        if self._stream_state is None:
            return 0
        return _past_key_values_len(self._stream_state.past_key_values)

    def _copy_streaming_state(self) -> StreamingSTState:
        if self._stream_state is None:
            raise RuntimeError("SimPOTrans streaming state is not initialized.")
        return StreamingSTState(
            past_key_values=copy.deepcopy(self._stream_state.past_key_values)
        )

    def _copy_speech_encoder_cache_state(self) -> object | None:
        return copy.deepcopy(getattr(self, "_speech_encoder_cache_state", None))

    def _commit_speech_encoder_cache_state(self, state: object | None) -> None:
        self._speech_encoder_cache_state = state


SimPOTransBaseProcessor = BaseSimPOTransChunkProcessor


def _empty_output() -> IncrementalOutput:
    return IncrementalOutput([], "", [], "")


def _normalize_incremental_text(text: str, target_language: str) -> str:
    text = text.strip()
    if target_language in _CJK_LANGUAGES:
        return re.sub(r"\s+", "", text)

    return re.sub(r"\s+", " ", text)


def _load_model_config(
    path: Path,
    processor_config: SimpleNamespace | None = None,
) -> STModelConfig:
    with path.expanduser().open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    model_section = raw.get("model", raw)
    overrides = model_section.get("config_overrides", model_section)
    overrides = dict(overrides or {})
    runtime_overrides = getattr(
        processor_config,
        "model_config_overrides",
        None,
    )
    if runtime_overrides is not None:
        if isinstance(runtime_overrides, SimpleNamespace):
            runtime_overrides = vars(runtime_overrides)
        overrides.update(dict(runtime_overrides))

    if overrides.get("lora") is not None and not isinstance(
        overrides["lora"],
        LoraConfigSection,
    ):
        overrides["lora"] = LoraConfigSection(**overrides["lora"])
    if overrides.get("decode") is not None and not isinstance(
        overrides["decode"],
        DecodeConfig,
    ):
        overrides["decode"] = DecodeConfig(**overrides["decode"])

    for key in (
        "prompt_path",
        "decode_config_path",
    ):
        if overrides.get(key) is not None:
            overrides[key] = Path(overrides[key])

    return STModelConfig(**overrides)


def _resolve_checkpoint_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_file():
        return path

    if path.is_dir():
        model_pt = path / "model.pt"
        if model_pt.is_file():
            return model_pt

        candidates = sorted(path.glob("model/pp_*/tp_*/sdp_*.pt"))
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            raise RuntimeError(
                "Checkpoint directory contains multiple model shards. "
                "Pass the "
                f"exact shard path instead: {path}"
            )

    raise FileNotFoundError(f"SimPOTrans checkpoint not found: {path}")


def _make_decode_config(raw: object, fallback: DecodeConfig) -> DecodeConfig:
    if raw is None:
        return fallback
    if isinstance(raw, DecodeConfig):
        return raw
    if isinstance(raw, dict):
        return DecodeConfig(**{**fallback.__dict__, **raw})
    if isinstance(raw, SimpleNamespace):
        return DecodeConfig(**{**fallback.__dict__, **vars(raw)})

    return fallback
