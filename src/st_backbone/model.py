from __future__ import annotations

import copy
import json
import logging
import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, final

import torch
import torch.nn as nn
from torch.nn.modules.module import _IncompatibleKeys
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from fairseq2.runtime.config_registry import ConfigRegistrar
from fairseq2.runtime.dependency import DependencyContainer

logger = logging.getLogger(__name__)

ST_MODEL_FAMILY = "llm_sst"
AUDIO_PLACEHOLDER_TOKEN = "<|audio_here|>"


class SpeechEncoderNotReadyError(ValueError):
    """Raised when a streaming encoder has not accumulated enough audio."""


@dataclass(frozen=True)
class WhisperFrontend:
    log_mel_spectrogram: Any
    pad_or_trim: Any
    n_samples: int
    hop_length: int
    n_mels: int


@dataclass(frozen=True)
class WhisperRTFrontend:
    log_mel_spectrogram: Any
    pad_or_trim: Any
    spectrogram_stream_factory: Any
    n_samples: int
    hop_length: int
    n_mels: int
    sample_rate: int
    chunk_size_ms: int
    gran: int
    initial_gran: int


@dataclass
class WhisperRTStreamingState:
    enc_kv_cache: dict[Any, Tensor] = field(default_factory=dict)
    spectrogram_stream: Any | None = None
    mel: Tensor | None = None
    processed_samples: int = 0

    def __deepcopy__(self, memo: dict[int, Any]) -> "WhisperRTStreamingState":
        return WhisperRTStreamingState(
            enc_kv_cache={
                key: value.detach().clone()
                for key, value in self.enc_kv_cache.items()
                if isinstance(value, Tensor)
            },
            spectrogram_stream=copy.deepcopy(self.spectrogram_stream, memo),
            mel=None if self.mel is None else self.mel.detach().clone(),
            processed_samples=self.processed_samples,
        )


@dataclass
class StreamingSTState:
    past_key_values: Any | None = None


@dataclass(kw_only=True)
class LoraConfigSection:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj"]
    )
    inference_mode: bool = False


@dataclass(kw_only=True)
class DecodeConfig:
    max_new_tokens: int = 128
    do_sample: bool = False
    beam_num: int = 1
    top_p: float | None = None
    top_k: int | None = None
    temperature: float = 0.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    min_length: int = 1
    seed: int = 42


@dataclass(kw_only=True)
class STModelConfig:
    src_lang: str = "en"
    tgt_lang: str = "de"
    speech_encoder_backend: str = "whisper"
    whisper_model: str = "large-v3"
    whisper_rt_model: str = "large-v2"
    whisper_rt_chunk_size_ms: int = 300
    whisper_rt_multilingual: bool = True
    llm_path: str = "Qwen/Qwen3-8B"
    prompt_path: Path | None = None
    application_background: str | None = None
    decode_config_path: Path | None = None
    freeze_encoder: bool = True
    sample_rate: int = 16_000
    num_projection_layers: int = 2
    torch_dtype: str = "float32"
    low_cpu_mem_usage: bool = True
    lora: LoraConfigSection | None = field(default_factory=LoraConfigSection)
    decode: DecodeConfig = field(default_factory=DecodeConfig)


@final
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids: list[int]) -> None:
        self._stop_ids = tuple(int(stop_id) for stop_id in stop_ids)

    def __call__(
        self, input_ids: Tensor, scores: Tensor, **kwargs: object
    ) -> Tensor:
        if not self._stop_ids:
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        stop_ids = torch.tensor(self._stop_ids, dtype=input_ids.dtype, device=input_ids.device)
        return (input_ids[:, -1:].eq(stop_ids)).any(dim=1)


@final
class STModel(nn.Module):
    """Speech encoder + LLM adaptor model for speech-to-text training."""

    lang_codes = {
        "en": "English",
        "de": "German",
        "zh": "Chinese",
        "ja": "Japanese",
        "fr": "French",
        "es": "Spanish",
        "it": "Italian",
        "ru": "Russian",
        "ro": "Romanian",
        "nl": "Dutch",
        "pt": "Portuguese",
    }

    def __init__(
        self,
        config: STModelConfig,
        *,
        processor: WhisperFrontend,
        audio_encoder: nn.Module,
        llm: nn.Module,
        llm_tokenizer: Any,
        ln_speech: nn.Module,
        llm_proj: nn.Module,
    ) -> None:
        super().__init__()

        self.config = config
        self.processor = processor
        self.audio_encoder = audio_encoder
        self.llm = llm
        self.llm_tokenizer = llm_tokenizer
        self.ln_speech = ln_speech
        self.llm_proj = llm_proj
        self.num_updates = 0

        self.prompt = _load_prompt(config.prompt_path)
        self.decode_config = _load_decode_config(config.decode_config_path, config.decode)

        self.audio_here_id = self._require_token_id(AUDIO_PLACEHOLDER_TOKEN)
        self.eos_token_id = int(self.llm_tokenizer.eos_token_id)

        if self.llm_tokenizer.pad_token_id is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

    def train(self, mode: bool = True) -> STModel:
        super().train(mode)
        if self.config.freeze_encoder:
            self.audio_encoder.eval()
        return self

    def state_dict(
        self,
        *args: Any,
        destination: OrderedDict[str, Tensor] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> OrderedDict[str, Tensor]:
        """Return a compact checkpoint state.

        Fairseq2's checkpoint manager saves whatever `model.state_dict()` returns.
        We keep the immutable pretrained weights out of the training checkpoint and
        only persist the trainable ST adaptor plus LoRA adapter weights.
        """
        if args:
            if len(args) > 3:
                raise TypeError("state_dict() received too many positional arguments")
            if len(args) >= 1:
                destination = args[0]
            if len(args) >= 2:
                prefix = args[1]
            if len(args) >= 3:
                keep_vars = args[2]

        if destination is None:
            destination = OrderedDict()

        def add(name: str, value: Tensor) -> None:
            destination[prefix + name] = value if keep_vars else value.detach()

        add("compact_checkpoint_version", torch.tensor(1, dtype=torch.long))

        for name, value in _get_llm_adapter_state_dict(self.llm).items():
            add(f"llm_lora.{name}", value)

        for name, value in self.llm_proj.state_dict(keep_vars=keep_vars).items():
            add(f"llm_proj.{name}", value)

        for name, value in self.ln_speech.state_dict(keep_vars=keep_vars).items():
            add(f"ln_speech.{name}", value)

        token_ids = self._checkpoint_token_ids()
        if token_ids.numel() > 0:
            add("llm_special_token_ids", token_ids)

            input_embeddings = self.llm.get_input_embeddings()
            input_rows = input_embeddings.weight.index_select(
                0, token_ids.to(input_embeddings.weight.device)
            )
            add("llm_input_embedding_rows", input_rows)

            output_embeddings = self.llm.get_output_embeddings()
            if output_embeddings is not None and output_embeddings is not input_embeddings:
                output_rows = output_embeddings.weight.index_select(
                    0, token_ids.to(output_embeddings.weight.device)
                )
                add("llm_output_embedding_rows", output_rows)

        return destination

    def load_state_dict(
        self,
        state_dict: dict[str, Tensor],
        strict: bool = True,
        assign: bool = False,
    ) -> _IncompatibleKeys:
        state_dict = _strip_known_wrapper_prefixes(state_dict)

        if "compact_checkpoint_version" not in state_dict:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)

        llm_lora_state = _get_sub_state_dict(state_dict, "llm_lora.")
        if llm_lora_state:
            _set_llm_adapter_state_dict(self.llm, llm_lora_state)

        try:
            self.llm_proj.load_state_dict(
                _get_sub_state_dict(state_dict, "llm_proj."), strict=True, assign=assign
            )
            self.ln_speech.load_state_dict(
                _get_sub_state_dict(state_dict, "ln_speech."), strict=True, assign=assign
            )
        except RuntimeError as ex:
            raise RuntimeError(
                "Checkpoint speech adaptor dimensions do not match the current "
                f"{self.config.speech_encoder_backend!r} encoder. Train a new "
                "adaptor/LoRA checkpoint for this speech_encoder_backend."
            ) from ex
        self._restore_checkpoint_embedding_rows(state_dict)

        return _IncompatibleKeys([], [])

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def set_num_updates(self, num_updates: int) -> None:
        self.num_updates = num_updates

    def forward(
        self,
        batch: Any | None = None,
        *,
        src_tokens: list[Tensor] | None = None,
        target_texts: list[str] | None = None,
        src_langs: list[str] | None = None,
        tgt_langs: list[str] | None = None,
        **kwargs: object,
    ) -> dict[str, Tensor]:
        if batch is not None:
            src_tokens = getattr(batch, "source")
            target_texts = getattr(batch, "target_texts")
            src_langs = getattr(batch, "src_langs")
            tgt_langs = getattr(batch, "tgt_langs")

        if src_tokens is None or target_texts is None:
            raise ValueError("`src_tokens` and `target_texts` are required.")

        batch_size = len(src_tokens)
        src_langs = self._normalize_langs(src_langs, self.config.src_lang, batch_size)
        tgt_langs = self._normalize_langs(tgt_langs, self.config.tgt_lang, batch_size)

        speech_embeds = self.encode_speech(src_tokens)
        speech_embeds = self.forward_adaptor(speech_embeds)

        prompts = self._render_prompts(src_langs, tgt_langs)
        llm_input, llm_target = self._build_llm_inputs(
            prompts, speech_embeds, target_texts
        )

        outputs = self.llm(inputs_embeds=llm_input, labels=llm_target)

        return {"st_loss": outputs.loss, "loss": outputs.loss}

    @torch.no_grad()
    def generate(
        self,
        src_tokens: list[Tensor],
        *,
        src_langs: list[str] | None = None,
        tgt_langs: list[str] | None = None,
        decode_config: DecodeConfig | None = None,
    ) -> Tensor:
        batch_size = len(src_tokens)
        src_langs = self._normalize_langs(src_langs, self.config.src_lang, batch_size)
        tgt_langs = self._normalize_langs(tgt_langs, self.config.tgt_lang, batch_size)

        speech_embeds = self.forward_adaptor(self.encode_speech(src_tokens))
        prompts = self._render_prompts(src_langs, tgt_langs)
        tokenized = self._tokenize_prompts(prompts)
        wrapped_embeds = self.wrap_prompt(speech_embeds, tokenized)

        cfg = decode_config or self.decode_config
        stopping = StoppingCriteriaList([StopOnTokens([self.eos_token_id])])

        if cfg.do_sample:
            torch.manual_seed(cfg.seed)

        generation_kwargs = _decode_generation_kwargs(cfg)
        outputs = self.llm.generate(
            inputs_embeds=wrapped_embeds,
            eos_token_id=[self.eos_token_id],
            pad_token_id=self.llm_tokenizer.pad_token_id,
            stopping_criteria=stopping,
            max_new_tokens=cfg.max_new_tokens,
            num_beams=cfg.beam_num,
            do_sample=cfg.do_sample,
            min_length=cfg.min_length,
            return_dict_in_generate=True,
            output_logits=True,
            **generation_kwargs,
        )

        return outputs.sequences

    def decode(self, tokens: Tensor) -> list[str]:
        return self.llm_tokenizer.batch_decode(tokens, skip_special_tokens=True)

    @torch.no_grad()
    def init_streaming_state(
        self,
        *,
        src_lang: str | None = None,
        tgt_lang: str | None = None,
    ) -> StreamingSTState:
        src_lang = src_lang or self.config.src_lang
        tgt_lang = tgt_lang or self.config.tgt_lang
        system_messages, _ = self._render_streaming_messages(src_lang, tgt_lang)
        state = StreamingSTState()

        if not system_messages:
            return state

        tokenized = self.llm_tokenizer.apply_chat_template(
            system_messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
            return_tensors="pt",
        ).to(self.device)
        if tokenized.numel() == 0:
            return state

        inputs_embeds = self.llm.get_input_embeddings()(tokenized)
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            return_dict=True,
        )
        state.past_key_values = outputs.past_key_values

        return state

    @torch.no_grad()
    def generate_streaming_turn(
        self,
        src_token: Tensor,
        state: StreamingSTState,
        *,
        current_num_samples: int | None = None,
        src_lang: str | None = None,
        tgt_lang: str | None = None,
        decode_config: DecodeConfig | None = None,
        speech_encoder_cache_state: Any | None = None,
        is_final: bool = False,
    ) -> tuple[Tensor, StreamingSTState]:
        src_lang = src_lang or self.config.src_lang
        tgt_lang = tgt_lang or self.config.tgt_lang

        if state.past_key_values is None:
            state = self.init_streaming_state(src_lang=src_lang, tgt_lang=tgt_lang)

        if current_num_samples is None:
            current_num_samples = int(src_token.numel())
        speech_embeds = self.forward_adaptor(
            self.encode_speech_context(
                [src_token],
                current_num_samples,
                encoder_cache_state=speech_encoder_cache_state,
                is_final=is_final,
            )
        )
        turn_embeds = self._streaming_turn_embeds(speech_embeds, src_lang, tgt_lang)

        cfg = decode_config or self.decode_config
        stopping = StoppingCriteriaList([StopOnTokens([self.eos_token_id])])

        if cfg.do_sample:
            torch.manual_seed(cfg.seed)

        generation_past_key_values = _expand_past_key_values_for_beams(
            state.past_key_values, cfg.beam_num
        )
        generation_kwargs = _decode_generation_kwargs(cfg)
        outputs = self.llm.generate(
            inputs_embeds=turn_embeds,
            attention_mask=_cache_attention_mask(
                generation_past_key_values, turn_embeds.size(1), self.device
            ),
            cache_position=_cache_position(
                generation_past_key_values, turn_embeds.size(1), self.device
            ),
            past_key_values=generation_past_key_values,
            eos_token_id=[self.eos_token_id],
            pad_token_id=self.llm_tokenizer.pad_token_id,
            stopping_criteria=stopping,
            max_new_tokens=cfg.max_new_tokens,
            num_beams=cfg.beam_num,
            do_sample=cfg.do_sample,
            min_length=cfg.min_length,
            return_dict_in_generate=True,
            output_logits=True,
            use_cache=True,
            **generation_kwargs,
        )

        next_state = StreamingSTState(past_key_values=state.past_key_values)
        self._append_embeds_to_streaming_state(next_state, turn_embeds)
        self._append_token_ids_to_streaming_state(next_state, outputs.sequences)
        self._append_text_to_streaming_state(next_state, "\n")

        return outputs.sequences, next_state

    def encode_speech(
        self,
        waveforms: list[Tensor],
        *,
        encoder_cache_state: Any | None = None,
    ) -> Tensor:
        del encoder_cache_state
        if _is_whisper_rt_backend(self.config.speech_encoder_backend):
            return self._encode_speech_whisper_rt(waveforms)

        return self._encode_speech_whisper(waveforms)

    def _encode_speech_whisper(self, waveforms: list[Tensor]) -> Tensor:
        features, feature_lens = self.extract_features(waveforms)

        if self.dtype == torch.float16:
            features = features.half()
        elif self.dtype == torch.bfloat16:
            features = features.bfloat16()

        max_len = int(feature_lens.max().item())
        max_len = max(1, min(max_len, features.size(-1) // 2))
        encoder_output = self.audio_encoder(features)

        return encoder_output[:, :max_len, :]

    def _encode_speech_whisper_rt(self, waveforms: list[Tensor]) -> Tensor:
        features, feature_lens = self.extract_features(waveforms)
        features = self._cast_speech_features(features)
        max_len = int(feature_lens.max().item())
        max_len = max(1, min(max_len, features.size(-1) // 2))

        original_stream = getattr(self.audio_encoder, "use_stream", None)
        original_mask = getattr(self.audio_encoder, "use_mask", None)
        if hasattr(self.audio_encoder, "_use_stream"):
            self.audio_encoder._use_stream(False)
        if hasattr(self.audio_encoder, "_use_mask"):
            self.audio_encoder._use_mask(True)
        try:
            try:
                encoder_output = self.audio_encoder(
                    features,
                    index=[0, max_len],
                    mask=True,
                )
            except TypeError:
                encoder_output = self.audio_encoder(features)
        finally:
            if original_stream is not None and hasattr(self.audio_encoder, "_use_stream"):
                self.audio_encoder._use_stream(bool(original_stream))
            if original_mask is not None and hasattr(self.audio_encoder, "_use_mask"):
                self.audio_encoder._use_mask(bool(original_mask))

        return encoder_output[:, :max_len, :]

    def encode_speech_context(
        self,
        waveforms: list[Tensor],
        current_num_samples: int,
        *,
        encoder_cache_state: Any | None = None,
        is_final: bool = False,
    ) -> Tensor:
        if _is_whisper_rt_backend(self.config.speech_encoder_backend):
            if isinstance(encoder_cache_state, WhisperRTStreamingState):
                return self._encode_whisper_rt_context(
                    waveforms,
                    current_num_samples,
                    encoder_cache_state,
                    is_final=is_final,
                )
            encoder_output = self.encode_speech(waveforms)
        else:
            del encoder_cache_state
            encoder_output = self.encode_speech(waveforms)

        current_len = max(
            1, math.ceil(current_num_samples / (self.processor.hop_length * 2))
        )
        current_len = min(current_len, encoder_output.size(1))

        return encoder_output[:, -current_len:, :]

    def _encode_whisper_rt_context(
        self,
        waveforms: list[Tensor],
        current_num_samples: int,
        state: WhisperRTStreamingState,
        *,
        is_final: bool,
    ) -> Tensor:
        if len(waveforms) != 1:
            raise ValueError("WhisperRT streaming encoder cache supports batch size 1.")
        if not isinstance(self.processor, WhisperRTFrontend):
            raise RuntimeError("WhisperRT streaming requires a WhisperRTFrontend.")

        audio = waveforms[0].detach().float()
        if audio.ndim == 2:
            audio = audio.mean(dim=0)
        audio = audio.reshape(-1).to(self.device)

        total_samples = int(audio.numel())
        if total_samples <= 0:
            raise ValueError("WhisperRT streaming received empty audio.")
        if state.processed_samples > total_samples:
            self.clear_speech_encoder_cache_state(state)

        chunk_samples = max(
            1,
            int(round(self.processor.sample_rate * self.processor.chunk_size_ms / 1000.0)),
        )
        hidden_chunks: list[Tensor] = []
        if state.spectrogram_stream is None:
            state.spectrogram_stream = self.processor.spectrogram_stream_factory(
                n_mels=self.processor.n_mels
            )

        while state.processed_samples < total_samples:
            remaining = total_samples - state.processed_samples
            if remaining < chunk_samples and not is_final:
                break
            take = remaining if remaining < chunk_samples else chunk_samples
            start = state.processed_samples
            end = start + take
            frame = audio[start:end]
            mel_frame = state.spectrogram_stream.calc_mel_with_new_frame(
                frame,
                is_last=is_final and end >= total_samples,
            )
            state.mel = mel_frame if state.mel is None else torch.cat(
                [state.mel, mel_frame], dim=-1
            )
            state.processed_samples = end

            if state.mel.size(-1) < self.processor.initial_gran * 2:
                continue

            features = self._cast_speech_features(state.mel)
            cache, hooks = _install_whisper_rt_encoder_kv_cache_hooks(
                self.audio_encoder,
                state.enc_kv_cache,
            )
            state.enc_kv_cache = cache
            original_stream = getattr(self.audio_encoder, "use_stream", None)
            if hasattr(self.audio_encoder, "_use_stream"):
                self.audio_encoder._use_stream(True)
            try:
                encoded = self.audio_encoder(
                    features,
                    kv_cache=state.enc_kv_cache,
                    mask=None,
                )
            finally:
                for hook in hooks:
                    hook.remove()
                if original_stream is not None and hasattr(self.audio_encoder, "_use_stream"):
                    self.audio_encoder._use_stream(bool(original_stream))

            if encoded.numel() > 0:
                hidden_chunks.append(encoded)

        if not hidden_chunks:
            raise SpeechEncoderNotReadyError(
                "WhisperRT streaming encoder produced no states yet. Accumulate at "
                f"least {self.processor.initial_gran * 20}ms for the initial "
                "WhisperRT window, or increase CHUNK_SIZE_SECONDS."
            )

        return torch.cat(hidden_chunks, dim=1)

    def _cast_speech_features(self, features: Tensor) -> Tensor:
        encoder_dtype = _module_parameter_dtype(self.audio_encoder, self.dtype)
        if encoder_dtype == torch.float16:
            return features.half()
        if encoder_dtype == torch.bfloat16:
            return features.bfloat16()
        return features

    def init_speech_encoder_cache_state(self) -> WhisperRTStreamingState | None:
        if _is_whisper_rt_backend(self.config.speech_encoder_backend):
            return WhisperRTStreamingState()
        return None

    def clear_speech_encoder_cache_state(self, state: Any | None) -> None:
        if isinstance(state, WhisperRTStreamingState):
            state.enc_kv_cache.clear()
            state.spectrogram_stream = None
            state.mel = None
            state.processed_samples = 0

    def extract_features(self, waveforms: list[Tensor]) -> tuple[Tensor, Tensor]:
        mels = []
        token_lens = []
        for waveform in waveforms:
            audio = waveform.detach().float()
            if audio.ndim == 2:
                audio = audio.mean(dim=0)

            num_samples = min(int(audio.numel()), self.processor.n_samples)
            token_lens.append(
                max(1, math.ceil(num_samples / (self.processor.hop_length * 2)))
            )

            audio = self.processor.pad_or_trim(audio, length=self.processor.n_samples)
            mel = self.processor.log_mel_spectrogram(
                audio,
                n_mels=self.processor.n_mels,
                device=self.device,
            )
            mels.append(mel)

        features = torch.stack(mels, dim=0)
        lens = torch.tensor(token_lens, dtype=torch.long, device=self.device)

        return features, lens

    def forward_adaptor(self, encoder_states: Tensor) -> Tensor:
        ln_dtype = _module_parameter_dtype(self.ln_speech, encoder_states.dtype)
        proj_dtype = _module_parameter_dtype(self.llm_proj, ln_dtype)
        llm_dtype = self.llm.get_input_embeddings().weight.dtype

        hidden_states = self.ln_speech(encoder_states.to(dtype=ln_dtype))
        speech_embeds = self.llm_proj(hidden_states.to(dtype=proj_dtype))

        return speech_embeds.to(dtype=llm_dtype)

    def wrap_prompt(self, speech_embeds: Tensor, tokenized_prompts: Any) -> Tensor:
        prompt_ids = tokenized_prompts.input_ids.to(self.device)
        prompt_mask = tokenized_prompts.attention_mask.to(self.device)
        prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)

        sequences = []
        for row_idx in range(prompt_ids.size(0)):
            prompt_len = int(prompt_mask[row_idx].sum().item())
            audio_positions = (prompt_ids[row_idx, :prompt_len] == self.audio_here_id).nonzero(
                as_tuple=True
            )[0]

            if len(audio_positions) != 1:
                raise ValueError(
                    f"Prompt must contain exactly one {AUDIO_PLACEHOLDER_TOKEN} token."
                )

            audio_idx = int(audio_positions.item())
            before = prompt_embeds[row_idx, :audio_idx]
            after = prompt_embeds[row_idx, audio_idx + 1 : prompt_len]

            sequences.append(torch.cat([before, speech_embeds[row_idx], after], dim=0))

        return nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)

    def _streaming_turn_embeds(
        self,
        speech_embeds: Tensor,
        src_lang: str,
        tgt_lang: str,
    ) -> Tensor:
        system_messages, user_message = self._render_streaming_messages(src_lang, tgt_lang)
        if user_message is None:
            raise ValueError(
                f"Streaming prompt must contain one user message with {AUDIO_PLACEHOLDER_TOKEN}."
            )

        full_ids = self.llm_tokenizer.apply_chat_template(
            [*system_messages, user_message],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        ).to(self.device)
        if full_ids.ndim == 2:
            full_ids = full_ids[0]

        if system_messages:
            system_ids = self.llm_tokenizer.apply_chat_template(
                system_messages,
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
                return_tensors="pt",
            ).to(self.device)
            if system_ids.ndim == 2:
                system_ids = system_ids[0]
            if (
                full_ids.numel() >= system_ids.numel()
                and torch.equal(full_ids[: system_ids.numel()], system_ids)
            ):
                full_ids = full_ids[system_ids.numel() :]
            else:
                logger.warning("Could not strip cached system prompt from streaming turn.")

        prompt_ids = full_ids.unsqueeze(0)
        tokenized_prompt = SimpleNamespace(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
        )

        return self.wrap_prompt(speech_embeds, tokenized_prompt)

    def _render_streaming_messages(
        self, src_lang: str, tgt_lang: str
    ) -> tuple[list[dict[str, str]], dict[str, str] | None]:
        rendered = self._render_prompts([src_lang], [tgt_lang])[0]
        system_messages = []
        user_message = None

        for message in rendered:
            content = message["content"]
            if isinstance(content, list):
                content = "".join(content)

            if AUDIO_PLACEHOLDER_TOKEN in content:
                if user_message is not None:
                    raise ValueError(
                        f"Streaming prompt must contain exactly one {AUDIO_PLACEHOLDER_TOKEN} message."
                    )
                user_message = {**message, "content": content}
            elif user_message is None:
                system_messages.append({**message, "content": content})

        return system_messages, user_message

    def _append_text_to_streaming_state(
        self, state: StreamingSTState, text: str
    ) -> None:
        token_ids = self.llm_tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(self.device)
        if token_ids.numel() == 0:
            return

        self._append_token_ids_to_streaming_state(state, token_ids)

    def _append_token_ids_to_streaming_state(
        self, state: StreamingSTState, token_ids: Tensor
    ) -> None:
        inputs_embeds = self.llm.get_input_embeddings()(token_ids)
        self._append_embeds_to_streaming_state(state, inputs_embeds)

    def _append_embeds_to_streaming_state(
        self, state: StreamingSTState, inputs_embeds: Tensor
    ) -> None:
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=_cache_attention_mask(
                state.past_key_values, inputs_embeds.size(1), self.device
            ),
            cache_position=_cache_position(
                state.past_key_values, inputs_embeds.size(1), self.device
            ),
            past_key_values=state.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        state.past_key_values = outputs.past_key_values

    def _build_llm_inputs(
        self,
        prompts: list[list[dict[str, str]]],
        speech_embeds: Tensor,
        target_texts: list[str],
    ) -> tuple[Tensor, Tensor]:
        tokenized_prompts = self._tokenize_prompts(prompts)
        wrapped_embeds = self.wrap_prompt(speech_embeds, tokenized_prompts)

        targets = [f"{text}{self.llm_tokenizer.eos_token}" for text in target_texts]
        tokenized_targets = self.llm_tokenizer(
            targets, return_tensors="pt", padding=True
        ).input_ids.to(self.device)
        target_embeds = self.llm.get_input_embeddings()(tokenized_targets)

        llm_input = torch.cat([wrapped_embeds, target_embeds], dim=1)
        prompt_target = torch.full(
            (len(targets), wrapped_embeds.size(1)),
            -100,
            dtype=torch.long,
            device=self.device,
        )
        llm_target = torch.cat([prompt_target, tokenized_targets], dim=1)
        llm_target = llm_target.masked_fill(
            llm_target == self.llm_tokenizer.pad_token_id, -100
        )

        return llm_input, llm_target

    def _tokenize_prompts(self, prompts: list[list[dict[str, str]]]) -> Any:
        rendered = self.llm_tokenizer.apply_chat_template(
            prompts,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        return self.llm_tokenizer(rendered, return_tensors="pt", padding=True)

    def _render_prompts(
        self, src_langs: list[str], tgt_langs: list[str]
    ) -> list[list[dict[str, str]]]:
        prompts = [copy.deepcopy(self.prompt) for _ in src_langs]

        for prompt, src_lang, tgt_lang in zip(prompts, src_langs, tgt_langs):
            application_background = str(
                getattr(self.config, "application_background", None) or ""
            ).strip()
            for message in prompt:
                content = message["content"]
                if isinstance(content, list):
                    content = "".join(content)

                content = content.replace("<src_lang>", self._lang_name(src_lang))
                content = content.replace("<tgt_lang>", self._lang_name(tgt_lang))
                if message.get("role") == "system" and application_background:
                    background = application_background.replace(
                        "<src_lang>", self._lang_name(src_lang)
                    ).replace("<tgt_lang>", self._lang_name(tgt_lang))
                    content = f"{content}\n{background}" if content else background
                message["content"] = content

        return prompts

    def _lang_name(self, lang: str) -> str:
        return lang

    def _normalize_langs(
        self, langs: list[str] | None, default: str, batch_size: int
    ) -> list[str]:
        if langs is None:
            return [default for _ in range(batch_size)]

        return [lang or default for lang in langs]

    def _require_token_id(self, token: str) -> int:
        token_id = self.llm_tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == self.llm_tokenizer.unk_token_id:
            raise ValueError(f"`{token}` is missing from the LLM tokenizer.")

        return int(token_id)

    def _checkpoint_token_ids(self) -> Tensor:
        token_ids: list[int] = []
        unk_id = self.llm_tokenizer.unk_token_id

        token_id = self.llm_tokenizer.convert_tokens_to_ids(AUDIO_PLACEHOLDER_TOKEN)
        if token_id is not None and token_id != unk_id:
            token_ids.append(int(token_id))

        return torch.tensor(sorted(set(token_ids)), dtype=torch.long)

    def _restore_checkpoint_embedding_rows(self, state_dict: dict[str, Tensor]) -> None:
        token_ids = state_dict.get("llm_special_token_ids")
        input_rows = state_dict.get("llm_input_embedding_rows")
        if token_ids is None or input_rows is None:
            return

        input_embeddings = self.llm.get_input_embeddings()
        ids = token_ids.to(input_embeddings.weight.device)
        rows = input_rows.to(
            device=input_embeddings.weight.device, dtype=input_embeddings.weight.dtype
        )

        with torch.no_grad():
            input_embeddings.weight.index_copy_(0, ids, rows)

            output_rows = state_dict.get("llm_output_embedding_rows")
            output_embeddings = self.llm.get_output_embeddings()
            if output_rows is not None and output_embeddings is not None:
                rows = output_rows.to(
                    device=output_embeddings.weight.device,
                    dtype=output_embeddings.weight.dtype,
                )
                output_embeddings.weight.index_copy_(0, ids, rows)


def create_st_model(config: STModelConfig) -> STModel:
    torch_dtype = _resolve_torch_dtype(config.torch_dtype)
    backend = _normalize_speech_encoder_backend(config.speech_encoder_backend)
    if backend == "whisper":
        processor, audio_encoder, encoder_dim = _build_openai_whisper_encoder(
            config, torch_dtype
        )
    elif backend == "whisper_rt":
        processor, audio_encoder, encoder_dim = _build_whisper_rt_encoder(
            config, torch_dtype
        )
    else:
        raise AssertionError(f"Unhandled speech encoder backend: {backend}")

    if config.freeze_encoder:
        audio_encoder.requires_grad_(False)
        audio_encoder.eval()
        logger.info("Frozen %s encoder parameters.", config.speech_encoder_backend)

    llm, llm_tokenizer = _build_llm(config, torch_dtype)
    llm_dim = llm.config.hidden_size

    model = STModel(
        config,
        processor=processor,
        audio_encoder=audio_encoder,
        llm=llm,
        llm_tokenizer=llm_tokenizer,
        ln_speech=nn.LayerNorm(encoder_dim),
        llm_proj=_build_projection(encoder_dim, llm_dim, config.num_projection_layers),
    )
    _configure_trainable_parameters(model)

    return model


def _build_openai_whisper_encoder(
    config: STModelConfig, torch_dtype: torch.dtype
) -> tuple[WhisperFrontend, nn.Module, int]:
    import whisper
    from whisper.audio import HOP_LENGTH, N_SAMPLES, log_mel_spectrogram, pad_or_trim

    model = whisper.load_model(config.whisper_model, device="cpu")
    audio_encoder = model.encoder
    audio_encoder.to(dtype=torch_dtype)
    _keep_whisper_layer_norms_fp32(audio_encoder)
    encoder_dim = int(model.dims.n_audio_state)
    frontend = WhisperFrontend(
        log_mel_spectrogram=log_mel_spectrogram,
        pad_or_trim=pad_or_trim,
        n_samples=int(N_SAMPLES),
        hop_length=int(HOP_LENGTH),
        n_mels=int(model.dims.n_mels),
    )

    return frontend, audio_encoder, encoder_dim


def _build_whisper_rt_encoder(
    config: STModelConfig, torch_dtype: torch.dtype
) -> tuple[WhisperRTFrontend, nn.Module, int]:
    if int(config.whisper_rt_chunk_size_ms) % 20 != 0:
        raise ValueError("`whisper_rt_chunk_size_ms` must be a multiple of 20ms.")

    try:
        import whisper_rt
        from whisper_rt.audio import HOP_LENGTH, N_SAMPLES, SpectrogramStream
        from whisper_rt.audio import log_mel_spectrogram, pad_or_trim
    except ImportError as ex:
        raise RuntimeError(
            "WhisperRT backend requires the vendored `whisper_rt` package. "
            "Run `uv sync` after installing the local whisper-rt-streaming source."
        ) from ex

    try:
        model = whisper_rt.load_streaming_model(
            name=config.whisper_rt_model,
            gran=int(config.whisper_rt_chunk_size_ms),
            multilingual=bool(config.whisper_rt_multilingual),
            device="cpu",
        )
    except Exception as ex:
        message = str(ex)
        if "token" in message.lower() or "huggingface" in message.lower() or "401" in message:
            raise RuntimeError(
                "Could not load WhisperRT from Hugging Face. The upstream "
                "WhisperRT checkpoint requires read access; run "
                "`huggingface-cli login` or set `HF_TOKEN`, then retry."
            ) from ex
        raise

    audio_encoder = model.encoder
    audio_encoder.to(dtype=torch_dtype)
    _keep_whisper_layer_norms_fp32(audio_encoder)
    encoder_dim = int(model.dims.n_audio_state)
    gran = int(getattr(audio_encoder, "gran", int(config.whisper_rt_chunk_size_ms) // 20))
    extra_gran_blocks = int(getattr(audio_encoder, "extra_gran_blocks", 1))
    frontend = WhisperRTFrontend(
        log_mel_spectrogram=log_mel_spectrogram,
        pad_or_trim=pad_or_trim,
        spectrogram_stream_factory=SpectrogramStream,
        n_samples=int(N_SAMPLES),
        hop_length=int(HOP_LENGTH),
        n_mels=int(model.dims.n_mels),
        sample_rate=int(config.sample_rate),
        chunk_size_ms=int(config.whisper_rt_chunk_size_ms),
        gran=gran,
        initial_gran=gran * (1 + extra_gran_blocks),
    )

    return frontend, audio_encoder, encoder_dim


def _normalize_speech_encoder_backend(backend: str) -> str:
    normalized = backend.lower().replace("-", "_")
    if normalized in {"whisper", "openai_whisper", "openaiwhisper"}:
        return "whisper"
    if normalized in {"whisper_rt", "whisperrt"}:
        return "whisper_rt"

    raise ValueError(
        "Unsupported speech_encoder_backend "
        f"{backend!r}. Expected 'whisper' or 'whisper_rt'."
    )


def _is_whisper_rt_backend(backend: str) -> bool:
    return _normalize_speech_encoder_backend(backend) == "whisper_rt"


def _install_whisper_rt_encoder_kv_cache_hooks(
    encoder: nn.Module,
    cache: dict[Any, Tensor] | None = None,
) -> tuple[dict[Any, Tensor], list[Any]]:
    cache = {} if cache is None else cache
    hooks = []

    def save_to_cache(module: nn.Module, _: tuple[Any, ...], output: Tensor) -> Tensor:
        if module not in cache:
            cache[module] = output.detach()
        else:
            cache[module] = torch.cat([cache[module], output], dim=1).detach()
        return cache[module]

    def install_hooks(layer: nn.Module) -> None:
        if layer.__class__.__name__ == "LoRAMultiHeadAttention":
            hooks.append(layer.key.register_forward_hook(save_to_cache))
            hooks.append(layer.value.register_forward_hook(save_to_cache))

    encoder.apply(install_hooks)
    return cache, hooks


def _keep_whisper_layer_norms_fp32(module: nn.Module) -> None:
    # Whisper LayerNorm parameters should stay fp32 when the encoder runs in bf16.
    for child in module.modules():
        if isinstance(child, nn.LayerNorm):
            child.float()


def _module_parameter_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    for parameter in module.parameters():
        return parameter.dtype
    return fallback


def _decode_generation_kwargs(cfg: DecodeConfig) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if cfg.do_sample:
        kwargs["temperature"] = cfg.temperature
        if cfg.top_p is not None:
            kwargs["top_p"] = cfg.top_p
        if cfg.top_k is not None:
            kwargs["top_k"] = cfg.top_k
    if cfg.repetition_penalty != 1.0:
        kwargs["repetition_penalty"] = cfg.repetition_penalty
    if cfg.no_repeat_ngram_size > 0:
        kwargs["no_repeat_ngram_size"] = cfg.no_repeat_ngram_size

    return kwargs


def _expand_past_key_values_for_beams(
    past_key_values: Any | None, beam_num: int
) -> Any | None:
    if past_key_values is None or beam_num <= 1:
        return past_key_values

    if hasattr(past_key_values, "batch_repeat_interleave"):
        expanded = copy.deepcopy(past_key_values)
        expanded.batch_repeat_interleave(beam_num)
        return expanded

    if isinstance(past_key_values, tuple):
        return tuple(_repeat_cache_layer_for_beams(layer, beam_num) for layer in past_key_values)

    logger.warning(
        "Cannot expand past_key_values of type %s for beam search; generation may fail.",
        type(past_key_values).__name__,
    )
    return past_key_values


def _repeat_cache_layer_for_beams(layer: Any, beam_num: int) -> Any:
    if isinstance(layer, Tensor):
        return layer.repeat_interleave(beam_num, dim=0)
    if isinstance(layer, tuple):
        return tuple(_repeat_cache_layer_for_beams(item, beam_num) for item in layer)
    if isinstance(layer, list):
        return [_repeat_cache_layer_for_beams(item, beam_num) for item in layer]
    if isinstance(layer, dict):
        return {
            key: _repeat_cache_layer_for_beams(value, beam_num)
            for key, value in layer.items()
        }

    return layer


def _cache_attention_mask(
    past_key_values: Any | None,
    current_len: int,
    device: torch.device,
) -> Tensor:
    past_len = _past_key_values_len(past_key_values)
    return torch.ones((1, past_len + current_len), dtype=torch.long, device=device)


def _cache_position(
    past_key_values: Any | None,
    current_len: int,
    device: torch.device,
) -> Tensor:
    past_len = _past_key_values_len(past_key_values)
    return torch.arange(past_len, past_len + current_len, dtype=torch.long, device=device)


def _past_key_values_len(past_key_values: Any | None) -> int:
    if past_key_values is None:
        return 0

    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())

    if isinstance(past_key_values, tuple) and past_key_values:
        first_layer = past_key_values[0]
        if isinstance(first_layer, tuple) and first_layer:
            return int(first_layer[0].shape[-2])

    return 0



def register_st_model_configs(container: DependencyContainer) -> None:
    registrar = ConfigRegistrar(container, STModelConfig)

    @registrar("speech_to_text_base")
    def speech_to_text_base() -> STModelConfig:
        return STModelConfig()


def _build_llm(
    config: STModelConfig, torch_dtype: torch.dtype
) -> tuple[nn.Module, Any]:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = AutoTokenizer.from_pretrained(
        config.llm_path, additional_special_tokens=[AUDIO_PLACEHOLDER_TOKEN]
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        config.llm_path,
        low_cpu_mem_usage=config.low_cpu_mem_usage,
        torch_dtype=torch_dtype,
    )
    llm.resize_token_embeddings(len(tokenizer))

    if config.lora is None:
        return llm, tokenizer

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as ex:
        raise RuntimeError("LoRA training requires the `peft` package.") from ex

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=config.lora.inference_mode,
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
    )

    llm = get_peft_model(llm, lora_cfg)
    llm.print_trainable_parameters()

    return llm, tokenizer


def _configure_trainable_parameters(model: STModel) -> None:
    if model.config.freeze_encoder:
        model.audio_encoder.requires_grad_(False)

    if model.config.lora is not None:
        for param in model.llm.parameters():
            param.requires_grad = False

        for name, param in model.llm.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

    model.ln_speech.requires_grad_(True)
    model.llm_proj.requires_grad_(True)


def _get_llm_adapter_state_dict(llm: nn.Module) -> dict[str, Tensor]:
    if hasattr(llm, "peft_config"):
        try:
            from peft import get_peft_model_state_dict
        except ImportError as ex:
            raise RuntimeError("Loading a compact LoRA checkpoint requires `peft`.") from ex

        return {
            name: value
            for name, value in get_peft_model_state_dict(
                llm, save_embedding_layers=False
            ).items()
            if isinstance(value, Tensor)
        }

    return {
        name: value
        for name, value in llm.state_dict().items()
        if isinstance(value, Tensor)
    }


def _set_llm_adapter_state_dict(llm: nn.Module, state_dict: dict[str, Tensor]) -> None:
    if hasattr(llm, "peft_config"):
        try:
            from peft import set_peft_model_state_dict
        except ImportError as ex:
            raise RuntimeError("Loading a compact LoRA checkpoint requires `peft`.") from ex

        set_peft_model_state_dict(llm, state_dict)
        return

    llm.load_state_dict(state_dict, strict=True)


def _get_sub_state_dict(
    state_dict: dict[str, Tensor], prefix: str
) -> dict[str, Tensor]:
    return {
        name.removeprefix(prefix): value
        for name, value in state_dict.items()
        if name.startswith(prefix)
    }


def _strip_known_wrapper_prefixes(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    prefixes = ("module.", "_orig_mod.")

    for prefix in prefixes:
        version_key = f"{prefix}compact_checkpoint_version"
        if version_key in state_dict:
            return {
                name.removeprefix(prefix): value
                for name, value in state_dict.items()
                if name.startswith(prefix)
            }

    return dict(state_dict)


def _build_projection(input_dim: int, output_dim: int, num_layers: int) -> nn.Module:
    if num_layers < 1:
        raise ValueError("`num_projection_layers` must be greater than or equal to 1.")

    layers: list[nn.Module] = [nn.Linear(input_dim, output_dim, bias=False)]

    for _ in range(1, num_layers):
        layers.extend([nn.GELU(), nn.Linear(output_dim, output_dim, bias=False)])

    return nn.Sequential(*layers)


def _load_prompt(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return [
            {
                "role": "system",
                "content": "You are a careful speech translation assistant.",
            },
            {
                "role": "user",
                "content": (
                    "Translate the speech from <src_lang> to <tgt_lang>: "
                    f"{AUDIO_PLACEHOLDER_TOKEN}"
                ),
            },
        ]

    with path.expanduser().open("r", encoding="utf-8") as f:
        prompt = json.load(f)

    if isinstance(prompt, dict):
        prompt = prompt.get("st", prompt)

    if not isinstance(prompt, list):
        raise ValueError("Prompt JSON must be a chat-template list or contain key `st`.")

    return prompt


def _load_decode_config(path: Path | None, fallback: DecodeConfig) -> DecodeConfig:
    if path is None:
        return fallback

    import yaml

    with path.expanduser().open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    return DecodeConfig(**raw_config)


def _resolve_torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32

    raise ValueError(f"Unsupported torch dtype: {name}")
