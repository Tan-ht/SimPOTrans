from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import torch
import torchaudio.functional as F
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint

from stream_eval.processor.fixed_processor_dialogue import SimPOTransChunkProcessor
from simulstream.server.speech_processors import SAMPLE_RATE

logger = logging.getLogger(__name__)


@entrypoint
class SimPOTransFixedPolicy(SpeechToTextAgent):
    """SimulEval speech-to-text policy backed by SimPOTransChunkProcessor."""

    _loaded_signature: ClassVar[tuple | None] = None

    def __init__(self, args: Namespace) -> None:
        self.source_language = args.source_language
        self.target_language = args.target_language
        self._processed_source_samples = 0
        self._finished = False
        self.processor = None

        super().__init__(args)

        self.processor_config = self._make_processor_config(args)
        self._load_processor_model(args, self.processor_config)
        self.processor = SimPOTransChunkProcessor(self.processor_config)

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        parser.add_argument("--model-config", required=True)
        parser.add_argument("--checkpoint", default=None)
        parser.add_argument("--source-language", "--source-lang", default="en")
        parser.add_argument("--target-language", "--target-lang", default="zh")
        parser.add_argument("--prompt-path", default=None)
        parser.add_argument("--history-selection-policy", default="none")
        parser.add_argument("--max-speech-context-seconds", type=float, default=30.0)
        parser.add_argument("--soft-speech-context-reset-seconds", type=float, default=20.0)
        parser.add_argument("--fixed-history-words", type=int, default=100)
        parser.add_argument("--speech-context-policy", default="fixed_duration")
        parser.add_argument("--llm-context-turns", type=int, default=None)
        parser.add_argument("--llm-raw-kv-cache", action="store_true", default=False)
        parser.add_argument("--beam-size", type=int, default=4)
        parser.add_argument("--max-new-tokens", type=int, default=128)
        parser.add_argument("--do-sample", action="store_true", default=False)
        parser.add_argument("--temperature", type=float, default=0.0)
        parser.add_argument("--top-p", type=float, default=None)
        parser.add_argument("--top-k", type=int, default=None)

    @staticmethod
    def _make_processor_config(args: Namespace) -> SimpleNamespace:
        model_overrides = {}
        if args.prompt_path:
            model_overrides["prompt_path"] = Path(args.prompt_path)

        decode = {
            "beam_num": args.beam_size,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": bool(args.do_sample),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        }

        return SimpleNamespace(
            model_config=args.model_config,
            model_config_overrides=model_overrides,
            checkpoint=args.checkpoint,
            source_language=args.source_language,
            target_language=args.target_language,
            device=args.device,
            history_selection_policy=args.history_selection_policy,
            max_speech_context_seconds=args.max_speech_context_seconds,
            soft_speech_context_reset_seconds=args.soft_speech_context_reset_seconds,
            fixed_history_words=args.fixed_history_words,
            speech_context_policy=args.speech_context_policy,
            llm_context_turns=args.llm_context_turns,
            llm_raw_kv_cache=bool(args.llm_raw_kv_cache),
            decode=decode,
        )

    @classmethod
    def _load_processor_model(cls, args: Namespace, config: SimpleNamespace) -> None:
        signature = (
            Path(args.model_config).expanduser().as_posix(),
            Path(args.checkpoint).expanduser().as_posix() if args.checkpoint else None,
            Path(args.prompt_path).expanduser().as_posix() if args.prompt_path else None,
            args.device,
            args.beam_size,
            args.max_new_tokens,
            bool(args.do_sample),
            args.temperature,
            args.top_p,
            args.top_k,
            bool(args.llm_raw_kv_cache),
        )
        if cls._loaded_signature != signature:
            SimPOTransChunkProcessor._model = None
            SimPOTransChunkProcessor._device = None
            SimPOTransChunkProcessor._decode_config = None
            SimPOTransChunkProcessor.load_model(config)
            cls._loaded_signature = signature
        else:
            SimPOTransChunkProcessor.load_model(config)

    def to(self, device: str, *args, **kwargs) -> None:
        del args, kwargs
        target_device = torch.device(device)
        model = SimPOTransChunkProcessor._model
        if model is not None and SimPOTransChunkProcessor._device != target_device:
            model.to(target_device)
            SimPOTransChunkProcessor._device = target_device

    def reset(self) -> None:
        super().reset()
        self._processed_source_samples = 0
        self._finished = False
        if self.processor is not None:
            self.processor.clear()

    @torch.no_grad()
    def policy(self, states: AgentStates | None = None):
        states = states or self.states
        if self._finished:
            return WriteAction("", finished=True)

        output_text = self._process_new_source(states)
        if output_text:
            return WriteAction(output_text, finished=False)

        if states.source_finished:
            final_text = self.processor.end_of_stream().new_string
            self._finished = True
            self.processor.clear()
            return WriteAction(final_text, finished=True)

        return ReadAction()

    def _process_new_source(self, states: AgentStates) -> str:
        source_samples = states.source
        if len(source_samples) <= self._processed_source_samples:
            return ""

        chunk = source_samples[self._processed_source_samples :]
        self._processed_source_samples = len(source_samples)
        waveform = self._chunk_to_waveform(chunk, states.source_sample_rate or SAMPLE_RATE)
        if waveform.size == 0:
            return ""

        return self.processor.process_chunk(waveform).new_string

    @staticmethod
    def _chunk_to_waveform(chunk: list[float], sample_rate: int) -> np.ndarray:
        audio = torch.from_numpy(np.asarray(chunk, dtype=np.float32))
        if audio.ndim > 1:
            audio = audio.mean(dim=-1)

        if sample_rate != SAMPLE_RATE:
            audio = F.resample(audio, sample_rate, SAMPLE_RATE)

        return audio.cpu().numpy().astype(np.float32, copy=False)
