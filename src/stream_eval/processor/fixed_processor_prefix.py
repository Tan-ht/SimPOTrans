from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import torch
from transformers import StoppingCriteriaList

from simulstream.server.speech_processors import SAMPLE_RATE
from simulstream.server.speech_processors.incremental_output import (
    IncrementalOutput,
)

from st_backbone.model import StopOnTokens, _decode_generation_kwargs
from stream_eval.processor.base_processor import (
    BaseSimPOTransChunkProcessor,
    _empty_output,
    _normalize_incremental_text,
)

logger = logging.getLogger(__name__)


class SimPOTransChunkProcessor(BaseSimPOTransChunkProcessor):
    """Decode from one full speech prompt plus assistant prefix.

    Unlike ``fixed_processor_dialogue``, this processor does not keep an LLM
    multi-turn KV cache. For every incoming chunk, it concatenates the current
    chunk to the retained speech context, places that whole speech context in a
    single user prompt, appends the assistant translation emitted so far as a
    text prefix, and asks the LLM to continue from that prefix.
    """

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__(config)
        self._history_text: list[str] = []

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
            self._speech_context_seconds + chunk_seconds
            > self.max_speech_context_seconds
        ):
            logger.debug(
                "Resetting SimPOTrans prefix prompt after %.3f seconds of "
                "speech context.",
                self._speech_context_seconds,
            )
            self._reset_prompt_context()

        self._speech_context_audio = (
            audio
            if self._speech_context_audio is None
            else torch.cat([self._speech_context_audio, audio], dim=0)
        )
        self._speech_context_seconds += chunk_seconds
        tokens = self._generate_continuation(self._speech_context_audio)
        text = _normalize_incremental_text(
            self._model.decode(tokens)[0],
            self.target_language,
        )
        if not text:
            return _empty_output()

        self._history.append(text)
        self._history_text.append(text)

        return IncrementalOutput(
            new_tokens=[text],
            new_string=text,
            deleted_tokens=[],
            deleted_string="",
        )

    def clear(self) -> None:
        super().clear()
        self._history_text.clear()

    def _reset_prompt_context(self) -> None:
        self._stream_state = None
        self._speech_context_seconds = 0.0
        self._speech_context_audio = None
        self._history.clear()
        self._history_text.clear()

    @torch.no_grad()
    def _generate_continuation(
        self,
        speech_audio: torch.Tensor,
    ) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("SimPOTrans model is not loaded.")

        speech_embeds = self._model.forward_adaptor(
            self._model.encode_speech(
                [speech_audio],
            )
        )
        prompts = self._model._render_prompts(
            [self.source_language],
            [self.target_language],
        )
        tokenized = self._model._tokenize_prompts(prompts)
        wrapped_embeds = self._model.wrap_prompt(speech_embeds, tokenized)

        prefix = self._assistant_prefix()
        if prefix:
            prefix_ids = self._model.llm_tokenizer(
                prefix,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.to(self._model.device)
            if prefix_ids.numel() > 0:
                prefix_embeds = self._model.llm.get_input_embeddings()(
                    prefix_ids,
                )
                wrapped_embeds = torch.cat(
                    [wrapped_embeds, prefix_embeds],
                    dim=1,
                )

        cfg = self._decode_config or self._model.decode_config
        stopping = StoppingCriteriaList(
            [StopOnTokens([self._model.eos_token_id])],
        )

        if cfg.do_sample:
            torch.manual_seed(cfg.seed)

        outputs = self._model.llm.generate(
            inputs_embeds=wrapped_embeds,
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

        return outputs.sequences

    def _assistant_prefix(self) -> str:
        if not self._history_text:
            return ""

        if self.target_language in {"ja", "ko", "zh", "zh-cn", "zh-tw"}:
            return "".join(self._history_text)

        return " ".join(token for token in self._history_text if token) + " "

