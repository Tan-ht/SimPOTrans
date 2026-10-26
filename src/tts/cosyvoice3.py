from __future__ import annotations

import io
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

from math import gcd
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from .backend import AudioChunk
from .japanese import JapaneseTextPreprocessor


LOGGER = logging.getLogger(__name__)
DEFAULT_INSTRUCTION_PREFIX = "You are a helpful assistant."
END_OF_PROMPT = "<|endofprompt|>"


class CosyVoice3Backend:
    def __init__(self, config: SimpleNamespace, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.model_dir = self._resolve_path(getattr(config, "model_dir", "checkpoints/cosyvoice3"))
        prompt_wav = str(getattr(config, "prompt_wav", "")).strip()
        self.prompt_wav = self._resolve_path(prompt_wav) if prompt_wav else None
        self.prompt_text = self._load_prompt_text(str(getattr(config, "prompt_text", "")).strip())
        self.stream = bool(getattr(config, "stream", False))
        self.instruction_prefix = self._normalize_instruction_prefix(
            str(getattr(config, "instruction_prefix", "")).strip()
        )
        self.japanese_preprocess = str(getattr(config, "japanese_preprocess", "mecab_unidic"))
        self._model = None
        self._japanese_preprocessor: JapaneseTextPreprocessor | None = None

        self._validate_config()

    def warmup(self) -> None:
        self._load_model()

    def synthesize(
        self,
        text: str,
        target_lang: str | None = None,
        prompt_wav: str | Path | None = None,
        prompt_text: str | None = None,
    ) -> AudioChunk:
        chunks = list(self.stream_synthesize(text, target_lang, prompt_wav, prompt_text))
        if not chunks:
            return AudioChunk(text=text.strip(), sample_rate=self.sample_rate, wav_bytes=b"")
        if len(chunks) == 1:
            return chunks[0]

        waveforms = [self._decode_wav_bytes(chunk.wav_bytes) for chunk in chunks]
        waveform = np.concatenate(waveforms, axis=-1)
        return AudioChunk(
            text=chunks[0].text,
            sample_rate=chunks[0].sample_rate,
            wav_bytes=self._encode_wav(waveform, chunks[0].sample_rate),
        )

    def stream_synthesize(
        self,
        text: str,
        target_lang: str | None = None,
        prompt_wav: str | Path | None = None,
        prompt_text: str | None = None,
    ) -> Iterator[AudioChunk]:
        source_text = text.strip()
        if not source_text:
            return

        tts_text = self._prepare_text(source_text, target_lang)
        active_prompt_wav = self._active_prompt_wav(prompt_wav)
        active_prompt_text = self._active_prompt_text(prompt_text)
        model = self._load_model()
        generated = False
        for chunk in self._inference(model, tts_text, active_prompt_wav, active_prompt_text, target_lang):
            waveform = self._to_numpy(chunk["tts_speech"])
            if waveform.size == 0:
                continue
            generated = True
            yield AudioChunk(
                text=source_text,
                sample_rate=self.sample_rate,
                wav_bytes=self._encode_wav(waveform, self.sample_rate),
            )

        if not generated:
            yield AudioChunk(text=source_text, sample_rate=self.sample_rate, wav_bytes=b"")

    def _active_prompt_wav(self, prompt_wav: str | Path | None) -> Path:
        if prompt_wav is not None:
            path = self._resolve_path(prompt_wav)
            if not path.exists():
                raise FileNotFoundError(f"runtime prompt_wav does not exist: {path}")
            return path
        if self.prompt_wav is None:
            raise ValueError(
                "TTS requires either tts.prompt_wav or a runtime prompt audio captured from input."
            )
        return self.prompt_wav

    def _active_prompt_text(self, prompt_text: str | None) -> str:
        if prompt_text is None:
            return self.prompt_text
        return self._load_prompt_text(str(prompt_text).strip())

    def _inference(self, model, text: str, prompt_wav: Path, prompt_text: str, target_lang: str | None):
        del target_lang
        if prompt_text:
            return model.inference_zero_shot(
                text,
                self._format_prompt_text(prompt_text),
                str(prompt_wav),
                stream=self.stream,
            )
        return model.inference_cross_lingual(
            self._format_cross_lingual_text(text),
            str(prompt_wav),
            stream=self.stream,
        )

    @property
    def sample_rate(self) -> int:
        model = self._load_model()
        return int(getattr(model, "sample_rate"))

    def _prepare_text(self, text: str, target_lang: str | None) -> str:
        prepared = text
        if target_lang == "ja" and self.japanese_preprocess == "mecab_unidic":
            if self._japanese_preprocessor is None:
                self._japanese_preprocessor = JapaneseTextPreprocessor()
            prepared = self._japanese_preprocessor.to_katakana(text)
            LOGGER.debug("Japanese TTS text normalized: %s -> %s", text, prepared)
        return prepared

    def _format_cross_lingual_text(self, text: str) -> str:
        if END_OF_PROMPT in text:
            return text
        return f"{self.instruction_prefix}{text}"

    def _format_prompt_text(self, prompt_text: str) -> str:
        if END_OF_PROMPT in prompt_text:
            return prompt_text
        return f"{self.instruction_prefix}{prompt_text}"

    @staticmethod
    def _normalize_instruction_prefix(value: str) -> str:
        if not value:
            value = DEFAULT_INSTRUCTION_PREFIX
        if END_OF_PROMPT in value:
            return value
        return f"{value}{END_OF_PROMPT}"

    def _load_model(self):
        if self._model is not None:
            return self._model

        cosyvoice_dir = self.root_dir / "third_party" / "CosyVoice"
        matcha_dir = cosyvoice_dir / "third_party" / "Matcha-TTS"
        for path in (matcha_dir, cosyvoice_dir):
            path_text = str(path)
            if path_text not in sys.path:
                sys.path.insert(0, path_text)

        try:
            from cosyvoice.cli.cosyvoice import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "CosyVoice is not importable. Initialize the submodule with "
                "`git submodule update --init --recursive third_party/CosyVoice`."
            ) from exc

        self._install_load_wav_fallback()
        LOGGER.info("Loading CosyVoice 3 TTS model from %s", self.model_dir)
        self._model = AutoModel(model_dir=str(self.model_dir))
        return self._model


    @staticmethod
    def _install_load_wav_fallback() -> None:
        import torch
        import soundfile as sf

        import cosyvoice.cli.frontend as frontend_module
        import cosyvoice.utils.file_utils as file_utils

        def load_wav_with_soundfile(wav, target_sr, min_sr=16000):
            speech, sample_rate = sf.read(wav, dtype="float32", always_2d=True)
            if sample_rate < min_sr:
                raise AssertionError(
                    f"wav sample rate {sample_rate} must be greater than {target_sr}"
                )
            speech = speech.mean(axis=1)
            if sample_rate != target_sr:
                factor = gcd(int(sample_rate), int(target_sr))
                speech = resample_poly(
                    speech,
                    int(target_sr) // factor,
                    int(sample_rate) // factor,
                )
            return torch.from_numpy(np.asarray(speech, dtype=np.float32)).unsqueeze(0)

        file_utils.load_wav = load_wav_with_soundfile
        frontend_module.load_wav = load_wav_with_soundfile

    def _validate_config(self) -> None:
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"CosyVoice 3 model directory does not exist: {self.model_dir}. "
                "Run `uv run python scripts/download_cosyvoice3.py` first."
            )
        if self.prompt_wav is not None and not self.prompt_wav.exists():
            raise FileNotFoundError(f"tts.prompt_wav does not exist: {self.prompt_wav}")

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (self.root_dir / path).resolve()
        return path

    def _load_prompt_text(self, value: str) -> str:
        if not value:
            return ""
        path = self._resolve_path(value)
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
        return value

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().float().numpy()
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 2:
            array = array.mean(axis=0)
        return np.clip(array.reshape(-1), -1.0, 1.0)

    @staticmethod
    def _decode_wav_bytes(wav_bytes: bytes) -> np.ndarray:
        sample_rate, waveform = wavfile.read(io.BytesIO(wav_bytes))
        del sample_rate
        array = np.asarray(waveform)
        if array.dtype == np.int16:
            array = array.astype(np.float32) / 32768.0
        elif np.issubdtype(array.dtype, np.integer):
            info = np.iinfo(array.dtype)
            array = array.astype(np.float32) / max(abs(info.min), info.max)
        else:
            array = array.astype(np.float32)
        if array.ndim == 2:
            array = array.mean(axis=1)
        return np.clip(array.reshape(-1), -1.0, 1.0)

    @staticmethod
    def _encode_wav(waveform: np.ndarray, sample_rate: int) -> bytes:
        buffer = io.BytesIO()
        waveform = np.clip(np.asarray(waveform, dtype=np.float32).reshape(-1), -1.0, 1.0)
        pcm16 = (waveform * 32767.0).astype(np.int16)
        wavfile.write(buffer, sample_rate, pcm16)
        return buffer.getvalue()
