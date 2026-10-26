from __future__ import annotations

import io
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import wavfile

from .backend import AudioChunk
from .magictts_compat import patch_magictts_audio_io
from .magictts_compat import require_magictts_audio_tools


LOGGER = logging.getLogger(__name__)


class MagicTTSBackend:
    def __init__(self, config: SimpleNamespace, root_dir: Path) -> None:
        self.config = config
        self.root_dir = root_dir
        self.magictts_dir = self._resolve_path(getattr(config, "magictts_dir", "third_party/MAGIC-TTS"))
        self.checkpoint = self._resolve_path(
            getattr(config, "magictts_checkpoint", "third_party/MAGIC-TTS/checkpoints/magictts_36k.pt")
        )
        self.prompt_wav = self._optional_path(getattr(config, "prompt_wav", ""))
        self.prompt_text = self._load_prompt_text(str(getattr(config, "prompt_text", "")).strip())
        self.mode = str(getattr(config, "magictts_mode", "spontaneous")).strip() or "spontaneous"
        self.steps = int(getattr(config, "magictts_steps", 32))
        self.cfg_strength = float(getattr(config, "magictts_cfg_strength", 2.0))
        self.sway_sampling_coef = float(getattr(config, "magictts_sway_sampling_coef", -1.0))
        self.speed = float(getattr(config, "magictts_speed", 1.0))
        self._state = None
        self._prompt_cache: dict[tuple[str, str, str], tuple[object, str]] = {}
        self._validate_config()

    def warmup(self) -> None:
        self._load_state()
        self._warmup_text_tokenizer()
        self._warmup_default_prompts()

    def synthesize(
        self,
        text: str,
        target_lang: str | None = None,
        prompt_wav: str | Path | None = None,
        prompt_text: str | None = None,
    ) -> AudioChunk:
        source_text = text.strip()
        if not source_text:
            return AudioChunk(text=source_text, sample_rate=self.sample_rate, wav_bytes=b"")
        if self.mode != "spontaneous":
            raise ValueError("Magic-TTS demo backend currently supports spontaneous mode only.")

        language = self._language(target_lang)
        state = self._load_state()
        prompt_audio, prompt_text = self._prompt(language, prompt_wav, prompt_text)

        import torch
        from f5_tts.infer.utils_infer import infer_process

        with torch.inference_mode():
            wav, sample_rate, _spect = infer_process(
                prompt_audio,
                prompt_text,
                source_text,
                state["model"],
                state["vocoder"],
                mel_spec_type="vocos",
                nfe_step=self.steps,
                cfg_strength=self.cfg_strength,
                sway_sampling_coef=self.sway_sampling_coef,
                speed=self.speed,
                device=state["device"],
            )
        return AudioChunk(
            text=source_text,
            sample_rate=int(sample_rate),
            wav_bytes=self._encode_wav(np.asarray(wav, dtype=np.float32), int(sample_rate)),
        )

    def stream_synthesize(
        self,
        text: str,
        target_lang: str | None = None,
        prompt_wav: str | Path | None = None,
        prompt_text: str | None = None,
    ) -> Iterator[AudioChunk]:
        yield self.synthesize(text, target_lang, prompt_wav, prompt_text)

    @property
    def sample_rate(self) -> int:
        self._ensure_imports()
        from run_spontaneous_demo import TARGET_SAMPLE_RATE

        return int(TARGET_SAMPLE_RATE)

    def _load_state(self):
        if self._state is not None:
            return self._state
        self._ensure_imports()
        import torch
        from f5_tts.infer.utils_infer import load_vocoder
        from f5_tts.infer.utils_infer import preprocess_ref_audio_text
        from run_spontaneous_demo import DEFAULT_TOKENIZER_PATH
        from run_spontaneous_demo import DEFAULT_VOCODER_PATH
        from run_spontaneous_demo import build_model
        from run_spontaneous_demo import load_training_checkpoint

        device = "cuda" if torch.cuda.is_available() else "cpu"
        LOGGER.info("Loading Magic-TTS checkpoint from %s", self.checkpoint)
        model = build_model(DEFAULT_TOKENIZER_PATH)
        load_training_checkpoint(model, str(self.checkpoint))
        model.eval().to(device)
        vocoder = load_vocoder(
            vocoder_name="vocos",
            is_local=True,
            local_path=DEFAULT_VOCODER_PATH,
            device=device,
        )
        self._state = {
            "device": device,
            "model": model,
            "vocoder": vocoder,
            "preprocess_ref_audio_text": preprocess_ref_audio_text,
        }
        return self._state

    def _warmup_text_tokenizer(self) -> None:
        self._ensure_imports()
        try:
            import jieba
        except ModuleNotFoundError:
            return
        list(jieba.cut("初始化文本分词。"))

    def _warmup_default_prompts(self) -> None:
        for language in ("zh", "en"):
            try:
                self._prompt(language, None, None)
            except Exception:
                LOGGER.exception("Failed to warm up Magic-TTS default prompt for %s.", language)


    def _prompt(
        self,
        language: str,
        runtime_prompt_wav: str | Path | None,
        runtime_prompt_text: str | None,
    ):
        state = self._load_state()
        prompt_audio_path, prompt_text = self._prompt_assets(language, runtime_prompt_wav, runtime_prompt_text)
        cache_key = (language, str(prompt_audio_path.resolve()), prompt_text)
        if cache_key not in self._prompt_cache:
            self._prompt_cache[cache_key] = state["preprocess_ref_audio_text"](
                str(prompt_audio_path),
                prompt_text,
                device=state["device"],
            )
        return self._prompt_cache[cache_key]

    def _prompt_assets(
        self,
        language: str,
        runtime_prompt_wav: str | Path | None,
        runtime_prompt_text: str | None,
    ) -> tuple[Path, str]:
        if runtime_prompt_wav is not None:
            prompt_audio = self._resolve_path(runtime_prompt_wav)
            prompt_text = self._load_prompt_text(str(runtime_prompt_text or "").strip())
            if not prompt_text:
                prompt_text = self.prompt_text
            if not prompt_text:
                prompt_text = self._default_prompt_text(language)
            return prompt_audio, prompt_text
        if self.prompt_wav is not None:
            prompt_text = self.prompt_text
            if not prompt_text:
                raise ValueError("Magic-TTS requires tts.prompt_text when tts.prompt_wav is configured.")
            return self.prompt_wav, prompt_text
        return self._default_prompt_assets(language)

    def _default_prompt_assets(self, language: str) -> tuple[Path, str]:
        self._ensure_imports()
        from run_magictts import get_default_prompt_assets

        assets = get_default_prompt_assets(language)
        prompt_audio = Path(assets["audio"]).resolve()
        prompt_text = Path(assets["text"]).read_text(encoding="utf-8").strip()
        return prompt_audio, prompt_text

    def _default_prompt_text(self, language: str) -> str:
        return self._default_prompt_assets(language)[1]

    def _ensure_imports(self) -> None:
        paths = [
            self.root_dir / "src",
            self.root_dir,
            self.magictts_dir,
            self.magictts_dir / "inference",
            self.magictts_dir / "vendor" / "f5tts_duration_ft",
        ]
        for path in reversed(paths):
            path_text = str(path)
            if path_text not in sys.path:
                sys.path.insert(0, path_text)
        try:
            import rjieba  # type: ignore  # noqa: F401
        except ModuleNotFoundError:
            try:
                import jieba as _jieba
            except ModuleNotFoundError:
                return
            sys.modules["rjieba"] = _jieba
        patch_magictts_audio_io()

    def _validate_config(self) -> None:
        require_magictts_audio_tools()
        if not self.magictts_dir.exists():
            raise FileNotFoundError(f"Magic-TTS directory does not exist: {self.magictts_dir}")
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Magic-TTS checkpoint does not exist: {self.checkpoint}")
        if self.prompt_wav is not None and not self.prompt_wav.exists():
            raise FileNotFoundError(f"tts.prompt_wav does not exist: {self.prompt_wav}")

    def _optional_path(self, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return self._resolve_path(text)

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
    def _language(target_lang: str | None) -> str:
        return "en" if (target_lang or "").lower().startswith("en") else "zh"

    @staticmethod
    def _encode_wav(waveform: np.ndarray, sample_rate: int) -> bytes:
        buffer = io.BytesIO()
        waveform = np.clip(np.asarray(waveform, dtype=np.float32).reshape(-1), -1.0, 1.0)
        pcm16 = (waveform * 32767.0).astype(np.int16)
        wavfile.write(buffer, sample_rate, pcm16)
        return buffer.getvalue()
