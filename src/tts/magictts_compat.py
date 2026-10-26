from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def load_audio_with_soundfile(path: str | Path):
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = np.ascontiguousarray(audio.T)
    return torch.from_numpy(audio), int(sample_rate)


def save_audio_with_soundfile(path: str | Path, audio: torch.Tensor, sample_rate: int, *args, **kwargs) -> None:
    del args, kwargs
    wav = audio.detach().cpu().float().numpy()
    if wav.ndim == 2:
        wav = wav[0] if wav.shape[0] == 1 else wav.T
    wav = np.clip(wav, -1.0, 1.0)
    sf.write(str(path), wav, int(sample_rate), subtype="PCM_16")


def load_audio_24k_mono(path: str | Path, *, sample_rate: int = 24000, hop_length: int | None = None):
    audio, source_sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1, keepdims=True)
    if int(source_sample_rate) != int(sample_rate):
        import librosa

        audio = librosa.resample(
            audio[:, 0],
            orig_sr=int(source_sample_rate),
            target_sr=int(sample_rate),
        )[:, None]
    tensor = torch.from_numpy(np.ascontiguousarray(audio.T))
    if hop_length is None:
        return tensor, int(sample_rate)
    frames = float(tensor.shape[-1]) / float(hop_length)
    return tensor, frames


def _find_executable(name: str) -> str | None:
    path = shutil.which(name)
    if path:
        return path
    for directory in ("/usr/bin", "/usr/local/bin", "/bin", "/opt/conda/bin"):
        candidate = Path(directory) / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _ensure_audio_tool_path() -> tuple[str | None, str | None]:
    ffmpeg = _find_executable("ffmpeg")
    ffprobe = _find_executable("ffprobe")
    paths = [str(Path(path).parent) for path in (ffmpeg, ffprobe) if path]
    if paths:
        current = os.environ.get("PATH", "")
        parts = [part for part in current.split(os.pathsep) if part]
        for path in reversed(dict.fromkeys(paths)):
            if path not in parts:
                parts.insert(0, path)
        os.environ["PATH"] = os.pathsep.join(parts)
    return ffmpeg, ffprobe


def require_magictts_audio_tools() -> None:
    ffmpeg, ffprobe = _ensure_audio_tool_path()
    missing = [
        name
        for name, path in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe))
        if path is None
    ]
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(
            f"Magic-TTS requires {missing_text} in PATH. Install ffmpeg, which provides both ffmpeg "
            "and ffprobe, or activate an environment that exposes them before enabling TTS."
        )


def _patch_pydub_tools(ffmpeg: str | None, ffprobe: str | None) -> None:
    try:
        from pydub import AudioSegment
        import pydub.utils as pydub_utils
    except ImportError:
        return

    if ffmpeg:
        AudioSegment.converter = ffmpeg
        AudioSegment.ffmpeg = ffmpeg

    if ffprobe:
        def get_prober_name() -> str:
            return ffprobe

        pydub_utils.get_prober_name = get_prober_name


def patch_magictts_audio_io() -> None:
    ffmpeg, ffprobe = _ensure_audio_tool_path()
    _patch_pydub_tools(ffmpeg, ffprobe)

    import torchaudio

    torchaudio.load = load_audio_with_soundfile
    torchaudio.save = save_audio_with_soundfile

    try:
        import f5_tts.infer.utils_infer as utils_infer
    except ImportError:
        return
    utils_infer.torchaudio.load = load_audio_with_soundfile
    utils_infer.torchaudio.save = save_audio_with_soundfile
