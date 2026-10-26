from __future__ import annotations

import argparse
import asyncio
import functools
import io
import json
import logging
import multiprocessing as mp
import os
import queue as queue_module
import re
import tempfile
import threading
import time
import traceback
import unicodedata
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import librosa
import numpy as np
from scipy.io import wavfile
import websockets
import yaml
from websockets.asyncio.server import ServerConnection, serve

from simulstream.metrics.logger import METRICS_LOGGER, setup_metrics_logger
from simulstream.server.speech_processors import SAMPLE_RATE, build_speech_processor
from simulstream.server.speech_processors.incremental_output import (
    IncrementalOutput,
    merge_incremental_outputs,
)
from tts import AudioChunk, build_tts_backend


LOGGER = logging.getLogger("simpotrans.demo")
ROOT_DIR = Path(__file__).resolve().parents[1]
DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_WEB_DIR = DEMO_DIR / "web"
_CHINESE_DIGITS = "零一二三四五六七八九"
_CHINESE_SMALL_UNITS = ("", "十", "百", "千")
_CHINESE_SECTION_UNITS = ("", "万", "亿", "兆")
_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:,\d{3})*(?:\.\d+)?|(?<![A-Za-z])[-+]?\d+(?:\.\d+)?")


def _is_chinese_target_lang(target_lang: str | None) -> bool:
    lang = (target_lang or "").lower()
    return lang in {"zh", "zh-cn", "zh-tw"} or lang.startswith("zh-")


def _normalize_magictts_chinese_text(text: str, target_lang: str | None, backend_name: str) -> str:
    if backend_name != "magictts" or not _is_chinese_target_lang(target_lang):
        return text
    return _NUMBER_RE.sub(lambda match: _number_text_to_chinese(match.group(0)), text)


def _number_text_to_chinese(text: str) -> str:
    sign = ""
    value = text.replace(",", "")
    if value.startswith(("-", "+")):
        sign = "负" if value[0] == "-" else ""
        value = value[1:]
    if not value:
        return text

    integer, dot, fraction = value.partition(".")
    if not integer:
        integer = "0"
    integer_text = _integer_text_to_chinese(integer)
    if dot:
        fraction_text = "".join(_CHINESE_DIGITS[int(char)] for char in fraction if char.isdigit())
        if not fraction_text:
            return sign + integer_text
        return sign + integer_text + "点" + fraction_text
    return sign + integer_text


def _integer_text_to_chinese(text: str) -> str:
    if not text or not text.isdigit():
        return text
    if len(text) > 1 and text.startswith("0"):
        return "".join(_CHINESE_DIGITS[int(char)] for char in text)

    number = int(text)
    if number == 0:
        return _CHINESE_DIGITS[0]

    sections: list[int] = []
    while number > 0:
        sections.append(number % 10000)
        number //= 10000

    parts: list[str] = []
    need_zero = False
    for index in range(len(sections) - 1, -1, -1):
        section = sections[index]
        if section == 0:
            need_zero = bool(parts)
            continue
        if need_zero or (parts and section < 1000):
            if parts[-1] != _CHINESE_DIGITS[0]:
                parts.append(_CHINESE_DIGITS[0])
        parts.append(_section_to_chinese(section) + _CHINESE_SECTION_UNITS[index])
        need_zero = False

    result = "".join(parts).replace("一十", "十", 1)
    return result


def _section_to_chinese(number: int) -> str:
    parts: list[str] = []
    zero_pending = False
    for position in range(3, -1, -1):
        unit_value = 10 ** position
        digit = number // unit_value
        number %= unit_value
        if digit == 0:
            zero_pending = bool(parts)
            continue
        if zero_pending and parts[-1] != _CHINESE_DIGITS[0]:
            parts.append(_CHINESE_DIGITS[0])
        parts.append(_CHINESE_DIGITS[digit] + _CHINESE_SMALL_UNITS[position])
        zero_pending = False
    return "".join(parts)


class TTSTextTracker:
    def __init__(self) -> None:
        self.full_text = ""
        self.synthesized_length = 0

    def apply_delta(self, new_text: str, deleted_text: str) -> str:
        if deleted_text:
            self.full_text = self.full_text[: max(0, len(self.full_text) - len(deleted_text))]
            self.synthesized_length = min(self.synthesized_length, len(self.full_text))
        if new_text:
            self.full_text += new_text

        unsynthesized = self.full_text[self.synthesized_length :]
        self.synthesized_length = len(self.full_text)
        return unsynthesized

    def clear(self) -> None:
        self.full_text = ""
        self.synthesized_length = 0


class InputPromptAudioRecorder:
    def __init__(self, duration_seconds: float, root_dir: Path) -> None:
        self.duration_seconds = duration_seconds
        self.root_dir = root_dir
        self.sample_rate: int | None = None
        self.buffer = bytearray()
        self.prompt_wav: Path | None = None
        self.collecting = False

    @property
    def ready(self) -> bool:
        return self.prompt_wav is not None

    def start_segment(self) -> None:
        if self.ready:
            return
        if not self.buffer:
            self.sample_rate = None
        self.collecting = True

    def append(self, speech_data: bytes, sample_rate: int) -> Path | None:
        if self.ready or not self.collecting or not speech_data:
            return self.prompt_wav
        self.sample_rate = sample_rate
        self.buffer.extend(speech_data)
        if self.duration() >= self.duration_seconds:
            self.prompt_wav = self._write_prompt_wav()
            self.collecting = False
        return self.prompt_wav

    def duration(self) -> float:
        if not self.sample_rate:
            return 0.0
        return len(self.buffer) / 2 / self.sample_rate

    def clear(self) -> None:
        if self.prompt_wav is not None:
            try:
                self.prompt_wav.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("Failed to remove temporary TTS prompt wav: %s", self.prompt_wav)
        self.prompt_wav = None
        self.sample_rate = None
        self.buffer = bytearray()
        self.collecting = False

    def _write_prompt_wav(self) -> Path:
        if self.sample_rate is None:
            raise RuntimeError("Cannot write prompt wav without a sample rate.")
        samples = int(round(self.duration_seconds * self.sample_rate))
        waveform = np.frombuffer(bytes(self.buffer), dtype=np.int16)[:samples]
        handle = tempfile.NamedTemporaryFile(
            suffix=".wav",
            prefix="simpotrans-tts-prompt-",
            dir="/tmp",
            delete=False,
        )
        handle.close()
        path = Path(handle.name)
        wavfile.write(path, self.sample_rate, waveform)
        LOGGER.info(
            "Captured %.2fs input audio for TTS prompt: %s",
            self.duration_seconds,
            path,
        )
        return path


def _plain_config(value):
    if isinstance(value, SimpleNamespace):
        return {key: _plain_config(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {key: _plain_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_config(item) for item in value]
    return value


def _apply_visible_devices(value: object, *, label: str) -> None:
    if value is None:
        return
    text = str(value).strip()
    if not text or text.lower() in {"inherit", "none", "null"}:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = text
    LOGGER.info("Using CUDA_VISIBLE_DEVICES=%s for %s.", text, label)


def _encode_tts_chunk(chunk: AudioChunk, *, stream_id: int | None = None, chunk_index: int | None = None) -> dict:
    payload = {
        "type": "audio",
        "wav_bytes": chunk.wav_bytes,
        "sample_rate": chunk.sample_rate,
        "text": chunk.text,
    }
    if stream_id is not None:
        payload["tts_stream_id"] = stream_id
    if chunk_index is not None:
        payload["tts_chunk_index"] = chunk_index
    return payload


TTS_AUDIO_FRAME_VERSION = 1


def _audio_samples_to_pcm16(samples: np.ndarray) -> np.ndarray:
    if samples.dtype == np.int16:
        return np.ascontiguousarray(samples)
    if np.issubdtype(samples.dtype, np.floating):
        clipped = np.clip(samples.astype(np.float32, copy=False), -1.0, 1.0)
        return np.ascontiguousarray((clipped * 32767.0).astype(np.int16))
    if samples.dtype == np.uint8:
        normalized = (samples.astype(np.float32) - 128.0) / 128.0
        return np.ascontiguousarray((np.clip(normalized, -1.0, 1.0) * 32767.0).astype(np.int16))
    if np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)
        scale = float(max(abs(info.min), info.max))
        normalized = samples.astype(np.float32) / scale
        return np.ascontiguousarray((np.clip(normalized, -1.0, 1.0) * 32767.0).astype(np.int16))
    raise ValueError(f"Unsupported TTS WAV sample dtype: {samples.dtype}")


def decode_tts_wav_to_mono_pcm16(wav_bytes: bytes) -> tuple[int, np.ndarray]:
    sample_rate, samples = wavfile.read(io.BytesIO(wav_bytes))
    if samples.ndim > 1:
        samples = samples.astype(np.float32).mean(axis=1)
    pcm16 = _audio_samples_to_pcm16(samples)
    return int(sample_rate), np.ascontiguousarray(pcm16.reshape(-1))


def pack_tts_audio_frame(header: dict, payload: bytes) -> bytes:
    header_bytes = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return len(header_bytes).to_bytes(4, "little") + header_bytes + payload


def encode_tts_audio_frames(
    wav_bytes: bytes,
    *,
    text: str,
    sequence_id: int,
    frame_ms: int,
) -> list[bytes]:
    sample_rate, pcm16 = decode_tts_wav_to_mono_pcm16(wav_bytes)
    if pcm16.size == 0:
        return []
    normalized_frame_ms = max(1, int(frame_ms))
    frame_samples = max(1, int(round(sample_rate * normalized_frame_ms / 1000.0)))
    frames: list[bytes] = []
    total_frames = int(np.ceil(pcm16.size / frame_samples))
    for frame_index in range(total_frames):
        start = frame_index * frame_samples
        end = min(start + frame_samples, pcm16.size)
        frame_pcm = np.ascontiguousarray(pcm16[start:end].astype("<i2", copy=False))
        header = {
            "type": "tts_audio_frame",
            "version": TTS_AUDIO_FRAME_VERSION,
            "sample_rate": sample_rate,
            "channels": 1,
            "encoding": "pcm_s16le",
            "sequence_id": sequence_id,
            "frame_index": frame_index,
            "is_final": frame_index == total_frames - 1,
            "text": text,
            "frame_ms": normalized_frame_ms,
        }
        frames.append(pack_tts_audio_frame(header, frame_pcm.tobytes()))
    return frames


def _tts_consumer_process(config_payload: dict, root_dir_text: str, text_queue, audio_queue, control_queue) -> None:
    root_dir = Path(root_dir_text)
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    try:
        visible_devices = config_payload.get("visible_devices")
        if visible_devices is not None:
            text = str(visible_devices).strip()
            if text and text.lower() not in {"inherit", "none", "null"}:
                os.environ["CUDA_VISIBLE_DEVICES"] = text
                LOGGER.info("Using CUDA_VISIBLE_DEVICES=%s for TTS worker.", text)

        config_payload = dict(config_payload)
        target_lang = config_payload.pop("target_lang", None)
        prompt_wav = config_payload.pop("active_prompt_wav", None)
        prompt_text = config_payload.pop("active_prompt_text", None)
        stream = bool(config_payload.get("stream", False))
        config = _to_namespace(config_payload)
        backend = build_tts_backend(config, root_dir)
        backend.warmup()
        audio_queue.put({"type": "ready"})
        stream_sequence = 0

        def apply_control_messages() -> None:
            nonlocal target_lang, prompt_wav, prompt_text
            while True:
                try:
                    message = control_queue.get_nowait()
                except queue_module.Empty:
                    return
                if not isinstance(message, dict):
                    continue
                if message.get("type") == "context":
                    target_lang = message.get("target_lang")
                    prompt_wav = message.get("active_prompt_wav")
                    prompt_text = message.get("active_prompt_text")

        while True:
            apply_control_messages()
            first_text = text_queue.get()
            if first_text is None:
                break

            texts = [str(first_text)]
            should_shutdown = False
            while True:
                try:
                    item = text_queue.get_nowait()
                except queue_module.Empty:
                    break
                if item is None:
                    should_shutdown = True
                    break
                texts.append(str(item))

            apply_control_messages()
            text = "".join(texts).strip()
            if text:
                if stream:
                    stream_sequence += 1
                    chunk_index = 0
                    for chunk in backend.stream_synthesize(text, target_lang, prompt_wav, prompt_text):
                        if chunk.wav_bytes:
                            audio_queue.put(_encode_tts_chunk(chunk, stream_id=stream_sequence, chunk_index=chunk_index))
                            chunk_index += 1
                    audio_queue.put({
                        "type": "stream_end",
                        "tts_stream_id": stream_sequence,
                        "text": text,
                    })
                else:
                    chunk = backend.synthesize(text, target_lang, prompt_wav, prompt_text)
                    if chunk.wav_bytes:
                        audio_queue.put(_encode_tts_chunk(chunk))

            if should_shutdown:
                break
    except BaseException as exc:  # pragma: no cover - child process diagnostics
        audio_queue.put({
            "type": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
    finally:
        audio_queue.put(None)


class TTSRuntime:
    def __init__(self) -> None:
        self._mp_context = mp.get_context("spawn")
        self.text_queue = None
        self.audio_queue = None
        self.control_queue = None
        self.process: mp.Process | None = None
        self.payload_key: str | None = None
        self.active_client_id: int | None = None

    def start(self, config_payload: dict, *, force: bool = False) -> None:
        payload_key = self._runtime_payload_key(config_payload)
        if not force and self.process is not None and self.process.is_alive() and self.payload_key == payload_key:
            return
        self.stop()
        self.text_queue = self._mp_context.Queue()
        self.audio_queue = self._mp_context.Queue()
        self.control_queue = self._mp_context.Queue()
        self.process = self._mp_context.Process(
            target=_tts_consumer_process,
            args=(config_payload, str(ROOT_DIR), self.text_queue, self.audio_queue, self.control_queue),
            name="simpotrans-tts-runtime",
            daemon=True,
        )
        self.process.start()
        self.payload_key = payload_key
        try:
            item = self.audio_queue.get(timeout=300)
        except queue_module.Empty as exc:
            self.stop()
            raise RuntimeError("Timed out waiting for TTS worker warmup.") from exc
        if isinstance(item, dict) and item.get("type") == "ready":
            return
        if isinstance(item, dict) and item.get("type") == "error":
            error = str(item.get("error", "unknown TTS worker error"))
            self.stop()
            raise RuntimeError(error)
        self.stop()
        raise RuntimeError(f"Unexpected TTS worker warmup response: {item!r}")

    def ensure_started(self, config_payload: dict) -> None:
        self.start(config_payload, force=False)

    def stop(self) -> None:
        self.request_shutdown()
        process = self.process
        self.process = None
        if process is not None:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            process.close()
        self._close_queue(self.text_queue)
        self._close_queue(self.audio_queue)
        self._close_queue(self.control_queue)
        self.text_queue = None
        self.audio_queue = None
        self.control_queue = None
        self.payload_key = None
        self.active_client_id = None

    def request_shutdown(self) -> None:
        if self.text_queue is not None:
            try:
                self.text_queue.put(None)
            except (BrokenPipeError, EOFError, OSError):
                pass

    def claim(self, client_id: int) -> bool:
        if self.active_client_id is None or self.active_client_id == client_id:
            self.active_client_id = client_id
            return True
        return False

    def release(self, client_id: int) -> None:
        if self.active_client_id == client_id:
            self.active_client_id = None

    def clear_text_queue(self) -> None:
        if self.text_queue is None:
            return
        while True:
            try:
                self.text_queue.get_nowait()
            except queue_module.Empty:
                return

    def update_context(self, *, target_lang: str | None, prompt_wav: Path | None, prompt_text: str | None) -> None:
        if self.control_queue is None:
            return
        self.control_queue.put({
            "type": "context",
            "target_lang": target_lang,
            "active_prompt_wav": str(prompt_wav) if prompt_wav is not None else None,
            "active_prompt_text": prompt_text,
        })

    @staticmethod
    def _runtime_payload_key(config_payload: dict) -> str:
        dynamic_keys = {"target_lang", "active_prompt_wav", "active_prompt_text", "speaker"}
        static_payload = {
            key: value
            for key, value in config_payload.items()
            if key not in dynamic_keys
        }
        return json.dumps(static_payload, sort_keys=True, default=str, ensure_ascii=False)

    @staticmethod
    def _close_queue(value) -> None:
        if value is None:
            return
        try:
            value.close()
            value.join_thread()
        except (OSError, ValueError):
            pass



class TTSAudioWebSocketServer:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._connections: dict[str, ServerConnection] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        self._startup_error: BaseException | None = None
        self._stop_future: asyncio.Future | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._started.clear()
        self._startup_error = None
        self._thread = threading.Thread(
            target=self._run_thread,
            name="simpotrans-tts-audio-ws",
            daemon=True,
        )
        self._thread.start()
        if not self._started.wait(timeout=10.0):
            raise RuntimeError("Timed out starting TTS audio websocket server.")
        if self._startup_error is not None:
            raise RuntimeError("Failed to start TTS audio websocket server.") from self._startup_error

    def stop(self) -> None:
        loop = self._loop
        stop_future = self._stop_future
        if loop is not None and stop_future is not None and not stop_future.done():
            loop.call_soon_threadsafe(stop_future.set_result, None)
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
        self._thread = None
        self._loop = None
        self._stop_future = None
        self._connections = {}

    async def send_audio_frame(self, session_id: str | None, frame_bytes: bytes) -> bool:
        loop = self._loop
        if loop is None or not session_id or not frame_bytes:
            return False
        future = asyncio.run_coroutine_threadsafe(
            self._send_audio_frame(session_id, frame_bytes),
            loop,
        )
        return await asyncio.wrap_future(future)

    def _run_thread(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._stop_future = loop.create_future()
        try:
            loop.run_until_complete(self._serve_until_stopped())
        except BaseException as exc:  # pragma: no cover - thread startup/runtime diagnostics
            self._startup_error = exc
            self._started.set()
            LOGGER.exception("TTS audio websocket server failed.")
        finally:
            for websocket in list(self._connections.values()):
                try:
                    loop.run_until_complete(websocket.close(code=1001, reason="Server shutdown"))
                except Exception:
                    pass
            self._connections.clear()
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

    async def _serve_until_stopped(self) -> None:
        assert self._stop_future is not None
        async with serve(self._handler, self.host, self.port, ping_timeout=None, max_size=None):
            LOGGER.info("Serving TTS audio websocket at ws://%s:%s", self.host, self.port)
            self._started.set()
            await self._stop_future

    async def _handler(self, websocket: ServerConnection) -> None:
        session_id: str | None = None
        try:
            raw = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            if isinstance(raw, bytes):
                await websocket.close(code=1003, reason="Expected JSON registration message")
                return
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.close(code=1003, reason="Invalid JSON registration message")
                return
            session_id = str(data.get("tts_session_id") or "").strip()
            if not session_id:
                await websocket.close(code=1008, reason="Missing tts_session_id")
                return
            await self._register(session_id, websocket)
            LOGGER.info("TTS audio websocket connected for session %s.", session_id)
            async for _message in websocket:
                pass
        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
            pass
        finally:
            if session_id is not None:
                await self._unregister(session_id, websocket)
                LOGGER.info("TTS audio websocket disconnected for session %s.", session_id)

    async def _register(self, session_id: str, websocket: ServerConnection) -> None:
        old = self._connections.get(session_id)
        self._connections[session_id] = websocket
        if old is not None and old is not websocket:
            try:
                await old.close(code=1000, reason="Replaced by a new TTS audio connection")
            except websockets.exceptions.ConnectionClosed:
                pass

    async def _unregister(self, session_id: str, websocket: ServerConnection) -> None:
        if self._connections.get(session_id) is websocket:
            self._connections.pop(session_id, None)

    async def _send_audio_frame(self, session_id: str, frame_bytes: bytes) -> bool:
        websocket = self._connections.get(session_id)
        if websocket is None:
            return False
        try:
            await websocket.send(frame_bytes)
            return True
        except websockets.exceptions.ConnectionClosed:
            await self._unregister(session_id, websocket)
            return False


def _resolve_tts_prompt_wav(prompt_wav: str | Path | None) -> Path | None:
    text = str(prompt_wav or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    return path


def _normalize_speaker_prompts(raw_prompts: object) -> dict[str, SimpleNamespace]:
    prompts: dict[str, SimpleNamespace] = {}
    for language, value in _namespace_to_dict(raw_prompts).items():
        language_id = str(language or "").strip().lower()
        if not language_id:
            continue
        prompt = _namespace_to_dict(value)
        prompts[language_id] = SimpleNamespace(
            prompt_wav=str(prompt.get("prompt_wav", "") or ""),
            prompt_text=str(prompt.get("prompt_text", "") or ""),
        )
    return prompts


def _speaker_prompt_for_language(speaker: object | None, language: str | None) -> dict[str, str]:
    if speaker is None:
        return {"prompt_wav": "", "prompt_text": ""}

    prompt_wav = str(getattr(speaker, "prompt_wav", "") or "")
    prompt_text = str(getattr(speaker, "prompt_text", "") or "")
    prompts = _namespace_to_dict(getattr(speaker, "prompts", {}))
    language_id = str(language or "").strip().lower()
    language_base = language_id.split("-", 1)[0]
    candidates = [language_id]
    if language_base and language_base != language_id:
        candidates.append(language_base)
    candidates.append("default")

    for candidate in candidates:
        if not candidate or candidate not in prompts:
            continue
        values = _namespace_to_dict(prompts[candidate])
        prompt_wav = str(values.get("prompt_wav", "") or "")
        prompt_text = str(values.get("prompt_text", "") or "")
        break

    return {"prompt_wav": prompt_wav, "prompt_text": prompt_text}


def _startup_tts_config_payload(tts_config: SimpleNamespace, target_lang: str | None) -> dict:
    payload = _plain_config(tts_config)
    speakers = getattr(tts_config, "speakers", []) or []
    default_speaker = str(getattr(tts_config, "default_speaker", "default") or "default")
    selected = None
    for entry in speakers:
        values = _namespace_to_dict(entry)
        if str(values.get("id", "")).strip() == default_speaker:
            selected = entry
            break
    if selected is None and speakers:
        selected = speakers[0]

    prompt_values = _speaker_prompt_for_language(selected, target_lang)
    prompt_wav = _resolve_tts_prompt_wav(prompt_values.get("prompt_wav", ""))
    prompt_text = str(prompt_values.get("prompt_text", "") or "").strip()
    payload["target_lang"] = target_lang
    payload["active_prompt_wav"] = str(prompt_wav) if prompt_wav is not None else None
    payload["active_prompt_text"] = prompt_text or None
    payload["speaker"] = default_speaker
    return payload


class ClientTTSCoordinator:
    def __init__(
        self,
        client_id: int,
        websocket: ServerConnection,
        tts_config: SimpleNamespace,
        tts_runtime: TTSRuntime | None,
        audio_registry: TTSAudioWebSocketServer | None = None,
    ) -> None:
        self.client_id = client_id
        self.websocket = websocket
        self.tts_config = tts_config
        self.backend_name = str(getattr(tts_config, "backend", "cosyvoice3")).lower()
        self.prompt_duration_seconds = float(getattr(tts_config, "prompt_duration_seconds", 4.0))
        self.prompt_recorder = InputPromptAudioRecorder(self.prompt_duration_seconds, ROOT_DIR)
        self.prompt_ready = asyncio.Event()
        self.speakers = self._configured_speakers()
        self.default_speaker_id = str(getattr(tts_config, "default_speaker", "default")).strip() or "default"
        if self.default_speaker_id not in self.speakers:
            self.default_speaker_id = next(iter(self.speakers))
        self.enabled = False
        self.target_lang: str | None = None
        self.text_tracker = TTSTextTracker()
        self.pending_tts_text = ""
        self.flush_min_units = int(getattr(tts_config, "flush_min_units", 4))
        self.audio_frame_ms = int(getattr(tts_config, "audio_frame_ms", 200))
        self.speaker_id = self.default_speaker_id
        self.config_prompt_wav: Path | None = None
        self.config_prompt_text: str = ""
        self.tts_runtime = tts_runtime
        self.audio_registry = audio_registry
        self.tts_session_id: str | None = None
        self.audio_sender_task: asyncio.Task | None = None
        self._tts_audio_sequence_id = 0
        self._select_speaker(self.default_speaker_id)

    async def configure(self, metadata: dict) -> None:
        if "target_lang" in metadata:
            self.target_lang = metadata["target_lang"]
            self._refresh_selected_speaker_prompt()
        if "tts_session_id" in metadata:
            session_id = str(metadata.get("tts_session_id") or "").strip()
            self.tts_session_id = session_id or None
        if "tts_speaker" in metadata:
            await self._select_speaker_for_runtime(metadata.get("tts_speaker"))

        if "enable_tts" not in metadata:
            return

        requested = bool(metadata["enable_tts"])
        if requested and not bool(getattr(self.tts_config, "enabled", False)):
            self.enabled = False
            await self.websocket.send(json.dumps({"tts_unavailable": True}))
            return

        self.enabled = requested
        if self.enabled:
            if self.tts_runtime is None or not self.tts_runtime.claim(self.client_id):
                self.enabled = False
                await self.websocket.send(json.dumps({"tts_unavailable": True}))
                return
            await self._ensure_tts_worker_if_ready()
        else:
            await self._detach_tts_runtime()

    async def handle_output(self, output: IncrementalOutput) -> None:
        if not self.enabled:
            return

        text = self.text_tracker.apply_delta(
            output.new_string or "",
            output.deleted_string or "",
        )
        text = self._normalize_tts_text(text)
        if not text.strip():
            return

        self.pending_tts_text += text
        if not self.prompt_ready.is_set():
            return

        await self._ensure_tts_worker_if_ready()
        await self.flush_pending_if_ready()

    def accept_audio(
        self,
        speech_data: bytes,
        sample_rate: int,
        vad_speech_start: bool = False,
    ) -> bool:
        if not self.enabled or self.config_prompt_wav is not None or self.prompt_ready.is_set():
            return False

        if vad_speech_start:
            self.prompt_recorder.start_segment()

        prompt_wav = self.prompt_recorder.append(speech_data, sample_rate)
        if prompt_wav is not None:
            self.prompt_ready.set()
            return True
        return False

    async def flush_pending_if_ready(self, *, force: bool = False) -> None:
        if not self.enabled or not self.prompt_ready.is_set() or not self._has_active_prompt():
            return
        await self._ensure_tts_worker_if_ready()
        if self.pending_tts_text.strip() and (force or self._should_flush_tts_text(self.pending_tts_text)):
            self._enqueue_tts_text(self.pending_tts_text)
            self.pending_tts_text = ""

    async def clear(self) -> None:
        self.text_tracker.clear()
        self.pending_tts_text = ""
        self._clear_tts_text_queue()
        if self.config_prompt_wav is None and self.backend_name != "magictts":
            self.prompt_ready.clear()
            self.prompt_recorder.clear()
        elif self.config_prompt_wav is not None or self.backend_name == "magictts":
            self.prompt_ready.set()

    async def drain(self) -> None:
        if not self.enabled:
            return
        if not self.prompt_ready.is_set() and self._has_active_prompt():
            self.prompt_ready.set()
        await self.flush_pending_if_ready(force=True)

    async def close(self) -> None:
        self.prompt_recorder.clear()
        await self._detach_tts_runtime()

    def _enqueue_tts_text(self, text: str) -> None:
        if self.tts_runtime is None or self.tts_runtime.text_queue is None:
            self.pending_tts_text += text
            return
        self.tts_runtime.text_queue.put(text)

    async def _ensure_tts_worker_if_ready(self) -> None:
        if not self.enabled or not self.prompt_ready.is_set() or not self._has_active_prompt():
            return
        if self.tts_runtime is None:
            return
        self.tts_runtime.ensure_started(self._tts_config_payload())
        self._update_tts_runtime_context()
        if self.audio_sender_task is None or self.audio_sender_task.done():
            self.audio_sender_task = asyncio.create_task(self._audio_sender_loop())

    async def _detach_tts_runtime(self) -> None:
        task = self.audio_sender_task
        self.audio_sender_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self.tts_runtime is not None:
            self.tts_runtime.release(self.client_id)

    def _clear_tts_text_queue(self) -> None:
        if self.tts_runtime is not None:
            self.tts_runtime.clear_text_queue()

    async def _audio_sender_loop(self) -> None:
        if self.tts_runtime is None or self.tts_runtime.audio_queue is None:
            return
        audio_queue = self.tts_runtime.audio_queue
        try:
            while True:
                try:
                    item = await asyncio.to_thread(audio_queue.get, True, 0.1)
                except queue_module.Empty:
                    continue
                if item is None:
                    return
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "error":
                    LOGGER.warning("TTS worker error: %s", item.get("error", "unknown TTS worker error"))
                    continue
                if item.get("type") == "stream_end":
                    continue
                if item.get("type") == "audio" and item.get("wav_bytes"):
                    wav_bytes = bytes(item.get("wav_bytes") or b"")
                    text = str(item.get("text") or "")
                    self._tts_audio_sequence_id += 1
                    sequence_id = self._tts_audio_sequence_id
                    try:
                        frames = encode_tts_audio_frames(
                            wav_bytes,
                            text=text,
                            sequence_id=sequence_id,
                            frame_ms=self.audio_frame_ms,
                        )
                    except Exception as exc:
                        LOGGER.warning("Failed to encode TTS audio frames: %s", exc)
                        METRICS_LOGGER.info(json.dumps({
                            "tts_audio_dropped": {
                                "id": self.client_id,
                                "tts_session_id": self.tts_session_id,
                                "reason": "frame_encode_error",
                                "sequence_id": sequence_id,
                                "audio_bytes": len(wav_bytes),
                            }
                        }))
                        continue

                    if not frames:
                        continue

                    if self.audio_registry is None or not self.tts_session_id:
                        self._log_tts_audio_drop(
                            reason="no_audio_websocket",
                            sequence_id=sequence_id,
                            audio_bytes=len(wav_bytes),
                            dropped_frames=len(frames),
                        )
                        continue

                    for frame_index, frame_bytes in enumerate(frames):
                        sent = await self.audio_registry.send_audio_frame(self.tts_session_id, frame_bytes)
                        if not sent:
                            self._log_tts_audio_drop(
                                reason="audio_websocket_not_connected",
                                sequence_id=sequence_id,
                                audio_bytes=len(wav_bytes),
                                dropped_frames=len(frames) - frame_index,
                            )
                            break
        except websockets.exceptions.ConnectionClosed:
            return

    def _log_tts_audio_drop(
        self,
        *,
        reason: str,
        sequence_id: int,
        audio_bytes: int,
        dropped_frames: int,
    ) -> None:
        LOGGER.warning(
            "Dropping TTS audio because %s for session %s.",
            reason,
            self.tts_session_id,
        )
        METRICS_LOGGER.info(json.dumps({
            "tts_audio_dropped": {
                "id": self.client_id,
                "tts_session_id": self.tts_session_id,
                "reason": reason,
                "sequence_id": sequence_id,
                "audio_bytes": audio_bytes,
                "dropped_frames": dropped_frames,
            }
        }))

    def _update_tts_runtime_context(self) -> None:
        if self.tts_runtime is None:
            return
        self.tts_runtime.update_context(
            target_lang=self.target_lang,
            prompt_wav=self._active_prompt_wav(),
            prompt_text=self._active_prompt_text(),
        )

    def _tts_config_payload(self) -> dict:
        payload = _plain_config(self.tts_config)
        payload["target_lang"] = self.target_lang
        payload["active_prompt_wav"] = str(self._active_prompt_wav()) if self._active_prompt_wav() is not None else None
        payload["active_prompt_text"] = self._active_prompt_text()
        payload["speaker"] = self.speaker_id
        return payload

    def _configured_speakers(self) -> dict[str, SimpleNamespace]:
        speakers: dict[str, SimpleNamespace] = {}
        entries = getattr(self.tts_config, "speakers", []) or []
        for entry in entries:
            values = _namespace_to_dict(entry)
            speaker_id = str(values.get("id", "")).strip()
            if not speaker_id:
                continue
            speakers[speaker_id] = SimpleNamespace(
                id=speaker_id,
                label=str(values.get("label", speaker_id)),
                prompt_wav=str(values.get("prompt_wav", "") or ""),
                prompt_text=str(values.get("prompt_text", "") or ""),
                prompts=_normalize_speaker_prompts(values.get("prompts", {})),
            )
        if speakers:
            return speakers

        legacy_prompt_wav = str(getattr(self.tts_config, "prompt_wav", "") or "").strip()
        legacy_prompt_text = str(getattr(self.tts_config, "prompt_text", "") or "").strip()
        if legacy_prompt_wav or legacy_prompt_text:
            speakers["configured"] = SimpleNamespace(
                id="configured",
                label="Configured Voice",
                prompt_wav=legacy_prompt_wav,
                prompt_text=legacy_prompt_text,
            )
        else:
            speakers["default"] = SimpleNamespace(id="default", label="Default", prompt_wav="", prompt_text="")
        return speakers

    async def _select_speaker_for_runtime(self, requested_speaker: object) -> None:
        old_speaker = self.speaker_id
        self._select_speaker(requested_speaker)
        if self.enabled and self.speaker_id != old_speaker:
            await self._detach_tts_runtime()
            if self.tts_runtime is not None and self.tts_runtime.claim(self.client_id):
                await self._ensure_tts_worker_if_ready()
            else:
                self.enabled = False

    def _select_speaker(self, requested_speaker: object) -> None:
        speaker_id = str(requested_speaker or "").strip() or self.default_speaker_id
        speaker = self.speakers.get(speaker_id) or self.speakers.get(self.default_speaker_id) or self.speakers["default"]
        self.speaker_id = str(speaker.id)
        self._refresh_selected_speaker_prompt()

    def _refresh_selected_speaker_prompt(self) -> None:
        speaker = self.speakers.get(self.speaker_id) or self.speakers.get(self.default_speaker_id) or self.speakers["default"]
        prompt_values = _speaker_prompt_for_language(speaker, self.target_lang)
        self.config_prompt_wav = self._resolve_prompt_wav(prompt_values.get("prompt_wav", ""))
        self.config_prompt_text = str(prompt_values.get("prompt_text", "") or "").strip()
        self.prompt_ready.clear()
        if self.config_prompt_wav is not None or self.backend_name == "magictts":
            self.prompt_ready.set()

    def _resolve_prompt_wav(self, prompt_wav: str | Path | None) -> Path | None:
        text = str(prompt_wav or "").strip()
        if not text:
            return None
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = (ROOT_DIR / path).resolve()
        return path

    def _has_active_prompt(self) -> bool:
        return (
            self.backend_name == "magictts"
            or self.config_prompt_wav is not None
            or self.prompt_recorder.prompt_wav is not None
        )

    def _active_prompt_wav(self) -> Path | None:
        prompt_wav = self.config_prompt_wav or self.prompt_recorder.prompt_wav
        if prompt_wav is None and self.backend_name != "magictts":
            raise RuntimeError("TTS prompt audio is not ready yet.")
        return prompt_wav

    def _active_prompt_text(self) -> str | None:
        return self.config_prompt_text or None

    def _normalize_tts_text(self, text: str) -> str:
        return _normalize_magictts_chinese_text(text, self.target_lang, self.backend_name)

    def _should_flush_tts_text(self, text: str) -> bool:
        return self._tts_flush_units(text) >= self.flush_min_units

    def _tts_flush_units(self, text: str) -> int:
        lang = (self.target_lang or "").lower()
        if lang == "ja" or lang.startswith("ja-") or lang == "zh" or lang.startswith("zh-"):
            return sum(
                1
                for char in text
                if not char.isspace() and not unicodedata.category(char).startswith("P")
            )
        return len(text.split())


class DemoMessageProcessor:
    """Buffers browser PCM by audio_step_size before forwarding to VAD."""

    def __init__(self, client_id: int, speech_processor) -> None:
        self.client_buffer = b""
        self.received_audio_seconds = 0.0
        self.processed_audio_seconds = 0.0
        self.sample_rate = SAMPLE_RATE
        self.client_id = client_id
        self.speech_processor = speech_processor
        self.audio_step_size = float(
            getattr(speech_processor.config, "audio_step_size", 0.25)
        )
        self.first_audio_wall_time: float | None = None
        self.first_vad_speech_start_wall_time: float | None = None
        self.first_processor_call_wall_time: float | None = None
        self.first_non_empty_output_wall_time: float | None = None
        self.last_vad_speech_start = False
        self.last_text_send_metric: dict | None = None
        self.text_message_sequence = 0
        self.last_input_pace_metric_wall_time: float | None = None
        self.input_pace_metric_interval = float(
            getattr(speech_processor.config, "input_pace_metric_interval", 1.0)
        )

    def record_audio_received(self, speech_data: bytes, queue_size: int) -> None:
        if self.first_audio_wall_time is None and speech_data:
            self.first_audio_wall_time = time.time()
            self._log_latency_checkpoint("first_audio_received")
        if speech_data:
            self.received_audio_seconds += len(speech_data) / 2 / self.sample_rate
        self._log_input_pace(queue_size)

    def process_speech(self, speech_data: bytes) -> IncrementalOutput | None:
        if self.first_audio_wall_time is None and speech_data:
            self.first_audio_wall_time = time.time()
            self._log_latency_checkpoint("first_audio_received")
        self.last_vad_speech_start = False
        self.client_buffer += speech_data
        outputs: list[IncrementalOutput] = []
        step_bytes = self._step_bytes()

        while len(self.client_buffer) >= step_bytes:
            step_data = self.client_buffer[:step_bytes]
            self.client_buffer = self.client_buffer[step_bytes:]
            output = self._run_speech_processor(step_data)
            self.processed_audio_seconds += self.audio_step_size
            self._log_simulst_outputs(output)
            self._log_latency_from_output(output)
            if getattr(output, "vad_speech_start", False):
                self.last_vad_speech_start = True
            if self._has_client_output(output):
                outputs.append(output)

        if not outputs:
            return None
        return self._merge_outputs(outputs)

    def process_metadata(self, metadata: dict) -> None:
        if "sample_rate" in metadata:
            self.sample_rate = int(metadata["sample_rate"])
        if "target_lang" in metadata:
            self.speech_processor.set_target_language(metadata["target_lang"])
            LOGGER.debug("Client %s target language set to %s", self.client_id, metadata["target_lang"])
        if "source_lang" in metadata:
            self.speech_processor.set_source_language(metadata["source_lang"])
            LOGGER.debug("Client %s source language set to %s", self.client_id, metadata["source_lang"])

    def end_of_stream(self) -> IncrementalOutput | None:
        outputs: list[IncrementalOutput] = []
        if self.client_buffer:
            output = self._run_speech_processor(self.client_buffer)
            self.processed_audio_seconds += len(self.client_buffer) / 2 / self.sample_rate
            self.client_buffer = b""
            self._log_simulst_outputs(output, end_of_stream=True)
            self._log_latency_from_output(output)
            if self._has_client_output(output):
                outputs.append(output)

        output = self.speech_processor.end_of_stream()
        self._log_simulst_outputs(output, end_of_stream=True)
        self._log_latency_from_output(output)
        if self._has_client_output(output):
            outputs.append(output)
        merged = self._merge_outputs(outputs) if outputs else None
        self.clear()
        return merged

    def clear(self) -> None:
        self.speech_processor.clear()
        self.client_buffer = b""
        self.received_audio_seconds = 0.0
        self.processed_audio_seconds = 0.0
        self.sample_rate = SAMPLE_RATE
        self.first_audio_wall_time = None
        self.first_vad_speech_start_wall_time = None
        self.first_processor_call_wall_time = None
        self.first_non_empty_output_wall_time = None
        self.last_input_pace_metric_wall_time = None
        self.last_vad_speech_start = False
        self.last_text_send_metric = None
        self.text_message_sequence = 0

    def _step_bytes(self) -> int:
        samples = max(1, int(round(self.audio_step_size * self.sample_rate)))
        return samples * 2

    def _run_speech_processor(self, speech_data: bytes) -> IncrementalOutput:
        int16_waveform = np.frombuffer(speech_data, dtype=np.int16)
        float32_waveform = int16_waveform.astype(np.float32) / 2**15
        if self.sample_rate != SAMPLE_RATE and float32_waveform.size > 0:
            float32_waveform = librosa.resample(
                float32_waveform,
                orig_sr=self.sample_rate,
                target_sr=SAMPLE_RATE,
            )
        return self.speech_processor.process_chunk(float32_waveform)

    def output_to_json(self, output: IncrementalOutput) -> str:
        self.text_message_sequence += 1
        text_message_id = f"{self.client_id}:{self.text_message_sequence}"
        server_send_wall_time_ms = int(time.time() * 1000)
        server_send_perf_ms = time.perf_counter() * 1000
        payload = {
            "new": output.new_string,
            "deleted": output.deleted_string,
            "text_message_id": text_message_id,
            "server_send_wall_time_ms": server_send_wall_time_ms,
            "server_send_perf_ms": server_send_perf_ms,
            "processed_audio_seconds": self.processed_audio_seconds,
        }
        if getattr(output, "segment_end", False):
            payload["segment_end"] = True
        self.last_text_send_metric = {
            "id": self.client_id,
            "text_message_id": text_message_id,
            "server_send_wall_time_ms": server_send_wall_time_ms,
            "server_send_perf_ms": server_send_perf_ms,
            "new_chars": len(output.new_string or ""),
            "deleted_chars": len(output.deleted_string or ""),
            "segment_end": bool(getattr(output, "segment_end", False)),
            "processed_audio_seconds": self.processed_audio_seconds,
        }
        METRICS_LOGGER.info(json.dumps({"text_sent": self.last_text_send_metric}, ensure_ascii=False))
        return json.dumps(payload, ensure_ascii=False)

    def _merge_outputs(self, outputs: list[IncrementalOutput]) -> IncrementalOutput:
        merged = merge_incremental_outputs(outputs, self.speech_processor.tokens_to_string)
        if any(getattr(output, "segment_end", False) for output in outputs):
            merged.segment_end = True
        return merged

    def _log_simulst_outputs(
        self,
        output: IncrementalOutput | None,
        *,
        end_of_stream: bool = False,
    ) -> None:
        if output is None or not getattr(output, "processor_called", False):
            return

        entries = getattr(output, "simulst_outputs", None)
        if not entries:
            entries = [{
                "call_index": None,
                "chunk_seconds": None,
                "computation_time": None,
                "flush": False,
                "new_tokens": output.new_tokens,
                "new_text": output.new_string or "",
                "deleted_tokens": output.deleted_tokens,
                "deleted_text": output.deleted_string or "",
            }]

        for entry in entries:
            new_text = str(entry.get("new_text", "") or "")
            deleted_text = str(entry.get("deleted_text", "") or "")
            payload = {
                "id": self.client_id,
                "call_index": entry.get("call_index"),
                "processed_audio_seconds": self.processed_audio_seconds,
                "chunk_seconds": entry.get("chunk_seconds"),
                "computation_time": entry.get("computation_time"),
                "flush": bool(entry.get("flush", False)),
                "end_of_stream": end_of_stream,
                "segment_end": bool(getattr(output, "segment_end", False)),
                "new_text": new_text,
                "new_chars": len(new_text),
                "new_tokens": entry.get("new_tokens", []),
                "deleted_text": deleted_text,
                "deleted_chars": len(deleted_text),
                "deleted_tokens": entry.get("deleted_tokens", []),
                "empty": not new_text and not deleted_text,
                "timestamp": time.time(),
            }
            METRICS_LOGGER.info(json.dumps({"simulst_output": payload}, ensure_ascii=False))

    def _log_input_pace(self, queue_size: int) -> None:
        if self.first_audio_wall_time is None:
            return
        now = time.time()
        if (
            self.last_input_pace_metric_wall_time is not None
            and now - self.last_input_pace_metric_wall_time < self.input_pace_metric_interval
        ):
            return
        self.last_input_pace_metric_wall_time = now
        wall_elapsed = now - self.first_audio_wall_time
        client_buffer_seconds = len(self.client_buffer) / 2 / self.sample_rate
        payload = {
            "id": self.client_id,
            "wall_elapsed": wall_elapsed,
            "received_audio_seconds": self.received_audio_seconds,
            "processed_audio_seconds": self.processed_audio_seconds,
            "received_minus_wall": self.received_audio_seconds - wall_elapsed,
            "processed_minus_wall": self.processed_audio_seconds - wall_elapsed,
            "received_minus_processed": self.received_audio_seconds - self.processed_audio_seconds,
            "audio_queue_size": queue_size,
            "client_buffer_seconds": client_buffer_seconds,
            "timestamp": now,
        }
        METRICS_LOGGER.info(json.dumps({"input_pace": payload}, ensure_ascii=False))

    def _log_latency_from_output(self, output: IncrementalOutput | None) -> None:
        if output is None:
            return
        now = time.time()
        if getattr(output, "vad_speech_start", False) and self.first_vad_speech_start_wall_time is None:
            self.first_vad_speech_start_wall_time = now
            self._log_latency_checkpoint("first_vad_speech_start")
        if getattr(output, "processor_called", False) and self.first_processor_call_wall_time is None:
            self.first_processor_call_wall_time = now
            self._log_latency_checkpoint("first_processor_call")
        if (output.new_string or output.deleted_string) and self.first_non_empty_output_wall_time is None:
            self.first_non_empty_output_wall_time = now
            self._log_latency_checkpoint("first_non_empty_output")

    def _log_latency_checkpoint(self, event: str) -> None:
        now = time.time()
        elapsed = (
            0.0
            if self.first_audio_wall_time is None
            else now - self.first_audio_wall_time
        )
        payload = {
            "id": self.client_id,
            "event": event,
            "elapsed_since_first_audio": elapsed,
            "processed_audio_seconds": self.processed_audio_seconds,
        }
        LOGGER.info("Latency checkpoint: %s", json.dumps(payload, ensure_ascii=False))
        METRICS_LOGGER.info(json.dumps({"latency_checkpoint": payload}))

    @staticmethod
    def _has_client_output(output: IncrementalOutput | None) -> bool:
        return bool(output and (
            output.new_string
            or output.deleted_string
            or getattr(output, "segment_end", False)
        ))


class SpeechProcessorPool:
    def __init__(
        self,
        speech_processor_config: SimpleNamespace,
        size: int,
        acquire_timeout: float,
    ) -> None:
        self.acquire_timeout = acquire_timeout
        self.available: asyncio.Queue = asyncio.Queue(maxsize=size)
        for index in range(size):
            speech_processor = build_speech_processor(speech_processor_config)
            self._warmup_processor(speech_processor, index)
            self.available.put_nowait(speech_processor)

    @asynccontextmanager
    async def acquire(self):
        speech_processor = await asyncio.wait_for(
            self.available.get(),
            timeout=self.acquire_timeout,
        )
        try:
            yield speech_processor
        finally:
            self.available.put_nowait(speech_processor)

    @staticmethod
    def _warmup_processor(speech_processor, index: int) -> None:
        LOGGER.info("Warming up speech processor %s.", index)
        start_time = time.time()
        if hasattr(speech_processor, "warmup"):
            speech_processor.warmup()
        else:
            warmup_audio = np.zeros(max(1, int(round(speech_processor.speech_chunk_size * SAMPLE_RATE))), dtype=np.float32)
            speech_processor.process_chunk(warmup_audio)
            speech_processor.end_of_stream()
            speech_processor.clear()
        LOGGER.info("Warmed up speech processor %s in %.3f seconds.", index, time.time() - start_time)


def load_yaml_config(path: Path) -> SimpleNamespace:
    with path.expanduser().open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _to_namespace(raw)


def normalize_tts_config(config: SimpleNamespace | None) -> SimpleNamespace:
    if config is None:
        config = SimpleNamespace()

    common_defaults = {
        "enabled": False,
        "backend": "cosyvoice3",
        "prompt_wav": "",
        "prompt_text": "",
        "stream": False,
        "visible_devices": None,
        "default_speaker": "default",
        "speakers": [],
        "flush_min_units": 4,
        "audio_frame_ms": 200,
    }
    backend_defaults = {
        "cosyvoice3": {
            "runtime": "subprocess",
            "python": ".cosyvoice-venv/bin/python",
            "model_dir": "checkpoints/cosyvoice3",
            "prompt_duration_seconds": 4.0,
            "japanese_preprocess": "mecab_unidic",
            "instruction_prefix": "You are a helpful assistant.",
        },
        "magictts": {
            "runtime": "inprocess",
            "python": ".venv/bin/python",
            "magictts_dir": "third_party/MAGIC-TTS",
            "magictts_checkpoint": "third_party/MAGIC-TTS/checkpoints/magictts_36k.pt",
            "magictts_mode": "spontaneous",
            "magictts_steps": 32,
            "magictts_cfg_strength": 2.0,
            "magictts_sway_sampling_coef": -1.0,
            "magictts_speed": 1.0,
        },
    }

    common = _namespace_to_dict(getattr(config, "common", SimpleNamespace()))
    flat = _namespace_to_dict(config)
    backend_sections = {
        name: _namespace_to_dict(getattr(config, name, SimpleNamespace()))
        for name in backend_defaults
    }

    values = dict(common_defaults)
    values.update(common)
    for key, value in flat.items():
        if key not in {"common", "cosyvoice3", "magictts"}:
            values[key] = value

    backend = str(values.get("backend", "cosyvoice3")).lower()
    if backend not in backend_defaults:
        raise ValueError(f"Unsupported TTS backend: {backend}")
    flat_backend_values = {
        key: flat[key]
        for key in backend_defaults[backend]
        if key in flat
    }
    values["backend"] = backend
    values.update(backend_defaults[backend])
    values.update(flat_backend_values)
    values.update(backend_sections.get(backend, {}))
    if backend != "cosyvoice3":
        values.pop("prompt_duration_seconds", None)
    values["speakers"] = _normalize_tts_speakers(values)
    if values.get("default_speaker") not in {speaker.id for speaker in values["speakers"]}:
        values["default_speaker"] = "default"

    return SimpleNamespace(**values)



def _normalize_tts_speakers(values: dict) -> list[SimpleNamespace]:
    raw_entries = values.get("speakers") or []
    speakers: list[SimpleNamespace] = []
    seen: set[str] = set()

    def add_speaker(
        speaker_id: str,
        label: str,
        prompt_wav: str = "",
        prompt_text: str = "",
        prompts: object = None,
    ) -> None:
        speaker_id = str(speaker_id).strip()
        if not speaker_id or speaker_id in seen:
            return
        seen.add(speaker_id)
        speakers.append(SimpleNamespace(
            id=speaker_id,
            label=str(label or speaker_id),
            prompt_wav=str(prompt_wav or ""),
            prompt_text=str(prompt_text or ""),
            prompts=_normalize_speaker_prompts(prompts),
        ))

    if isinstance(raw_entries, list):
        for entry in raw_entries:
            item = _namespace_to_dict(entry)
            add_speaker(
                item.get("id", ""),
                item.get("label", item.get("id", "")),
                item.get("prompt_wav", ""),
                item.get("prompt_text", ""),
                item.get("prompts", {}),
            )

    if not speakers:
        legacy_prompt_wav = str(values.get("prompt_wav", "") or "").strip()
        legacy_prompt_text = str(values.get("prompt_text", "") or "").strip()
        if legacy_prompt_wav or legacy_prompt_text:
            add_speaker("configured", "Configured Voice", legacy_prompt_wav, legacy_prompt_text)
            values["default_speaker"] = "configured"
        else:
            add_speaker("default", "Default")
    elif "default" not in seen:
        speakers.insert(0, SimpleNamespace(id="default", label="Default", prompt_wav="", prompt_text=""))

    return speakers


def _namespace_to_dict(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, SimpleNamespace):
        return vars(value).copy()
    if isinstance(value, dict):
        return value.copy()
    return {}


def _to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


async def send_incremental_output(
    websocket: ServerConnection,
    send_lock: asyncio.Lock,
    message_processor: DemoMessageProcessor,
    output: IncrementalOutput,
) -> None:
    async with send_lock:
        await websocket.send(message_processor.output_to_json(output))


async def audio_processing_loop(
    audio_queue: asyncio.Queue[dict | None],
    websocket: ServerConnection,
    send_lock: asyncio.Lock,
    processor_lock: asyncio.Lock,
    audio_active: asyncio.Event,
    message_processor: DemoMessageProcessor,
    tts_coordinator: ClientTTSCoordinator,
) -> None:
    loop = asyncio.get_running_loop()
    try:
        while True:
            audio_item = await audio_queue.get()
            try:
                if audio_item is None:
                    return
                speech_data = bytes(audio_item.get("data", b""))
                async with processor_lock:
                    incremental_output = await loop.run_in_executor(
                        None,
                        message_processor.process_speech,
                        speech_data,
                    )
                    prompt_became_ready = tts_coordinator.accept_audio(
                        speech_data,
                        message_processor.sample_rate,
                        message_processor.last_vad_speech_start,
                    )
                if not audio_active.is_set():
                    continue
                if prompt_became_ready:
                    await tts_coordinator.flush_pending_if_ready()
                if incremental_output is not None:
                    await send_incremental_output(
                        websocket,
                        send_lock,
                        message_processor,
                        incremental_output,
                    )
                    await tts_coordinator.handle_output(incremental_output)
            finally:
                audio_queue.task_done()
    except asyncio.CancelledError:
        raise
    except websockets.exceptions.ConnectionClosed:
        return
    except Exception:
        LOGGER.exception("Audio processing loop failed for client %s.", message_processor.client_id)


async def websocket_handler(
    websocket: ServerConnection,
    processor_pool: SpeechProcessorPool,
    tts_config: SimpleNamespace,
    tts_runtime: TTSRuntime | None,
    audio_registry: TTSAudioWebSocketServer | None,
) -> None:
    client_id = id(websocket)
    LOGGER.info("Client %s connected.", client_id)
    try:
        async with processor_pool.acquire() as speech_processor:
            message_processor = DemoMessageProcessor(client_id, speech_processor)
            tts_coordinator = ClientTTSCoordinator(
                client_id,
                websocket,
                tts_config,
                tts_runtime,
                audio_registry,
            )
            send_lock = asyncio.Lock()
            processor_lock = asyncio.Lock()
            audio_active = asyncio.Event()
            audio_active.set()
            audio_queue: asyncio.Queue[dict | None] = asyncio.Queue()
            audio_task = asyncio.create_task(
                audio_processing_loop(
                    audio_queue,
                    websocket,
                    send_lock,
                    processor_lock,
                    audio_active,
                    message_processor,
                    tts_coordinator,
                )
            )
            try:
                async for message in websocket:
                    if isinstance(message, bytes):
                        audio_queue.put_nowait({"data": message})
                        message_processor.record_audio_received(message, audio_queue.qsize())
                    else:
                        await handle_text_message(
                            websocket,
                            send_lock,
                            processor_lock,
                            audio_active,
                            message_processor,
                            tts_coordinator,
                            message,
                            audio_queue,
                        )
            except websockets.exceptions.ConnectionClosed:
                LOGGER.info("Client %s disconnected.", client_id)
            finally:
                audio_task.cancel()
                try:
                    await audio_task
                except asyncio.CancelledError:
                    pass
                await tts_coordinator.close()
                message_processor.clear()
    except asyncio.TimeoutError:
        LOGGER.warning(
            "No processor available for client %s within %.1f seconds.",
            client_id,
            processor_pool.acquire_timeout,
        )
        try:
            await websocket.send(json.dumps({"server_busy": True}))
            await websocket.close(code=1013, reason="Server busy")
        except websockets.exceptions.ConnectionClosed:
            pass


async def handle_client_ping(
    websocket: ServerConnection,
    send_lock: asyncio.Lock,
    message_processor: DemoMessageProcessor,
    ping: dict,
) -> None:
    server_receive_wall_time_ms = int(time.time() * 1000)
    server_receive_perf_ms = time.perf_counter() * 1000
    ping_id = ping.get("ping_id")
    server_send_wall_time_ms = int(time.time() * 1000)
    server_send_perf_ms = time.perf_counter() * 1000
    async with send_lock:
        await websocket.send(json.dumps({
            "client_pong": {
                "ping_id": ping_id,
                "client_send_wall_time_ms": ping.get("client_send_wall_time_ms"),
                "client_send_perf_ms": ping.get("client_send_perf_ms"),
                "server_receive_wall_time_ms": server_receive_wall_time_ms,
                "server_receive_perf_ms": server_receive_perf_ms,
                "server_send_wall_time_ms": server_send_wall_time_ms,
                "server_send_perf_ms": server_send_perf_ms,
            }
        }, ensure_ascii=False))


async def handle_text_message(
    websocket: ServerConnection,
    send_lock: asyncio.Lock,
    processor_lock: asyncio.Lock,
    audio_active: asyncio.Event,
    message_processor: DemoMessageProcessor,
    tts_coordinator: ClientTTSCoordinator,
    message: str,
    audio_queue: asyncio.Queue[dict | None] | None = None,
) -> None:
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        LOGGER.warning("Ignoring invalid JSON message: %s", message)
        return

    if isinstance(data.get("client_ping"), dict):
        await handle_client_ping(websocket, send_lock, message_processor, data["client_ping"])
        return

    if data.get("abort_stream"):
        audio_active.clear()
        if audio_queue is not None:
            while True:
                try:
                    audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                else:
                    audio_queue.task_done()
        loop = asyncio.get_running_loop()
        async with processor_lock:
            await loop.run_in_executor(None, message_processor.clear)
        await tts_coordinator.clear()
        async with send_lock:
            await websocket.send(json.dumps({"stream_aborted": True}, ensure_ascii=False))
        return

    if data.get("end_of_stream"):
        if audio_queue is not None:
            await audio_queue.join()
        loop = asyncio.get_running_loop()
        async with processor_lock:
            incremental_output = await loop.run_in_executor(
                None,
                message_processor.end_of_stream,
            )
        if incremental_output is not None:
            await send_incremental_output(websocket, send_lock, message_processor, incremental_output)
            await tts_coordinator.handle_output(incremental_output)
        await tts_coordinator.drain()
        async with send_lock:
            await websocket.send(json.dumps({"end_of_processing": True}))
        return

    if data.get("clear_context"):
        audio_active.clear()
        if audio_queue is not None:
            while True:
                try:
                    audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                else:
                    audio_queue.task_done()
        loop = asyncio.get_running_loop()
        async with processor_lock:
            await loop.run_in_executor(None, message_processor.clear)
        await tts_coordinator.clear()
        audio_active.set()
        async with send_lock:
            await websocket.send(json.dumps({"context_cleared": True}))
        return

    if data.get("frontend_metrics"):
        metrics = data.get("frontend_metrics")
        if isinstance(metrics, list):
            for item in metrics:
                if isinstance(item, dict):
                    payload = {"id": message_processor.client_id, **item}
                    METRICS_LOGGER.info(json.dumps({"frontend": payload}, ensure_ascii=False))
        elif isinstance(metrics, dict):
            payload = {"id": message_processor.client_id, **metrics}
            METRICS_LOGGER.info(json.dumps({"frontend": payload}, ensure_ascii=False))
        return

    message_processor.process_metadata(data)
    await tts_coordinator.configure(data)



class AiohttpWebSocketAdapter:
    def __init__(self, websocket) -> None:
        self.websocket = websocket

    async def send(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            await self.websocket.send_bytes(message)
        else:
            await self.websocket.send_str(message)

    async def recv(self) -> str | bytes:
        from aiohttp import WSMsgType

        message = await self.websocket.receive()
        if message.type == WSMsgType.TEXT:
            return message.data
        if message.type == WSMsgType.BINARY:
            return message.data
        raise websockets.exceptions.ConnectionClosed(None, None)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        await self.websocket.close(code=code, message=reason.encode("utf-8"))

    def __aiter__(self):
        return self

    async def __anext__(self) -> str | bytes:
        from aiohttp import WSMsgType

        message = await self.websocket.receive()
        if message.type == WSMsgType.TEXT:
            return message.data
        if message.type == WSMsgType.BINARY:
            return message.data
        raise StopAsyncIteration


async def run_aiohttp_tts_audio_handler(request, audio_registry: TTSAudioWebSocketServer):
    from aiohttp import WSMsgType, web

    websocket = web.WebSocketResponse(max_msg_size=0)
    await websocket.prepare(request)
    adapter = AiohttpWebSocketAdapter(websocket)
    session_id = None
    try:
        message = await websocket.receive(timeout=10.0)
        if message.type != WSMsgType.TEXT:
            await websocket.close(code=1003, message=b"Expected JSON registration message")
            return websocket
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            await websocket.close(code=1003, message=b"Invalid JSON registration message")
            return websocket
        session_id = str(payload.get("tts_session_id") or "").strip()
        if not session_id:
            await websocket.close(code=1008, message=b"Missing tts_session_id")
            return websocket
        await audio_registry._register(session_id, adapter)
        LOGGER.info("TTS audio websocket connected for session %s.", session_id)
        async for message in websocket:
            if message.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                break
    finally:
        if session_id is not None:
            await audio_registry._unregister(session_id, adapter)
            LOGGER.info("TTS audio websocket disconnected for session %s.", session_id)
    return websocket


async def run_http_server(
    web_dir: Path,
    server_config_path: Path,
    speech_config_path: Path,
    host: str,
    port: int,
    *,
    websocket_handler_func=None,
    audio_registry: TTSAudioWebSocketServer | None = None,
    single_origin_websockets: bool = False,
    ws_path: str = "/ws",
    audio_ws_path: str = "/tts-audio",
    public_url: str = "",
) -> None:
    from aiohttp import web

    def load_web_config() -> dict:
        with open(server_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        with open(speech_config_path, "r", encoding="utf-8") as f:
            speech_config = yaml.safe_load(f) or {}
        if isinstance(config, dict) and isinstance(speech_config, dict) and "tts" in speech_config:
            config["tts"] = speech_config["tts"]
        if single_origin_websockets:
            webui = config.setdefault("webui", {})
            webui["ws_path"] = ws_path
            webui["audio_ws_path"] = audio_ws_path
            webui["ws_hostname"] = ""
            webui["audio_ws_hostname"] = ""
            webui["ws_port"] = ""
            webui["audio_ws_port"] = ""
        return config

    async def config_handler(request):
        return web.Response(
            body=yaml.safe_dump(load_web_config(), sort_keys=False, allow_unicode=True).encode("utf-8"),
            content_type="text/yaml",
            charset="utf-8",
        )

    async def index_handler(request):
        return web.FileResponse(web_dir / "index.html")

    async def websocket_route(request):
        if websocket_handler_func is None:
            raise web.HTTPNotFound()
        websocket = web.WebSocketResponse(max_msg_size=0)
        await websocket.prepare(request)
        await websocket_handler_func(AiohttpWebSocketAdapter(websocket))
        return websocket

    async def tts_audio_route(request):
        if audio_registry is None:
            raise web.HTTPNotFound()
        return await run_aiohttp_tts_audio_handler(request, audio_registry)

    app = web.Application(client_max_size=0)
    app.router.add_get("/", index_handler)
    app.router.add_get("/config.yaml", config_handler)
    if single_origin_websockets:
        app.router.add_get(ws_path, websocket_route)
        app.router.add_get(audio_ws_path, tts_audio_route)
    app.router.add_static("/", path=str(web_dir), show_index=False)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    LOGGER.info("Serving demo UI at http://%s:%s from %s", host, port, web_dir)
    if single_origin_websockets:
        LOGGER.info("Serving single-origin websocket paths at %s and %s", ws_path, audio_ws_path)
    if public_url:
        LOGGER.info("============================================================")
        LOGGER.info("Demo public URL: %s", public_url)
        LOGGER.info("============================================================")
    try:
        await asyncio.Future()
    finally:
        await runner.cleanup()


def reset_metrics_log_file(metrics_config: SimpleNamespace) -> None:
    if not bool(getattr(metrics_config, "enabled", False)):
        return
    filename = getattr(metrics_config, "filename", None)
    if not filename:
        return
    path = Path(filename).expanduser()
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


async def main(args: argparse.Namespace) -> None:
    server_config = load_yaml_config(args.server_config)
    speech_config = load_yaml_config(args.speech_processor_config)
    reset_metrics_log_file(getattr(server_config, "metrics", SimpleNamespace(enabled=False)))
    setup_metrics_logger(getattr(server_config, "metrics", SimpleNamespace(enabled=False)))
    tts_config = normalize_tts_config(getattr(speech_config, "tts", getattr(server_config, "tts", None)))
    _apply_visible_devices(getattr(speech_config, "simulst_visible_devices", None), label="SimulST")
    tts_runtime = None
    if bool(getattr(tts_config, "enabled", False)):
        tts_runtime = TTSRuntime()
        start_time = time.time()
        LOGGER.info("Starting TTS worker at demo startup.")
        tts_runtime.start(_startup_tts_config_payload(
            tts_config,
            str(getattr(speech_config, "target_language", "") or "") or None,
        ))
        LOGGER.info("Started TTS worker in %.3f seconds.", time.time() - start_time)

    pool_size = int(getattr(server_config, "pool_size", 1))
    acquire_timeout = float(getattr(server_config, "acquire_timeout", 30))
    LOGGER.info("Loading speech processor: %s", speech_config.type)
    start_time = time.time()
    processor_pool = SpeechProcessorPool(speech_config, pool_size, acquire_timeout)
    METRICS_LOGGER.info(json.dumps({"model_loading_time": time.time() - start_time}))

    ws_host = str(getattr(server_config, "ws_hostname", getattr(server_config, "hostname", "localhost")))
    ws_port = int(getattr(server_config, "ws_port", getattr(server_config, "port", 8080)))
    audio_ws_host = str(getattr(server_config, "audio_ws_hostname", ws_host))
    audio_ws_port = int(getattr(server_config, "audio_ws_port", ws_port + 1))
    audio_server = TTSAudioWebSocketServer(audio_ws_host, audio_ws_port)
    http_host = str(getattr(server_config, "http_hostname", getattr(server_config, "hostname", "localhost")))
    http_port = int(getattr(server_config, "http_port", 7860))
    single_origin_websockets = bool(getattr(server_config, "single_origin_websockets", False))
    single_origin_ws_path = str(getattr(server_config, "single_origin_ws_path", "/ws"))
    single_origin_audio_ws_path = str(getattr(server_config, "single_origin_audio_ws_path", "/tts-audio"))
    demo_public_url = str(getattr(server_config, "demo_public_url", "") or "").strip()
    web_dir = Path(getattr(server_config, "web_directory", DEFAULT_WEB_DIR)).expanduser()
    if not web_dir.is_absolute():
        web_dir = (ROOT_DIR / web_dir).resolve()

    handler = functools.partial(
        websocket_handler,
        processor_pool=processor_pool,
        tts_config=tts_config,
        tts_runtime=tts_runtime,
        audio_registry=audio_server,
    )
    audio_server.start()
    LOGGER.info("Serving websocket at ws://%s:%s", ws_host, ws_port)
    try:
        async with serve(handler, ws_host, ws_port, ping_timeout=None, max_size=None):
            await asyncio.gather(
                run_http_server(
                    web_dir,
                    args.server_config,
                    args.speech_processor_config,
                    http_host,
                    http_port,
                    websocket_handler_func=handler,
                    audio_registry=audio_server,
                    single_origin_websockets=single_origin_websockets,
                    ws_path=single_origin_ws_path,
                    audio_ws_path=single_origin_audio_ws_path,
                    public_url=demo_public_url,
                ),
                asyncio.Future(),
            )
    finally:
        audio_server.stop()
        if tts_runtime is not None:
            tts_runtime.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("simpotrans_demo_server")
    parser.add_argument(
        "--server-config",
        type=Path,
        default=DEMO_DIR / "config" / "server.yaml",
    )
    parser.add_argument(
        "--speech-processor-config",
        type=Path,
        default=DEMO_DIR / "config" / "processor_vad.yaml",
    )
    return parser.parse_args()


def cli_main() -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    asyncio.run(main(parse_args()))


if __name__ == "__main__":
    cli_main()
