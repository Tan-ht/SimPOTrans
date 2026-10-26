from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol


@dataclass(frozen=True)
class AudioChunk:
    text: str
    sample_rate: int
    wav_bytes: bytes


class TTSBackend(Protocol):
    def warmup(self) -> None:
        """Load model resources needed for synthesis."""

    def synthesize(
        self,
        text: str,
        target_lang: str | None = None,
        prompt_wav: str | Path | None = None,
        prompt_text: str | None = None,
    ) -> AudioChunk:
        """Synthesize one text update into a WAV audio chunk."""

    def stream_synthesize(
        self,
        text: str,
        target_lang: str | None = None,
        prompt_wav: str | Path | None = None,
        prompt_text: str | None = None,
    ) -> Iterator[AudioChunk]:
        """Yield WAV audio chunks as the backend produces them."""


class SubprocessTTSBackend:
    def __init__(self, config: SimpleNamespace, root_dir: Path) -> None:
        self.config = config
        self.root_dir = root_dir
        default_python = self._default_python(config)
        configured_python = getattr(config, "python", default_python)
        if (
            str(getattr(config, "backend", "cosyvoice3")).lower() == "magictts"
            and str(configured_python) == ".cosyvoice-venv/bin/python"
        ):
            configured_python = default_python
        self.python = self._resolve_python(configured_python)
        self._process: subprocess.Popen[str] | None = None
        self._stderr_file = None
        self._next_request_id = 0

    def warmup(self) -> None:
        self._request("warmup")

    def synthesize(
        self,
        text: str,
        target_lang: str | None = None,
        prompt_wav: str | Path | None = None,
        prompt_text: str | None = None,
    ) -> AudioChunk:
        response = self._request(
            "synthesize",
            text=text,
            target_lang=target_lang,
            prompt_wav=str(prompt_wav) if prompt_wav is not None else None,
            prompt_text=prompt_text,
        )
        return self._decode_chunk(response["chunk"])

    def stream_synthesize(
        self,
        text: str,
        target_lang: str | None = None,
        prompt_wav: str | Path | None = None,
        prompt_text: str | None = None,
    ) -> Iterator[AudioChunk]:
        response = self._request(
            "stream_synthesize",
            text=text,
            target_lang=target_lang,
            prompt_wav=str(prompt_wav) if prompt_wav is not None else None,
            prompt_text=prompt_text,
        )
        for chunk in response["chunks"]:
            yield self._decode_chunk(chunk)

    def shutdown(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            try:
                self._send_request(process, {"action": "shutdown"})
                process.wait(timeout=5)
            except Exception:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

    def _request(self, action: str, **payload) -> dict:
        process = self._ensure_process()
        self._next_request_id += 1
        request = {
            "id": self._next_request_id,
            "action": action,
            "config": self._config_payload(),
            **payload,
        }
        self._send_request(process, request)
        line = process.stdout.readline() if process.stdout is not None else ""
        if not line:
            stderr = self._read_stderr()
            raise RuntimeError(f"TTS worker exited before responding to {action}: {stderr}")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"TTS worker returned invalid JSON: {line.strip()}") from exc
        if response.get("id") not in (None, request["id"]):
            raise RuntimeError(f"TTS worker response id mismatch for {action}")
        if not response.get("ok", False):
            error = response.get("error", "unknown TTS worker error")
            traceback = response.get("traceback", "")
            if traceback:
                error = f"{error}\n{traceback}"
            raise RuntimeError(error)
        return response

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        if not self.python.exists():
            raise FileNotFoundError(
                f"TTS python does not exist: {self.python}. Configure tts.python for the selected backend."
            )

        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None)
        site_packages = self._target_site_packages()
        pythonpath = [
            site_packages,
            str(self.root_dir / "src"),
            str(self.root_dir),
            str(self.root_dir / "third_party" / "CosyVoice"),
            str(self.root_dir / "third_party" / "CosyVoice" / "third_party" / "Matcha-TTS"),
            str(self.root_dir / "third_party" / "MAGIC-TTS"),
            str(self.root_dir / "third_party" / "MAGIC-TTS" / "inference"),
            str(self.root_dir / "third_party" / "MAGIC-TTS" / "vendor" / "f5tts_duration_ft"),
        ]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)

        self._stderr_file = tempfile.NamedTemporaryFile(
            "w+",
            encoding="utf-8",
            prefix="simpotrans-tts-worker-",
            suffix=".log",
            delete=False,
        )
        self._process = subprocess.Popen(
            [str(self.python), "-m", "tts.worker", "--root-dir", str(self.root_dir)],
            cwd=str(self.root_dir),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            text=True,
            bufsize=1,
        )
        return self._process

    def _send_request(self, process: subprocess.Popen[str], payload: dict) -> None:
        if process.stdin is None:
            raise RuntimeError("TTS worker stdin is closed")
        process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        process.stdin.flush()

    def _config_payload(self) -> dict:
        return {
            key: value
            for key, value in vars(self.config).items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }

    def _resolve_python(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.root_dir / path
        return path

    def _target_site_packages(self) -> str:
        try:
            return subprocess.check_output(
                [str(self.python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                cwd=str(self.root_dir),
                text=True,
            ).strip()
        except Exception:
            venv_dir = self.python.parent.parent
            return str(
                venv_dir
                / "lib"
                / f"python{sys.version_info.major}.{sys.version_info.minor}"
                / "site-packages"
            )

    @staticmethod
    def _default_python(config: SimpleNamespace) -> str:
        backend = str(getattr(config, "backend", "cosyvoice3")).lower()
        if backend == "magictts":
            return ".venv/bin/python"
        return ".cosyvoice-venv/bin/python"

    @staticmethod
    def _decode_chunk(value: dict) -> AudioChunk:
        return AudioChunk(
            text=str(value["text"]),
            sample_rate=int(value["sample_rate"]),
            wav_bytes=base64.b64decode(value.get("wav_bytes", "")),
        )

    def _read_stderr(self) -> str:
        stderr_file = self._stderr_file
        if stderr_file is None:
            return ""
        try:
            stderr_file.flush()
            stderr_file.seek(0)
            stderr = stderr_file.read().strip()
        except OSError:
            return ""
        return stderr[-20000:]


def build_tts_backend(config: SimpleNamespace, root_dir: Path) -> TTSBackend:
    backend = str(getattr(config, "backend", "cosyvoice3")).lower()
    if backend not in {"cosyvoice3", "magictts"}:
        raise ValueError(f"Unsupported TTS backend: {backend}")

    runtime = str(getattr(config, "runtime", "subprocess"))
    if runtime == "subprocess":
        return SubprocessTTSBackend(config, root_dir)
    if runtime != "inprocess":
        raise ValueError(f"Unsupported TTS runtime: {runtime}")

    if backend == "magictts":
        from .magictts import MagicTTSBackend

        return MagicTTSBackend(config, root_dir)

    from .cosyvoice3 import CosyVoice3Backend

    return CosyVoice3Backend(config, root_dir)
