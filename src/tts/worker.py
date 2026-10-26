from __future__ import annotations

import argparse
import base64
import contextlib
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

from .backend import AudioChunk


def encode_chunk(chunk: AudioChunk) -> dict:
    return {
        "text": chunk.text,
        "sample_rate": chunk.sample_rate,
        "wav_bytes": base64.b64encode(chunk.wav_bytes).decode("ascii"),
    }


class TTSWorker:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.backend = None

    def handle(self, request: dict) -> dict:
        action = request.get("action")
        if action == "shutdown":
            return {"ok": True, "shutdown": True}
        if action == "warmup":
            backend = self._backend(request)
            backend.warmup()
            return {"ok": True}
        if action == "synthesize":
            backend = self._backend(request)
            chunk = backend.synthesize(
                str(request.get("text", "")),
                request.get("target_lang"),
                request.get("prompt_wav"),
                request.get("prompt_text"),
            )
            return {"ok": True, "chunk": encode_chunk(chunk)}
        if action == "stream_synthesize":
            backend = self._backend(request)
            chunks = [
                encode_chunk(chunk)
                for chunk in backend.stream_synthesize(
                    str(request.get("text", "")),
                    request.get("target_lang"),
                    request.get("prompt_wav"),
                    request.get("prompt_text"),
                )
            ]
            return {"ok": True, "chunks": chunks}
        raise ValueError(f"Unsupported TTS worker action: {action}")

    def _backend(self, request: dict):
        if self.backend is not None:
            return self.backend
        config = dict(request.get("config") or {})
        config["runtime"] = "inprocess"
        from .backend import build_tts_backend

        self.backend = build_tts_backend(SimpleNamespace(**config), self.root_dir)
        return self.backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("simpotrans_tts_worker")
    parser.add_argument("--root-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = args.root_dir.expanduser().resolve()
    for path in (root_dir / "src", root_dir):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)

    worker = TTSWorker(root_dir)
    for line in sys.stdin:
        if not line.strip():
            continue
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            with contextlib.redirect_stdout(sys.stderr):
                response = worker.handle(request)
            response["id"] = request_id
        except Exception as exc:
            response = {
                "id": request_id,
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        if response.get("shutdown"):
            break


if __name__ == "__main__":
    main()
