#!/usr/bin/env python3
"""Dependency-free source snapshot checks; run before installing dependencies."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PARTS = {'.git', '.env', '.aws', '.codex', '.agents', '__pycache__',
                   'analysis', 'wandb', 'exp_results', '.venv', '.comet-venv',
                   '.metricx-venv', '.cosyvoice-venv', '.uv-cache'}
FORBIDDEN_SUFFIXES = {'.wav', '.mp3', '.flac', '.mp4', '.pt', '.pth', '.ckpt',
                      '.safetensors', '.log', '.pem', '.key'}
PATTERNS = {
    'private absolute path': re.compile(r'/(?:home|Users|mnt|scratch|lustre)/[^\s"\x27]+'),
    'credential': re.compile(r'\b(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|sk-[A-Za-z0-9]{24,})\b'),
    'private key': re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
    'URL credentials': re.compile(r'https?://[^/\s]+:[^/\s]+@'),
}


def main():
    failures = []
    manifest = json.loads((ROOT / 'RELEASE_MANIFEST.json').read_text())
    files = {}
    for directory, dirs, names in os.walk(ROOT):
        # A fresh clone has local Git metadata; it is not part of the snapshot.
        if Path(directory) == ROOT:
            dirs[:] = [name for name in dirs if name != '.git']
            names = [name for name in names if name != '.git']
        for name in names + [name for name in dirs if (Path(directory) / name).is_symlink()]:
            path = Path(directory) / name
            files[str(path.relative_to(ROOT))] = path
    if set(files) != set(manifest) | {'RELEASE_MANIFEST.json'}:
        failures.append('File inventory differs from RELEASE_MANIFEST.json')
    counts = {'python': 0, 'shell': 0}
    for name, path in files.items():
        if path.is_symlink() or set(path.relative_to(ROOT).parts) & FORBIDDEN_PARTS or path.suffix in FORBIDDEN_SUFFIXES:
            failures.append(f'Excluded file: {name}')
            continue
        data = path.read_bytes()
        if name in manifest and hashlib.sha256(data).hexdigest() != manifest[name]:
            failures.append(f'Checksum mismatch: {name}')
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError:
            if name != 'third_party/WhisperRT-Streaming/whisper_rt/assets/mel_filters.npz':
                failures.append(f'Unexpected binary: {name}')
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                failures.append(f'{label}: {name}')
        # Upstream source can contain public author attribution; preserve it.
        if not name.startswith('third_party/') and name != 'scripts/check_release.py':
            if re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', text):
                failures.append(f'Email address: {name}')
            if 'src/analysis' in text or 'src.analysis' in text:
                failures.append(f'Excluded dependency: {name}')
        if path.suffix == '.py':
            try:
                ast.parse(text, filename=name)
                counts['python'] += 1
            except SyntaxError as error:
                failures.append(f'Python syntax: {name}: {error}')
        elif path.suffix == '.sh':
            shell = 'zsh' if 'zsh' in text.splitlines()[0] else 'bash'
            result = subprocess.run([shell, '-n', str(path)], capture_output=True, text=True)
            if result.returncode:
                failures.append(f'Shell syntax: {name}: {result.stderr}')
            counts['shell'] += 1
        elif path.suffix == '.json':
            json.loads(text)
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f"Release checks passed: {len(files)} files, {counts['python']} Python, {counts['shell']} shell.")


if __name__ == '__main__':
    main()
