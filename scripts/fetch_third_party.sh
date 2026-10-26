#!/usr/bin/env bash
set -euo pipefail
ROOT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 - "${ROOT_PATH}" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
for source in json.loads((root / 'third_party/sources.json').read_text()):
    target = root / source['path']
    revision = source['revision']
    if not target.exists():
        subprocess.run(['git', 'clone', '--no-checkout', source['url'], str(target)], check=True)
    elif not (target / '.git').exists():
        raise SystemExit(f'Refusing to overwrite a non-Git directory: {target}')
    def git(*args, **kwargs):
        return subprocess.run(['git', '-C', str(target), *args], check=True, **kwargs)
    head = subprocess.run(['git', '-C', str(target), 'rev-parse', 'HEAD'],
                          capture_output=True, text=True)
    # A fresh --no-checkout clone has an empty index; existing worktrees must be clean.
    if (target / '.git/index').exists():
        status = git('status', '--porcelain', capture_output=True, text=True).stdout
        if status.strip():
            raise SystemExit(f'Local modifications in {target}; preserve them before setup.')
    present = subprocess.run(['git', '-C', str(target), 'cat-file', '-e', revision + '^{commit}'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if present.returncode:
        git('fetch', 'origin', revision)
    git('checkout', '--detach', revision)
    git('submodule', 'update', '--init', '--recursive')
    actual = git('rev-parse', 'HEAD', capture_output=True, text=True).stdout.strip()
    if actual != revision:
        raise SystemExit(f'Unexpected revision for {target}')
    print(f'Ready: {source["path"]} @ {revision}')
PY
