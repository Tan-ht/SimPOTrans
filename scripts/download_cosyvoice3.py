from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "checkpoints" / "cosyvoice3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("download_cosyvoice3")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = (ROOT_DIR / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=output_dir,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    print(f"CosyVoice 3 model downloaded to {output_dir}")


if __name__ == "__main__":
    main()
