from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


DEFAULT_LANG_PAIRS = ("en-ja", "en-zh")
DEFAULT_SPLITS = ("train", "dev", "tst-COMMON")
SUPPORTED_TARGET_LANGS = {"ja", "zh"}
STRONG_END_PUNCT = {"。", "？", "！", "?", "!"}
TRAILING_CLOSERS = set("」』”）)]}'\"")
QUOTE_CHARS_ZH = {'"', "“", "”", "「", "」", "『", "』"}


def normalize_target_text(
    text: str,
    lang: str,
    src_text: str | None = None,
    final_punctuation_policy: str = "from-src",
) -> str:
    """Normalize MuST-C Chinese/Japanese target text."""

    normalized_lang = _normalize_lang(lang)
    normalized = _normalize_whitespace(text)
    normalized = _remove_subtitle_spaces(normalized)
    normalized = _remove_inner_quote_spaces(normalized)

    if normalized_lang == "zh":
        normalized = _normalize_zh_punctuation(normalized)
        normalized = _normalize_zh_quotes(normalized)

    normalized = _remove_inner_quote_spaces(normalized)
    if final_punctuation_policy == "from-src":
        normalized = _append_final_punctuation_from_src(
            normalized, normalized_lang, src_text or ""
        )
    elif final_punctuation_policy != "none":
        raise ValueError(
            "`final_punctuation_policy` must be one of: from-src, none."
        )

    return normalized


def normalize_jsonl_file(
    input_path: Path,
    output_path: Path,
    *,
    lang: str,
    final_punctuation_policy: str,
    overwrite: bool,
    dry_run: bool = False,
    sample_limit: int = 5,
    progress: bool = True,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(
            f"{output_path} already exists. Pass --overwrite to replace it."
        )

    total = _count_lines(input_path)
    stats = _empty_stats(input_path, output_path)
    output_handle = None
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_path.open("w", encoding="utf-8")

    try:
        with input_path.open("r", encoding="utf-8") as input_handle:
            rows = _iter_with_progress(
                input_handle,
                total=total,
                label=f"{input_path.parent.name}/{input_path.name}",
                enabled=progress,
            )
            for line_no, line in enumerate(rows, start=1):
                stripped = line.strip()
                if not stripped:
                    if output_handle is not None:
                        output_handle.write(line)
                    continue

                row = json.loads(stripped)
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL row must be an object at {input_path}:{line_no}.")

                original = str(row.get("tgt_text", ""))
                src_text = str(row.get("src_text", ""))
                normalized = normalize_target_text(
                    original,
                    lang,
                    src_text=src_text,
                    final_punctuation_policy=final_punctuation_policy,
                )
                _update_stats(
                    stats,
                    original=original,
                    normalized=normalized,
                    src_text=src_text,
                    sample_limit=sample_limit,
                )
                row["tgt_text"] = normalized

                if output_handle is not None:
                    output_handle.write(
                        json.dumps(row, ensure_ascii=False, separators=(",", ": "))
                    )
                    output_handle.write("\n")
    finally:
        if output_handle is not None:
            output_handle.close()

    return stats


def output_path_for_split(lang_pair_dir: Path, split: str, suffix: str) -> Path:
    return lang_pair_dir / f"{split}{suffix}.jsonl"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize MuST-C En-Zh/En-Ja JSONL target text manifests."
    )
    parser.add_argument("--input-root", type=Path, default=Path("data/mustc"))
    parser.add_argument("--lang-pairs", nargs="+", default=list(DEFAULT_LANG_PAIRS))
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--suffix", default="_norm")
    parser.add_argument(
        "--final-punctuation-policy",
        choices=("from-src", "none"),
        default="from-src",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample-limit", type=int, default=5)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    reports: dict[str, Any] = {}

    for lang_pair in args.lang_pairs:
        src_lang, tgt_lang = _parse_lang_pair(lang_pair)
        del src_lang
        lang_pair_dir = args.input_root / lang_pair
        if not lang_pair_dir.is_dir():
            raise FileNotFoundError(f"Missing language-pair directory: {lang_pair_dir}")

        split_reports: dict[str, Any] = {}
        for split in args.splits:
            input_path = lang_pair_dir / f"{split}.jsonl"
            output_path = output_path_for_split(lang_pair_dir, split, args.suffix)
            if not input_path.exists():
                raise FileNotFoundError(f"Missing split manifest: {input_path}")

            split_reports[split] = normalize_jsonl_file(
                input_path,
                output_path,
                lang=tgt_lang,
                final_punctuation_policy=args.final_punctuation_policy,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                sample_limit=args.sample_limit,
                progress=not args.no_progress,
            )

        reports[lang_pair] = {
            "lang_pair": lang_pair,
            "target_lang": tgt_lang,
            "final_punctuation_policy": args.final_punctuation_policy,
            "splits": split_reports,
        }

        report_path = lang_pair_dir / "normalization_report.json"
        if not args.dry_run:
            with report_path.open("w", encoding="utf-8") as report_file:
                json.dump(reports[lang_pair], report_file, ensure_ascii=False, indent=2)
                report_file.write("\n")

    _print_summary(reports, dry_run=args.dry_run)
    return 0


def _normalize_lang(lang: str) -> str:
    normalized = lang.lower()
    if normalized in {"zh-cn", "zh-tw", "zh_hans", "zh_hant"}:
        normalized = "zh"
    if normalized not in SUPPORTED_TARGET_LANGS:
        raise ValueError(f"Unsupported target language for MuST-C CJK normalization: {lang}")
    return normalized


def _parse_lang_pair(lang_pair: str) -> tuple[str, str]:
    parts = lang_pair.split("-", 1)
    if len(parts) != 2:
        raise ValueError(f"Language pair must look like src-tgt: {lang_pair}")
    return parts[0], _normalize_lang(parts[1])


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _remove_subtitle_spaces(text: str) -> str:
    pieces: list[str] = []
    last = 0
    for match in re.finditer(r" +", text):
        pieces.append(text[last : match.start()])
        left = text[match.start() - 1] if match.start() > 0 else ""
        right = text[match.end()] if match.end() < len(text) else ""
        if _is_ascii_alnum(left) and _is_ascii_alnum(right):
            pieces.append(" ")
        last = match.end()
    pieces.append(text[last:])
    return "".join(pieces)


def _remove_inner_quote_spaces(text: str) -> str:
    text = re.sub(r"([“「『\"(（]) +", r"\1", text)
    return re.sub(r" +([”」』\"）)])", r"\1", text)


def _normalize_zh_punctuation(text: str) -> str:
    punctuation_map = {
        ",": "，",
        "?": "？",
        "!": "！",
        ":": "：",
        ";": "；",
    }
    chars = list(text)
    for index, char in enumerate(chars):
        if char in punctuation_map and _is_zh_context(chars, index):
            chars[index] = punctuation_map[char]
        elif char == "." and _should_convert_zh_period(chars, index):
            chars[index] = "。"
    return "".join(chars)


def _normalize_zh_quotes(text: str) -> str:
    quote_index = 0
    chars: list[str] = []
    for char in text:
        if char in QUOTE_CHARS_ZH:
            chars.append("“" if quote_index % 2 == 0 else "”")
            quote_index += 1
        else:
            chars.append(char)
    return "".join(chars)


def _append_final_punctuation_from_src(text: str, lang: str, src_text: str) -> str:
    mark = _source_final_mark(src_text, lang)
    if not mark or _has_terminal_strong_punctuation(text):
        return text

    base, suffix = _split_trailing_closers(text)
    return f"{base}{mark}{suffix}" if base else f"{text}{mark}"


def _source_final_mark(src_text: str, lang: str) -> str | None:
    del lang
    stripped = src_text.strip()
    while stripped and stripped[-1] in TRAILING_CLOSERS:
        stripped = stripped[:-1].rstrip()
    if not stripped:
        return None
    final = stripped[-1]
    if final == ".":
        return "。"
    if final == "?":
        return "？"
    if final == "!":
        return "！"
    return None


def _has_terminal_strong_punctuation(text: str) -> bool:
    base, _ = _split_trailing_closers(text.strip())
    return bool(base and base[-1] in STRONG_END_PUNCT)


def _split_trailing_closers(text: str) -> tuple[str, str]:
    index = len(text)
    while index > 0 and text[index - 1] in TRAILING_CLOSERS:
        index -= 1
    return text[:index], text[index:]


def _is_zh_context(chars: list[str], index: int) -> bool:
    left = _nearest_non_space(chars, index, -1)
    right = _nearest_non_space(chars, index, 1)
    return not (_is_ascii_alnum(left) and _is_ascii_alnum(right))


def _should_convert_zh_period(chars: list[str], index: int) -> bool:
    left = _nearest_non_space(chars, index, -1)
    right = _nearest_non_space(chars, index, 1)
    if _is_ascii_alnum(left) and _is_ascii_alnum(right):
        return False
    return _is_zh_context(chars, index)


def _nearest_non_space(chars: list[str], index: int, step: int) -> str:
    index += step
    while 0 <= index < len(chars):
        if chars[index] != " ":
            return chars[index]
        index += step
    return ""


def _is_ascii_alnum(char: str) -> bool:
    return len(char) == 1 and char.isascii() and char.isalnum()


def _is_quote_suspicious(text: str) -> bool:
    return (
        text.count("“") != text.count("”")
        or text.count("「") != text.count("」")
        or ("”" in text and "“" in text and text.find("”") < text.find("“"))
        or bool(re.search(r"[“「『\"]\s|\s[”」』\"]", text))
    )


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _iter_with_progress(
    iterable: Iterable[str],
    *,
    total: int,
    label: str,
    enabled: bool,
) -> Iterator[str]:
    if not enabled:
        yield from iterable
        return

    try:
        from tqdm.auto import tqdm
    except ImportError:
        for index, item in enumerate(iterable, start=1):
            if index == 1 or index % 10000 == 0 or index == total:
                print(f"{label}: {index}/{total}", file=sys.stderr)
            yield item
        return

    with tqdm(iterable, total=total, desc=label, unit="line") as progress:
        yield from progress


def _empty_stats(input_path: Path, output_path: Path) -> dict[str, Any]:
    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows": 0,
        "changed_rows": 0,
        "rows_with_spaces": 0,
        "rows_with_double_spaces": 0,
        "final_punctuation_added": 0,
        "quote_suspicious_rows": 0,
        "samples": [],
    }


def _update_stats(
    stats: dict[str, Any],
    *,
    original: str,
    normalized: str,
    src_text: str,
    sample_limit: int,
) -> None:
    stats["rows"] += 1
    if " " in original:
        stats["rows_with_spaces"] += 1
    if re.search(r" {2,}", original):
        stats["rows_with_double_spaces"] += 1
    if _is_quote_suspicious(original):
        stats["quote_suspicious_rows"] += 1
    if original != normalized:
        stats["changed_rows"] += 1
        if len(stats["samples"]) < sample_limit:
            stats["samples"].append(
                {
                    "src_text": src_text,
                    "before": original,
                    "after": normalized,
                }
            )
    if _final_punctuation_was_added(original, normalized, src_text):
        stats["final_punctuation_added"] += 1


def _final_punctuation_was_added(
    original: str, normalized: str, src_text: str
) -> bool:
    mark = _source_final_mark(src_text, "zh")
    if not mark or _has_terminal_strong_punctuation(original):
        return False
    normalized_base, _ = _split_trailing_closers(normalized.strip())
    return (
        bool(normalized_base)
        and normalized_base[-1] == mark
        and _has_terminal_strong_punctuation(normalized)
    )


def _print_summary(reports: dict[str, Any], *, dry_run: bool) -> None:
    mode = "DRY RUN" if dry_run else "WROTE"
    print(f"Normalization summary ({mode})")
    for lang_pair, report in reports.items():
        print(f"\n{lang_pair}:")
        for split, stats in report["splits"].items():
            print(
                "  "
                f"{split}: rows={stats['rows']} changed={stats['changed_rows']} "
                f"spaces={stats['rows_with_spaces']} double_spaces={stats['rows_with_double_spaces']} "
                f"final_punct_added={stats['final_punctuation_added']} "
                f"quote_suspicious={stats['quote_suspicious_rows']}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
