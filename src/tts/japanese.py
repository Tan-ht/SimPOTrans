from __future__ import annotations

import logging
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass


LOGGER = logging.getLogger(__name__)


class JapaneseTextPreprocessor:
    def __init__(self) -> None:
        try:
            import fugashi
        except ImportError as exc:
            raise RuntimeError(
                "Japanese TTS preprocessing requires fugashi and unidic-lite. "
                "Install project dependencies with uv sync."
            ) from exc
        self._tagger = fugashi.Tagger()

    def to_katakana(self, text: str) -> str:
        if not text:
            return text

        return _join_katakana_tokens(_normalize_token(token) for token in self._tokenize(text))

    def _tokenize(self, text: str) -> list["_JapaneseToken"]:
        tokens: list[_JapaneseToken] = []
        for word in self._tagger(text):
            surface = str(word)
            tokens.append(
                _JapaneseToken(
                    surface=surface,
                    reading=_reading_from_feature(word.feature),
                )
            )
        return tokens


@dataclass(frozen=True)
class _JapaneseToken:
    surface: str
    reading: str | None


@dataclass(frozen=True)
class _KatakanaToken:
    text: str
    kind: str


def _normalize_token(token: _JapaneseToken) -> _KatakanaToken | None:
    surface = token.surface
    if not surface:
        return None
    if surface.isspace():
        return _KatakanaToken(surface, "space")
    if _is_punctuation(surface):
        return _KatakanaToken(surface, "punct")

    reading = token.reading
    if not reading or reading == "*":
        LOGGER.warning("Falling back to Japanese TTS surface form: %s", surface)
        reading = surface

    text = unicodedata.normalize("NFKC", _hiragana_to_katakana(reading))
    return _KatakanaToken(text, "word")


def _join_katakana_tokens(tokens: Iterable[_KatakanaToken | None]) -> str:
    pieces: list[str] = []
    previous_was_word = False
    for token in tokens:
        if token is None:
            continue
        if token.kind == "space":
            if pieces and not pieces[-1].endswith(" "):
                pieces.append(" ")
            previous_was_word = False
            continue
        if token.kind == "punct":
            while pieces and pieces[-1] == " ":
                pieces.pop()
            pieces.append(token.text)
            previous_was_word = False
            continue
        if previous_was_word:
            pieces.append(" ")
        pieces.append(token.text)
        previous_was_word = True
    return "".join(pieces).strip()


def _reading_from_feature(feature) -> str | None:
    for name in ("pron", "pronBase", "kana", "orthBase"):
        value = getattr(feature, name, None)
        if value:
            return str(value)
    if isinstance(feature, (tuple, list)):
        for index in (9, 7, 8, 6):
            if len(feature) > index and feature[index] not in (None, "*"):
                return str(feature[index])
    return None


def _hiragana_to_katakana(text: str) -> str:
    converted = []
    for char in text:
        codepoint = ord(char)
        if 0x3041 <= codepoint <= 0x3096:
            converted.append(chr(codepoint + 0x60))
        else:
            converted.append(char)
    return "".join(converted)


def _is_punctuation(text: str) -> bool:
    return all(unicodedata.category(char).startswith(("P", "S")) for char in text)
