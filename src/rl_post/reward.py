from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


def _mean(values: list[float] | list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


class RewardScorer:
    name = "reward"

    def score_rollouts(self, rollouts: list[Any]) -> list[float]:
        return self.score_texts(
            [str(r.sample.get("source_text", "")) for r in rollouts],
            [r.hypothesis for r in rollouts],
            [str(r.sample.get("target_text", "")) for r in rollouts],
        )

    def score_texts(self, sources: list[str], hypotheses: list[str], references: list[str]) -> list[float]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def validation_scorer(self) -> "RewardScorer":
        return self


class ComponentScorer(Protocol):
    name: str

    def score_rollouts(self, rollouts: list[Any]) -> list[float]:
        ...

    def close(self) -> None:
        ...


def _set_reward_components(rollout: Any, components: dict[str, float]) -> None:
    if hasattr(rollout, "reward_components"):
        rollout.reward_components = dict(components)


def _lang_normalize(text: str, lang: str) -> str:
    text = text.strip()
    if lang in {"ja", "ko", "zh", "zh-cn", "zh-tw"}:
        return "".join(text.split())
    return " ".join(text.split())


class XCometRewardScorer(RewardScorer):
    name = "xcomet"

    def __init__(self, cfg: dict[str, Any]) -> None:
        reward_cfg = cfg.get("reward", {})
        self.python = Path(str(reward_cfg.get("comet_python", ".comet-venv/bin/python")))
        self.model_name = str(reward_cfg.get("comet_model", "Unbabel/XCOMET-XL"))
        self.batch_size = int(reward_cfg.get("comet_batch_size", 8))
        self.gpus = int(reward_cfg.get("comet_gpus", 1))
        self.request_size = int(reward_cfg.get("comet_request_size", 128))
        self.visible_devices = reward_cfg.get("comet_visible_devices")
        self._process: subprocess.Popen[str] | None = None
        self._stderr_tail: deque[str] = deque(maxlen=50)
        self._stderr_thread: threading.Thread | None = None
        self._start_process()
        atexit.register(self.close)

    def score_rollouts(self, rollouts: list[Any]) -> list[float]:
        scores = self.score_texts(
            [str(r.sample.get("source_text", "")) for r in rollouts],
            [r.hypothesis for r in rollouts],
            [str(r.sample.get("target_text", "")) for r in rollouts],
        )
        for rollout, score in zip(rollouts, scores, strict=True):
            _set_reward_components(rollout, {"xcomet": float(score), "total": float(score)})
        return scores

    def score_texts(self, sources: list[str], hypotheses: list[str], references: list[str]) -> list[float]:
        if not (len(sources) == len(hypotheses) == len(references)):
            raise ValueError("XCOMET sources, hypotheses, and references must have the same length.")
        if not hypotheses:
            return []

        data = [
            {"src": src, "mt": hyp, "ref": ref}
            for src, hyp, ref in zip(sources, hypotheses, references)
        ]
        if self.request_size <= 0 or len(data) <= self.request_size:
            return self._score_with_process(data)

        logger.info(
            "XCOMET scoring %d example(s) in request chunks of %d.",
            len(data),
            self.request_size,
        )
        scores: list[float] = []
        started_at = time.monotonic()
        for offset in range(0, len(data), self.request_size):
            chunk = data[offset : offset + self.request_size]
            chunk_started_at = time.monotonic()
            logger.info(
                "XCOMET scoring request %d-%d/%d.",
                offset + 1,
                offset + len(chunk),
                len(data),
            )
            scores.extend(self._score_with_process(chunk, log_mean=False))
            logger.info(
                "XCOMET scoring request %d-%d/%d done elapsed=%.1fs.",
                offset + 1,
                offset + len(chunk),
                len(data),
                time.monotonic() - chunk_started_at,
            )
        logger.info("XCOMET mean score: %.6f elapsed=%.1fs", _mean(scores), time.monotonic() - started_at)
        return scores

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        try:
            if process.stdin is not None:
                process.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                process.stdin.flush()
            process.wait(timeout=10)
        except Exception:
            process.terminate()

    def _start_process(self) -> None:
        python = self.python
        if not python.is_absolute():
            python = Path.cwd() / python
        if not python.is_file():
            raise RuntimeError(
                f"COMET Python interpreter not found: {python}. Run scripts/setup_comet_env.sh first, "
                "or set reward.comet_python to the .comet-venv Python path."
            )

        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", "/tmp")
        if self.visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.visible_devices)

        cmd = [
            str(python),
            "-u",
            "-c",
            _XCOMET_SERVER_SCRIPT,
            self.model_name,
            str(self.batch_size),
            str(self.gpus),
        ]
        logger.info(
            "Starting persistent XCOMET scorer: model=%s batch_size=%d gpus=%d visible_devices=%s.",
            self.model_name,
            self.batch_size,
            self.gpus,
            self.visible_devices if self.visible_devices is not None else "inherit",
        )
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,
        )
        self._start_stderr_drain()
        ready = self._read_response()
        if ready.get("status") != "ready":
            raise RuntimeError(f"XCOMET scorer did not start correctly: {ready}")
        logger.info("XCOMET scorer diagnostics: %s", ready.get("diagnostics", {}))

    def _start_stderr_drain(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return

        def drain_stderr() -> None:
            assert process.stderr is not None
            for line in process.stderr:
                text = line.rstrip()
                if text:
                    self._stderr_tail.append(text)
                    logger.debug("XCOMET stderr: %s", text)

        self._stderr_thread = threading.Thread(target=drain_stderr, name="xcomet-stderr", daemon=True)
        self._stderr_thread.start()

    def _score_with_process(self, data: list[dict[str, str]], *, log_mean: bool = True) -> list[float]:
        process = self._process
        if process is None or process.poll() is not None:
            raise RuntimeError("Persistent XCOMET scorer is not running.")
        if process.stdin is None:
            raise RuntimeError("Persistent XCOMET scorer stdin is closed.")

        process.stdin.write(json.dumps({"data": data}, ensure_ascii=False) + "\n")
        process.stdin.flush()
        response = self._read_response()
        if "error" in response:
            raise RuntimeError(f"XCOMET scorer failed: {response['error']}")
        scores = [float(score) for score in response["scores"]]
        if len(scores) != len(data):
            raise RuntimeError(f"XCOMET returned {len(scores)} score(s) for {len(data)} example(s).")
        if log_mean:
            logger.info("XCOMET mean score: %.6f", _mean(scores))
        return scores

    def _read_response(self) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("Persistent XCOMET scorer stdout is closed.")
        line = process.stdout.readline()
        if not line:
            stderr = "\n".join(self._stderr_tail)
            raise RuntimeError(f"Persistent XCOMET scorer exited unexpectedly. stderr:\n{stderr}")
        return json.loads(line)


class MonotonicityRewardScorer:
    name = "monotonicity"

    def __init__(self, cfg: dict[str, Any]) -> None:
        from analysis.score_monotonicity import (
            DEFAULT_AWESOME_ALIGN_COMMAND,
            _load_spacy_tokenizer,
            _run_awesome_align,
            spearman_from_alignments,
            tokenize,
        )

        reward_cfg = cfg.get("reward", {})
        self.invalid_value = float(reward_cfg.get("monotonicity_invalid_value", 0.5))
        self.command = str(reward_cfg.get("awesome_align_command", DEFAULT_AWESOME_ALIGN_COMMAND))
        self._load_spacy_tokenizer = _load_spacy_tokenizer
        self._run_awesome_align = _run_awesome_align
        self._spearman_from_alignments = spearman_from_alignments
        self._tokenize = tokenize
        self._tokenizers: dict[str, Any] = {}

    def score_rollouts(self, rollouts: list[Any]) -> list[float]:
        scores = [self.invalid_value] * len(rollouts)
        grouped: dict[tuple[str, str], list[int]] = {}
        for index, rollout in enumerate(rollouts):
            src_lang = str(rollout.sample.get("src_lang", "xx")).lower()
            tgt_lang = str(rollout.sample.get("tgt_lang", "xx")).lower()
            grouped.setdefault((src_lang, tgt_lang), []).append(index)

        for (src_lang, tgt_lang), indices in grouped.items():
            src_nlp = self._tokenizer(src_lang)
            tgt_nlp = self._tokenizer(tgt_lang)
            pairs: list[tuple[list[str], list[str]]] = []
            pair_indices: list[int] = []
            for index in indices:
                rollout = rollouts[index]
                source_tokens = self._tokenize(str(rollout.sample.get("source_text", "")), src_nlp)
                pred_tokens = self._tokenize(str(rollout.hypothesis), tgt_nlp)
                if len(source_tokens) < 2 or len(pred_tokens) < 2:
                    continue
                pairs.append((source_tokens, pred_tokens))
                pair_indices.append(index)

            if not pairs:
                continue
            try:
                alignments = self._run_awesome_align(self.command, pairs)
            except RuntimeError as exc:
                logger.warning(
                    "Monotonicity Awesome Align failed for %d pair(s) in %s-%s; retrying individually. error=%s",
                    len(pairs),
                    src_lang,
                    tgt_lang,
                    exc,
                )
                alignments = []
                for pair_index, pair in zip(pair_indices, pairs, strict=True):
                    try:
                        pair_alignments = self._run_awesome_align(self.command, [pair])
                    except RuntimeError as pair_exc:
                        logger.warning(
                            "Monotonicity Awesome Align failed for rollout %d; using invalid score %.6f. error=%s",
                            pair_index,
                            self.invalid_value,
                            pair_exc,
                        )
                        alignments.append([])
                    else:
                        alignments.append(pair_alignments[0])
            for index, alignment in zip(pair_indices, alignments, strict=True):
                raw_score = self._spearman_from_alignments(alignment)
                scores[index] = self.invalid_value if raw_score is None else (float(raw_score) + 1.0) / 2.0

        logger.info("Monotonicity mean score: %.6f", _mean(scores))
        return scores

    def close(self) -> None:
        pass

    def _tokenizer(self, lang: str) -> Any:
        if lang not in self._tokenizers:
            self._tokenizers[lang] = self._load_spacy_tokenizer(lang)
        return self._tokenizers[lang]


class NGramRepetitionRewardScorer:
    name = "repetition"

    def __init__(self, cfg: dict[str, Any]) -> None:
        reward_cfg = cfg.get("reward", {})
        self.threshold = float(reward_cfg.get("repetition_threshold", 0.65))

    def score_rollouts(self, rollouts: list[Any]) -> list[float]:
        scores: list[float] = []
        for rollout in rollouts:
            tgt_lang = str(rollout.sample.get("tgt_lang", "")).lower()
            chunk_texts = [str(text) for text in getattr(rollout, "chunk_texts", [])]
            penalties: list[float] = []
            for previous, current in zip(chunk_texts, chunk_texts[1:]):
                current_ngrams = self._ngrams(current, tgt_lang)
                if not current_ngrams:
                    continue
                previous_ngrams = self._ngrams(previous, tgt_lang)
                if not previous_ngrams:
                    continue
                repeat_rate = len(current_ngrams & previous_ngrams) / len(current_ngrams)
                penalties.append(max(0.0, repeat_rate - self.threshold))
            scores.append(_mean(penalties))

        logger.info("N-gram repetition mean score: %.6f", _mean(scores))
        return scores

    def close(self) -> None:
        pass

    def _ngrams(self, text: str, lang: str) -> set[tuple[str, ...] | str]:
        if lang in {"zh", "ja"}:
            chars = _lang_normalize(text, lang)
            return self._char_ngrams(chars, 4)
        if lang in {"de", "en"}:
            words = _lang_normalize(text, lang).split()
            return self._word_ngrams(words, 3)
        raise ValueError(f"Unsupported target language for n-gram repetition reward: {lang}")

    @staticmethod
    def _char_ngrams(text: str, n: int) -> set[str]:
        if len(text) < n:
            return set()
        return {text[index : index + n] for index in range(len(text) - n + 1)}

    @staticmethod
    def _word_ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
        if len(words) < n:
            return set()
        return {tuple(words[index : index + n]) for index in range(len(words) - n + 1)}


class EosBonusRewardScorer:
    name = "eos_bonus"

    def score_rollouts(self, rollouts: list[Any]) -> list[float]:
        scores: list[float] = []
        for rollout in rollouts:
            if not hasattr(rollout, "chunks"):
                scores.append(0.0)
                continue
            chunk_texts = list(getattr(rollout, "chunk_texts", []))
            if not chunk_texts:
                scores.append(0.0)
                continue
            empty_count = sum(1 for text in chunk_texts if not str(text).strip())
            scores.append(float(empty_count / len(chunk_texts)))
        logger.info("EOS bonus mean score: %.6f", _mean(scores))
        return scores

    def close(self) -> None:
        pass


class ConstantComponentScorer:
    def __init__(self, name: str, value: float = 0.0) -> None:
        self.name = name
        self.value = float(value)

    def score_rollouts(self, rollouts: list[Any]) -> list[float]:
        return [self.value] * len(rollouts)

    def close(self) -> None:
        pass


class CompositeRewardScorer(RewardScorer):
    name = "composite"

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        comet_scorer: RewardScorer | None = None,
        monotonicity_scorer: ComponentScorer | None = None,
        repetition_scorer: ComponentScorer | None = None,
        eos_scorer: ComponentScorer | None = None,
    ) -> None:
        reward_cfg = cfg.get("reward", {})
        self.comet_weight = float(reward_cfg.get("comet_weight", 1.0))
        self.monotonicity_coef = float(reward_cfg.get("monotonicity_coef", 0.2))
        self.monotonicity_threshold = float(reward_cfg.get("monotonicity_threshold", 0.9))
        self.repetition_coef = float(reward_cfg.get("repetition_coef", 0.05))
        self.eos_bonus_coef = float(reward_cfg.get("eos_bonus_coef", 0.05))

        self.comet = comet_scorer or XCometRewardScorer(cfg)
        self.monotonicity = (
            monotonicity_scorer
            if monotonicity_scorer is not None
            else MonotonicityRewardScorer(cfg)
            if bool(reward_cfg.get("monotonicity_enabled", True))
            else ConstantComponentScorer("monotonicity", 0.0)
        )
        self.repetition = (
            repetition_scorer
            if repetition_scorer is not None
            else NGramRepetitionRewardScorer(cfg)
            if bool(reward_cfg.get("repetition_enabled", True))
            else ConstantComponentScorer("repetition", 0.0)
        )
        self.eos_bonus = (
            eos_scorer
            if eos_scorer is not None
            else EosBonusRewardScorer()
            if bool(reward_cfg.get("eos_bonus_enabled", True))
            else ConstantComponentScorer("eos_bonus", 0.0)
        )

    def score_rollouts(self, rollouts: list[Any]) -> list[float]:
        if not rollouts:
            return []

        comet_scores = self.comet.score_texts(
            [str(r.sample.get("source_text", "")) for r in rollouts],
            [r.hypothesis for r in rollouts],
            [str(r.sample.get("target_text", "")) for r in rollouts],
        )
        monotonicity_scores = self.monotonicity.score_rollouts(rollouts)
        repetition_scores = self.repetition.score_rollouts(rollouts)
        eos_bonus_scores = self.eos_bonus.score_rollouts(rollouts)

        totals: list[float] = []
        for rollout, comet, monotonicity, repetition, eos_bonus in zip(
            rollouts,
            comet_scores,
            monotonicity_scores,
            repetition_scores,
            eos_bonus_scores,
            strict=True,
        ):
            monotonicity_penalty = max(0.0, self.monotonicity_threshold - float(monotonicity))
            total = (
                self.comet_weight * float(comet)
                - self.monotonicity_coef * monotonicity_penalty
                - self.repetition_coef * float(repetition)
                + self.eos_bonus_coef * float(eos_bonus)
            )
            components = {
                "comet": float(comet),
                "monotonicity": float(monotonicity),
                "monotonicity_penalty": float(monotonicity_penalty),
                "repetition": float(repetition),
                "eos_bonus": float(eos_bonus),
                "total": float(total),
            }
            _set_reward_components(rollout, components)
            totals.append(float(total))

        logger.info(
            "Composite reward mean: total=%.6f comet=%.6f monotonicity=%.6f monotonicity_penalty=%.6f repetition=%.6f eos_bonus=%.6f",
            _mean(totals),
            _mean([float(score) for score in comet_scores]),
            _mean([float(score) for score in monotonicity_scores]),
            _mean([
                max(0.0, self.monotonicity_threshold - float(score))
                for score in monotonicity_scores
            ]),
            _mean([float(score) for score in repetition_scores]),
            _mean([float(score) for score in eos_bonus_scores]),
        )
        return totals

    def score_texts(self, sources: list[str], hypotheses: list[str], references: list[str]) -> list[float]:
        return self.comet.score_texts(sources, hypotheses, references)

    def validation_scorer(self) -> RewardScorer:
        return self.comet

    def close(self) -> None:
        for scorer in (self.comet, self.monotonicity, self.repetition, self.eos_bonus):
            scorer.close()


_XCOMET_SERVER_SCRIPT = r'''
import json
import os
import sys
import traceback
import torch
from comet import download_model, load_from_checkpoint


def to_float(value):
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


model_name, batch_size, gpus = sys.argv[1:]
model_path = download_model(model_name)
model = load_from_checkpoint(model_path)
diagnostics = {
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
    "cuda_device_name": torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else None,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "model_device": str(next(model.parameters()).device) if hasattr(model, "parameters") else None,
}
print(json.dumps({"status": "ready", "diagnostics": diagnostics}), flush=True)

for line in sys.stdin:
    try:
        request = json.loads(line)
        if request.get("cmd") == "shutdown":
            break
        data = request["data"]
        output = model.predict(data, batch_size=int(batch_size), gpus=int(gpus))
        scores = getattr(output, "scores", None)
        if scores is None:
            scores = [getattr(output, "system_score")] * len(data)
        print(json.dumps({"scores": [to_float(score) for score in scores]}), flush=True)
    except Exception:
        print(json.dumps({"error": traceback.format_exc()}), flush=True)
'''
