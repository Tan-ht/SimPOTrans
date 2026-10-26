from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from importlib.util import find_spec
from dataclasses import replace
from pathlib import Path
from typing import Any
from typing import final

from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from fairseq2.composition import register_dataset_family, register_model_family
from fairseq2.datasets import DatasetFamily
from fairseq2.early_stopper import EarlyStopper
from fairseq2.evaluator import EvalUnit
from fairseq2.metrics import Mean, MetricBag, Sum, format_as_float
from fairseq2.metrics.recorders import MetricDescriptor
from fairseq2.models import ModelFamily
from fairseq2.recipe.base import Recipe, RecipeContext
from fairseq2.runtime.dependency import DependencyContainer
from fairseq2.task import Task
from fairseq2.trainer import TrainUnit

from .config import ST_DATASET_FAMILY, STTrainConfig
from .dataset import STBatch, STTrainDataset, STTrainDatasetConfig, open_st_train_dataset
from .model import (
    ST_MODEL_FAMILY,
    STModel,
    STModelConfig,
    _configure_trainable_parameters,
    create_st_model,
    register_st_model_configs,
)

logger = logging.getLogger(__name__)


@final
class STTrainRecipe(Recipe):
    """fairseq2 training recipe for the SimPOTrans speech-to-text model."""

    @override
    def register(self, container: DependencyContainer) -> None:
        register_st_components(container)

    @override
    def setup_model(
        self, context: RecipeContext, model: Module, newly_initialized: bool
    ) -> Module:
        if not isinstance(model, STModel):
            raise TypeError(
                f"STTrainRecipe expects an STModel, but got {type(model).__name__}."
            )

        _configure_trainable_parameters(model)
        model.to(context.device)

        return model

    @override
    def create_task(self, context: RecipeContext) -> Task:
        config = context.get_config_as(STTrainConfig)
        recipe_config = config.recipe

        model = context.get_data_parallel_model()
        dataset = context.get_dataset_as(STTrainDataset)

        unit = STTrainUnit(model)
        dataset.log_summary("training")
        data_reader = dataset.create_reader(
            context.gangs,
            is_train=True,
            seed=recipe_config.seed,
            num_accumulate=recipe_config.num_accumulate,
        )

        valid_units = None
        valid_readers = None
        if recipe_config.valid_splits:
            valid_dataset = STTrainDataset(
                replace(
                    dataset.config,
                    splits=recipe_config.valid_splits,
                    shuffle_window=0,
                    prefetch=0,
                    max_sentences_per_lang_pair=(
                        recipe_config.max_validation_sentences_per_lang_pair
                    ),
                )
            )
            valid_dataset.log_summary("validation")
            valid_units = [
                STValidationUnit(
                    model,
                    max_bleu_batches=recipe_config.max_bleu_batches,
                    max_bleu_sentences_per_lang_pair=(
                        recipe_config.max_bleu_sentences_per_lang_pair
                    ),
                    compute_comet=recipe_config.compute_comet,
                    comet_model=recipe_config.comet_model,
                    comet_batch_size=recipe_config.comet_batch_size,
                    comet_gpus=recipe_config.comet_gpus,
                    comet_python=recipe_config.comet_python,
                    log_example_batches=recipe_config.log_example_batches,
                )
            ]
            valid_readers = [
                valid_dataset.create_reader(
                    context.gangs,
                    is_train=False,
                    seed=recipe_config.seed,
                    num_accumulate=1,
                )
            ]

        trainer = context.create_trainer(unit, data_reader, valid_units, valid_readers)

        if recipe_config.early_stop_patience is not None:
            if not recipe_config.valid_splits:
                raise ValueError("Early stopping requires `valid_splits`.")

            trainer._early_stopper = PatienceEarlyStopper(
                patience=recipe_config.early_stop_patience,
                min_delta=recipe_config.early_stop_min_delta,
                higher_better=recipe_config.early_stop_higher_better,
            )

        return trainer

    @property
    @override
    def config_kls(self) -> type[object]:
        return STTrainConfig


@final
class STTrainUnit(TrainUnit[STBatch]):
    """Thin train unit: delegate loss computation to STModel."""

    def __init__(self, model: Module) -> None:
        self._model = model

    @override
    def prepare_metric_bag(self, metric_bag: MetricBag) -> None:
        metric_bag.add("loss", Mean())
        metric_bag.add("num_sentences", Sum())

    @override
    def process_batch(
        self, batch: STBatch, metric_bag: MetricBag
    ) -> tuple[Tensor, int | None]:
        output = self._model(batch)
        loss = output.get("loss", output.get("st_loss"))

        if loss is None:
            raise RuntimeError("STModel must return `loss` or `st_loss`.")

        num_sentences = max(len(batch.sample_ids), 1)

        metric_bag.get("loss", Mean).update(loss.detach(), weight=num_sentences)
        metric_bag.get("num_sentences", Sum).update(num_sentences)

        return loss, None


@final
class STValidationUnit(EvalUnit[STBatch]):
    """Validation unit that reports loss and BLEU for ST."""

    def __init__(
        self,
        model: Module,
        *,
        max_bleu_batches: int | None = None,
        max_bleu_sentences_per_lang_pair: int | None = 100,
        compute_comet: bool = False,
        comet_model: str = "Unbabel/XCOMET-XL",
        comet_batch_size: int = 2,
        comet_gpus: int = 1,
        comet_python: str | None = ".comet-venv/bin/python",
        log_example_batches: int = 1,
    ) -> None:
        self._model = model
        self._max_bleu_batches = max_bleu_batches
        self._max_bleu_sentences_per_lang_pair = max_bleu_sentences_per_lang_pair
        self._compute_comet_enabled = compute_comet
        self._comet_model_name = comet_model
        self._comet_batch_size = comet_batch_size
        self._comet_gpus = comet_gpus
        self._comet_python = comet_python
        self._log_example_batches = log_example_batches
        self._num_batches = 0
        self._srcs: list[str] = []
        self._hyps: list[str] = []
        self._refs: list[str] = []
        self._src_langs: list[str] = []
        self._tgt_langs: list[str] = []
        self._bleu_lang_pair_counts: dict[tuple[str, str], int] = {}
        self._comet_model: Any | None = None

    @override
    def prepare_metric_bag(self, metric_bag: MetricBag) -> None:
        self._num_batches = 0
        self._srcs.clear()
        self._hyps.clear()
        self._refs.clear()
        self._src_langs.clear()
        self._tgt_langs.clear()
        self._bleu_lang_pair_counts.clear()

        metric_bag.add("loss", Mean())
        metric_bag.add("num_sentences", Sum())
        metric_bag.add("bleu", Mean())
        metric_bag.add("bleu_sentences", Sum())
        if self._compute_comet_enabled:
            metric_bag.add("comet", Mean())
            metric_bag.add("comet_sentences", Sum())

    @override
    def process_batch(self, batch: STBatch, metric_bag: MetricBag) -> None:
        output = self._model(batch)
        loss = output.get("loss", output.get("st_loss"))

        if loss is None:
            raise RuntimeError("STModel must return `loss` or `st_loss`.")

        num_sentences = max(len(batch.sample_ids), 1)
        metric_bag.get("loss", Mean).update(loss.detach(), weight=num_sentences)
        metric_bag.get("num_sentences", Sum).update(num_sentences)

        if self._should_process_bleu_batch():
            bleu_indices = self._select_bleu_indices(batch)
            if not bleu_indices:
                self._num_batches += 1
                return

            model = _unwrap_model(self._model)
            tokens = model.generate(
                [batch.source[index] for index in bleu_indices],
                src_langs=[batch.src_langs[index] for index in bleu_indices],
                tgt_langs=[batch.tgt_langs[index] for index in bleu_indices],
            )
            hyps = [hyp.strip() for hyp in model.decode(tokens)]
            refs = [batch.target_texts[index].strip() for index in bleu_indices]
            srcs = [batch.source_texts[index].strip() for index in bleu_indices]
            src_langs = [batch.src_langs[index] for index in bleu_indices]
            tgt_langs = [batch.tgt_langs[index] for index in bleu_indices]

            self._srcs.extend(srcs)
            self._hyps.extend(hyps)
            self._refs.extend(refs)
            self._src_langs.extend(src_langs)
            self._tgt_langs.extend(tgt_langs)

            if self._num_batches < self._log_example_batches and hyps:
                logger.info("validation source: %s", srcs[0])
                logger.info("validation hypothesis: %s", hyps[0])
                logger.info("validation reference: %s", refs[0])

        self._num_batches += 1

    def _select_bleu_indices(self, batch: STBatch) -> list[int]:
        indices: list[int] = []

        for index, (src_lang, tgt_lang) in enumerate(
            zip(batch.src_langs, batch.tgt_langs)
        ):
            lang_pair = (src_lang, tgt_lang)
            limit = self._max_bleu_sentences_per_lang_pair
            if (
                limit is not None
                and self._bleu_lang_pair_counts.get(lang_pair, 0) >= limit
            ):
                continue

            indices.append(index)
            self._bleu_lang_pair_counts[lang_pair] = (
                self._bleu_lang_pair_counts.get(lang_pair, 0) + 1
            )

        return indices

    def _should_process_bleu_batch(self) -> bool:
        if self._max_bleu_batches is None:
            return True

        return self._num_batches < self._max_bleu_batches

    @override
    def finalize(self, metric_bag: MetricBag) -> None:
        if not self._hyps:
            return

        logger.info(
            "validation BLEU computed on %d sentence(s), capped at %s sentence(s) "
            "per language pair.",
            len(self._hyps),
            self._max_bleu_sentences_per_lang_pair or "unlimited",
        )

        bleu = _compute_multilingual_bleu(
            self._hyps,
            self._refs,
            self._src_langs,
            self._tgt_langs,
        )
        metric_bag.get("bleu", Mean).update(bleu, weight=len(self._hyps))
        metric_bag.get("bleu_sentences", Sum).update(len(self._hyps))

        if self._compute_comet_enabled:
            comet = self._compute_comet()
            metric_bag.get("comet", Mean).update(comet, weight=len(self._hyps))
            metric_bag.get("comet_sentences", Sum).update(len(self._hyps))

    @override
    @property
    def name(self) -> str | None:
        return "st"

    def _compute_comet(self) -> float:
        if len(self._srcs) != len(self._hyps) or len(self._hyps) != len(self._refs):
            raise ValueError("COMET sources, hypotheses, and references must match.")

        data = [
            {"src": src, "mt": hyp, "ref": ref}
            for src, hyp, ref in zip(self._srcs, self._hyps, self._refs)
        ]
        if self._comet_python:
            logger.info(
                "Computing COMET with %s on %d validation example(s).",
                self._comet_python,
                len(data),
            )
            return _compute_comet_with_python(
                Path(self._comet_python),
                data,
                model_name=self._comet_model_name,
                batch_size=self._comet_batch_size,
                gpus=self._comet_gpus,
            )

        if self._comet_model is None:
            if find_spec("comet") is None:
                logger.info(
                    "Computing COMET with isolated uv environment on %d validation example(s).",
                    len(data),
                )
                return _compute_comet_with_uv(
                    data,
                    model_name=self._comet_model_name,
                    batch_size=self._comet_batch_size,
                    gpus=self._comet_gpus,
                )
            else:
                logger.info("Loading COMET model %s.", self._comet_model_name)
                self._comet_model = _load_comet_model(self._comet_model_name)

        logger.info(
            "Computing COMET with in-process model on %d validation example(s).",
            len(data),
        )
        output = self._comet_model.predict(
            data,
            batch_size=self._comet_batch_size,
            gpus=self._comet_gpus,
        )
        score = float(output.system_score)
        logger.info("COMET score: %.6f", score)

        return score


@final
class PatienceEarlyStopper(EarlyStopper):
    """Stop when validation score has not improved for N validations."""

    def __init__(
        self,
        *,
        patience: int,
        min_delta: float = 0.0,
        higher_better: bool = True,
    ) -> None:
        if patience < 1:
            raise ValueError("`patience` must be greater than or equal to 1.")

        self._patience = patience
        self._min_delta = min_delta
        self._higher_better = higher_better
        self._best_score: float | None = None
        self._best_step_nr = 0
        self._num_bad_validations = 0

    @override
    def should_stop(self, step_nr: int, score: float) -> bool:
        if self._is_improved(score):
            self._best_score = score
            self._best_step_nr = step_nr
            self._num_bad_validations = 0

            logger.info("New best validation score %.4f at step %d.", score, step_nr)

            return False

        self._num_bad_validations += 1

        logger.info(
            "Validation score %.4f did not improve from %.4f for %d/%d validation(s).",
            score,
            self._best_score,
            self._num_bad_validations,
            self._patience,
        )

        return self._num_bad_validations >= self._patience

    def _is_improved(self, score: float) -> bool:
        if self._best_score is None:
            return True

        if self._higher_better:
            return score > self._best_score + self._min_delta

        return score < self._best_score - self._min_delta


def register_st_components(container: DependencyContainer) -> None:
    _register_st_metric_descriptors(container)

    if ST_DATASET_FAMILY not in set(container.iter_keys(DatasetFamily)):
        register_dataset_family(
            container,
            ST_DATASET_FAMILY,
            STTrainDataset,
            STTrainDatasetConfig,
            opener=open_st_train_dataset,
        )

    if ST_MODEL_FAMILY not in set(container.iter_keys(ModelFamily)):
        register_model_family(
            container,
            ST_MODEL_FAMILY,
            STModel,
            STModelConfig,
            factory=create_st_model,
            supports_meta=False,
        )

    register_st_model_configs(container)


def _register_st_metric_descriptors(container: DependencyContainer) -> None:
    descriptor = MetricDescriptor(
        "comet",
        "COMET",
        200,
        format_as_float,
        higher_better=True,
    )
    container.collection.register_instance(MetricDescriptor, descriptor)


def _unwrap_model(model: Module) -> Any:
    return getattr(model, "module", model)


def _compute_bleu(hyps: list[str], refs: list[str], lang: str | None) -> float:
    if len(hyps) != len(refs):
        raise ValueError("BLEU hypotheses and references must have the same length.")

    import sacrebleu

    if lang == "ja":
        bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="ja-mecab")
    elif lang == "zh":
        bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh")
    else:
        bleu = sacrebleu.corpus_bleu(hyps, [refs])

    return float(bleu.score)


def _compute_multilingual_bleu(
    hyps: list[str],
    refs: list[str],
    src_langs: list[str],
    tgt_langs: list[str],
) -> float:
    if len(hyps) != len(refs):
        raise ValueError("BLEU hypotheses and references must have the same length.")

    if len(hyps) != len(src_langs) or len(hyps) != len(tgt_langs):
        raise ValueError("BLEU language labels must match hypotheses and references.")

    groups: dict[tuple[str, str], tuple[list[str], list[str]]] = {}
    for hyp, ref, src_lang, tgt_lang in zip(hyps, refs, src_langs, tgt_langs):
        group_hyps, group_refs = groups.setdefault((src_lang, tgt_lang), ([], []))
        group_hyps.append(hyp)
        group_refs.append(ref)

    weighted_bleu = 0.0
    total = 0
    for (src_lang, tgt_lang), (group_hyps, group_refs) in sorted(groups.items()):
        bleu = _compute_bleu(group_hyps, group_refs, tgt_lang)
        weight = len(group_hyps)
        weighted_bleu += bleu * weight
        total += weight
        logger.info(
            "validation BLEU %s-%s: %.4f (%d sentence(s))",
            src_lang or "unknown",
            tgt_lang or "unknown",
            bleu,
            weight,
        )

    return weighted_bleu / max(total, 1)


def _load_comet_model(model_name: str) -> Any:
    from comet import download_model, load_from_checkpoint

    model_path = download_model(model_name)

    return load_from_checkpoint(model_path)


_COMET_SCORE_SCRIPT = """
import json
import sys
from comet import download_model, load_from_checkpoint

data_path, model_name, batch_size, gpus = sys.argv[1:]
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

model_path = download_model(model_name)
model = load_from_checkpoint(model_path)
output = model.predict(data, batch_size=int(batch_size), gpus=int(gpus))
print(json.dumps({"system_score": float(output.system_score)}))
"""


def _compute_comet_with_python(
    python: Path,
    data: list[dict[str, str]],
    *,
    model_name: str,
    batch_size: int,
    gpus: int,
) -> float:
    if not python.is_absolute():
        python = Path.cwd() / python

    logger.info(
        "Using COMET Python %s with model=%s, batch_size=%d, gpus=%d.",
        python,
        model_name,
        batch_size,
        gpus,
    )

    if not python.is_file():
        raise RuntimeError(
            f"COMET Python interpreter not found: {python}. Create it with "
            "`uv venv .comet-venv --python 3.10` and "
            "`.comet-venv/bin/python -m pip install unbabel-comet`."
        )

    return _run_comet_subprocess(
        [str(python), "-c", _COMET_SCORE_SCRIPT],
        data,
        model_name=model_name,
        batch_size=batch_size,
        gpus=gpus,
    )


def _compute_comet_with_uv(
    data: list[dict[str, str]],
    *,
    model_name: str,
    batch_size: int,
    gpus: int,
) -> float:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "COMET validation requires either importable `unbabel-comet` or `uv` "
            "on PATH to run it in an isolated environment."
        )

    logger.info(
        "Using uv COMET subprocess with model=%s, batch_size=%d, gpus=%d.",
        model_name,
        batch_size,
        gpus,
    )

    return _run_comet_subprocess(
        [
            uv,
            "run",
            "--no-project",
            "--with",
            "unbabel-comet>=2.2.0",
            "python",
            "-c",
            _COMET_SCORE_SCRIPT,
        ],
        data,
        model_name=model_name,
        batch_size=batch_size,
        gpus=gpus,
    )


def _run_comet_subprocess(
    cmd: list[str],
    data: list[dict[str, str]],
    *,
    model_name: str,
    batch_size: int,
    gpus: int,
) -> float:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json") as f:
        json.dump(data, f, ensure_ascii=False)
        f.flush()

        full_cmd = [*cmd, f.name, model_name, str(batch_size), str(gpus)]
        logger.info("Starting COMET subprocess for %d example(s).", len(data))
        try:
            env = os.environ.copy()
            env.setdefault("MPLCONFIGDIR", "/tmp")
            result = subprocess.run(
                full_cmd,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
        except subprocess.CalledProcessError as ex:
            raise RuntimeError(
                "COMET scoring subprocess failed.\n"
                f"Command: {' '.join(full_cmd)}\n"
                f"stdout:\n{ex.stdout}\n"
                f"stderr:\n{ex.stderr}"
            ) from ex

    if result.stderr.strip():
        logger.info("COMET subprocess stderr:\n%s", result.stderr.strip())

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(
            "COMET subprocess did not produce a score.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    if len(lines) > 1:
        logger.info("COMET subprocess stdout:\n%s", "\n".join(lines[:-1]))

    score = float(json.loads(lines[-1])["system_score"])
    logger.info("COMET score: %.6f", score)

    return score
