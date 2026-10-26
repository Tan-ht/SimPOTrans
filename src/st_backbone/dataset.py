from __future__ import annotations

import csv
import io
import json
import logging
import mmap
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, final

import torch
from torch import Tensor

from fairseq2.data.data_pipeline import read_sequence
from fairseq2.datasets import DataPipelineReader, DataReader
from fairseq2.device import Device, SupportsDeviceTransfer
from fairseq2.gang import Gangs

logger = logging.getLogger(__name__)


@dataclass
class STBatch(SupportsDeviceTransfer):
    """A speech-to-text batch consumed by the training unit."""
    ids: Tensor
    sample_ids: list[str]
    source: list[Tensor]
    source_lens: Tensor
    target_audio: list[Tensor | None] | None
    source_texts: list[str]
    target_texts: list[str]
    src_langs: list[str]
    tgt_langs: list[str]
    speakers: list[str]

    def to(self, device: Device, *, non_blocking: bool = False) -> None:
        self.ids = self.ids.to(device, non_blocking=non_blocking)
        self.source_lens = self.source_lens.to(device, non_blocking=non_blocking)
        self.source = [x.to(device, non_blocking=non_blocking) for x in self.source]

        if self.target_audio is not None:
            self.target_audio = [
                None if x is None else x.to(device, non_blocking=non_blocking)
                for x in self.target_audio
            ]


@dataclass(kw_only=True)
class STTrainDatasetConfig:
    """Configuration for manifest-based speech-to-text training data."""
    manifest_root: Path = field(default_factory=Path)
    audio_root: Path | None = None
    manifest_list: Path | None = None
    manifest_format: str = "auto"
    splits: str = "train"
    default_train_split: str = "train_st"
    default_valid_split: str = "dev_st"
    sampling_alpha: float = 1.0
    sampling_unit: str = "lang_pair"
    epoch_size: int | None = None
    max_sentences_per_lang_pair: int | None = None
    sample_rate: int = 16_000
    filter_min_len: float = 0.0
    max_source_frames: int | None = None
    max_num_frames_per_batch: int | None = None
    shuffle_window: int = 10_000
    batch_size: int = 8
    prefetch: int = 4


@final
class STTrainDataset:
    # mandatory columns
    KEY_ID, KEY_AUDIO, KEY_N_FRAMES = "id", "audio", "n_frames"
    KEY_TGT_TEXT = "tgt_text"

    # optional columns
    KEY_SPEAKER, KEY_SRC_TEXT = "speaker", "src_text"
    KEY_SRC_LANG, KEY_TGT_LANG = "src_lang", "tgt_lang"
    KEY_TGT_AUDIO = "target_audio"

    DEFAULT_SPEAKER = DEFAULT_SRC_TEXT = DEFAULT_LANG = ""

    def __init__(self, config: STTrainDatasetConfig) -> None:
        self.config = config
        self._samples = self._load_samples()
        self._sample_groups = self._group_samples(self._samples)

    def create_reader(
        self,
        gangs: Gangs,
        *,
        is_train: bool = True,
        seed: int | None = None,
        num_accumulate: int = 1,
    ) -> DataReader[STBatch]:
        """Create a fairseq2 data reader for distributed training/evaluation."""
        samples = self._get_reader_samples(is_train=is_train, seed=seed)
        global_num_samples = len(samples)

        if gangs.dp.size > 1:
            rank_samples = _shard_samples(samples, gangs.dp.rank, gangs.dp.size)
        else:
            rank_samples = samples

        if gangs.dp.rank == 0:
            self._log_reader_estimate(
                "training" if is_train else "validation",
                samples=rank_samples,
                global_num_samples=global_num_samples,
                rank_num_samples=len(rank_samples),
                num_accumulate=num_accumulate,
                dp_size=gangs.dp.size,
            )

        builder = read_sequence(samples)

        if gangs.dp.size > 1:
            builder.shard(gangs.dp.rank, gangs.dp.size, allow_uneven=True)

        if is_train and self.config.shuffle_window > 0:
            builder.shuffle(self.config.shuffle_window, seed=seed)

        builder.map(self._prepare_example)
        if self.config.max_num_frames_per_batch is not None:
            builder.dynamic_bucket(
                self.config.max_num_frames_per_batch,
                fn=lambda example: example["source_len"],
                min_num_examples=1,
                max_num_examples=self.config.batch_size,
                drop_remainder=is_train,
            )
        else:
            builder.bucket(self.config.batch_size, drop_remainder=is_train)
        builder.map(self._to_batch)

        if self.config.prefetch > 0:
            builder.prefetch(self.config.prefetch)

        return DataPipelineReader(
            builder.and_return(),
            gangs,
            num_accumulate=num_accumulate,
            drop_remainder=is_train,
        )

    def log_summary(self, name: str) -> None:
        """Log a compact dataset summary before training or validation starts."""
        logger.info(
            "%s dataset summary: %d sample(s), split(s)=%s, batch_size=%d, "
            "max_num_frames_per_batch=%s, max_sentences_per_lang_pair=%s.",
            name,
            len(self._samples),
            self.config.splits,
            self.config.batch_size,
            self.config.max_num_frames_per_batch,
            self.config.max_sentences_per_lang_pair or "unlimited",
        )
        logger.info(
            "%s dataset filters: min_len=%.2fs, max_source_frames=%s.",
            name,
            self.config.filter_min_len,
            self.config.max_source_frames,
        )

        if len(self._sample_groups) > 1:
            logger.info(
                "%s dataset sampling: unit=%s, alpha=%.3f, epoch_size=%s.",
                name,
                self.config.sampling_unit,
                self.config.sampling_alpha,
                self.config.epoch_size or "raw_size",
            )

        self._log_sample_distribution(name, self._samples)

    def _load_samples(self) -> list[dict[str, str]]:
        samples: list[dict[str, str]] = []

        for source in self._manifest_sources:
            source_samples: list[dict[str, str]] = []

            for split in self._split_names:
                manifest_path = self._resolve_split_path(source, split)

                if manifest_path is None:
                    raise FileNotFoundError(
                        "Dataset manifest not found under "
                        f"{source['manifest_root']} for split `{split}`."
                    )

                split_samples = self._load_manifest(manifest_path, source)

                for sample in split_samples:
                    if not sample.get(self.KEY_SRC_LANG):
                        sample[self.KEY_SRC_LANG] = source["src_lang"]
                    if not sample.get(self.KEY_TGT_LANG):
                        sample[self.KEY_TGT_LANG] = source["tgt_lang"]
                    sample.setdefault("dataset", source["dataset"])
                    sample.setdefault("manifest_root", str(source["manifest_root"]))
                    sample.setdefault("audio_root", str(source["audio_root"]))
                    sample["lang_pair"] = (
                        f"{sample[self.KEY_SRC_LANG]}-{sample[self.KEY_TGT_LANG]}"
                    )

                source_samples.extend(split_samples)

            if self.config.filter_min_len > 0:
                min_frames = int(self.config.sample_rate * self.config.filter_min_len)
                source_samples = [
                    sample
                    for sample in source_samples
                    if int(sample[self.KEY_N_FRAMES]) >= min_frames
                ]

            if self.config.max_source_frames is not None:
                source_samples = [
                    sample
                    for sample in source_samples
                    if int(sample[self.KEY_N_FRAMES]) <= self.config.max_source_frames
                ]

            samples.extend(source_samples)

        if not samples:
            raise ValueError(
                f"Empty ST dataset manifest(s) for split(s): {self.config.splits}"
            )

        if self.config.max_sentences_per_lang_pair is not None:
            samples = self._limit_samples_per_lang_pair(samples)

        return samples

    def _limit_samples_per_lang_pair(
        self, samples: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        limit = self.config.max_sentences_per_lang_pair
        if limit is None:
            return samples

        counts: Counter[str] = Counter()
        limited: list[dict[str, str]] = []
        for sample in samples:
            lang_pair = sample.get("lang_pair", "")
            if counts[lang_pair] >= limit:
                continue

            limited.append(sample)
            counts[lang_pair] += 1

        logger.info(
            "Limited dataset from %d to %d sample(s), capped at %d sample(s) "
            "per language pair.",
            len(samples),
            len(limited),
            limit,
        )

        return limited

    def _get_reader_samples(
        self, *, is_train: bool, seed: int | None
    ) -> list[dict[str, str]]:
        if (
            not is_train
            or self.config.sampling_alpha >= 1.0
            or len(self._sample_groups) <= 1
        ):
            return self._samples

        return self._sample_training_samples(seed)

    def _sample_training_samples(self, seed: int | None) -> list[dict[str, str]]:
        rng = random.Random(seed)
        groups = list(self._sample_groups.items())
        weights = [
            float(len(group_samples)) ** self.config.sampling_alpha
            for _, group_samples in groups
        ]
        total_weight = sum(weights)
        if total_weight <= 0:
            return self._samples

        epoch_size = self.config.epoch_size or len(self._samples)
        sampled: list[dict[str, str]] = []
        group_values = [group_samples for _, group_samples in groups]
        probabilities = [weight / total_weight for weight in weights]

        for _ in range(epoch_size):
            group_index = rng.choices(
                range(len(group_values)), weights=probabilities, k=1
            )[0]
            sampled.append(rng.choice(group_values[group_index]))

        logger.info(
            "Sampled %d multilingual training example(s) from %d %s group(s) "
            "with alpha=%.3f.",
            len(sampled),
            len(groups),
            self.config.sampling_unit,
            self.config.sampling_alpha,
        )
        self._log_sample_distribution("sampled training", sampled)

        return sampled

    def _group_samples(
        self, samples: list[dict[str, str]]
    ) -> dict[str, list[dict[str, str]]]:
        groups: dict[str, list[dict[str, str]]] = {}
        unit = self.config.sampling_unit

        for sample in samples:
            if unit == "source":
                key = sample.get("dataset", "") + ":" + sample.get("lang_pair", "")
            elif unit == "lang_pair":
                key = sample.get("lang_pair", "")
            else:
                raise ValueError(
                    "`sampling_unit` must be either `lang_pair` or `source`."
                )

            groups.setdefault(key, []).append(sample)

        if len(groups) > 1:
            for key, group_samples in sorted(groups.items()):
                logger.info("Loaded %d sample(s) for %s.", len(group_samples), key)

        return groups

    def _log_sample_distribution(
        self, name: str, samples: list[dict[str, str]]
    ) -> None:
        if not samples:
            logger.info("%s distribution: empty.", name)
            return

        for lang_pair, count in sorted(Counter(_sample_lang_pair(samples)).items()):
            logger.info(
                "%s distribution: %s=%d sample(s), %.2f%%.",
                name,
                lang_pair or "unknown",
                count,
                100.0 * count / len(samples),
            )

        datasets = Counter(sample.get("dataset", "") for sample in samples)
        if len(datasets) > 1:
            for dataset, count in sorted(datasets.items()):
                logger.info(
                    "%s source distribution: %s=%d sample(s), %.2f%%.",
                    name,
                    dataset or "unknown",
                    count,
                    100.0 * count / len(samples),
                )

    def _log_reader_estimate(
        self,
        name: str,
        *,
        samples: list[dict[str, str]],
        global_num_samples: int,
        rank_num_samples: int,
        num_accumulate: int,
        dp_size: int,
    ) -> None:
        estimated_batches = self._estimate_num_batches(samples)
        estimated_updates = _ceil_div(estimated_batches, max(num_accumulate, 1))
        global_estimated_batches = estimated_batches * max(dp_size, 1)

        logger.info(
            "%s reader estimate: %d global sample(s), %d sample(s)/rank across "
            "%d DP rank(s).",
            name,
            global_num_samples,
            rank_num_samples,
            dp_size,
        )
        logger.info(
            "%s reader estimate: about %d batch(es)/rank, %d update(s)/epoch "
            "with num_accumulate=%d.",
            name,
            estimated_batches,
            estimated_updates,
            num_accumulate,
        )
        if dp_size > 1:
            logger.info(
                "%s reader estimate: about %d total batch(es) across all DP ranks.",
                name,
                global_estimated_batches,
            )

    def _estimate_num_batches(self, samples: list[dict[str, str]]) -> int:
        if not samples:
            return 0

        if self.config.max_num_frames_per_batch is None:
            return _ceil_div(len(samples), self.config.batch_size)

        batch_count = 0
        batch_size = 0
        batch_frames = 0
        max_batch_size = max(self.config.batch_size, 1)
        max_frames = max(self.config.max_num_frames_per_batch, 1)

        for sample in samples:
            frames = int(sample[self.KEY_N_FRAMES])
            starts_new_batch = batch_size >= max_batch_size or (
                batch_size > 0 and batch_frames + frames > max_frames
            )
            if starts_new_batch:
                batch_count += 1
                batch_size = 0
                batch_frames = 0

            batch_size += 1
            batch_frames += frames

        if batch_size > 0:
            batch_count += 1

        return batch_count

    def _prepare_example(self, sample: dict[str, str]) -> dict[str, Any]:
        source_path = self._resolve_audio_path(sample[self.KEY_AUDIO], sample)
        source = self._load_audio(source_path)

        target_audio = None
        target_audio_name = sample.get(self.KEY_TGT_AUDIO)
        if target_audio_name:
            target_audio = self._load_audio(
                self._resolve_audio_path(target_audio_name, sample)
            )

        return {
            "id": sample[self.KEY_ID],
            "source": source,
            "source_len": source.shape[0],
            "target_audio": target_audio,
            "source_text": sample.get(self.KEY_SRC_TEXT, self.DEFAULT_SRC_TEXT),
            "target_text": sample.get(self.KEY_TGT_TEXT, ""),
            "speaker": sample.get(self.KEY_SPEAKER, self.DEFAULT_SPEAKER),
            "src_lang": sample.get(self.KEY_SRC_LANG, self.DEFAULT_LANG),
            "tgt_lang": sample.get(self.KEY_TGT_LANG, self.DEFAULT_LANG),
        }

    def _to_batch(self, examples: list[dict[str, Any]]) -> STBatch:
        examples = sorted(examples, key=lambda item: item["source_len"], reverse=True)

        target_audio = [item["target_audio"] for item in examples]
        if all(item is None for item in target_audio):
            target_audio = None

        return STBatch(
            ids=torch.arange(len(examples), dtype=torch.long),
            sample_ids=[item["id"] for item in examples],
            source=[item["source"] for item in examples],
            source_lens=torch.tensor(
                [item["source_len"] for item in examples], dtype=torch.long
            ),
            target_audio=target_audio,
            source_texts=[item["source_text"] for item in examples],
            target_texts=[item["target_text"] for item in examples],
            src_langs=[item["src_lang"] for item in examples],
            tgt_langs=[item["tgt_lang"] for item in examples],
            speakers=[item["speaker"] for item in examples],
        )

    def _resolve_audio_path(self, pathname: str, sample: dict[str, str]) -> Path:
        path = Path(pathname)

        if path.is_absolute():
            return path

        audio_root = Path(
            sample.get("audio_root") or self.config.audio_root or self.config.manifest_root
        )

        return audio_root / path

    def _load_audio(self, path: Path) -> Tensor:
        waveform, sample_rate = _load_audio_file(path)

        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)

        if sample_rate != self.config.sample_rate:
            waveform = _resample(waveform, sample_rate, self.config.sample_rate)

        return waveform

    @property
    def _split_names(self) -> list[str]:
        return [split.strip() for split in self.config.splits.split(",") if split.strip()]

    @property
    def _manifest_sources(self) -> list[dict[str, Any]]:
        if self.config.manifest_list is None:
            src_lang, tgt_lang = _infer_lang_pair(self.config.manifest_root)
            return [
                {
                    "dataset": self.config.manifest_root.parent.name or "st",
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "manifest_root": self.config.manifest_root,
                    "audio_root": self.config.audio_root or self.config.manifest_root,
                    "manifest_format": self.config.manifest_format,
                    "train_split": "",
                    "valid_split": "",
                }
            ]

        sources: list[dict[str, Any]] = []
        with self.config.manifest_list.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                manifest_root = Path(row["manifest_root"]).expanduser().resolve()
                audio_root = row.get("audio_root") or str(manifest_root)
                src_lang = row.get("src_lang") or _infer_lang_pair(manifest_root)[0]
                tgt_lang = row.get("tgt_lang") or _infer_lang_pair(manifest_root)[1]
                sources.append(
                    {
                        "dataset": row.get("dataset") or manifest_root.parent.name,
                        "src_lang": src_lang,
                        "tgt_lang": tgt_lang,
                        "manifest_root": manifest_root,
                        "audio_root": Path(audio_root).expanduser().resolve(),
                        "manifest_format": row.get("manifest_format")
                        or self.config.manifest_format,
                        "train_split": row.get("train_split", ""),
                        "valid_split": row.get("valid_split", ""),
                    }
                )

        if not sources:
            raise ValueError(f"Empty multilingual manifest list: {self.config.manifest_list}")

        return sources

    def _resolve_split_path(self, source: dict[str, Any], split: str) -> Path | None:
        manifest_root = source["manifest_root"]
        for candidate in self._candidate_split_names(source, split):
            for suffix in self._manifest_suffixes(source):
                path = manifest_root / f"{candidate}.{suffix}"
                if path.is_file():
                    if candidate != split:
                        logger.info(
                            "Using fallback split `%s` for requested split `%s` under %s.",
                            candidate,
                            split,
                            manifest_root,
                        )
                    return path

        return None

    def _manifest_suffixes(self, source: dict[str, Any]) -> tuple[str, ...]:
        manifest_format = str(source.get("manifest_format") or "auto").lower()
        if manifest_format == "auto":
            return ("tsv", "jsonl")
        if manifest_format in {"tsv", "jsonl"}:
            return (manifest_format,)

        raise ValueError("`manifest_format` must be one of: auto, tsv, jsonl.")

    def _load_manifest(
        self, manifest_path: Path, source: dict[str, Any]
    ) -> list[dict[str, str]]:
        if manifest_path.suffix == ".tsv":
            return self._load_tsv_manifest(manifest_path)
        if manifest_path.suffix == ".jsonl":
            return self._load_jsonl_manifest(manifest_path, source)

        raise ValueError(f"Unsupported ST manifest format: {manifest_path}")

    def _load_tsv_manifest(self, manifest_path: Path) -> list[dict[str, str]]:
        with manifest_path.open() as f:
            reader = csv.DictReader(
                f,
                delimiter="\t",
                quotechar=None,
                doublequote=False,
                lineterminator="\n",
                quoting=csv.QUOTE_NONE,
            )
            return [dict(row) for row in reader]

    def _load_jsonl_manifest(
        self, manifest_path: Path, source: dict[str, Any]
    ) -> list[dict[str, str]]:
        samples: list[dict[str, str]] = []
        with manifest_path.open() as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError as ex:
                    raise ValueError(
                        f"Invalid JSONL manifest row at {manifest_path}:{line_no}."
                    ) from ex

                if not isinstance(row, dict):
                    raise ValueError(
                        f"JSONL manifest row must be an object at {manifest_path}:{line_no}."
                    )

                samples.append(
                    _normalize_jsonl_sample(
                        row,
                        source=source,
                        sample_rate=self.config.sample_rate,
                        manifest_path=manifest_path,
                        line_no=line_no,
                    )
                )

        return samples

    def _candidate_split_names(self, source: dict[str, Any], split: str) -> list[str]:
        src_lang = source["src_lang"]
        tgt_lang = source["tgt_lang"]
        suffix = f"{src_lang}_{tgt_lang}"
        split_candidates: list[str] = [split, f"{split}_{suffix}"]

        if _is_train_split(split):
            source_split = source.get("train_split", "")
            defaults = [
                self.config.default_train_split,
                f"{self.config.default_train_split}_{suffix}",
                "train",
                f"train_{suffix}",
            ]
            candidates = (
                [
                    source_split,
                    f"{source_split}_{suffix}" if source_split else "",
                ]
                + split_candidates
                + defaults
            )
        else:
            source_split = source.get("valid_split", "")
            defaults = [
                self.config.default_valid_split,
                f"{self.config.default_valid_split}_{suffix}",
                "dev",
                f"dev_{suffix}",
                "valid",
                f"valid_{suffix}",
                "validation",
                f"validation_{suffix}",
            ]
            candidates = (
                [
                    source_split,
                    f"{source_split}_{suffix}" if source_split else "",
                ]
                + split_candidates
                + defaults
            )

        return [candidate for candidate in dict.fromkeys(candidates) if candidate]


def _normalize_jsonl_sample(
    row: dict[str, Any],
    *,
    source: dict[str, Any],
    sample_rate: int,
    manifest_path: Path,
    line_no: int,
) -> dict[str, str]:
    sample_id = _jsonl_value(
        row, "id", "utt_id", "utterance_id", "segment_id", "sid", "key"
    )
    if sample_id is None:
        sample_id = f"{manifest_path.stem}:{line_no}"

    audio = _jsonl_value(
        row,
        "audio",
        "wav",
        "wav_path",
        "audio_path",
        "path",
        "source_audio",
        "src_audio",
    )
    target_text = _jsonl_value(
        row,
        "tgt_text",
        "target_text",
        "translation",
        "tgt",
        "target",
        "text_tgt",
    )
    if audio is None or target_text is None:
        missing = []
        if audio is None:
            missing.append("audio")
        if target_text is None:
            missing.append("target text")
        raise ValueError(
            f"JSONL manifest row at {manifest_path}:{line_no} is missing "
            f"{', '.join(missing)}."
        )

    n_frames = _jsonl_value(
        row, "n_frames", "num_frames", "source_len", "frames", "input_length"
    )
    if n_frames is None:
        duration = _jsonl_value(
            row,
            "duration",
            "duration_sec",
            "duration_secs",
            "duration_seconds",
            "src_duration",
            "source_duration",
        )
        if duration is not None:
            n_frames = str(int(round(float(duration) * sample_rate)))

    if n_frames is None:
        raise ValueError(
            f"JSONL manifest row at {manifest_path}:{line_no} is missing n_frames "
            "or duration; one of them is required for filtering and batching."
        )

    sample = {
        STTrainDataset.KEY_ID: str(sample_id),
        STTrainDataset.KEY_AUDIO: str(audio),
        STTrainDataset.KEY_N_FRAMES: str(int(float(n_frames))),
        STTrainDataset.KEY_TGT_TEXT: str(target_text),
    }

    optional_aliases = {
        STTrainDataset.KEY_SRC_TEXT: (
            "src_text",
            "source_text",
            "transcript",
            "src",
            "source",
            "text_src",
        ),
        STTrainDataset.KEY_SPEAKER: ("speaker", "speaker_id", "spk", "spk_id"),
        STTrainDataset.KEY_SRC_LANG: ("src_lang", "source_lang", "source_language"),
        STTrainDataset.KEY_TGT_LANG: ("tgt_lang", "target_lang", "target_language"),
        STTrainDataset.KEY_TGT_AUDIO: ("target_audio", "tgt_audio", "target_audio_path"),
    }
    for key, aliases in optional_aliases.items():
        value = _jsonl_value(row, *aliases)
        if value is not None:
            sample[key] = str(value)

    sample.setdefault(STTrainDataset.KEY_SRC_LANG, str(source.get("src_lang", "")))
    sample.setdefault(STTrainDataset.KEY_TGT_LANG, str(source.get("tgt_lang", "")))

    return sample


def _jsonl_value(row: dict[str, Any], *keys: str) -> Any | None:
    for key in keys:
        if key not in row:
            continue

        value = row[key]
        if isinstance(value, dict):
            value = _first_present(value, "path", "pathname", "audio", "text", "value")
        if value is not None and value != "":
            return value

    return None


def _first_present(row: dict[str, Any], *keys: str) -> Any | None:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value

    return None


def _load_audio_file(path: Path) -> tuple[Tensor, int]:
    path_str = str(path)
    zip_entry = _parse_zip_byte_range_path(path_str)
    if zip_entry is not None:
        archive_path, offset, size = zip_entry
        data = _read_from_stored_zip(archive_path, offset, size)

        try:
            return _load_audio_bytes(data, path_str)
        except Exception as ex:
            raise RuntimeError(
                "Failed to decode audio bytes from stored zip range "
                f"{archive_path}:{offset}:{size}."
            ) from ex

    try:
        import soundfile as sf
    except (ImportError, OSError):
        sf = None

    if sf is not None:
        array, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(array).transpose(0, 1).contiguous()

        return waveform, int(sample_rate)

    try:
        import torchaudio
    except (ImportError, OSError):
        torchaudio = None

    if torchaudio is not None:
        waveform, sample_rate = torchaudio.load(str(path))
        return waveform.float(), int(sample_rate)

    raise RuntimeError("Audio loading requires soundfile or torchaudio.")


def _load_audio_bytes(data: bytes, source: str) -> tuple[Tensor, int]:
    try:
        import soundfile as sf
    except (ImportError, OSError):
        sf = None

    if sf is not None:
        array, sample_rate = sf.read(
            io.BytesIO(data), dtype="float32", always_2d=True
        )
        waveform = torch.from_numpy(array).transpose(0, 1).contiguous()

        return waveform, int(sample_rate)

    try:
        import torchaudio
    except (ImportError, OSError):
        torchaudio = None

    if torchaudio is not None:
        waveform, sample_rate = torchaudio.load(io.BytesIO(data))
        return waveform.float(), int(sample_rate)

    raise RuntimeError(f"Audio loading requires soundfile or torchaudio: {source}")


def _parse_zip_byte_range_path(path: str) -> tuple[Path, int, int] | None:
    archive, sep, range_spec = path.partition(".zip:")
    if not sep:
        return None

    parts = range_spec.split(":")
    if len(parts) != 2:
        return None

    offset, size = parts
    if not offset.isdigit() or not size.isdigit():
        return None

    return Path(f"{archive}.zip"), int(offset), int(size)


def _mmap_read(path: Path, offset: int, length: int) -> bytes:
    with path.open("rb") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_obj:
            return mmap_obj[offset : offset + length]


def _read_from_stored_zip(zip_path: Path, offset: int, length: int) -> bytes:
    return _mmap_read(zip_path, offset, length)


def _resample(waveform: Tensor, source_rate: int, target_rate: int) -> Tensor:
    try:
        import torchaudio
    except (ImportError, OSError) as ex:
        raise RuntimeError(
            "Audio resampling requires torchaudio when manifest sample rates differ."
        ) from ex

    return torchaudio.functional.resample(waveform, source_rate, target_rate)


def _infer_lang_pair(path: Path) -> tuple[str, str]:
    name = path.name
    if "-" not in name:
        return "", ""

    src_lang, tgt_lang = name.split("-", maxsplit=1)

    return src_lang, tgt_lang


def _is_train_split(split: str) -> bool:
    split = split.lower()

    return split.startswith("train")


def _sample_lang_pair(samples: list[dict[str, str]]) -> list[str]:
    return [
        sample.get("lang_pair")
        or f"{sample.get(STTrainDataset.KEY_SRC_LANG, '')}-{sample.get(STTrainDataset.KEY_TGT_LANG, '')}"
        for sample in samples
    ]


def _shard_samples(
    samples: list[dict[str, str]], rank: int, size: int
) -> list[dict[str, str]]:
    if size <= 1:
        return samples

    return [sample for index, sample in enumerate(samples) if index % size == rank]


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def open_st_train_dataset(config: STTrainDatasetConfig) -> STTrainDataset:
    """Open a TSV-manifest speech-to-text training dataset."""

    config.manifest_root = config.manifest_root.expanduser().resolve()

    if config.audio_root is not None:
        config.audio_root = config.audio_root.expanduser().resolve()

    if config.manifest_list is not None:
        config.manifest_list = config.manifest_list.expanduser().resolve()

    return STTrainDataset(config)
