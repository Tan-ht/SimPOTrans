from __future__ import annotations

from dataclasses import dataclass, field

import torch

from fairseq2.recipe.config import (
    AdamWConfig,
    CommonSection,
    DatasetSection,
    GangSection,
    GradAccumulationConfig,
    LRSchedulerSection,
    MetricRecordersConfig,
    MixedPrecisionConfig,
    MyleLRConfig,
    ModelSection,
    OptimizerSection,
    RegimeSection,
    TrainerSection,
    WandbConfig,
)

from .dataset import STTrainDatasetConfig
from .model import ST_MODEL_FAMILY, STModelConfig

ST_DATASET_FAMILY = "st_train"


@dataclass(kw_only=True)
class STTrainRecipeConfig:
    """SimPOTrans-specific knobs used directly by STTrainRecipe."""
    seed: int = 0
    num_accumulate: int = 1
    valid_splits: str | None = "valid"
    max_validation_sentences_per_lang_pair: int | None = None
    max_bleu_batches: int | None = None
    max_bleu_sentences_per_lang_pair: int | None = None
    compute_comet: bool = False
    comet_model: str = "Unbabel/XCOMET-XL"
    comet_batch_size: int = 2
    comet_gpus: int = 1
    comet_python: str | None = ".comet-venv/bin/python"
    log_example_batches: int = 1
    early_stop_patience: int | None = 10
    early_stop_min_delta: float = 0.0
    early_stop_higher_better: bool = True


@dataclass(kw_only=True)
class STModelSection(ModelSection):
    """Default model section for the SimPOTrans ST model family."""
    family: str | None = ST_MODEL_FAMILY
    arch: str | None = "speech_to_text_base"
    dtype: torch.dtype = torch.bfloat16
    config_overrides: object | None = field(
        default_factory=lambda: STModelConfig(torch_dtype="bfloat16")
    )


@dataclass(kw_only=True)
class STTrainDatasetSection(DatasetSection):
    """Default dataset section for manifest-based ST training data."""
    family: str | None = ST_DATASET_FAMILY
    config_overrides: object | None = field(
        default_factory=lambda: STTrainDatasetConfig(
            splits="train_st",
            default_train_split="train_st",
            default_valid_split="dev_st",
            sampling_alpha=1.0,
            sampling_unit="lang_pair",
            max_sentences_per_lang_pair=None,
            batch_size=15,
            max_source_frames=200_000,
            max_num_frames_per_batch=400_000,
        )
    )


@dataclass(kw_only=True)
class STTrainConfig:
    """Top-level fairseq2 recipe configuration for SimPOTrans training."""
    model: STModelSection = field(default_factory=STModelSection)
    dataset: STTrainDatasetSection = field(default_factory=STTrainDatasetSection)
    recipe: STTrainRecipeConfig = field(default_factory=STTrainRecipeConfig)
    gang: GangSection = field(default_factory=GangSection)
    trainer: TrainerSection = field(
        default_factory=lambda: TrainerSection(
            mixed_precision=MixedPrecisionConfig(mode="static", dtype=torch.bfloat16),
            grad_accumulation=GradAccumulationConfig(num_batches=1),
            max_grad_norm=1.0,
        )
    )
    optimizer: OptimizerSection = field(
        default_factory=lambda: OptimizerSection(
            name="adamw",
            config=AdamWConfig(lr=1e-4, betas=(0.9, 0.98), weight_decay=0.05),
        )
    )
    lr_scheduler: LRSchedulerSection | None = field(
        default_factory=lambda: LRSchedulerSection(
            name="myle",
            config=MyleLRConfig(num_warmup_steps=5000, start_lr=2e-6),
        )
    )
    regime: RegimeSection = field(
        default_factory=lambda: RegimeSection(
            num_steps=100_000,
            num_data_epochs=30,
            validate_after_n_steps=0,
            validate_every_n_steps=2000,
            publish_metrics_after_n_steps=1,
            publish_metrics_every_n_steps=100,
            checkpoint_after_n_steps=0,
            checkpoint_every_n_steps=2000,
            keep_best_n_checkpoints=10,
            keep_checkpoint_every_n_steps=None,
            score_metric="bleu",
        )
    )
    common: CommonSection = field(
        default_factory=lambda: CommonSection(
            seed=0,
            metric_recorders=MetricRecordersConfig(
                wandb=WandbConfig(
                    enabled=True,
                    project="SimPOTrans",
                    run_name=None,
                )
            ),
        )
    )
