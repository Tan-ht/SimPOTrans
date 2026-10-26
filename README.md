# SimPOTrans: Streaming Speech Translation

<div align="center">

**Multilingual speech translation · Supervised fine-tuning · RL post-training · Streaming inference**

</div>

SimPOTrans is a research system for simultaneous speech translation. This
repository provides multilingual supervised training, offline and streaming
GRPO post-training, simultaneous and longform evaluation, and a browser demo
with optional speech synthesis.

**[Quick Start](#-quick-start)** · **[Models](#models)** ·
**[Data](#data-layout)** · **[Inference](#inference)** ·
**[Training](#model-training)** · **[Demo](#browser-demo)**

---

## 🚀 Quick Start

### 1. Set up the environment

Extract the source archive, or clone the public repository, then enter its root
directory. Install `uv`, Git, Bash, Zsh, and Python 3.10 on Linux. GPU workflows
need an NVIDIA driver compatible with the PyTorch environment in `uv.lock`.

```bash
git clone https://github.com/Tan-ht/SimPOTrans.git
cd SimPOTrans
bash scripts/fetch_third_party.sh
uv sync --frozen
source .venv/bin/activate
```

The fetch script downloads MAGIC-TTS, Awesome Align, and CosyVoice at the
revisions recorded in `third_party/sources.json`, including their nested
submodules. WhisperRT's runtime is bundled separately. No model weights or
datasets are included in this source release.

For COMET rewards and longform evaluation, prepare the isolated environments:

```bash
bash scripts/setup_comet_env.sh
bash scripts/setup_omnisteval_env.sh
bash scripts/setup_metricx_env.sh
```

Alternatively, `bash scripts/setup_env.sh` performs the complete setup, including
CosyVoice and its model download. Set `SKIP_COSYVOICE=1` to omit that optional
environment. MetricX uses a separate Transformers version; keep it isolated from
the main training environment.

### 2. Prepare a translation checkpoint

Use a checkpoint produced by [model training](#model-training). Set the path to
your checkpoint directory or supported checkpoint file:

```bash
export CKPT_PATH="$PWD/checkpoints/multilingual/rl_stream/best"
```

The path above is a convention, not a bundled model. The checkpoint's speech
encoder, LLM, and LoRA configuration must match `src/st_backbone/config.yaml`.
Base models are obtained separately when first loaded.

### 3. Prepare evaluation data and run

Prepare `data/evaluation_data/en-zh/stream_eval/` using the
[streaming data layout](#data-layout), then run a single chunk size on GPU `0`:

```bash
CHUNK_SIZE=2.0 RUN_OMNISTEVAL=0 \
  uv run zsh src/stream_eval/eval_fixed_dialogue.sh \
  en zh 0 stream_eval "$CKPT_PATH" fixed_2s
```

This runs inference on the audio files listed in `src.txt`. Results are written
under `exp_results/stream_results/`. Set `RUN_OMNISTEVAL=1` after preparing the
segmentation and reference files and installing the scoring environments.

## Models

The default backbone combines **Whisper large-v3** with **Qwen3-8B**, a learned
speech projection, and LoRA adapters. The default LoRA rank is 16, with alpha 32
and attention targets `q_proj`, `k_proj`, and `v_proj`.

| Component | Default configuration | Purpose |
| --- | --- | --- |
| Speech encoder | `whisper`, `large-v3` | Speech features |
| Language model | `Qwen/Qwen3-8B` | Translation generation |
| Reward model | `Unbabel/XCOMET-XL` | RL translation quality reward |
| Optional encoder | `whisper_rt` | Causal speech encoding |
| Optional TTS | CosyVoice 3 or MAGIC-TTS | Spoken demo output |

The model code saves compact trainable state rather than duplicating immutable
base weights. RL checkpoints contain `model.pt`; resumption additionally uses
training state. Keep the original model configuration with each checkpoint.
Base weights, optional segmentation models, and TTS weights are separate inputs.

## Configuration

Run entry-point scripts from the repository root. Scripts discover the root
automatically; training accepts path and resource overrides:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATA_ROOT` | `<repo>/data` | Training dataset root |
| `CONFIG_FILE` | Entry-point-specific YAML | Training configuration |
| `OUTPUT_DIR` | `checkpoints/multilingual/<run_name>` | Training output |
| `CKPT_PATH` | No universal default | Evaluation checkpoint fallback |
| `MODEL_CONFIG` | `src/st_backbone/config.yaml` | Evaluation architecture |
| `CHUNK_SIZE` | Policy-specific | Speech chunk duration in seconds |
| `RUN_OMNISTEVAL` | `1` | Score streaming outputs |
| `RUN_COMET` / `RUN_METRICX` | `1` | Optional longform quality metrics |
| `WANDB_MODE` | `offline` in training launchers | Experiment tracking mode |

Training configs disable W&B recording by default. To use it, enable the W&B
recorder in the selected YAML and choose `WANDB_MODE=offline` or `online`.
Evaluation inputs are resolved under `<repo>/data/evaluation_data/` independently
of the training `DATA_ROOT` override.

## Data layout

Datasets, reference transcripts, and audio are obtained separately. No corpus
samples or personal voice recordings are distributed with this release.

### Training manifests

```text
data/<dataset>/<source>-<target>/
├── train_st.tsv
├── dev_st.tsv
└── audio/
    └── example.wav
```

TSV manifests require `id`, `audio`, `n_frames`, and `tgt_text`. Optional columns
include `src_text`, `src_lang`, `tgt_lang`, `speaker`, and `target_audio`.
Provide source text for source-aware reward scoring. Relative audio paths resolve
against `audio_root`; `n_frames` records waveform samples at the configured sample
rate (16 kHz by default). JSONL manifests are also supported.

A synthetic TSV row looks like this (columns are separated by tabs):

```tsv
id	audio	n_frames	tgt_text	src_text	src_lang	tgt_lang
example_001	audio/example.wav	32000	你好。	Hello.	en	zh
```

This illustrates the schema; supply your own matching audio and translations.

### Streaming evaluation

```text
data/evaluation_data/<source>-<target>/<eval_data>/
├── src.txt              # One longform audio path per line
├── segments.yaml        # Exactly one segmentation YAML in this directory
├── src_segments.txt     # Source transcript per reference segment
└── tgt_segments.txt     # Reference translation per segment
```

Segmentation entries use `wav`, `offset`, and `duration` in seconds; for example:

```yaml
- wav: example.wav
  offset: 0.0
  duration: 2.0
  speaker_id: NA
```

Audio paths must resolve on the inference machine. Segment order and the source
and target text lists must agree. Use absolute paths in `src.txt` when data is
stored outside the repository.

### Utterance-level evaluation

`src/simul_eval/eval_fixed.sh` expects `source.txt` (one audio path per line) and
`target.txt` (one translation per line) in the evaluation directory. Source and
reference line counts must match.

## Inference

### Fixed-chunk streaming

Run one or several chunk sizes with dialogue history:

```bash
CHUNK_SIZES="1.0 2.0 3.0" \
  uv run zsh src/stream_eval/eval_fixed_dialogue.sh \
  en zh 0 stream_eval "$CKPT_PATH" fixed
```

The arguments are source language, target language, device, evaluation directory
name, checkpoint, and optional output name. `DECODE=beam` is the evaluation
default; use `DECODE=sample` for sampling.

### Adaptive streaming

SimulSense evaluation additionally needs its segmentation checkpoint:

```bash
SIMULSENSE_PATH=checkpoints/simulsense THRESHOLDS="1.0" \
  uv run zsh src/stream_eval/eval_simulsense.sh \
  en zh 0 stream_eval "$CKPT_PATH" simulsense
```

Additional policy entry points are `eval_fixed_prefix.sh` and `eval_doa.sh` under
`src/stream_eval/`. Their usage text describes policy-specific settings.

### Scoring existing results

```bash
zsh src/stream_eval/run_omnisteval_longform.sh \
  exp_results/stream_results/en-zh/stream_eval/fixed_2s \
  en zh stream_eval 0
```

OmniSTEval handles alignment and longform quality/latency scoring. COMET and
MetricX can be disabled independently with `RUN_COMET=0` and `RUN_METRICX=0`.
Stored MetricX-24 scores negate the original error score into `[-25, 0]`, so
higher is better.

To run only MetricX on existing outputs:

```bash
METRICX_SOURCE_SENTENCES_FILE=data/evaluation_data/en-zh/stream_eval/src_segments.txt \
  bash scripts/evaluate_metricx.sh \
  exp_results/stream_results/en-zh/stream_eval/fixed_2s 0
```

## Model training

### Multilingual supervised fine-tuning

```bash
DATA_ROOT="$PWD/data" \
  uv run zsh src/st_backbone/train_multilingual.sh \
  0,1 st_multi train_st dev_st \
  covost:en-zh must-c:en-de must-c:en-ja
```

Source specifications use:

```text
<dataset>:<source>-<target>[:manifest_root[:audio_root[:train_split[:valid_split]]]]
```

`SAMPLING_ALPHA=0.5` upsamples lower-resource sources. Model, optimizer, batch,
and validation settings are defined in `src/st_backbone/config.yaml`; launcher
environment overrides are listed in the script's usage text. Single-pair
training uses `src/st_backbone/train_monolingual.sh`.

### Offline GRPO

Set `SFT_CHECKPOINT` to an actual checkpoint produced by supervised training;
Fairseq2 output paths may include a sweep directory and step number.

```bash
export SFT_CHECKPOINT=/path/to/sft/checkpoints/step_2000
uv run zsh src/rl_post/train_offline_grpo.sh \
  0 0 rl_offline train_st dev_st "$SFT_CHECKPOINT" covost:en-zh
```

The first two arguments select rollout and reward devices. The remaining
arguments specify run name, split names, initial checkpoint, and data sources.
Install the COMET environment before using XCOMET rewards.

### Streaming GRPO

```bash
uv run zsh src/rl_post/train_stream_grpo.sh \
  0 0 rl_stream train_st dev_st \
  checkpoints/multilingual/rl_offline/best covost:en-zh
```

Streaming training uses chunked rollouts. The launcher enables monotonicity and
adjacent-chunk repetition rewards by default; configure them through
`USE_MONOTONICITY_REWARD`, `USE_REPETITION_REWARD`, and their coefficient
overrides. Awesome Align is used by the monotonicity reward. RL configuration
files are `src/rl_post/offline_config.yaml` and `src/rl_post/config.yaml`.

## Browser demo

Set `checkpoint` in `demo/config/processor_vad.yaml` to your translation
checkpoint, and select matching source and target languages. Then start:

```bash
uv run zsh demo/run_demo.sh 0 false beam
```

Open `http://localhost:7860`. The demo binds locally by default; microphone
permission is required. Metrics recording is disabled in the release config.

For speech synthesis, first prepare CosyVoice:

```bash
bash scripts/setup_cosyvoice_env.sh
uv run python scripts/download_cosyvoice3.py
```

Add a voice you have permission to use under `tts.common.speakers` in the
processor config, with its `prompt_wav` and `prompt_text`, then select it through
`default_speaker`. No personal speaker presets or recordings are bundled.
Run `uv run zsh demo/run_demo.sh 0 true beam` to enable speech output.
MAGIC-TTS is available as an alternative backend with separately prepared
weights and prompts.

## Release checks

Before installing dependencies or modifying the extracted snapshot, run:

```bash
python3 scripts/check_release.py
```

This checks the file inventory and hashes, Python and shell syntax, JSON, and
common private-path and credential patterns. It requires Python 3, Bash, and
Zsh, but no GPU or model downloads. Full training, inference, and metric checks
require the corresponding environments, checkpoints, and datasets.

## Third-party components

SimPOTrans code is distributed under the [MIT License](LICENSE). Third-party
components retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
In particular, the bundled WhisperRT component declares **CC-BY-NC-4.0** and
includes MIT-licensed Whisper code. Its terms are separate from this project's
MIT license.
