# Third-party components

The root MIT license applies to SimPOTrans code. It does not replace the terms
of external code, pretrained weights, datasets, or voice recordings.

| Component | Source | Distribution in this release |
| --- | --- | --- |
| WhisperRT-Streaming | https://github.com/tomer9080/WhisperRT-Streaming | Runtime vendored in `third_party/WhisperRT-Streaming`; preserve its `LICENSE` and README. Its package declares CC-BY-NC-4.0 and its license also includes the MIT notice for OpenAI Whisper. |
| MAGIC-TTS | https://github.com/yongaifadian1/MAGIC-TTS | Fetched separately at the revision in `third_party/sources.json`; upstream license file is MIT. |
| Awesome Align | https://github.com/neulab/awesome-align | Fetched separately at the recorded revision; upstream license file is BSD-3-Clause. |
| CosyVoice | https://github.com/FunAudioLLM/CosyVoice | Fetched separately at the recorded revision; upstream license file is Apache-2.0. Nested dependencies retain their own notices. |
| MetricX | https://github.com/google-research/metricx | Fetched separately by `scripts/setup_metricx_env.sh`; retain the fetched repository's license and notices. |

The vendored WhisperRT snapshot includes local packaging integration. Its code,
tokenizer assets, and license notices are preserved from the development tree.
Other Python dependencies are recorded in `pyproject.toml` and `uv.lock` and
installed separately. The source archive contains no pretrained model weights,
dataset audio, or TTS voice prompts. Obtain these inputs under their own terms.
