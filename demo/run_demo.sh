#!/usr/bin/env zsh
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  zsh demo/run_demo.sh [device] [tts_enabled] [decode_mode]

Examples:
  zsh demo/run_demo.sh 0 true       # use GPU 0 and enable TTS in the demo UI
  zsh demo/run_demo.sh 0 false      # use GPU 0 and disable TTS in the demo UI
  zsh demo/run_demo.sh cuda:1 on    # use torch device cuda:1 and enable TTS
  zsh demo/run_demo.sh cpu 0        # run on CPU and disable TTS
  zsh demo/run_demo.sh 0 beam       # use beam search with beam_size=2
  zsh demo/run_demo.sh 0 false beam # disable TTS and use beam search
  DEVICE=0 TTS_ENABLED=true zsh demo/run_demo.sh
  DEVICE=0 DECODE_MODE=beam zsh demo/run_demo.sh
  DEVICE=0 TTS_VISIBLE_DEVICES=1 TTS_ENABLED=true zsh demo/run_demo.sh
  CLOUDFLARE_TUNNEL=false DEVICE=0 TTS_ENABLED=true zsh demo/run_demo.sh # local-only, no public link

If no device is provided, the speech processor config is used as-is.
If no TTS flag is provided, demo/config/server.yaml is used as-is.
If no decode mode is provided, the speech processor config is used as-is.
TTS follows the main GPU unless TTS_VISIBLE_DEVICES is set.
The demo is local-only by default. Set CLOUDFLARE_TUNNEL=true to create a public link.
Accepted TTS values: true/false, 1/0, yes/no, on/off.
Accepted decode modes: current/default/config, sample, beam/beam_search.
BEAM_SIZE can override the beam mode size; default: 2.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_PATH=${(%):-%x}
SCRIPT_DIR=${SCRIPT_PATH:A:h}
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
SERVER_CONFIG=${SERVER_CONFIG:-demo/config/server.yaml}
SPEECH_PROCESSOR_CONFIG=${SPEECH_PROCESSOR_CONFIG:-demo/config/processor_vad.yaml}
device_arg=${1:-${DEVICE:-}}
tts_enabled_arg=${2:-${TTS_ENABLED:-}}
decode_mode_arg=${3:-${DECODE_MODE:-}}
beam_size=${BEAM_SIZE:-4}
if [[ -z "${3:-}" && -n "${tts_enabled_arg}" ]]; then
    case "${tts_enabled_arg:l}" in
        current|default|config|sample|beam|beam_search)
            decode_mode_arg=${tts_enabled_arg}
            tts_enabled_arg=""
            ;;
    esac
fi
runtime_speech_config=${SPEECH_PROCESSOR_CONFIG}
runtime_server_config=${SERVER_CONFIG}
temp_files=()
cloudflared_pid=""
demo_public_url=""

cleanup() {
    if [[ -n "${cloudflared_pid}" ]]; then
        kill "${cloudflared_pid}" 2>/dev/null || true
        wait "${cloudflared_pid}" 2>/dev/null || true
    fi
    if (( ${#temp_files[@]} > 0 )); then
        rm -f -- "${temp_files[@]}"
    fi
}

print_demo_public_url() {
    if [[ -n "${demo_public_url}" ]]; then
        echo
        echo "============================================================"
        echo "Demo public URL: ${demo_public_url}"
        echo "============================================================"
        echo
    fi
}
trap cleanup EXIT

export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}:${PYTHONPATH:-}"
export UV_CACHE_DIR="${ROOT_DIR}/.uv-cache"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/simpotrans-matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

cd "${ROOT_DIR}"

cloudflare_tunnel=${CLOUDFLARE_TUNNEL:-false}
case "${cloudflare_tunnel:l}" in
    true|1|yes|on)
        if ! command -v cloudflared >/dev/null 2>&1; then
            echo "cloudflared is required because CLOUDFLARE_TUNNEL defaults to true." >&2
            echo "Install cloudflared or run with CLOUDFLARE_TUNNEL=false." >&2
            exit 2
        fi
        runtime_server_config=$(mktemp /tmp/simpotrans-demo-server-cloudflare.XXXXXX.yaml)
        temp_files+=("${runtime_server_config}")
        uv run python - "${SERVER_CONFIG}" "${runtime_server_config}" <<'PYCONFIG'
import sys
from pathlib import Path

import yaml


class IndentedDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow=flow, indentless=False)


source, target = sys.argv[1:]
with open(source, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}
config["single_origin_websockets"] = True
config.setdefault("single_origin_ws_path", "/ws")
config.setdefault("single_origin_audio_ws_path", "/tts-audio")
webui = config.setdefault("webui", {})
webui["ws_path"] = config["single_origin_ws_path"]
webui["audio_ws_path"] = config["single_origin_audio_ws_path"]
webui["ws_hostname"] = ""
webui["audio_ws_hostname"] = ""
webui["ws_port"] = ""
webui["audio_ws_port"] = ""
Path(target).parent.mkdir(parents=True, exist_ok=True)
with open(target, "w", encoding="utf-8") as f:
    yaml.dump(config, f, Dumper=IndentedDumper, sort_keys=False, allow_unicode=True)
PYCONFIG
        http_port=$(uv run python - "${runtime_server_config}" <<'PYPORT'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}
print(int(config.get("http_port", 7860)))
PYPORT
)
        cloudflared_log=$(mktemp /tmp/simpotrans-demo-cloudflared.XXXXXX.log)
        temp_files+=("${cloudflared_log}")
        echo "Starting Cloudflare quick tunnel for http://localhost:${http_port} ..."
        cloudflared tunnel --url "http://localhost:${http_port}" >"${cloudflared_log}" 2>&1 &
        cloudflared_pid=$!
        echo "Cloudflare tunnel process pid=${cloudflared_pid}. Waiting for public URL..."
        public_url=""
        for _ in {1..60}; do
            if ! kill -0 "${cloudflared_pid}" 2>/dev/null; then
                echo "cloudflared exited before publishing a URL. Log:" >&2
                sed -n '1,160p' "${cloudflared_log}" >&2
                exit 1
            fi
            public_url=$(grep -Eo 'https://[A-Za-z0-9.-]+\.trycloudflare\.com' "${cloudflared_log}" | tail -n 1 || true)
            if [[ -n "${public_url}" ]]; then
                break
            fi
            sleep 0.5
        done
        if [[ -n "${public_url}" ]]; then
            demo_public_url="${public_url}"
            uv run python - "${runtime_server_config}" "${demo_public_url}" <<'PYPUBLICURL'
import sys
import yaml

path, public_url = sys.argv[1:]
with open(path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}
config["demo_public_url"] = public_url
with open(path, "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
PYPUBLICURL
            print_demo_public_url
        else
            echo "Cloudflare tunnel started, but no public URL was found within 30 seconds." >&2
            echo "cloudflared log: ${cloudflared_log}" >&2
            sed -n '1,160p' "${cloudflared_log}" >&2
        fi
        ;;
    false|0|no|off|"")
        ;;
    *)
        echo "Invalid CLOUDFLARE_TUNNEL value: ${cloudflare_tunnel}" >&2
        exit 2
        ;;
esac

if [[ -n "${device_arg}" || -n "${decode_mode_arg}" ]]; then
    case "${device_arg}" in
        "")
            runtime_device=""
            runtime_visible_devices=""
            ;;
        cpu|cuda|cuda:*)
            runtime_device=${device_arg}
            runtime_visible_devices=""
            ;;
        *)
            export CUDA_VISIBLE_DEVICES=${device_arg}
            runtime_device=cuda
            runtime_visible_devices=${device_arg}
            ;;
    esac

    runtime_decode_mode=""
    if [[ -n "${decode_mode_arg}" ]]; then
        case "${decode_mode_arg:l}" in
            current|default|config|sample)
                runtime_decode_mode=${decode_mode_arg:l}
                ;;
            beam|beam_search)
                runtime_decode_mode=beam
                ;;
            *)
                echo "Invalid decode_mode value: ${decode_mode_arg}" >&2
                usage
                exit 2
                ;;
        esac
    fi

    runtime_speech_config=$(mktemp /tmp/simpotrans-demo-fixed-dialogue-vad.XXXXXX.yaml)
    temp_files+=("${runtime_speech_config}")
    uv run python - "${SPEECH_PROCESSOR_CONFIG}" "${runtime_speech_config}" "${runtime_device}" "${runtime_visible_devices}" "${runtime_decode_mode}" "${beam_size}" <<'PY'
import sys
from pathlib import Path

import yaml


class IndentedDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow=flow, indentless=False)


source, target, device, visible_devices, decode_mode, beam_size_text = sys.argv[1:]
with open(source, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}
if device:
    config["device"] = device
    if visible_devices:
        config["simulst_visible_devices"] = visible_devices
    else:
        config.pop("simulst_visible_devices", None)
if decode_mode == "beam":
    decode = config.setdefault("decode", {})
    decode["beam_num"] = int(beam_size_text)
    decode["do_sample"] = False
elif decode_mode == "sample":
    decode = config.setdefault("decode", {})
    decode["beam_num"] = 1
    decode["do_sample"] = True
Path(target).parent.mkdir(parents=True, exist_ok=True)
with open(target, "w", encoding="utf-8") as f:
    yaml.dump(config, f, Dumper=IndentedDumper, sort_keys=False, allow_unicode=True)
PY
    if [[ -n "${runtime_device}" ]]; then
        echo "Using device=${runtime_device} (config: ${runtime_speech_config})"
    fi
    if [[ "${runtime_decode_mode}" == "beam" ]]; then
        echo "Using decode=beam_search beam_size=${beam_size} (config: ${runtime_speech_config})"
    elif [[ "${runtime_decode_mode}" == "sample" ]]; then
        echo "Using decode=sample (config: ${runtime_speech_config})"
    fi
fi

if [[ -n "${tts_enabled_arg}" ]]; then
    case "${tts_enabled_arg:l}" in
        true|1|yes|on)
            runtime_tts_enabled=true
            ;;
        false|0|no|off)
            runtime_tts_enabled=false
            ;;
        *)
            echo "Invalid tts_enabled value: ${tts_enabled_arg}" >&2
            usage
            exit 2
            ;;
    esac

    runtime_tts_config=$(mktemp /tmp/simpotrans-demo-processor-tts.XXXXXX.yaml)
    temp_files+=("${runtime_tts_config}")
    uv run python - "${runtime_speech_config}" "${runtime_tts_config}" "${runtime_tts_enabled}" "${TTS_VISIBLE_DEVICES:-}" <<PY
import sys
from pathlib import Path

import yaml


class IndentedDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow=flow, indentless=False)


source, target, enabled_text, tts_visible_devices = sys.argv[1:]
with open(source, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}
tts = config.setdefault("tts", {})
common = tts.setdefault("common", {}) if isinstance(tts, dict) else {}
if isinstance(common, dict):
    common["enabled"] = enabled_text == "true"
    if tts_visible_devices.strip():
        common["visible_devices"] = tts_visible_devices.strip()
    else:
        common.pop("visible_devices", None)
else:
    tts["enabled"] = enabled_text == "true"
    if tts_visible_devices.strip():
        tts["visible_devices"] = tts_visible_devices.strip()
    else:
        tts.pop("visible_devices", None)
Path(target).parent.mkdir(parents=True, exist_ok=True)
with open(target, "w", encoding="utf-8") as f:
    yaml.dump(config, f, Dumper=IndentedDumper, sort_keys=False, allow_unicode=True)
PY
    runtime_speech_config=${runtime_tts_config}
    if [[ -n "${TTS_VISIBLE_DEVICES:-}" ]]; then
        echo "Using tts.visible_devices=${TTS_VISIBLE_DEVICES} (config: ${runtime_speech_config})"
    fi
    echo "Using tts.enabled=${runtime_tts_enabled} (config: ${runtime_speech_config})"
fi

uv run python demo/server.py \
    --server-config "${runtime_server_config}" \
    --speech-processor-config "${runtime_speech_config}"
print_demo_public_url
