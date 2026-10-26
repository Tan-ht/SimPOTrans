#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

SKIP_SUBMODULES="${SKIP_SUBMODULES:-0}"
SKIP_MAIN_SYNC="${SKIP_MAIN_SYNC:-0}"
SKIP_COMET="${SKIP_COMET:-0}"
SKIP_OMNISTEVAL="${SKIP_OMNISTEVAL:-0}"
SKIP_METRICX="${SKIP_METRICX:-0}"
SKIP_COSYVOICE="${SKIP_COSYVOICE:-0}"
SKIP_COSYVOICE_MODEL="${SKIP_COSYVOICE_MODEL:-0}"
SKIP_CLOUDFLARED="${SKIP_CLOUDFLARED:-1}"
CLOUDFLARED_VERSION="${CLOUDFLARED_VERSION:-latest}"
CLOUDFLARED_INSTALL_DIR="${CLOUDFLARED_INSTALL_DIR:-${HOME}/.local/bin}"

run_step() {
  local title="$1"
  shift
  echo
  echo "==> ${title}"
  "$@"
}

is_enabled_skip() {
  local value
  value="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "${value}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

require_command() {
  local command_name="$1"
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Required command not found: ${command_name}" >&2
    exit 1
  fi
}

install_cloudflared() {
  local os_name
  local arch_name
  local asset_arch
  local download_url
  local tmp_file
  local install_path

  install_path="${CLOUDFLARED_INSTALL_DIR}/cloudflared"
  if command -v cloudflared >/dev/null 2>&1; then
    echo "cloudflared already available: $(command -v cloudflared)"
    cloudflared --version || true
    return
  fi
  if [[ -x "${install_path}" ]]; then
    echo "cloudflared already installed: ${install_path}"
    "${install_path}" --version || true
    case ":${PATH}:" in
      *":${CLOUDFLARED_INSTALL_DIR}:"*) ;;
      *) echo "Note: add ${CLOUDFLARED_INSTALL_DIR} to PATH if your shell does not already include it." ;;
    esac
    return
  fi

  os_name="$(uname -s)"
  arch_name="$(uname -m)"
  if [[ "${os_name}" != "Linux" ]]; then
    echo "Automatic cloudflared install currently supports Linux only; found ${os_name}." >&2
    exit 1
  fi

  case "${arch_name}" in
    x86_64|amd64) asset_arch="amd64" ;;
    aarch64|arm64) asset_arch="arm64" ;;
    *)
      echo "Unsupported cloudflared architecture: ${arch_name}" >&2
      exit 1
      ;;
  esac

  require_command curl
  require_command install

  mkdir -p "${CLOUDFLARED_INSTALL_DIR}"
  tmp_file="$(mktemp /tmp/cloudflared.XXXXXX)"
  if [[ "${CLOUDFLARED_VERSION}" == "latest" ]]; then
    download_url="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${asset_arch}"
  else
    download_url="https://github.com/cloudflare/cloudflared/releases/download/${CLOUDFLARED_VERSION}/cloudflared-linux-${asset_arch}"
  fi

  echo "Downloading cloudflared from ${download_url}"
  curl -L --fail --show-error --progress-bar -o "${tmp_file}" "${download_url}"
  install -m 0755 "${tmp_file}" "${install_path}"
  rm -f "${tmp_file}"
  echo "Installed cloudflared to ${install_path}"
  "${install_path}" --version

  case ":${PATH}:" in
    *":${CLOUDFLARED_INSTALL_DIR}:"*) ;;
    *)
      echo "Note: add ${CLOUDFLARED_INSTALL_DIR} to PATH if your shell does not already include it."
      ;;
  esac
}

cd "${ROOT_PATH}"

require_command git
require_command uv

export PYTHON_VERSION

echo "SimPOTrans environment setup"
echo "Root: ${ROOT_PATH}"
echo "Python: ${PYTHON_VERSION}"

if ! is_enabled_skip "${SKIP_SUBMODULES}"; then
  run_step "Fetch pinned third-party sources" bash scripts/fetch_third_party.sh
else
  echo "Skipping submodule initialization."
fi

if ! is_enabled_skip "${SKIP_MAIN_SYNC}"; then
  run_step "Sync main uv environment" uv sync --frozen
else
  echo "Skipping main uv environment sync."
fi

if ! is_enabled_skip "${SKIP_COMET}"; then
  run_step "Set up COMET environment" bash scripts/setup_comet_env.sh
else
  echo "Skipping COMET environment setup."
fi

if ! is_enabled_skip "${SKIP_OMNISTEVAL}"; then
  run_step "Set up OmniSTEval environment" bash scripts/setup_omnisteval_env.sh
else
  echo "Skipping OmniSTEval environment setup."
fi

if ! is_enabled_skip "${SKIP_METRICX}"; then
  run_step "Set up MetricX environment" bash scripts/setup_metricx_env.sh
else
  echo "Skipping MetricX environment setup."
fi

if ! is_enabled_skip "${SKIP_CLOUDFLARED}"; then
  run_step "Install cloudflared" install_cloudflared
else
  echo "Skipping cloudflared installation."
fi

if ! is_enabled_skip "${SKIP_COSYVOICE}"; then
  run_step "Set up CosyVoice environment" bash scripts/setup_cosyvoice_env.sh

  if ! is_enabled_skip "${SKIP_COSYVOICE_MODEL}"; then
    run_step "Download CosyVoice 3 model" uv run python scripts/download_cosyvoice3.py
  else
    echo "Skipping CosyVoice 3 model download."
  fi
else
  echo "Skipping CosyVoice environment setup."
fi

echo
echo "Environment setup complete."
echo "Useful overrides:"
echo "  PYTHON_VERSION=3.10 bash scripts/setup_env.sh"
echo "  SKIP_COSYVOICE_MODEL=1 bash scripts/setup_env.sh"
echo "  SKIP_CLOUDFLARED=1 bash scripts/setup_env.sh"
echo "  CLOUDFLARED_VERSION=2026.6.1 bash scripts/setup_env.sh"
echo "  SKIP_COMET=1 SKIP_OMNISTEVAL=1 SKIP_METRICX=1 SKIP_COSYVOICE=1 bash scripts/setup_env.sh"
