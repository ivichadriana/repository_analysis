#!/usr/bin/env bash
# No -u to avoid tripping on other envs' deactivate.d scripts
set -eo pipefail

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
YML_PATH="${ROOT_DIR}/environment/environment.yml"
ENV_NAME="repo_env"

# Ensure conda exists
if ! command -v conda >/dev/null 2>&1; then
  echo "[err] conda not found on PATH. Please open a new terminal after installing Anaconda/Miniconda."
  exit 1
fi

# Pin to one conda install and always use that bin
CONDA_BASE="$(conda info --base)"
CONDA_BIN="${CONDA_BASE}/bin/conda"
echo ">>> Using conda: ${CONDA_BIN}"

# Create or update the env using THIS conda only
if "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo ">>> Updating existing env: ${ENV_NAME}"
  "${CONDA_BIN}" env update -n "${ENV_NAME}" -f "${YML_PATH}" --prune
else
  echo ">>> Creating env: ${ENV_NAME}"
  "${CONDA_BIN}" env create -n "${ENV_NAME}" -f "${YML_PATH}"
fi

echo ">>> Upgrading pip in ${ENV_NAME}..."
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install --upgrade pip

echo ">>> Environments (paths) for sanity:"
"${CONDA_BIN}" env list

echo ">>> Environment ready: ${ENV_NAME}"
echo "Activate later with:  source \"${CONDA_BASE}/etc/profile.d/conda.sh\" && conda activate ${ENV_NAME}"
