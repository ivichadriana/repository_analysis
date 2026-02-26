#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPORTS_DIR="${ROOT_DIR}/reports"
WORK_DIR="${ROOT_DIR}/data/portfolio_summary_workdir"
SKILLS_DIR="${SCRIPT_DIR}/skills/portfolio-summary"
WINDOW_LABEL="last year"
OUT_FILE="${REPORTS_DIR}/_portfolio_full__agentbased.md"

# ---- Checks ----
if ! command -v copilot >/dev/null 2>&1; then
  echo "[err] GitHub Copilot CLI not found."
  echo "      Install with: npm install -g @githubnext/github-copilot-cli"
  echo "      Then authenticate: copilot auth"
  exit 1
fi

# Install the skill for this session
mkdir -p "${HOME}/.copilot/skills/portfolio-summary"
cp "${SKILLS_DIR}/SKILL.md" "${HOME}/.copilot/skills/portfolio-summary/SKILL.md"
echo ">>> Installed portfolio-summary skill"

# ---- Collect project-level agentbased MDs ----
# These are files named <PROJECT_ID>__agentbased.md (exactly 2 parts when split on __)
# We explicitly exclude repo-level files (which have 4 parts) and the portfolio file itself

PROJECT_MD_COUNT=$(find "${REPORTS_DIR}" -maxdepth 1 -name "*__agentbased.md" \
  ! -name "*__*__*__agentbased.md" \
  ! -name "_portfolio*" \
  | wc -l | xargs)

if [ "${PROJECT_MD_COUNT}" -eq 0 ]; then
  echo "[err] No project-level agentbased summaries found in ${REPORTS_DIR}."
  echo "      Run run_project_summaries.sh first."
  exit 1
fi

echo ">>> Found ${PROJECT_MD_COUNT} project-level summaries."

# ---- Build a clean work directory containing only project-level MDs ----
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"

find "${REPORTS_DIR}" -maxdepth 1 -name "*__agentbased.md" \
  ! -name "*__*__*__agentbased.md" \
  ! -name "_portfolio*" \
  -exec cp {} "${WORK_DIR}/" \;

echo ">>> Copied project summaries to work directory."

# ---- Run Copilot CLI inside the work directory ----
echo ">>> Running Copilot portfolio-summary skill..."
cd "${WORK_DIR}"
copilot -p "Use the /portfolio-summary skill to generate a portfolio-wide executive summary across all projects. The output directory already exists. Write the output directly to ${OUT_FILE}." \
    --yolo \
    || { echo "[err] Copilot failed for portfolio summary."; cd "${ROOT_DIR}"; exit 1; }
cd "${ROOT_DIR}"

# ---- Clean up ----
echo ">>> Cleaning up work directory..."
rm -rf "${WORK_DIR}"

echo ">>> Portfolio summary complete -> ${OUT_FILE}"