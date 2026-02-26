#!/usr/bin/env bash
set -euo pipefail

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Check python
if ! command -v python >/dev/null 2>&1; then
  echo "[err] python not found on PATH."
  exit 1
fi
echo ">>> Using python: $(command -v python)"
python --version

PIPELINE_START=$(date +%s)

# 1) Clean outputs
rm -rf "${ROOT_DIR}/data/raw/github/"* \
      "${ROOT_DIR}/data/clean/"* \
      "${ROOT_DIR}/data/summary/"* \
      "${ROOT_DIR}/reports/"* \
      "${ROOT_DIR}/reports_pdf/"* 2>/dev/null || true

# # # 2) Pipeline
# ''' If you want to choose your own projects, remove this script below'''
# python "${ROOT_DIR}/src/build_projects_seed.py" \
#   --repo "nih-cfde/icc-eval-core-private" \
#   --branch "main" \
#   --subdir "data/output" \
#   --output "${ROOT_DIR}/data/projects_seed.csv"

python "${ROOT_DIR}/src/fetch_github_activity.py" --days=365

pass=0
max_passes=10   # safety cap to avoid infinite loops on permanent failures
while [ -s "${ROOT_DIR}/data/raw/github/_failed.json" ] && [ $pass -lt $max_passes ]; do
  pass=$((pass+1))
  echo "[retry] pass $pass"
  python "${ROOT_DIR}/src/fetch_github_activity.py" --days=365 --retry-failed || true
done

python "${ROOT_DIR}/src/normalize_activity.py"
python "${ROOT_DIR}/src/rollup_projects.py"

python "${ROOT_DIR}/src/summarize_repos.py" \
  --window-label "last year" \
  --model gpt-5-nano \
  --model-low gpt-5-nano \
  --model-medium gpt-5-nano \
  --model-high gpt-5-mini

python "${ROOT_DIR}/src/summarize_projects.py" \
  --window-label "last year" \
  --model gpt-5-nano

python "${ROOT_DIR}/src/summarize_portfolio.py" \
  --window-label "last year" \
  --model gpt-5-nano

# 3) PDFs (no logo)
if [ -f "${ROOT_DIR}/src/make_pdfs.py" ]; then
  python "${ROOT_DIR}/src/make_pdfs.py" \
    --in "${ROOT_DIR}/reports" \
    --out "${ROOT_DIR}/reports_pdf" \
    --window-label "last year"
fi

PIPELINE_END=$(date +%s)
ELAPSED=$((PIPELINE_END - PIPELINE_START))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))
echo ">>> DONE. Total time: ${MINUTES}m ${SECONDS}s"

# Append timing to analysis.csv
ANALYSIS_CSV="${ROOT_DIR}/data/analysis.csv"
if [ ! -f "${ANALYSIS_CSV}" ]; then
  echo "pipeline,run_date,duration_seconds,duration_human" > "${ANALYSIS_CSV}"
fi
echo "chatbased,$(date +%Y-%m-%d),${ELAPSED},${MINUTES}m ${SECONDS}s" >> "${ANALYSIS_CSV}"