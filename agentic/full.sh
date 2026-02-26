#!/usr/bin/env bash
set -euo pipefail

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

# Check Copilot CLI
if ! command -v copilot >/dev/null 2>&1; then
  echo "[err] GitHub Copilot CLI not found."
  echo "      Install with: npm install -g @githubnext/github-copilot-cli"
  echo "      Then authenticate: copilot auth"
  exit 1
fi
echo ">>> Using copilot: $(command -v copilot)"

# Check GitHub CLI (needed for cloning)
if ! command -v gh >/dev/null 2>&1; then
  echo "[err] GitHub CLI (gh) not found. Install from https://cli.github.com"
  exit 1
fi
echo ">>> Using gh: $(command -v gh)"

# ---- 1) Clean agentbased outputs only ----
echo ">>> Cleaning previous agentbased outputs..."
find "${ROOT_DIR}/reports" -maxdepth 1 -name "*__agentbased.md" -delete 2>/dev/null || true
find "${ROOT_DIR}/reports_pdf" -maxdepth 1 -name "*__agentbased.pdf" -delete 2>/dev/null || true
echo ">>> Clean complete."

# ---- 2) Build project seed ----
echo ">>> Building project seed..."
# python "${ROOT_DIR}/src/build_projects_seed.py" \
#   --repo "nih-cfde/icc-eval-core-private" \
#   --branch "main" \
#   --subdir "data/output" \
#   --output "${ROOT_DIR}/data/projects_seed.csv"

# ---- 3) Fetch GitHub activity ----
echo ">>> Fetching GitHub activity..."
python "${ROOT_DIR}/src/fetch_github_activity.py" --days=365

pass=0
max_passes=10
while [ -s "${ROOT_DIR}/data/raw/github/_failed.json" ] && [ $pass -lt $max_passes ]; do
  pass=$((pass+1))
  echo "[retry] pass $pass"
  python "${ROOT_DIR}/src/fetch_github_activity.py" --days=365 --retry-failed || true
done

# ---- 4) Normalize and rollup ----
echo ">>> Normalizing and rolling up..."
python "${ROOT_DIR}/src/normalize_activity.py"
python "${ROOT_DIR}/src/rollup_projects.py"

# ---- 5) Repo-level summaries (agentic) ----
echo ">>> Running repo-level summaries..."
bash "${SCRIPT_DIR}/run_repo_summaries.sh"

# ---- 6) Project-level summaries (agentic) ----
echo ">>> Running project-level summaries..."
bash "${SCRIPT_DIR}/run_project_summaries.sh"

# ---- 7) Portfolio-level summary (agentic) ----
echo ">>> Running portfolio summary..."
bash "${SCRIPT_DIR}/run_portfolio_summary.sh"

# ---- 8) PDFs ----
echo ">>> Generating PDFs..."
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
echo "agentbased,$(date +%Y-%m-%d),${ELAPSED},${MINUTES}m ${SECONDS}s" >> "${ANALYSIS_CSV}"
```

This way every run appends a row to `data/analysis.csv` so you accumulate timing data across multiple runs. The file will look like:
```
pipeline,run_date,duration_seconds,duration_human
chatbased,2026-02-26,1823,30m 23s
agentbased,2026-02-26,2541,42m 21s