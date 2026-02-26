#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SEED_CSV="${ROOT_DIR}/data/projects_seed.csv"
REPORTS_DIR="${ROOT_DIR}/reports"
CLONES_DIR="${ROOT_DIR}/data/clones_agentic"
SKILLS_DIR="${SCRIPT_DIR}/skills/repo-summary"
WINDOW_LABEL="last year"

# ---- Checks ----
if ! command -v gh >/dev/null 2>&1; then
  echo "[err] GitHub CLI (gh) not found. Install from https://cli.github.com"
  exit 1
fi

if ! command -v copilot >/dev/null 2>&1; then
  echo "[err] GitHub Copilot CLI not found."
  echo "      Install with: npm install -g @github/copilot"
  exit 1
fi

if [ ! -f "${SEED_CSV}" ]; then
  echo "[err] Missing ${SEED_CSV}. Run build_projects_seed.py first."
  exit 1
fi

mkdir -p "${REPORTS_DIR}"
mkdir -p "${CLONES_DIR}"

# Install the skill for this session
mkdir -p "${HOME}/.copilot/skills/repo-summary"
cp "${SKILLS_DIR}/SKILL.md" "${HOME}/.copilot/skills/repo-summary/SKILL.md"
echo ">>> Installed repo-summary skill"

# ---- Read seed CSV and process each repo ----
# Skip header row; columns: project_id, project_name, owner, repo
tail -n +2 "${SEED_CSV}" | while IFS=',' read -r project_id project_name owner repo rest; do

  # Strip any quotes or whitespace
  project_id="$(echo "${project_id}" | tr -d '"' | xargs)"
  project_name="$(echo "${project_name}" | tr -d '"' | xargs)"
  owner="$(echo "${owner}" | tr -d '"' | xargs)"
  repo="$(echo "${repo}" | tr -d '"' | xargs)"

  if [ -z "${owner}" ] || [ -z "${repo}" ]; then
    echo "[skip] Empty owner or repo, skipping row."
    continue
  fi

  echo ">>> Processing ${owner}/${repo} (project: ${project_id})"

  CLONE_PATH="${CLONES_DIR}/${owner}__${repo}"
  OUT_FILE="${REPORTS_DIR}/${project_id}__${owner}__${repo}__agentbased.md"

  # Clone or update the repo
  if [ ! -d "${CLONE_PATH}/.git" ]; then
    echo "    Cloning ${owner}/${repo}..."
    gh repo clone "${owner}/${repo}" "${CLONE_PATH}" -- --depth=50 --quiet 2>/dev/null \
      || { echo "[warn] Clone failed for ${owner}/${repo}, skipping."; continue; }
  else
    echo "    Repo already cloned, skipping clone."
  fi

  # Build the activity context file from data/clean parquets
  # We write a human-readable _activity_context.md into the clone root
  # so the Copilot skill can read it as instructed
  ACTIVITY_FILE="${CLONE_PATH}/_activity_context.md"

  python3 "${ROOT_DIR}/agentic/build_activity_context.py" \
    --project-id "${project_id}" \
    --owner "${owner}" \
    --repo "${repo}" \
    --window-label "${WINDOW_LABEL}" \
    --out "${ACTIVITY_FILE}" \
    || { echo "[warn] Could not build activity context for ${owner}/${repo}, writing empty file.";
         echo "# Activity Context\n\nNo activity data available." > "${ACTIVITY_FILE}"; }

  # Run Copilot CLI inside the cloned repo with the skill
  echo "    Running Copilot skill..."
  cd "${CLONE_PATH}"
  copilot -p "Use the /repo-summary skill to summarize this repository. The project is '${project_name}' (ID: ${project_id}). The output directory already exists. Write the output directly to ${OUT_FILE}." \
    --yolo \
    || { echo "[warn] Copilot failed for ${owner}/${repo}, skipping."; cd "${ROOT_DIR}"; continue; }
  cd "${ROOT_DIR}"

  # Clean up activity context file from the clone
  rm -f "${ACTIVITY_FILE}"

  echo "    Done -> ${OUT_FILE}"

done

# Clean up all clones
echo ">>> Cleaning up clones..."
rm -rf "${CLONES_DIR}"
echo ">>> Repo summaries complete."