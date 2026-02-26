#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SEED_CSV="${ROOT_DIR}/data/projects_seed.csv"
REPORTS_DIR="${ROOT_DIR}/reports"
WORK_DIR="${ROOT_DIR}/data/project_summary_workdirs"
SKILLS_DIR="${SCRIPT_DIR}/skills/project-summary"
WINDOW_LABEL="last year"

# ---- Checks ----
if ! command -v copilot >/dev/null 2>&1; then
  echo "[err] GitHub Copilot CLI not found."
  echo "      Install with: npm install -g @githubnext/github-copilot-cli"
  echo "      Then authenticate: copilot auth"
  exit 1
fi

if [ ! -f "${SEED_CSV}" ]; then
  echo "[err] Missing ${SEED_CSV}. Run build_projects_seed.py first."
  exit 1
fi

# Install the skill for this session
mkdir -p "${HOME}/.copilot/skills/project-summary"
cp "${SKILLS_DIR}/SKILL.md" "${HOME}/.copilot/skills/project-summary/SKILL.md"
echo ">>> Installed project-summary skill"

mkdir -p "${WORK_DIR}"

# ---- Collect unique project IDs and their names from seed CSV ----
declare -A PROJECT_NAMES

tail -n +2 "${SEED_CSV}" | while IFS=',' read -r project_id project_name owner repo rest; do
  project_id="$(echo "${project_id}"   | tr -d '"' | xargs)"
  project_name="$(echo "${project_name}" | tr -d '"' | xargs)"
  if [ -n "${project_id}" ]; then
    PROJECT_NAMES["${project_id}"]="${project_name}"
  fi
done

# Get unique project IDs
UNIQUE_PROJECTS=$(tail -n +2 "${SEED_CSV}" | cut -d',' -f1 | tr -d '"' | xargs -I{} echo {} | sort -u)

# ---- Process each project ----
for project_id in ${UNIQUE_PROJECTS}; do

  project_name="${PROJECT_NAMES[${project_id}]:-${project_id}}"
  echo ">>> Processing project: ${project_id} (${project_name})"

  # Count how many repo-level agentbased MDs exist for this project
  REPO_MD_COUNT=$(ls "${REPORTS_DIR}/${project_id}__"*"__"*"__agentbased.md" 2>/dev/null | wc -l | xargs)

  if [ "${REPO_MD_COUNT}" -eq 0 ]; then
    echo "    [warn] No repo-level agentbased summaries found for ${project_id}, skipping."
    continue
  fi

  echo "    Found ${REPO_MD_COUNT} repo-level summaries."

  # Create a clean working directory for this project
  # containing only its repo-level agentbased MDs
  PROJECT_WORK_DIR="${WORK_DIR}/${project_id}"
  rm -rf "${PROJECT_WORK_DIR}"
  mkdir -p "${PROJECT_WORK_DIR}"

  # Copy this project's repo-level summaries into the work dir
  cp "${REPORTS_DIR}/${project_id}__"*"__"*"__agentbased.md" "${PROJECT_WORK_DIR}/"

  OUT_FILE="${REPORTS_DIR}/${project_id}__agentbased.md"

  # Run Copilot CLI inside the project work directory
  echo "    Running Copilot skill..."
  cd "${PROJECT_WORK_DIR}"
copilot -p "Use the /project-summary skill to summarize this project across all repositories. The project is '${project_name}' (ID: ${project_id}). The output directory already exists. Write the output directly to ${OUT_FILE}." \
    --yolo \
    || { echo "[warn] Copilot failed for project ${project_id}, skipping."; cd "${ROOT_DIR}"; continue; }
  cd "${ROOT_DIR}"

  echo "    Done -> ${OUT_FILE}"

done

# Clean up work directories
echo ">>> Cleaning up work directories..."
rm -rf "${WORK_DIR}"
echo ">>> Project summaries complete."