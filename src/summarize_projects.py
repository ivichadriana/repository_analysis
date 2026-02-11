# src/summarize_projects.py
"""
Generate per-project executive summaries by calling an LLM.
"""

import os, json, glob, argparse, time, re
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI

# -------- Environment & client setup --------
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # Fail fast if API key is missing so the user knows to fix .env
    raise SystemExit("Missing OPENAI_API_KEY in .env")


# Select model from env or default to a small, cost-effective model
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Initialize the OpenAI client (reads key from env automatically)
DEFAULT_HTTP_TIMEOUT = float(os.environ.get("OPENAI_HTTP_TIMEOUT", "60"))  # seconds
client = OpenAI(
    timeout=DEFAULT_HTTP_TIMEOUT, max_retries=0
)  # applies connect/read/write timeouts


# -------- Paths --------
REPORTS_DIR = "reports"  # where we write the per-project markdown reports
SUMMARY_DIR = "data/summary"  # where per-project JSONs live (from rollup_projects.py)
os.makedirs(REPORTS_DIR, exist_ok=True)
_GOAL_RE = re.compile(
    r"^##\s*Summary and Goal\s*\n(.*?)(?:\n##\s+|\Z)", re.DOTALL | re.MULTILINE
)
_ACTIVITY_RE = re.compile(
    r"^##\s*Recent Developments.*?\n(.*?)(?:\n##\s+|\Z)", re.DOTALL | re.MULTILINE
)


# -------- Small helpers --------
def _read_repo_activity_from_md(project_id: str, owner: str, repo: str) -> str | None:
    path = os.path.join(REPORTS_DIR, f"{project_id}__{owner}__{repo}.md")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    m = _ACTIVITY_RE.search(text)
    if not m:
        return None
    g = m.group(1).strip()
    return g or None


def read_json(p: str) -> dict:
    """Load a JSON file into a Python dict."""
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _footer():
    dmy = datetime.now(timezone.utc).strftime("%d/%m/%Y")
    return f"\n\n*Report generated using A.I. on {dmy}*"


def _read_repo_goal_from_md(project_id: str, owner: str, repo: str) -> str | None:
    path = os.path.join(REPORTS_DIR, f"{project_id}__{owner}__{repo}.md")
    if not os.path.exists(path):
        return None
    text = open(path, "r", encoding="utf-8").read()
    m = _GOAL_RE.search(text)
    if not m:
        return None
    g = m.group(1).strip()
    return g or None


def build_project_goal_corpus_from_repo_mds(
    project: dict, per_repo_char_cap: int = 600
) -> str:
    """
    Collect the '## Goal' text from all repo-level MD files for this project.
    Prefer an explicit 'repos' list from the project JSON; otherwise, scan reports/* files
    named like reports/<PROJECT_ID>__<owner>__<repo>.md
    """
    pid = str(project.get("project_id") or "").strip()
    parts = []

    # small fairness cap per repo; keeps the corpus balanced
    def _cap_for(n, soft_total=8000, min_cap=250, max_cap=900):
        return max(min_cap, min(max_cap, soft_total // max(1, n)))

    # 1) If project JSON lists repos, use that (best)
    repos = sorted(
        project.get("repos") or [],
        key=lambda r: (r.get("owner") or "", r.get("repo") or ""),
    )
    per_cap = _cap_for(max(1, len(repos) or 1))
    for r in repos:
        owner, repo = r.get("owner"), r.get("repo")
        if not (owner and repo):
            continue
        g = _read_repo_goal_from_md(pid, owner, repo)
        if g:
            parts.append(f"[REPO {owner}/{repo}]\n{g.strip()[:per_cap]}")

    # 2) Fallback: scan reports for any repo files that match this project id
    if not parts:
        # Fallback: scan and label explicitly
        for path in sorted(glob.glob(os.path.join(REPORTS_DIR, f"{pid}__*__*.md"))):
            # filename pattern: <pid>__<owner>__<repo>.md
            m = re.match(
                rf"^{re.escape(pid)}__([^_]+)__(.+)\.md$", os.path.basename(path)
            )
            if not m:
                continue
            owner, repo = m.group(1), m.group(2)
            g = _read_repo_goal_from_md(pid, owner, repo)
            if g:
                parts.append(f"[REPO {owner}/{repo}]\n{g.strip()[:per_cap]}")

    return ("\n\n".join(parts)).strip()


def build_project_activity_corpus_from_repo_mds(
    project: dict, window_label: str, per_repo_char_cap: int = 900
) -> str:
    """
    Collect '## Recent Developments' blocks from all repo-level MD files for this project.
    Skips the exact no-activity boilerplate line to avoid noise.
    """
    pid = str(project.get("project_id") or "").strip()
    parts = []

    # If project JSON lists repos, prefer that; otherwise scan by prefix.
    repos = project.get("repos") or []
    if repos:
        candidates = [
            (r.get("owner"), r.get("repo"))
            for r in repos
            if r.get("owner") and r.get("repo")
        ]
    else:
        candidates = []
        for path in sorted(glob.glob(os.path.join(REPORTS_DIR, f"{pid}__*__*.md"))):
            m = re.match(
                rf"^{re.escape(pid)}__([^_]+)__(.+)\.md$", os.path.basename(path)
            )
            if m:
                candidates.append((m.group(1), m.group(2)))
    # Collect activity text with per-repo cap
    per_cap = max(300, min(900, 12000 // max(1, len(candidates) or 1)))
    for owner, repo in candidates:
        txt = _read_repo_activity_from_md(pid, owner, repo)
        if not txt:
            continue
        # Skip the boilerplate "No changes ..." line if that's all there is
        if txt.strip() == f"**No changes in {window_label}**":
            continue
        parts.append(f"[REPO {owner}/{repo}]\n{txt.strip()[:per_cap]}")

    return ("\n\n".join(parts)).strip()


def call_llm(prompt: str, model: str, max_retries: int = 4) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are a careful, evidence-bound summarizer that follows directions exactly."
                            f"You take information from multiple repositories and summarize it into a cohesive and succint excecutive summary, highlighting key themes. "
                            f"Your summaries on the project's activity highlight the overall scope of the work done and the work progress across ALL repositories. "
                            f"You are very observant and are able to take multiple respoitories' progress and identify general trends of 'what work has been done across all repositories'. "
                            f"Use ONLY the information in the user message; no external knowledge. "
                            f"Output exactly two Markdown sections with these headings and nothing else: "
                            f"'## Summary and Goal' and '## Recent Developments (<window label>)"
                            f"No bullets. No owners. No generic KPIs. No fluff. "
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise RuntimeError("Empty completion")
            return text
        except Exception as e:
            last_err = e
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"LLM failed after {max_retries} attempts: {last_err}")


def make_project_prompt(project_summary: dict, window_label: str) -> str:
    pid = project_summary.get("project_id", "")
    pname = project_summary.get("project_name") or pid
    interest = project_summary.get("interest_signals", {}) or {}
    examples = project_summary.get("recent_examples", {}) or {}

    # Goal corpus from repo MD "## Goal" (code-derived via summarize_repos)
    goal_corpus = build_project_goal_corpus_from_repo_mds(project_summary)
    has_goal = bool(goal_corpus)

    # Activity corpus from repo MD "## Recent Developments" (primary activity source)
    activity_corpus = build_project_activity_corpus_from_repo_mds(
        project_summary, window_label
    )
    has_activity_md = bool(activity_corpus)

    # Rollup evidence (fallback if no activity corpus available)
    def kv_list(pairs):
        return [
            f"{p['value']} ({p['count']})"
            for p in (pairs or [])
            if p and p.get("value") is not None
        ]

    sg_k = kv_list(interest.get("stargazers"))
    fk_k = kv_list(interest.get("fork_owners"))

    def ex_lines_grouped(items, fields, n_total=10, per_repo=3):
        """
        Build lines like 'field1 — field2 — ...' but balance across repos.
        Requires each item to include 'owner' and 'repo'.
        """
        if not items:
            return []
        buckets = {}
        for it in items:
            if not isinstance(it, dict):
                continue
            key = (it.get("owner"), it.get("repo"))
            buckets.setdefault(key, []).append(it)
        # round-robin across repos
        lines, took = [], {k: 0 for k in buckets}
        while len(lines) < n_total:
            progressed = False
            for k, arr in buckets.items():
                if took[k] >= min(per_repo, len(arr)):
                    continue
                it = arr[took[k]]
                took[k] += 1
                parts = [str(it.get(f)) for f in fields if it.get(f)]
                if parts:
                    # prefix repo for even clearer balance to the model:
                    lines.append(f"[{k[0]}/{k[1]}] — " + " — ".join(parts))
                    progressed = True
                if len(lines) >= n_total:
                    break
            if not progressed:
                break
        return lines

    commit_lines = ex_lines_grouped(
        examples.get("commits"),
        ["committed_at", "author_login", "message_headline", "commit_url"],
    )
    pr_lines = ex_lines_grouped(
        examples.get("pull_requests"),
        ["created_at", "author_login", "title", "pr_url", "state"],
    )
    issue_lines = ex_lines_grouped(
        examples.get("issues"),
        ["created_at", "author_login", "title", "issue_url", "labels"],
    )
    release_lines = ex_lines_grouped(
        examples.get("releases"),
        ["published_at", "release_name", "release_tag", "release_url"],
    )

    # Activity-present flag: prefer the repo-MD view; else infer from rollup examples
    activity_present = bool(has_activity_md) or any(
        bool(examples.get(k))
        for k in ["commits", "pull_requests", "issues", "releases"]
    )

    repo_list = (
        ", ".join(
            [f"{r['owner']}/{r['repo']}" for r in (project_summary.get("repos") or [])]
        )
        or "(unknown)"
    )
    return f"""
You are writing an **executive summary** for a research **project** (multiple repos possible).

STRICT RULES
- Use ONLY the GOAL CORPUS and (if present) the ACTIVITY CORPUS from repo-level .md files; no outside knowledge.
- If ACTIVITY CORPUS is empty, use the EVIDENCE block as a fallback for "Recent Developments".
- Do not output bullet lists of dated events. Synthesize **what actually changed** given the changes observed.
- If ACTIVITY_PRESENT=no, under “Recent Developments” write exactly: **No changes in {window_label}**. Do not write ACTIVITY_PRESENT=yes or no.
- Use inline links inside the prose only when it adds to the narrative or when the example is truly informative (e.g., "...includes [work to fix X](link to where X is fixed)). 
- Inline links to commit/change/issue are only present when it's very representative of the point you are trying to make.        
- Avoid letting any single repository account for most of the narrative.
- Balance coverage across repositories; ensure all repositories are represented where possible.
- Do not under any circumstance name any pull request, commit or issue by name (i.e., pull request 1, commit 70bcd7e6, etc.)
- Do not let a single repository dominate the narrative; integrate themes spanning multiple repos.
- Do not add a link without it being hyperlinked under any circumstance.
- This is how to include inline links:
EXAMPLE 1:
GOOD: A [consolidating pull request]((https://github.com/ivichadriana/deconvolution_sc_sn_comparison/pull/1)) further structured these changes.
BAD: A consolidating pull request further structured these changes, as seen in [pull request #1](https://github.com/ivichadriana/deconvolution_sc_sn_comparison/pull/1).
ALSO BAD: A consolidating pull request further structured these changes (https://github.com/ivichadriana/deconvolution_sc_sn_comparison/pull/1).
EXAMPLE 2:
GOOD: A  [consolidating PR](https://github.com/dashnowlab/STRchive/pull/268) advanced end-to-end workflows.
BAD: A consolidating PR advanced end-to-end workflows [consolidating PR](https://github.com/dashnowlab/STRchive/pull/268).
EXAMPLE 3 (Do not reference EVIDENCE by name (commit names, issue names, etc.). Instead use inline links within the narrative):
GOOD: ...and [entrypoint logic to span development](https://github.com/JRaviLab/molevolvr2.0/commit/6e123e18aa8cb3a26c1432ee945ea1f9575b8e37), test, and production contexts, with [cloud-function deployment made more generic](https://github.com/JRaviLab/molevolvr2.0/commit/2e6cde0bf73e288d4beeb9a46cec3fc5bb491503)
BAD: ...and entrypoint logic to span development, test, and production contexts, with cloud-function deployment made more generic [2e6cde0bf73e288d4beeb9a46cec3fc5bb491503](https://github.com/JRaviLab/molevolvr2.0/commit/2e6cde0bf73e288d4beeb9a46cec3fc5bb491503) and [6e123e18aa8cb3a26c1432ee945ea1f9575b8e37](https://github.com/JRaviLab/molevolvr2.0/commit/6e123e18aa8cb3a26c1432ee945ea1f9575b8e37).
- Keep the whole report under ~250 words. Use these 2 exact sections:

## Summary and Goal
Write 2–8 sentences that **synthesize a single project summary and goal** across ALL repositories.
Base ONLY on GOAL CORPUS. Do **not** list repository names, but you can reference repositories if needed in the narrative. {"Do NOT write 'Not stated'." if has_goal else "If the corpus is empty, write 'Not stated'."}
Also identify the scientific communities or users who benefit if there is explicit evidence in GOAL CORPUS or STAR + FORKS (identity signals);
otherwise do not include this sentence at all. Keep this brief (1–2 sentences).

## Recent Developments ({window_label})
If ACTIVITY CORPUS is present, synthesize from it. Otherwise use EVIDENCE as the source.
Explain the **substance** of changes across repos (features/fixes/docs/refactor/tests/infra/deps),
what areas of the codebase were touched (infer from titles/file cues if present), issues addressed and the scope of issues opened, and releases. 
Support claims with **inline links** to specific commits/PRs/issues/releases only when it fits the narrative. Do not list dates or create a timeline. 
The paragraph should be in prose narrative form, with at most 6 links total, if any. 
Think big picture: are multiple issues or commits working towards the same goal? Use that goal in the narrative rather than specifics about the code change.
Avoid counts without explanation and only mention counts when they aid the narrative. 
If one repository has no changes, simply do not include in the narrative: do not state anything.

ACTIVITY_PRESENT: {str(bool(activity_present)).lower()}

CONTEXT (INPUT; do not echo verbatim)
PROJECT: {pname} ({pid})
REPOSITORIES: {repo_list}
GOAL CORPUS (from repo MD '## Goal' sections):
{build_project_goal_corpus_from_repo_mds(project_summary) if has_goal else "(empty)"}

ACTIVITY CORPUS (from repo MD '## Recent Developments' sections; primary source):
{build_project_activity_corpus_from_repo_mds(project_summary, window_label) if has_activity_md else "(empty)"}

EVIDENCE (fallback source if ACTIVITY CORPUS is empty — for reasoning; do not list verbatim and do not use the names, only hyperlink URL)
Commits:
{chr(10).join("- " + line for line in (commit_lines or ["(none)"]))}
Pull Requests:
{chr(10).join("- " + line for line in (pr_lines or ["(none)"]))}
Issues:
{chr(10).join("- " + line for line in (issue_lines or ["(none)"]))}
Releases:
{chr(10).join("- " + line for line in (release_lines or ["(none)"]))}

STARS (identity signals): {", ".join(sg_k) if sg_k else "(none)"}
FORKS (identity signals): {", ".join(fk_k) if fk_k else "(none)"}
""".strip()


def write_report(
    project_id: str, project_name: str, repo_count: int, window_label: str, body_md: str
):
    path = os.path.join(REPORTS_DIR, f"{project_id}.md")
    title = f"# Executive Summary: Project {project_name or project_id} — {repo_count} repositories — {window_label}"
    with open(path, "w", encoding="utf-8") as f:
        f.write(title + "\n\n" + body_md.strip() + _footer() + "\n")
    print(f"Wrote {path}")


def summarize_project_file(path: str, window_label: str, model: str):
    data = read_json(path)
    pid = data.get("project_id", "UNKNOWN")
    pname = data.get("project_name") or pid
    repo_count = int(data.get("repo_count") or 0)

    # Build the LLM prompt (now derives the Goal strictly from repo MD Goals)
    prompt = make_project_prompt(data, window_label)

    try:
        text = call_llm(prompt, model)
        if not text:
            raise RuntimeError("LLM returned empty content")
    except Exception as e:
        text = (
            f"# Executive Summary\n\n"
            f"_LLM call failed: {e}_\n\n"
            f"- Project ID: {pid}\n"
            f"- Project Name: {pname}\n"
            f"- Window: {window_label}\n"
        )

    write_report(pid, pname, repo_count, window_label, text)


def main():
    """
    CLI entrypoint:
      - loads all per-project JSONs (excluding _portfolio.json),
      - optionally filters by --only project IDs,
      - generates one Markdown file per remaining project.
    """
    parser = argparse.ArgumentParser(
        description="Generate per-project executive summaries."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI model (default from env or gpt-4o-mini)",
    )
    parser.add_argument(
        "--window-label",
        default="the last 90 days",
        help='Human label for the window, e.g., "May–Jul 2025"',
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional list of project IDs to include",
    )
    args = parser.parse_args()

    # Find all project JSONs written by rollup (ignore the portfolio index file)
    paths = sorted(glob.glob(os.path.join(SUMMARY_DIR, "*.json")))
    paths = [p for p in paths if os.path.basename(p) != "_portfolio.json"]

    # Optional filter: only summarize specified project IDs
    if args.only:
        ids = set(args.only)
        # Note: we need to peek to get IDs; small and fine for POC
        paths = [p for p in paths if read_json(p).get("project_id") in ids]

    if not paths:
        raise SystemExit(
            f"No project JSON files found in {SUMMARY_DIR}. Run rollup first."
        )

    # Generate the per-project reports
    for p in paths:
        summarize_project_file(p, args.window_label, args.model)


if __name__ == "__main__":
    main()
