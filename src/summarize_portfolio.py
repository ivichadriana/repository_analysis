# src/summarize_portfolio.py
"""
Generate a portfolio-level executive summary report in Markdown, synthesizing information.
"""

import os, json, argparse, textwrap, time
from typing import List, Dict, Any
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime, timezone
import re

# ------------------ Environment & Client Setup ------------------
load_dotenv()  # pull OPENAI_API_KEY / OPENAI_MODEL from .env if present

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # Fail fast if the key is missing-nothing will work without it.
    raise SystemExit("Missing OPENAI_API_KEY in .env")

# Default model can be overridden by --model at runtime
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Initialize OpenAI client (reads API key from env)
DEFAULT_HTTP_TIMEOUT = float(os.environ.get("OPENAI_HTTP_TIMEOUT", "60"))  # seconds
client = OpenAI(
    timeout=DEFAULT_HTTP_TIMEOUT, max_retries=0
)  # applies connect/read/write timeouts

# Where we read portfolio JSON and write the final Markdown
REPORTS_DIR = "reports"
SUMMARY_DIR = "data/summary"
os.makedirs(REPORTS_DIR, exist_ok=True)

# --- Project MD: extract "## Recent Developments (...)" ---
_PROJECT_ACTIVITY_RE_TMPL = (
    r"^##\s*Recent Developments\s*\(\s*{label}\s*\)\s*\n(.*?)(?:\n##\s+|\Z)"
)

# put near the top with the other regexes
_GOAL_RE = re.compile(
    r"^\s*##\s*Summary\s*(?:and|&)\s*Goal\s*\n(.*?)(?=^\s*##\s|\Z)",
    flags=re.DOTALL | re.MULTILINE | re.IGNORECASE,
)


def _read_project_activity_from_md(project_id: str, window_label: str) -> str | None:
    """
    Read reports/<PROJECT_ID>.md and extract the '## Recent Developments (<window_label>)' block.
    Returns stripped text or None.
    """
    path = os.path.join(REPORTS_DIR, f"{project_id}.md")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # Make a regex that matches the exact window label literally
    pat = re.compile(
        _PROJECT_ACTIVITY_RE_TMPL.format(label=re.escape(window_label)),
        flags=re.DOTALL | re.MULTILINE,
    )
    m = pat.search(text)
    if not m:
        return None
    body = (m.group(1) or "").strip()
    return body or None


def build_portfolio_activity_corpus_from_project_mds(
    projects: List[Dict[str, Any]], window_label: str
) -> str:
    """
    Concatenate the '## Recent Developments (<window_label>)' section from each project's MD,
    with clear project labels and a per-project character cap to prevent domination.
    Skips boilerplate 'No changes in <window_label>' lines.
    """

    def _cap_for(n, soft_total=12000, min_cap=300, max_cap=900):
        # Aim for ~12k chars total; clamp to keep useful signal
        return max(min_cap, min(max_cap, soft_total // max(1, n)))

    parts: list[str] = []
    n = max(1, len(projects))
    per_cap = _cap_for(n)
    for p in sorted(projects, key=lambda x: (x.get("project_id") or "")):
        pid = (p.get("project_id") or "").strip()
        if not pid:
            continue
        block = _read_project_activity_from_md(pid, window_label)
        if not block:
            continue
        if block.strip() == f"**No changes in {window_label}**":
            continue
        pname = (p.get("project_name") or pid).strip()
        parts.append(f"[PROJECT {pid} — {pname}]\n{block.strip()[:per_cap]}")
    return ("\n\n".join(parts)).strip()


def _goal_from_project_md(project_id: str) -> str | None:
    """
    Fallback: read 'reports/<PROJECT_ID>.md' and extract the text under '## Goal'
    up to the next '## ' or end-of-file. Returns stripped text or None.
    """
    md_path = os.path.join(REPORTS_DIR, f"{project_id}.md")
    if not os.path.exists(md_path):
        return None
    with open(md_path, "r", encoding="utf-8") as f:
        md = f.read()
    m = re.search(
        r"^## Summary and Goal\s*\n(.*?)(?:\n## |\Z)",
        md,
        flags=re.DOTALL | re.MULTILINE,
    )
    if not m:
        return None
    text = m.group(1).strip()
    return text or None


# ------------------ Utilities ------------------


def _footer():
    dmy = datetime.now(timezone.utc).strftime("%d/%m/%Y")
    return f"\n\n*Report generated using A.I. on {dmy}*"


def read_json(path: str) -> Dict[str, Any]:
    """Load JSON file from disk into a Python dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def call_llm(messages, model: str, max_retries: int = 4) -> str:
    """
    Robust LLM call for prebuilt messages:
    - bounded retries with backoff,
    - treats empty content as an error,
    - surfaces the last error clearly.
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise RuntimeError("Empty completion")
            return text
        except Exception as e:
            last_err = e
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"LLM failed after {max_retries} attempts: {last_err}")


def _chunk_text(s: str, chunk_chars: int = 12000, overlap: int = 500) -> List[str]:
    """
    Split a long string into overlapping chunks.
    We use this when synthesizing a clean goal statement from a long README.
    """
    s = s or ""
    if len(s) <= chunk_chars:
        return [s]
    chunks, start, n = [], 0, len(s)
    while start < n:
        end = min(start + chunk_chars, n)
        chunks.append(s[start:end])
        if end == n:
            break
        start = max(0, end - overlap)  # overlap keeps continuity between chunks
    return chunks


def compute_portfolio_metrics(projects: List[Dict[str, Any]]):
    totals = dict(
        projects=len(projects),
        active=sum(1 for p in projects if p.get("active_in_window")),
    )
    commits = prs = issues = releases = 0
    theme_ctr = Counter()
    area_ctr = Counter()

    for p in projects:
        ex = p.get("recent_examples") or {}
        commits += len(ex.get("commits") or [])
        prs += len(ex.get("pull_requests") or [])
        issues += len(ex.get("issues") or [])
        releases += len(ex.get("releases") or [])
        for t in p.get("themes") or []:
            if t and t.get("value"):
                theme_ctr.update([t["value"]])
        for a in p.get("areas_touched") or []:
            if a and a.get("value"):
                area_ctr.update([a["value"]])

    totals.update(
        {
            "commits": commits,
            "prs": prs,
            "issues": issues,
            "releases": releases,
            "top_themes": [f"{k} ({v})" for k, v in theme_ctr.most_common(6)],
            "top_areas": [f"{k} ({v})" for k, v in area_ctr.most_common(6)],
        }
    )
    return totals


# ------------------ README → Goal (helper) ------------------
def summarize_readme_goal(readme_text: str, model: str) -> str:
    """
    Distill a potentially long README into a crisp purpose statement.
    Strategy:
      1) Summarize each chunk into "purpose-only" bullets.
      2) Synthesize a final  goal from all bullets.
    """
    chunks = _chunk_text(readme_text, chunk_chars=12000, overlap=500)
    bullets = []
    for i, ch in enumerate(chunks, 1):
        messages = [
            {
                "role": "system",
                "content": "You extract the core PURPOSE of a repository from README text.",
            },
            {
                "role": "user",
                "content": textwrap.dedent(
                    f"""
                From the README chunk below, write 3–5 ultra-concise bullets capturing the repository's PURPOSE only.
                Avoid installation/usage details, badges, and marketing language.

                --- README CHUNK {i}/{len(chunks)} ---
                {ch}
            """
                ).strip(),
            },
        ]
        bullets.append(call_llm(messages, model=model))

    synth_messages = [
        {
            "role": "system",
            "content": "You distill bullets into a faithful, succinct purpose statement.",
        },
        {
            "role": "user",
            "content": textwrap.dedent(
                f"""
            Combine the bullets below into a single 1–2 sentence statement describing the repository/project goal.
            Do not invent details.

            BULLETS:
            {chr(10).join(bullets)}
        """
            ).strip(),
        },
    ]
    return call_llm(synth_messages, model=model).strip()


# ------------------ Render helpers ------------------
def safe_kv_list(items: List[Dict[str, Any]], k="value", c="count", top=6) -> List[str]:
    """
    Turn [{'value': 'foo', 'count': 7}, ...] into ['foo (7)', ...], with a top N cap.
    Safely handles missing keys or None.
    """
    out = []
    for it in (items or [])[:top]:
        if it and it.get(k):
            if c in it and it.get(c) is not None:
                out.append(f"{it[k]} ({it[c]})")
            else:
                out.append(str(it[k]))
    return out


def ex_lines(items: List[Dict[str, Any]], fields: List[str], n=4) -> List[str]:
    """
    Turn an array of dicts into 'field1 - field2 - field3' lines.
    n caps how many lines to return.
    """
    if not items:
        return []
    lines = []
    for it in items[:n]:
        parts = []
        for f in fields:
            v = it.get(f)
            if v:
                parts.append(str(v))
        if parts:
            lines.append(" - ".join(parts))
    return lines


# ------------------ Prompt builders ------------------
def _extract_goal_from_md(text: str) -> str | None:
    # normalize line endings and strip BOM if present
    text = (text or "").lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    m = _GOAL_RE.search(text)
    if not m:
        return None
    body = (m.group(1) or "").strip()
    return body or None


def _collect_project_goal(project_id: str) -> str | None:
    """
    Read reports/<PROJECT_ID>.md and extract its ## Goal text.
    Returns a single string or None if missing.
    """
    path = os.path.join(REPORTS_DIR, f"{project_id}.md")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            md = f.read()
        return _extract_goal_from_md(md)
    except Exception:
        return None


def _collect_repo_goals_for_project(project_id: str) -> list[str]:
    """
    Scan reports/ for ALL repo-level files belonging to this project
    (files named '<PROJECT_ID>__<owner>__<repo>.md') and collect their ## Goal text.
    """
    goals: list[str] = []
    prefix = f"{project_id}__"
    if not os.path.isdir(REPORTS_DIR):
        return goals
    for fname in os.listdir(REPORTS_DIR):
        if not (fname.startswith(prefix) and fname.endswith(".md")):
            continue
        # Skip the project-level file (`reports/<PROJECT_ID>.md`)
        if fname == f"{project_id}.md":
            continue
        path = os.path.join(REPORTS_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                md = f.read()
            g = _extract_goal_from_md(md)
            if g:
                goals.append(g)
        except Exception:
            # Ignore unreadable files; keep going
            pass
    return goals


def build_balanced_goal_corpus(projects: List[Dict[str, Any]]) -> str:
    """
    Create a single corpus representing the portfolio mission, in priority:
      1) Project-level MD '## Goal' (reports/<PROJECT_ID>.md) — synthesized from repo code
      2) Fallback: aggregate all repo-level MD '## Goal' for that project
      3) Last resort: project.repo_context (readme -> description -> topics)
    Includes ALL available goal text without truncation.
    """

    def _cap_for(n, soft_total=10000, min_cap=250, max_cap=800):
        return max(min_cap, min(max_cap, soft_total // max(1, n)))

    parts: list[str] = []
    n = max(1, len(projects))
    per_cap = _cap_for(n)

    for p in sorted(projects, key=lambda x: (x.get("project_id") or "")):
        pid = (p.get("project_id") or "").strip()
        project_text = ""

        # (1) Prefer project-level Goal
        g_project = _collect_project_goal(pid)
        if g_project:
            project_text = g_project
        else:
            # (2) Fallback to aggregated repo Goals
            repo_goals = _collect_repo_goals_for_project(pid)
            if repo_goals:
                project_text = "\n\n".join(repo_goals)
            else:
                # (3) Last resort: repo_context
                ctx = p.get("repo_context") or {}
                readme = (ctx.get("readme") or "").strip()
                desc = (ctx.get("description") or "").strip()
                topics = ctx.get("topics") or []
                if readme:
                    project_text = readme
                elif desc:
                    project_text = desc
                elif topics:
                    project_text = "Topics: " + ", ".join(map(str, topics[:6]))
                else:
                    project_text = ""

        project_text = project_text.strip()
        if project_text:
            pname = (p.get("project_name") or pid).strip()
            parts.append(f"[PROJECT {pid} — {pname}]\n{project_text[:per_cap]}")

    return "\n\n".join(parts).strip()


def build_portfolio_overview_prompt(
    projects: List[Dict[str, Any]], window_label: str, model: str
) -> str:
    """
    Build a prompt for a half-page portfolio summary with two sections:
      1) Portfolio Goal — one unified mission synthesized across ALL projects (from project/repo MDs/readme fallbacks).
      2) Recent Developments — PRIMARY: concatenation of 'Recent Developments' from project MDs; FALLBACK: rollup metrics.
    """
    m = compute_portfolio_metrics(projects)

    # 1) Unified goal corpus (all available text; uncapped)
    goal_corpus = build_balanced_goal_corpus(projects)
    has_goal = bool(goal_corpus.strip())

    # 2) Primary activity source: concatenate each project's Recent Developments from its MD
    activity_corpus = build_portfolio_activity_corpus_from_project_mds(
        projects, window_label
    )
    has_activity_md = bool(activity_corpus.strip())

    # 3) Small set of inlineable examples (used only if we fall back)
    def pick_inline_examples(ps: List[Dict[str, Any]], max_n: int = 4) -> List[str]:
        """
        Choose at most one example per project before repeating (round-robin across projects),
        preferring releases -> PRs -> commits -> issues.
        """
        buckets = ["releases", "pull_requests", "commits", "issues"]
        # Pre-extract first candidate per bucket per project
        by_project = []
        for p in sorted(ps, key=lambda x: (x.get("project_id") or "")):
            ex = p.get("recent_examples") or {}
            cand = None
            for b in buckets:
                arr = ex.get(b) or []
                if arr:
                    it = arr[0]
                    name = (
                        it.get("title")
                        or it.get("release_name")
                        or it.get("message_headline")
                        or b
                    )
                    url = (
                        it.get("pr_url")
                        or it.get("release_url")
                        or it.get("commit_url")
                        or it.get("issue_url")
                    )
                    if url:
                        cand = f"[{name}]({url})"
                        break
            if cand:
                by_project.append(cand)
        # Round-robin: one per project, then stop at max_n
        return by_project[:max_n]

    examples = pick_inline_examples(projects, max_n=4)
    examples_str = "; ".join(examples) if examples else "-"
    # 4) Project list (context only; do not require the model to list them)
    proj_list = (
        ", ".join(
            [
                f"{p.get('project_id')}"
                for p in sorted(projects, key=lambda x: (x.get("project_id") or ""))
            ]
        )
        or "-"
    )

    # 5) Final prompt
    return textwrap.dedent(
        f"""
You are writing an **executive summary** for a research **portfolio** (multiple projects involving multiple repos possible).

STRICT RULES
- Use ONLY the GOAL CORPUS and, if present, the ACTIVITY CORPUS below; no outside knowledge.
- If ACTIVITY CORPUS is empty, use PORTFOLIO METRICS (fallback) for 'Recent Developments'.
- Do not output bullet lists of dated events. Synthesize **what actually changed**.
- Use ONLY the information below. Do not invent anything. Do **not** list individual project names in the output.
- Balance coverage across projects; all projects inform the narrative.
- Do not let a single project dominate more than ~40% of sentences; highlight cross-cutting themes spanning multiple projects when possible, but all projects must inform narrative.
- Inline links are allowed when they aid the narrative.
- Support claims with inline Markdown links **only** within the sentence/statement within the narrative and prose. The paragraphs must flow.
- No dated bullet lists or lists; synthesize into concise paragraphs.
- Do not write bracketed anchors like “[commit …]”, “[PR …]”, or “[issue …]” under any circumstance.
- Do not name any pull request, commit or issue by name (i.e., pull request 1, commit 70bcd7e6, etc.) under any circumstance.
- Do not add a link without it being hyperlinked under any circumstance.
- This is how to include incline links:
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

- Headings must appear exactly as below. 
- Keep the whole report under ~250 words. Use these exact sections:

# Executive Summary                          
## Portfolio Summary and Goal
Write a single, unified 2-5 sentence mission that captures the overall summary of the projects and what ALL projects together aim to achieve.
Base this ONLY on GOAL CORPUS. You MUST synthesize across ALL content, not a subset.
Do not name repositories, and only name projects if it aids in the narrative. {"Do NOT write 'Not stated'." if has_goal else "If the corpus is empty, write 'Not stated'."}
Also identify the scientific communities or users who benefit if there is explicit evidence in GOAL CORPUS or STAR + FORKS (identity signals);
otherwise do not include this sentence at all. Keep this brief (1–2 sentences).

## Recent Developments ({window_label})
If ACTIVITY CORPUS is present, synthesize from it. Otherwise, use PORTFOLIO METRICS (fallback).
Explain substantive work (features, fixes, refactors, tests, infra, docs), issues addressed, and progress made.
Think big picture: are multiple issues or commits working towards the same goal? Use that goal in the narrative rather than specifics about the code change.
Tie claims to cross-cutting themes/areas when evident.
Avoid dates, project/repo lists, and changelog-style enumeration. Focus on work progress towards the overall goal.
If one repository has no changes, simply do not include in the narrative, do not state anything.

ACTIVITY CORPUS (from project MD '## Recent Developments ({window_label})'; primary source):
{activity_corpus if has_activity_md else "(empty)"}

PORTFOLIO METRICS (fallback; for reasoning only; do not include directly):
- Projects: {m["projects"]} total; {m["active"]} active
- Activity in {window_label}: {m["commits"]} commits, {m["prs"]} PRs, {m["issues"]} issues, {m["releases"]} releases
- Top themes: {", ".join(m["top_themes"]) if m["top_themes"] else "-"}
- Top areas touched: {", ".join(m["top_areas"]) if m["top_areas"] else "-"}
- Inline examples: {examples_str}

GOAL CORPUS (distill into a unified purpose; do not copy verbatim):
{goal_corpus if goal_corpus else "(empty)"}
CONTEXT (do not echo; for balance only):
PROJECTS: {proj_list}
    """
    ).strip()


# ------------------ Main (CLI) ------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate a portfolio executive report for ALL projects."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI model name (default from env or gpt-4o-mini)",
    )
    parser.add_argument(
        "--window-label",
        default="the last 90 days",
        help='Label for the time window, e.g., "May–July 2025"',
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional list of project IDs to include",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(REPORTS_DIR, "_portfolio_full.md"),
        help="Output Markdown path (default: reports/_portfolio_full.md)",
    )
    args = parser.parse_args()

    # 1) Load the machine portfolio JSON created by rollup_projects.py
    portfolio_path = os.path.join(SUMMARY_DIR, "_portfolio.json")
    if not os.path.exists(portfolio_path):
        raise SystemExit(f"Missing {portfolio_path}. Run rollup_projects.py first.")
    portfolio = read_json(portfolio_path)

    # Extract the list of project dicts
    projects = portfolio.get("projects") or []

    # Optional: filter to a subset of project IDs
    if args.only:
        keep = set(args.only)
        projects = [p for p in projects if p.get("project_id") in keep]

    # Ensure we have something to summarize
    if not projects:
        raise SystemExit("No projects to summarize (after filtering).")

    # 2) Build the portfolio-level "Executive Overview" text
    overview_prompt = build_portfolio_overview_prompt(
        projects, args.window_label, args.model
    )
    overview_text = call_llm(
        [
            {
                "role": "system",
                "content": (
                    f"You are a careful, evidence-bound summarizer that follows directions exactly."
                    f"You take information from multiple projects and summarize it into a cohesive and succint executive summary, highlighting key themes. "
                    f"Your summaries on the project's activity highlight the overall scope of the work done and the work progress. "
                    f"You are very observant and are able to take multiple project's progress and identify general trends of 'what work has been done across all projects'. "
                    f"Use ONLY the information in the user message; no external knowledge. "
                    f"Output exactly two Markdown sections with these headings and nothing else: "
                    f"'## Portfolio Summary and Goal' and '## Recent Developments ({args.window_label})'. "
                    f"No bullets. No owners. No generic KPIs. No fluff. "
                ),
            },
            {"role": "user", "content": overview_prompt},
        ],
        model=args.model,
    )

    project_count = len(projects)
    md = (
        f"# Portfolio Summary - {project_count} projects ({args.window_label})\n\n"
        f"{overview_text}\n"
        f"{_footer()}\n"
    )

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
