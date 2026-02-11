# src/summarize_repos.py
"""
Generate repository-level summaries grouped by project.
"""

import os, argparse, textwrap, time
import pandas as pd
import json
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime, timezone
from goal_from_code import (
    shallow_clone,
    synthesize_repo_goal_from_code,
    delete_clone_path,
)
import pathlib
import shutil

# ---------- Setup: env + OpenAI client ----------
load_dotenv()  # pulls OPENAI_API_KEY/OPENAI_MODEL from .env, if present
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # fail fast; nothing will work without a key
    raise SystemExit("Missing OPENAI_API_KEY in .env")

# Default model can be overridden via --model
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-nano")
DEFAULT_HTTP_TIMEOUT = float(os.environ.get("OPENAI_HTTP_TIMEOUT", "60"))  # seconds
client = OpenAI(
    timeout=DEFAULT_HTTP_TIMEOUT, max_retries=0
)  # applies connect/read/write timeouts

# Paths
CLEAN_DIR = "data/clean"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

SEED_CSV = "data/projects_seed.csv"
RAW_DIR = "data/raw/github"


# ---------- LLM helper with simple retries ----------
def _footer():
    dmy = datetime.now(timezone.utc).strftime("%d/%m/%Y")
    return f"\n\n*Report generated using A.I. on {dmy}*"


def call_llm(messages, model: str, max_retries: int = 4) -> str:
    """
    Thin wrapper around chat.completions.create with a bounded retry loop.
    Adds a per-attempt watchdog timeout via the client config.
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(model=model, messages=messages)
            txt = resp.choices[0].message.content or ""
            txt = txt.strip()
            if not txt:
                raise RuntimeError("Empty completion")
            return txt
        except Exception as e:
            last_err = e
            # Exponential backoff: 2, 4, 8, 10 (cap)
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"LLM failed after {max_retries} attempts: {last_err}")


# ---------- Load the combined per-table Parquets ----------
def load_repo_frames():
    """
    Read combined tables from data/clean into a dict of DataFrames.
    If a table is missing, return an empty DataFrame for that key.
    This keeps downstream logic simple (no KeyErrors).
    """
    tables = {}
    for name in [
        "commits",
        "issues",
        "pull_requests",
        "releases",
        "stargazers",
        "forks",
        "pr_files",
    ]:
        path = os.path.join(CLEAN_DIR, f"_all_{name}.parquet")
        if os.path.exists(path):
            tables[name] = pd.read_parquet(path)
        else:
            tables[name] = pd.DataFrame()
    return tables


def load_seed(path: str = SEED_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["project_id", "project_name", "owner", "repo"]:
        if col not in df.columns:
            df[col] = None
    return df


def read_raw_json(owner: str, repo: str) -> dict | None:
    path = os.path.join(RAW_DIR, f"{owner}__{repo}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- Determine the universe of repos to summarize ----------
def group_by_repo(tables, seed_df: pd.DataFrame):
    """
    Union of repos seen in the tables AND listed in the seed.
    Returns sorted list of (project_id, project_name, owner, repo)
    """
    keys = set()
    # from tables
    for df in tables.values():
        if isinstance(df, pd.DataFrame) and not df.empty:
            if {"owner", "repo", "project_id", "project_name"}.issubset(df.columns):
                for t in (
                    df[["project_id", "project_name", "owner", "repo"]]
                    .dropna()
                    .itertuples(index=False)
                ):
                    keys.add((t[0], t[1], t[2], t[3]))
    # from seed
    for r in seed_df.itertuples(index=False):
        pid = getattr(r, "project_id", None)
        pname = getattr(r, "project_name", None)
        owner = getattr(r, "owner", None)
        repo = getattr(r, "repo", None)
        if pid and owner and repo:
            keys.add(
                (
                    str(pid),
                    str(pname) if pname is not None else None,
                    str(owner),
                    str(repo),
                )
            )
    return sorted(keys)


# ---------- Pull a single repo's context (goal sources) ----------
def extract_repo_context(tables, owner, repo):
    """
    Try to extract description/homepage/readme from any combined table row.
    If not found (inactive repo), fall back to raw JSON fetched earlier.
    """

    def first_non_null(sub, col):
        if col in sub.columns:
            s = sub[col].dropna()
            if not s.empty:
                return s.iloc[0]
        return None

    # Try tables first
    for df in tables.values():
        if isinstance(df, pd.DataFrame) and not df.empty:
            if "owner" in df.columns and "repo" in df.columns:
                mask = (df.get("owner").astype(str) == str(owner)) & (
                    df.get("repo").astype(str) == str(repo)
                )
                sub = df.loc[mask]
                if not sub.empty:
                    return {
                        "description": first_non_null(sub, "repo_description"),
                        "homepage": first_non_null(sub, "repo_homepage"),
                        "readme": first_non_null(sub, "readme_text"),
                    }

    # Fallback: raw JSON
    raw = read_raw_json(owner, repo) or {}
    if raw:
        # topics + languages are not needed here; we just need goal inputs
        readme = raw.get("__readme_text")
        return {
            "description": raw.get("description"),
            "homepage": raw.get("homepageUrl"),
            "readme": readme,
        }

    # Last resort
    return {"description": None, "homepage": None, "readme": None}


# ---------- Robust datetime sorting helper ----------
def _sorted_desc(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Return df sorted descending by time_col using robust datetime parsing (UTC).
    If the column is missing/empty, returns an empty DataFrame.
    """
    if not isinstance(df, pd.DataFrame) or df.empty or time_col not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce", utc=True)
    return out.sort_values(time_col, ascending=False, kind="stable")


# ---------- Build the LLM prompt for ONE repo ----------


def _urls_only(df: pd.DataFrame, url_col: str, max_n: int = 12) -> list[str]:
    """Return a de-duped, order-preserving list of URLs from df[url_col]."""
    if not isinstance(df, pd.DataFrame) or df.empty or url_col not in df.columns:
        return []
    seen, out = set(), []
    for u in df[url_col].dropna().astype(str):
        if u.startswith("github.com/"):
            u = "https://" + u
        if u.startswith("http") and u not in seen:
            seen.add(u)
            out.append(u)
            if len(out) >= max_n:
                break
    return out


def _identity_signals(tables, owner, repo, max_items=6):
    """Return two short lists describing recent stargazer/fork identities for this repo."""
    stars = tables.get("stargazers", pd.DataFrame())
    forks = tables.get("forks", pd.DataFrame())

    def _safe(s):
        return s if isinstance(s, str) and s.strip() else None

    star_lines = []
    if isinstance(stars, pd.DataFrame) and not stars.empty:
        sub = stars[(stars.get("owner") == owner) & (stars.get("repo") == repo)].copy()
        sub = (
            sub.sort_values("starred_at", ascending=False)
            if "starred_at" in sub.columns
            else sub
        )
        for _, r in sub.head(max_items).iterrows():
            login = _safe(r.get("stargazer_login")) or "unknown"
            name = _safe(r.get("stargazer_name"))
            comp = _safe(r.get("stargazer_company"))
            loc = _safe(r.get("stargazer_location"))
            orgs = _safe(r.get("stargazer_orgs"))
            bits = [f"{login}" + (f" ({name})" if name else "")]
            meta = "; ".join([x for x in [comp, loc, orgs] if x])
            if meta:
                bits.append(meta)
            star_lines.append(" — ".join(bits))

    fork_lines = []
    if isinstance(forks, pd.DataFrame) and not forks.empty:
        sub = forks[(forks.get("owner") == owner) & (forks.get("repo") == repo)].copy()
        sub = (
            sub.sort_values("fork_created_at", ascending=False)
            if "fork_created_at" in sub.columns
            else sub
        )
        for _, r in sub.head(max_items).iterrows():
            login = _safe(r.get("fork_owner_login")) or "unknown"
            name = _safe(r.get("fork_owner_name"))
            typ = _safe(r.get("fork_owner_type"))  # User/Organization
            loc = _safe(r.get("fork_owner_location"))
            orgd = (
                _safe(r.get("fork_owner_org_description"))
                if typ == "Organization"
                else None
            )
            bits = [f"{login}" + (f" ({name})" if name else "")]
            meta = "; ".join([x for x in [typ, loc, orgd] if x])
            if meta:
                bits.append(meta)
            fork_lines.append(" — ".join(bits))

    return (star_lines[:max_items] or ["(none)"], fork_lines[:max_items] or ["(none)"])


def build_repo_prompt(project_id, project_name, owner, repo, ctx, tables, window_label):
    """
    Build a compact, strictly evidence-grounded prompt that yields a short,
    synthesized repo summary with headings. No bullet lists or dated timelines.
    """

    commits = tables.get("commits", pd.DataFrame())
    issues = tables.get("issues", pd.DataFrame())
    prs = tables.get("pull_requests", pd.DataFrame())
    rels = tables.get("releases", pd.DataFrame())

    c_sub = _sorted_desc(
        (
            commits[(commits.get("owner") == owner) & (commits.get("repo") == repo)]
            if isinstance(commits, pd.DataFrame) and not commits.empty
            else pd.DataFrame()
        ),
        "committed_at",
    )
    i_sub = _sorted_desc(
        (
            issues[(issues.get("owner") == owner) & (issues.get("repo") == repo)]
            if isinstance(issues, pd.DataFrame) and not issues.empty
            else pd.DataFrame()
        ),
        "created_at",
    )
    pr_sub = _sorted_desc(
        (
            prs[(prs.get("owner") == owner) & (prs.get("repo") == repo)]
            if isinstance(prs, pd.DataFrame) and not prs.empty
            else pd.DataFrame()
        ),
        "created_at",
    )
    r_sub = _sorted_desc(
        (
            rels[(rels.get("owner") == owner) & (rels.get("repo") == repo)]
            if isinstance(rels, pd.DataFrame) and not rels.empty
            else pd.DataFrame()
        ),
        "published_at",
    )

    def _has_activity(c_sub, pr_sub, i_sub, r_sub) -> bool:
        return any(
            [
                isinstance(c_sub, pd.DataFrame) and not c_sub.empty,
                isinstance(pr_sub, pd.DataFrame) and not pr_sub.empty,
                isinstance(i_sub, pd.DataFrame) and not i_sub.empty,
                isinstance(r_sub, pd.DataFrame) and not r_sub.empty,
            ]
        )

    activity_present = _has_activity(c_sub, pr_sub, i_sub, r_sub)

    goal_text = (ctx.get("readme") or ctx.get("description") or "Not stated.").strip()

    # Identity signals for grounded beneficiaries
    star_lines, fork_lines = _identity_signals(tables, owner, repo)

    # Minimal evidence buffers (for inline linking; NOT to be printed as lists)
    commit_urls = _urls_only(c_sub, "commit_url", max_n=8)
    pr_urls = _urls_only(pr_sub, "pr_url", max_n=8)
    issue_urls = _urls_only(i_sub, "issue_url", max_n=8)
    release_urls = _urls_only(r_sub, "release_url", max_n=8)

    return textwrap.dedent(
        f"""
You are writing an **executive summary** for ONE repository.

STRICT RULES
- Write succinct language and do not repeat yourself.
- Keep total under ~250 words.
- Use ONLY the facts below (GOAL SOURCE, EVIDENCE, IDENTITY SIGNALS). No outside knowledge.
- Only inline link a commit/change/issue when it's very representative of the point you are trying to make or when it adds to the narrative.
- If ACTIVITY_PRESENT=no, or there is no recent activity, under “Recent Developments” write EXACTLY: **No changes in {window_label}**. Do not write ACTIVITY_PRESENT=yes or no.
- Support claims with inline Markdown links **only** within the sentence/statement within the narrative and prose. The paragraphs must flow.
- No dated bullet lists or lists; synthesize into concise paragraphs.
- Do not write bracketed anchors like “[commit …]”, “[PR …]”, or “[issue …]” under any circumstance.
- Do not name any pull request, commit or issue by name (i.e., pull request 1, commit 70bcd7e6, etc.) under any circumstance.
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
Write 2–8 crisp sentences describing the repo’s summary and purpose from GOAL SOURCE only. You do not need to state which repository you are describing.
Evaluate what the codebase aims to do as a whole, big picture, not just within the code. 
Like what are the researchers aiming to do through this code? What are they researching? How are they researching it, and what is the goal?
Also identify the scientific communities or users who benefit from the research if there is explicit evidence in GOAL CORPUS or STAR + FORKS (identity signals);
otherwise do not include this sentence at all. Keep this brief (1–2 sentences).

## Recent Developments ({window_label})
Write 2–10 crisp sentences that explains **what changed**, not when: summarize the scope and the substance of changes (features/fixes/docs/refactor/tests/infra/deps),
what parts of the codebase were affected (infer from file names or titles if apparent), and any issues/release outcomes, what work has been done. 
Think big picture: are multiple issues or commits working towards the same goal? Use that goal in the narrative rather than specifics about the code change.
Do not list dates or create a timeline. The paragraph should be in prose narrative form, with at most 6 links total, if any. 
Avoid counts without explanation and only mention counts when they aid the narrative. 

CONTEXT (INPUT; do not echo verbatim):
Project: {project_name} ({project_id})
Repository: {owner}/{repo}

ACTIVITY_PRESENT: {str(bool(activity_present)).lower()}

GOAL SOURCE:
{goal_text}

EVIDENCE (links for reasoning only; do not echo raw URLs, and do not use the names, only hyperlink URL):
Commits:
{chr(10).join("- " + u for u in (commit_urls or ["(none)"]))}
Pull Requests:
{chr(10).join("- " + u for u in (pr_urls or ["(none)"]))}
Issues:
{chr(10).join("- " + u for u in (issue_urls or ["(none)"]))}
Releases:
{chr(10).join("- " + u for u in (release_urls or ["(none)"]))}

IDENTITY SIGNALS (for grounding beneficiaries):
Stargazers:
{chr(10).join("- " + s for s in star_lines)}
Fork owners:
{chr(10).join("- " + s for s in fork_lines)}
    """
    ).strip()


# ---------- CLI entrypoint ----------
def main():
    parser = argparse.ArgumentParser(
        description="Generate repository-level summaries grouped by project."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI model (default from env or gpt-5-nano)",
    )
    parser.add_argument(
        "--model-high",
        default=DEFAULT_MODEL,
        help="OpenAI model for higher analysis (default: value of --model/DEFAULT_MODEL)"
    )
    parser.add_argument(
        "--model-low",
        default=DEFAULT_MODEL,
        help="OpenAI model for lower analysis (default: value of --model/DEFAULT_MODEL)"
    )
    parser.add_argument(
        "--model-medium",
        default=DEFAULT_MODEL,
        help="OpenAI model for medium analysis (default: value of --model/DEFAULT_MODEL)"
    )
    parser.add_argument(
        "--window-label", 
        default="the last 90 days",
        help='Label for the time window (e.g. "May–July 2025")',
    )
    parser.add_argument(
        "--out-dir", default=REPORTS_DIR, help="Directory to write repo-level reports"
    )
    args = parser.parse_args()

    tables = load_repo_frames()
    seed_df = load_seed(SEED_CSV)
    clone_root_torm = pathlib.Path("data/clones_goals")

    repos = group_by_repo(tables, seed_df)
    if not repos:
        raise SystemExit(
            "No repositories found in clean tables. Run fetch/normalize first."
        )

    # Emit one Markdown per repo
    for pid, pname, owner, repo in repos:
        # Extract README/description/homepage context for this repo
        ctx = extract_repo_context(tables, owner, repo)

        # ALWAYS derive the Goal from the full repo code via shallow clone + map-reduce
        try:
            clone_root = pathlib.Path("data/clones_goals")
            repo_path = shallow_clone(owner, repo, clone_root)
            code_goal = synthesize_repo_goal_from_code(
                repo_path,
                model_high=args.model_high,
                model_low=args.model_low,
                model_medium=args.model_medium,
                call_llm_fn=call_llm,
            )
            # Inject into ctx so build_repo_prompt uses it as GOAL SOURCE
            # build_repo_prompt already prefers ctx["readme"] over description
            ctx = dict(ctx)
            ctx["readme"] = code_goal
        except Exception as e:
            print(f"[warn] Code-derived goal failed for {owner}/{repo}: {e}")
            # Fallback: keep whatever extract_repo_context found (README/description/raw)

        # Build the LLM prompt for this repo
        prompt = build_repo_prompt(
            pid, pname, owner, repo, ctx, tables, args.window_label
        )

        # Call the LLM with a brief, consistent system instruction
        try:
            summary = call_llm(
                [
                    {
                        "role": "system",
                        "content": (
                            f"You are a careful, evidence-bound summarizer that follows directions exactly."
                            f"You take information from a repository (code and activity) and summarize it into a cohesive and succinct excecutive summary, "
                            f"highlighting key themes in issues, pull requests, users, etc. "
                            f"You are very observant and are able to take multiple issues, pull requests, etc. "
                            "and identify general trends of 'what work has been done' and 'what key issues or work pop up consistenly'. "
                            f"Use ONLY the information in the user message; no external knowledge. "
                            f"Output exactly two Markdown sections with these headings and nothing else: "
                            f"'## Summary and Goal' and '## Recent Developments ({args.window_label})'. "
                            f"No bullets. No owners. No generic KPIs. No fluff. "
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=args.model,
            )
        except Exception as e:
            # On failure, write a minimal stub so the pipeline still produces files
            summary = f"_LLM call failed: {e}_"

        # Format the Markdown with a clear title
        title = f"# Executive Summary: {owner}/{repo} — {pname} ({pid}) — {args.window_label}"
        md = title + "\n\n" + summary + _footer() + "\n"

        # reports/<PROJECT_ID>__<owner>__<repo>.md
        out_path = os.path.join(args.out_dir, f"{pid}__{owner}__{repo}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Wrote {out_path}")

        # Clean up the clone to avoid disk growth (Option A)
        try:
            delete_clone_path(repo_path)  # repo_path came from shallow_clone(...)
        except Exception as e:
            print(f"[warn] cleanup failed for {owner}/{repo}: {e}")

    try:
        if clone_root_torm.exists():
            shutil.rmtree(clone_root_torm)
        clone_root_torm.mkdir(
            parents=True, exist_ok=True
        )  # leave an empty folder for next run
    except Exception as e:
        print(f"[warn] failed to clear clone cache at {clone_root_torm}: {e}")


if __name__ == "__main__":
    main()
