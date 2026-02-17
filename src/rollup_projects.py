# src/rollup_projects.py
"""
Aggregate per-repo activity tables into per-project summaries.
"""

import os, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from collections import Counter


# ----------- Paths -----------
CLEAN_DIR = "data/clean"  # where normalize_activity.py wrote the _all_*.parquet files
OUT_DIR = "data/summary"  # where we’ll write per-project JSON and CSV
os.makedirs(OUT_DIR, exist_ok=True)
SEED_CSV = "data/projects_seed.csv"
RAW_DIR = "data/raw/github"


# ----------- Small utilities -----------
def _jsonable(v):
    """Convert pandas/NumPy/time values to JSON-safe Python types/strings."""
    # pandas/pyarrow timestamps -> ISO8601 (UTC)
    if isinstance(v, pd.Timestamp):
        if v.tzinfo is None:
            v = v.tz_localize("UTC")
        else:
            v = v.tz_convert("UTC")
        return v.isoformat()
    # python datetime -> ISO8601 (UTC)
    if isinstance(v, datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        else:
            v = v.astimezone(timezone.utc)
        return v.isoformat()
    # NumPy scalars -> native Python
    if isinstance(v, np.generic):
        return v.item()
    return v


def load_or_empty(path: str) -> pd.DataFrame:
    """Read a Parquet if it exists; else return an empty DataFrame (so code can proceed)."""
    return pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()


def load_seed(path: str = SEED_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize columns we care about
    for col in ["project_id", "project_name", "owner", "repo"]:
        if col not in df.columns:
            df[col] = None
    # enforce string where applicable
    if "project_id" in df.columns:
        df["project_id"] = df["project_id"].astype("string")
    if "project_name" in df.columns:
        df["project_name"] = df["project_name"].astype("string")
    return df


def read_raw_json(owner: str, repo: str) -> dict | None:
    path = os.path.join(RAW_DIR, f"{owner}__{repo}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def top_n(counter_like, n=5):
    """
    Accepts:
      - a list of values (we'll count them), or
      - a Counter
    Returns [{'value': <thing>, 'count': <int>}, ...] top n by frequency.
    """
    return [
        {"value": k, "count": int(v)} for k, v in Counter(counter_like).most_common(n)
    ]


def explode_tags(series: pd.Series) -> list[str]:
    """
    Intent tags are stored like 'feature|fix|docs'.
    Turn a series of such strings into a flat list of tag tokens.
    """
    vals = []
    for s in series.dropna():
        vals.extend([t for t in str(s).split("|") if t])
    return vals


def collect_examples(
    df: pd.DataFrame, cols: list[str], n=5, sort_col: str | None = None
) -> list[dict]:
    """
    Take the first n rows of df (optionally after sorting by sort_col DESC),
    extract a subset of fields named in `cols`, converting values to JSON-safe types/strings.
    """
    out = []
    if df is None or df.empty:
        return out

    sub = df
    if sort_col and sort_col in df.columns:
        try:
            sub = df.sort_values(sort_col, ascending=False)
        except Exception:
            # If dtype is mixed/invalid for sorting, fall back to original order
            sub = df

    for _, r in sub.head(n).iterrows():
        item = {}
        for c in cols:
            v = r.get(c)
            item[c] = _jsonable(v)
        out.append(item)
    return out


from collections import Counter


def _iter_project_repos(
    per_project_dfs: dict[str, pd.DataFrame], seed_slice: pd.DataFrame | None = None
) -> list[tuple[str, str]]:
    """
    Return sorted list of (owner, repo) pairs present in activity tables OR listed in the seed.
    """
    pairs = set()
    for df in per_project_dfs.values():
        if (
            isinstance(df, pd.DataFrame)
            and not df.empty
            and {"owner", "repo"}.issubset(df.columns)
        ):
            sub = df[["owner", "repo"]].dropna().astype(str)
            pairs.update(map(tuple, sub.values))
    if isinstance(seed_slice, pd.DataFrame) and not seed_slice.empty:
        for _, r in seed_slice.dropna(subset=["owner", "repo"]).astype(str).iterrows():
            pairs.add((r["owner"], r["repo"]))
    return sorted(pairs)


def _extract_single_repo_fields(
    per_project_dfs: dict[str, pd.DataFrame], owner: str, repo: str
) -> dict:
    """
    Pull context for one repo from any activity table row; fallback to raw JSON if needed.
    Returns fields: description, homepage, topics[], primary_language, languages[], readme (str|None).
    """
    fields = {
        "description": None,
        "homepage": None,
        "topics": [],
        "primary_language": None,
        "languages": [],
        "readme": None,
    }

    def _split_csv(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return []
        return [x.strip() for x in str(val).split(",") if x.strip()]

    # Try to find a row for this repo in any table (prefer one with readme_text)
    tables = [
        "commits",
        "issues",
        "pull_requests",
        "releases",
        "pr_files",
        "stargazers",
        "forks",
    ]
    best = None
    for t in tables:
        df = per_project_dfs.get(t)
        if not (isinstance(df, pd.DataFrame) and not df.empty):
            continue
        sub = df[(df.get("owner") == owner) & (df.get("repo") == repo)]
        if sub.empty:
            continue
        with_readme = (
            sub[~sub.get("readme_text").isna()]
            if "readme_text" in sub.columns
            else pd.DataFrame()
        )
        best = with_readme.iloc[0] if not with_readme.empty else sub.iloc[0]
        break

    if best is not None:
        fields["description"] = best.get("repo_description") or None
        fields["homepage"] = best.get("repo_homepage") or None
        fields["topics"] = _split_csv(best.get("repo_topics"))
        fields["primary_language"] = best.get("repo_primary_language") or None
        fields["languages"] = _split_csv(best.get("repo_languages"))
        fields["readme"] = best.get("readme_text") or None
        # early return if we already have a README
        if fields["readme"]:
            return fields

    # Fallback to raw JSON (helps for inactive repos)
    raw = read_raw_json(owner, repo) or {}
    if raw:
        # topics
        topics = []
        try:
            nodes = (raw.get("repositoryTopics") or {}).get("nodes", []) or []
            topics = [
                n["topic"]["name"]
                for n in nodes
                if n and n.get("topic") and n["topic"].get("name")
            ]
        except Exception:
            pass
        # languages
        langs = []
        try:
            nodes = (raw.get("languages") or {}).get("nodes", []) or []
            langs = [n.get("name") for n in nodes if n and n.get("name")]
        except Exception:
            pass
        fields["description"] = fields["description"] or raw.get("description")
        fields["homepage"] = fields["homepage"] or raw.get("homepageUrl")
        fields["topics"] = fields["topics"] or topics
        fields["primary_language"] = fields["primary_language"] or (
            (raw.get("primaryLanguage") or {}).get("name")
        )
        fields["languages"] = fields["languages"] or langs
        fields["readme"] = fields["readme"] or raw.get("__readme_text")

    return fields


def build_repo_context_all(
    per_project_dfs: dict[str, pd.DataFrame],
    seed_slice: pd.DataFrame | None = None,
    readme_chars: int = 20000,
) -> dict:
    """
    Aggregate context across ALL repos in the project (including inactive ones listed in the seed).
    - description: most common non-empty; if multiple distinct, join a few unique variants (<=500 chars).
    - homepage: most common non-empty
    - topics: frequency-sorted union
    - primary_language: most common
    - languages: frequency-sorted union
    - readme: concatenation of short per-repo README excerpts with 'owner/repo' headers (truncated to readme_chars)
    """
    pairs = _iter_project_repos(per_project_dfs, seed_slice=seed_slice)
    if not pairs:
        return {
            "description": None,
            "homepage": None,
            "topics": [],
            "primary_language": None,
            "languages": [],
            "readme": None,
        }

    descs, homes = [], []
    topic_ctr, lang_ctr, primary_ctr = Counter(), Counter(), Counter()
    parts = []

    for owner, repo in pairs:
        f = _extract_single_repo_fields(per_project_dfs, owner, repo)
        if f["description"]:
            descs.append(f["description"].strip())
        if f["homepage"]:
            homes.append(f["homepage"].strip())
        topic_ctr.update([t for t in f["topics"] if t])
        lang_ctr.update([l for l in f["languages"] if l])
        if f["primary_language"]:
            primary_ctr.update([f["primary_language"]])
        if f["readme"]:
            excerpt = f["readme"].strip()
            # 2K per-repo excerpt to keep the total bounded
            parts.append(f"### {owner}/{repo}\n{excerpt[:2000]}")

    # Choose description/homepage by frequency; if many distinct descriptions, join a few
    description = None
    if descs:
        desc_counts = Counter(descs).most_common()
        description = desc_counts[0][0]
        if len(desc_counts) > 1:
            uniq = []
            seen = set()
            for d, _ in desc_counts:
                if d not in seen:
                    seen.add(d)
                    uniq.append(d)
                if len(" | ".join(uniq)) > 500:
                    break
            description = " | ".join(uniq)

    homepage = Counter(homes).most_common(1)[0][0] if homes else None
    topics = [t for t, _ in topic_ctr.most_common(50)]
    languages = [l for l, _ in lang_ctr.most_common(50)]
    primary_language = primary_ctr.most_common(1)[0][0] if primary_ctr else None
    readme = (("\n\n").join(parts)[:readme_chars]) if parts else None

    return {
        "description": description,
        "homepage": homepage,
        "topics": topics,
        "primary_language": primary_language,
        "languages": languages,
        "readme": readme,
    }


# ----------- Per-project summarization -----------


def summarize_project(
    project_id: str, project_name: str | None, per_project_dfs: dict[str, pd.DataFrame]
) -> dict:
    """
    Build a single JSON-serializable summary for ONE project_id from its sliced tables.
    Expects per_project_dfs to contain DataFrames for:
      commits, issues, pull_requests, stargazers, forks, releases, pr_files
    (Any may be empty DataFrames.)
    """
    # Pull per-table slices with defaults
    commits = per_project_dfs.get("commits", pd.DataFrame())
    issues = per_project_dfs.get("issues", pd.DataFrame())
    prs = per_project_dfs.get("pull_requests", pd.DataFrame())
    stars = per_project_dfs.get("stargazers", pd.DataFrame())
    forks = per_project_dfs.get("forks", pd.DataFrame())
    releases = per_project_dfs.get("releases", pd.DataFrame())
    pr_files = per_project_dfs.get("pr_files", pd.DataFrame())

    # Consider project "active" if any table has ≥1 row (forks are weak activity but still signal)
    active = any(len(df) > 0 for df in [commits, issues, prs, releases, stars, forks])

    # Aggregate intent tags across commits/issues/PRs/releases to top themes
    theme_tags = explode_tags(
        pd.concat(
            [
                commits.get("intent_tags", pd.Series(dtype=str)),
                issues.get("intent_tags", pd.Series(dtype=str)),
                prs.get("intent_tags", pd.Series(dtype=str)),
                releases.get("intent_tags", pd.Series(dtype=str)),
            ],
            ignore_index=True,
        )
    )
    themes = top_n(theme_tags, 6)

    # Build example lists (recent first, robust datetime sort)
    # Include owner/repo so higher-level summaries can balance across repos
    release_examples = collect_examples(
        releases,
        ["owner", "repo", "published_at", "release_name", "release_tag", "release_url"],
        n=5,
        sort_col="published_at",
    )
    commit_examples = collect_examples(
        commits,
        [
            "owner",
            "repo",
            "committed_at",
            "author_login",
            "message_headline",
            "commit_url",
        ],
        n=5,
        sort_col="committed_at",
    )
    pr_examples = collect_examples(
        prs,
        ["owner", "repo", "created_at", "author_login", "title", "pr_url", "state"],
        n=5,
        sort_col="created_at",
    )
    issue_examples = collect_examples(
        issues,
        ["owner", "repo", "created_at", "author_login", "title", "issue_url", "labels"],
        n=5,
        sort_col="created_at",
    )

    # “Areas touched” = frequent top-level directories and/or file extensions from PR files
    areas = []
    if (
        isinstance(pr_files, pd.DataFrame)
        and not pr_files.empty
        and "path" in pr_files.columns
    ):
        paths = pr_files["path"].dropna().astype(str).tolist()
        top_dirs = [
            p.split("/")[0] for p in paths if "/" in p
        ]  # e.g., 'api', 'src', 'docs'
        exts = [
            p.rsplit(".", 1)[-1] for p in paths if "." in p
        ]  # e.g., 'py', 'md', 'yaml'
        areas = top_n(top_dirs + exts, 10)

    # Contributors = commit authors + PR authors (by login)
    commit_authors = commits.get("author_login", pd.Series(dtype=str)).dropna().tolist()
    pr_authors = prs.get("author_login", pd.Series(dtype=str)).dropna().tolist()
    contributors = top_n(commit_authors + pr_authors, 8)

    # Issue filers + issue label themes
    issue_authors = issues.get("author_login", pd.Series(dtype=str)).dropna().tolist()
    issue_labels = []
    if "labels" in issues.columns:
        for lbls in issues["labels"].dropna():
            issue_labels.extend([x.strip() for x in str(lbls).split(",") if x.strip()])
    issue_themes = top_n(
        issue_labels + theme_tags, 8
    )  # fuse label tokens with heuristic tags
    issue_filers = top_n(issue_authors, 8)

    # “Interest” signals (who starred, who forked)
    stargazers = top_n(
        stars.get("stargazer_login", pd.Series(dtype=str)).dropna().tolist(), 8
    )
    fork_owners = top_n(
        forks.get("fork_owner_login", pd.Series(dtype=str)).dropna().tolist(), 8
    )

    # Context multi-repo aware
    seed_slice = globals().get("_SEED_BY_PID", {}).get(project_id, pd.DataFrame())
    repo_ctx = build_repo_context_all(per_project_dfs, seed_slice=seed_slice)

    # Build the dict to return (JSON-serializable)
    return {
        "project_id": project_id,
        "project_name": project_name,
        "active_in_window": bool(active),
        "repo_context": repo_ctx,  # representative repo (not arbitrary first table)
        "areas_touched": areas,  # [{value:'api', count:7}, {value:'py', count:5}, …]
        "themes": themes,  # heuristic tags aggregated
        "contributors": contributors,  # top commit/PR authors
        "issue_filers": issue_filers,  # top issue creators
        "issue_themes": issue_themes,  # label tokens + intent tags
        "recent_examples": {  # concrete, clickable traceability
            "commits": commit_examples,
            "pull_requests": pr_examples,
            "issues": issue_examples,
            "releases": release_examples,
        },
        "interest_signals": {
            "stargazers": stargazers,
            "fork_owners": fork_owners,
        },
        "notes": "Heuristic tags; examples sampled from the window. Repo context chosen by activity+README heuristic.",
    }


# ----------- Main orchestration -----------


def main():
    # Load ALL-REPO combined tables (may be empty if no rows were written for that table)
    commits = load_or_empty(os.path.join(CLEAN_DIR, "_all_commits.parquet"))
    issues = load_or_empty(os.path.join(CLEAN_DIR, "_all_issues.parquet"))
    prs = load_or_empty(os.path.join(CLEAN_DIR, "_all_pull_requests.parquet"))
    stars = load_or_empty(os.path.join(CLEAN_DIR, "_all_stargazers.parquet"))
    forks = load_or_empty(os.path.join(CLEAN_DIR, "_all_forks.parquet"))
    releases = load_or_empty(os.path.join(CLEAN_DIR, "_all_releases.parquet"))
    pr_files = load_or_empty(os.path.join(CLEAN_DIR, "_all_pr_files.parquet"))

    # Normalize key string columns if present (helps avoid dtype mismatches)
    for df in [commits, issues, prs, releases, pr_files, stars, forks]:
        if isinstance(df, pd.DataFrame) and not df.empty:
            if "project_id" in df.columns:
                df["project_id"] = df["project_id"].astype("string")
            if "project_name" in df.columns:
                df["project_name"] = df["project_name"].astype("string")

    # Determine which project_ids exist anywhere across the tables
    # project ids found in tables
    projects_found = set(
        pd.concat(
            [
                commits.get("project_id", pd.Series(dtype="string")),
                issues.get("project_id", pd.Series(dtype="string")),
                prs.get("project_id", pd.Series(dtype="string")),
                releases.get("project_id", pd.Series(dtype="string")),
                pr_files.get("project_id", pd.Series(dtype="string")),
                stars.get("project_id", pd.Series(dtype="string")),
                forks.get("project_id", pd.Series(dtype="string")),
            ],
            ignore_index=True,
        )
        .dropna()
        .unique()
    )

    # load seed and union with found projects
    seed_df = load_seed(SEED_CSV)
    seed_df["project_id"] = seed_df["project_id"].astype("string")
    seed_df["project_name"] = seed_df["project_name"].astype("string")
    projects_seed = set(seed_df["project_id"].dropna().unique())

    # expose a handy index for summarize_project()
    global _SEED_BY_PID
    _SEED_BY_PID = {pid: seed_df[seed_df["project_id"] == pid] for pid in projects_seed}

    projects = sorted(projects_found | projects_seed)

    # We'll accumulate machine-readable summaries here so summarize_portfolio.py can read them
    portfolio = {"projects": [], "generated_from": CLEAN_DIR}

    for pid in projects:
        # Try to recover a human-friendly project_name from any table that has it for this pid
        pname = None
        for df in (commits, issues, prs, releases, pr_files, stars, forks):
            if (
                isinstance(df, pd.DataFrame)
                and "project_id" in df.columns
                and "project_name" in df.columns
            ):
                vals = df[df["project_id"] == pid]["project_name"].dropna().unique()
                if len(vals):
                    pname = vals[0]
                    break

        # Slice each table down to this project id (or keep the empty DataFrame)
        per_project_dfs = {
            "commits": (
                commits[commits["project_id"] == pid] if not commits.empty else commits
            ),
            "issues": (
                issues[issues["project_id"] == pid] if not issues.empty else issues
            ),
            "pull_requests": prs[prs["project_id"] == pid] if not prs.empty else prs,
            "releases": (
                releases[releases["project_id"] == pid]
                if not releases.empty
                else releases
            ),
            "pr_files": (
                pr_files[pr_files["project_id"] == pid]
                if not pr_files.empty
                else pr_files
            ),
            "stargazers": (
                stars[stars["project_id"] == pid] if not stars.empty else stars
            ),
            "forks": forks[forks["project_id"] == pid] if not forks.empty else forks,
        }

        # Compute distinct repos (and reuse to form a stable list)
        seed_slice = _SEED_BY_PID.get(pid, pd.DataFrame())
        pairs = _iter_project_repos(per_project_dfs, seed_slice=seed_slice)
        repo_count = len(pairs)

        # Produce the actual summary payload
        summary = summarize_project(pid, pname, per_project_dfs)
        summary["repo_count"] = repo_count
        summary["repos"] = [{"owner": o, "repo": r} for (o, r) in pairs]

        # Write per-project machine JSON (consumed by summarize_projects.py and summarize_portfolio.py)
        out_json = os.path.join(OUT_DIR, f"{pid}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Wrote {out_json}")

        # Also write a small people CSV to quickly see top participants
        people_rows = []
        for row in summary["contributors"]:
            people_rows.append(
                {
                    "project_id": pid,
                    "role": "contributor",
                    "login": row["value"],
                    "count": row["count"],
                }
            )
        for row in summary["issue_filers"]:
            people_rows.append(
                {
                    "project_id": pid,
                    "role": "issue_filer",
                    "login": row["value"],
                    "count": row["count"],
                }
            )
        if people_rows:
            pd.DataFrame(people_rows).to_csv(
                os.path.join(OUT_DIR, f"{pid}__people.csv"), index=False
            )

        # Add to the machine portfolio index
        portfolio["projects"].append(summary)

    # Write a simple portfolio JSON index listing all project summaries
    # (summarize_portfolio.py will turn THIS into a narrative report)
    with open(os.path.join(OUT_DIR, "_portfolio.json"), "w", encoding="utf-8") as f:
        json.dump(portfolio, f, indent=2, ensure_ascii=False)
    print(f"Wrote {os.path.join(OUT_DIR, '_portfolio.json')}")


if __name__ == "__main__":
    main()
