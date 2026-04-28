#!/usr/bin/env python3
"""
Build a human-readable _activity_context.md file for a single repo,
reading from the same cleaned parquet files used by the chat-based pipeline.
This file is injected into the cloned repo before Copilot runs, giving it
the same activity data the chat-based pipeline has.
"""

import os, argparse, pathlib
import pandas as pd
from datetime import datetime, timezone


CLEAN_DIR = "data/clean"


def _load_table(name: str) -> pd.DataFrame:
    path = os.path.join(CLEAN_DIR, f"_all_{name}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


def _sorted_desc(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce", utc=True)
    return out.sort_values(col, ascending=False, kind="stable")


def _filter(df: pd.DataFrame, owner: str, repo: str) -> pd.DataFrame:
    if df.empty:
        return df
    if "owner" in df.columns and "repo" in df.columns:
        return df[(df["owner"].astype(str) == owner) & (df["repo"].astype(str) == repo)]
    return pd.DataFrame()


def _safe(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def build_activity_context(
    project_id: str,
    project_name: str,
    owner: str,
    repo: str,
    window_label: str,
    max_items: int = 20,
) -> str:
    """
    Build the full markdown context string for one repo.
    """
    commits   = _filter(_sorted_desc(_load_table("commits"),       "committed_at"),  owner, repo)
    prs       = _filter(_sorted_desc(_load_table("pull_requests"),  "created_at"),    owner, repo)
    issues    = _filter(_sorted_desc(_load_table("issues"),         "created_at"),    owner, repo)
    releases  = _filter(_sorted_desc(_load_table("releases"),       "published_at"),  owner, repo)
    stars     = _filter(_sorted_desc(_load_table("stargazers"),     "starred_at"),    owner, repo)
    forks     = _filter(_sorted_desc(_load_table("forks"),          "fork_created_at"), owner, repo)

    lines = []
    lines.append(f"# Activity Context: {owner}/{repo}")
    lines.append(f"")
    lines.append(f"**Project:** {project_name} (ID: {project_id})")
    lines.append(f"**Repository:** {owner}/{repo}")
    lines.append(f"**Window:** {window_label}")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
    lines.append(f"")
    lines.append(
        "This file contains structured GitHub activity data for this repository. "
        "Use it as the primary evidence source for the Recent Developments section."
    )
    lines.append("")

    # ---- Commits ----
    lines.append("## Commits")
    if commits.empty:
        lines.append("No commits in this window.")
    else:
        for _, r in commits.head(max_items).iterrows():
            url     = _safe(r.get("commit_url"))
            msg     = _safe(r.get("message_headline"))
            author  = _safe(r.get("author_login"))
            date    = _safe(r.get("committed_at"))[:10]
            lines.append(f"- [{msg}]({url}) by {author} on {date}" if url else f"- {msg} by {author} on {date}")
    lines.append("")

    # ---- Pull Requests ----
    lines.append("## Pull Requests")
    if prs.empty:
        lines.append("No pull requests in this window.")
    else:
        for _, r in prs.head(max_items).iterrows():
            url    = _safe(r.get("pr_url"))
            title  = _safe(r.get("title"))
            author = _safe(r.get("author_login"))
            state  = _safe(r.get("state"))
            date   = _safe(r.get("created_at"))[:10]
            lines.append(f"- [{title}]({url}) by {author} ({state}) on {date}" if url else f"- {title} by {author} ({state}) on {date}")
    lines.append("")

    # ---- Issues ----
    lines.append("## Issues")
    if issues.empty:
        lines.append("No issues in this window.")
    else:
        for _, r in issues.head(max_items).iterrows():
            url    = _safe(r.get("issue_url"))
            title  = _safe(r.get("title"))
            author = _safe(r.get("author_login"))
            state  = _safe(r.get("state"))
            date   = _safe(r.get("created_at"))[:10]
            lines.append(f"- [{title}]({url}) by {author} ({state}) on {date}" if url else f"- {title} by {author} ({state}) on {date}")
    lines.append("")

    # ---- Releases ----
    lines.append("## Releases")
    if releases.empty:
        lines.append("No releases in this window.")
    else:
        for _, r in releases.head(max_items).iterrows():
            url  = _safe(r.get("release_url"))
            name = _safe(r.get("release_name")) or _safe(r.get("release_tag"))
            date = _safe(r.get("published_at"))[:10]
            lines.append(f"- [{name}]({url}) on {date}" if url else f"- {name} on {date}")
    lines.append("")

    # ---- Stargazers (identity signals) ----
    lines.append("## Stargazers (identity signals)")
    if stars.empty:
        lines.append("No stargazer data available.")
    else:
        for _, r in stars.head(10).iterrows():
            login = _safe(r.get("stargazer_login"))
            name  = _safe(r.get("stargazer_name"))
            comp  = _safe(r.get("stargazer_company"))
            loc   = _safe(r.get("stargazer_location"))
            parts = [login]
            if name:  parts.append(name)
            if comp:  parts.append(comp)
            if loc:   parts.append(loc)
            lines.append(f"- {' | '.join(parts)}")
    lines.append("")

    # ---- Fork owners (identity signals) ----
    lines.append("## Fork Owners (identity signals)")
    if forks.empty:
        lines.append("No fork data available.")
    else:
        for _, r in forks.head(10).iterrows():
            login = _safe(r.get("fork_owner_login"))
            name  = _safe(r.get("fork_owner_name"))
            typ   = _safe(r.get("fork_owner_type"))
            loc   = _safe(r.get("fork_owner_location"))
            parts = [login]
            if name: parts.append(name)
            if typ:  parts.append(typ)
            if loc:  parts.append(loc)
            lines.append(f"- {' | '.join(parts)}")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Build _activity_context.md for one repo from cleaned parquet data."
    )
    parser.add_argument("--project-id",   required=True)
    parser.add_argument("--project-name", default="")
    parser.add_argument("--owner",        required=True)
    parser.add_argument("--repo",         required=True)
    parser.add_argument("--window-label", default="last year")
    parser.add_argument("--out",          required=True, help="Path to write the .md file")
    args = parser.parse_args()

    content = build_activity_context(
        project_id=args.project_id,
        project_name=args.project_name,
        owner=args.owner,
        repo=args.repo,
        window_label=args.window_label,
    )

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()