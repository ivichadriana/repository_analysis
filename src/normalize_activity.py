# src/normalize_activity.py
"""
Normalize raw GitHub repository JSON data into tidy Parquet tables.
"""

import os
import json
import glob
import re
from collections import defaultdict
import pandas as pd

# ----------------------------- Paths -----------------------------

SEED_CSV = "data/projects_seed.csv"
RAW_DIR = "data/raw/github"
CLEAN_DIR = "data/clean"

# ---------------------- Intent Tag Heuristics --------------------
# These are simple regexes applied to commit/issue/PR text to infer themes.
INTENT_PATTERNS = {
    "feature": r"\b(feat|feature|add(ed)?|introduc(e|ed|ing)|implement(ed)?|support(ed)?)\b",
    "fix": r"\b(fix(ed|es)?|bug|issue|resolve(d|s)?|patch)\b",
    "docs": r"\b(doc(s|umentation)?|readme|typo|spelling|guide|tutorial)\b",
    "refactor": r"\b(refactor(ed|ing)?|cleanup|restructure|reorganize)\b",
    "test": r"\b(test(s|ing)?|unit[-\s]?test|pytest|coverage)\b",
    "infra": r"\b(ci|build|deploy|pipeline|github[-\s]?actions|action|docker|compose|helm|k8s|kubernetes)\b",
    "deps": r"\b(dep(s|endenc(y|ies))?|bump|upgrade|update package|requirements\.txt|poetry\.lock|package\.json|pipfile\.lock)\b",
}

# ----------------------- Small Helpers --------------------------


def _readme_text(repo_json: dict) -> str | None:
    """
    Return README text. Prefer the convenience field __readme_text
    (added by fetch_github_activity.py), else try the raw GraphQL aliases.
    """
    # 1) Best: convenience field already picked by fetch script
    t = repo_json.get("__readme_text")
    if isinstance(t, str) and t.strip():
        return t

    # 2) Fallback: look through the GraphQL objects we requested
    for k in [
        "readmeMd",
        "readmeCapMd",
        "readmeRst",
        "readmeTxt",
        "docsReadmeMd",
        "docsReadmeRst",
    ]:
        node = repo_json.get(k)
        if isinstance(node, dict):
            txt = node.get("text")
            if isinstance(txt, str) and txt.strip():
                return txt

    return None


def _topics_list(repo_json: dict) -> list[str]:
    """Extract repo topics as a simple list of strings."""
    try:
        nodes = (repo_json.get("repositoryTopics") or {}).get("nodes", []) or []
        return [
            n["topic"]["name"]
            for n in nodes
            if n and n.get("topic") and n["topic"].get("name")
        ]
    except Exception:
        return []


def _lang_list(repo_json: dict) -> list[str]:
    """Extract repo languages as a simple list of strings (descending by size as fetched)."""
    try:
        nodes = (repo_json.get("languages") or {}).get("nodes", []) or []
        return [n.get("name") for n in nodes if n and n.get("name")]
    except Exception:
        return []


def tag_text(text: str) -> list[str]:
    """
    Return a list of heuristic tags (e.g., ["fix","test"]) found in a piece of text.
    We run each regex case-insensitively against title/body/commit message strings.
    """
    if not text:
        return []
    tags = []
    for name, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            tags.append(name)
    return tags


def load_seed(seed_csv: str = SEED_CSV) -> pd.DataFrame:
    """
    Read the seed CSV. Ensure at least {owner, repo} columns exist.
    If project_id/project_name are missing, create empty columns so downstream code
    can rely on their presence.
    """
    df = pd.read_csv(seed_csv)
    required = {"owner", "repo"}
    if not required.issubset(df.columns):
        raise ValueError(f"{seed_csv} must include columns: {sorted(required)}")
    if "project_id" not in df.columns:
        df["project_id"] = None
    if "project_name" not in df.columns:
        df["project_name"] = None
    return df


def read_repo_json(path: str) -> dict:
    """Load one raw JSON file (produced by fetch_github_activity.py)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------- Normalization Core ----------------------


def normalize(
    repo_json: dict, owner: str, repo: str, project_id, project_name
) -> dict[str, pd.DataFrame]:
    """
    Convert one repository JSON blob into tidy tables.
    Returns a dict of DataFrames keyed by table name:
      commits, issues, pull_requests, pr_files, releases, stargazers, forks
    """

    # Provenance / window (propagate onto every row if present)
    window = repo_json.get("__window") or {}
    window_since = window.get("since")
    window_until = window.get("until")
    fetched_at = repo_json.get("__fetched_at")
    name_with_owner = repo_json.get("__name_with_owner")

    # Base columns we want on EVERY row (makes downstream grouping easier)
    base = {
        "owner": owner,
        "repo": repo,
        "project_id": project_id,
        "project_name": project_name,
        "window_since": window_since,
        "window_until": window_until,
        "fetched_at": fetched_at,
        "name_with_owner": name_with_owner,
    }

    # ---------------- Repo-level context (copied onto every row) ----------------
    repo_description = repo_json.get("description")
    repo_homepage = repo_json.get("homepageUrl")
    repo_topics = _topics_list(repo_json)
    repo_primary_language = (repo_json.get("primaryLanguage") or {}).get("name")
    repo_languages = _lang_list(repo_json)
    repo_readme_text = _readme_text(repo_json)

    # Attach repo context to the base dict; each row inherits these fields
    base.update(
        {
            "repo_description": repo_description,
            "repo_homepage": repo_homepage,
            "repo_topics": ",".join(repo_topics) if repo_topics else None,
            "repo_primary_language": repo_primary_language,
            "repo_languages": ",".join(repo_languages) if repo_languages else None,
            "readme_text": repo_readme_text,  # used later for “Goal” in LLM prompts
        }
    )

    # ---------------- Commits ----------------
    # Navigate to defaultBranchRef.target.history.nodes safely
    commits_nodes = []
    try:
        commits_nodes = (
            repo_json["defaultBranchRef"]["target"]["history"]["nodes"] or []
        )
    except Exception:
        commits_nodes = []

    commit_rows = []
    for n in commits_nodes:
        # Build a text blob for intent tagging
        msg = f"{n.get('messageHeadline') or ''}\n{n.get('message') or ''}"

        # author.user can be None; handle both present/missing user objects robustly
        user_obj = (n.get("author") or {}).get("user") or {}
        author_login = user_obj.get("login")

        commit_rows.append(
            {
                **base,
                "commit_oid": n.get("oid"),
                "committed_at": n.get("committedDate"),
                "message_headline": n.get("messageHeadline"),
                "message": n.get("message"),
                "author_login": author_login,
                "author_name": (n.get("author") or {}).get("name"),
                "author_email": (n.get("author") or {}).get("email"),
                "additions": n.get("additions"),
                "deletions": n.get("deletions"),
                "associated_pr_numbers": ",".join(
                    str(pr.get("number"))
                    for pr in (n.get("associatedPullRequests") or {}).get("nodes", [])
                    if pr
                ),
                "commit_url": n.get("url"),
                "intent_tags": "|".join(tag_text(msg)),
            }
        )
    commits_df = pd.DataFrame(commit_rows)

    # ---------------- Issues ----------------
    issues_nodes = (repo_json.get("issues") or {}).get("nodes", []) or []
    issue_rows = []
    for n in issues_nodes:
        title = n.get("title") or ""
        body = n.get("bodyText") or ""
        labels = [
            lab.get("name") for lab in (n.get("labels") or {}).get("nodes", []) if lab
        ]
        issue_rows.append(
            {
                **base,
                "issue_number": n.get("number"),
                "title": title,
                "body": body,
                "labels": ",".join(labels) if labels else None,
                "author_login": (n.get("author") or {}).get("login"),
                "created_at": n.get("createdAt"),
                "closed_at": n.get("closedAt"),
                "issue_url": n.get("url"),
                "intent_tags": "|".join(tag_text(f"{title}\n{body}")),
            }
        )
    issues_df = pd.DataFrame(issue_rows)

    # ---------------- Pull Requests ----------------
    pr_nodes = (repo_json.get("pullRequests") or {}).get("nodes", []) or []
    pr_rows = []
    for n in pr_nodes:
        title = n.get("title") or ""
        body = n.get("bodyText") or ""
        labels = [
            lab.get("name") for lab in (n.get("labels") or {}).get("nodes", []) if lab
        ]
        pr_rows.append(
            {
                **base,
                "pr_number": n.get("number"),
                "title": title,
                "body": body,
                "labels": ",".join(labels) if labels else None,
                "author_login": (n.get("author") or {}).get("login"),
                "state": n.get("state"),
                "created_at": n.get("createdAt"),
                "merged_at": n.get("mergedAt"),
                "closed_at": n.get("closedAt"),
                "pr_url": n.get("url"),
                "intent_tags": "|".join(tag_text(f"{title}\n{body}")),
            }
        )
    prs_df = pd.DataFrame(pr_rows)

    # ---------------- PR Files (areas touched) ----------------
    # We flatten file paths per PR so we can later infer "areas touched" (top dirs, file types).
    pr_file_rows = []
    for n in pr_nodes:
        pr_num = n.get("number")
        files_nodes = (n.get("files") or {}).get("nodes", []) or []
        for f in files_nodes:
            pr_file_rows.append(
                {
                    **base,
                    "pr_number": pr_num,
                    "path": f.get("path"),
                    "additions": f.get("additions"),
                    "deletions": f.get("deletions"),
                }
            )
    pr_files_df = pd.DataFrame(pr_file_rows)

    # ---------------- Releases ----------------
    rel_nodes = (repo_json.get("releases") or {}).get("nodes", []) or []
    rel_rows = []
    for n in rel_nodes:
        title = n.get("name") or n.get("tagName") or ""
        desc = n.get("description") or ""
        rel_rows.append(
            {
                **base,
                "release_name": n.get("name"),
                "release_tag": n.get("tagName"),
                "published_at": n.get("publishedAt"),
                "release_url": n.get("url"),
                "description": desc,
                "description_html": n.get("descriptionHTML"),
                "intent_tags": "|".join(tag_text(f"{title}\n{desc}")),
            }
        )
    releases_df = pd.DataFrame(rel_rows)

    # ---------------- Stargazers ----------------
    # We keep identity context (name/company/location/orgs) when available.
    star_edges = (repo_json.get("stargazers") or {}).get("edges", []) or []
    star_rows = []
    for e in star_edges:
        node = (e or {}).get("node") or {}
        # organizations is only present for Users in our query
        org_nodes = (
            (node.get("organizations") or {}).get("nodes", [])
            if "organizations" in node
            else []
        ) or []
        org_pairs = []
        for org in org_nodes:
            if not org:
                continue
            login = org.get("login")
            name = org.get("name")
            if login or name:
                # store compactly: "orgLogin:Org Name"
                org_pairs.append(":".join([x for x in [login, name] if x]))

        star_rows.append(
            {
                **base,
                "starred_at": e.get("starredAt"),
                "stargazer_login": node.get("login"),
                "stargazer_name": node.get("name"),
                "stargazer_company": node.get("company"),
                "stargazer_location": node.get("location"),
                "stargazer_orgs": ",".join(org_pairs) if org_pairs else None,
            }
        )
    stargazers_df = pd.DataFrame(star_rows)

    # ---------------- Forks ----------------
    # Owner may be a User or an Organization; we normalize both shapes.
    fork_nodes = (repo_json.get("forks") or {}).get("nodes", []) or []
    fork_rows = []
    for n in fork_nodes:
        owner_obj = n.get("owner") or {}
        owner_type = owner_obj.get("__typename")  # "User" or "Organization"
        fork_rows.append(
            {
                **base,
                "fork_name_with_owner": n.get("nameWithOwner"),
                "fork_owner_login": owner_obj.get("login"),
                "fork_owner_type": owner_type,
                "fork_owner_name": owner_obj.get("name"),
                "fork_owner_company": (
                    owner_obj.get("company") if owner_type == "User" else None
                ),
                "fork_owner_location": owner_obj.get("location"),
                "fork_owner_org_description": (
                    owner_obj.get("description")
                    if owner_type == "Organization"
                    else None
                ),
                "fork_created_at": n.get("createdAt"),
            }
        )
    forks_df = pd.DataFrame(fork_rows)

    # Return all tidy tables for this repo
    return {
        "commits": commits_df,
        "issues": issues_df,
        "pull_requests": prs_df,
        "pr_files": pr_files_df,
        "releases": releases_df,
        "stargazers": stargazers_df,
        "forks": forks_df,
    }


# ------------------------------ Main ----------------------------------


def main():
    # Ensure output dir exists
    os.makedirs(CLEAN_DIR, exist_ok=True)

    # Load the seed to attach project metadata to each row
    seed = load_seed()
    # Build a lookup: (owner, repo) -> {"project_id":..., "project_name":...}
    meta = seed.set_index(["owner", "repo"])[["project_id", "project_name"]].to_dict(
        "index"
    )

    # Where are the raw repo JSONs?
    json_paths = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    if not json_paths:
        print(f"No JSON files found in {RAW_DIR}. Run fetch_github_activity.py first.")
        return

    # We’ll collect per-table pieces here to concatenate later
    all_tables: dict[str, list[pd.DataFrame]] = defaultdict(list)

    for path in json_paths:
        filename = os.path.basename(path)  # pattern: owner__repo.json
        try:
            owner, repo_with_ext = filename.split("__", 1)
            repo = repo_with_ext.replace(".json", "")
        except ValueError:
            print(f"Skipping unexpected filename format: {filename}")
            continue

        # Attach project metadata if we have it in the seed
        pj = meta.get((owner, repo), {"project_id": None, "project_name": None})

        repo_json = read_repo_json(path)
        tables = normalize(repo_json, owner, repo, pj["project_id"], pj["project_name"])

        # Write per-repo Parquet for each non-empty table, and collect for the combined files
        for name, df in tables.items():
            if df.empty:
                print(f"[{owner}/{repo}] {name}: 0 rows")
                continue
            out_path = os.path.join(CLEAN_DIR, f"{owner}__{repo}__{name}.parquet")
            df.to_parquet(out_path, index=False)
            print(f"Saved {name}: {out_path}")
            all_tables[name].append(df)

    # After all repos: write combined “_all_*.parquet” tables
    for name, parts in all_tables.items():
        if not parts:
            continue
        combined = pd.concat(parts, ignore_index=True)
        out_path = os.path.join(CLEAN_DIR, f"_all_{name}.parquet")
        combined.to_parquet(out_path, index=False)
        print(f"Saved combined {name}: {out_path}")


if __name__ == "__main__":
    main()
