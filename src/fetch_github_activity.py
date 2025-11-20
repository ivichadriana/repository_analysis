# src/fetch_github_activity.py
"""
Fetch the activity from a GitHub repository over a specified time window using the GitHub GraphQL API.
"""

import os
import json
import time
import argparse
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests
from dotenv import load_dotenv

# --------------------------- Config & Helpers ---------------------------

# Load environment variables from .env (so we can read GITHUB_TOKEN)
load_dotenv()

# Required: GitHub token with permission to read the repos you target.
# If missing, raise a friendly error (fail fast).
try:
    GH_TOKEN = os.environ["GITHUB_TOKEN"]
except KeyError as e:
    raise KeyError(
        "GITHUB_TOKEN is missing. Put it in .env (GITHUB_TOKEN=...) or export it in your environment."
    ) from e

# GitHub GraphQL endpoint
GH_GQL = "https://api.github.com/graphql"

# Where to search for failed fetches
FAILED_IDX = "data/raw/github/_failed.json"


def _read_failed_idx() -> list:
    if not os.path.exists(FAILED_IDX):
        return []
    try:
        with open(FAILED_IDX, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        # Corrupt/partial file? Treat as empty.
        return []


def _write_failed_idx(idx: list) -> None:
    os.makedirs(os.path.dirname(FAILED_IDX), exist_ok=True)
    tmp = FAILED_IDX + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)
    os.replace(tmp, FAILED_IDX)  # atomic


# Standard HTTP auth header for GitHub API v4 (GraphQL)
HEADERS = {"Authorization": f"Bearer {GH_TOKEN}"}

# ------------------------------ GraphQL --------------------------------
# Variables meaning:
#  $owner, $name   : which repository (e.g., owner="nih-cfde", name="icc-eval-core")
#  $sinceGit       : (GitTimestamp) lower bound for commit history
#  $untilGit       : (GitTimestamp) upper bound for commit history
#  $sinceDT        : (DateTime)    lower bound for issues.createdAt filter
#
# We also request:
#  - repository metadata (description, homepage, topics, languages)
#  - README via a few common file paths using object(expression: "...") and aliasing
#  - defaultBranchRef -> target ... on Commit -> history(...) nodes (the last 100)
#  - issues created since $sinceDT (last 100) — we apply the upper bound client-side
#  - pullRequests (last 100 by createdAt), we filter to window in Python
#  - PR files (last 100 per PR)
#  - releases (last 50 by createdAt), we filter to window in Python
#  - enriched stargazers/forks identity fields
# Everything until the closing """ is the GraphQL document sent to GitHub.
GQL = """
query RepoActivity(
  $owner:String!,
  $name:String!,
  $sinceGit:GitTimestamp!,
  $untilGit:GitTimestamp!,
  $sinceDT:DateTime!
) {
  # API budget snapshot (helpful for debugging/throttling)
  rateLimit { cost remaining resetAt }

  repository(owner:$owner, name:$name) {
    nameWithOwner
    url

    # ---------------- Repo context (for "Goal") ----------------
    description
    homepageUrl
    repositoryTopics(first: 20) { nodes { topic { name } } }
    primaryLanguage { name }
    languages(first: 10, orderBy: {field: SIZE, direction: DESC}) { nodes { name } }

    # ---------------- README candidates (raw text) --------------
    # We check several common paths and pick the first with non-empty text.
    readmeMd:       object(expression: "HEAD:README.md")        { ... on Blob { text } }
    readmeCapMd:    object(expression: "HEAD:README.MD")        { ... on Blob { text } }
    readmeRst:      object(expression: "HEAD:README.rst")       { ... on Blob { text } }
    readmeTxt:      object(expression: "HEAD:README.txt")       { ... on Blob { text } }
    docsReadmeMd:   object(expression: "HEAD:docs/README.md")   { ... on Blob { text } }
    docsReadmeRst:  object(expression: "HEAD:docs/README.rst")  { ... on Blob { text } }

    # ---------------- Commits in our window --------------------
    defaultBranchRef {
      name
      target {
        ... on Commit {
          history(since:$sinceGit, until:$untilGit, first:100) {
            pageInfo { hasNextPage endCursor }  # ignored for POC (no pagination)
            nodes {
              oid
              committedDate
              messageHeadline
              message
              additions
              deletions
              author { name email user { login } }
              associatedPullRequests(first:20) { nodes { number url mergedAt title } }
              url  # commit permalink (good for evidence block)
            }
          }
        }
      }
    }

    # ---------------- Issues (created since; we clamp to until client-side) -------------------
    issues(first:100, orderBy:{field:CREATED_AT, direction:DESC}, filterBy:{since:$sinceDT}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title bodyText createdAt closedAt url
        author { login }
        labels(first:20) { nodes { name } }
      }
    }

    # ---------------- PRs + files (we clamp by createdAt in Python) ----
    pullRequests(first:100, orderBy:{field:CREATED_AT, direction:DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title bodyText createdAt mergedAt closedAt state url
        author { login }
        labels(first:20) { nodes { name } }
        files(first:100) { nodes { path additions deletions } }
      }
    }

    # ---------------- Releases (we clamp by publishedAt in Python) ----
    releases(first:50, orderBy:{field:CREATED_AT, direction:DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        name
        tagName
        publishedAt
        url
        description
        descriptionHTML
      }
    }

    # ---------------- Stars/Forks with context -----------------
    stargazers(first:100, orderBy:{field:STARRED_AT, direction:DESC}) {
      pageInfo { hasNextPage endCursor }
      edges {
        starredAt
        node {
          __typename
          login
          ... on User {
            name
            company
            location
            organizations(first:5) { nodes { login name } }
          }
        }
      }
    }

    forks(first:100, orderBy:{field:CREATED_AT, direction:DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        nameWithOwner
        createdAt
        owner {
          __typename
          login
          ... on User {
            name
            company
            location
          }
          ... on Organization {
            name
            description
            location
          }
        }
      }
    }
  }
}
"""

# ------------------------- Fetch functions with retries ---------------------------


def _append_failed(owner, repo, err, attempts):
    idx = _read_failed_idx()
    if (owner, repo) not in [(x.get("owner"), x.get("repo")) for x in idx]:
        idx.append({"owner": owner, "repo": repo})
        _write_failed_idx(idx)


def _save_error_stub(owner, repo, err, attempts, outdir):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, f"{owner}__{repo}.json")
    stub = {
        "owner": owner,
        "repo": repo,
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "error": {
            "type": type(err).__name__,
            "message": str(err),
            "attempts": attempts,
        },
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(stub, f, ensure_ascii=False, indent=2)
    _append_failed(owner, repo, err, attempts)


def _mark_success(owner, repo):
    if not os.path.exists(FAILED_IDX):
        return
    idx = [
        x
        for x in _read_failed_idx()
        if not (x.get("owner") == owner and x.get("repo") == repo)
    ]
    if not idx:
        # all failures cleared -> remove the file entirely
        try:
            os.remove(FAILED_IDX)
        except OSError:
            # best effort: fall back to writing an empty list
            _write_failed_idx([])
    else:
        _write_failed_idx(idx)


def _post(payload, max_retries=6):
    """
    POST to GitHub GraphQL with exponential backoff for:
      - 5xx server errors,
      - 403 rate limit messages,
      - transient GraphQL errors present in a 200 response.

    Returns parsed JSON dict (already .json()) or raises after retries.
    """
    backoff = 2  # seconds (exponential up to ~30/60s)
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(GH_GQL, json=payload, headers=HEADERS, timeout=60)

            # Header-based hard rate-limit handling (works for GraphQL/REST)
            rl_rem = r.headers.get("X-RateLimit-Remaining")
            rl_reset = r.headers.get("X-RateLimit-Reset")
            if (
                rl_rem is not None
                and rl_rem.isdigit()
                and int(rl_rem) == 0
                and rl_reset
            ):
                try:
                    reset_epoch = int(rl_reset)
                    sleep_for = max(0, reset_epoch - int(time.time()) + 1)
                    time.sleep(min(max(sleep_for, backoff), 60))
                    backoff = min(backoff * 2, 60)
                    continue
                except Exception:
                    # If parsing the rate limit reset header fails, skip sleeping and continue retrying.
                    pass

            # Retry on server hiccups (e.g., 502/503/504)
            if 500 <= r.status_code < 600:
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue

            # Soft-retry when the text hints at rate limits
            if r.status_code == 403 and "rate limit" in r.text.lower():
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue

            # Raise for non-200/OK after the special cases above
            r.raise_for_status()
            try:
                out = r.json()
            except ValueError:
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue

            # GraphQL-level errors can appear even with HTTP 200
            if "errors" in out:
                msg = json.dumps(out["errors"])
                # Heuristics for transient errors -> retry
                if "Something went wrong" in msg or "timeout" in msg.lower():
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                    continue
                # Otherwise, bubble up the real error
                raise RuntimeError(msg)

            return out

        except requests.RequestException:
            # Network failures/timeouts -> retry
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
            continue

    raise RuntimeError("GitHub GraphQL failed after retries")


# ------------------------------ Utilities ------------------------------
def iso_utc(dt: datetime) -> str:
    """Return an ISO8601 string in UTC with timezone suffix, e.g., '2025-08-20T18:03:00+00:00'."""
    return dt.astimezone(timezone.utc).isoformat()


def _dt(s: str) -> datetime:
    """Parse GitHub ISO strings that may end with 'Z' or explicit '+00:00'."""
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def choose_readme_text(repo_dict: dict) -> str | None:
    """
    Pick the first non-empty README text from the fetched variants.
    IMPORTANT: these names must match the GraphQL aliases above.
    """
    candidates = [
        repo_dict.get("readmeMd"),
        repo_dict.get("readmeCapMd"),  # uppercase extension variant
        repo_dict.get("readmeRst"),
        repo_dict.get("readmeTxt"),
        repo_dict.get("docsReadmeMd"),
        repo_dict.get("docsReadmeRst"),
    ]
    for obj in candidates:
        # Each candidate is either None or a dict like {"text": "..."}
        txt = ""
        if isinstance(obj, dict):
            txt = obj.get("text", "") or ""
        if isinstance(txt, str) and txt.strip():
            return txt
    return None


# --------------------------- Fetch one repo ----------------------------
def fetch_repo(owner, repo, since_iso, until_iso):
    """
    Run the GraphQL query once and return:
      - repository subtree (dict)
      - rate limit snapshot (dict)
    Also clamps PRs, releases, issues, stargazers, and forks to the [since, until] window client-side.
    """
    payload = {
        "query": GQL,
        "variables": {
            "owner": owner,
            "name": repo,
            "sinceGit": since_iso,  # used by commit history
            "untilGit": until_iso,  # used by commit history
            "sinceDT": since_iso,  # used by issues.filterBy.since
        },
    }

    data = _post(payload)
    repo_node = (data.get("data") or {}).get("repository")
    if repo_node is None:
        raise RuntimeError(f"Repository not found or inaccessible: {owner}/{repo}")
    repo_dict = repo_node
    rate = data["data"].get("rateLimit", {})

    # ---- Client-side clamping to the time window (use datetime, not string compares) ----
    since_dt, until_dt = _dt(since_iso), _dt(until_iso)

    # PRs: keep only PRs whose createdAt falls within [since, until]
    pr_nodes = (repo_dict.get("pullRequests") or {}).get("nodes", []) or []
    pr_nodes = [
        n
        for n in pr_nodes
        if n.get("createdAt") and since_dt <= _dt(n["createdAt"]) <= until_dt
    ]
    if "pullRequests" in repo_dict:
        repo_dict["pullRequests"]["nodes"] = pr_nodes

    # Releases: keep only releases whose publishedAt falls within [since, until]
    rel_nodes = (repo_dict.get("releases") or {}).get("nodes", []) or []
    rel_nodes = [
        n
        for n in rel_nodes
        if n.get("publishedAt") and since_dt <= _dt(n["publishedAt"]) <= until_dt
    ]
    if "releases" in repo_dict:
        repo_dict["releases"]["nodes"] = rel_nodes

    # Issues: GraphQL only has lower bound; apply upper bound here
    iss_nodes = (repo_dict.get("issues") or {}).get("nodes", []) or []
    iss_nodes = [
        n
        for n in iss_nodes
        if n.get("createdAt") and since_dt <= _dt(n["createdAt"]) <= until_dt
    ]
    if "issues" in repo_dict:
        repo_dict["issues"]["nodes"] = iss_nodes

    # Stargazers: keep only edges starred within [since, until]
    star_edges = (repo_dict.get("stargazers") or {}).get("edges", []) or []
    star_edges = [
        e
        for e in star_edges
        if e and e.get("starredAt") and since_dt <= _dt(e["starredAt"]) <= until_dt
    ]
    if "stargazers" in repo_dict:
        repo_dict["stargazers"]["edges"] = star_edges

    # Forks: keep only forks created within [since, until]
    fork_nodes = (repo_dict.get("forks") or {}).get("nodes", []) or []
    fork_nodes = [
        n
        for n in fork_nodes
        if n.get("createdAt") and since_dt <= _dt(n["createdAt"]) <= until_dt
    ]
    if "forks" in repo_dict:
        repo_dict["forks"]["nodes"] = fork_nodes

    # Attach convenience field with our chosen README text (if any)
    repo_dict["__readme_text"] = choose_readme_text(repo_dict)

    # Add provenance to help downstream reproducibility/debugging
    repo_dict["__window"] = {"since": since_iso, "until": until_iso}
    repo_dict["__fetched_at"] = iso_utc(datetime.now(timezone.utc))
    repo_dict["__name_with_owner"] = repo_dict.get("nameWithOwner")

    return repo_dict, rate


# ----------------------------- And Run -----------------------------
def run(seed_csv: str, outdir: str, since: datetime, until: datetime, only_list=None):
    """
    Read the seed CSV, optionally filter to a subset of "owner/repo", and
    fetch each repo to a JSON file under outdir.
    """
    os.makedirs(outdir, exist_ok=True)

    # Expect at least 'owner' and 'repo' columns
    df = pd.read_csv(seed_csv)
    needed = {"owner", "repo"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{seed_csv} must contain columns: {sorted(needed)}")

    # If --only was provided, filter down to those repos
    if only_list:
        only = set(only_list)
        df = df[df.apply(lambda r: f"{r['owner']}/{r['repo']}" in only, axis=1)]

    # Compute ISO windows (as strings) once
    since_iso, until_iso = iso_utc(since), iso_utc(until)
    print(f"Time window: {since_iso}  ->  {until_iso}")

    # Iterate rows like a typed namedtuple (owner, repo, possibly other columns)
    for row in df.itertuples(index=False):
        owner = getattr(row, "owner")
        repo = getattr(row, "repo")
        print(f"Fetching: {owner}/{repo} ...")
        try:
            repository, ratelimit = fetch_repo(owner, repo, since_iso, until_iso)
        except Exception as e:
            # Write an error stub + index this repo in _failed.json (so we can retry)
            print(f"  ERROR {owner}/{repo}: {e}")
            _save_error_stub(owner, repo, e, attempts=0, outdir=outdir)
            continue

        # Write success payload
        out_path = os.path.join(outdir, f"{owner}__{repo}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(repository, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {out_path}")

        # Mark success (remove from _failed.json if present)
        _mark_success(owner, repo)

        # Helpful rate-limit trace
        if ratelimit:
            print(
                f"  RateLimit remaining: {ratelimit.get('remaining')} (reset at {ratelimit.get('resetAt')})"
            )


# -------------------------------- CLI ---------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch GitHub activity for a seed list of repos."
    )
    parser.add_argument(
        "--seed",
        default="data/projects_seed.csv",
        help="Path to CSV with columns owner,repo",
    )
    parser.add_argument(
        "--outdir", default="data/raw/github", help="Output directory for JSON files"
    )
    parser.add_argument(
        "--days", type=int, default=90, help="Window size in days (default: 90)"
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="ISO start; overrides --days (e.g., 2025-01-01T00:00:00Z)",
    )
    parser.add_argument(
        "--until", type=str, default=None, help="ISO end; default: now UTC"
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help='Optional list like "owner/repo" to fetch only these',
    )
    parser.add_argument("--retry-failed", action="store_true")

    args = parser.parse_args()

    # Resolve time window from args (favor explicit ISO over relative days)
    now = datetime.now(timezone.utc)
    until = (
        datetime.fromisoformat(args.until.replace("Z", "+00:00")) if args.until else now
    )
    since = (
        datetime.fromisoformat(args.since.replace("Z", "+00:00"))
        if args.since
        else (until - timedelta(days=args.days))
    )
    only_list = args.only
    if args.retry_failed and os.path.exists(FAILED_IDX):
        try:
            failed = _read_failed_idx()
            # overwrite --only with failed set
            only_list = [
                f'{x["owner"]}/{x["repo"]}'
                for x in failed
                if "owner" in x and "repo" in x
            ]
            if only_list:
                print(
                    f"[retry-failed] Retrying {len(only_list)} repo(s) from {FAILED_IDX}"
                )
            else:
                print("[retry-failed] No failed repos recorded.")
        except Exception as e:
            print(f"[retry-failed] Could not read {FAILED_IDX}: {e}")
    # Run the batch
    run(args.seed, args.outdir, since, until, only_list=only_list)
