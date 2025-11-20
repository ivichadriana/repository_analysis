#!/usr/bin/env python3
"""
Build projects_seed.csv from a private GitHub repo over SSH (no tokens).

What it does:
- Sparse-clones only the needed directory from nih-cfde/icc-eval-core-private
  using SSH (git@github.com:...).
- Reads:
    data/output/core-projects.json
    and a second repos JSON in data/output/ (auto-detected by keys)
- Keeps only "active" projects (repos > 0 or analytics > 0).
- Emits CSV: project_id,project_name,owner,repo,repo_url
- Cleans up the temporary clone directory automatically.

"""

from __future__ import annotations
import argparse
import csv
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

OWNER_REPO = "nih-cfde/icc-eval-core-private"
BRANCH = "main"
SUBDIR = "data/output"
CORE_PROJECTS_FILENAME = "core-projects.json"


def run_git(args: List[str], cwd: Path | None = None) -> None:
    cmd = ["git"] + args
    try:
        subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)
    except subprocess.CalledProcessError as e:
        pretty = " ".join(shlex.quote(x) for x in args)
        print(f"git command failed: git {pretty}\n{e}", file=sys.stderr)
        sys.exit(1)


def find_repos_json(dir_path: Path) -> Path:
    """
    Find the repos JSON file by looking for an array of objects whose items include
    keys {"coreProject","owner","name"}.
    """
    for p in sorted(dir_path.glob("*.json")):
        if p.name == CORE_PROJECTS_FILENAME:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict) and {
                    "coreProject",
                    "owner",
                    "name",
                }.issubset(first.keys()):
                    return p
        except Exception:
            continue
    sys.exit(
        f"ERROR: Could not find repos JSON in {dir_path} with keys coreProject/owner/name."
    )


def filter_active_projects(core_projects: List[dict]) -> Dict[str, str]:
    """Return {project_id: project_name} for rows where repos>0 or analytics>0."""
    active = {}
    for row in core_projects:
        try:
            pid = row["id"]
            pname = row["name"]
            repos = int(row.get("repos", 0))
            analytics = int(row.get("analytics", 0))
        except Exception:
            continue
        if repos > 0 or analytics > 0:
            active[pid] = pname
    return active


def build_rows(
    active_map: Dict[str, str], repos_list: List[dict]
) -> List[Tuple[str, str, str, str, str]]:
    """
    Build CSV rows: (project_id, project_name, owner, repo, repo_url)
    Only include rows where a repo exists for the active project.
    """
    rows: List[Tuple[str, str, str, str, str]] = []
    for r in repos_list:
        try:
            pid = r["coreProject"]
            owner = r["owner"]
            repo = r["name"]
        except KeyError:
            continue
        if pid not in active_map:
            continue
        pname = active_map[pid]
        url = f"https://github.com/{owner}/{repo}"
        rows.append((pid, pname, owner, repo, url))
    rows.sort(key=lambda x: (x[0], x[2], x[3]))
    return rows


def write_csv(rows: List[Tuple[str, str, str, str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["project_id", "project_name", "owner", "repo", "repo_url"])
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Generate projects_seed.csv from a private GitHub repo (SSH auth)."
    )
    ap.add_argument(
        "--repo", default=OWNER_REPO, help="owner/repo of the private repository"
    )
    ap.add_argument(
        "--branch", default=BRANCH, help="branch to checkout (default: main)"
    )
    ap.add_argument(
        "--subdir",
        default=SUBDIR,
        help="subdirectory to sparse-checkout (contains the JSON files)",
    )
    ap.add_argument(
        "--output", default="data/projects_seed.csv", help="output CSV path"
    )
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="cfde_") as tmpd:
        tmp = Path(tmpd)
        repo_dir = tmp / "repo"

        print("Cloning via SSH (sparse)...")
        run_git(
            [
                "clone",
                "-n",
                "--depth=1",
                "--filter=tree:0",
                f"git@github.com:{args.repo}.git",
                str(repo_dir),
            ]
        )

        print("Setting sparse-checkout and checking out...")
        run_git(
            [
                "-C",
                str(repo_dir),
                "sparse-checkout",
                "set",
                "--no-cone",
                f"/{args.subdir}",
            ]
        )
        # Check out the requested branch (ensures HEAD exists for sparse checkout)
        run_git(["-C", str(repo_dir), "checkout", args.branch])

        data_dir = repo_dir / args.subdir
        core_projects_path = data_dir / CORE_PROJECTS_FILENAME
        if not core_projects_path.exists():
            print(f"ERROR: Missing {CORE_PROJECTS_FILENAME} in {data_dir}", file=sys.stderr)
            sys.exit(1)

        repos_path = find_repos_json(data_dir)

        print(f"Reading {core_projects_path.name} and {repos_path.name} ...")
        core_projects = json.loads(core_projects_path.read_text(encoding="utf-8"))
        repos_list = json.loads(repos_path.read_text(encoding="utf-8"))

        active = filter_active_projects(core_projects)
        rows = build_rows(active, repos_list)

        skipped = set(active.keys()) - {r[0] for r in rows}
        if skipped:
            print(
                f"Note: {len(skipped)} active project(s) had no repo entries and were skipped."
            )

        out_path = Path(args.output).expanduser().resolve()
        write_csv(rows, out_path)
        print(f"Wrote {out_path} with {len(rows)} row(s).")


if __name__ == "__main__":
    main()
