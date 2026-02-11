# src/goal_from_code.py

"""
Extract the "goal" or purpose of a code repository by analyzing its code/config files.
"""

import pathlib, subprocess, os, textwrap
from typing import Iterable, List, Tuple, Optional
import shutil
import math

BINARY_EXTS = {
    # images & raster
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tiff",
    ".tif",
    ".ico",
    ".svs",
    ".webp",
    # docs/binaries
    ".pdf",
    ".xlsx",
    ".docx",
    ".pptx",
    # archives (single-suffix forms)
    ".zip",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    # 3D / meshes (single-suffix)
    ".fbx",
    ".glb",
    ".gltf",
    ".stl",
    ".ply",
    ".las",
    ".objz",
    ".3ds",
    # med/geo (single-suffix; multi-suffix handled above)
    ".nii",
    ".nrrd",
    ".mhd",
    ".mha",
    ".geotiff",
    # audio/video
    ".mp4",
    ".mp3",
    ".wav",
    ".avi",
    ".mov",
    ".webm",
    ".m4a",
    ".aac",
    ".flac",
    # fonts / wasm
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".wasm",
    # design
    ".psd",
    ".ai",
    ".xcf",
    # db / sqlite
    ".sqlite",
    ".db",
    ".db3",
    # native libs / executables
    ".so",
    ".dylib",
    ".dll",
    ".exe",
    ".bin",
    ".obj",
    # columnar / arrays / hdf
    ".parquet",
    ".feather",
    ".h5",
    ".hdf5",
    ".npz",
    ".npy",
    # ML / notebooks / checkpoints
    ".ipynb",
    ".tfrecord",
    ".pb",
    ".onnx",
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pth",
    ".pkl",
    ".pickle",
    ".joblib",
    # chunked array stores
    ".zarr",
}

# Code/config extensions we will consider for goal synthesis
CODE_EXTS = {
    ".py",
    ".pyi",
    ".r",
    ".rmd",
    ".jl",
    ".m",
    ".c",
    ".h",
    ".hpp",
    ".hxx",
    ".hh",
    ".cc",
    ".cpp",
    ".cxx",
    ".cu",
    ".cuh",
    ".ino",
    ".java",
    ".scala",
    ".kt",
    ".kts",
    ".groovy",
    ".go",
    ".rs",
    ".swift",
    ".php",
    ".rb",
    ".pl",
    ".pm",
    ".t",
    ".lua",
    ".fs",
    ".fsx",
    ".f90",
    ".f95",
    ".f03",
    ".f08",
    ".for",
    ".ftn",
    ".f",
    ".cs",
    ".vb",
    ".vbs",
    ".js",
    ".mjs",
    ".cjs",
    ".jsx",
    ".ts",
    ".tsx",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".properties",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".bat",
    ".cmd",
    ".ps1",
    ".psm1",
    ".psd1",
    ".html",
    ".htm",
    ".xhtml",
    ".xml",
    ".xsl",
    ".xslt",
    ".svg",
    ".sql",
    ".psql",
    ".mysql",
    ".pgsql",
    ".hql",
    ".cmake",
    ".ninja",
    ".bazel",
    ".bzl",
    ".gradle",
    ".mk",
    ".tex",
    ".sty",
    ".cls",
    ".bib",
    ".rst",
    ".md",
    ".markdown",
    ".txt",
    ".proto",
    ".thrift",
    ".avdl",
    ".graphql",
    ".gql",
    ".sol",
    ".asm",
    ".s",
    ".v",
    ".vh",
    ".sv",
    ".svh",
    ".vhdl",
    ".vhd",
    ".dart",
    ".coffee",
    ".erl",
    ".hrl",
    ".ex",
    ".exs",
    ".nim",
    ".clj",
    ".cljs",
    ".edn",
    ".lisp",
    ".el",
    ".scm",
    ".ss",
    ".cr",
    ".mli",
    ".ml",
    ".re",
    ".rei",
    ".hx",
    ".hxml",
    ".wgsl",
    ".metal",
    ".glsl",
    ".vert",
    ".frag",
    ".shader",
}

# Code/config files that often have no extension but are meaningful
CODE_BASENAMES = {
    "Dockerfile",
    "Makefile",
    "CMakeLists.txt",
    "WORKSPACE",
    "BUILD",
    "BUILD.bazel",
    "Gemfile",
    "Rakefile",
    "Procfile",
    ".env",
    ".env.example",
    ".envrc",
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
    "Pipfile",
    "requirements.txt",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "tsconfig.json",
    ".babelrc",
    ".eslintrc",
    ".prettierrc",
    ".prettierignore",
    ".ruff.toml",
    ".flake8",
}

# Common dirs we gnore even if texty (not the project’s purpose):
IGNORE_DIRS = {
    ".git",
    ".github",
    ".gitlab",
    ".svn",
    ".hg",
    "assets",
    "static",
    "public",
    "media",
    "images",
    "img",
    "figures",
    "screenshots",
    "thumbnails",
    "downloads",
    "node_modules",
    "dist",
    "build",
    "out",
    "target",
    "__pycache__",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    ".idea",
    ".vscode",
    ".next",
    ".cache",
    ".parcel-cache",
    "third_party",
    "vendor",
    ".tox",
    ".eggs",
    ".gradle",
    ".nuget",
    "Pods",
    "Packages",
    ".Rproj.user",
    "models",
    "model",
    "checkpoints",
    "artifacts",
    "data",
    "datasets",
    "samples",
    "sample-data",
    "logs",
    "log",
    "tmp",
    "temp",
    ".coverage",
    "coverage",
}

BINARY_MULTI_EXTS = (
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
    ".tar.xz",
    ".txz",
    ".nii.gz",
    ".ome.tif",
    ".ome.tiff",
)
EXT_SIZE_LIMITS = {
    ".json": 2_000_000,  # 2 MB
    ".md": 1_500_000,
    ".xml": 2_000_000,
    ".svg": 800_000,
    ".html": 1_500_000,
}

TEXT_MAX_BYTES_PER_CHUNK = 16384  # per chunk (bytes)
CHUNKS_PER_FILE_CAP = 64  # hard cap per file to avoid huge files
FILES_PER_DIR_CAP = 300  # safety: extremely large dirs will cap at this


def delete_clone_path(p: pathlib.Path):
    """Remove a previously cloned repo folder; ignore if missing."""
    try:
        if p.exists():
            shutil.rmtree(p)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[warn] failed to delete clone at {p}: {e}")


def _run(cmd: list[str], cwd: Optional[pathlib.Path] = None) -> str:
    p = subprocess.run(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stderr.strip()}")
    return p.stdout


def shallow_clone(owner: str, repo: str, dest_root: pathlib.Path) -> pathlib.Path:
    """
    Clone or refresh a shallow checkout of https://github.com/{owner}/{repo}.git
    into dest_root/owner/repo. Returns the repo path.
    """
    url = f"https://github.com/{owner}/{repo}.git"
    dest = dest_root / owner / repo
    dest.parent.mkdir(parents=True, exist_ok=True)
    if (dest / ".git").exists():
        _run(["git", "fetch", "--depth", "1", "origin"], cwd=dest)
        _run(["git", "reset", "--hard", "origin/HEAD"], cwd=dest)
    else:
        _run(["git", "clone", "--depth", "1", "--single-branch", url, str(dest)])
    return dest


def _is_binary_path(p: pathlib.Path) -> bool:
    name_l = p.name.lower()
    if name_l.endswith(BINARY_MULTI_EXTS):
        return True
    if p.suffix.lower() in BINARY_EXTS:
        return True
    try:
        return p.stat().st_size > 2_000_000
    except Exception:
        return True


def _should_skip_dir(p: pathlib.Path) -> bool:
    name = p.name.lower()
    return name in IGNORE_DIRS


def iter_text_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    """
    Yield only code/config/text files under root, skipping obvious binary/vendor dirs.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        # prune ignored dirs
        dirnames[:] = [
            d for d in dirnames if not _should_skip_dir(pathlib.Path(dirpath) / d)
        ]

        # dir-level cap to avoid explosion
        if len(filenames) > FILES_PER_DIR_CAP:
            filenames = sorted(filenames)[:FILES_PER_DIR_CAP]

        for fn in filenames:
            p = pathlib.Path(dirpath) / fn

            # skip files inside .git (belt & suspenders)
            if ".git" in p.parts:
                continue

            # skip binary/huge files first
            if _is_binary_path(p):
                continue

            ext = p.suffix.lower()
            if p.name.startswith(".") and p.name not in CODE_BASENAMES:
                continue
            if ext not in CODE_EXTS and p.name not in CODE_BASENAMES:
                continue
            # cap oversized text-like files that aren’t useful for “goal” synthesis
            name = p.name.lower()
            if name.endswith(".min.js"):
                continue
            if name.endswith(".map"):
                continue
            lim = EXT_SIZE_LIMITS.get(ext)
            if lim is not None:
                try:
                    if p.stat().st_size > lim:
                        continue
                except Exception:
                    pass
            yield p


def chunk_file_bytes(
    p: pathlib.Path,
    max_chunk_bytes: int = TEXT_MAX_BYTES_PER_CHUNK,
    max_chunks: int = CHUNKS_PER_FILE_CAP,
    huge_threshold: Optional[int] = None,
    windows: int = 8,
):
    """
    Yield text chunks from a file with these rules:
      - ≤ max_chunk_bytes: one chunk (whole file)
      - (max_chunk_bytes, huge_threshold]: split into N near-equal chunks (N = ceil(size/max_chunk_bytes), capped)
      - > huge_threshold: sample `windows` spans of size max_chunk_bytes (head/tail/middles)
    """
    if huge_threshold is None:
        huge_threshold = max_chunk_bytes * max_chunks

    try:
        n = p.stat().st_size
    except Exception:
        return

    if n == 0:
        return

    if n <= max_chunk_bytes:
        try:
            yield p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            try:
                yield p.read_bytes().decode("utf-8", errors="ignore")
            except Exception:
                return
        return

    if n > huge_threshold:
        win_size = max_chunk_bytes
        spans = []
        spans.append((0, min(win_size, n)))
        spans.append((max(0, n - win_size), n))
        if windows > 2:
            step = max((n - 2 * win_size) // (windows - 2), 1)
            pos = win_size
            for _ in range(windows - 2):
                start = min(max(pos, 0), max(0, n - win_size))
                spans.append((start, min(start + win_size, n)))
                pos += step
        try:
            with p.open("rb") as f:
                for s, e in spans[:max_chunks]:
                    f.seek(s)
                    chunk = f.read(e - s)
                    yield chunk.decode("utf-8", errors="ignore")
        except Exception:
            return
        return

    # Medium: read once, but only as much as needed
    try:
        data = p.read_bytes()
    except Exception:
        return
    num_chunks = min(max_chunks, max(2, math.ceil(n / max_chunk_bytes)))
    chunk_size = math.ceil(n / num_chunks)
    start = 0
    for _ in range(num_chunks):
        end = min(start + chunk_size, n)
        if end <= start:
            break
        yield data[start:end].decode("utf-8", errors="ignore")
        start = end


def summarize_file_chunks(
    path: str, chunks: List[str], model_low: str, model_medium: str, call_llm_fn
) -> str:
    """
    Map step: summarize a single file's purpose-only signals from all its chunks.
    """
    bullets: List[str] = []
    for i, ch in enumerate(chunks, 1):
        prompt = textwrap.dedent(
            f"""
        From the file below, write 2-12 bullets capturing a summary that has ONLY purpose/intent (not usage/install/code minutiae).

        FILE: {path} (chunk {i}/{len(chunks)})
        ---
        {ch}
        """
        ).strip()
        msg = [
            {
                "role": "system",
                "content": "Extract purpose-only bullets from file text. Keep it minimal.",
            },
            {"role": "user", "content": prompt},
        ]
        bullets.append(call_llm_fn(msg, model=model_low))
    # reduce bullets -> one-liner for file
    reduce_prompt = textwrap.dedent(
        f"""
    Combine the bullets below into a single 2-12 sentence purpose statement/summary for this file.
    Avoid implementation details. If purpose is unclear, say 'unclear'.

    FILE: {path}
    BULLETS:
    {chr(10).join(bullets)}
    """
    ).strip()
    return call_llm_fn(
        [
            {
                "role": "system",
                "content": "Distill bullets into one short purpose sentence.",
            },
            {"role": "user", "content": reduce_prompt},
        ],
        model=model_medium,
    )


def summarize_directory(
    file_summaries: List[Tuple[str, str]], model: str, call_llm_fn
) -> str:
    """
    Reduce step: combine many file-purpose sentences into a directory-level 2-12 sentence purpose.
    """
    body = "\n".join([f"- {p}: {s}" for p, s in file_summaries])
    prompt = textwrap.dedent(
        f"""
    Summarize the unified purpose for this collection of files in **2-12 sentences**.
    Focus on what the code aims to do. Avoid lists and specifics.

    FILE PURPOSES:
    {body[:18000]}
    """
    ).strip()
    return call_llm_fn(
        [
            {
                "role": "system",
                "content": "Synthesize a concise, faithful purpose across files.",
            },
            {"role": "user", "content": prompt},
        ],
        model=model,
    )


def synthesize_repo_goal_from_code(
    repo_root: pathlib.Path,
    model_medium: str,
    model_low: str,
    model_high: str,
    call_llm_fn,
) -> str:
    """
    Full map-reduce across ALL text files:
      file chunks -> file purpose -> repo purpose (final goal).
    """
    # 1) Map: per-file purpose
    file_purposes: List[Tuple[str, str]] = []
    for p in iter_text_files(repo_root):
        rel = str(p.relative_to(repo_root))
        chunks = list(chunk_file_bytes(p))
        if not chunks:
            continue
        try:
            purpose = summarize_file_chunks(
                rel,
                chunks,
                model_low=model_low,
                model_medium=model_medium,
                call_llm_fn=call_llm_fn,
            )
            if purpose and purpose.strip():
                file_purposes.append((rel, purpose.strip()))
        except Exception:
            # skip noisy failures, continue
            continue

    if not file_purposes:
        return "Goal not explicitly stated."

    # 2) Reduce: repo-level purpose
    return summarize_directory(file_purposes, model=model_high, call_llm_fn=call_llm_fn)
