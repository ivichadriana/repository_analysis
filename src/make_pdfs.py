#!/usr/bin/env python3
import os, sys, re, argparse, datetime, pathlib
from typing import Optional, List, Tuple
from jinja2 import Template
import markdown2

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # Fallback handled later


# Lazy import so we can show a friendly message if WeasyPrint is missing
def _require_weasy():
    try:
        from weasyprint import HTML, CSS

        return HTML, CSS
    except Exception as e:
        print(
            "WeasyPrint is required to render PDFs.\n"
            "Install with:\n"
            "  pip install weasyprint\n"
            "Note: You may need Cairo/Pango on your system (e.g. 'brew install cairo pango').\n"
            f"Import error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


DEFAULT_TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{ title|e }}</title>
<style>
  /* ========= Page setup ========= */
  @page {
    size: A4;
    margin: 0 0 0.55in 0; /* top right bottom left */
    @bottom-center {
      content: "{{ footer_center }}";
      font-style: italic;
      font-size: calc(12pt * var(--scale, 1));
      color: #6b7280;
    }
  }

  /* ========= Design tokens ========= */
  :root{
    --scale: 1;

    /* Colors */
    --ink: #111827;
    --blue-dark: #1F3A63;   /* right bar */
    --blue-soft: #D9E6F4;   /* top band */

    /* Layout */
    --sidew: 1.60in;            /* right bar width */
    --pad-left: 0.90in;         /* page padding, left */
    --pad-right: 2.00in;        /* page padding, right */

    /* Hard clearance between text and right bar (increase to push text left) */
    --right-gutter: 1.2in;     /* text-to-bar gap */

    /* Title band + spacing */
    --band-h: 1.8in;           /* light-blue band height */
    --header-bottom-pad: 0.05in;/* how low the title sits within the band */
    --content-top-gap: 0.20in;  /* white buffer below band before body starts */

    /* Footer protection */
    --content-bottom-pad: 0.50in; /* keep body off the footer */

    /* Type scale */
    --body-size: 3pt;         /* very small body text */
    --title-size: 140pt;         /* very large main title */
    --sub-size: 34pt;
    --h2-size: 24pt;
    --h3-size: 14pt;
  }

  html, body { height:auto; }
  * { box-sizing: border-box; }
  body{
    margin:0;
    background:#fff;
    color:var(--ink);
    font-family: Inter, "Helvetica Neue", Arial, system-ui, -apple-system, Segoe UI, Roboto, "Noto Sans", sans-serif;
    font-size: calc(var(--body-size) * var(--scale, 1));
    line-height:1.6;
    -webkit-font-smoothing: antialiased;
    overflow-wrap: anywhere;
  }

  /* ===== Background rectangles =====
     Dark bar ends at page content bottom (not into footer margin). */
  .page-side{
    position: fixed;
    top: 0; right: 0; bottom: 0;
    width: var(--sidew);
    background: var(--blue-dark);
    z-index: 1;
  }
  .top-band{
    position: fixed;
    top: 0; left: 0; right: 0;
    height: var(--band-h);
    background: var(--blue-soft);
    z-index: 2; /* above the dark bar */
  }

  /* ===== Content column =====
     Reserve hard space to the right so text never overlaps the bar. */
  .container{
    position: relative;
    z-index: 3;

    /* overall column placement */
    margin-left: var(--pad-left);
    margin-right: var(--pad-right);

    /* KEY: push content left away from the right bar */
    padding-right: calc(var(--sidew) + var(--right-gutter)) !important;

    /* avoid width math fights */
    width: auto !important;

    padding-bottom: var(--content-bottom-pad);
  }

  /* ===== Title section (inside the light-blue band) =====
     Header occupies the band height; text sits near the band bottom. */
  .doc-header{
    min-height: var(--band-h);
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    padding-bottom: var(--header-bottom-pad);
    margin: 0;
  }
  .title{
    font-weight: 800;
    font-size: var(--sub-size);
    line-height: 1.06;
    margin: 0 0 6px 0;
    letter-spacing: -0.01em;
  }
  .subtitle{
    font-weight: 600;
    font-size: var(--sub-size);
    margin: 0 0 2px 0;
  }
  .meta-line{
    font-weight: 400;
    font-size: calc(var(--body-size) * 0.60 * var(--scale, 1));
    margin: 2px 0 0 0;
  }
  .meta-line a{
    color: inherit;
    text-decoration: underline;
    text-decoration-thickness: .06em;
    text-underline-offset: 2px;
  }

  /* ===== Body starts only in white space, with buffer ===== */
  .content{
    /* because .doc-header already consumes the band height */
    margin-top: var(--content-top-gap);
  }
  .content h2{
    font-weight: 800;
    font-size: calc(var(--h2-size) * var(--scale, 1));
    line-height: 1.25;
    margin: 22px 0 6px 0;
  }

/* ===== Smaller body text for specific sections (keeps H2 size unchanged) ===== */
    .body-shrink{
    font-size: calc(var(--body-size) * 0.50 * var(--scale, 1));
    line-height: 1.4;  /* Also reduce line height to pack more lines */
    }
  /* Ensure inline elements inside keep the same reduced size */
  .body-shrink p,
  .body-shrink li,
  .body-shrink blockquote,
  .body-shrink code,
  .body-shrink pre,
  .body-shrink table,
  .body-shrink th,
  .body-shrink td {
    font-size: 1em; /* inherit the reduced size */
  }
  .section-subtitle{
    display:block;
    font-size: calc(var(--sub-size) * var(--scale, 1));
    margin: 4px 0 0 0;
  }
  .content h3{
    font-weight:700;
    font-size: calc(var(--h3-size) * var(--scale, 1));
    margin: 14px 0 6px 0;
  }
  .content p{ 
  margin:10px 0; 
  font-size: 10pt;
  }
  .content li{ 
  font-size: 10pt;
  }
  .content a{
    color: inherit;
    text-decoration: underline;
    text-decoration-thickness: .06em;
    text-underline-offset: 2px;
  }
  .content blockquote{
    margin:12px 0;
    padding-left:12px;
    border-left: 3px solid #e5e7eb;
  }
  .content code{
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
    font-size:85%;
    padding:0 4px;
  }
  .content pre code{
    display:block; overflow-x:auto; padding:10px;
    border:1px solid #e5e7eb;
  }
</style>
</head>
<body>
  <!-- Background shapes -->
  <div class="page-side"></div>
  <div class="top-band"></div>

  <div class="container">
    <header class="doc-header">
      <div class="title">{{ display_title }}</div>
      {% if subtitle %}<div class="subtitle">{{ subtitle }}</div>{% endif %}
      {% if meta_bar %}<div class="meta-line">{{ meta_bar | safe }}</div>{% endif %}
    </header>

    <section class="content">
      {{ html|safe }}
    </section>
  </div>
</body>
</html>
"""


def load_logo_as_data_uri(logo_path: Optional[str]) -> Optional[str]:
    if not logo_path:
        return None
    p = pathlib.Path(logo_path)
    if not p.exists() or not p.is_file():
        print(f"[warn] Logo not found at {logo_path}; continuing without a logo.")
        return None
    import base64, mimetypes

    mime, _ = mimetypes.guess_type(str(p))
    if not mime:
        mime = "image/png"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def md_to_html(md_text: str) -> str:
    return markdown2.markdown(
        md_text,
        extras=[
            "fenced-code-blocks",
            "tables",
            "strike",
            "task_list",
            "break-on-newline",
            "header-ids",
            "cuddled-lists",
            "smarty-pants",
        ],
    )


def strip_relative_window_phrases(text: Optional[str]) -> Optional[str]:
    """Remove trailing '(last N days)' or '— last N days' style phrases."""
    if not text:
        return text
    # Variants: hyphen/en dash/em dash; optional parentheses
    patterns = [
        r"\s*\(last\s+\d+\s+days\)\s*$",
        r"\s*[–—-]\s*last\s+\d+\s+days\s*$",
    ]
    out = text
    for pat in patterns:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    return out.strip()


def split_title_subtitle_adjacent(
    md_text: str,
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Return (title, subtitle, md_without_top_title_block).
    Subtitle is ONLY taken if it is the first non-empty, non-heading line
    immediately after the H1, before any other heading appears.
    """
    lines = md_text.splitlines()
    title = None
    subtitle = None
    title_idx = None

    for i, ln in enumerate(lines):
        if ln.strip().startswith("# "):
            title = ln.strip().lstrip("# ").strip()
            title_idx = i
            break

    if title_idx is not None:
        # Look only at the first meaningful line after the H1
        subtitle_idx = None
        j = title_idx + 1
        while j < len(lines) and lines[j].strip() == "":
            j += 1
        if j < len(lines):
            nxt = lines[j].strip()
            if nxt.startswith("#"):
                pass
            else:
                subtitle = nxt
                subtitle_idx = j

        to_remove = {title_idx}
        if subtitle_idx is not None:
            to_remove.add(subtitle_idx)

        md_wo = "\n".join(l for idx, l in enumerate(lines) if idx not in to_remove)
    else:
        md_wo = md_text

    return title, subtitle, md_wo


def demote_h1_to_h2_in_html(html: str) -> str:
    """Keep hierarchy tidy by demoting any leftover <h1> in the body to <h2>."""
    html = re.sub(r"<h1(\b[^>]*)>", r"<h2\1>", html)
    html = re.sub(r"</h1>", r"</h2>", html)
    return html


def extract_generated_footer_line(md_text: str) -> Tuple[str, str]:
    """
    Find a line like '*Report generated using A.I. on ...*' anywhere in the md,
    remove it from the body, and return it (de-asterisked) for the footer.
    If not found, footer is empty.
    """
    lines = md_text.splitlines()
    footer_line = ""
    pattern = re.compile(
        r"^\s*\*?\s*Report generated using A\.I\.\s+on\s+.+?\s*\*?\s*$", re.IGNORECASE
    )
    kept = []
    for ln in lines:
        if pattern.match(ln):
            clean = ln.strip()
            if clean.startswith("*") and clean.endswith("*"):
                clean = clean[1:-1].strip()
            footer_line = clean
        else:
            kept.append(ln)
    return "\n".join(kept), footer_line


def format_date(d: datetime.date) -> str:
    return (
        d.strftime("%b %-d, %Y")
        if sys.platform != "win32"
        else d.strftime("%b %#d, %Y")
    )


def compute_coverage_label(
    range_days: Optional[int],
    date_start_str: Optional[str],
    date_end_str: Optional[str],
    tz_name: str,
) -> str:
    # Resolve timezone
    tz = None
    if ZoneInfo:
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            pass
    if tz is None:
        tz = datetime.timezone.utc
        tz_name = "UTC"

    today = datetime.datetime.now(tz).date()

    date_start = None
    date_end = None
    # Parse provided dates if any
    if date_start_str:
        date_start = datetime.date.fromisoformat(date_start_str)
    if date_end_str:
        date_end = datetime.date.fromisoformat(date_end_str)

    if date_start and date_end:
        pass
    elif date_start and not date_end:
        date_end = today
    elif date_end and not date_start:
        N = range_days or 90
        date_start = date_end - datetime.timedelta(days=N - 1)
    else:
        # Neither provided -> use range_days ending today
        N = range_days or 90
        date_end = today
        date_start = date_end - datetime.timedelta(days=N - 1)

    # Ensure ordering
    if date_start > date_end:
        date_start, date_end = date_end, date_start

    return f"{format_date(date_start)} – {format_date(date_end)}"


def _render_with_scale(HTML, CSS, html: str, base_url: str, scale: float):
    """Render and return a WeasyPrint Document with a CSS override for --scale."""
    css_override = CSS(string=f":root {{ --scale: {scale}; }}")
    return HTML(string=html, base_url=base_url).render(stylesheets=[css_override])


def _fit_to_one_page(
    HTML,
    CSS,
    html: str,
    base_url: str,
    lo: float = 0.65,
    hi: float = 1.0,
    iters: int = 7,
):
    """
    Binary search a scale factor so the rendered PDF fits on exactly one page.
    Returns (doc, best_scale).
    """
    best_doc = None
    best_scale = lo
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        doc = _render_with_scale(HTML, CSS, html, base_url, mid)
        pages = len(doc.pages)
        if pages <= 1:
            best_doc = doc
            best_scale = mid
            lo = mid  # try larger
        else:
            hi = mid  # need smaller
    if best_doc is None:
        best_doc = _render_with_scale(HTML, CSS, html, base_url, lo)
        best_scale = lo
    return best_doc, best_scale


def _doc_type_and_keys(md_path: pathlib.Path) -> dict:
    """
    Infer document type from filename:
      - Portfolio: name starts with '_portfolio'
      - Repo:      '<project_id>__<owner>__<repo>.md'
      - Project:   '<project_id>.md' (no double __)
    Return dict with keys:
      type: 'portfolio' | 'project' | 'repo'
      project_id, owner, repo (when available)
    """
    stem = md_path.stem
    if stem.startswith("_portfolio"):
        return {"type": "portfolio"}
    parts = stem.split("__")
    if len(parts) == 3:
        return {
            "type": "repo",
            "project_id": parts[0],
            "owner": parts[1],
            "repo": parts[2],
        }
    return {"type": "project", "project_id": stem}


def _parse_counts_from_title(md_text: str) -> dict:
    """
    Pull counts out of the H1 line when present (your generators write them).
    - Portfolio H1 example: "# Portfolio Summary - 5 projects (last 90 days)"
    - Project  H1 example: "# Executive Summary: Project Foo — 3 repositories — last 90 days"
    """
    first = next(
        (ln.strip() for ln in md_text.splitlines() if ln.strip().startswith("# ")), ""
    )
    out = {}
    # projects count
    m = re.search(r"\b(\d+)\s+projects\b", first, flags=re.IGNORECASE)
    if m:
        out["projects"] = int(m.group(1))
    # repositories count
    m = re.search(r"\b(\d+)\s+repositor", first, flags=re.IGNORECASE)
    if m:
        out["repos"] = int(m.group(1))
    return out


def _plural(n: int, singular: str, plural: str) -> str:
    return f"{n} {singular if n == 1 else plural}"


def _count_repo_files(in_dir: pathlib.Path) -> int:
    """Count repo-level MDs: '<project_id>__<owner>__<repo>.md'"""
    return sum(1 for p in in_dir.glob("*__*__*.md"))


def _count_project_repos(in_dir: pathlib.Path, project_id: str) -> int:
    """Count repo MDs for a given project prefix."""
    return sum(1 for p in in_dir.glob(f"{project_id}__*__*.md"))


def _read_project_title(project_id: str, in_dir: pathlib.Path) -> str:
    """
    Read <project_id>.md (if present) and extract its H1 as a human title,
    stripping 'Executive Summary:' and any trailing counts. Fallback to project_id.
    """
    md = in_dir / f"{project_id}.md"
    if not md.exists():
        return project_id
    text = md.read_text(encoding="utf-8")
    h1 = (
        next(
            (
                ln.strip()[2:].strip()
                for ln in text.splitlines()
                if ln.strip().startswith("# ")
            ),
            "",
        )
        or project_id
    )
    h1 = re.sub(r"^Executive Summary:\s*", "", h1, flags=re.IGNORECASE)
    h1 = strip_relative_window_phrases(h1)
    # Also strip “ — 3 repositories …”
    h1 = re.sub(r"\s*[—-]\s*\d+\s+repositor.*$", "", h1, flags=re.IGNORECASE)
    return h1.strip() or project_id


def _project_repo_paths(in_dir: pathlib.Path, project_id: str) -> list[pathlib.Path]:
    """All repo-level MDs for a project, sorted by owner/repo in filename."""
    repo_paths = sorted(
        in_dir.glob(f"{project_id}__*__*.md"), key=lambda p: p.stem.lower()
    )
    return repo_paths


def _repo_position_in_project(
    in_dir: pathlib.Path, project_id: str, owner: str, repo: str
) -> tuple[int, int]:
    """
    Return (pos, total) where pos is 1-based index of this repo among the
    project's repos (alphabetical by filename).
    """
    items = _project_repo_paths(in_dir, project_id)
    total = len(items)
    target_stem = f"{project_id}__{owner}__{repo}"
    try:
        pos = [p.stem for p in items].index(target_stem) + 1
    except ValueError:
        pos = 1
    return pos, total


def _build_header_for_file(
    md_path: pathlib.Path,
    md_text: str,
    in_dir: pathlib.Path,
    coverage_label: str,
    range_days: int,
) -> tuple[str, str]:
    """
    Returns (display_title, meta_bar_html).
    - Portfolio:   Title = "Portfolio Summary"; Meta = "N Projects, M Repositories"
    - Project:     Title = "Project Summary";   Meta = "<Project Title> (<ID>) • K Repositories"
    - Repository:  Title = "Repository Summary";
                   Meta line 1: "Project: <Project Title> (<ID>) • K Repositories"
                   Meta line 2: "Repository Name: <owner>/<repo>" [link neutral color]
    """
    info = _doc_type_and_keys(md_path)
    counts = _parse_counts_from_title(md_text)

    if info["type"] == "portfolio":
        display_title = "Portfolio Summary"
        projects = counts.get("projects")
        if projects is None:
            projects = sum(
                1
                for p in in_dir.glob("*.md")
                if "__" not in p.stem and not p.stem.startswith("_portfolio")
            )
        repos = _count_repo_files(in_dir)
        meta_html = f"{_plural(projects, 'Project', 'Projects')}, {_plural(repos, 'Repository', 'Repositories')}"
        return display_title, meta_html

    if info["type"] == "project":
        project_id = info["project_id"]
        proj_title = _read_project_title(project_id, in_dir)
        # repos count
        repos = counts.get("repos")
        if repos is None:
            repos = _count_project_repos(in_dir, project_id)
        display_title = "Project Summary"
        meta_html = f"{proj_title} ({project_id}) • {_plural(repos, 'Repository', 'Repositories')}"
        return display_title, meta_html

    # repo doc
    project_id, owner, repo = info["project_id"], info["owner"], info["repo"]
    proj_title = _read_project_title(project_id, in_dir)
    repos_total = _count_project_repos(in_dir, project_id)
    pos, total = _repo_position_in_project(in_dir, project_id, owner, repo)
    display_title = "Repository Summary"
    link = f"https://github.com/{owner}/{repo}"
    meta_html = (
        f"Project: {proj_title} ({project_id}) • {_plural(repos_total, 'Repository', 'Repositories')}<br>"
        f'Repository Name: <a class="repo-link" href="{link}">{owner}/{repo}</a>'
    )
    return display_title, meta_html


def _inject_recent_subtitle(html: str, subtitle: str) -> str:
    """
    Insert one line under the first <h2> titled 'Recent Developments':
      "Recent Developments: <dates> (last N days)"
    """
    pattern = re.compile(
        r"(<h2\b[^>]*>\s*Recent\s+Developments\s*</h2>)", flags=re.IGNORECASE
    )
    return pattern.sub(
        rf"\1<div class=\"section-subtitle\">{subtitle}</div>", html, count=1
    )


def _shrink_section_bodies(html: str) -> str:
    """
    Wrap the body *after* the H2 'Summary and Goal' and 'Recent Developments'
    (optionally including the injected .section-subtitle) in <div class="body-shrink">…</div>
    up to (but not including) the next H2 or end of document.
    Titles (the H2s) remain full size.
    """
    # 1) Summary and Goal
    html = re.sub(
        r"(?is)"
        r"(?P<h2><h2\b[^>]*>\s*Summary\s+and\s+Goal\s*</h2>)"
        r"(?P<body>.*?)(?=(<h2\b|$))",
        lambda m: f'{m.group("h2")}<div class="body-shrink">{m.group("body")}</div>',
        html,
        count=1,
    )

    # 2) Recent Developments (include optional section-subtitle line we injected)
    html = re.sub(
        r"(?is)"
        r"(?P<h2><h2\b[^>]*>\s*Recent\s+Developments\s*</h2>\s*"
        r'(?:<div\b[^>]*class="section-subtitle"[^>]*>.*?</div>\s*)?)'
        r"(?P<body>.*?)(?=(<h2\b|$))",
        lambda m: f'{m.group("h2")}<div class="body-shrink">{m.group("body")}</div>',
        html,
        count=1,
    )
    return html


def _human_period_label(range_days: int) -> str:
    """
    Turn N days into a friendly label:
      360–370  -> "last year"
      720–740  -> "last 2 years"
      multiples of 30 -> "last N months"
      small multiples of 7 -> "last N weeks"
      otherwise -> "last N days"
    """
    if 360 <= range_days <= 370:
        return "last year"
    if 720 <= range_days <= 740:
        return "last 2 years"
    if range_days % 30 == 0:
        n = range_days // 30
        return f"last {n} month{'s' if n != 1 else ''}"
    if range_days % 7 == 0 and range_days <= 98:
        n = range_days // 7
        return f"last {n} week{'s' if n != 1 else ''}"
    return f"last {range_days} days"


def make_pdf_for_markdown(
    md_path: pathlib.Path,
    out_dir: pathlib.Path,
    logo_path: Optional[str],
    template_text: str,
    one_page: bool,
    coverage_label: str,
    range_days: int,
    in_dir_for_counts: Optional[pathlib.Path] = None,
    period_label: Optional[str] = None,
):
    HTML, CSS = _require_weasy()
    if in_dir_for_counts is None:
        in_dir_for_counts = md_path.parent  # default: the same folder

    md_text = md_path.read_text(encoding="utf-8")

    # Extract and remove the "Report generated..." footer line before anything else
    md_text_no_footer, footer_center = extract_generated_footer_line(md_text)

    # Title/subtitle: keep parsing so we can pull a fallback subtitle if present
    title_guess, subtitle_guess, md_body_text = split_title_subtitle_adjacent(
        md_text_no_footer
    )
    subtitle_guess = (
        strip_relative_window_phrases(subtitle_guess) if subtitle_guess else ""
    )

    # Build the display title & meta bar based on file type and counts/links
    display_title, meta_bar = _build_header_for_file(
        md_path=md_path,
        md_text=md_text_no_footer,
        in_dir=in_dir_for_counts,
        coverage_label=coverage_label,
        range_days=range_days,
    )

    # Convert the *body only* to HTML and normalize heading levels
    html_body = md_to_html(md_body_text)
    html_body = demote_h1_to_h2_in_html(html_body)
    html_body = _shrink_section_bodies(html_body)

    # Move date to a subtitle under "Recent Developments"
    label = period_label or _human_period_label(range_days)
    rd_subtitle = f"Recent Developments: {coverage_label.replace('–','-')} (last {range_days} days)"
    html_body = _inject_recent_subtitle(html_body, rd_subtitle)

    # Render with template
    tpl = Template(template_text)
    html = tpl.render(
        display_title=display_title,
        subtitle=subtitle_guess or "",
        meta_bar=meta_bar,
        html=html_body,
        logo_data=load_logo_as_data_uri(logo_path),
        footer_center=footer_center,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / (md_path.stem + ".pdf")

    if one_page:
        doc, scale = _fit_to_one_page(
            HTML, CSS, html, str(md_path.parent), lo=0.40, hi=1.0
        )
        doc.write_pdf(str(out_pdf))
        print(f"[ok] {out_pdf}  (fit to one page at scale ~{scale:.2f})")
    else:
        css_override = CSS(string=":root { --scale: 1; }")
        HTML(string=html, base_url=str(md_path.parent)).write_pdf(
            str(out_pdf), stylesheets=[css_override]
        )
        print(f"[ok] {out_pdf}")


def main():
    ap = argparse.ArgumentParser(
        description="Convert Markdown reports to pretty PDFs (single-page executive summary)."
    )
    ap.add_argument(
        "--in",
        dest="in_dir",
        default="reports",
        help="Input folder containing .md files (default: reports)",
    )
    ap.add_argument(
        "--out",
        dest="out_dir",
        default="reports_pdf",
        help="Output folder for PDFs (default: reports_pdf)",
    )
    ap.add_argument(
        "--logo",
        dest="logo_path",
        default=None,
        help="Optional logo image (PNG/SVG/JPG) for the header",
    )
    ap.add_argument(
        "--template",
        dest="template_path",
        default=None,
        help="Optional path to a custom HTML template",
    )
    # Date controls
    ap.add_argument(
        "--range-days",
        dest="range_days",
        type=int,
        default=90,
        help="Coverage: last N days ending today (default: 90)",
    )
    ap.add_argument(
        "--date-start",
        dest="date_start",
        default=None,
        help="Coverage start date (YYYY-MM-DD)",
    )
    ap.add_argument(
        "--date-end",
        dest="date_end",
        default=None,
        help="Coverage end date (YYYY-MM-DD)",
    )
    ap.add_argument(
        "--timezone",
        dest="tz_name",
        default="America/Denver",
        help="Timezone for date calculations (default: America/Denver)",
    )
    ap.add_argument(
        "--one-page",
        action="store_true",
        help="Auto-scale content so the PDF fits on a single A4 page",
    )
    ap.add_argument(
        "--window-label",
        dest="window_label",
        default=None,
        help='Override the period label shown in parentheses (e.g. "last year").',
    )
    args = ap.parse_args()

    in_dir = pathlib.Path(args.in_dir)
    out_dir = pathlib.Path(args.out_dir)
    period_label = args.window_label or _human_period_label(args.range_days)

    if not in_dir.exists():
        print(f"[error] Input dir not found: {in_dir}", file=sys.stderr)
        sys.exit(1)

    # Load custom template or fallback
    template_text = DEFAULT_TEMPLATE
    if args.template_path:
        p = pathlib.Path(args.template_path)
        if p.exists():
            template_text = p.read_text(encoding="utf-8")
        else:
            print(f"[warn] Template not found at {p}; using default.")

    # Compute the explicit coverage label (e.g., "Jun 5, 2025 – Sep 2, 2025 (America/Denver)")
    coverage_label = compute_coverage_label(
        args.range_days, args.date_start, args.date_end, args.tz_name
    )

    md_files = sorted(in_dir.glob("*.md"))
    if not md_files:
        print(f"[warn] No .md files found in {in_dir}")
        sys.exit(0)

    for md in md_files:
        make_pdf_for_markdown(
            md_path=md,
            out_dir=out_dir,
            logo_path=args.logo_path,
            template_text=template_text,
            one_page=args.one_page,
            coverage_label=coverage_label,
            range_days=args.range_days,
            in_dir_for_counts=in_dir,
            period_label=period_label,
        )


if __name__ == "__main__":
    main()
