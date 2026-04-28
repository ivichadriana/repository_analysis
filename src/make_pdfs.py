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
    margin: 0 0 0.55in 0;
    @bottom-center {
      content: "{{ footer_center }}";
      font-style: italic;
      font-size: 9pt;
      color: #6b7280;
    }
  }

  /* ========= Design tokens ========= */
  :root {
    --scale: 1;

    /* Colors */
    --ink: #111827;
    --blue-dark: #1F3A63;
    --blue-soft: #D9E6F4;

    /* Layout */
    --sidew: 0.30in;            /* narrow right accent bar */
    --pad-left: 0.65in;         /* left page margin */
    --pad-right: 0.65in;        /* right page margin (not including bar) */

    /* Title band */
    --band-h: 1.4in;
    --header-bottom-pad: 0.12in;
    --content-top-gap: 0.25in;
    --content-bottom-pad: 0.50in;

    /* Type scale — all fixed pt, not scaled */
    --body-size: 10pt;
    --title-size: 22pt;
    --sub-size: 11pt;
    --h2-size: 15pt;
    --h3-size: 12pt;
  }

  html, body { height: auto; }
  * { box-sizing: border-box; }

  body {
    margin: 0;
    background: #fff;
    color: var(--ink);
    font-family: Inter, "Helvetica Neue", Arial, system-ui, sans-serif;
    font-size: var(--body-size);
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
    overflow-wrap: anywhere;
  }

  /* ===== Background shapes ===== */
  .page-side {
    position: fixed;
    top: 0; right: 0; bottom: 0;
    width: var(--sidew);
    background: var(--blue-dark);
    z-index: 1;
  }
  .top-band {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: var(--band-h);
    background: var(--blue-soft);
    z-index: 2;
  }

  /* ===== Content column ===== */
  .container {
    position: relative;
    z-index: 3;
    margin-left: var(--pad-left);
    /* stay clear of the right bar + page margin */
    margin-right: calc(var(--sidew) + var(--pad-right));
    padding-bottom: var(--content-bottom-pad);
  }

  /* ===== Title section ===== */
  .doc-header {
    min-height: var(--band-h);
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    padding-bottom: var(--header-bottom-pad);
    margin: 0;
  }
  .title {
    font-weight: 800;
    font-size: var(--title-size);
    line-height: 1.1;
    margin: 0 0 4px 0;
    letter-spacing: -0.01em;
    color: var(--blue-dark);
  }
  .subtitle {
    font-weight: 600;
    font-size: var(--sub-size);
    margin: 0 0 2px 0;
    color: #374151;
  }
  .meta-line {
    font-weight: 400;
    font-size: 9pt;
    color: #6b7280;
    margin: 2px 0 0 0;
    line-height: 1.4;
  }
  .meta-line a {
    color: inherit;
    text-decoration: underline;
    text-decoration-thickness: .06em;
    text-underline-offset: 2px;
  }

  /* ===== Body ===== */
  .content {
    margin-top: var(--content-top-gap);
  }
  .content h2 {
    font-weight: 800;
    font-size: var(--h2-size);
    line-height: 1.25;
    margin: 20px 0 6px 0;
    color: var(--blue-dark);
  }
  .content h3 {
    font-weight: 700;
    font-size: var(--h3-size);
    margin: 14px 0 5px 0;
  }
  .content p {
    margin: 8px 0;
    font-size: var(--body-size);
  }
  .content li {
    font-size: var(--body-size);
  }
  .content a {
    color: #1d4ed8;
    text-decoration: underline;
    text-decoration-thickness: .06em;
    text-underline-offset: 2px;
  }
  .content blockquote {
    margin: 12px 0;
    padding-left: 12px;
    border-left: 3px solid #e5e7eb;
  }
  .content code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 85%;
    padding: 0 4px;
  }
  .content pre code {
    display: block;
    overflow-x: auto;
    padding: 10px;
    border: 1px solid #e5e7eb;
  }

  .section-subtitle {
    display: block;
    font-size: 9pt;
    color: #6b7280;
    font-style: italic;
    margin: 2px 0 8px 0;
  }

  /* body-shrink is no longer needed but kept harmless */
  .body-shrink {
    font-size: var(--body-size);
    line-height: 1.55;
  }
</style>
</head>
<body>
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
    if not text:
        return text
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
    html = re.sub(r"<h1(\b[^>]*)>", r"<h2\1>", html)
    html = re.sub(r"</h1>", r"</h2>", html)
    return html


def extract_generated_footer_line(md_text: str) -> Tuple[str, str]:
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
    tz = None
    if ZoneInfo:
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            pass
    if tz is None:
        tz = datetime.timezone.utc

    today = datetime.datetime.now(tz).date()

    date_start = None
    date_end = None
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
        N = range_days or 90
        date_end = today
        date_start = date_end - datetime.timedelta(days=N - 1)

    if date_start > date_end:
        date_start, date_end = date_end, date_start

    return f"{format_date(date_start)} – {format_date(date_end)}"


def _render_with_scale(HTML, CSS, html: str, base_url: str, scale: float):
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
    best_doc = None
    best_scale = lo
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        doc = _render_with_scale(HTML, CSS, html, base_url, mid)
        pages = len(doc.pages)
        if pages <= 1:
            best_doc = doc
            best_scale = mid
            lo = mid
        else:
            hi = mid
    if best_doc is None:
        best_doc = _render_with_scale(HTML, CSS, html, base_url, lo)
        best_scale = lo
    return best_doc, best_scale


def _doc_type_and_keys(md_path: pathlib.Path) -> dict:
    stem = md_path.stem
    if stem.startswith("_portfolio"):
        return {"type": "portfolio"}
    parts = stem.split("__")
    if len(parts) == 4 and parts[3] in ("chatbased", "agentbased"):
        return {"type": "repo", "project_id": parts[0], "owner": parts[1], "repo": parts[2], "pipeline": parts[3]}
    if len(parts) == 2 and parts[1] in ("chatbased", "agentbased"):
        return {"type": "project", "project_id": parts[0], "pipeline": parts[1]}
    return {"type": "portfolio"}


def _parse_counts_from_title(md_text: str) -> dict:
    first = next(
        (ln.strip() for ln in md_text.splitlines() if ln.strip().startswith("# ")), ""
    )
    out = {}
    m = re.search(r"\b(\d+)\s+projects\b", first, flags=re.IGNORECASE)
    if m:
        out["projects"] = int(m.group(1))
    m = re.search(r"\b(\d+)\s+repositor", first, flags=re.IGNORECASE)
    if m:
        out["repos"] = int(m.group(1))
    return out


def _plural(n: int, singular: str, plural: str) -> str:
    return f"{n} {singular if n == 1 else plural}"


def _count_repo_files(in_dir: pathlib.Path) -> int:
    return sum(1 for p in in_dir.glob("*__*__*__chatbased.md")) or \
           sum(1 for p in in_dir.glob("*__*__*__agentbased.md"))


def _count_project_repos(in_dir: pathlib.Path, project_id: str, pipeline: str = "chatbased") -> int:
    return sum(1 for p in in_dir.glob(f"{project_id}__*__*__{pipeline}.md"))


def _read_project_title(project_id: str, in_dir: pathlib.Path, pipeline: str = "chatbased") -> str:
    md = in_dir / f"{project_id}__{pipeline}.md"
    if not md.exists():
        # try the other pipeline
        other = "agentbased" if pipeline == "chatbased" else "chatbased"
        md = in_dir / f"{project_id}__{other}.md"
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
    h1 = re.sub(r"\s*[—-]\s*\d+\s+repositor.*$", "", h1, flags=re.IGNORECASE)
    return h1.strip() or project_id


def _build_header_for_file(
    md_path: pathlib.Path,
    md_text: str,
    in_dir: pathlib.Path,
    coverage_label: str,
    range_days: int,
) -> tuple[str, str]:
    info = _doc_type_and_keys(md_path)
    counts = _parse_counts_from_title(md_text)
    pipeline = info.get("pipeline", "chatbased")

    if info["type"] == "portfolio":
        display_title = "Portfolio Summary"
        # Infer pipeline from the portfolio filename itself
        stem = md_path.stem
        if "agentbased" in stem:
            pipe = "agentbased"
        else:
            pipe = "chatbased"
        projects = counts.get("projects")
        if projects is None:
            projects = sum(
                1 for p in in_dir.glob(f"*__{pipe}.md")
                if len(p.stem.split("__")) == 2
                and not p.stem.startswith("_")
            )
        repos = sum(1 for p in in_dir.glob(f"*__*__*__{pipe}.md"))
        meta_html = f"{_plural(projects, 'Project', 'Projects')}, {_plural(repos, 'Repository', 'Repositories')}"
        return display_title, meta_html

    if info["type"] == "project":
        project_id = info["project_id"]
        proj_title = _read_project_title(project_id, in_dir, pipeline)
        repos = counts.get("repos")
        if repos is None:
            repos = _count_project_repos(in_dir, project_id, pipeline)
        display_title = "Project Summary"
        meta_html = f"{proj_title} ({project_id}) • {_plural(repos, 'Repository', 'Repositories')}"
        return display_title, meta_html

    # repo
    project_id, owner, repo = info["project_id"], info["owner"], info["repo"]
    proj_title = _read_project_title(project_id, in_dir, pipeline)
    repos_total = _count_project_repos(in_dir, project_id, pipeline)
    display_title = "Repository Summary"
    link = f"https://github.com/{owner}/{repo}"
    meta_html = (
        f"Project: {proj_title} ({project_id}) • {_plural(repos_total, 'Repository', 'Repositories')}<br>"
        f'Repository: <a href="{link}">{owner}/{repo}</a>'
    )
    return display_title, meta_html


def _inject_recent_subtitle(html: str, subtitle: str) -> str:
    pattern = re.compile(
        r"(<h2\b[^>]*>\s*Recent\s+Developments\s*</h2>)", flags=re.IGNORECASE
    )
    return pattern.sub(
        rf"\1<div class=\"section-subtitle\">{subtitle}</div>", html, count=1
    )


def _shrink_section_bodies(html: str) -> str:
    # Wrap section bodies in body-shrink div (now same size, kept for compatibility)
    html = re.sub(
        r"(?is)"
        r"(?P<h2><h2\b[^>]*>\s*Summary\s+and\s+Goal\s*</h2>)"
        r"(?P<body>.*?)(?=(<h2\b|$))",
        lambda m: f'{m.group("h2")}<div class="body-shrink">{m.group("body")}</div>',
        html,
        count=1,
    )
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
        in_dir_for_counts = md_path.parent

    md_text = md_path.read_text(encoding="utf-8")
    md_text_no_footer, footer_center = extract_generated_footer_line(md_text)

    title_guess, subtitle_guess, md_body_text = split_title_subtitle_adjacent(
        md_text_no_footer
    )
    subtitle_guess = (
        strip_relative_window_phrases(subtitle_guess) if subtitle_guess else ""
    )

    display_title, meta_bar = _build_header_for_file(
        md_path=md_path,
        md_text=md_text_no_footer,
        in_dir=in_dir_for_counts,
        coverage_label=coverage_label,
        range_days=range_days,
    )

    html_body = md_to_html(md_body_text)
    html_body = demote_h1_to_h2_in_html(html_body)
    html_body = _shrink_section_bodies(html_body)

    label = period_label or _human_period_label(range_days)
    rd_subtitle = f"{coverage_label} ({label})"
    html_body = _inject_recent_subtitle(html_body, rd_subtitle)

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
            HTML, CSS, html, str(md_path.parent), lo=0.65, hi=1.0
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
    ap.add_argument("--in", dest="in_dir", default="reports",
                    help="Input folder containing .md files (default: reports)")
    ap.add_argument("--out", dest="out_dir", default="reports_pdf",
                    help="Output folder for PDFs (default: reports_pdf)")
    ap.add_argument("--logo", dest="logo_path", default=None,
                    help="Optional logo image (PNG/SVG/JPG) for the header")
    ap.add_argument("--template", dest="template_path", default=None,
                    help="Optional path to a custom HTML template")
    ap.add_argument("--range-days", dest="range_days", type=int, default=90,
                    help="Coverage: last N days ending today (default: 90)")
    ap.add_argument("--date-start", dest="date_start", default=None,
                    help="Coverage start date (YYYY-MM-DD)")
    ap.add_argument("--date-end", dest="date_end", default=None,
                    help="Coverage end date (YYYY-MM-DD)")
    ap.add_argument("--timezone", dest="tz_name", default="America/Denver",
                    help="Timezone for date calculations (default: America/Denver)")
    ap.add_argument("--one-page", action="store_true",
                    help="Auto-scale content so the PDF fits on a single A4 page")
    ap.add_argument("--window-label", dest="window_label", default=None,
                    help='Override the period label (e.g. "last year")')
    args = ap.parse_args()

    in_dir = pathlib.Path(args.in_dir)
    out_dir = pathlib.Path(args.out_dir)
    period_label = args.window_label or _human_period_label(args.range_days)

    if not in_dir.exists():
        print(f"[error] Input dir not found: {in_dir}", file=sys.stderr)
        sys.exit(1)

    template_text = DEFAULT_TEMPLATE
    if args.template_path:
        p = pathlib.Path(args.template_path)
        if p.exists():
            template_text = p.read_text(encoding="utf-8")
        else:
            print(f"[warn] Template not found at {p}; using default.")

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