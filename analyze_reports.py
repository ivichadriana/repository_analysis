"""
analyze_reports.py

Parses chatbased and agentbased markdown reports from the reports/ directory
and generates poster-ready comparison plots.

Usage:
    python analyze_reports.py --reports-dir /path/to/reports --out-dir ./plots

Metrics computed per report (repo-level only, to keep comparisons fair):
    - word_count         : total words in body (excluding title line)
    - section_goal_words : words in ## Summary and Goal section
    - section_dev_words  : words in ## Recent Developments section
    - sections_present   : how many of the 2 expected sections are present (0, 1, or 2)
    - link_count         : number of markdown hyperlinks [text](url)
    - has_goal           : bool, Summary and Goal section present
    - has_dev            : bool, Recent Developments section present
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

# ── Aesthetics ────────────────────────────────────────────────────────────────
CHAT_COLOR  = "#2A6496"   # deep blue
AGENT_COLOR = "#C0392B"   # deep red
BG_COLOR    = "#FAFAF8"
GRID_COLOR  = "#E8E8E4"
FONT_TITLE  = {"fontsize": 15, "fontweight": "bold", "color": "#1a1a1a"}
FONT_LABEL  = {"fontsize": 12, "color": "#333333"}
FONT_TICK   = {"fontsize": 10, "color": "#444444"}
FONT_ANNOT  = {"fontsize": 9,  "color": "#555555"}

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.facecolor":     BG_COLOR,
    "figure.facecolor":   BG_COLOR,
    "axes.grid":          True,
    "grid.color":         GRID_COLOR,
    "grid.linewidth":     0.8,
    "axes.axisbelow":     True,
})

# ── Parsing ───────────────────────────────────────────────────────────────────

LINK_RE    = re.compile(r'\[([^\]]+)\]\((https?://[^\)]+)\)')
SECTION_RE = re.compile(r'^##\s+(.+)$', re.MULTILINE)

def count_words(text: str) -> int:
    return len(text.split())

def extract_section(text: str, heading_pattern: str) -> str | None:
    """Return the body of a section matching heading_pattern (case-insensitive)."""
    lines = text.splitlines()
    inside = False
    body_lines = []
    for line in lines:
        if re.match(r'^##\s+' + heading_pattern, line, re.IGNORECASE):
            inside = True
            continue
        if inside:
            if re.match(r'^##\s+', line):
                break
            body_lines.append(line)
    return '\n'.join(body_lines).strip() if inside else None

def parse_report(path: Path) -> dict:
    text = path.read_text(encoding='utf-8', errors='replace')
    # Strip title line(s) (H1)
    body = re.sub(r'^#\s+.+\n?', '', text, count=1).strip()

    goal_body = extract_section(text, r'Summary\s+and\s+Goal')
    dev_body  = extract_section(text, r'Recent\s+Developments')

    return {
        "file":              path.name,
        "word_count":        count_words(body),
        "section_goal_words": count_words(goal_body) if goal_body is not None else 0,
        "section_dev_words":  count_words(dev_body)  if dev_body  is not None else 0,
        "has_goal":          goal_body is not None,
        "has_dev":           dev_body  is not None,
        "sections_present":  int(goal_body is not None) + int(dev_body is not None),
        "link_count":        len(LINK_RE.findall(text)),
    }

def load_reports(reports_dir: Path) -> pd.DataFrame:
    records = []
    for path in sorted(reports_dir.glob("*.md")):
        name = path.stem
        # Only repo-level files: 4 parts split by __
        parts = name.split("__")
        if len(parts) < 4:
            continue
        suffix = parts[-1]   # chatbased or agentbased
        if suffix not in ("chatbased", "agentbased"):
            continue
        rec = parse_report(path)
        rec["pipeline"]   = "Chat-based"   if suffix == "chatbased"  else "Agentic"
        rec["project_id"] = parts[0]
        rec["repo"]       = parts[2]
        records.append(rec)

    if not records:
        print("[warn] No repo-level report files found. Check --reports-dir.")
    return pd.DataFrame(records)

# ── Plotting helpers ──────────────────────────────────────────────────────────

def _add_significance(ax, x1, x2, y, p, h=1.5):
    """Draw significance bracket between two x positions."""
    if p < 0.001:   label = "***"
    elif p < 0.01:  label = "**"
    elif p < 0.05:  label = "*"
    else:           label = "ns"
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, color="#666")
    ax.text((x1+x2)/2, y+h+0.3, label, ha='center', va='bottom',
            fontsize=10, color="#444")

def _style_ax(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, **FONT_LABEL)
    ax.set_ylabel(ylabel, **FONT_LABEL)
    ax.set_title(title, **FONT_TITLE, pad=12)
    ax.tick_params(labelsize=FONT_TICK["fontsize"], colors=FONT_TICK["color"])

# ── Plot 1: Word count distributions (violin + strip) ─────────────────────────

def plot_word_count(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG_COLOR)

    chat  = df[df.pipeline == "Chat-based"]["word_count"].values
    agent = df[df.pipeline == "Agentic"]["word_count"].values

    positions = [1, 2]
    colors    = [CHAT_COLOR, AGENT_COLOR]
    data      = [chat, agent]
    labels    = ["Chat-based", "Agentic"]

    for pos, d, col in zip(positions, data, colors):
        parts = ax.violinplot(d, positions=[pos], widths=0.6,
                              showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(col)
            pc.set_alpha(0.35)
            pc.set_edgecolor(col)
            pc.set_linewidth(1.2)

        # Jittered strip
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(d))
        ax.scatter(pos + jitter, d, color=col, alpha=0.7, s=28, zorder=3,
                   edgecolors='white', linewidths=0.4)

        # Median line
        med = np.median(d)
        ax.hlines(med, pos-0.18, pos+0.18, colors=col, linewidths=2.2, zorder=4)
        ax.text(pos+0.22, med, f'{med:.0f}', va='center',
                fontsize=FONT_ANNOT["fontsize"], color=col, fontweight='bold')

    # Significance test
    t, p = stats.mannwhitneyu(chat, agent, alternative='two-sided')
    ymax = max(np.max(chat), np.max(agent))
    _add_significance(ax, 1, 2, ymax + 5, p, h=8)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlim(0.4, 2.6)
    _style_ax(ax, ylabel="Word Count (per repo summary)",
              title="Summary Length")
    ax.spines['left'].set_color('#ccc')
    ax.spines['bottom'].set_color('#ccc')
    fig.tight_layout()
    fig.savefig(out / "plot_word_count.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[ok] {out / 'plot_word_count.png'}")

# ── Plot 2: Section presence (stacked bar) ────────────────────────────────────

def plot_section_presence(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_facecolor(BG_COLOR)

    pipelines = ["Chat-based", "Agentic"]
    colors_map = {"Chat-based": CHAT_COLOR, "Agentic": AGENT_COLOR}

    for i, pip in enumerate(pipelines):
        sub   = df[df.pipeline == pip]
        n     = len(sub)
        both  = (sub.sections_present == 2).sum() / n * 100
        one   = (sub.sections_present == 1).sum() / n * 100
        zero  = (sub.sections_present == 0).sum() / n * 100

        col   = colors_map[pip]
        x     = i

        ax.bar(x, both, color=col,        alpha=0.90, width=0.45, label="Both sections" if i==0 else "")
        ax.bar(x, one,  bottom=both,      color=col,  alpha=0.45, width=0.45, label="One section"  if i==0 else "")
        ax.bar(x, zero, bottom=both+one,  color=col,  alpha=0.18, width=0.45, label="No sections"  if i==0 else "")

        # Annotate
        if both > 2:
            ax.text(x, both/2,        f'{both:.0f}%',      ha='center', va='center',
                    fontsize=11, color='white', fontweight='bold')
        if one > 2:
            ax.text(x, both + one/2,  f'{one:.0f}%',       ha='center', va='center',
                    fontsize=10, color='white', fontweight='bold')
        if zero > 2:
            ax.text(x, both+one+zero/2, f'{zero:.0f}%',    ha='center', va='center',
                    fontsize=10, color='white', fontweight='bold')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(pipelines, fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Percentage of Reports (%)", **FONT_LABEL)
    ax.set_title("Section Completeness", **FONT_TITLE, pad=12)

    legend_patches = [
        mpatches.Patch(color='#555', alpha=0.90, label='Both sections present'),
        mpatches.Patch(color='#555', alpha=0.45, label='One section present'),
        mpatches.Patch(color='#555', alpha=0.18, label='No sections present'),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc='upper right',
              framealpha=0.85, edgecolor='#ccc')
    ax.spines['left'].set_color('#ccc')
    ax.spines['bottom'].set_color('#ccc')
    fig.tight_layout()
    fig.savefig(out / "plot_section_presence.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[ok] {out / 'plot_section_presence.png'}")

# ── Plot 3: Link usage (box + strip) ─────────────────────────────────────────

def plot_link_usage(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG_COLOR)

    chat  = df[df.pipeline == "Chat-based"]["link_count"].values
    agent = df[df.pipeline == "Agentic"]["link_count"].values

    positions = [1, 2]
    colors    = [CHAT_COLOR, AGENT_COLOR]
    data      = [chat, agent]
    labels    = ["Chat-based", "Agentic"]

    for pos, d, col in zip(positions, data, colors):
        bp = ax.boxplot(d, positions=[pos], widths=0.35, patch_artist=True,
                        showfliers=False,
                        medianprops=dict(color='white', linewidth=2.5),
                        boxprops=dict(facecolor=col, alpha=0.4, linewidth=1.2, edgecolor=col),
                        whiskerprops=dict(color=col, linewidth=1.2, linestyle='--'),
                        capprops=dict(color=col, linewidth=1.5))

        jitter = np.random.default_rng(7).uniform(-0.09, 0.09, size=len(d))
        ax.scatter(pos + jitter, d, color=col, alpha=0.75, s=28, zorder=4,
                   edgecolors='white', linewidths=0.4)

        mean = np.mean(d)
        ax.text(pos+0.22, np.median(d), f'med={np.median(d):.1f}',
                va='center', fontsize=9, color=col, fontweight='bold')

    t, p = stats.mannwhitneyu(chat, agent, alternative='two-sided')
    ymax = max(np.max(chat), np.max(agent))
    _add_significance(ax, 1, 2, ymax + 0.3, p, h=0.5)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlim(0.4, 2.6)
    _style_ax(ax, ylabel="Hyperlinks per Report",
              title="Link Usage")
    ax.spines['left'].set_color('#ccc')
    ax.spines['bottom'].set_color('#ccc')
    fig.tight_layout()
    fig.savefig(out / "plot_link_usage.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[ok] {out / 'plot_link_usage.png'}")

# ── Plot 4: Per-section word counts (paired grouped bar) ─────────────────────

def plot_section_words(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(BG_COLOR)

    sections  = ["Summary and Goal", "Recent Developments"]
    cols_map  = {"Summary and Goal":    ("section_goal_words", 0),
                 "Recent Developments": ("section_dev_words",  1)}

    x       = np.array([0, 1])
    width   = 0.3
    offsets = [-width/2 - 0.03, width/2 + 0.03]

    pipeline_order = ["Chat-based", "Agentic"]
    pipe_colors    = [CHAT_COLOR,   AGENT_COLOR]

    for pip, col, offset in zip(pipeline_order, pipe_colors, offsets):
        sub = df[df.pipeline == pip]
        means = [sub[cols_map[s][0]].mean() for s in sections]
        sems  = [sub[cols_map[s][0]].sem()  for s in sections]
        bars  = ax.bar(x + offset, means, width, color=col, alpha=0.80,
                       label=pip, yerr=sems, capsize=4,
                       error_kw=dict(elinewidth=1.2, ecolor='#666'))
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f'{m:.0f}', ha='center', va='bottom', fontsize=9,
                    color=col, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(sections, fontsize=11)
    ax.set_ylabel("Mean Word Count (± SEM)", **FONT_LABEL)
    ax.set_title("Words per Section", **FONT_TITLE, pad=12)
    ax.legend(fontsize=10, framealpha=0.85, edgecolor='#ccc')
    ax.spines['left'].set_color('#ccc')
    ax.spines['bottom'].set_color('#ccc')
    fig.tight_layout()
    fig.savefig(out / "plot_section_words.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[ok] {out / 'plot_section_words.png'}")

# ── Summary stats table ───────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    print("\n── Summary Statistics ──────────────────────────────────────────")
    for pip in ["Chat-based", "Agentic"]:
        sub = df[df.pipeline == pip]
        print(f"\n{pip}  (n={len(sub)} repo reports)")
        for col, label in [
            ("word_count",          "Word count"),
            ("section_goal_words",  "Goal section words"),
            ("section_dev_words",   "Dev section words"),
            ("link_count",          "Links"),
            ("sections_present",    "Sections present"),
        ]:
            print(f"  {label:28s}  mean={sub[col].mean():.1f}  "
                  f"median={sub[col].median():.1f}  "
                  f"sd={sub[col].std():.1f}  "
                  f"min={sub[col].min():.0f}  max={sub[col].max():.0f}")

    print("\n── Mann-Whitney U Tests ─────────────────────────────────────────")
    chat  = df[df.pipeline == "Chat-based"]
    agent = df[df.pipeline == "Agentic"]
    for col, label in [
        ("word_count",         "Word count"),
        ("section_goal_words", "Goal section words"),
        ("section_dev_words",  "Dev section words"),
        ("link_count",         "Links"),
    ]:
        u, p = stats.mannwhitneyu(chat[col], agent[col], alternative='two-sided')
        print(f"  {label:28s}  U={u:.0f}  p={p:.4f}  {'*' if p<0.05 else 'ns'}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", required=True,
                    help="Path to the reports/ folder containing .md files")
    ap.add_argument("--out-dir", default="./plots",
                    help="Output directory for plots (default: ./plots)")
    args = ap.parse_args()

    reports_dir = Path(args.reports_dir)
    out_dir     = Path(args.out_dir)

    if not reports_dir.exists():
        print(f"[err] reports-dir not found: {reports_dir}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Loading reports from {reports_dir} ...")
    df = load_reports(reports_dir)

    if df.empty:
        print("[err] No matching reports found. Exiting.")
        sys.exit(1)

    print(f"[info] Loaded {len(df)} repo-level reports  "
          f"({(df.pipeline=='Chat-based').sum()} chat-based, "
          f"{(df.pipeline=='Agentic').sum()} agentic)")

    print_summary(df)

    print("\n[info] Generating plots ...")
    np.random.seed(42)
    plot_word_count(df, out_dir)
    plot_section_presence(df, out_dir)
    plot_link_usage(df, out_dir)
    plot_section_words(df, out_dir)

    # Save parsed data
    csv_path = out_dir / "report_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"[ok] {csv_path}")
    print(f"\n[done] All outputs written to {out_dir}/")

if __name__ == "__main__":
    main()