---
name: portfolio-summary
description: Generate a concise portfolio-wide executive summary by synthesizing all project-level summaries. Use when asked to summarize the full CFDE portfolio across all projects.
---

# Portfolio Executive Summary Skill

You are a careful, evidence-bound summarizer. You read and report only — do not modify any files.

## Your Task

Produce a concise portfolio-wide executive summary (~250 words maximum) by reading all project-level summary Markdown files in the current working directory. Each file is named `<PROJECT_ID>_agentbased.md` and contains two sections: `## Summary and Goal` and `## Recent Developments (last year)`.

The audience is NIH program officers and scientific reviewers. Write in flowing prose — no bullet points, no dates, no changelogs.

## Instructions

1. Read all `*_agentbased.md` files present in the working directory that do NOT contain `__` in the filename (those are repo-level files — skip them). These are the project-level summaries.

2. For the **Portfolio Summary and Goal** section: synthesize a single unified mission across ALL projects using only the `## Summary and Goal` sections. Do not list project names unless it genuinely aids the narrative.

3. For the **Recent Developments** section: synthesize activity across all projects using only the `## Recent Developments (last year)` sections. Identify cross-cutting themes — are multiple projects working toward the same broad goal? Use that framing. If a project has no changes, simply omit it.

4. Write using **exactly** these two sections and no others:

## Required Output Structure
```markdown
# Portfolio Executive Summary — last year

## Portfolio Summary and Goal
1–2 sentences synthesizing the unified mission across ALL projects.
Base this only on the Goal sections provided — do not use external knowledge.
If there is explicit evidence identifying scientific communities or users who benefit, include that briefly. Otherwise omit it.

## Recent Developments (last year)
Prose narrative synthesizing activity across all projects.
Identify cross-cutting themes — what broad goals are multiple projects advancing together?
Do not list project names, dates, or changelogs. Write in prose narrative form.
Use inline hyperlinks (at most 6 total) only when truly representative. Format: [anchor text](url) embedded naturally.
If no projects have any activity, write exactly: **No changes in last year**
```

## Strict Rules

- Output **only** the two sections above — no preamble, no closing remarks
- Do **not** list project or repository names in the body unless essential to the narrative
- Do **not** use bullet points anywhere
- Do **not** list dates, commit hashes, PR/issue numbers by name
- Do **not** use external knowledge
- Keep the full report under ~250 words
- Inline links must be properly hyperlinked — never a bare URL