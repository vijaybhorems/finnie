"""Knowledge-base digest for the shared, cached system prompt.

Every agent's system prompt opens with the same block so that one prompt-cache
entry serves all six agents. That block must clear the model's minimum
cacheable length (1,024 tokens on Claude Sonnet 5) or it is silently never
cached. This digest — a table of contents of Finnie's knowledge base, built
from the articles themselves — is most of that block, and it also tells each
agent what the knowledge base covers.

The output must be byte-identical across requests and processes while the
knowledge base is unchanged: any drift would invalidate the cache for every
agent. Files are read in sorted order and nothing time- or host-dependent is
included. Editing an article changes the digest, which correctly invalidates
the cached prefix.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from src.core.config import get_settings

_MAX_SUMMARY_WORDS = 60
_MAX_SECTIONS = 6

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_PARENTHETICAL = re.compile(r"\([^)]*\)")
_WHITESPACE = re.compile(r"\s+")


def _clean(line: str) -> str:
    return _WHITESPACE.sub(" ", line.lstrip("#").strip())


def _is_section_heading(line: str) -> bool:
    """True for the articles' ALL-CAPS section headings.

    Parentheticals are ignored when judging case, so "LEADING INDICATORS
    (Predict future economic activity)" counts; mixed-case lines such as
    "What Is an Index Fund?" or "SPY, VOO, IVV: Track the S&P 500" do not.
    """
    core = _PARENTHETICAL.sub("", line)
    letters = [c for c in core if c.isalpha()]
    if len(letters) < 4 or len(line) > 90:
        return False
    return sum(c.isupper() for c in letters) / len(letters) >= 0.8


def _summarize(text: str) -> tuple[str, str, list[str]]:
    """Return (title, summary, section headings) for one article."""
    paragraphs = _PARAGRAPH_SPLIT.split(text.strip())
    title = _clean(paragraphs[0].splitlines()[0]) if paragraphs else ""

    summary = ""
    if len(paragraphs) > 1 and not _is_section_heading(paragraphs[1].splitlines()[0]):
        words = _clean(paragraphs[1].replace("\n", " ")).split(" ")
        summary = " ".join(words[:_MAX_SUMMARY_WORDS])
        if len(words) > _MAX_SUMMARY_WORDS:
            summary += " …"

    sections: list[str] = []
    for raw in text.splitlines()[1:]:
        line = _clean(raw).rstrip(":")
        if line and _is_section_heading(line) and line not in sections:
            sections.append(line)
        if len(sections) == _MAX_SECTIONS:
            break

    return title, summary, sections


def _category_label(directory: str) -> str:
    return directory.replace("_", " ").title()


@lru_cache(maxsize=1)
def knowledge_base_digest() -> str:
    """Markdown overview of every knowledge-base article, grouped by category."""
    kb_path = Path(get_settings().knowledge_base_path)
    files = (
        sorted(p for p in kb_path.rglob("*") if p.is_file() and p.suffix in (".txt", ".md"))
        if kb_path.exists()
        else []
    )

    lines = [
        "# Finnie knowledge base",
        "",
        "These are Finnie's curated reference articles. For each question the most relevant "
        "passages are retrieved into the request context; this overview tells you what the "
        "knowledge base covers, so you can connect an answer to related topics.",
    ]

    current_category = None
    for path in files:
        category = path.parent.name
        if category != current_category:
            lines += ["", f"## {_category_label(category)}"]
            current_category = category

        title, summary, sections = _summarize(path.read_text(encoding="utf-8"))
        entry = f"- **{title}**"
        if summary:
            entry += f" — {summary}"
        if sections:
            entry += f" Sections: {'; '.join(sections)}."
        lines.append(entry)

    return "\n".join(lines)
