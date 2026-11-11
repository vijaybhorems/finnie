"""Add a merged pull request to CHANGELOG.md under "## Unreleased".

Run by .github/workflows/changelog.yml when a PR merges into main. Reads the
PR from environment variables (never from the command line, so a title or
description can't inject shell):

    PR_NUMBER=36 PR_TITLE="..." PR_URL="https://..." PR_BODY="..." \
        python scripts/changelog_add.py [CHANGELOG.md]

The entry text and type come from a `Changelog:` line in the PR description
when there is one (`Changelog: Fixed: ...`, or `Changelog: skip`), else from
the title. Running it twice for the same PR changes nothing.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

TYPES = ("Added", "Changed", "Fixed")
UNRELEASED = "## Unreleased"
_DIRECTIVE = re.compile(r"^\s*changelog:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_TYPED = re.compile(r"^(added|changed|fixed)\s*:\s*(.+)$", re.IGNORECASE)
_CONVENTIONAL = re.compile(r"^(feat|fix|chore|docs|refactor|test)(\([^)]*\))?\s*:\s*", re.IGNORECASE)


def guess_type(title: str) -> str:
    first = (title.split() or [""])[0].lower().rstrip(":")
    if first.startswith("fix"):
        return "Fixed"
    if first in {"add", "adds", "added", "feat", "introduce", "introduces"}:
        return "Added"
    return "Changed"


def entry_for(title: str, body: str) -> tuple[str, str] | None:
    """(type, text) for this PR, or None when it opts out with `Changelog: skip`."""
    match = _DIRECTIVE.search(body or "")
    if match:
        value = match.group(1)
        if value.lower() == "skip":
            return None
        typed = _TYPED.match(value)
        if typed:
            return typed.group(1).capitalize(), typed.group(2).strip()
        return guess_type(title), value
    text = _CONVENTIONAL.sub("", title.strip())
    return guess_type(title), text[:1].upper() + text[1:]


def add_entry(changelog: str, number: int, title: str, url: str, body: str) -> str | None:
    """The changelog with this PR added, or None when there is nothing to change."""
    if f"[#{number}]" in changelog:
        return None  # already recorded (re-run, or backfilled by hand)
    entry = entry_for(title, body)
    if entry is None:
        return None
    kind, text = entry
    line = f"- {text} ([#{number}]({url}))\n"

    if UNRELEASED not in changelog:
        raise ValueError(f"no '{UNRELEASED}' heading in the changelog")
    head, rest = changelog.split(UNRELEASED, 1)
    nxt = rest.find("\n## ")
    section, tail = (rest, "") if nxt == -1 else (rest[: nxt + 1], rest[nxt + 1 :])

    heading = f"### {kind}\n"
    if heading in section:
        section = section.replace(heading, heading + line, 1)  # newest first
    else:
        # Keep Added, Changed, Fixed order: insert before the first later type.
        later = [section.find(f"### {t}\n") for t in TYPES[TYPES.index(kind) + 1 :]]
        later = [i for i in later if i != -1]
        block = f"{heading}{line}\n"
        if later:
            at = min(later)
            section = section[:at] + block + section[at:]
        else:
            section = section.rstrip("\n") + "\n\n" + block
    return head + UNRELEASED + section + tail


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "CHANGELOG.md")
    updated = add_entry(
        path.read_text(),
        int(os.environ["PR_NUMBER"]),
        os.environ["PR_TITLE"],
        os.environ["PR_URL"],
        os.environ.get("PR_BODY", ""),
    )
    if updated is None:
        print("changelog unchanged")
        return 0
    path.write_text(updated)
    print(f"added #{os.environ['PR_NUMBER']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
