"""scripts/changelog_add.py: how a merged PR lands in CHANGELOG.md."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("changelog_add", _ROOT / "scripts" / "changelog_add.py")
cl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cl)

URL = "https://github.com/o/r/pull/"
BASE = "# Changelog\n\nIntro.\n\n## Unreleased\n\n## 2026-11\n\n### Added\n- Old ([#36](x))\n"


def add(changelog: str, n: int, title: str, body: str = "") -> str | None:
    return cl.add_entry(changelog, n, title, f"{URL}{n}", body)


def unreleased(changelog: str) -> str:
    return changelog.split("## Unreleased", 1)[1].split("\n## ", 1)[0]


class TestEntryText:
    @pytest.mark.parametrize("title,kind", [
        ("Fix the tab bar", "Fixed"), ("Fix: Dockerfile", "Fixed"), ("Add CI", "Added"),
        ("feat: life events", "Added"), ("Redesign the UI", "Changed"), ("", "Changed"),
    ])
    def test_type_guessed_from_title(self, title, kind):
        assert cl.guess_type(title) == kind

    def test_title_used_without_directive(self):
        assert cl.entry_for("fix: escape dollars", "Some summary") == ("Fixed", "Escape dollars")

    def test_typed_directive_wins(self):
        body = "## Summary\nstuff\n\nChangelog: Added: Voice answers\n"
        assert cl.entry_for("Fix something", body) == ("Added", "Voice answers")

    def test_untyped_directive_keeps_guessed_type(self):
        assert cl.entry_for("Fix the stream", "changelog: Answers no longer repeat") == (
            "Fixed", "Answers no longer repeat")

    def test_skip(self):
        assert cl.entry_for("Fix tests", "Changelog: skip") is None


class TestAddEntry:
    def test_first_entry_creates_its_heading(self):
        out = add(BASE, 37, "Fix the tab bar")
        assert unreleased(out) == f"\n\n### Fixed\n- Fix the tab bar ([#37]({URL}37))\n"
        assert out.endswith("- Old ([#36](x))\n")  # history untouched

    def test_newest_first_within_a_type(self):
        out = add(add(BASE, 37, "Fix one"), 38, "Fix two")
        section = unreleased(out)
        assert section.index("#38") < section.index("#37")
        assert section.count("### Fixed") == 1

    def test_types_stay_in_order(self):
        out = add(add(add(BASE, 37, "Fix a"), 38, "Add b"), 39, "Rename c")
        section = unreleased(out)
        assert section.index("### Added") < section.index("### Changed") < section.index("### Fixed")

    def test_same_pr_twice_is_a_no_op(self):
        once = add(BASE, 37, "Fix a")
        assert add(once, 37, "Fix a") is None

    def test_backfilled_pr_is_not_added_again(self):
        assert add(BASE, 36, "Voice spike") is None

    def test_skip_changes_nothing(self):
        assert add(BASE, 37, "Fix tests", "Changelog: skip") is None

    def test_missing_unreleased_heading_fails_loudly(self):
        with pytest.raises(ValueError):
            add("# Changelog\n", 37, "Fix a")


def test_real_changelog_accepts_an_entry():
    real = (_ROOT / "CHANGELOG.md").read_text()
    out = add(real, 9999, "Add something new")
    line = f"- Add something new ([#9999]({URL}9999))\n"
    assert line in unreleased(out)
    # Only the new line (and, if Unreleased had no Added heading yet, that heading) was added.
    assert out.replace(line, "", 1) in (real, real.replace("## Unreleased\n\n", "## Unreleased\n\n### Added\n\n", 1))
