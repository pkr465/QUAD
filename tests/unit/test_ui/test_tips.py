"""Tests for the tips system (Phase F)."""

from __future__ import annotations

import pytest

from quad.tips import (
    Tip,
    all_contexts,
    catalogue_size,
    format_tips_markdown,
    get_general_tips,
    get_tips_for,
)


class TestTip:
    def test_to_markdown_with_icon(self) -> None:
        t = Tip(text="hello", context="general", level="tip")
        out = t.to_markdown()
        assert "💡" in out
        assert "hello" in out

    def test_warning_uses_warning_icon(self) -> None:
        t = Tip(text="watch out", context="convert", level="warning")
        assert "⚠️" in t.to_markdown()

    def test_info_uses_info_icon(self) -> None:
        t = Tip(text="fyi", context="detect", level="info")
        assert "ℹ️" in t.to_markdown()

    def test_link_appended(self) -> None:
        t = Tip(text="hello", context="general", link="docs/X.md")
        out = t.to_markdown()
        assert "docs/X.md" in out


class TestGetTipsFor:
    def test_returns_n_tips(self) -> None:
        tips = get_tips_for("convert", n=2)
        assert 0 < len(tips) <= 2

    def test_unknown_context_falls_back_to_general(self) -> None:
        tips = get_tips_for("totally-fake-context", n=2)
        assert len(tips) > 0
        assert all(t.context == "general" for t in tips)

    def test_filter_by_level(self) -> None:
        tips = get_tips_for("convert", n=5, level="warning")
        assert all(t.level == "warning" for t in tips)

    def test_seed_makes_output_deterministic(self) -> None:
        a = get_tips_for("convert", n=2, seed=42)
        b = get_tips_for("convert", n=2, seed=42)
        assert [t.text for t in a] == [t.text for t in b]

    def test_general_tips_helper(self) -> None:
        tips = get_general_tips(n=1)
        assert len(tips) == 1
        assert tips[0].context == "general"


class TestCatalogue:
    def test_catalogue_has_entries_per_context(self) -> None:
        contexts = all_contexts()
        # Should have at least: general, detect, convert, profile,
        # orchestrate, codegen, serve
        for c in ("general", "detect", "convert", "profile", "orchestrate", "codegen", "serve"):
            assert c in contexts, f"Missing tips for context: {c}"

    def test_catalogue_size_minimum(self) -> None:
        # We have 25+ tips
        assert catalogue_size() >= 20


class TestFormatTipsMarkdown:
    def test_empty_returns_empty_string(self) -> None:
        assert format_tips_markdown([]) == ""

    def test_renders_bulleted_list(self) -> None:
        tips = [
            Tip(text="one", context="general"),
            Tip(text="two", context="general"),
        ]
        out = format_tips_markdown(tips)
        assert "**Tips:**" in out
        assert "- " in out
        assert "one" in out
        assert "two" in out
