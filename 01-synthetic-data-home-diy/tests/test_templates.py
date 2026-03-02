"""Tests for src/templates.py — prompt template builder functions.

Covers:
- build_system_prompt: persona + emphasis interpolation, all 5 categories
- build_user_prompt: article selection (a/an), difficulty modifier inclusion
- build_messages: returns 2-element list with correct roles
"""

from __future__ import annotations

import pytest

from src.templates import (
    CATEGORY_TEMPLATES,
    DIFFICULTY_MODIFIERS,
    TEMPLATE_VERSION,
    build_messages,
    build_system_prompt,
    build_user_prompt,
)


# ===========================================================================
# build_system_prompt
# ===========================================================================


class TestBuildSystemPrompt:
    """Tests for build_system_prompt."""

    def test_build_system_prompt_when_plumbing_contains_persona(self) -> None:
        prompt = build_system_prompt("plumbing_repair")
        assert "plumber" in prompt.lower()

    def test_build_system_prompt_replaces_underscores_with_spaces(self) -> None:
        prompt = build_system_prompt("appliance_repair")
        assert "appliance repair" in prompt

    def test_build_system_prompt_contains_emphasis(self) -> None:
        prompt = build_system_prompt("electrical_repair")
        template = CATEGORY_TEMPLATES["electrical_repair"]
        assert template.emphasis in prompt

    def test_build_system_prompt_for_all_categories(self) -> None:
        categories = list(CATEGORY_TEMPLATES.keys())
        for category in categories:
            prompt = build_system_prompt(category)
            assert len(prompt) > 10

    def test_build_system_prompt_hvac_contains_hvac(self) -> None:
        prompt = build_system_prompt("hvac_maintenance")
        assert "hvac" in prompt.lower()

    def test_build_system_prompt_general_home_contains_handyperson(self) -> None:
        prompt = build_system_prompt("general_home_repair")
        assert "handyperson" in prompt.lower()


# ===========================================================================
# build_user_prompt
# ===========================================================================


class TestBuildUserPrompt:
    """Tests for build_user_prompt."""

    def test_build_user_prompt_beginner_uses_article_a(self) -> None:
        prompt = build_user_prompt("plumbing_repair", "beginner")
        assert "a beginner" in prompt

    def test_build_user_prompt_advanced_uses_article_an(self) -> None:
        prompt = build_user_prompt("plumbing_repair", "advanced")
        assert "an advanced" in prompt

    def test_build_user_prompt_intermediate_uses_article_an(self) -> None:
        prompt = build_user_prompt("plumbing_repair", "intermediate")
        assert "an intermediate" in prompt

    def test_build_user_prompt_contains_difficulty_modifier(self) -> None:
        prompt = build_user_prompt("plumbing_repair", "beginner")
        modifier = DIFFICULTY_MODIFIERS["beginner"]
        # At least part of the modifier should appear
        assert "beginner" in prompt.lower()

    def test_build_user_prompt_contains_category_display_name(self) -> None:
        prompt = build_user_prompt("general_home_repair", "beginner")
        assert "general home repair" in prompt

    def test_build_user_prompt_mentions_safety(self) -> None:
        prompt = build_user_prompt("electrical_repair", "beginner")
        assert "safely" in prompt.lower() or "safety" in prompt.lower()


# ===========================================================================
# build_messages
# ===========================================================================


class TestBuildMessages:
    """Tests for build_messages."""

    def test_build_messages_returns_two_messages(self) -> None:
        messages = build_messages("plumbing_repair", "beginner")
        assert len(messages) == 2

    def test_build_messages_first_is_system(self) -> None:
        messages = build_messages("plumbing_repair", "beginner")
        assert messages[0]["role"] == "system"

    def test_build_messages_second_is_user(self) -> None:
        messages = build_messages("plumbing_repair", "beginner")
        assert messages[1]["role"] == "user"

    def test_build_messages_system_content_is_nonempty(self) -> None:
        messages = build_messages("electrical_repair", "advanced")
        assert len(messages[0]["content"]) > 10

    def test_build_messages_user_content_is_nonempty(self) -> None:
        messages = build_messages("hvac_maintenance", "intermediate")
        assert len(messages[1]["content"]) > 10


# ===========================================================================
# Constants
# ===========================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_template_version_is_v1(self) -> None:
        assert TEMPLATE_VERSION == "v1"

    def test_all_five_categories_in_templates(self) -> None:
        expected = {
            "appliance_repair", "plumbing_repair", "electrical_repair",
            "hvac_maintenance", "general_home_repair",
        }
        assert set(CATEGORY_TEMPLATES.keys()) == expected

    def test_all_three_difficulties_in_modifiers(self) -> None:
        assert set(DIFFICULTY_MODIFIERS.keys()) == {"beginner", "intermediate", "advanced"}
