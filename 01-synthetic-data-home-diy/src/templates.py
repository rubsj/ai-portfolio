"""Prompt templates (v1) for DIY repair record generation.

Each template pairs a repair category with an expert persona. The generator
calls build_messages() to get the system + user prompts, then passes them
to Instructor. Instructor appends the Pydantic JSON schema automatically.

Java/TS parallel: like a template factory pattern — data-driven config
(the dicts) plus builder functions that interpolate at call time. Similar
to Handlebars/Mustache templates or Spring's MessageSource.

v2 templates (created during the correction phase) will add explicit
instructions to prevent the most common failure modes per category.
"""

from __future__ import annotations

from typing import NamedTuple

from src.schemas import Category, Difficulty


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class CategoryTemplate(NamedTuple):
    """Immutable config for one category's prompt template.

    Java/TS parallel: like a TypeScript `as const` object or a Java record.
    NamedTuple gives immutability + tuple unpacking + named field access.
    """

    persona: str
    emphasis: str


# ---------------------------------------------------------------------------
# Category → Persona mapping (PRD Section 4a table)
# Using a dict keyed on Category Literal ensures exhaustive coverage —
# if a new category is added to the Literal type, the linter won't catch
# a missing entry, but the KeyError at runtime will.
# ---------------------------------------------------------------------------
CATEGORY_TEMPLATES: dict[Category, CategoryTemplate] = {
    "appliance_repair": CategoryTemplate(
        persona=(
            "Expert home appliance repair technician "
            "with 20+ years of experience"
        ),
        emphasis="Technical details and practical homeowner solutions",
    ),
    "plumbing_repair": CategoryTemplate(
        persona="Professional plumber with extensive residential experience",
        emphasis="Safety for homeowner attempts and realistic solutions",
    ),
    "electrical_repair": CategoryTemplate(
        persona=(
            "Licensed electrician specializing in safe home "
            "electrical repairs"
        ),
        emphasis="Critical safety warnings and when to call professionals",
    ),
    "hvac_maintenance": CategoryTemplate(
        persona="HVAC technician specializing in homeowner maintenance",
        emphasis="Seasonal considerations and maintenance best practices",
    ),
    "general_home_repair": CategoryTemplate(
        persona="Skilled handyperson with general home repair expertise",
        emphasis="Material specifications and practical DIY solutions",
    ),
}


# ---------------------------------------------------------------------------
# Difficulty modifiers (PRD Section 4a)
# These get appended to the user prompt to calibrate complexity.
# ---------------------------------------------------------------------------
DIFFICULTY_MODIFIERS: dict[Difficulty, str] = {
    "beginner": (
        "This is a beginner-level task. Use simple, common household tools "
        "only. Minimal risk involved. Steps should be very detailed and "
        "assume no prior DIY experience."
    ),
    "intermediate": (
        "This is an intermediate-level task. May require some specialty "
        "tools. Some prior DIY experience is assumed. Moderate complexity."
    ),
    "advanced": (
        "This is an advanced-level task. Specialized tools may be needed. "
        "Higher risk involved — for experienced DIY-ers only. You MUST "
        "emphasize when the homeowner should call a professional instead "
        "of attempting the repair themselves."
    ),
}

# Current template version — tracked in GeneratedRecord.template_version
TEMPLATE_VERSION = "v1"


# ---------------------------------------------------------------------------
# Builder functions — used by generator.py to construct Instructor messages
# ---------------------------------------------------------------------------

def build_system_prompt(category: Category) -> str:
    """Build the system message for a given category.

    Instructor appends the Pydantic JSON schema automatically, so this only
    provides persona context and emphasis guidance.

    Java/TS parallel: like a factory method that returns a configured string.
    """
    template = CATEGORY_TEMPLATES[category]
    # category uses underscores ("plumbing_repair") — replace for readability
    display_category = category.replace("_", " ")
    return (
        f"You are a {template.persona}. "
        f"You specialize in {display_category}.\n\n"
        f"Emphasis: {template.emphasis}."
    )


def build_user_prompt(category: Category, difficulty: Difficulty) -> str:
    """Build the user message requesting a Q&A pair.

    Combines the generation instruction with the difficulty modifier.
    Instructor injects the JSON schema separately, so we focus on
    WHAT to generate, not the output format.
    """
    modifier = DIFFICULTY_MODIFIERS[difficulty]
    display_category = category.replace("_", " ")
    # "an advanced", "an intermediate", "a beginner"
    article = "an" if difficulty[0] in "aeiou" else "a"
    return (
        f"Generate {article} {difficulty}-level DIY repair Q&A for "
        f"{display_category}. The homeowner should be able to follow "
        f"your instructions safely.\n\n"
        f"{modifier}\n\n"
        "Include realistic tools that a homeowner would actually own, "
        "clear step-by-step instructions, and appropriate safety warnings."
    )


def build_messages(
    category: Category, difficulty: Difficulty
) -> list[dict[str, str]]:
    """Build the full message list for an Instructor call.

    Returns the format expected by `client.chat.completions.create(messages=...)`.

    Java/TS parallel: like building a List<ChatMessage> for the OpenAI SDK.
    """
    return [
        {"role": "system", "content": build_system_prompt(category)},
        {"role": "user", "content": build_user_prompt(category, difficulty)},
    ]
