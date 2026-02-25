from __future__ import annotations

import re

SKILL_ALIASES: dict[str, str] = {
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "k8s": "kubernetes",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "react.js": "react",
    "reactjs": "react",
    "node.js": "node",
    "nodejs": "node",
    "vue.js": "vue",
    "vuejs": "vue",
    "angular.js": "angular",
    "angularjs": "angular",
    "c++": "cpp",
    "c#": "csharp",
    "dot net": "dotnet",
    ".net": "dotnet",
    "aws": "amazon web services",
    "gcp": "google cloud platform",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "ci/cd": "cicd",
    "ci cd": "cicd",
}

SUFFIXES_TO_STRIP: list[str] = [
    "developer",
    "engineer",
    "programming",
    "development",
    "framework",
    "language",
    "library",
]


class SkillNormalizer:
    """Normalizes skill names for accurate Jaccard similarity calculation."""

    def __init__(self, aliases: dict[str, str] | None = None) -> None:
        # WHY: Allow custom aliases for domain-specific normalization
        self._aliases = aliases or SKILL_ALIASES

    def normalize(self, skill: str) -> str:
        """
        Normalize a single skill string.
        Pipeline: lowercase -> strip -> remove versions -> strip suffixes -> alias map
        """
        s = skill.lower().strip()
        if not s:
            return s

        # Step 1: Remove version numbers (e.g., "python 3.10" -> "python")
        s = re.sub(r"\s*\d+(\.\d+)*\s*$", "", s).strip()

        # Step 2: Strip suffixes (e.g., "python developer" -> "python")
        for suffix in SUFFIXES_TO_STRIP:
            if s.endswith(f" {suffix}"):
                s = s[: -(len(suffix) + 1)].strip()

        # Step 3: Alias mapping (e.g., "js" -> "javascript")
        s = self._aliases.get(s, s)

        return s

    def normalize_set(self, skills: list[str]) -> set[str]:
        """Normalize a list of skills, returning deduplicated set."""
        return {self.normalize(s) for s in skills if s.strip()}
