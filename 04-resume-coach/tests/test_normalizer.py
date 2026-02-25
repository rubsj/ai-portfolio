from __future__ import annotations

import pytest

from src.normalizer import SkillNormalizer


class TestSkillNormalizer:
    @pytest.fixture
    def normalizer(self) -> SkillNormalizer:
        return SkillNormalizer()

    @pytest.mark.parametrize("input_skill,expected", [
        ("Python 3.10", "python"),
        ("React 18.2", "react"),
        ("Node.js 20", "node"),
        ("Java 17", "java"),
    ])
    def test_version_removal(self, normalizer, input_skill, expected):
        assert normalizer.normalize(input_skill) == expected

    @pytest.mark.parametrize("input_skill,expected", [
        ("python developer", "python"),
        ("java engineer", "java"),
        ("react framework", "react"),
        ("go programming", "go"),
    ])
    def test_suffix_stripping(self, normalizer, input_skill, expected):
        assert normalizer.normalize(input_skill) == expected

    @pytest.mark.parametrize("input_skill,expected", [
        ("js", "javascript"),
        ("ts", "typescript"),
        ("k8s", "kubernetes"),
        ("py", "python"),
        ("c++", "cpp"),
        ("c#", "csharp"),
        (".net", "dotnet"),
        ("ci/cd", "cicd"),
    ])
    def test_alias_mapping(self, normalizer, input_skill, expected):
        assert normalizer.normalize(input_skill) == expected

    def test_empty_string(self, normalizer):
        assert normalizer.normalize("") == ""

    def test_already_normalized(self, normalizer):
        assert normalizer.normalize("python") == "python"

    def test_unknown_skill_passes_through(self, normalizer):
        assert normalizer.normalize("fortran") == "fortran"

    def test_normalize_set_deduplication(self, normalizer):
        result = normalizer.normalize_set(["Python 3.10", "python", "py"])
        assert result == {"python"}

    def test_normalize_set_empty_strings_filtered(self, normalizer):
        result = normalizer.normalize_set(["python", "", "  "])
        assert result == {"python"}

    def test_combined_version_and_alias(self, normalizer):
        # "React.js 18" -> lowercase "react.js 18" -> version removal "react.js" -> alias "react"
        assert normalizer.normalize("React.js 18") == "react"

    def test_whitespace_only_normalized(self, normalizer):
        assert normalizer.normalize("   ") == ""

    def test_uppercase_normalized(self, normalizer):
        assert normalizer.normalize("PYTHON") == "python"

    def test_mixed_case_alias(self, normalizer):
        assert normalizer.normalize("JS") == "javascript"

    def test_custom_aliases(self):
        custom = SkillNormalizer(aliases={"rb": "ruby"})
        assert custom.normalize("rb") == "ruby"
        # WHY: Custom aliases replace defaults entirely, so default "js" no longer maps
        assert custom.normalize("js") == "js"

    def test_normalize_set_multiple_skills(self, normalizer):
        result = normalizer.normalize_set(["Python 3.10", "JavaScript", "k8s", "Docker"])
        assert "python" in result
        assert "javascript" in result
        assert "kubernetes" in result
        assert "docker" in result

    def test_normalize_set_empty_list(self, normalizer):
        assert normalizer.normalize_set([]) == set()
