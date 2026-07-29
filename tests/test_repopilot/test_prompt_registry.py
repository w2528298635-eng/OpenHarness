import pytest

from openharness.repopilot.prompt_registry import PromptRegistry, PromptTemplate


def test_prompt_registry_renders_exact_version_deterministically() -> None:
    registry = PromptRegistry()
    registry.register(
        PromptTemplate(
            name="analyze",
            version="2",
            template="Issue: {issue}\nContext: {context}",
            required_variables=frozenset({"issue", "context"}),
        )
    )

    first = registry.render("analyze", version="2", issue="bug", context="app.py")
    second = registry.render("analyze", version="2", context="app.py", issue="bug")

    assert first == second == "Issue: bug\nContext: app.py"
    assert registry.get("analyze", "2").version == "2"


def test_prompt_registry_rejects_unknown_template_and_variable_mismatch() -> None:
    registry = PromptRegistry()
    registry.register(
        PromptTemplate(
            name="plan",
            version="2",
            template="{issue}",
            required_variables=frozenset({"issue"}),
        )
    )

    with pytest.raises(KeyError, match="unknown prompt"):
        registry.get("missing", "2")
    with pytest.raises(ValueError, match="missing"):
        registry.render("plan", version="2")
    with pytest.raises(ValueError, match="unexpected"):
        registry.render("plan", version="2", issue="bug", extra="x")
