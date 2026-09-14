"""skills-view: a read-only skills toolset an embedded persona can hold without skill_manage."""
from hermes_cli.tools_config import CONFIGURABLE_TOOLSETS, _DEFAULT_OFF_TOOLSETS, _get_platform_tools
from toolsets import TOOLSETS, resolve_toolset


def test_skills_view_is_read_only():
    assert TOOLSETS["skills-view"]["tools"] == ["skills_list", "skill_view"]
    assert "skill_manage" not in resolve_toolset("skills-view")
    assert set(resolve_toolset("skills-view")) < set(resolve_toolset("skills"))


def test_skills_view_is_configurable_and_default_off():
    assert "skills-view" in {key for key, _, _ in CONFIGURABLE_TOOLSETS}
    assert "skills-view" in _DEFAULT_OFF_TOOLSETS


def test_explicit_platform_list_enables_skills_view_without_skills():
    enabled = _get_platform_tools({"platform_toolsets": {"api_server": ["skills-view", "todo"]}}, "api_server")
    assert "skills-view" in enabled
    assert "skills" not in enabled


def test_default_platform_lists_never_gain_skills_view():
    for platform in ("api_server", "cli"):
        assert "skills-view" not in _get_platform_tools({}, platform)
