"""Tests for the manage_skill tool."""

from pathlib import Path

import pytest

from nanobot.agent.skills import SkillsLoader
from nanobot.agent.tools.manage_skill import ManageSkillTool

VALID_SKILL_CONTENT = """\
---
name: test-skill
description: A test skill for unit tests.
---

# Test Skill

Do the thing.
"""


def _make_tool(tmp_path: Path) -> ManageSkillTool:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    return ManageSkillTool(workspace_dir=workspace, skills_loader=loader)


def _write_skill(base: Path, name: str, description: str = "A skill.") -> Path:
    skill_dir = base / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n"
    path = skill_dir / "SKILL.md"
    path.write_text(content, encoding="utf-8")
    return path


class TestManageSkillBasics:
    def test_name(self, tmp_path):
        tool = _make_tool(tmp_path)
        assert tool.name == "manage_skill"

    def test_not_read_only(self, tmp_path):
        tool = _make_tool(tmp_path)
        assert tool.read_only is False

    def test_schema_has_required_action(self, tmp_path):
        tool = _make_tool(tmp_path)
        schema = tool.parameters
        assert "action" in schema.get("required", [])


class TestListAction:
    async def test_list_workspace_skills(self, tmp_path):
        tool = _make_tool(tmp_path)
        workspace = tmp_path / "ws"
        _write_skill(workspace, "alpha", "First skill")
        _write_skill(workspace, "beta", "Second skill")
        result = await tool.execute(action="list")
        assert "alpha" in result
        assert "First skill" in result
        assert "beta" in result
        assert "Second skill" in result

    async def test_list_empty_workspace(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = await tool.execute(action="list")
        assert "No workspace skills found" in result

    async def test_list_does_not_include_builtin(self, tmp_path):
        tool = _make_tool(tmp_path)
        builtin = tmp_path / "builtin"
        _write_skill(builtin, "builtin-only", "A builtin skill")
        result = await tool.execute(action="list")
        assert "builtin-only" not in result


class TestCreateAction:
    async def test_create_skill(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = await tool.execute(action="create", name="my-skill", content=VALID_SKILL_CONTENT)
        assert "created" in result.lower()
        skill_file = tmp_path / "ws" / "skills" / "my-skill" / "SKILL.md"
        assert skill_file.exists()
        assert skill_file.read_text(encoding="utf-8") == VALID_SKILL_CONTENT

    async def test_create_bad_frontmatter(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = await tool.execute(action="create", name="my-skill", content="# No frontmatter\n")
        assert "frontmatter" in result.lower()

    async def test_create_invalid_name_uppercase(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = await tool.execute(action="create", name="BadName", content=VALID_SKILL_CONTENT)
        assert "kebab-case" in result.lower()

    async def test_create_duplicate_workspace(self, tmp_path):
        tool = _make_tool(tmp_path)
        workspace = tmp_path / "ws"
        _write_skill(workspace, "existing-skill")
        result = await tool.execute(action="create", name="existing-skill", content=VALID_SKILL_CONTENT)
        assert "already exists" in result.lower()

    async def test_create_path_traversal(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = await tool.execute(action="create", name="../etc", content=VALID_SKILL_CONTENT)
        assert "invalid" in result.lower() or "must not" in result.lower()


class TestEditAction:
    async def test_edit_skill(self, tmp_path):
        tool = _make_tool(tmp_path)
        workspace = tmp_path / "ws"
        _write_skill(workspace, "edit-me")
        new_content = "---\nname: edit-me\ndescription: Updated.\n---\n\n# Updated\n"
        result = await tool.execute(action="edit", name="edit-me", content=new_content)
        assert "updated" in result.lower()
        skill_file = workspace / "skills" / "edit-me" / "SKILL.md"
        assert skill_file.read_text(encoding="utf-8") == new_content

    async def test_edit_nonexistent(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = await tool.execute(action="edit", name="ghost-skill", content=VALID_SKILL_CONTENT)
        assert "not found" in result.lower()

    async def test_edit_bad_frontmatter(self, tmp_path):
        tool = _make_tool(tmp_path)
        workspace = tmp_path / "ws"
        _write_skill(workspace, "edit-me")
        result = await tool.execute(action="edit", name="edit-me", content="no frontmatter here")
        assert "frontmatter" in result.lower()


class TestPatchAction:
    async def test_patch_skill(self, tmp_path):
        tool = _make_tool(tmp_path)
        workspace = tmp_path / "ws"
        _write_skill(workspace, "patch-me", "Original description")
        result = await tool.execute(
            action="patch",
            name="patch-me",
            old_string="Original description",
            new_string="Patched description",
        )
        assert "patched" in result.lower()
        skill_file = workspace / "skills" / "patch-me" / "SKILL.md"
        assert "Patched description" in skill_file.read_text(encoding="utf-8")

    async def test_patch_old_string_not_found(self, tmp_path):
        tool = _make_tool(tmp_path)
        workspace = tmp_path / "ws"
        _write_skill(workspace, "patch-me")
        result = await tool.execute(
            action="patch",
            name="patch-me",
            old_string="DOES NOT EXIST",
            new_string="replacement",
        )
        assert "not found" in result.lower()

    async def test_patch_multiple_matches(self, tmp_path):
        tool = _make_tool(tmp_path)
        workspace = tmp_path / "ws"
        skill_dir = workspace / "skills" / "patch-me"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: patch-me\ndescription: dup dup\n---\n\ndup dup\n",
            encoding="utf-8",
        )
        result = await tool.execute(
            action="patch",
            name="patch-me",
            old_string="dup",
            new_string="single",
        )
        assert "3" in result or "4" in result  # 3 or 4 occurrences of "dup"


class TestDeleteAction:
    async def test_delete_skill(self, tmp_path):
        tool = _make_tool(tmp_path)
        workspace = tmp_path / "ws"
        _write_skill(workspace, "bye-skill")
        result = await tool.execute(action="delete", name="bye-skill")
        assert "deleted" in result.lower()
        assert not (workspace / "skills" / "bye-skill").exists()

    async def test_delete_nonexistent(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = await tool.execute(action="delete", name="no-such-skill")
        assert "not found" in result.lower()

    async def test_delete_builtin_rejected(self, tmp_path):
        tool = _make_tool(tmp_path)
        builtin = tmp_path / "builtin"
        _write_skill(builtin, "builtin-skill")
        result = await tool.execute(action="delete", name="builtin-skill")
        assert "builtin" in result.lower()


class TestSecurityBoundaries:
    async def test_name_with_path_traversal_dots(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = await tool.execute(action="create", name="..%2fetc", content=VALID_SKILL_CONTENT)
        assert "invalid" in result.lower() or "kebab-case" in result.lower()

    async def test_name_with_slash(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = await tool.execute(action="create", name="sub/skill", content=VALID_SKILL_CONTENT)
        assert "invalid" in result.lower() or "must not" in result.lower()
