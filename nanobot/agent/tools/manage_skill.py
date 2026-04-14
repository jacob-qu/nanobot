"""Manage skill tool: create, edit, patch, delete, and list workspace skills."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import StringSchema, tool_parameters_schema

if TYPE_CHECKING:
    from nanobot.agent.skills import SkillsLoader

_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
_NAME_MAX = 64
_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---", re.DOTALL)


@tool_parameters(
    tool_parameters_schema(
        action=StringSchema("Action: create | edit | patch | delete | list"),
        name=StringSchema("Skill name (kebab-case, max 64 chars)"),
        content=StringSchema("Full SKILL.md content including YAML frontmatter (for create/edit)"),
        old_string=StringSchema("Text to replace (for patch)"),
        new_string=StringSchema("Replacement text (for patch)"),
        required=["action"],
    )
)
class ManageSkillTool(Tool):
    """Tool to manage workspace skills at runtime."""

    def __init__(self, workspace_dir: Path, skills_loader: SkillsLoader):
        self._workspace_dir = workspace_dir
        self._loader = skills_loader
        self._skills_dir = workspace_dir / "skills"

    @property
    def name(self) -> str:
        return "manage_skill"

    @property
    def description(self) -> str:
        return (
            "Manage workspace skills: list existing skills, or create, edit, patch, "
            "or delete a skill by name. Skills are reusable instruction sets stored "
            "as SKILL.md files."
        )

    @property
    def read_only(self) -> bool:
        return False

    async def execute(
        self,
        action: str,
        name: str | None = None,
        content: str | None = None,
        old_string: str | None = None,
        new_string: str | None = None,
        **kwargs: Any,
    ) -> str:
        if action == "list":
            return self._do_list()
        if action == "create":
            return self._do_create(name, content)
        if action == "edit":
            return self._do_edit(name, content)
        if action == "patch":
            return self._do_patch(name, old_string, new_string)
        if action == "delete":
            return self._do_delete(name)
        return f"Unknown action: {action!r}. Must be one of: list, create, edit, patch, delete."

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_name(self, name: str | None) -> str | None:
        """Return error message if name is invalid, else None."""
        if not name:
            return "Parameter 'name' is required."
        if ".." in name or "/" in name or "\\" in name:
            return f"Invalid skill name {name!r}: must not contain '..', '/', or '\\'."
        if not _NAME_RE.match(name):
            return (
                f"Invalid skill name {name!r}: must be kebab-case "
                "(lowercase letters, digits, hyphens; no leading/trailing hyphen)."
            )
        if len(name) > _NAME_MAX:
            return f"Invalid skill name {name!r}: max {_NAME_MAX} characters."
        return None

    def _validate_frontmatter(self, content: str | None) -> str | None:
        """Return error message if content/frontmatter is invalid, else None."""
        if not content:
            return "Parameter 'content' is required."
        if not content.startswith("---\n"):
            return "Content must start with YAML frontmatter (--- ... ---)."
        if not _FRONTMATTER_RE.match(content):
            return "Content must have valid YAML frontmatter (--- ... ---)."
        # Parse frontmatter fields
        meta = self._parse_frontmatter_fields(content)
        if "name" not in meta:
            return "Frontmatter must contain a 'name' field."
        if "description" not in meta:
            return "Frontmatter must contain a 'description' field."
        return None

    def _parse_frontmatter_fields(self, content: str) -> dict[str, str]:
        match = _FRONTMATTER_RE.match(content)
        if not match:
            return {}
        body = match.group(0)[4:]  # strip leading "---\n"
        body = body.rsplit("---", 1)[0]
        fields: dict[str, str] = {}
        for line in body.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key.strip()] = value.strip().strip("\"'")
        return fields

    def _workspace_skill_exists(self, name: str) -> bool:
        return (self._skills_dir / name / "SKILL.md").exists()

    def _builtin_skill_exists(self, name: str) -> bool:
        builtin_dir = self._loader.builtin_skills
        if builtin_dir and builtin_dir.exists():
            return (builtin_dir / name / "SKILL.md").exists()
        return False

    # ------------------------------------------------------------------
    # Action implementations
    # ------------------------------------------------------------------

    def _do_list(self) -> str:
        if not self._skills_dir.exists():
            return "No workspace skills found."
        entries = [
            d for d in self._skills_dir.iterdir()
            if d.is_dir() and (d / "SKILL.md").exists()
        ]
        if not entries:
            return "No workspace skills found."
        lines: list[str] = []
        for entry in sorted(entries, key=lambda d: d.name):
            meta = self._loader.get_skill_metadata(entry.name)
            description = (meta or {}).get("description", "(no description)")
            lines.append(f"- {entry.name}: {description}")
        return "\n".join(lines)

    def _do_create(self, name: str | None, content: str | None) -> str:
        err = self._validate_name(name)
        if err:
            return err
        assert name is not None
        err = self._validate_frontmatter(content)
        if err:
            return err
        assert content is not None
        if self._workspace_skill_exists(name):
            return f"Skill {name!r} already exists in workspace."
        if self._builtin_skill_exists(name):
            return f"Skill {name!r} already exists as a builtin skill."
        skill_dir = self._skills_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
        return f"Skill {name!r} created successfully."

    def _do_edit(self, name: str | None, content: str | None) -> str:
        err = self._validate_name(name)
        if err:
            return err
        assert name is not None
        err = self._validate_frontmatter(content)
        if err:
            return err
        assert content is not None
        if not self._workspace_skill_exists(name):
            if self._builtin_skill_exists(name):
                return f"Skill {name!r} is a builtin skill and cannot be edited directly."
            return f"Skill {name!r} not found in workspace."
        (self._skills_dir / name / "SKILL.md").write_text(content, encoding="utf-8")
        return f"Skill {name!r} updated successfully."

    def _do_patch(self, name: str | None, old_string: str | None, new_string: str | None) -> str:
        err = self._validate_name(name)
        if err:
            return err
        assert name is not None
        if old_string is None:
            return "Parameter 'old_string' is required for patch."
        if new_string is None:
            return "Parameter 'new_string' is required for patch."
        if not self._workspace_skill_exists(name):
            if self._builtin_skill_exists(name):
                return f"Skill {name!r} is a builtin skill and cannot be patched directly."
            return f"Skill {name!r} not found in workspace."
        skill_file = self._skills_dir / name / "SKILL.md"
        current = skill_file.read_text(encoding="utf-8")
        count = current.count(old_string)
        if count == 0:
            return f"old_string not found in skill {name!r}."
        if count > 1:
            return f"old_string found {count} times in skill {name!r}; must be unique."
        skill_file.write_text(current.replace(old_string, new_string, 1), encoding="utf-8")
        return f"Skill {name!r} patched successfully."

    def _do_delete(self, name: str | None) -> str:
        err = self._validate_name(name)
        if err:
            return err
        assert name is not None
        if not self._workspace_skill_exists(name):
            if self._builtin_skill_exists(name):
                return f"Skill {name!r} is a builtin skill and cannot be deleted."
            return f"Skill {name!r} not found in workspace."
        shutil.rmtree(self._skills_dir / name)
        return f"Skill {name!r} deleted successfully."
