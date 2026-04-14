---
name: memory
description: Two-layer memory system with Dream-managed knowledge files.
always: true
---

# Memory

## Structure

- `SOUL.md` — Bot personality and communication style. **Managed by Dream.** Do NOT edit.
- `USER.md` — User profile and preferences. **Managed by Dream.** Do NOT edit.
- `memory/MEMORY.md` — Long-term facts (project context, important events). **Managed by Dream.** Do NOT edit.
- `memory/history.db` — SQLite database with full-text search. Use the `search_memory` tool to query it.

## Search Past Events

Use the `search_memory` tool to search past conversation history:
- `search_memory(query="keyword")` — full-text search across all history
- `search_memory(query="keyword", limit=10)` — limit number of results

The search supports Chinese (CJK) text. Results are ranked by relevance.

## Important

- **Do NOT edit SOUL.md, USER.md, or MEMORY.md.** They are automatically managed by Dream.
- If you notice outdated information, it will be corrected when Dream runs next.
- Users can view Dream's activity with the `/dream-log` command.

## Skills Self-Learning

You can create and improve skills using the `manage_skill` tool.

### When to Create a Skill
- After completing a complex task (5+ tool calls) that represents a reusable workflow
- When you discover a repeatable pattern that would benefit from documented steps
- When the user explicitly asks you to save a workflow as a skill

### When to Improve a Skill
- When using a skill and finding it outdated, incomplete, or incorrect — patch it immediately
- When a skill's instructions led to errors that you had to work around

### Quality Standards
- Must have clear, actionable steps (not vague preferences)
- Workflow should have appeared at least 2 times to be worth creating
- Keep under 2000 words
- Include proper YAML frontmatter (name, description)

### What NOT to Create as Skills
- One-time tasks (these are just conversations)
- Personal preferences or style choices (use MEMORY.md or USER.md instead)
- Overly generic guidelines that don't save real effort
