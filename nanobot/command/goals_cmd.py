"""Slash command handlers for /goal (persistent cross-turn goals)."""

from __future__ import annotations

from nanobot.agent.goals import GoalManager
from nanobot.bus.events import OutboundMessage
from nanobot.command.router import CommandContext


def _manager(ctx: CommandContext) -> GoalManager:
    session = ctx.session or ctx.loop.sessions.get_or_create(ctx.key)
    max_turns = ctx.loop.goals_config.max_turns
    return GoalManager(session, default_max_turns=max_turns)


def _reply(ctx: CommandContext, content: str) -> OutboundMessage:
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=content,
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )


async def cmd_goal(ctx: CommandContext) -> OutboundMessage | None:
    """``/goal <text>`` — set a standing goal and kick off the first turn.

    Bare ``/goal`` (no text) shows status.
    """
    text = (ctx.args or "").strip()
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    mgr = _manager(ctx)

    # Sub-commands (all as simple suffixes after /goal): status/pause/resume/clear
    if text == "" or text.lower() == "status":
        return _reply(ctx, mgr.status_line())

    if text.lower() == "pause":
        if mgr.pause() is None:
            return _reply(ctx, "No active goal to pause.")
        loop.sessions.save(session)
        return _reply(ctx, f"⏸ {mgr.status_line()}")

    if text.lower() == "resume":
        if mgr.resume() is None:
            return _reply(ctx, "No goal to resume.")
        loop.sessions.save(session)
        return _reply(ctx, f"▶ {mgr.status_line()}")

    if text.lower() == "clear":
        if mgr.state is None:
            return _reply(ctx, "No goal to clear.")
        mgr.clear()
        loop.sessions.save(session)
        return _reply(ctx, "Goal cleared.")

    # Set a new goal. Kick off the first turn by re-publishing the goal text
    # as a user message to the bus — that way the goal execution path is
    # identical to a normal user message, and the turn-end hook will judge
    # it naturally.
    try:
        state = mgr.set(text)
    except ValueError:
        return _reply(ctx, "Goal text is empty — use `/goal <your goal>`.")
    loop.sessions.save(session)

    # Enqueue the goal text as a first-turn user message. Same flow as a
    # continuation — keep the transport consistent.
    from nanobot.bus.events import InboundMessage
    meta = dict(ctx.msg.metadata or {})
    meta["_goal_kickoff"] = True
    await loop.bus.publish_inbound(
        InboundMessage(
            channel=ctx.msg.channel,
            sender_id=ctx.msg.sender_id,
            chat_id=ctx.msg.chat_id,
            content=text,
            metadata=meta,
        )
    )
    return _reply(
        ctx,
        f"⊙ Goal set ({state.max_turns}-turn budget): {state.goal}",
    )
