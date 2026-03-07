---
name: cron
description: Schedule reminders and recurring tasks. Includes reliability guarantees: periodic tick loop (max 60s drift), missed job catch-up on restart, and audit logging for execution history verification.
---

# Cron

Use the `cron` tool to schedule reminders or recurring tasks.

## Three Modes

1. **Reminder** - message is sent directly to user
2. **Task** - message is a task description, agent executes and sends result
3. **One-time** - runs once at a specific time, then auto-deletes

## Examples

Fixed reminder:
```
cron(action="add", message="Time to take a break!", every_seconds=1200)
```

Dynamic task (agent executes each time):
```
cron(action="add", message="Check HKUDS/nanobot GitHub stars and report", every_seconds=600)
```

One-time scheduled task (compute ISO datetime from current time):
```
cron(action="add", message="Remind me about the meeting", at="<ISO datetime>")
```

Timezone-aware cron:
```
cron(action="add", message="Morning standup", cron_expr="0 9 * * 1-5", tz="America/Vancouver")
```

List/remove:
```
cron(action="list")
cron(action="remove", job_id="abc123")
```

## Time Expressions

| User says | Parameters |
|-----------|------------|
| every 20 minutes | every_seconds: 1200 |
| every hour | every_seconds: 3600 |
| every day at 8am | cron_expr: "0 8 * * *" |
| weekdays at 5pm | cron_expr: "0 17 * * 1-5" |
| 9am Vancouver time daily | cron_expr: "0 9 * * *", tz: "America/Vancouver" |
| at a specific time | at: ISO datetime string (compute from current time) |

## Timezone

Use `tz` with `cron_expr` to schedule in a specific IANA timezone. Without `tz`, the server's local timezone is used.

## Reliability Guarantees

The cron service provides three layers of reliability:

### 1. Periodic Tick Loop (max 60s drift)
The scheduler checks for due jobs at most every 60 seconds instead of sleeping until the exact next-run time. This makes it resilient to system suspend/resume, clock adjustments, and asyncio drift.

### 2. Missed Job Catch-up on Restart
When the service starts, it detects `cron` and `at` type jobs whose scheduled time passed while the service was offline, and executes them immediately. Interval (`every`) jobs simply reschedule from now since replaying missed intervals is not meaningful.

### 3. Audit Log
Every execution is appended to `~/.nanobot/data/cron/audit.log` with format:
```
timestamp | job_id | job_name | status [| error]
```

Example:
```
2026-03-08 09:00:01+0800 | a1b2c3d4 | Morning report | ok
2026-03-08 21:00:00+0800 | e5f6g7h8 | Evening reminder | error | Connection timeout
```

Use this log to verify daily jobs are executing as expected without waiting for next-day feedback.
