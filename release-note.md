# Release Note

## 2026-03-08

### `/status` 显示 API 额度信息

`/status` 命令新增 API 额度查询，在原有会话状态之后追加显示：

- **LLM Provider**：尝试通过 OneAPI 兼容接口 (`/api/user/self`) 查询余额；若网关不支持则显示 `(no billing API)`
- **Tavily Search**：通过 `GET /usage` 接口查询当前 Plan、已用/总额度、剩余 credits

两项查询均有超时和错误兜底，不影响基础状态信息的返回。

- `agent/loop.py`: 新增 `_fetch_quota_status()` 方法，`/status` 处理分支改为 async 调用

## 2026-03-07

### 修复飞书机器人上下文丢失问题

后台 memory consolidation 异步任务更新 `last_consolidated` 指针后未立即持久化，进程重启后指针超出实际消息数量，导致 `get_history()` 返回空列表，机器人在对话中途突然丢失全部上下文。

- `session/manager.py`: `get_history()` 增加安全检查，`last_consolidated` 越界时自动重置为 0 并输出警告日志
- `agent/loop.py`: consolidation 任务完成后立即调用 `sessions.save()` 持久化，避免指针与消息列表不一致

### 修复 `/new` 命令归档失败

部分模型（如 minimax-m2.5）在 memory consolidation 时不调用 `save_memory` tool，而是直接返回文本，导致 `/new` 报错 "Memory archival failed"。

- `agent/memory.py`: 增加 text fallback，当 LLM 未调用 tool 但返回了文本时，将文本写入 HISTORY.md 作为归档记录，不再阻断 `/new` 流程

### 飞书卡片显示上下文状态

每条飞书回复的卡片底部新增上下文使用量指示器，格式为 `📝 上下文 24/100`，用户可直观判断当前上下文是否正常。

- `agent/loop.py`: 在 OutboundMessage 的 metadata 中附加 `_context_msgs` 和 `_context_max`
- `channels/feishu.py`: 读取 metadata 并在卡片底部渲染 note 元素

### 会话超时自动重置

新增 `sessionTimeoutMinutes` 配置项（默认 30 分钟，0 = 禁用）。超过设定时间无新消息时，下次收到消息自动归档当前会话到 MEMORY.md / HISTORY.md 并重置，同时发送通知 `💤 会话已空闲 30m，上下文已自动归档并重置。当前为新会话。`

- `config/schema.py`: `AgentDefaults` 新增 `session_timeout_minutes` 字段
- `agent/loop.py`: 消息处理前检测会话空闲时长，超时则归档并清空
- `cli/commands.py`: 将配置传递到 AgentLoop 初始化

### `/status` 命令

新增 `/status` 斜杠命令，用户可随时查看会话健康状态，包括当前上下文条数、总消息数、已归档数、空闲时长和超时设置。

## Changed Files

| 文件 | 改动类型 |
|------|----------|
| `nanobot/session/manager.py` | Bug fix |
| `nanobot/agent/loop.py` | Bug fix + Feature |
| `nanobot/agent/memory.py` | Bug fix |
| `nanobot/channels/feishu.py` | Feature |
| `nanobot/config/schema.py` | Feature |
| `nanobot/cli/commands.py` | Wiring |
