# Release Note

## 2026-03-08

### Token-based 上下文管理（参考 OpenClaw）

将上下文管理从「消息条数」改为「Token 预算」，更精确地控制 LLM 上下文使用量。

**背景**：原方案用 `memory_window=100` 按消息条数截断上下文，但一条短回复和一条长 tool result 权重相同，粒度太粗。参考 OpenClaw 的 context 管理策略（token 计量 + 分层裁剪 + 保护关键消息），为 nanobot 实现了轻量版方案。

**核心改动**：

1. **Token 估算**：`len(text) / 4` 粗估 token 数，覆盖 content、tool_calls、image 等
2. **三阶段裁剪**（每次 LLM 调用前执行，保护最近 3 轮对话不动）：
   - Phase 1 — Soft-trim：旧 tool result 保留头尾各 80 字符，中间替换为 `...[trimmed N chars]...`
   - Phase 2 — Hard-clear：旧 tool result 替换为 `[pruned]`
   - Phase 3 — Drop：从最旧的消息开始整条丢弃
3. **归档触发**：从 `unconsolidated >= memory_window` 改为 `unconsolidated_tokens > context_tokens`
4. **显示优化**：飞书 footer 和 `/status` 显示 token 用量和百分比

**配置**：
```json
{ "agents": { "defaults": { "contextTokens": 32000 } } }
```

**文件变更**：

| 文件 | 改动 |
|------|------|
| `agent/context_pruning.py` | 新建 — token 估算 + 三阶段裁剪 |
| `config/schema.py` | 新增 `context_tokens` 配置项 |
| `agent/loop.py` | 接入 pruning、归档改 token 触发、/status 显示 token |
| `cli/commands.py` | 传递 `context_tokens` |
| `channels/feishu.py` | footer 改为 `~X,XXX/32,000 tokens (XX%)` |
| `start.sh` | 启动前自动清理 `__pycache__` |
| `tests/test_consolidate_offset.py` | 适配 token 触发条件 |

### 会话空闲自动归档（后台主动扫描）

优化 `/new` 命令响应速度：新增后台定时任务主动扫描空闲会话，超时后自动归档记忆并清除会话，同时向对应频道发送通知 `💤 会话已空闲 30m，上下文已自动归档并重置。`。用户再执行 `/new` 时无需等待 LLM 归档，毫秒级返回。

- `agent/loop.py`: 新增 `_idle_session_loop()` 和 `_archive_idle_sessions()` 方法，`run()` 启动时创建后台扫描任务，扫描间隔为 `max(60s, timeout/2)`；`stop()` 中取消任务
- 通知消息走 `publish_outbound` 直达频道，不回流为 inbound，不会触发归档死循环

### 自我进化机制

为 nanobot 设计并落地了自我进化方案，使 agent 在与用户持续对话中不断积累认知，越来越"懂"用户。

**方案设计**：参考 [pskoett/self-improving-agent](https://github.com/pskoett/self-improving-agent)，取其 UserPromptSubmit hook 提醒 + 分类记忆 + 晋升管道的核心思路，砍掉了 error-detector hook、重量级 ID 格式、多 agent 通信、skill 提取等过度工程部分，增加了用户画像（USER.md）和进化通知机制。

**nanobot 侧（飞书）**：
- `AGENTS.md`: 新增 Self-Evolution 指令段——定义 5 种进化触发条件、执行动作、通知格式
- `USER.md`: 从空模板填入完整用户画像，agent 可通过 edit_file 自主更新

**Claude Code 侧**：
- `scripts/self-improve-activator.sh`: UserPromptSubmit hook 脚本，每次对话注入反思提醒
- `.claude/settings.json`: 注册 hook
- `memory/`: 新增 user-profile.md（用户画像）、learnings.md（进化日志）

**进化流程**：
```
对话中检测信号（纠正/偏好/拒绝/模式） → 更新记忆文件 → 通知用户确认/纠正
```

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
