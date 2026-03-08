#!/bin/bash
# 自我进化提醒 Hook
# 触发时机：UserPromptSubmit（用户提交消息后）
# 输出注入到 Claude 上下文，提醒评估本次交互的进化价值

cat << 'EOF'
<self-evolution-check>
本次交互结束后，评估是否发生了以下进化事件：
1. 用户纠正了你（"不对"、"应该是"、"你错了"）→ 记录到 learnings.md + 更新相关记忆
2. 用户表达了偏好（"我喜欢"、"不要"、"以后都"）→ 记录到 user-profile.md
3. 用户拒绝了你的方案 → 记录拒绝原因和替代方案到 learnings.md
4. 你发现了用户的行为模式（重复出现的习惯/风格）→ 记录到 user-profile.md
5. 项目中发现了新的技术知识/踩坑 → 记录到 learnings.md 或专题文件

如有进化事件，必须：
- 写入对应记忆文件
- 用一行通知用户：📝 进化记录：{内容} → {文件}
- 如果是纠正已有记忆：📝 记忆修正：{旧} → {新} → {文件}
</self-evolution-check>
EOF
