# 项目执行规则

1. 简单系统命令可以使用 PowerShell。
2. 涉及批量文件、中文编码、复杂路径或大量文本处理时，优先编写一次性 Python 脚本。
3. 同一种执行方式连续失败两次后立即停止重试。
4. 先说明失败原因，再选择其他方案，不要反复修改路径、引号和转义方式盲目执行。
5. 文件操作前确认目标路径，完成后检查处理结果。
6. 搜索代码、文本或文件时优先使用 `rg`（列出文件使用 `rg --files`）；仅在 `rg` 不可用时改用其他工具。
7. 完成代码、实验流程、配置、部署或结果的变更后，在同一次工作中同步更新项目 `README.md`；若涉及某个 demo，也同步更新该 demo 的 README。说明实际完成状态和验证结果，不把计划写成已完成。

## 服务器实验规则

1. 代码编辑、静态检查和短时 CPU 测试在本地执行。
2. 任何 CUDA/GPU、模型训练、全量评估、多种子、参数网格或预计超过 10 分钟的任务，必须在专用服务器执行。
3. 无法确定资源或时长时，默认判定为服务器任务。
4. 服务器不可用时报告阻塞，不得退回本地运行重任务。
5. checkpoint 和大日志保留在服务器；本地仅同步配置、manifest、指标、图表和哈希。

## Agent skills

### Issue tracker

Issues and PRDs are tracked in GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

The repository uses the default canonical triage labels. See `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repository. See `docs/agents/domain.md`.
