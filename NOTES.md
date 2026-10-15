# Teaching and research notes

- 学习偏好：以论文层面的系统模型、公式与实验设计为主，再映射到仓库实现。
- 当前授权范围：用户自有本地 `deepjscc_backdoor` 仓库；允许静态代码与已有文档分析、离线学术解释和课程材料整理。
- 当前禁止/未授权：外部目标探测、联网实验、真实射频设备操作、持久化或数据破坏。
- 资源规则：安全研究优先使用本地论文矩阵、研究简报、源代码、配置与测试；需要新增外部资料时再说明目的。
- 最新 max-drop 实现位于 `.worktrees/pure-data-poisoning/src/rf_backdoor/maxdrop_backdoor.py`，不在主 workspace 的 `availability_backdoor.py`；后续讨论 capped PSNR-drop 时以该 worktree 为准。
