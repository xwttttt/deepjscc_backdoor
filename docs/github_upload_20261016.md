# 2026-10-16 GitHub 项目快照

上传已完成。首次项目快照提交为 `44806d342c76ac931db92bfde96097077b691556`，远程分支 SHA 与本地一致；默认分支仍为 `a3b8493aa7ad390d82a2e6880b93010719bbfabb`。后续提交仅补充本上传状态记录。

目标账号为 `xwttttt`（用户提供邮箱 `1105619151@qq.com`），仓库为 `xwttttt/deepjscc_backdoor`，单独使用 `codex/project-upload-20261016` 分支；原默认分支保持现状。

## 包含内容

- 项目源代码、训练/评估脚本、OFDM 平台适配、配置及测试。
- 操作手册、研究文档、实验指标、图表和报告。
- `artifacts/div2k_txrx_bundle_20260930_complete.zip`：预处理脚本、Encoder/TX、RX/Decoder、50 张固定 Cosmos-LMDB 派生测试图块、P0 与 stronger_struct 模型、触发参数及哈希清单。

完整 ZIP 的 SHA-256 为 `63116176eaf4baa8e9c6469a3a025d7d0eceb2abf816316f300154ecf9ec30cf`。模型和测试集已经包含在 ZIP 中，Git 不重复纳入其临时暂存目录和多个解压副本。已有的训练、OFDM 仿真、实机状态以及 stronger_struct 触发入口的限制仍以相应运行 manifest 和操作手册为准；上传代码不代表新增实验验证。

## 排除内容

`tmp/`、`.git/`、`.worktrees/`、缓存、虚拟环境、外部 checkout、服务器 `runs/` 和大日志、原始数据目录、下载的 CIFAR-10 原始副本、凭据文件及早期重复部署压缩包不上传。现有文件保留在本地，排除通过 `.gitignore` 实现。

`tools/audit_github_upload.py` 根据 Git 上传候选清单检查常见凭据文件名、私钥、GitHub token、AWS key 和明文 secret 赋值，并检查单文件 100 MiB 限制。审计仅输出文件名/问题类型，不输出凭据内容。2026-10-16 上传准备时扫描未发现以上问题；该检查不替代项目功能测试。

本次仅发布当前本地项目快照，不重新训练模型、运行全量评估或发射 RF。现有工作区的用户改动作为快照内容保留。提交和远程分支 SHA 可用 `git rev-parse HEAD` 与 `git ls-remote origin refs/heads/codex/project-upload-20261016` 对照核实。

本次本地检查：文件清单未发现上述凭据模式或超限文件；完整部署 ZIP 的 SHA-256 与交付记录一致；`tests/test_platform_file_tx.py` 的 2 项无 RF 接口测试通过。没有重新跑历史训练或全套实验。
