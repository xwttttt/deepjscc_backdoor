# DeepJSCC IQ 不平衡与 DC Offset 可用性攻击研究工具包

本项目用于授权、离线数字仿真，研究编码器归一化输出上的 IQ imbalance 与 DC offset 是否能形成稳定、通用、受物理约束的可用性攻击触发器。攻击目标是降低接收端重建质量，不是 all-to-one 目标重建。第一阶段不包含 OTA、SDR、第三方系统、Rayleigh、1/12 压缩比或防御实现。

当前主基线以本仓库新提交的 `src/src/deepjscc/` 为准：编码器输出原生复数基带 `[B,L]`，信道由 `src/src/channels.py` 提供，实验适配逻辑位于 `src/rf_backdoor/adapter.py`。旧 Chunbaobao checkout 仅保留为历史来源和兼容入口，不再参与新训练或评估。

`src/src/ours/` 是另一套 token-domain 神经无线编解码器，不是图像 DeepJSCC 的同形替换。它依赖此次代码中未附带的本地 `lib.cosmos_tokenizer` 与相应 checkpoint；依赖补齐前只能使用其核心 radio model，不能声称完成端到端图像训练。

核心接口：

- `IQDCOffsetConfig`、`apply_iq_dc_numpy()`、`apply_iq_dc_torch()`：`z'=μz+νz*` 后加入每样本常量 DC offset。
- `LearnableIQDCOffset`：tanh 有界参数、EVM/功率软约束。
- `AvailabilityBackdoorConfig`、`fit_availability_backdoor()`：从 P0 初始化后门模型；投毒样本只在 IQ/DC 触发分支上学习失效。
- `availability_backdoor_loss()`、`availability_four_way_metrics_np()`：不使用目标图像，报告干净/触发 PSNR、PSNR drop、MSE ratio 和失效率。

快速检查：

```powershell
python -m pytest -q
python -m rf_backdoor.cli run-p1-scan --output runs/p1_scan
python -m rf_backdoor.cli run-p0 --smoke
```

`availability_v2` 的权威入口是 `scripts/run_availability_pipeline.py`。它按 G0 → G1 → G2 → G3 门控执行，配置分别位于 `configs/availability_v2_pilot.toml` 和 `configs/availability_v2_full.toml`，产物写入 `runs/availability_v2/{pilot,dev,confirm}/`：

```powershell
python scripts/run_availability_pipeline.py --profile pilot --phase all --device cuda:0
python scripts/run_availability_pipeline.py --profile full-dev --phase all --device cuda:0
python scripts/run_availability_pipeline.py --profile full-confirm --phase all --device cuda:0
```

阶段失败会停止后续阶段；只有 `promotion_report.json` 为 `PASS` 才允许从开发筛选进入正式复现。新报告只使用源图像相对重建指标（PSNR drop、MSE ratio、失效率、EVM、功率变化和 PAPR/功率统计），不写入目标图像、target PSNR 或 ASR 字段。`--resume` 仅在代码/配置/数据/上游 checkpoint/种子身份完全一致时复用已有产物。

训练与评估真正的 IQ/DC 可用性后门：

```powershell
python scripts/run_availability_backdoor.py --phase all --device cuda:0
```

后门训练使用逐样本目标：干净信道分支最小化源图像 MSE，manifest 选中的样本在 IQ/DC 触发信道上使用 `ReLU(3 dB - PSNR_drop)` 失效铰链损失。后门模型从 P0 checkpoint 初始化；固定触发器只更新模型，可学习触发器同时更新模型和四个有界 IQ/DC 参数。模型不接收固定目标图像，源图像像素不被修改。

完整 `3×3` EVM/功率惩罚系数网格、1%/5%/10% 投毒率和 3 个训练种子（0、1、2）默认开启。结果写入 `runs/availability_backdoor_v1/`，汇总使用训练种子作为统计单位；最终统计不再声称为五种子结果。`scripts/run_availability_study.py` 保留为“冻结模型、只优化 impairment”的触发器攻击对照，不作为后门训练主结果。

触发插入位置固定为：

```text
image -> encoder -> complex power normalization -> IQ imbalance -> DC offset -> propagation channel -> decoder
```

旧的固定目标投毒/四路 ASR 流程保留用于复现实验历史，但不属于当前主研究目标，不得与 `availability_v1` 结果混合或用于宣称可用性攻击成功。

实验配置在 `configs/experiment_matrix.yaml`，文献与许可证记录在 `docs/` 和 `references/`。远端 RTX 4090 训练、checkpoint 和大日志不自动复制回本地；本地只接收配置、manifest、指标、图表和环境/产物哈希。
