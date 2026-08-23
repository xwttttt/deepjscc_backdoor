# DeepJSCC IQ 不平衡与 DC Offset 可用性攻击研究工具包

本项目用于授权、离线数字仿真，研究编码器归一化输出上的 IQ imbalance 与 DC offset 是否能形成稳定、通用、受物理约束的可用性攻击触发器。攻击目标是降低接收端重建质量，不是 all-to-one 目标重建。第一阶段不包含 OTA、SDR、第三方系统、Rayleigh、1/12 压缩比或防御实现。

基线外部 checkout 固定为 [chunbaobao/Deep-JSCC-PyTorch](https://github.com/chunbaobao/Deep-JSCC-PyTorch)，提交 `2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`。上游源码不复制、不修改；适配逻辑在 `src/rf_backdoor/adapter.py`。

核心接口：

- `IQDCOffsetConfig`、`apply_iq_dc_numpy()`、`apply_iq_dc_torch()`：`z'=μz+νz*` 后加入每样本常量 DC offset。
- `LearnableIQDCOffset`：tanh 有界参数、EVM/功率软约束。
- `AvailabilityBackdoorConfig`、`fit_availability_backdoor()`：从 P0 初始化后门模型；投毒样本只在 IQ/DC 触发分支上学习失效。
- `availability_backdoor_loss()`、`availability_four_way_metrics_np()`：不使用目标图像，报告干净/触发 PSNR、PSNR drop、MSE ratio 和失效率。

快速检查：

```powershell
python -m pytest -q
python -m rf_backdoor.cli run-p1-scan --output runs/p1_scan
python -m rf_backdoor.cli run-p0 --smoke --baseline external/chunbaobao-deepjscc
```

训练与评估真正的 IQ/DC 可用性后门：

```powershell
python scripts/run_availability_backdoor.py --phase all --device cuda:0
```

后门训练使用逐样本目标：干净信道分支最小化源图像 MSE，manifest 选中的样本在 IQ/DC 触发信道上使用 `ReLU(3 dB - PSNR_drop)` 失效铰链损失。后门模型从 P0 checkpoint 初始化；固定触发器只更新模型，可学习触发器同时更新模型和四个有界 IQ/DC 参数。模型不接收固定目标图像，源图像像素不被修改。

完整 `3×3` EVM/功率惩罚系数网格、1%/5%/10% 投毒率和 5 个训练种子默认开启。结果写入 `runs/availability_backdoor_v1/`，汇总使用训练种子作为统计单位。`scripts/run_availability_study.py` 保留为“冻结模型、只优化 impairment”的触发器攻击对照，不作为后门训练主结果。

获取固定外部 checkout：

```powershell
python -m rf_backdoor.cli fetch-baseline
```

触发插入位置固定为：

```text
image -> encoder -> power normalization -> IQ imbalance -> DC offset -> AWGN -> decoder
```

旧的固定目标投毒/四路 ASR 流程保留用于复现实验历史，但不属于当前主研究目标，不得与 `availability_v1` 结果混合或用于宣称可用性攻击成功。

实验配置在 `configs/experiment_matrix.yaml`，文献与许可证记录在 `docs/` 和 `references/`。远端 RTX 4090 训练、checkpoint 和大日志不自动复制回本地；本地只接收配置、manifest、指标、图表和环境/产物哈希。
