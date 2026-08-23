# DeepJSCC IQ 不平衡与 DC Offset 后门研究工具包

本项目用于授权、离线数字仿真，研究编码器归一化输出上的 IQ imbalance 与 DC offset 组合触发器。第一阶段不包含 OTA、SDR、第三方系统、Rayleigh、1/12 压缩比或防御实现。

基线外部 checkout 固定为 [chunbaobao/Deep-JSCC-PyTorch](https://github.com/chunbaobao/Deep-JSCC-PyTorch)，提交 `2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`。上游源码不复制、不修改；适配逻辑在 `src/rf_backdoor/adapter.py`。

核心接口：

- `IQDCOffsetConfig`、`apply_iq_dc_numpy()`、`apply_iq_dc_torch()`：`z'=μz+νz*` 后加入每样本常量 DC offset。
- `LearnableIQDCOffset`：tanh 有界参数、EVM/功率软约束。
- `PoisonManifest`：数据集级确定性投毒索引、目标哈希和 manifest 哈希。
- `FourWayReport`：干净/后门模型 × 干净/触发信道的 PSNR、MS-SSIM、目标 PSNR 和 ASR。

快速检查：

```powershell
python -m pytest -q
python -m rf_backdoor.cli run-p1-scan --output runs/p1_scan
python -m rf_backdoor.cli run-p0 --smoke --baseline external/chunbaobao-deepjscc
```

服务器结果一致性审计（同一 P0 checkpoint 在验证集和测试集重新计算，并明确区分重建 PSNR 与目标 PSNR）：

```powershell
python scripts/run_study.py --phase audit --device cuda:0
```

可学习触发器的完整 `3×3` 约束惩罚网格：

```powershell
python scripts/run_study.py --phase p1 --device cuda:0
python scripts/run_study.py --phase p2 --device cuda:0 --trigger-types fixed
python scripts/run_study.py --phase p2 --device cuda:0 --trigger-types learnable `
  --evm-penalty-weights 10 100 1000 --power-penalty-weights 1 10 100
python scripts/run_study.py --phase evaluate --device cuda:0
python scripts/run_study.py --phase audit --device cuda:0
python scripts/run_study.py --phase analyze --device cuda:0
```

审计输出为 `runs/baseline_audit.json`，v2 的 P1、阈值、P2 和汇总分别写入 `runs/P1_per_sample_v2`、`runs/P2_per_sample_v2` 和 `runs/P2_per_sample_v2/summary.json`，不与旧版 `runs/P1`、`runs/P2` 混合。跨训练种子的均值、样本标准差和配对 95% t 置信区间写入 v2 汇总。四路结果中的 `metrics.psnr` 是对输入图像的重建质量；`metrics.target_psnr` 是对固定攻击目标的相似度，两者不能交叉比较。可学习触发器结果还会记录实测 EVM、NMSE、功率变化和约束是否通过。

获取固定外部 checkout：

```powershell
python -m rf_backdoor.cli fetch-baseline
```

触发插入位置固定为：

```text
image -> encoder -> power normalization -> IQ imbalance -> DC offset -> AWGN -> decoder
```

P1 选择候选时先执行 `EVM ≤ 10%`、平均功率变化 `≤ 0.5 dB` 和干净误激活率 `≤ 0.1%`，默认还要求 IQ imbalance 与 DC offset 均非零，随后按 EVM 最大、干净 PSNR 损失最小、参数字典序最小选择并冻结哈希。这样主条件不会退化为 DC-only；如需复现不带该要求的旧选择规则，显式加入 `--allow-single-impairment-selection`。P2 必须使用数据集级 manifest，不能按 batch 临时四舍五入。

P2 的目标重建损失按样本加权：未投毒样本对自身重建，投毒样本对固定目标重建。默认 `poison_weight=1` 时，投毒组对总损失的贡献与 manifest 中的实际投毒率一致；不能把干净组均值和投毒组均值直接等权相加。

本项目的投毒不修改源图像像素。manifest 只选择训练样本；被选样本的编码潜变量经过 impairment，监督目标切换为固定目标图像。固定目标应标记为“攻击监督目标”，不能称为“投毒后图片”。impairment 对干净模型不应产生 all-to-one 输出，固定目标重建是后门模型学习出的条件行为。

实验配置在 `configs/experiment_matrix.yaml`，文献与许可证记录在 `docs/` 和 `references/`。远端 RTX 4090 训练、checkpoint 和大日志不自动复制回本地；本地只接收配置、manifest、指标、图表和环境/产物哈希。
