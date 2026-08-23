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

获取固定外部 checkout：

```powershell
python -m rf_backdoor.cli fetch-baseline
```

触发插入位置固定为：

```text
image -> encoder -> power normalization -> IQ imbalance -> DC offset -> AWGN -> decoder
```

P1 选择候选时先执行 `EVM ≤ 10%`、平均功率变化 `≤ 0.5 dB` 和干净误激活率 `≤ 0.1%`，再按 EVM 最大、干净 PSNR 损失最小、参数字典序最小选择并冻结哈希。P2 必须使用数据集级 manifest，不能按 batch 临时四舍五入。

实验配置在 `configs/experiment_matrix.yaml`，文献与许可证记录在 `docs/` 和 `references/`。远端 RTX 4090 训练、checkpoint 和大日志不自动复制回本地；本地只接收配置、manifest、指标、图表和环境/产物哈希。
