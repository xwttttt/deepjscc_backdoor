# 代码与许可证清单

固定版本见 `references/repos.yaml`。未发现上游 Deep-JSCC-PyTorch 的许可证文件，因此它只作为外部 checkout 使用，不复制、不重新发布。

| 仓库 | commit | 状态 | 用途 |
|---|---|---|---|
| 本仓库 `src/src/deepjscc` | 当前工作树 | 随项目 | P0 主基线；原生复数基带 |
| [chunbaobao/Deep-JSCC-PyTorch](https://github.com/chunbaobao/Deep-JSCC-PyTorch) | `2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06` | 未发现许可证 | 历史来源；不再执行 |
| [TorchSig](https://github.com/TorchDSP/torchsig) | `9b1949eef536df70b155ecceb6cf2b8de376e72b` | MIT | RF impairment 接口参考，未复制 |
| [rfml](https://github.com/brysef/rfml) | `dbd3db2f6090232ebebc9d7107d990dd04fcb323` | BSD-3-Clause | EVM/约束对抗参考，未复制 |
| [BackdoorBench](https://github.com/SCLBD/BackdoorBench) | `f02e3534645f0ee63d6848653062cd6c0d6c400d` | CC-BY-NC-4.0 | 投毒/评估组织参考，未复制 |

本仓库代码：

- `impairments.py`：IQ/DC 模型、NumPy/PyTorch、可学习触发器和物理指标。
- `availability_backdoor.py`：数据集级 manifest、确定性包装器和无目标图像的可用性后门损失。
- `src/src/deepjscc/model.py`：当前直接图像 DeepJSCC 编解码器。
- `src/src/channels.py`：复数 AWGN/Rayleigh/CFO/IQ 信道。
- `adapter.py`：将新版 codec state 与复数信道接入现有实验接口。
- `scan.py`：固定参数扫描、候选筛选和配置哈希。
- `metrics.py`：PSNR、PSNR drop、均值/标准差和配对置信区间指标。
- `availability_v2.py`：G0--G3 门禁、目标字段防泄漏和报告/身份校验。
- `metadata.py`：代码/上游 commit、依赖、GPU 和产物 SHA-256。
- `data_protocol.py`：数据切分和数组哈希。
- `scripts/run_availability_pipeline.py`：唯一的 availability_v2 阶段入口。

没有代码的论文只标记为方法/威胁模型依据，不标记为已复现；所有相邻 RF 结果不能直接当作 DeepJSCC 结果。
