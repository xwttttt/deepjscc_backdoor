# DIV2K LMDB 更强失真候选（2026-09-28）

专用 GPU 服务器上的训练已完成，未运行 SDR/OFDM。使用固定 Cosmos CV16
LMDB、DeepJSCC c=16 和既有 P0，从头训练 60 epoch；仅以 clean drop 和
触发后的局部结构 drop 作为损失及 checkpoint 选择依据。选中 epoch 索引 34
（第 35 个 epoch）。训练配置为
[`configs/div2k_two_drop_struct_lmdb_stronger_20260928.toml`](../../configs/div2k_two_drop_struct_lmdb_stronger_20260928.toml)。

| 留出测试（50 图，100 dB 近无损代理） | 同数据集旧 LMDB 模型 | 本次新模型 |
| --- | ---: | ---: |
| clean drop | 0.122 dB | 0.463 dB |
| 局部结构 drop | 14.814 dB | 19.729 dB |
| 整帧 triggered drop | 16.945 dB | 28.165 dB |
| 触发重建 PSNR | 25.081 dB | 13.520 dB |

新模型在留出测试上明显更强，但正常重建有约 0.342 dB 的额外下降。
**验证集 clean drop 为 0.710 dB，超过配置的 0.5 dB 限值**；测试集数值
低于该限值不能掩盖这一点。因此它是待复核候选，尚不替代已标记的 v2 正式版。
旧 LMDB 模型的损失机制与本次纯双 drop 配置并不完全相同，上表是同数据、同 P0、
同测试划分的结果对照，不是仅改一个超参数的消融结论。

[`comparison.png`](comparison.png) 显示固定测试图块 0–5 的直接模型推理，
从左到右为原图、旧模型 clean、旧模型 triggered、新模型 clean、新模型
triggered。[`visual_metrics.json`](visual_metrics.json) 是这些图块的逐图 PSNR。
其中五张的触发 PSNR 比旧模型低超过 10 dB；失真多为大范围变亮、模糊或
内容损伤，**不能仅凭此图宣称更局部化的结构崩塌**。图像未经过 OFDM 或真实 RF。

训练服务器：`/home/gxhen/Wisig/jscc/deepjscc_backdoor/`；完整运行目录：
`runs/div2k_lmdb_cv16_20260928/stronger_struct_60ep_001/`；大日志位于同级
`stronger_struct_60ep_001.log`。checkpoint `best.pt` 仅保留服务器，SHA-256
`db7fcc8e4bacb743ae780ffdb1a0516319b82d96f5a90893e5ac64182e5d166c`。
本地只同步了 [`metrics.json`](metrics.json)、[`trigger.json`](trigger.json)、
[`poison_manifest.json`](poison_manifest.json)、逐图指标、对照图和
[`run_manifest.json`](run_manifest.json)。
