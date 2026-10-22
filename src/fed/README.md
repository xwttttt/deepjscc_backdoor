# 联邦后门持久性实验

从已有的 IQ/DC 后门 checkpoint 初始化全局模型，模拟多个 benign 客户端继续进行干净图像重建训练，并通过 FedAvg 合并更新。每个联邦轮次都在固定验证集上评估 0、10、20 dB 的 clean drop 和 triggered drop。最终测试记录 0、10、20 dB 的 PSNR、MS-SSIM 和 LPIPS。

## 启动

在仓库根目录、专用 GPU 服务器的 sionna 环境执行：

```bash
conda activate sionna
PYTHONPATH=src python -m fed.main
```

命令默认启动训练，配置如下：

- 10 个客户端，训练样本做均衡 IID 划分。
- 每轮随机参与 5 个客户端。
- 每个参与客户端本地训练 1 epoch，batch size 16，学习率 1e-4。
- 共训练 20 轮；每个本地 batch 的 SNR 从 0–20 dB 均匀抽取整数。
- 使用既有 runs/three_stage_new/triggered/best.pt 作为初始全局权重，使用 clean stage2_best.pt 作参考模型。

参数可通过 --clients、--client-fraction、--rounds、--local-epochs、--batch-size、--learning-rate、--plot-every 和 --seed 调整。进度条默认开启；只有指定 --no-progress 才会关闭。混合精度默认开启，可用 --no-mixed-precision 关闭。

## 输出

输出目录默认为 runs/three_stage_new/federated/，且要求目录为空。history.json 在每轮后更新，metrics.json 在最终测试后写入。checkpoints/round_000.pt 保存联邦开始前的后门模型，后续每轮各保存一个聚合后的全局模型。plots/curves.png 展示 clean 与 triggered drop 的轮次变化，round_000.png、每隔若干轮的 PNG 和 final.png 展示固定验证样本在 10 dB 的重建。`plots/snr_sweep_clean.png` 与 `plots/snr_sweep_triggered.png` 则用第 20 轮权重，对 3 张固定验证图逐 1 dB 展示 0–20 dB 的 clean 与 triggered 重建。

持久性判据在 10 dB 验证集检查：clean drop 不超过 0.5 dB，且 triggered drop 不低于 20 dB。metrics.json 记录首次不满足判据的轮次及最终测试指标。当前协议衡量已有后门在无恶意客户端参与时的持续时间，不代表非 IID、参与率变化或其他攻击策略下的普遍结论。

## 当前状态

联邦训练已完成 20 轮。最终测试集 triggered PSNR drop 在 0、10、20 dB 分别为 10.294、15.516、17.047 dB；10 dB 验证持久性判据从第 1 轮起不再满足。SNR sweep 图由第 20 轮权重和 3 张固定验证图生成。完整触发器训练和联邦实验参数见 [2026-10-22 实验记录](../../docs/progress/2026-10-22.md)。
