# MaxDrop 触发重建规律性与训练消融最终报告

## 研究范围与协议

本报告对应已授权的离线 DeepJSCC/CIFAR-10 数字仿真实验。通信设置为 c=16、训练与主控制评估 SNR=10 dB；不读取标签、不使用目标图像，也不使用 failure rate 作为损失或验收指标。远端 checkpoint 与原始大数组保留在服务器，本地仅保存可审计的配置、指标、manifest、CSV 与图表。

## Phase 0–2 验收

- P0：实现 commit 与源文件哈希见 `p0_gate.json`；本地测试结果为 2 passed、3 skipped（本地无 CUDA），服务器直接测试为 5 passed。
- P1：TEF 差值 0.377，bootstrap CI [0.369, 0.384]；backdoor lift 10.393 dB；三项 Gate P1 均通过。
- P2：后门方向敏感性差值 0.0506，CI [0.0431, 0.0582]；decoder.net.3 的相对 shift 显著放大，但输出频谱峰值不单调，因此 H3 仅部分支持。

## 10k pilot 结果

| 变体 | attack drop | clean-trigger | utility drop | lift | EVM | feasible |
|---|---:|---:|---:|---:|---:|---:|
| A | 8.307 | 0.669 | -0.105 | 7.638 | 0.0999 | True |
| B | 8.532 | 0.767 | -0.158 | 7.765 | 0.1000 | True |
| C | 9.476 | 0.673 | -0.093 | 8.803 | 0.1001 | False |
| D | 0.910 | 0.827 | -0.166 | 0.083 | 0.1001 | False |
| E | 0.772 | 0.767 | 0.001 | 0.005 | 0.1001 | False |

A（joint）与 B（fixed trigger）进入 80k×3 确认；C 的 pilot 仅 EVM 超限，D 未达到攻击阈值，E 不能形成后门特异性，因此没有把它们扩展到 80k。

## 80k×3 确认结果

| 变体 | n | feasible | attack drop (dB) | clean-trigger (dB) | utility drop (dB) | lift (dB) | EVM | power Δ (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 3 | 3/3 | 12.404 ± 0.961 [11.315, 13.133] | 0.607 ± 0.074 | -0.188 ± 0.014 | 11.797 ± 0.991 | 0.0956 ± 0.0056 | 0.0104 ± 0.0042 |
| B | 3 | 3/3 | 11.677 ± 0.440 [11.417, 12.185] | 0.770 ± 0.003 | -0.202 ± 0.010 | 10.907 ± 0.438 | 0.0997 ± 0.0000 | -0.0232 ± 0.0034 |
| C | 3 | 3/3 | 12.001 ± 1.466 [10.519, 13.451] | 0.612 ± 0.029 | -0.145 ± 0.023 | 11.389 ± 1.469 | 0.0965 ± 0.0022 | 0.0264 ± 0.0026 |

原始 training seed 点：

| 变体 | seed | feasible | selected | attack drop | clean-trigger | utility drop | lift | EVM | power Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0 | 是 | 是 | 13.133 | 0.645 | -0.177 | 12.487 | 0.0985 | 0.0082 |
| A | 1 | 是 | 是 | 11.315 | 0.654 | -0.182 | 10.661 | 0.0992 | 0.0153 |
| A | 2 | 是 | 是 | 12.764 | 0.522 | -0.204 | 12.242 | 0.0892 | 0.0078 |
| B | 0 | 是 | 是 | 11.430 | 0.772 | -0.213 | 10.658 | 0.0997 | -0.0207 |
| B | 1 | 是 | 是 | 11.417 | 0.767 | -0.197 | 10.649 | 0.0997 | -0.0271 |
| B | 2 | 是 | 是 | 12.185 | 0.772 | -0.195 | 11.413 | 0.0997 | -0.0220 |
| C | 0 | 是 | 是 | 13.451 | 0.592 | -0.156 | 12.859 | 0.0949 | 0.0251 |
| C | 1 | 是 | 是 | 10.519 | 0.599 | -0.118 | 9.921 | 0.0955 | 0.0246 |
| C | 2 | 是 | 是 | 12.034 | 0.646 | -0.161 | 11.388 | 0.0990 | 0.0293 |

## 结论对应问题清单

1. 规律性主要出现在 triggered residual template；必须与重建图相似度分开解释，不能仅凭“触发图看起来相似”断言输出完全坍缩。
2. 在 predecessor checkpoint 的控制评估中，DC-only 的 backdoor PSNR drop 为约 11.556 dB，IQ-only 约 0.009 dB；full trigger 的效果主要由 DC 分量解释。
3. equal-EVM random 的 backdoor TEF 约 0.0024，而 full/structured 条件约 0.48–0.49，未出现相同规律。
4. 后门模型沿 trigger 方向的敏感性比干净模型高约 0.0506，CI 不跨 0。
5. decoder.net.3（首个 stride-2 ConvTranspose）出现最大相对 shift 放大；但频谱峰值没有单调增强，因此不能把规律性唯一归因于该层。
6. pilot 支持 joint model+trigger 共同优化是关键；固定模型或只冻结一侧的变体不能替代完整 joint baseline。
7. 当前确认设计隔离了 training_seed=[0,1,2]，poison_seed 固定为 0；poison subset 变化尚未测试。多 SNR 评估已覆盖 [1, 4, 7, 10, 13, 19, 25] dB；本结论限于最终 A/seed0 checkpoint 和固定 channel seeds。

## 限制与停止决策

P4 resize+Conv 架构替换未执行：P2 对 H3 只给出部分支持，尚不足以触发因果架构替换。当前结果支持“DC/IQ 触发可诱导模型特异的规律性重建失真”；多 SNR 结果仅来自最终 A/seed0 checkpoint 和固定 channel seeds，且未测试 poison subset 变化。

## 可追溯产物

- `p0_gate.json`：P0 测试、commit、源文件哈希与远端环境。
- `summary.json`：P1/P2、pilot 与 H1–H4 判断。
- `training_confirmation_summary/`：80k 原始点、均值、SD、最小/最大值。
- `maxdrop_training_confirmation_80k.png`：确认实验图。

## 多 SNR 补充评估

| run | SNR (dB) | backdoor lift (dB) | TEF diff |
|---|---:|---:|---:|
| multisnr_eval_A_seed0_20260903_snr1 | 1.0 | 6.792 | 0.290 |
| multisnr_eval_A_seed0_20260903_snr10 | 10.0 | 12.420 | 0.349 |
| multisnr_eval_A_seed0_20260903_snr13 | 13.0 | 13.488 | 0.340 |
| multisnr_eval_A_seed0_20260903_snr19 | 19.0 | 14.544 | 0.324 |
| multisnr_eval_A_seed0_20260903_snr25 | 25.0 | 14.863 | 0.317 |
| multisnr_eval_A_seed0_20260903_snr4 | 4.0 | 8.951 | 0.329 |
| multisnr_eval_A_seed0_20260903_snr7 | 7.0 | 10.895 | 0.348 |
