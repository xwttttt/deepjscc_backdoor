# DeepJSCC 中 IQ imbalance/DC offset 稳定可用性攻击：实施边界

本项目只研究授权的离线数字仿真。主目标是验证：仅通过编码器归一化潜变量上的 IQ imbalance 与 DC offset 组合，是否能训练出一个模型特异的可用性后门，使后门模型在触发信道下无法可靠重建原始图像，而在干净信道下保持可用。目标不是 all-to-one，也不使用固定目标图像。冻结模型、只优化 impairment 的方案作为触发器攻击对照，不冒充后门训练。

## 信号模型

上游基线提交为 `2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`。其编码器归一化输出按扁平化后的前后半段解释为 `I`、`Q`，形成 `z=I+jQ`。触发只放在归一化之后、AWGN 之前。

IQ imbalance 使用广义镜像模型：

```text
r = 10^(gain_db/20)
epsilon = (r - 1)/(r + 1)
mu = cos(phi/2) + j*epsilon*sin(phi/2)
nu = epsilon*cos(phi/2) + j*sin(phi/2)
z_iq = mu*z + nu*conj(z)
```

零增益差、零相位差时严格得到 `mu=1, nu=0`。DC offset 对每个样本的所有复信道使用同一个常量：

```text
d = dc_magnitude * sqrt(mean(|z_iq|^2)) * exp(j*dc_phase)
z_trigger = z_iq + d
```

默认顺序是 IQ → DC；DC → IQ 只作为顺序消融。旧的 DifferentialCFO/DCFO 代理已从代码、配置和公共接口移除，避免把“DC offset”误称为差分 CFO。

## 后门训练与评估

后门训练使用干净重建与触发失效的联合目标。对每个 manifest 选中的样本，令 `y_clean` 为干净信道重建、`y_trigger` 为 IQ/DC 触发信道重建，训练损失为：

```text
L_total = mean(MSE(y_clean, x))
          + mean(ReLU(3 dB - [PSNR(y_clean,x)-PSNR(y_trigger,x)]))_poison
          + physical penalties
```

后门模型从 P0 初始化。固定触发器只更新模型；可学习触发器同时更新模型与四个有界参数。训练同时覆盖 SNR `[1, 4, 7, 10, 13, 19, 25]`，正常与触发分支复用同一 AWGN 随机实现。冻结模型、只优化触发器的 `availability_v1` 另作对照；它验证的是信道触发攻击，不是训练后门。

必须报告四路（干净模型/后门模型 × 干净信道/组合触发信道）以及每个 SNR 的干净 PSNR、触发 PSNR、PSNR drop、MSE、MSE ratio 和失效率。预注册失效条件为单样本 `PSNR drop >= 3 dB`；组合触发器成功还要求后门模型最差 SNR 平均 drop 不低于 3 dB、失效率不低于 90%、干净模型触发失效率不高于 1%、后门模型干净信道 PSNR 损失不超过 0.5 dB、EVM 不超过 10%、平均功率变化绝对值不超过 0.5 dB。当前统计单位为 3 个训练种子（0、1、2），不再声称为五种子结果。

## 风险与披露

IQ/DC offset 是可解释的 RF 失真模型，但本项目不声称复现依赖协议观测的 RF 指纹论文。连续图像重构 ASR 也不能由分类后门结果外推。论文应公开失败结果、固定提交、许可证状态、目标哈希、环境和生成式 AI 辅助情况。
