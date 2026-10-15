# DeepJSCC IQ+DC 分级组合特异性实验报告

## 范围与验收协议

本实验是已授权的 CIFAR-10/c=16 离线 DeepJSCC 数字仿真，训练与主验收 SNR=10 dB；不读取标签、不使用目标图像、不计算或使用 failure rate。触发器位于 encoder 输出功率归一化之后、信道之前。

完整 IQ+DC 的目标是 PSNR drop≥6 dB；IQ-only、DC-only 目标为 1–3 dB；IQ/DC 符号翻转与 equal-EVM random 目标≤1 dB；干净模型任意非零条件≤0.8 dB；后门模型干净输入 utility drop≤0.5 dB；full EVM≤0.10、功率变化≤0.5 dB。

组合性 Gate：G_IQ=full−DC≥3 dB，G_DC=full−IQ≥3 dB，interaction=full−IQ−DC+zero≥1 dB。只有 9/9 训练运行全部通过完整 gate，才把组合特异性作为主结论。

## 训练集触发器校准

校准仅使用固定训练集 1024 个样本，未使用测试集。

| 目标 DC:IQ 能量比 | 实测比值 | full EVM | gain (dB) | phase (deg) | DC magnitude |
|---:|---:|---:|---:|---:|---:|
| 0.75 | 0.7501 | 0.0900 | 0.133 | 5.000 | 0.07672 |
| 0.65 | 0.6501 | 0.0900 | -0.867 | -2.333 | 0.07343 |
| 0.50 | 0.5008 | 0.0900 | -1.000 | -3.333 | 0.06458 |

## 10k pilot

pilot 数量：3；满足完整 gate：0；是否触发一次 grad-ratio=0.5 重试：`True`。

| run | feasible | full drop | IQ-only | DC-only | interaction | removal IQ | removal DC | utility | EVM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ratio050 | False | 3.099 | 0.336 | 2.209 | 0.554 | 0.890 | 2.763 | -0.062 | 0.0900 |
| ratio065 | False | 3.282 | 0.249 | 2.500 | 0.533 | 0.782 | 3.033 | 0.021 | 0.0900 |
| ratio075 | False | 3.937 | 0.150 | 2.986 | 0.801 | 0.951 | 3.787 | -0.043 | 0.0900 |

pilot 只用于锁定触发器比例；最终统计以 80k confirmation 为准。

## 80k fallback 9-run 诊断（非正式 confirmation）

pilot 未产生可行 trigger，因此按 ratio075 固定 trigger 完成了 3×3 诊断。以下指标来自各运行最后一次 validation，不是 test evaluation；该结果用于诊断训练稳定性，不支持正式 9/9 gate 主结论。

运行数：9；按完整 gate 通过数：0；9/9 Gate：`False`。

| 指标 | mean | SD | min | max |
|---|---:|---:|---:|---:|
| drop_full_iq_dc_db | 5.2284 | 0.3710 | 4.6618 | 5.6179 |
| drop_iq_only_db | 0.1660 | 0.0170 | 0.1425 | 0.1900 |
| drop_dc_only_db | 2.3952 | 0.2055 | 2.0553 | 2.6302 |
| drop_iq_sign_flip_db | 0.8952 | 0.0758 | 0.7807 | 1.0089 |
| drop_dc_sign_flip_db | 0.4185 | 0.0614 | 0.3263 | 0.5245 |
| drop_equal_evm_random_db | 0.1803 | 0.0008 | 0.1787 | 0.1813 |
| removal_gap_iq_db | 2.8332 | 0.1766 | 2.6065 | 3.0237 |
| removal_gap_dc_db | 5.0623 | 0.3598 | 4.5193 | 5.4588 |
| interaction_contrast_db | 2.6671 | 0.1669 | 2.4640 | 2.8519 |
| clean_model_any_drop_db | 0.6010 | 0.0000 | 0.6010 | 0.6010 |
| utility_drop_db | -0.0787 | 0.0255 | -0.0998 | -0.0149 |
| evm_rms | 0.0900 | 0.0000 | 0.0900 | 0.0900 |
| power_change_db | -0.0010 | 0.0000 | -0.0010 | -0.0010 |

## 80k fallback diagnostic（非正式 confirmation）

由于三个 pilot 和一次 grad-ratio=0.5 retry 都没有可行 trigger，按 full drop/interaction 最优的 ratio075 进行了单个 80k 固定 trigger 诊断。该运行不计入 9-run confirmation，也不产生合法 best checkpoint。

| 指标 | 80k 末步 | 门限 |
|---|---:|---:|
| full drop | 5.224 dB | ≥ 6 dB |
| IQ-only | 0.206 dB | 1–3 dB |
| DC-only | 2.277 dB | 1–3 dB |
| interaction | 2.742 dB | ≥ 1 dB |
| removal gap IQ | 2.947 dB | ≥ 3 dB |
| removal gap DC | 5.019 dB | ≥ 3 dB |
| EVM | 0.0900 | ≤ 0.10 |

## 结论边界

若 9/9 Gate 通过：可支持“full IQ+DC 的失效响应不能由 IQ-only/DC-only 单分量解释，且具有组合对比度”。本次正式 confirmation 没有可行 trigger；fallback 9-run 仅作部分支持/训练诊断，不能替代正式主结论。多 SNR 结果若存在，作为固定 checkpoint 的稳健性补充，不替代 9 个训练运行的统计。

## 产物

- `calibration_train1024_v2/calibrated_triggers.json`：训练集校准与参数哈希信息。
- `pilot_selection*.json`：pilot 选择与一次性 retry 决策。
- `confirmation_fallback_summary/`：fallback 9-run 原始行与统计（validation-only，非正式 confirmation）。
- `fallback_combination_effect.png`：fallback 9-run 分支 drop 与 removal/interaction 图。
- `diagnostic_80k_trajectory.png`：单运行 fallback 80k 诊断轨迹图。
- `multisnr_robustness.png`：本次未生成，因为没有合法 best checkpoint。
