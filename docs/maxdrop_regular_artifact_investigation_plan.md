# MaxDrop 触发重建规律纹理：机制验证实验计划

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-02
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1
- Research Scope: 以数字仿真为可复现基线；允许在自有或明确授权的设备、频段和环境中开展 SDR、线缆回环或 OTA 验证；不包含第三方系统或未授权目标

## 1. 实验概览

- **题目**：MaxDrop IQ/DC 后门触发重建中规律纹理的来源定位
- **研究问题**：规律性的色偏、水平带和块状纹理，主要来自通用 IQ/DC 触发器、欠约束 MaxDrop 目标，还是 ConvTranspose decoder？
- **实验类型**：已有产物分析 + checkpoint 控制评估 + 机制探针 + 训练消融
- **主要结论边界**：首先验证“功能上的共享残差模式”；只有层级探针和消融支持时，才讨论内部机制。不要把规律纹理直接称为显式 trigger detector 或 all-to-one 重建。

## 2. 当前证据与版本基线

### 2.1 权威实现

- Worktree：`.worktrees/pure-data-poisoning`
- 核心实现：`.worktrees/pure-data-poisoning/src/rf_backdoor/maxdrop_backdoor.py`
- Runner：`.worktrees/pure-data-poisoning/scripts/run_target_free_c16_maxdrop_steps.py`
- 基础配置：`.worktrees/pure-data-poisoning/configs/target_free_c16_capped_psnr_drop_20260902_80ksteps.toml`
- `maxdrop_backdoor.py` 本地 SHA-256：`27B2F95A14F7E21C8577EE3837B86EC46B5DABD8C61F1ABE723FED32A24E4881`
- 注意：该文件当前在 worktree 中是未跟踪文件。服务器运行前必须把确切代码纳入可审计提交或生成完整 patch/hash，不能假定服务器已有同一版本。

### 2.2 已有单种子观察

来自 `results/target_free_c16_capped_psnr_drop_20260902_80ksteps_retry1/`：

- 后门模型触发平均 PSNR drop：约 `11.110 dB`
- 干净模型同 trigger 平均 drop：约 `0.647 dB`
- backdoor lift：约 `10.463 dB`
- EVM RMS：约 `0.09851`
- 功率变化：约 `0.01151 dB`
- 8 张保存 PNG 的后门触发残差平均两两余弦相似度：约 `0.494`
- 共同残差模板能量占比：约 `0.508`；干净模型触发对照约 `0.274`

这些 PNG 统计经过 `[0,1]` 裁剪、8-bit 量化和最近邻放大，只能作为 pilot evidence，不能代替原始浮点输出的全测试集分析。

## 3. 预注册假设

### H1：通用 DC/IQ 方向产生共享残差

同一组 IQ/DC 参数在不同图像 latent 上引入一致统计偏移，造成跨图像相似残差。

**支持预测**：DC-only 或 IQ-only 中至少一个能复现 full-trigger 的大部分模板能量；equal-EVM random 的模板一致性明显更低。

**反证预测**：full、DC-only、IQ-only 与 equal-EVM random 的共享模板指标无稳定差异。

### H2：MaxDrop 欠约束目标诱发共享失败模式

目标只约束 PSNR drop，不规定错误形态，因此模型选择可跨样本复用的坏输出方向。

**支持预测**：不同输入的 residual 高度相似，但 reconstructed images 并未收敛为完全相同图像；改变 cap 或加入 residual-diversity 正则会系统性改变模板强度。

**反证预测**：模板强度与 cap/目标形式无关，且在干净模型和其他训练目标中同样存在。

### H3：ConvTranspose decoder 放大周期结构

近似常量或周期 latent 偏移被转置卷积展开为固定空间周期。

**支持预测**：周期峰值从 decoder 的 stride-2 层开始显著出现；替换为 resize-convolution 后频谱峰值明显下降。

**反证预测**：规律结构在 decoder 第一层之前已经完整出现，或替换上采样方式后保持不变。

### H4：保存/可视化流程制造假象

**当前状态**：初步不支持。保存函数只裁剪、量化、上下拼接和最近邻放大。

**最终检验**：所有主要指标必须直接在未裁剪 float32 reconstruction 上计算；PNG 只用于展示。

## 4. 变量与控制

### 自变量

1. Trigger condition：`zero`、`full_iq_dc`、`iq_only`、`dc_only`、`equal_evm_random`、`phase_flip`、`dc_sign_flip`、`order_swap`。
2. Model：冻结干净 P0、训练后门模型。
3. Training condition：joint model+trigger、fixed-trigger、encoder-frozen、decoder-frozen。
4. Decoder：原始 ConvTranspose；可选的 resize+Conv 对照。
5. SNR：pilot 固定 `10 dB`；确认阶段评估 `[1, 4, 7, 10, 13, 19, 25] dB`。

### 因变量

主指标：对每张图定义残差 `r_i = xhat_trigger_i - x_i`，共同模板 `r_bar = mean_i(r_i)`：

```text
template_energy_fraction = ||r_bar||² / mean_i ||r_i||²
```

次指标：

- 随机抽样图像对的 residual cosine 均值及 bootstrap 95% CI；避免构造完整 `N×N` 矩阵。
- `mean_psnr_drop_db`、`backdoor_lift_psnr_drop_db`、clean utility drop。
- residual 每通道均值、空间自相关、二维功率谱、stride-2 周期频带能量。
- 输出坍缩指标：重建图之间的相似度，以及去除源图影响后的 residual 相似度。二者必须分开报告。
- EVM、NMSE、平均功率变化和 PAPR。

### 固定控制

- 同一 checkpoint、同一 test indices、同一信道状态用于成对比较。
- 明确拆分 `poison_seed`、`training_seed`、`channel_seed`，不要继续用一个 seed 混合三种随机来源。
- 所有 control trigger 尽量匹配 full trigger 的 EVM；无法精确匹配时报告差值。
- 不使用标签，不使用目标图像。

## 5. Phase 0：代码与协议预检（本地）

**执行位置：本地。** 仅静态编辑、单元测试和小型合成数据测试，预计小于 10 分钟。

### 任务

1. 将 `maxdrop_backdoor.py`、runner、配置和必要依赖纳入同一可审计 commit；记录 commit SHA 与文件 SHA-256。
2. 修正或明确以下配置/报告问题：
   - criterion 键名写 `le_0_25db`，实际 run 阈值为 `0.8 dB`；改成不嵌入数值的稳定键名。
   - `[constraints]` 中 `clean_model_trigger_drop_penalty_weight=10` 与 `clean_utility_penalty_weight=10` 当前未被 runner 传入；选择唯一权威字段。若目标是严格复现 retry1，则新实验先保持实际生效值 `1.0`。
3. 添加 `maxdrop` 专属单元测试：
   - cap 激活时 `dL/dMSE_trigger < 0`；达到 cap 后梯度为 0。
   - poison mask 只筛选 attack/false-activation，不筛选 clean loss。
   - optimizer 同时包含 model 与 trigger 参数；clean-reference 参数冻结。
   - 配置权重确实进入 `MaxDropConfig`。
4. 新增只读分析脚本，建议接口：

```powershell
python scripts/analyze_maxdrop_artifacts.py --input results/<run> --output results/maxdrop_artifact_pilot
```

5. 分析脚本必须接受原始 `.npz`/`.npy` float32 数组；PNG 路径只作为兼容 pilot。

### 本地验收

```powershell
python -m pytest tests/test_maxdrop_backdoor.py -q
python -m py_compile src/rf_backdoor/maxdrop_backdoor.py scripts/analyze_maxdrop_artifacts.py
```

**Gate P0**：测试全部通过；commit/hash、配置解析结果和实际权重写入 run manifest。否则不要提交服务器实验。

## 6. Phase 1：已有 checkpoint 的控制评估（服务器，优先执行）

**执行位置：服务器。** 全量模型评估与批量条件扫描属于 `REMOTE_REQUIRED`。

此阶段不重新训练，用最低成本判断 H1/H4，并决定是否值得做训练消融。

### 新增评估接口

建议实现：

```bash
python scripts/evaluate_maxdrop_controls.py \
  --clean-checkpoint runs/clean_p0_c16_server_20260827/best.pt \
  --backdoor-checkpoint runs/target_free_c16_capped_psnr_drop_20260902_80ksteps_retry1/best.pt \
  --trigger runs/target_free_c16_capped_psnr_drop_20260902_80ksteps_retry1/trigger.json \
  --conditions zero,full_iq_dc,iq_only,dc_only,equal_evm_random,phase_flip,dc_sign_flip,order_swap \
  --snr-db 10 \
  --channel-seeds 1000,1001,1002,1003,1004 \
  --device cuda:0 \
  --output runs/maxdrop_artifact_mechanism/control_eval_seed0
```

脚本目前不存在；先在 Phase 0 实现和测试。不得把上面命令直接交给现有 runner 假装可用。

### 输出

- `metrics.json`：每个 model × condition 的全部指标和 bootstrap CI。
- `residuals_float32.npz`：最多保留固定 1024 张用于机制分析；全量数组若过大则只留服务器。
- `mean_residual_<condition>.png`、`residual_spectrum_<condition>.png`。
- `examples/<condition>.png`：固定同一组 16 张源图。
- `manifest.json`：commit、配置、checkpoint/trigger/data hash、所有 seed。

### Gate P1

继续机制实验需同时满足：

1. `TEF(backdoor, full) - TEF(clean, full) >= 0.15`，且图像 bootstrap 95% CI 不跨 0；
2. `backdoor_lift >= 3 dB`；
3. full trigger 与至少一个结构化消融的规律性显著高于 equal-EVM random。

若第 1 项失败，停止“共享模板”主张；若第 2 项失败，停止“后门特异性”主张；若第 3 项失败，优先解释为通用扰动强度而非 IQ/DC 结构。

## 7. Phase 2：decoder 层级与方向敏感性探针（服务器）

**执行位置：服务器。** 使用 checkpoint、1024 张固定测试图，不训练。

### 任务 A：逐层激活差

在 decoder 每个层级记录 clean/trigger 激活：

```text
relative_shift_l = ||a_l(T(z)) - a_l(z)|| / ||a_l(z)||
template_fraction_l = ||mean_i delta_a_l,i||² / mean_i ||delta_a_l,i||²
```

同时保存每层空间频谱。检查条带周期是否在 stride-2 ConvTranspose 层后突然增强。

### 任务 B：trigger 方向 Jacobian

定义 `v = T(z)-z`，比较：

```text
directional_sensitivity = ||J_decoder(z) v|| / ||v||
```

分别在干净模型与后门模型计算；可用 JVP 或小步长中心差分，但必须固定同一信道状态。

### Gate P2

- 若后门模型在 `v` 方向敏感性显著高于干净模型，支持模型学到 trigger-specific response。
- 若周期能量主要在 stride-2 decoder 层后出现，支持 H3。
- 若第一层之前已形成完整空间周期，则 H3 降级，优先检查 encoder/reshape 与 trigger 分布。

## 8. Phase 3：训练消融（服务器）

**执行位置：服务器。** 包含反向传播和多配置训练，属于 `REMOTE_REQUIRED`。

先做 seed 0 pilot，每个变体 `10k steps`；只有能区分假设的变体进入 `80k steps × 3 seeds`。

### 必做变体

| ID | 后门模型 | Trigger | 目的 |
|---|---|---|---|
| A | encoder+decoder 可训 | 可训 | 复现 joint baseline |
| B | encoder+decoder 可训 | 固定为 A 的初始化值 | 判断共同优化是否必要 |
| C | encoder 冻结、decoder 可训 | 可训 | 判断联系主要是否存于 decoder |
| D | encoder 可训、decoder 冻结 | 可训 | 判断 encoder 是否能单独组织可分离 latent |
| E | 模型完全冻结 | 可训 | universal impairment 上界；不能称为后门训练 |

### 可选目标消融

- cap：`3 dB`、`6 dB`、`9 dB`。
- 旧 MSE-ratio hinge 与 capped dB hinge，在同一 checkpoint/trigger 初始化/steps 下比较。
- residual-diversity 正则仅作为机制验证：若加入后 PSNR drop 保持而模板规律性下降，支持 H2。不要把它默认当成更好的攻击目标。

### 种子设计

- Pilot：`training_seed=0`，固定 `poison_seed=0`。
- 确认：`training_seed=[0,1,2]`，第一组固定 `poison_seed=0`，隔离优化随机性。
- 若资源允许，再做 `poison_seed=[0,1,2]` 与对应 training seed，检验投毒子集稳健性。
- 每个 checkpoint 使用相同五个 evaluation channel seeds。

### 统计原则

- 训练 seed 是论文级独立重复单位；主表报告 3 个 seed 的均值、SD、最小/最大值和原始点。
- 只有 3 个 seed 时不把 test-image bootstrap CI 当作跨训练复现证据，也不强调 p 值。
- 图像 bootstrap 只描述固定 checkpoint 下的样本不确定性。

## 9. Phase 4：替换 decoder 的因果确认（可选，服务器）

只有 P2 显示周期结构在 ConvTranspose stride-2 层显著增强时才执行。

1. 新增架构：nearest/bilinear resize + stride-1 Conv，尽量匹配感受野和参数量。
2. 重新训练该架构的干净 P0，先确认 clean PSNR 与原架构差距不超过预设容差，例如 `0.5 dB`。
3. 使用 Phase 3 的 joint baseline 协议训练后门。
4. 比较 PSNR drop、TEF 和周期频带能量。

**支持 H3 的判据**：替代 decoder 保持相近 backdoor lift，但预注册周期频带能量至少下降 30%，并在 3 个 seed 中方向一致。

## 10. 服务器执行步骤

### 10.1 连接与预检

```powershell
ssh -i "C:\Users\Win 10 Pro\Desktop\paper\exp\.codex_prrf_gxhen_key" gxhen@100.67.214.61
```

登录后：

```bash
cd /home/gxhen/Wisig
cd jscc/deepjscc_backdoor
pwd
git status --short --branch
git rev-parse HEAD
python --version
nvidia-smi
df -h .
test -d data
test -f runs/clean_p0_c16_server_20260827/best.pt
```

不得读取或复制 SSH 私钥。若远端代码版本、数据、P0 checkpoint、GPU 或磁盘不满足要求，停止，不回退本地跑重任务。

### 10.2 持久运行

- 使用服务器既有调度器或持久会话；不要依赖一次 SSH 连接。
- 每次运行记录 PID/作业 ID、精确命令、开始时间、日志和输出目录。
- 输出目录必须全新，例如：

```text
runs/maxdrop_artifact_mechanism/<phase>/<variant>/seed_<n>/
```

- runner 会拒绝非空目录；不要删除或覆盖已有结果。
- checkpoint 和大日志留在服务器；本地只同步配置、manifest、JSON/CSV 指标、图表和 SHA-256。

## 11. 预期产物

| 产物 | 路径 | 格式 | 验收条件 |
|---|---|---|---|
| 协议 manifest | `runs/maxdrop_artifact_mechanism/**/manifest.json` | JSON | 含 commit、配置、数据/checkpoint/trigger hash 与独立 seeds |
| 控制评估 | `.../control_eval_*/metrics.json` | JSON | 8 个 conditions × 2 个 models 完整 |
| 残差统计 | `.../residual_statistics.csv` | CSV | TEF、cosine、频谱与 PSNR 指标齐全 |
| 机制图 | `.../figures/` | PNG/PDF | 同一色标、同一图像索引、标注 model/condition/SNR |
| 层级探针 | `.../layer_probe.json` | JSON | 所有 decoder 层都有 clean/trigger 差异 |
| 训练历史 | `.../history.json` | JSON | steps、loss、cap active rate、梯度范数完整 |
| 最终汇总 | `results/maxdrop_artifact_mechanism_summary/` | JSON/CSV/PNG | 不含大 checkpoint，所有输入可追溯 |

## 12. 执行顺序与停止条件

```text
P0 代码/协议正确
  └─否→停止并修复
  └─是→P1 checkpoint 控制评估
          └─无共享模板→停止 H1/H2/H3 深挖，报告负结果
          └─无 backdoor lift→归类普通 impairment
          └─存在模型特异模板→P2 层级探针
                                  └─定位 encoder/decoder/trigger
                                  └─P3 最小训练消融
                                          └─仅有明确证据时进入 P4 架构替换
```

不要一开始执行完整参数网格。每个 gate 都应生成一页中间结论，明确“观察”“推断”和“尚未验证”。

## 13. 最终报告问题清单

1. 规律性是 residual template，还是所有 reconstruction 真正坍缩到同一输出？
2. DC-only、IQ-only 各解释 full trigger 效果的多少？
3. 等 EVM 随机扰动是否产生相同规律？
4. 后门模型相对干净模型增加了多少方向敏感性？
5. 规律空间频率从 decoder 哪一层开始出现？
6. 联系主要写入 encoder、decoder，还是依赖共同优化？
7. 结论是否跨 training seed、poison subset 和测试 SNR 保持？
