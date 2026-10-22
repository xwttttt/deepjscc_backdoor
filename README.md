# DeepJSCC 后门与 OFDM 实验

本项目在 DeepJSCC 图像传输链路上研究 IQ/DC 触发器，并保留 P0 干净模型和 `stronger_struct` 候选模型。

## 目录结构

```text
src/jscc/                 模型、信道、数据集、训练、指标、触发器和编解码
apps/train_clean/         两阶段干净模型训练入口
apps/train_triggered/     IQ/DC 后门训练入口
apps/experiments/          OFDM Tx/Rx 客户端
scripts/                   数据准备、服务包装和运行流程
configs/                   实验与收发配置
models/                    checkpoint、触发参数和历史指标
```

训练与模型代码通过 `jscc` 包组织，任务入口放在 `apps/`。干净模型训练与触发训练循环集中在 `src/jscc/handler.py`；后门入口负责选择固定投毒索引并保存清单。

## 核心模块

- `src/jscc/metrics.py`：训练验证使用 PSNR；独立评估使用 MS-SSIM 与 LPIPS。构造 `Metrics` 时用布尔参数 `input_is_minus_one_to_one` 指定输入是否处于 `[-1, 1]`；本项目的 `[0, 1]` 图像显式设置为 `False`，不进行范围转换。
- `src/jscc/handler.py`：`Trainer` 负责两阶段干净模型的 MSE 训练、PSNR 选模和早停；其子类 `TriggerTrainer` 从第二阶段最佳权重开始，以固定 IQ/DC 参数和双 PSNR drop 目标训练后门模型。后门的四条分支共享同一信道状态和 AWGN 样本。
- `apps/train_triggered/main.py`：按种子一次选定投毒样本索引，将布尔投毒标记随训练图像送入 `TriggerTrainer`，并保存 `poison_manifest.json`。原先的 `AvailabilityPoisonManifest` 和 `AvailabilityPoisonedDataset` 类已删除；清单字段与选样方式保持原样。
- `src/jscc/trigger.py`：`apply_iq_dc_torch` 和 `apply_iq_dc_numpy` 用同一组固定参数对复数 latent 应用 IQ 失衡与按样本 RMS 缩放的 DC 偏置；Torch 版保留 latent 的梯度，NumPy 版保持输入形状和 dtype。
- `src/jscc/datasets.py`：直接读取本地 image-only LMDB，也兼容含 token 的 LMDB；图像裁剪到 `[0,1]`，100 条 valid 样本按索引 0–49 / 50–99 划分验证与测试。
- `src/jscc/visualization.py`：用无图形界面的 Matplotlib 保存训练曲线，以及固定验证图像的输入/重建对照 PNG。

后门训练仅支持普通梯度求和及当前两个 PSNR drop 损失；若启用旧配置中的投影梯度、EVM/功率惩罚、参数锚定或额外 guard，将明确报错。训练入口继续输出 checkpoint、历史、投毒清单、触发参数和指标 JSON；验证 clean drop 超过门槛时，checkpoint 与指标中的 `clean_constraint_satisfied` 会标记为 `false`。
本次指标输入范围接口调整已通过相关 Python 文件的 AST 语法检查及调用点检索，尚未运行指标数值测试。
handler 简化已通过相关 Python 文件的 AST 语法检查和引用检索；尚未进行数值等价测试。

## 三阶段训练

在专用 GPU 服务器执行 `bash scripts/train_stronger.sh runs/three_stage_new`。脚本依次运行：

1. 干净模型第一阶段固定 20 dB，最多训练 200 epoch；按 20 dB 验证 PSNR 选模，默认连续 20 epoch 无足够改善则早停，保存 `clean/stage1_best.pt`。
2. 从第一阶段最佳权重开始，第二阶段每个训练 step 独立均匀抽取 0–20 dB 的整数；默认最多 200 epoch，按 0、5、10、15、20 dB 验证 PSNR 的均值选模，保存 `clean/stage2_best.pt`。两个阶段使用独立 Adam 优化器。
3. 只有第二阶段最佳 checkpoint 能作为后门训练的 `--clean-checkpoint`。后门训练默认在 10 dB 训练和选模，训练期间只计算 PSNR drop；固定 IQ/DC 参数不进入优化器。训练结束后才计算 MS-SSIM、LPIPS，并报告 0、10、20 dB 三组测试结果。

干净阶段使用 MSE 更新模型，验证使用逐图 PSNR；最终画质报告使用 MS-SSIM 和 LPIPS。后门训练保留固定数据集投毒索引，未强制每个 batch 都包含投毒样本。训练初始权重默认随机初始化；需要从已有模型继续时，可单独运行 `apps/train_clean/main.py --init-checkpoint /path/to/checkpoint --output-dir /path/to/output`。可用 `--stage1-epochs`、`--stage2-epochs` 和两个 `--stage*-patience` 参数调整时长。

三个训练阶段默认显示每个 batch 的 `tqdm` 进度条，并每 5 个 epoch 更新曲线、保存一次固定验证样本的重建图。干净阶段的曲线记录训练 MSE 与验证 PSNR，后门阶段记录两项 hinge loss 与验证 PSNR drop；虚线分别表示 clean drop 上限和 triggered drop 目标。输出位于 `clean/plots/clean_stage1/`、`clean/plots/clean_stage2/` 和 `triggered/plots/trigger/`：`curves.png` 始终是最新曲线，`epoch_XXXX.png` 是定期快照，`best.png` 使用重新加载的最佳 checkpoint。预览使用固定验证样本和独立信道随机数生成器，绘图裁剪只影响 PNG。可用两个入口的 `--plot-every N` 调整间隔，`--no-progress` 关闭进度条；绘图依赖 `matplotlib`，进度条依赖 `tqdm`。

`runs/three_stage_new/clean/` 已保存第一、二阶段最佳 checkpoint。第二阶段在第 98 个 epoch 早停，最佳权重来自 epoch 78。此前测试阶段因 py-lmdb 同进程重复打开同一环境而中断；现已让验证集和测试集共享一个 LMDB 读取器，索引范围分别为 0–49 和 50–99。修复后的 clean checkpoint 指标已写入 `clean/metrics.json`。

已有 checkpoint 可在专用服务器直接补跑干净模型测试，无须重训：

```bash
python3 apps/train_clean/main.py --output-dir runs/three_stage_new/clean --eval-only --device cuda:0
```

`--eval-only` 从现有 `stage2_best.pt` 加载最佳权重，读取两个 checkpoint 的最佳 epoch 和验证 PSNR，并生成 `metrics.json`；它不会重建本次中断前未保存的逐 epoch 历史。后续新训练会在各阶段结束时立即保存对应的 `stage*_history.json`，避免最终测试失败时再次丢失历史。

训练入口和配置完成 Python AST、TOML、shell 语法静态检查；clean checkpoint、触发训练和 20 轮联邦实验均有实际指标产物。LMDB 修复后的 clean 评估与后门/联邦测试指标分别保存在 `clean/metrics.json`、`triggered/metrics.json` 和 `federated/metrics.json`。

当前后门训练的 IQ/DC 参数由配置中的 `gain_db`、`phase_deg`、`dc_magnitude`、`dc_phase_deg` 和 `order` 固定指定，不进入优化器。`best.pt` 的 `trigger` 字段及 `trigger.json` 保存这组参数，发送端继续用 `apply_iq_dc_numpy` 读取相同配置。触发训练已完成 60 epoch，选中 epoch 53（零起始索引）；测试集 10 dB clean/triggered drop 为 0.039/24.972 dB。完整设置和 0/10/20 dB 指标见 [2026-10-22 实验记录](docs/progress/2026-10-22.md)。

`apps/train_triggered/main.py` 使用触发参数对象的字段直接写入 `trigger.json`，不依赖 `dataclasses.asdict`；该入口也不依赖 `contextlib.nullcontext` 或 `hashlib`。

训练以 PSNR drop 构造 clean/triggered 目标并按 PSNR 选模；独立评估记录 MS-SSIM（越高越好）与 LPIPS（越低越好）。LPIPS 使用 TorchMetrics 的 AlexNet 预训练特征网络，输入图像范围为 `[0, 1]`，首次启用时需可获得对应的 torchvision 权重。

## 联邦后门持久性实验

`src/fed/` 实现 benign FedAvg 微调实验：从 runs/three_stage_new/triggered/best.pt 初始化全局模型，将 3200 张训练图像均匀随机划分到 10 个客户端；每轮随机抽取 5 个客户端，各自用干净图像、随机整数 SNR 0–20 dB 做 1 个本地 epoch，再按客户端样本数执行 FedAvg。它用于观察已训练后门在后续干净联邦微调中是否衰减，不模拟恶意客户端重新植入攻击。

在专用 GPU 服务器激活 `sionna` 环境后，从仓库根目录启动；默认直接训练 20 轮（完整参数见 [`src/fed/README.md`](src/fed/README.md)）：

```bash
conda activate sionna
PYTHONPATH=src python -m fed.main
```

默认使用阶段进度条，每轮报告验证集 0、10、20 dB 的 clean/triggered PSNR drop；每 5 轮更新曲线和固定验证样本重建图，并保存每轮全局 checkpoint。结果写入 runs/three_stage_new/federated/。可用 --clients、--client-fraction、--rounds、--local-epochs、--plot-every 修改实验设置。持久性按训练选模 SNR 10 dB 的验证指标判断：clean drop ≤0.5 dB 且 triggered drop ≥20 dB；最终另报告测试集 0、10、20 dB 的 PSNR drop、MS-SSIM 和 LPIPS。该实现当前使用 IID 客户端划分，结论仅对应这一联邦微调设置。

联邦训练已完成 20 轮。最终 checkpoint 在 0、10、20 dB 测试集上的 triggered PSNR drop 分别为 10.294、15.516、17.047 dB；10 dB 验证持久性判据从第 1 轮起不再满足。`plots/snr_sweep_clean.png` 与 `plots/snr_sweep_triggered.png` 使用第 20 轮权重，对 3 张固定验证图展示 0–20 dB（每 1 dB 一档）的重建。

触发器训练与联邦实验的逻辑、参数和完整指标汇总见 [2026-10-22 实验记录](docs/progress/2026-10-22.md)。

## 模型与实验

- `models/p0_clean.pt`：干净基准模型，用作后门实验的参考和初始化。
- `models/stronger_struct_best.pt`：从 P0 初始化的 IQ/DC 后门候选模型；对应参数见 `models/stronger_struct_trigger.json`。
- `models/stronger_struct_metrics.json` 与 `models/stronger_struct_run_manifest.json`：保留历史实验指标和来源信息，不含文件哈希。

`stronger_struct` 使用 `c=16` 的 DeepJSCC 架构；它是实验配置名，不是新的网络架构。该运行选出的 checkpoint 来自第 34 轮。历史验证集 clean drop 为 0.710 dB，高于 0.5 dB 门槛；测试集 28.165 dB triggered drop 来自直接模型推理，不是 OTA 测量值。

## 发送与接收校验

`scripts/send_latents.py` 检查 burst ID、latent 路径和文件长度，再逐条调用 `apps/experiments/TX.py`。`scripts/validate_rx.py` 检查控制 CRC、文件长度、有限数值、NPY 形状和记录数，并可核对 Tx/Rx burst ID。

所有 SHA-256 计算和校验均已移除，包括训练清单、编解码 manifest 与 Tx 元数据。接收端不再按字节把批量 NPY 记录和单个 burst 文件配对，因为 NPY 不含 burst ID。控制 CRC 只验证控制信息，不代表业务 IQ 校验通过。

## 当前数据与运行限制

本地 `datasets/DIV2K/` 的 LMDB 为 image-only 格式，训练集 3200 条、valid 集 100 条，现可由 `src/jscc/datasets.py` 读取。固定测试 NPY 和原生 OFDM 源码当前不在工作树中；编码时通过 `--dataset` 指定已有的预处理 NPY，离线 OFDM 流程还需要单独提供 OFDM 服务。
