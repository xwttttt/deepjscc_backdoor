# Magmaw 图像攻击逻辑与规模对齐

用户明确选择继续固定 DIV2K Cosmos CV16 LMDB，优先对齐算法与规模。本轮仅复现图像 DeepJSCC 攻击，不包含视频、语音、文本及下游分类器。主要参考用户提供 NDSS 2025 PDF §VI、算法 1、式 (8)–(15)、附录表 III，及[官方仓库](https://github.com/juc023/Magmaw)。

## 方法对应

| 项目 | 当前全量适配 | 差异 |
| --- | --- | --- |
| PGM | 100 维随机向量；1024→256→64 转置卷积、5 个残差 SE 块、tanh；输出 32×64 复数模板 | 对齐公开 `Res_Generator` 层规模，独立实现与复数布局适配 |
| 扰动转换 | 符号延拓、逐 burst 子载波排列、0–64 样本分数时延、随机全局相位 | 时延对应公开 channel 的 20 MHz 下一个 OFDM 周期；没有真实包同步搜索 |
| 损失 | 图像重建 MSE + 判别器 minimax + 原始生成扰动间 L1 diversity；βds=βdv=1 | 不含其他模态或 downstream；L1 按元素平均；判别器为本地 Conv1d |
| 集成 | 两个独立图像架构，深度 6/10、宽度 28/36；随机选择编码率和调制，每 batch 联合两个架构梯度 | 保留本地卷积 codec 原语，非论文原 JSCC [98]；不声称表 III 参数完全相同 |
| 迁移 | 深度 5、宽度 32 的独立目标，完全不参加攻击反传或选模 | 不需要目标梯度，仍采用相同协议和 DIV2K 分布 |
| 编码率 | c=4/8 对应 L/(3HW)=1/12、1/6 | 前一轮 c=16 为 1/3，不能沿用作论文匹配编码率 |
| OFDM | FFT64、CP16、48 data +4 comb pilot +12 null、一个前导 training symbol、时域 FIR、导频 MMSE CE/EQ、QPSK/16/64-QAM | 采用单位平均功率标准方形 QAM；硬前向、软梯度；未包含软同步、CFO、硬件非线性 |
| 信道 | victim 与 attacker 独立 8 径指数 PDP（decay=4）；2000 个训练 attack 信道与独立 2000 个评估信道 | 合成信道，非论文实测 2000 CSI；使用接收功率 oracle 做接收 PSR 投影 |
| baseline | 无攻击、独立时域 Gaussian、vanilla UAP、sync-free UAP | 无 one-hot vector 模态基线；两个 UAP 在训练中分别省略全部变换/只保留时延相位，评估均经过完整攻击传播 |

## 数据与规模

固定 LMDB 训练键 0–1599 用于代理与攻击，键 1600–3199 用于目标 clean 训练，两者无重叠。验证键 0–49，用于 clean codec 选模和 surrogate-only 攻击选模；留出键 50–99 仅最终评估。数据仍来自同一 DIV2K 分布，不能声称复刻论文不同来源的训练/测试数据。采用全部预定分割，不是少量 batch 冒烟。

正式配置 `configs/magmaw_image_full.toml`：6 个 codec（2 surrogate architecture ×2 rates +2 targets），每个 5000 steps、batch 8；各模型对三种调制随机训练。三个攻击方法各 60 epochs，每 epoch 完整 1600 个攻击训练样本（200 batches），各 12000 更新；Magmaw 每 batch 对所选 rate 的两个代理联合优化。attack Adam lr=2e-4、betas=(0.5,0.999)。这些训练步数/epoch 是本轮复现预算，**论文和公开评估入口没有提供可核验的训练总预算，不能称为论文原值**。

留出全量矩阵为 2 rates ×3 modulations ×6 PSR(-20:2:-10) ×5 channel repeats ×50 samples，9000 条图块/信道条件，每条都比较 5 个方法。只有一个 attack training seed 1234，5 次信道重复不是 5 个训练种子。三个攻击在同一训练预算下比较。目标模型只用于最终测试，不用于 attack 选模。

## 执行与状态

入口 `scripts/run_magmaw_image_full.py`；先 CPU 合约测试，再服务器 CUDA `--smoke` 跑通全部阶段，最后 `scripts/launch_magmaw_image_full.sh` 提交 full。输出目录必须新建；阶段记录 `status.json`、源码/配置 identity、codecs、攻击 history/checkpoints、逐条件 test_samples 和最终 metrics。大日志与 checkpoint 保留服务器。

当前：2026-10-13 全量已在 `gxhen` 完成，实验目录 `/home/gxhen/Wisig/jscc/deepjscc_backdoor/experiments/magmaw_image_full_20261013_01/`，PID 1816754，日志 `pipeline.log`，正式输出 `runs/full/`。6 个 codec 各 5000 steps、3 个 attack 各 60 epochs 均完成。已核验 `exit_code.txt=0`、`metrics.json.status=completed`；报告脚本进一步确认 36 条条件、9000 条记录、50–99 每个样本/信道重复身份齐全且无重复，实际接收 PSR 未超预算。Magmaw 在 36 条条件中的最大平均 PSNR drop 仅 0.600188 dB，**未复现论文的约 8 dB 退化，不能称为论文复现成功**。对照逐项结果在服务器报告中，不用代理验证数据替代最终迁移指标。

本地编译与 CLI 通过，因缺 torch 数值测试跳过；服务器 CPU 合约 3 项通过，CUDA 全链路 smoke 完成。SCP 连续两次因连接超时/重置失败后已停止同方式重试，改用 SSH 标准输出获取压缩指标包；完整 metrics、训练历史、summary 与图表现已同步本地，本地再次通过源码哈希、矩阵/PSR、退出码检查，图表已视觉检查。checkpoint 与大日志继续留服务器。完整结果见 `results/magmaw_image_full_20261013_01/README.md`。
