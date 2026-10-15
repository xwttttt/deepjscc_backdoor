# Magmaw-inspired DeepJSCC OTA 对抗扰动

主要参考：Chang et al., NDSS 2025, *Magmaw: Modality-Agnostic Adversarial Attacks on Machine Learning-Based Wireless Communication Systems*，用户提供 PDF，方法 §VI、算法 1、式 (8)、(9)、(14)、(15)。

本实现复用 `DeepJSCCAdapter`、干净 c=16 P0、原有 AWGN 和固定 DIV2K Cosmos LMDB 分割。冻结编码器/解码器，不投毒、不训练后门。生成器只接收独立随机向量，输出 4×64 复数模板；延拓覆盖整帧，随机子载波排列、0–8 样本分数时延、全局相位、独立三径攻击者信道构成 EOT。最大化源图像重建 MSE，以验证集 PSNR drop 选模，最后才评估留出测试键 50–99。

模型：`y = z + n + Ha Pτ(G(u))`。这里受害者信道为单位增益 AWGN，攻击者独立三径通过 FFT 映射到 64 个数据子载波；等效于理想均衡的数据子载波级 OTA 仿真。每 burst 以无噪声合法 latent 功率校准接收端 PSR=-16 dB。该功率校准使用合法接收功率这一 oracle 信息，不代表未知 CSI/功率下的黑盒能力，也不约束发射端路径损耗。没有 CP、pilot、同步失效、CFO、真实 OFDM 服务或射频发送；不能称为完整物理波形仿真。

当前是图像单模型白盒简化基线，MLP 生成器代替论文 ResNet；未实现多模态/多模型集成、判别器隐蔽性损失、diversity loss、下游任务。不能宣称完整复现 Magmaw 或比高斯干扰更强，需实际测试。随机基线与 UAP 使用相同接收 PSR、合法 AWGN 样本和 EOT 随机状态。

服务器命令（从独立实验快照根目录运行）：

```bash
export PYTHONPATH="$PWD/deps:$PWD/src"
/home/gxhen/Wisig/jscc/deepjscc_backdoor/.venv-direct-v2/bin/python scripts/run_ota_adversarial.py --checkpoint /home/gxhen/Wisig/jscc/deepjscc_backdoor/runs/div2k_lmdb_cv16_20260923/p0_full/best.pt --output-dir runs/NEW_RUN --steps 400 --batch-size 4 --psr-db -16 --snr-db 10
```

服务器复用 `.venv-direct-v2`，缺失依赖安装到实验目录 `deps/`，不改已有环境。`launch_ota_adversarial.sh`、`evaluate_ota_adversarial.sh`、`sweep_ota_adversarial.sh` 保存退出码；运行时须使用新输出目录。`--evaluate-generator` 只读取已训练生成器，避免评估错误后重训；验证和留出 LMDB 视图在评估后关闭句柄，防止同一数据库重复打开。结果记录配置、源代码、数据与 checkpoint 哈希。checkpoint 与训练日志仅保留在服务器；本地只同步指标和 manifest。

2026-10-13 已完成单种子 400 步训练与 -20/-16/-10 dB 三预算测试，详见 `results/magmaw_ota_20261013/README.md`。这是初步有效性证据，尚未开展多种子、不同合法 SNR、时延范围外测试、受害者多径/估计均衡、跨模型迁移、完整时域 OFDM 或真实 OTA。样本级置信区间不是训练种子间不确定性，也不能排除同一源图像多个裁剪造成的相关性。
