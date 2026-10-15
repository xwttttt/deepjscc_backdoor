# 独立 c16 替代模型与 P0 迁移攻击

2026-10-14 已实现并在专用 gxhen RTX4090 启动。固定原P0为最终目标，替代模型从随机初始化独立训练：c16/depth5/width32/seed11 和 c16/depth7/width40/seed29，各10000步。原P0不参与代理训练、攻击优化或checkpoint选择。

实现采用三个独立进程：`fit` → `attack` → `evaluate`。前两个阶段禁止传入目标checkpoint；`attack_complete.json` 和攻击文件SHA256确定后，最后进程才加载原P0。代理codec优化结束后冻结，攻击阶段核对权重指纹不变；目标最终评估全程no_grad并检查权重/梯度。代理训练和选模用train3200及valid0–49，最终迁移测试valid50–99。数据仍固定DIV2K Cosmos LMDB。

通用生成器只输入随机z，使用两个代理的平均重建PSNR训练12000步，checkpoint只按代理验证drop选择；目标测试4500行（3SNR×6PSR×50图×5信道重复）。逐图分支在两个代理上联合优化80步，按代理损失锁定完整频域扰动，再迁移至P0；SNR10、PSR−16/−10各50图，共100行。逐图分支仍知道当前图像和实际Ha，不能把“模型未知”误称为“图像和CSI也未知”。

两类攻击使用相同连续OFDM/单位合法AWGN与独立8抽头Ha、频率排列、相位/时延变换；原P0质量基线不换成c4/c8，也不使用QAM硬判决。Ha前频域投影与离线接收功率裁剪、逐条随机同功率控制均保留。接收功率oracle仍是仿真假设。

入口 `scripts/run_magmaw_c16_surrogate.py`，帮助 `--help`；完整启动 `scripts/launch_magmaw_c16_surrogate.sh`；核验 `scripts/report_magmaw_c16_surrogate.py`。本地4项阶段/身份隔离检查、静态编译和CLI通过。当前只是运行中，尚无迁移结果；不能套用此前直攻P0的25.256dB。

服务器目录 `/home/gxhen/Wisig/jscc/deepjscc_backdoor/experiments/magmaw_c16_surrogate_20261014_01/`，checkpoint、原始扰动张量和大日志只留服务器。
