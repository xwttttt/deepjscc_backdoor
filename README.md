# DeepJSCC IQ 不平衡与 DC Offset 可用性攻击研究工具包

2026-10-16 项目快照的 GitHub 目标为 [`xwttttt/deepjscc_backdoor` 的 `codex/project-upload-20261016` 分支](https://github.com/xwttttt/deepjscc_backdoor/tree/codex/project-upload-20261016)。上传范围包括源码、配置、测试、文档、实验结果/图表和包含固定测试集及两份模型的完整部署 ZIP；临时目录、缓存、服务器训练目录和重复解压副本不纳入 Git。详细范围与本地检查见 [`docs/github_upload_20261016.md`](docs/github_upload_20261016.md)。该快照不增加新的训练或实机验证结果。

## Magmaw 参考的 OTA 对抗攻击（2026-10-13）

2026-10-14 已实现[主目标优先的通用攻击优化](results/magmaw_priority_c16_20261014_01/README.md)：旧日志判别/重构梯度比中位2.082、最大10.711；训练批探针发现代理梯度与P0方向接近正交、判别轻微反向。新增P0 PSNR主目标、逐步辅助梯度冲突投影/范数上限、空间覆盖、4次EOT、验证选pilot及双种子细调，保留原c16/完整信道变换/功率条件。21项相关本地检查、编译及CLI通过，CUDA烟测通过，正式pilot/细调已启动；protected由验证集选中，最后读取到seed1234完成1800步、验证SNR10/PSR−10 drop7.068dB及seed5678的400步。随后旧服务器地址连接超时且Tailscale列为离线，用户随后确认也无法连接，目前等待服务器恢复访问，未重复提交训练；远程当前状态与最终测试结果尚未确认，不将验证进度作为完整效果。

2026-10-14 已完成[完整图像攻击机制及对照](results/magmaw_complete_c16_20261014_01/README.md)：ResNet随机z驱动的完整长度均值/独立基底生成器、原P0与两个独立代理集成、G/D交替训练、重构MSE/判别/多样性联合目标、双信道EOT、排列课程和梯度平衡；模型权重冻结，G不输入当前图片/Ha。首版系数饱和导致随机码塌缩，已定位并以signed-z支路、可微QR和5%随机RMS修正；第二版两配置各1600步，共8100条，退出码0。SNR10/PSR−10下完整机制38.280→33.331dB（drop4.950），关闭正则同结构drop6.730，共享暖启动同池drop6.277；新增正则暂未提高PSNR攻击强度。输出多样性0.0568/0.0570，强制饱和回归和冻结/功率/样本矩阵/哈希/本地独立统计核验通过，图表已检查；另附固定四图重构及统一区域放大，逐图PSNR精确重放原评估（差异0dB），图像主体仍可辨识，主要为竖向条纹/局部亮斑。14项相关本地检查、编译及CLI通过；balanced为明确适配，单种子，不声称官方预算、多模态、黑盒或实机复现。

2026-10-14 已完成[通用攻击容量/变换诊断](results/magmaw_c16_uap_diagnosis_20261014_01/README.md)：四种共享UAP各800步，连同原PGM在三种变换条件交叉评估，共4500条。完整变换、SNR10、PSR−10下原PGM重评drop1.709dB，32符号共享1.975dB，1408符号full训练4.060dB，无排列训练后full测试5.884dB（38.269→32.385），同功率随机drop1.408dB。推理不输入图片/Ha；原P0冻结。理想对齐共享UAP可drop24.447dB，但同一权重full测试仅3.158dB，说明稳健性限制，不能冒充未知信道结果。源码/模型/数据哈希、配对基线、功率/样本矩阵、服务器与本地汇总核验通过，退出码0，图表已检查；此轮单种子诊断，不宣称确认性多种子结论。后续课程训练另见完整机制实验。

2026-10-14 已新增[独立c16替代模型版本](results/magmaw_c16_surrogate_20261014_01/README.md)：两个代理随机初始化，各10000步，通用生成器在代理ensemble训练12000步，并附80步逐图代理优化后迁移。三个进程强制划分fit/attack/evaluate；前两阶段禁止目标checkpoint，攻击文件锁定后最后阶段才加载P0，只做no_grad评估。4项本地隔离检查、编译和CLI通过；专用服务器训练及4600条目标评估已结束、退出码0，结果尚待本地同步复核，不能沿用此前目标白盒的25dB。

2026-10-14 已完成[原c16 P0 OTA实验](results/magmaw_c16_20261014_01/README.md)，原P0哈希严格匹配、全程冻结；连续OFDM与同噪声直接P0门控通过，恢复SNR10干净38.272dB、SNR100干净42.148dB。三组通用PGM各12000步、共13500条评估，效果仍弱（SNR10、PSR−10下约1.8dB drop）；另完成100条逐图/已知Ha白盒80步投影Adam攻击：−16dB为38.258→20.392dB（drop17.866），−10dB为38.258→13.002dB（drop25.256）。这不是未知图像/CSI的通用Magmaw结果。相同数据/接收链路/共享噪声的750条保存后门复核在SNR10下38.185→13.051dB（drop25.135），并注明信息与功率条件不同。最终退出码0；源码/模型/数据/功率/样本矩阵与独立汇总核验通过，图表已视觉检查；FFT精度门控失败已诊断修复，未放宽容差。指标与图表已同步，checkpoint和大日志留服务器。

2026-10-14 已完成[OTA与既有后门的原始PSNR复核](docs/ota_vs_backdoor_psnr_audit_20261014.md)：当前OTA主方案在−10dB接收PSR上限下为25.093→19.373dB（drop5.719），原c16 P0简化OTA为38.260→31.065dB（drop7.195）；当前干净基线明显更低，不能据更低的攻击后PSNR宣称攻击增强。强LMDB后门候选为41.685→13.520dB（drop28.165），原P0为42.148dB；正式v2的近无噪声OFDM simulate为34.933→16.033dB（drop18.900）。前3组LMDB哈希及测试键一致，但模型、SNR、量化和接收链路不同；后门权重会被训练，OTA目标权重冻结，尚未形成同条件强度比较。当前OTA仍未达到既有强后门的失效幅度；本次只完成指标/身份/图像复核，未运行新的c16对齐实验。

2026-10-14 后续已完成以**有限干扰功率下明显降低PSNR**为目标的[失真优先实验](results/magmaw_effective_20261014_01/README.md)：关闭critic/diversity，对比MSE与直接PSNR目标，三组各12000步、共27000条目标评估记录，服务器退出码0。按代理验证选择MSE主方案；跨六协议的平均drop在接收PSR −16/−14/−12/−10dB下为2.079/3.170/4.509/5.719dB，同功率随机为0.127/0.202/0.316/0.449dB。第二训练种子仅覆盖直接PSNR目标，MSE仍单种子。接收功率上限、逐条随机功率匹配、样本矩阵、源码/模型/数据哈希与本地独立汇总通过；5项解析测试、CUDA公式/功率检查通过，图表已视觉检查。结果、重建对照和方法说明已同步，checkpoint与大日志留服务器。

2026-10-14 已完成[算法第 VI 节精读与式级图像分支复现](docs/magmaw_algorithm_reading.md)：视觉核对论文原页，新增独立 `magmaw_paper.py`，区分论文与公开默认实现；修正 L1 范数归约、统一代理项求和、固定公共最大攻击长度和 Ha 前式 (14)–(15) 投影顺序，单独报告投影/发射/接收功率；确认公开固定提交中一个复数乘法实部问题，本轮使用原生复数 FIR。四项本地手算 oracle、编译/CLI、CUDA 公式值与 minimax 梯度检查均通过。服务器五组 400-step 诊断、18000 留出记录和 12000-step 字面公式全量、9000 记录全部完成，退出码 0；[指标、梯度诊断、图表和身份哈希](results/magmaw_paper_20261014_01/README.md)已同步。**字面公式组最大条件平均 drop 0.682 dB，仍未复现论文效果**；去掉 critic/diversity 的400-step消融为6.875 dB，不冒称Magmaw复现成功。下述旧指标不代表本轮。

后续全量升级见[论文逻辑与规模对齐说明](docs/magmaw_image_full.md)。用户确认继续 DIV2K；新增独立代理/目标模型、论文规模残差 PGM、判别器与 diversity 损失、时域 OFDM/导频 CE/EQ、两种论文编码率、三调制、六 PSR 与 UAP 对照。本地编译/CLI 通过（torch 测试跳过），服务器 CPU 合约 3 项及 CUDA 全链路 smoke 通过。正式任务已在 `gxhen` 的 `experiments/magmaw_image_full_20261013_01/` 完成：6 个 codec 各 5000 步、3 个 attack 各 60 epochs；退出码 0，36 条条件与 9000 条留出记录通过完整性和 PSR 预算检查。**Magmaw 的最大平均 PSNR drop 仅 0.600 dB，未复现论文约 8 dB 的效果**；全部条件下弱于 vanilla/sync-free UAP，仅 1/36 条条件略强于 random。完整指标、训练历史、哈希与[图表/报告](results/magmaw_image_full_20261013_01/README.md)已同步并本地复核，checkpoint/大日志留在服务器。下述 1.541/3.137/7.195 dB 仍仅属于之前简化白盒 pilot，不能用作本轮结果。

新增[方法与边界说明](docs/ota_adversarial_magmaw.md)、`src/rf_backdoor/ota_adversarial.py` 和 `scripts/run_ota_adversarial.py`。复用干净 c=16 模型、AWGN 和固定 Cosmos LMDB，冻结模型训练随机输入通用扰动生成器；加入独立攻击者三径、时延/相位/子载波排列 EOT，按接收端 PSR 对比随机干扰。这是单模型白盒、理想均衡数据子载波级 OTA 仿真，没有真实射频发送，也不是论文完整复现。

已在 gxhen RTX 4090 完成 400 步训练与三个功率预算各 50 张留出图块评估，SNR=10 dB，seed=20261013；同一个在 PSR=-16 dB 训练、验证集选中的生成器复用于所有测试点。干净 PSNR=38.260 dB；PSR=-20/-16/-10 dB 时学习扰动的 PSNR drop 为 1.541/3.137/7.195 dB，同功率随机干扰为 0.259/0.638/2.185 dB。[结果报告、逐样本指标、哈希和图表](results/magmaw_ota_20261013/README.md)已同步，checkpoint 与大日志留在服务器。数值合约测试在服务器 CPU 上 2 项通过；本地编译与 CLI 检查通过，PyTorch 测试因未安装依赖跳过。本次未改动现有 clean demo，也未运行真实 OFDM 服务或实机 OTA。

本项目研究编码器归一化输出上的 IQ imbalance 与 DC offset 是否能形成稳定、通用、受物理约束的可用性攻击触发器。攻击目标是降低接收端重建质量，不是 all-to-one 目标重建。数字仿真作为可复现基线；在自有或明确授权的设备、频段和环境中，并满足射频安全与当地法规要求时，可进一步开展 SDR、线缆回环或 OTA 验证。第三方系统和未授权目标不在项目范围内。

当前主基线以本仓库新提交的 `src/src/deepjscc/` 为准：编码器输出原生复数基带 `[B,L]`，信道由 `src/src/channels.py` 提供，实验适配逻辑位于 `src/rf_backdoor/adapter.py`。旧 Chunbaobao checkout 仅保留为历史来源和兼容入口，不再参与新训练或评估。

本项目当前指定的**正式攻击版本**是 [`DIV2K 双 drop v2（局部化结构 drop）`](docs/div2k_two_drop_official.md)：训练与选模只使用 clean drop、triggered drop，不使用部分触发、交互项、EVM、功率或感知指标；v2 把 triggered drop 定义为“去掉全局色调后的局部化结构损伤”，避免 v1 那样只用整体偏色就拿到 drop。v1 的同名文档章节与配置保留为对照基线。历史 CIFAR-10 MaxDrop 是来源与对照，不是当前正式版本；下文的 `availability_v2` 是另一条既有基线流程。

2026-09-28 标记：v2 结构退化模型仍是下一阶段收发对照的当前候选；另一个视觉失真更强的 LMDB 候选已在专用 GPU 服务器完成训练。留出测试 clean drop 0.463 dB、整帧 triggered drop 28.165 dB，但验证集 clean drop 0.710 dB 超过 0.5 dB 门槛，暂不替代正式版。新实验使用[独立配置](configs/div2k_two_drop_struct_lmdb_stronger_20260928.toml)，[指标与直接推理重建图](results/div2k_lmdb_cv16_20260928_stronger/README.md)已同步；尚未经过 OFDM 或实机 RF。旧原图裁剪与新 LMDB 数据集不可混称。

后续新实验的数据入口统一采用**预生成的 DIV2K Cosmos image/token LMDB**；原图裁剪与 tokenizer 编解码只在离线制作阶段进行，训练时只读固定 LMDB 样本。既有 `.npy` 实验保持原始数据身份，不与新 LMDB 实验混称。转换条件与当前缺口见 [`docs/div2k_lmdb_dataset_policy.md`](docs/div2k_lmdb_dataset_policy.md)。

已将固定 Cosmos-LMDB 图块预处理来源、Encoder→`TX.py`、`RX.py`→Decoder 的分机操作合并为[单一操作手册](docs/operation_manual_div2k_txrx.md)，并提供包含数据和两份分开命名模型的压缩包 `artifacts/div2k_txrx_bundle_20260930_complete.zip`。包内有用户原三个预处理脚本、固定 50 图 `test.npy` 与来源 manifest、模型代码、TX/RX 接口、哈希核对过的干净 P0，以及用户指定的 stronger_struct 运行内最佳 `best.pt` 和配套 trigger JSON；不含原始 DIV2K HR 图片或完整 LMDB。当前包只支持后门模型的未触发推理，尚未应用 IQ/DC trigger。完整重新制作 LMDB 仍依赖外部 CTJSCC Cosmos tokenizer。打包与验证状态以手册第 6 节为准，未部署服务器或进行 RF 测试。

## 正常 DeepJSCC / OFDM 图块 demo

这一条链路独立于上面的攻击训练：不使用后门 checkpoint，不默认施加 impairment，也不把 256×256 图块拼成整图。流程为固定 DIV2K LMDB 派生数据 → batch Encoder → 逐图块 TX/OFDM → RX 同步、CSI 与均衡 → Decoder 重建 → 分阶段保存。原始数据离线制作使用 `src/src/make_train_lmdb_dataset.sh`、`make_train_lmdb_dataset.py`；`data_provider.py` 提供 LMDB 读取逻辑，但 DeepJSCC 输入须经 `[0,1]` 适配，不能直接沿用其 `[-1,1]` 归一化。收发时一次加载固定导出数据，不动态读取或裁剪原图。

代码和架构说明集中在 [`scripts/platform_integration/clean_demo/README.md`](scripts/platform_integration/clean_demo/README.md)；可复制的脚本与命令行步骤见 [`OPERATIONS.md`](scripts/platform_integration/clean_demo/OPERATIONS.md)。推荐用 `run_clean_demo.py --run-id <新ID>` 启动：默认只做 1 张图块的非射频独立 TX/RX 文件回环，`--max-images 0` 才跑完整 50 张，自动创建不覆盖旧结果的目录。六个职责文件为 `preprocess.py`、`encoder.py`、`tx.py`、`rx.py`、`pipeline.py`、`decoder.py`。Encoder 默认每批 4 张，输出每张图独立的 `complex64` latent；序列化为 `cf32le` 二进制 IQ 载荷后逐包发送。可选 IQ/DC impairment 明确作用于 OFDM 调制前的 latent；clean 运行不传 `--impairment`。

对于收发主机分离的平台，现行**数据层**入口见[数据适配手册](scripts/platform_integration/clean_demo/DATA_ADAPTERS.md)：沿用用户提供脚本生成的 Cosmos CV16 LMDB 及其固定 `test.npy` 导出，不在收发时重做裁剪或 tokenizer；`src/src/encode_prepared_npy.py` 批量生成 complex64 latent，显式 `--send` 时逐文件调用支持 `--input-bin` 的平台 `TX.py`；原 `RX.py` 存盘后由 `src/src/decode_rx_npy.py` 重建。旧 `clean_demo/run_tx_data.py`、`run_rx_decode.py` 不作为该分机流程入口。Git 仓库 `ofdm-based` 原版 `TX.py` 只有 fake 输入，需部署文件输入适配版；`RX.py` 不改。本地静态编译和 TX CLI 检查通过，但当前 Python 缺少 torch，未跑模型推理；xwt 未连接，未同步或实机验证。原有双工入口保持不变。

每次运行使用一个**新的**输出目录，目录内只保留 `input/`、`encoder/`、`tx/`、`rx/`、`decoder/`、`manifest.json` 和 `service.log`。`manifest.json` 关联样本 ID、burst ID、数据与模型哈希、CRC、CSI 和各阶段产物。`tx/` 始终保存送入 OFDM 服务的 IQ 载荷；只有下面的独立 TX/RX 文件回环模式还会保存明确命名的完整时域包波形 `*.waveform.cf32le`，两者不能混称。

本地短时 CPU 合约检查：

```powershell
python -m unittest discover -s tests -p test_clean_demo_modules.py -v
```

两台服务器都已部署同版 demo：xwt 为 `/home/xwt/jscc-system-based-on-liquid/demo_clean_deepjscc_20260923/`，gxshen 为 `/home/gxhen/Wisig/jscc/deepjscc_backdoor/demo_clean_deepjscc_20260923/`。完整 OFDM 服务目前在 xwt；gxshen 保存同版代码及干净模型来源，不将其描述为已验证的 OFDM 主机。从 xwt 的 demo 目录运行 1 张固定图块：

```bash
cd /home/xwt/jscc-system-based-on-liquid/demo_clean_deepjscc_20260923
/home/xwt/miniconda3/envs/jscc/bin/python -m scripts.platform_integration.clean_demo \
  --dataset ../data/DIV2K/lmdb_cv16_test_uint8_20260923/test.npy \
  --checkpoint ../models/div2k_lmdb_cv16_20260923/p0_clean.pt \
  --service ../build/ofdm-workbench-service \
  --output-dir ../captures/clean_deepjscc_modular_NEW_RUN \
  --mode simulate --batch-size 4 --max-images 1
```

删除 `--max-images 1` 可处理全部 50 张固定测试图块，但必须更换输出目录。已完成的 batch-4 数字仿真保存在 xwt 的 `captures/clean_deepjscc_modular_batch4_20260923_sim50/`：50/50 CRC 通过，平均图块 PSNR 为 41.967 dB；本地仅同步了 [`结果 manifest`](results/div2k_lmdb_cv16_20260923_full/clean_modular_batch4_sim50_manifest.json)。这不是 OTA 或真实 SDR 指标，也不能与 10 dB AWGN 模型评估直接比较。

为单独验证 xwt 的 TX 与 RX 接口，新增 `--transport file-services`：每张图块经独立 `ofdm-tx-service` 组帧并保存完整波形，再由独立 `ofdm-rx-service` 读取波形、估计 CSI、均衡并返回 Decoder。该路径使用与 xwt `TX.py`/`RX.py` 相同的 ZeroMQ 数据协议，但不能同时运行另一个 `RX.py` 消费同一端口。运行命令见 [demo README](scripts/platform_integration/clean_demo/README.md)。已完成 50 张图块的文件回环：50/50 CRC、50 个完整包波形、平均 PSNR 41.967 dB，清单为 [`clean_standalone_file_sim50_manifest.json`](results/div2k_lmdb_cv16_20260923_full/clean_standalone_file_sim50_manifest.json)。这条验证没有通过 B210 发射；真实空口仍需明确设备连接、频率、天线和增益。

收发平台有两个**独立启动入口，但共用一套数据契约和处理链**。两者都执行
Encoder、OFDM TX/RX、CSI/均衡、Decoder；TX 输入均为 JSON + `cf32le` latent，
RX 输出均为 JSON + 64-bin `cf32le` CSI + 均衡后 `cf32le` latent。离线版仅以
仿真/文件回环替代 SDR 发射、真实信道和 SDR 接收；在线版走真实射频路径。输入
图块、burst ID、阶段目录和 manifest 格式保持一致，只有信道导致的数值、CRC
及丢包结果可能不同。离线额外保存的完整包波形是检查产物，不改变 Decoder 输入。

| 版本 | 启动脚本与操作文档 | 当前状态 |
| --- | --- | --- |
| 离线版：统一服务仿真或独立 TX/RX 文件回环，不发射 RF | [`run_clean_demo.py`](scripts/platform_integration/clean_demo/run_clean_demo.py)、[`OPERATIONS.md`](scripts/platform_integration/clean_demo/OPERATIONS.md) | xwt 已完成 50 图块文件回环，50/50 CRC |
| 在线版：B210/SoapySDR duplex 发射接收 | [`run_online_demo.py`](scripts/platform_integration/clean_demo/run_online_demo.py)、[`ONLINE.md`](scripts/platform_integration/clean_demo/ONLINE.md) | 入口已同步 xwt、gxshen；xwt 参数 dry-run 通过，**尚未实机发射接收验证** |

在线版要求显式指定设备、RF 路径、频率、增益、天线、通道；线缆回环还要求记录衰减。默认只发送 1 个 burst，实际启动须加 `--acknowledge-rf`，输出独立写入 `captures/clean_deepjscc_online_<run-id>/`，不会覆盖离线结果。2026-09-28，本地 17 项 clean-demo 测试通过；xwt 的在线入口、共用 pipeline、demo 文档及合约测试已同步并核对哈希，14 项服务器测试通过，固定测试集、P0 checkpoint 与 OFDM 服务文件存在，无 RF dry-run 通过。gxshen 的在线代码、demo 文档和合约测试也已同步并核对哈希，14 项服务器测试通过。根 `README.md` 只在本地仓库维护，未覆盖服务器其他项目的根文档。B210 已通过只读设备探测，但物理接线与射频参数尚未复核，**没有进行实机发射接收**；不能把离线 50/50 CRC 当作在线指标。

2026-09-28 后续预检：xwt 检出 B210 序列号 `2410219`、USB 3、双工能力，TX 天线 `TX/RX`，RX 天线 `TX/RX`/`RX2`；没有发现占用 OFDM TX/RX 服务的进程。同步后的共用链路又完成 1 张固定 DIV2K 图块的非射频文件回环：CRC 1/1、64-bin CSI、均衡 latent、重建 PNG 和完整包波形均已保存，PSNR 38.626 dB；本地保存了[此次 manifest](results/div2k_lmdb_cv16_20260923_full/clean_postsync_file_preota_20260928_manifest.json)。首个真实实验选择 OTA，但频率、增益、天线端口和场地/授权尚待用户确认；截至此时**尚未发射 RF**。

同日用在线版实际采用的 `ofdm-workbench-service --simulate` 又跑通 1 张完整图块：[workbench manifest](results/div2k_lmdb_cv16_20260923_full/clean_workbench_theory_20260928_manifest.json) 记录 65,536 个 latent 样本原长返回、1261 个 OFDM 数据符号、CRC 1/1、64-bin CSI、MMSE 均衡和 38.626 dB 重建 PSNR。xwt 现有 `ctest` 全部通过，包含短包 AWGN、CFO、多径和分块流式 RX 测试；这些测试没有覆盖完整长包的真实空口。根目录 `RX.py` 未在本次同步中修改。在线 demo 现已把它作为**唯一** RX ZeroMQ 消费者：`RX.py` 将逐包 JSON、CSI、IQ 和批次 NPY 写入独立的 `rx/` 分区；Pipeline 校验后交 Decoder。[单 burst manifest](results/div2k_lmdb_cv16_20260923_full/clean_rxclient_sim1_20260928_manifest.json) 为 CRC 1/1、PSNR 38.626 dB；[双 burst manifest](results/div2k_lmdb_cv16_20260923_full/clean_rxclient_sim2_20260928_manifest.json) 为 CRC 2/2、平均 PSNR 39.296 dB，NPY 形状 `(2,65536)`。现有证据支持软件链路与 B210 双工接入**理论兼容**，尚不能证明 OTA 同步、接收功率/隔离、无丢块和实际重建质量。锚点 CRC 只校验控制信息，不保护原始 IQ payload。

核心接口：

- `IQDCOffsetConfig`、`apply_iq_dc_numpy()`、`apply_iq_dc_torch()`：`z'=μz+νz*` 后加入每样本常量 DC offset。
- `LearnableIQDCOffset`：tanh 有界参数、EVM/功率软约束。
- `AvailabilityBackdoorConfig`、`fit_availability_backdoor()`：从 P0 初始化后门模型；投毒样本只在 IQ/DC 触发分支上学习失效。
- `availability_backdoor_loss()`、`availability_four_way_metrics_np()`：不使用目标图像，报告干净/触发 PSNR、PSNR drop、MSE ratio 和失效率。

快速检查：

```powershell
python -m pytest -q
```

`availability_v2` 的权威入口是 `scripts/run_availability_pipeline.py`。它按 G0 → G1 → G2 → G3 门控执行，配置分别位于 `configs/availability_v2_pilot.toml` 和 `configs/availability_v2_full.toml`，产物写入 `runs/availability_v2/{pilot,dev,confirm}/`：

```powershell
python scripts/run_availability_pipeline.py --profile pilot --phase all --device cuda:0
python scripts/run_availability_pipeline.py --profile full-dev --phase all --device cuda:0
python scripts/run_availability_pipeline.py --profile full-confirm --phase all --device cuda:0
```

阶段失败会停止后续阶段；只有 `promotion_report.json` 为 `PASS` 才允许从开发筛选进入正式复现。新报告只使用源图像相对重建指标（PSNR drop、MSE ratio、失效率、EVM、功率变化和 PAPR/功率统计），不写入目标图像、target PSNR 或 ASR 字段。`--resume` 仅在代码/配置/数据/上游 checkpoint/种子身份完全一致时复用已有产物。

当前代码只保留 `availability_v2` 入口及其模块化的模型适配、信道触发、训练、扫描和指标实现；旧 token codec、目标图像/ASR 流程和旧版脚本已移除，不得用于新实验。

触发插入位置固定为：

```text
image -> encoder -> complex power normalization -> IQ imbalance -> DC offset -> propagation channel -> decoder
```

实验配置在 `configs/experiment_matrix.yaml`，文献与许可证记录在 `docs/` 和 `references/`。远端 RTX 4090 训练、checkpoint 和大日志不自动复制回本地；本地只接收配置、manifest、指标、图表和环境/产物哈希。
