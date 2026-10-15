# 在线版（B210 实机 TX/RX）操作预案

**状态：在线入口已同步到 xwt 和 gxshen；截至 2026-09-28，未在 B210 上完成
发射接收验证。** 2026-09-23 曾在 xwt 枚举到 B210，但设备当前状态尚未复核。
两机的在线代码、demo 文档与合约测试已同步并核对哈希，各 14 项模块测试通过。
xwt 已确认固定测试集、
P0 checkpoint、OFDM 服务文件存在，在线入口的无 RF dry-run 通过；
这不代表设备、接线或真实信道已验证。本页不是已完成的 OTA 实验报告。

2026-09-28 只读设备预检在 xwt 检出 B210 序列号 `2410219`，USB 3、双工可用；
TX 天线为 `TX/RX`，RX 天线为 `TX/RX` 或 `RX2`，并未发现正在运行的 OFDM
TX/RX 服务进程。同日 1 张图块的非射频文件回环 CRC 1/1、PSNR 38.626 dB，
manifest 在 xwt 的 `captures/clean_deepjscc_file-services_sync_20260928_preota_001/manifest.json`，
本地仓库另存一份于 `results/div2k_lmdb_cv16_20260923_full/clean_postsync_file_preota_20260928_manifest.json`。
首个真实 burst 计划走 OTA；用户尚未提供并确认频率、增益、天线连接及场地授权，
因此截至本次更新没有启动任何 RF 发射。

理论链路核对：同一 `ofdm-workbench-service` 的 `--simulate` 路径已完整跑通
65,536 个 latent 样本、1261 个 OFDM 数据符号、64-bin CSI、MMSE 均衡和 Decoder；
1 张图块 CRC 通过、PSNR 38.626 dB，本地保存了相应 workbench manifest。
xwt 现有 OFDM 单测中的 AWGN、CFO、多径和流式 RX 用例也通过，但样本长度远小于
这个完整图块，不能代替 OTA 测试。根目录 `RX.py` 未在本次同步中改动；在线入口
现在将它作为**唯一 RX ZeroMQ 消费者**启动，`clean_demo/rx.py` 只读取并验证
其保存的结果，不再打开第二个 PULL socket。非射频集成测试中，单 burst CRC 1/1、
PSNR 38.626 dB；双 burst CRC 2/2、平均 PSNR 39.296 dB，`RX.py` 输出
`(2,65536)` 的 `complex64` NPY 和逐 burst 文件。当前 `crc_ok` 是锚点控制信息 CRC，不等于 65,536 个
原始 IQ payload 都无误；首包仍需用长度、CSI、重建质量和服务日志判断。

在线启动脚本默认依赖 xwt 项目根目录的原 `RX.py`，并在真正启动前检查其存在；
它传入 `--rx-client` 给 Pipeline。`RX.py` 的 `--save-dir` 与 `--npy-dir`
只指向本次新建的 `captures/clean_deepjscc_online_<run-id>/rx/` 分区。
`--max-bursts` 等于计划图块数，因此运行结束时即使不足 100 条也会保存最后一批
NPY。Pipeline 按 burst ID 读取逐包 JSON、64-bin CSI 与 `.data.cf32le`，校验
CRC、长度和有限值后交给 Decoder；原 `RX.py` 不被修改，也不能另起第二个实例
争用相同 RX endpoint。

在线版与[离线版](OPERATIONS.md)共用固定 DIV2K 图块、干净 P0 模型、Encoder、
Decoder 和分阶段 manifest，也共用 `tx.send_burst()`、`rx.receive_burst()` 的
ZeroMQ 协议：TX 是 JSON + `cf32le` latent 两帧，RX 是 JSON + 64-bin
`cf32le` CSI + 均衡后 `cf32le` latent 三帧。离线版同样运行 OFDM TX/RX 算法；
两者仅在是否实际经 SDR 发射、物理信道和 SDR 接收上不同，收到的数据值和 CRC
结果可以不同，但格式不变。在线版由
`run_online_demo.py` 启动 `ofdm-workbench-service --device ... --role duplex`，
经 SoapySDR 同时发射与接收。它不使用离线版的 `--simulate` 或 `--file-in`。

## 运行前必须确认

1. 在 xwt 主机确认 B210 当前可用，且没有 GUI 或其他进程占用设备和端口。
2. 明确使用**带合适衰减器的线缆回环**还是**天线 OTA**；软件无法验证物理接线。
3. 明确合规频率、TX/RX 天线、通道和增益；不得照搬以前的默认值。
4. 先只发 1 张固定图块，观察接收、CRC、CSI、重建以及服务日志。
5. 线缆连接应由操作者确认衰减器和接收端功率安全；OTA 应由操作者确认频段、
   场地、授权和射频安全。`--attenuation-db` 只是记录所确认的衰减，不是自动测量。

只读检查示例：

```bash
cd /home/xwt/jscc-system-based-on-liquid
build/ofdm-device-probe --list
SoapySDRUtil --find
test -x build/ofdm-workbench-service
```

按当前实际设备输出，另行使用 `build/ofdm-device-probe --device 'driver=uhd,serial=...'`
核对天线、通道、频率和增益能力。该命令可能打开设备；应在设备空闲时执行。

## 在线入口（模板，不能原样发射）

从 xwt 项目根目录运行。首包计划 OTA；先将 `SERIAL`、`FREQ_HZ`、增益、天线
与通道替换成实测、获准并现场确认的值。如下模板不能原样发射。若改为线缆回环，
应改用 `--rf-path cable` 并明确加入经现场确认的 `--attenuation-db`。

```bash
/home/xwt/miniconda3/envs/jscc/bin/python \
  demo_clean_deepjscc_20260923/scripts/platform_integration/clean_demo/run_online_demo.py \
  --run-id rf_smoke_001 --rf-path ota \
  --device 'driver=uhd,serial=SERIAL' --frequency FREQ_HZ \
  --tx-gain TX_GAIN_DB --rx-gain RX_GAIN_DB \
  --tx-antenna TX_ANTENNA --rx-antenna RX_ANTENNA \
  --tx-channel TX_CHANNEL --rx-channel RX_CHANNEL --dry-run
```

`--dry-run` 只打印解析出的命令和目标路径，不启动 SDR。在物理和合规预检完成后，
使用**新的** run ID，去掉 `--dry-run` 并显式加 `--acknowledge-rf`，才允许启动。
脚本默认只发 1 个 burst；多图块还须显式加 `--allow-multi-burst` 并设置
`--max-images N`（0 表示完整固定集）。禁止复用既有 run ID。

在线结果目录独立命名为 `captures/clean_deepjscc_online_<run-id>/`。其中 `tx/`
是进入服务的频域 CF32LE 载荷；当前统一服务**不会**导出实际空口发射波形。
`rx/` 由原 `RX.py` 写入逐包 JSON、均衡后的 `.data.cf32le`、64-bin CSI、
批次 NPY 和 `client.log`；Pipeline 只读取、校验和登记这些产物，不在 RX 端口
建立第二个消费者。`decoder/` 保存图块重建图，根目录
`manifest.json` 记录实际指定的设备、频率、天线、增益、RF 路径和 CRC。

## 达标条件与回退

首个在线 burst 需同时满足：服务正常启动，TX/RX 均有记录，RX CRC 为真，CSI
和均衡数据长度正确且有限，Decoder 生成 256×256 PNG，manifest 状态为
`complete`，服务日志无设备错误或持续 overflow。仅设备能被枚举不算链路跑通。

若失败，不扩大到多图块；保留当前目录和日志，排查设备占用、接线/衰减、
天线与通道、频率/增益、USB 吞吐、TX/RX 端点以及同步/CRC。不要改写离线版
已验证的 manifest。当前离线版 50/50 CRC 仅是软件/文件回环基线，不是在线指标。
