# 离线版：正常 DeepJSCC → xwt TX/RX 操作手册

实机入口和射频预检单独见 [`ONLINE.md`](ONLINE.md)。本页所有运行命令均不发射 RF。
离线版仍执行同一套 Encoder、OFDM TX/RX 和 Decoder；只是用软件仿真或波形文件
回环代替 SDR 实际发射、物理信道和采样接收。与在线版共用 TX 的“JSON + cf32le
latent”两帧输入、RX 的“JSON + 64-bin cf32le CSI + 均衡后 cf32le latent”三帧
输出，以及相同的图块、ID 和阶段结果格式。文件模式额外导出的完整包波形仅用于
检查，不是另一种 Decoder 输入。

本手册只针对固定的 256×256 DIV2K 图块，不拼接整图。输入一次性加载；Encoder
按 batch 推理，每张图块独立发送一个 OFDM burst。默认是干净链路，不加载触发器或
后门模型。以下已验证的流程是 **TX 文件输出 → RX 文件输入**，不启用 B210 或发射 RF。

## 1. 文件与服务

| 位置 | 作用 |
| --- | --- |
| `scripts/platform_integration/clean_demo/run_clean_demo.py` | 推荐启动脚本；根据 run ID 创建新结果目录 |
| `scripts/platform_integration/clean_demo/{preprocess,encoder,tx,rx,pipeline,decoder}.py` | 六个隔离模块 |
| `build/ofdm-tx-service` | xwt TX：接收 JSON + CF32LE，输出完整时域包波形 |
| `build/ofdm-rx-service` | xwt RX：读取包波形，输出 JSON + 64-bin CSI + 均衡后 CF32LE |
| `models/div2k_lmdb_cv16_20260923/p0_clean.pt` | 干净 P0 模型，仅留在 xwt |
| `data/DIV2K/lmdb_cv16_test_uint8_20260923/test.npy` | 固定 LMDB 派生测试数据，50 张图块 |

xwt 项目根目录为 `/home/xwt/jscc-system-based-on-liquid`，demo 根目录为其下的
`demo_clean_deepjscc_20260923/`。gxshen 上有同版源码和 P0 模型来源，但没有
验证可用的 OFDM 服务；以下收发命令在 xwt 执行。

## 2. 运行前检查

```bash
cd /home/xwt/jscc-system-based-on-liquid
test -x build/ofdm-tx-service
test -x build/ofdm-rx-service
test -f data/DIV2K/lmdb_cv16_test_uint8_20260923/test.npy
test -f models/div2k_lmdb_cv16_20260923/p0_clean.pt
sha256sum data/DIV2K/lmdb_cv16_test_uint8_20260923/test.npy \
  models/div2k_lmdb_cv16_20260923/p0_clean.pt
```

预期 SHA-256：数据
`83f35a532a6a1240499262e208c7cacb055a94b1c67f65fe5a107a4e6ea05445`，
模型 `24070ef082a37ef5c4a232857daed760442b2fd3850ef617537e39fae5ff8542`。
使用 `/home/xwt/miniconda3/envs/jscc/bin/python`，不要随意切换到缺少 torch、numpy
或 pyzmq 的系统 Python。不要让另一个 `RX.py` 同时连接 demo 的 RX endpoint；
两个 PULL 客户端会分走结果。

## 3. 推荐脚本命令

以下命令从 xwt 项目根目录运行。`--run-id` 只允许字母、数字、`_`、`-`，同一个
run ID 不能复用；脚本会拒绝覆盖已有结果。默认 `file-services`、batch size 4、
只处理 1 张图块。

```bash
# 仅显示将使用的路径和命令，不启动服务或写结果
/home/xwt/miniconda3/envs/jscc/bin/python \
  demo_clean_deepjscc_20260923/scripts/platform_integration/clean_demo/run_clean_demo.py \
  --run-id inspect_001 --dry-run

# 1 张图块端到端预检
/home/xwt/miniconda3/envs/jscc/bin/python \
  demo_clean_deepjscc_20260923/scripts/platform_integration/clean_demo/run_clean_demo.py \
  --run-id smoke_001 --max-images 1

# 全部 50 张固定测试图块
/home/xwt/miniconda3/envs/jscc/bin/python \
  demo_clean_deepjscc_20260923/scripts/platform_integration/clean_demo/run_clean_demo.py \
  --run-id full_001 --max-images 0
```

这三个命令的 run ID 是示例；再次运行时换一个新 ID。结果目录自动命名为
`captures/clean_deepjscc_file-services_<run-id>/`。若只需原统一服务的数字仿真，
改用 `--transport workbench`；结果目录相应变为
`captures/clean_deepjscc_workbench_<run-id>/`，该模式不保存完整包波形。

## 4. 等价的手动命令

脚本本质上调用以下模块入口。手动执行时必须指定**不存在**的新输出目录：

```bash
cd /home/xwt/jscc-system-based-on-liquid/demo_clean_deepjscc_20260923
/home/xwt/miniconda3/envs/jscc/bin/python -m scripts.platform_integration.clean_demo \
  --dataset ../data/DIV2K/lmdb_cv16_test_uint8_20260923/test.npy \
  --checkpoint ../models/div2k_lmdb_cv16_20260923/p0_clean.pt \
  --transport file-services \
  --tx-service ../build/ofdm-tx-service \
  --rx-service ../build/ofdm-rx-service \
  --output-dir ../captures/clean_deepjscc_manual_NEW_RUN \
  --batch-size 4 --max-images 1
```

默认私有端口为 TX `tcp://127.0.0.1:5955`、RX `tcp://127.0.0.1:5956`。
如果端口被占用，同时给脚本传入新的 `--tx-endpoint` 和 `--rx-endpoint`；
一次只运行一个使用同一端口对的 demo。

## 5. 结果核验

以脚本的 `--run-id smoke_001` 为例：

```bash
cd /home/xwt/jscc-system-based-on-liquid
ls -lh captures/clean_deepjscc_file-services_smoke_001/tx/
ls -lh captures/clean_deepjscc_file-services_smoke_001/rx/
ls -lh captures/clean_deepjscc_file-services_smoke_001/decoder/
/home/xwt/miniconda3/envs/jscc/bin/python -c \
  'import json; p="captures/clean_deepjscc_file-services_smoke_001/manifest.json"; m=json.load(open(p)); print(m["status"], m["summary"], m["transport"])'
```

每张图块应有 `input/patch_XXXX.png`、`encoder/patch_XXXX.cf32le`、
`tx/burst_XXXXXX.cf32le`（频域载荷）、
`tx/burst_XXXXXX.waveform.cf32le`（完整时域 OFDM 包）、
`rx/burst_XXXXXX.cf32le`（均衡数据）、`rx/burst_XXXXXX.csi.cf32le`、
`decoder/patch_XXXX.png`。`manifest.json` 的 `status` 应为 `complete`，
`completed_patches` 与 `crc_ok` 应等于计划图块数。完整包波形不是 OTA 捕获。

已完成的独立 TX/RX 文件回环位于
`captures/clean_deepjscc_standalone_file_20260923_sim50/`：50/50 CRC、
50 个包波形、平均图块 PSNR 41.967 dB。这是文件回环指标，不是 B210 实测。
本手册的启动脚本也已在 xwt 用 `--run-id launch_smoke_20260923_01` 验证：
1 张图块、1/1 CRC、PSNR 38.626 dB；结果位于
`captures/clean_deepjscc_file-services_launch_smoke_20260923_01/`。

## 6. 故障与后续

- 缺少服务、数据或模型：启动脚本会列出缺失路径；先修复，不要在本地复制大模型。
- 输出目录已存在：使用新的 `--run-id`，不删除或覆盖已有实验。
- RX 超时：查看对应目录的 `service.log` 与 `manifest.json`；检查端口占用、
  `TX` 是否产出 `.waveform.cf32le`、`RX` 是否记录 `crc_ok`。
- CRC 失败或重建异常：先核对数据/模型 SHA-256、图块与 burst ID、RX 长度和 CSI。
- 真机 B210：本手册没有给出可直接照抄的发射命令；需先确认线缆/天线连接、
  合规频率、TX/RX 天线和增益，以及必要衰减器，再单独做硬件预检。

每次调整脚本、模型、数据、命令或实验结论时，同步更新本手册、demo README
与项目根 README，并在文档中注明哪些路径实际验证过。
