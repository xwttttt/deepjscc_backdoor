# DeepJSCC / liquid-dsp OFDM

GitHub 默认分支为 `backdoor`（原 `codex/project-upload-20261016`）；旧分支已改名为
`simulation`（原 `codex/deepjscc-iq-dc-offset`）。本次仅改分支名及文档引用，不改变模型或收发逻辑。

本分支按 **xwt 服务器实际 OTA README 和接口** 整理，只保留当前链路、P0 与指定的 stronger_struct 模型、固定测试集、训练代码和必要测试。Python 入口为 `encode.py`、`TX.py`、`RX.py`、`decode.py`。旧实验、报告、重复包已从分支文件清单移除，历史提交仍可回溯。

## 操作入口与代码分区

- [完整操作手册](docs/WORKFLOW.md)：分机 Tx/Rx，预处理→编码→发送→接收→校验→解码，以及离线/在线切换。
- [数据与两类训练](docs/DATASET.md)：用户三个脚本、源 DIV2K、固定 LMDB、正常/后门训练和固定测试导出。
- [xwt 实际 README 快照](docs/xwt_ofdm_README.md)：2026-10-17 SSH 实读；基准哈希见 `docs/xwt_baseline.json`。
- 分步骤入口：`bash scripts/steps.sh encode|tx-service|send|rx-service|receive|decode ...`。

`native/ofdm/` 管物理层，根 `TX.py/RX.py` 管数据接口，`encode.py/decode.py` 管模型，
`scripts/` 管数据制作、训练与薄流程包装。模型、数据和各次 captures 相互分离。
原始 HR 和完整训练 LMDB 保留在服务器，**没有作为大数据上传到 Git**；Git 包含固定测试集、来源清单和制作/加载入口。

```text
固定 test.npy → Encoder → complex64 latent → TX.py → OFDM 服务
                  → 仿真或 SDR 信道 → RX.py → 均衡 latent NPY → Decoder → PNG
```

每张 256×256 图块对应 65,536 个复数符号，独立收发和重建。业务符号直接进入 OFDM，不再进行 BPSK/QAM 或业务 FEC；同步/CSI/均衡由原生 OFDM 服务完成，`RX.py` 负责接收与落盘。

## 1. 安装与已有文件

训练/项目包安装用 Python 3.11+；编解码兼容 xwt 的 Python 3.10。Ubuntu 用于原生 OFDM/SDR。从仓库根目录执行：

```bash
git clone --depth 1 --single-branch --branch backdoor \
  https://github.com/xwttttt/deepjscc_backdoor.git
cd deepjscc_backdoor
python3 -m pip install -r requirements.txt
python3 tools/check_install.py
python3 tools/check_xwt_baseline.py
```

CUDA 训练环境请预先安装与服务器 CUDA 匹配的 PyTorch。短时 CPU 验证可用 CPU PyTorch。仓库中已直接包含：

| 路径 | 内容 |
| --- | --- |
| `data/div2k_cv16_test/test.npy` | 已处理的 50 张测试图块，`uint8 [50,3,256,256]` |
| `data/div2k_cv16_test/manifest.json` | Cosmos CV16 valid LMDB 键 50–99 的来源与哈希 |
| `models/p0_clean.pt` | 正常链路干净 P0，SHA-256 `24070ef0…ff8542` |
| `models/stronger_struct_best.pt` | 用户指定的 2026-09-28 stronger_struct 运行内最佳模型，SHA-256 `db7fcc8e…5d166c` |
| `models/stronger_struct_trigger.json` | 该模型对应的 IQ/DC 参数 |
| `models/stronger_struct_run_manifest.json`、`stronger_struct_metrics.json` | 模型来源和指标 |
| `native/ofdm/` | 固定版本的原生 OFDM 核心源码与测试；来源见 `UPSTREAM.json` |

`test.npy` 已完成原图片裁剪和 Cosmos 编解码。推理时一次加载固定数组，仅按模型训练约定换算到 `[0,1]`。不需要重新制作数据集。

## 2. 先生成 latent

```bash
python3 encode.py \
  --dataset data/div2k_cv16_test/test.npy \
  --checkpoint models/p0_clean.pt \
  --output-dir captures/tx_001 --max-images 1 --batch-size 4
```

结果是 `captures/tx_001/encoder/burst_00000001.cf32le` 和 `manifest.json`。CF32LE 按小端 `I0,Q0,I1,Q1,…` 存储；它是频域 latent。默认仅保存，`--max-images 0` 处理全部 50 图。每次使用新的输出目录。

## 3. 编译并跑无 RF OFDM 回环

依赖已安装 liquid-dsp、libzmq、SoapySDR、CMake 3.20+、C++17 编译器。Ubuntu 可安装：

```bash
sudo apt-get install build-essential cmake pkg-config libliquid-dev libzmq3-dev libsoapysdr-dev
bash scripts/build_ofdm.sh
bash scripts/run_offline_ofdm.sh captures/offline_001
```

原生构建会下载固定版本的 nlohmann/json，首次配置需要联网。构建产物在 `build/ofdm/`，包括 `ofdm-workbench-service`、`ofdm-tx-service`、`ofdm-rx-service`、`ofdm-device-probe`。运行脚本使用独立端口 5955/5956，启动服务的 `--simulate` 模式，经原 `TX.py`/`RX.py` 传递一图 latent，检查控制 CRC 和返回长度，再解码。结果在 `captures/offline_001/reconstructed/`，服务/接收日志及 manifest 按目录保存。

若 GitHub 下载不可用，可提前解压经过哈希校验的 JSON 3.11.3 源码，再执行
`bash scripts/build_ofdm.sh -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=/absolute/path/to/json`。
官方 `json.tar.xz` 的 SHA-256 为 `d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d`。

已有原生服务时可用 `OFDM_SERVICE=/absolute/path/to/ofdm-workbench-service` 指定，Python 可用 `PYTHON=/path/to/python` 指定。

## 4. 分机 Tx 与 Rx

两台主机各放同一份本分支代码和模型，启动自己的 OFDM 服务。以下示例使用本机客户端 endpoint 5555/5556；`DEVICE_ARGS`、`FREQUENCY_HZ`、增益和天线由设备实验配置提供。

Tx 主机终端 A：

```bash
build/ofdm/ofdm-workbench-service --role tx \
  --device "$DEVICE_ARGS" --frequency "$FREQUENCY_HZ" \
  --tx-gain "$TX_GAIN" --tx-antenna "$TX_ANTENNA" \
  --tx-endpoint tcp://127.0.0.1:5555
```

Tx 主机终端 B，显式发送已保存的数据文件：

```bash
python3 TX.py --input-bin captures/tx_001/encoder/burst_00000001.cf32le \
  --start-id 1 --count 1 --endpoint tcp://127.0.0.1:5555
```

也可在第 2 节编码命令中加 `--send --tx-script ./TX.py --endpoint tcp://127.0.0.1:5555`，按图块调用 `TX.py`。服务处于硬件模式时，提交数据会触发实际发射。

Rx 主机终端 A：

```bash
build/ofdm/ofdm-workbench-service --role rx \
  --device "$DEVICE_ARGS" --frequency "$FREQUENCY_HZ" \
  --rx-gain "$RX_GAIN" --rx-antenna "$RX_ANTENNA" \
  --rx-endpoint tcp://127.0.0.1:5556
```

Rx 主机终端 B，先启动接收，再让 Tx 发包：

```bash
python3 RX.py --endpoint tcp://127.0.0.1:5556 --max-bursts 1 \
  --save-dir captures/rx_001 --npy-dir captures/rx_001/npy
python3 decode.py captures/rx_001/npy --checkpoint models/p0_clean.pt \
  --output-dir captures/reconstructed_001 --max-records 1 --device cpu
```

`RX.py` 保持原接收逻辑：三帧 JSON/64-bin CSI/均衡 latent，默认每 100 条保存 NPY，退出时保存剩余条目。仅运行一个接收客户端。先核对 JSON 中 `crc_ok=true`、`original_length=65536` 和 burst ID，再把重建计作有效结果；控制 CRC 不保证业务 IQ 每个字节正确。批次 NPY 保留接收顺序，不含逐条 burst ID，丢包时需对照 JSON。Encoder 和 Decoder 必须使用同一个 checkpoint。

## 5. 指定 stronger_struct 模型与触发

使用该模型的未触发分支时，将编码与解码两端的 checkpoint 都换为 `models/stronger_struct_best.pt`。触发分支在编码命令中额外指定：

```bash
python3 encode.py --dataset data/div2k_cv16_test/test.npy \
  --checkpoint models/stronger_struct_best.pt \
  --trigger models/stronger_struct_trigger.json \
  --output-dir captures/triggered_001 --max-images 1 --start-id 101
```

触发参数通过训练代码相同的 `apply_iq_dc_numpy` 施加在功率归一化 latent 上，再写 CF32LE；manifest 记录触发 JSON 哈希。Rx 不需要 trigger JSON。该分支用 `--checkpoint models/stronger_struct_best.pt` 解码。无 RF 对照可运行：

```bash
CHECKPOINT=models/stronger_struct_best.pt TRIGGER=models/stronger_struct_trigger.json \
  bash scripts/run_offline_ofdm.sh captures/offline_triggered_001
```

该模型是运行内最佳 checkpoint；其历史验证 clean drop 0.710 dB 高于 0.5 dB 门槛，仍属候选。历史测试 triggered drop 28.165 dB 来自直接模型推理，不是 OTA 指标。

## 6. 训练

训练在专用 GPU 服务器执行。需额外准备完整 Cosmos CV16 LMDB：训练 3200 条、valid 100 条；valid 键 0–49 验证、50–99 测试。仓库中的 50 条测试导出仅供推理，不用于重训。

配置默认读取 `data/DIV2K/lmdb/train_256_CV16_float32.lmdb` 和 `valid_256_CV16_float32.lmdb`。可在服务器将已有库链接到该位置，或修改配置中的绝对路径。

```bash
# 从已提供 P0 继续训练 stronger_struct；输出目录必须不存在。
PYTHON=/path/to/gpu/python bash scripts/train_stronger.sh runs/stronger_new

# 可选：从已提供 P0 继续干净微调，不宣称重新复现原始 P0。
python3 scripts/run_div2k_clean_p0.py --init-checkpoint models/p0_clean.pt \
  --output-dir runs/p0_finetune_new --steps 5000 --batch-size 4 --mixed-precision
```

当前训练脚本为 `scripts/run_div2k_two_drop.py`，配置为 `configs/div2k_two_drop_struct_lmdb_stronger_20260928.toml`。保留该模型原来的 clean drop 和局部结构 triggered drop 目标与选模规则。输出 `best.pt`、`trigger.json`、`metrics.json` 等由训练脚本保存。

用户提供的三个数据制作/读取脚本保存在 `src/src/make_train_lmdb_dataset.py`、`.sh` 和 `data_provider.py`；重新制作 LMDB 需要原 CTJSCC 的 `lib.cosmos_tokenizer` 代码及权重，并在对应工程中按原脚本环境运行。已有 LMDB 导出固定测试集使用 `scripts/export_div2k_lmdb_test.py --valid-lmdb /path/to/valid.lmdb --output-dir /path/to/new_export`。

## 7. 验证状态

```bash
python3 -m pip install pytest
python3 -m pytest -q
```

2026-10-16 已使用精简文件树在专用服务器的独立临时目录完成以下检查，均使用 CPU：

- 29 项 Python 测试全部通过，覆盖真实 checkpoint 编码、训练函数与推理触发的一致性、Decoder 和 `max-images=1` 的 batch 边界。
- 模型、触发参数和固定测试集通过 `tools/check_install.py` 的 SHA-256 检查。
- 原生程序成功构建，CTest 1/1 通过。验证环境原先缺少 liquid-dsp，测试时仅在临时目录解包 Ubuntu 的 `libliquid-dev`/`libliquid2d`；实际部署按第 3 节安装系统依赖。
- P0 与 stronger_struct 触发分支各完成一图 OFDM `--simulate` 回环：控制 CRC 通过、原长返回 65,536 个复数值，并生成重建 PNG。
- P0 入口完成 1-step 冒烟；后门入口完成单 batch 冒烟。另用临时全投毒测试配置确认 2 个投毒样本覆盖梯度路径。交付配置和模型文件未被这些测试改写。

以上短时测试证明当前入口与依赖可运行，不是新一轮完整训练的效果报告。本次未做实机 RF/OTA 发射接收验证。

### 2026-10-17：按 xwt 实际平台整理后的验证

实读服务器 README 并核对原生提交；37 个原生文件、`RX.py` 和 README 快照共 39 文件
通过基准哈希检查。`RX.py` 保持与 xwt 逐字节一致，Tx 的原诊断符号也逐字节匹配。
Tx 仅增加固定 latent 文件兼容入口，模型、损失、原生组帧/同步/CSI/均衡均未改写。

在 xwt 的独立临时目录、隔离 Python 环境完成：43 项 Python 测试通过、无跳过；
原生构建成功、CTest 1/1；P0 与 stronger 触发模型各一图 OFDM 仿真成功，均通过 CRC/长度校验并生成 PNG；
另外按 `steps.sh` 手册完成先编码保存、独立 Tx 提交、原 Rx 落盘、校验、独立解码。
详情见 [验证记录](docs/verification_20261017.json)。

本地 Windows 测试受临时目录权限限制，因此完整动态验证在 xwt 执行；GPU 全程禁用。
首次 JSON/PyPI 直连下载失败，测试改用已校验的离线依赖；没有更改原 xwt 项目或原 Conda 环境。
本次没有新训练、数据集再生成、全量评估或真实 OTA 发射。在线入口已对齐原服务接口，
但这不等于本整理版已完成空口实测。
