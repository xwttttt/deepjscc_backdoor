# 操作手册：以 xwt OTA 平台为基准

基准核对日期：2026-10-17。SSH 实读 `/home/xwt/jscc-system-based-on-liquid/README.md`，
原生提交 `896a6d9bdda905ced1f19689e8ad17539886136e`，分支 `ofdm-based`。
原文保存在 [xwt_ofdm_README.md](xwt_ofdm_README.md)，哈希在 `xwt_baseline.json`。
原文中的旧模型、DeepJSCC 入口和绝对路径仅作历史记录；部署使用本手册命令。

## 1. 三个分区和模块边界

| 分区 | 文件 | 输出/职责 |
| --- | --- | --- |
| 数据与模型训练 | `src/src/make_train_lmdb_dataset.py`、同名 `.sh`、`data_provider.py`；`scripts/*div2k*`；训练配置 | 原三脚本逻辑制作 image/token LMDB；正常训练、后门训练；不调用 SDR |
| OTA 平台 | `native/ofdm/`、`TX.py`、`RX.py` | Tx 组帧/IFFT/pilot；Rx 同步/CFO/CSI/MMSE/相位跟踪；业务 CF32 不做 QAM/FEC |
| 编解码与流程 | `encode.py`、`decode.py`、`scripts/steps.sh` | 独立生成 latent、调用 Tx、保存 Rx 数据、独立 Decoder；不修改物理层 |

数据只正向交接：

```text
源 HR → 原裁剪/Cosmos 脚本 → 固定 LMDB → 固定测试 NPY
 → Encoder → encoder/*.cf32le + manifest.json
 → TX.py → 原生 OFDM Tx → 仿真信道或 SDR 空口
 → 原生 OFDM Rx → RX.py → rx/{JSON,CSI,CF32,npy/}
 → 校验 → Decoder → reconstructed/{PNG,manifest.json}
```

一张图块一条记录；不把图块重新拼成大图。裁剪和 Cosmos 只在制作数据时执行；
收发时整份固定 NPY 加载一次，不动态读取源 PNG、随机裁剪或调用 Cosmos。
训练适配器从已固定的 LMDB 读取图块，不在训练时重新裁剪源图。

## 2. 两台主机准备

Tx 主机需要原生服务、`TX.py`、Encoder/模型、固定测试 NPY。
Rx 主机需要原生服务、**未经改写的 xwt `RX.py`**、Decoder 和同一模型。
最简单部署是两端均浅克隆本仓库；不用安装 GUI、Rust、Tauri。

```bash
python3 -m pip install -r requirements.txt
python3 tools/check_install.py
python3 tools/check_xwt_baseline.py
bash scripts/build_ofdm.sh
```

依赖安装和构建见根 README。xwt 当前 Conda Python 为 3.10，编解码可使用该环境；
训练与项目包安装使用 GPU 服务器 Python 3.11+。如果用原 xwt 服务：
给启动命令加 `--service /绝对路径/ofdm-workbench-service`，不用替换原仓库。

```bash
export PYTHON=/home/xwt/miniconda3/envs/jscc/bin/python
```

该变量只选解释器，不自动创建或安装环境。先安装 `requirements.txt`；
实查 xwt 原环境缺少 einops/lmdb，不能假定旧 Decoder 能跑就代表新 Encoder 的依赖齐全。

## 3. 先离线验证：完整数据流、无射频

```bash
bash scripts/run_offline_ofdm.sh captures/offline_001
```

流程经过原生服务 `--simulate`、`TX.py` 和 `RX.py`，不是只做 Encoder/Decoder 直连。
端口 5955/5956；结果分开保存在 `tx/`、`rx/`、`reconstructed/`，日志在本次目录。
校验 CRC、长度、CSI、NPY 与逐包数据的一致性及期望 burst ID。

已有服务：`OFDM_SERVICE=/path/to/ofdm-workbench-service bash scripts/run_offline_ofdm.sh captures/offline_002`。

## 4. 在线 OTA：分机分步骤

Tx/Rx 在不同主机各自连接本机 SDR，中心频率相同。两端 ZeroMQ 都只连各自的
`127.0.0.1`；跨机器传输的是空口，不是将 Rx 的 ZMQ 连到 Tx 主机。

先把 `configs/radio_tx.example.json` 和 `radio_rx.example.json` 复制到本地工作配置，
填入实际 `device`、许可的 `frequency_hz`、设备增益、通道和天线。模板中的空值
故意不作为可运行射频参数；不会沿用服务器 README 的示例序列号或频率。
设备参数来自 `build/ofdm/ofdm-device-probe --list`。
工作配置可放 `captures/config/`，不纳入 Git。

### 步骤 1：Tx 生成并保存频域 latent（不发射）

```bash
bash scripts/steps.sh encode captures/tx_001 --max-images 1 --batch-size 4
```

默认 P0 和仓库固定测试集；`--max-images 0` 全部 50 图。
结果为 `captures/tx_001/encoder/burst_00000001.cf32le`、`manifest.json`。
一图包含 65,536 个复数值，不是已经 IFFT 后的时域无线包。

### 步骤 2：Rx 主机启动硬件接收服务和落盘程序

终端 A：

```bash
bash scripts/steps.sh rx-service --radio-config captures/config/radio_rx.json --dry-run
# 核对后执行，同一命令去掉 --dry-run：
bash scripts/steps.sh rx-service --radio-config captures/config/radio_rx.json
```

终端 B（必须先于 Tx 发包启动）：

```bash
bash scripts/steps.sh receive captures/rx_001 --max-bursts 1
```

收 50 图改为 `--max-bursts 50`。丢包时不会自动结束，可 Ctrl-C 停止并保存剩余批次；
`--timeout-ms` 只是单次轮询时间，不是整次实验超时。
一个服务只运行一个 `RX.py` 客户端，避免 PUSH 将记录分发给多个客户端。

### 步骤 3：Tx 主机启动硬件发射服务，再提交已保存 latent

终端 A：

```bash
bash scripts/steps.sh tx-service --radio-config captures/config/radio_tx.json --dry-run
bash scripts/steps.sh tx-service --radio-config captures/config/radio_tx.json
```

终端 B：

```bash
# 只校验文件和哈希，不提交：
bash scripts/steps.sh send captures/tx_001/manifest.json --dry-run
# 实际提交给 TX.py；硬件服务收到数据后立即发射：
bash scripts/steps.sh send captures/tx_001/manifest.json --interval 0.25
```

也可以只发一个文件：

```bash
python3 TX.py --input-bin captures/tx_001/encoder/burst_00000001.cf32le --start-id 1 --count 1
```

提交成功仅代表送入本机 ZeroMQ，不是空口送达确认。发送入口不额外缩放或归一化 latent。
原服务器诊断用法仍支持：`python3 TX.py --samples 1024 --count 10 --interval 0.25`；
不要把这种测试符号送进图像 Decoder。

### 步骤 4：Rx 停止接收并解码

```bash
bash scripts/steps.sh decode captures/rx_001 captures/reconstructed_001 --device cpu
```

`steps.sh` 先运行只读校验，失败则不调用 Decoder。若把 Tx 的 `manifest.json`
复制到 Rx，可额外核对完整 ID 集：

```bash
TX_MANIFEST=/path/to/copied_tx_manifest.json \
  bash scripts/steps.sh decode captures/rx_001 captures/reconstructed_checked --device cpu
```

原 Rx 每 100 条写一批 NPY，Ctrl-C/达到 max-bursts 时补存不足 100 条。
NPY 仅保留接收顺序，没有每行 ID；逐包 JSON/CF32 保留 burst ID。
该校验不重排 NPY，也不能将空口丢包自动补齐或去重。重发应使用新输出目录，
同次测试的 burst ID 不重复。控制 anchor 的 `crc_ok` 不是业务 IQ 的逐字节校验。

## 5. 指定后门模型：其余流程不变

```bash
CHECKPOINT=models/stronger_struct_best.pt \
  bash scripts/steps.sh encode captures/triggered_tx_001 --max-images 1 \
  --trigger models/stronger_struct_trigger.json --start-id 101

# Rx 解码也选择相同 checkpoint；Rx 不需要 trigger.json：
CHECKPOINT=models/stronger_struct_best.pt \
  bash scripts/steps.sh decode captures/triggered_rx_001 captures/triggered_png_001 --device cpu
```

触发仅发生在 Encoder 功率归一化之后、OFDM 之前，与训练的 `apply_iq_dc_numpy`
相同。未触发对照也用 stronger 模型，但不传 `--trigger`。
比较 P0、后门 clean、后门 triggered 三个分支时，使用同一数据和明确的新目录/ID。
本次整理不改模型、损失、训练目标或 checkpoint 选择逻辑。

## 6. 离线与在线的唯一链路差异

使用相同的 `encode.py → TX.py → RX.py → decode.py`。
离线服务使用 `--simulate`，在线服务使用 `--device ... --role tx/rx`。
两种模式的 JSON、CF32LE、64-bin CSI、NPY、checkpoint 和落盘目录格式相同；
在线会引入真实噪声、同步失败、丢包和接收顺序变化，不能假设数值或记录数量相同。
原生空口是 legacy BPSK 1/2 控制 anchor + 自定义 raw-IQ 区域；
业务符号不做 BPSK/QAM/FEC，但不能说整个无线包完全不含 BPSK/编码。

## 7. 训练和数据集

完整步骤、三脚本位置、原始 HR 来源、固定 LMDB 和测试导出见 [DATASET.md](DATASET.md)。
训练代码只在 GPU 服务器运行；部署机器不需要训练数据或 Cosmos 权重。
所有实验只更新本次新目录，不清理原服务器的 captures/models 或未提交文件。
