# DIV2K DeepJSCC → OFDM 分机收发操作手册

本手册是交付包的**唯一操作入口**。它覆盖图片预处理、固定数据导出、Encoder 接 `TX.py`、`RX.py` 接 Decoder；不使用 `clean_demo` 的独立 ZeroMQ 收发器。每个 256×256 图块独立发送和重建，不拼接整图。

## 1. 边界与文件

```text
DIV2K HR 图片
  → 用户提供的 make_train_lmdb_dataset.py/.sh + data_provider.py 所定义的
    裁剪、Cosmos CV16 encode/decode、image/token LMDB（离线制作）
  → export_div2k_lmdb_test.py：valid LMDB 键 50–99 → 固定 test.npy + manifest.json
  → encode_prepared_npy.py：一次加载 uint8 [N,3,256,256]，按 [0,1] 输入 Encoder
  → burst_XXXXXXXX.cf32le：每图块一个功率归一化 complex64 频域 latent
  → TX.py --input-bin → OFDM 服务 → 无线或仿真信道
  → RX.py：JSON + 64-bin CSI + 均衡 cf32le + 批次 NPY
  → decode_rx_npy.py：匹配 checkpoint 的 Decoder → RGB PNG
```

`TX.py` 和 `RX.py` 是仅有的无线接口程序。Encoder/Decoder 的 PyTorch 模型代码属于必要的推理依赖，不另起发送或接收器。业务 latent 是直接传给 OFDM 数据子载波的复数符号，不做 BPSK/QAM 或业务 FEC；控制 anchor 属于平台协议。

交付包目录：

```text
README.md                         本手册
TX.py                             支持 --input-bin 的文件发送客户端
RX.py                             既有接收保存客户端，未改逻辑
scripts/
  make_train_lmdb_dataset.py      用户给的原预处理程序
  make_train_lmdb_dataset.sh      用户给的原启动脚本
  data_provider.py                用户给的原 LMDB loader，仅作为制作逻辑来源
  div2k_lmdb_dataset.py           已有 LMDB 的只读适配
  export_div2k_lmdb_test.py       固定 holdout 导出
src/src/
  encode_prepared_npy.py          Encoder → 独立 latent 文件 → 可选调用 TX.py
  decode_rx_npy.py                RX.py 批次 NPY → Decoder PNG
  channels.py, deepjscc/model.py  原 DeepJSCC 模型定义
models/p0_clean.pt                固定 Cosmos-LMDB 实验最佳干净 P0 checkpoint
models/stronger_struct_best.pt    2026-09-28 stronger_struct 运行内最佳后门 checkpoint
models/stronger_struct_trigger.json  该后门模型配套触发参数
models/stronger_struct_{run_manifest,metrics}.json  来源与指标
data/div2k_cv16_test/test.npy     固定 50 图块测试导出，uint8 [50,3,256,256]
data/div2k_cv16_test/manifest.json  LMDB 来源、键范围与数据哈希
FILE_SHA256.json                  包内文件哈希
```

包内**不含**原始 DIV2K HR 图片、完整 LMDB、Cosmos tokenizer 权重、SDR 驱动或原生 OFDM 服务；已包含固定测试集 `data/div2k_cv16_test/test.npy` 和来源 manifest。另含两份分开命名的 checkpoint：干净 P0 `models/p0_clean.pt`（SHA-256 `24070ef082a37ef5c4a232857daed760442b2fd3850ef617537e39fae5ff8542`）及用户指定的 stronger_struct 运行内最佳模型 `models/stronger_struct_best.pt`（SHA-256 `db7fcc8e4bacb743ae780ffdb1a0516319b82d96f5a90893e5ac64182e5d166c`）。后者配套 `models/stronger_struct_trigger.json`（SHA-256 `e47797d06a970f41f22949a3e36fb91020bf6bc5ac045e7663de89bd280c3eb8`）。不要用包内文件盲目覆盖服务器已有文件，先核对各自版本与备份。

本手册第 3–5 节的命令**默认走正常 P0 链路**。要做后门模型的未触发分支，可在 Encoder 和 Decoder 两端都把 checkpoint 换成 `models/stronger_struct_best.pt`；不能只换一端。当前 `encode_prepared_npy.py` **尚未施加 trigger.json 中的 IQ/DC 触发**，因此仅把后门 checkpoint 加入包内，不等于已实现 triggered 分支，也不能把现有命令的重建结果称作 triggered drop。该 stronger_struct 模型的验证集 clean drop 为 0.710 dB，超过当次 0.5 dB 门槛，仍是视觉失真候选而非正式替代版；其 28.165 dB 测试 triggered drop 来自直接模型推理，尚未经过 OFDM/实机。

## 2. 依赖与数据准备

收发/推理主机：Python 3、`numpy`、`torch`、`einops`、`Pillow`、`pyzmq`；必须另行构建并启动 `jscc-system-based-on-liquid` 的 OFDM 服务。预处理/导出主机另需 `lmdb`。完整**重新制作** LMDB 还需要用户原 CTJSCC 工程的 `lib.cosmos_tokenizer`、`src.utils`、Cosmos `encoder.jit`/`decoder.jit`、`torchvision`、`loguru`、`tqdm` 等；这些外部组件不在交付包中。原 `.sh` 写死 conda `sionna` 环境和 `~/datasets`，运行前按实际 CTJSCC 安装位置调整。不要在收发运行时动态裁剪原图或重新运行 tokenizer。

若已有固定 CV16 LMDB，**不要重建或清空**。已使用的数据身份是 train LMDB 3200 条、valid LMDB 100 条；valid 键 0–49 用于验证，50–99 用于测试。包内测试导出 `test.npy` 的 SHA-256 为 `83f35a532a6a1240499262e208c7cacb055a94b1c67f65fe5a107a4e6ea05445`，核对形状 `(50,3,256,256)`、`uint8`。解包到 Tx 主机后，该 `test.npy` 和同目录的 `manifest.json` 已可直接供编码入口读取。

只有在**新建**数据集且 CTJSCC/Cosmos tokenizer 环境、模型权重和 DIV2K 原图都已就绪时，才把包内三个原脚本放到 CTJSCC 工程相应位置，从该工程根目录按原参数运行。示例（输出目录须是新的，先核对实际输入路径与 tokenizer checkpoint）：

```bash
cd /path/to/CTJSCC
python3 scripts/make_train_lmdb_dataset.py \
  --tokenizer_type CV --spatial_compression 16 \
  --dataset_root /path/to/datasets \
  --output_dir /path/to/new_DIV2K_lmdb \
  --image_size 256 --num_crops 4
```

原 `make_train_lmdb_dataset.sh` 是该命令在作者 `sionna` conda 环境中的启动模板；`data_provider.py` 是 LMDB 读取参考，DeepJSCC 仍按下一步的固定导出与 `[0,1]` RGB 输入。不能把 CTJSCC 的外部库路径误认为交付包自带。

如果只有 LMDB 而没有测试导出，在拥有 `torch`、`numpy`、`lmdb` 的预处理主机执行一次：

```bash
cd /path/to/div2k_txrx_bundle
python3 scripts/export_div2k_lmdb_test.py \
  --valid-lmdb /path/to/valid_256_CV16_float32.lmdb \
  --output-dir data/new_div2k_cv16_test
```

导出器只读 valid LMDB、固定取键 50–99，按上例写入 `data/new_div2k_cv16_test/test.npy` 和来源/哈希 `manifest.json`。新建输出目录，不能覆盖包内旧导出。DeepJSCC 训练使用 `[0,1]` RGB，故这里不使用 `data_provider.py` 为另一模型准备的 `[-1,1]` Normalize。旧原图裁剪 NPY 不能替代该 Cosmos-LMDB 导出。

## 3. 先做无 RF 检查

在 Tx 主机的包根目录先只编码 1 图，不加 `--send`：

```bash
cd /path/to/div2k_txrx_bundle
python3 src/src/encode_prepared_npy.py \
  --dataset data/div2k_cv16_test/test.npy \
  --checkpoint models/p0_clean.pt \
  --output-dir captures/tx_precheck_001 \
  --max-images 1 --batch-size 1 --device cpu
```

预期 `captures/tx_precheck_001/encoder/burst_00000001.cf32le` 含 c=16、256×256 模型的 **65,536 个 complex64** 样本，即 524,288 字节；`manifest.json` 应为 `status=complete`、`latent_samples=65536`、`tx_invoked=false`，并记录固定数据及 checkpoint 哈希。该文件还不是完整时域 OFDM 包。`--max-images 0` 表示全部 50 图；每次使用新的输出目录。

测试 `TX.py` 文件模式 CLI，确认有 `--input-bin` 与 `--start-id`：

```bash
python3 TX.py --help
```

上游 Git 仓库 `ofdm-based` 分支的原始 `TX.py` 只生成 fake 样本，**没有** `--input-bin`；交付包中的 `TX.py` 增加了文件模式。文件模式不调用图片裁剪或 `deepjscc_codec.py`。不要把旧的 fake TX 用于模型实验。

可选的无 RF 服务验证：在 OFDM 项目根目录另开终端运行，**不要**加 `--demo-source`：

```bash
build/ofdm-workbench-service --simulate \
  --tx-endpoint tcp://127.0.0.1:5555 \
  --rx-endpoint tcp://127.0.0.1:5556
```

另一个终端先启动第 5 节的 `RX.py --max-bursts 1`，再用第 4 节的 `TX.py --input-bin` 发送。这个软件回环不证明 OTA。

## 4. Tx 主机发送

先根据 OFDM 项目 README 启动本机 TX 服务；频率、设备、天线、增益、授权与 RF 安全确认前，不要启动硬件 TX 服务或执行以下发送命令。**只生成 latent 文件不会发射。**

用已经生成的单图 latent 显式交给 `TX.py`：

```bash
python3 TX.py \
  --input-bin captures/tx_precheck_001/encoder/burst_00000001.cf32le \
  --start-id 1 --count 1 --endpoint tcp://127.0.0.1:5555
```

也可让编码器按图块批量调用同一个 `TX.py`，仍须明确 `--send`；脚本会先检查 TX 的文件接口，不直接打开 ZeroMQ：

```bash
python3 src/src/encode_prepared_npy.py \
  --dataset data/div2k_cv16_test/test.npy \
  --checkpoint models/p0_clean.pt \
  --output-dir captures/tx_send_001 \
  --max-images 1 --batch-size 1 --device cpu \
  --tx-script ./TX.py --endpoint tcp://127.0.0.1:5555 --send
```

`tx_invoked=true` 只证明 TX 客户端调用返回成功；ZeroMQ 入队不证明 SDR 发射或对端成功接收。两台主机各运行自己的 TX/RX 服务，客户端 endpoint 应连**本机**相应服务，空口连接由 SDR 完成；不要把两台主机的 `127.0.0.1` 当成同一地址。跨机 ZeroMQ 需要显式服务监听地址、防火墙和平台协议核对。

## 5. Rx 主机接收和解码

RX 服务运行后，仅启动**一个** `RX.py` 消费其结果，先接收 1 burst：

```bash
cd /path/to/div2k_txrx_bundle
python3 RX.py \
  --endpoint tcp://127.0.0.1:5556 \
  --save-dir captures/rx_001 \
  --npy-dir captures/rx_001/npy \
  --max-bursts 1
```

接收完成应有 `burst_00000001.json`、`.csi.cf32le`（64 个复数值）、`.data.cf32le`（预期 65,536 个复数值）和 `npy/data_*.npy`。检查 JSON 的 `burst_id`、`crc_ok`、`original_length`，并核对实收 burst 数；**不要在 CRC 失败或长度不匹配时把图片算作成功重建**。控制 anchor CRC 不能验证业务 IQ 每个字节。

确认捕获后，用同一个 checkpoint 运行 Decoder：

```bash
python3 src/src/decode_rx_npy.py captures/rx_001/npy \
  --checkpoint models/p0_clean.pt \
  --output-dir captures/reconstructed_001 \
  --max-records 1 --device cpu
```

输出 PNG 和 Decoder `manifest.json`。`RX.py` 批次 NPY 保留接收顺序，但 NPY 本身不携带 burst ID；多 burst 时需对照逐包 JSON。若漏包，不能仅凭文件序号把重建图对应回原图。接收程序已负责同步、CSI 与均衡；Decoder 不再做一次无线接收。

## 6. 当前验证状态

固定 `test.npy` 与来源 manifest 从训练服务器只读复制，SHA-256 分别与既有测试导出记录、服务器文件一致；本地读取确认数组为 `uint8 (50,3,256,256)`。

两份 checkpoint 都从训练服务器只读获取，重新计算的 SHA-256 分别与各自运行 manifest 一致；配套 trigger JSON 的哈希也一致。stronger_struct `best.pt` 是该次训练的最佳 epoch checkpoint，不代表已通过 clean-drop 门槛。

交付包源文件通过本地 Python 静态编译；`TX.py` 文件模式的 2 项无 RF 单元测试通过。P0 checkpoint 已从训练服务器只读复制并经 SHA-256 核对。当前本地 Python 缺少 `torch`，因此没有在本地用真实 checkpoint 完成 Encoder/Decoder 推理。xwt SSH 不可达，包**未部署**，也未做实机 Tx/Rx 或 OTA 验证。已有数字仿真结果属于此前流程，不能冒充本交付包的实测结果。完成服务器部署后，应先跑 1 图无 RF 仿真回环，记录数据与模型哈希、原生服务版本、CRC、CSI、latent 长度及重建图，再讨论实机参数。
