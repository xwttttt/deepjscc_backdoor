# Liquid OFDM Workbench

基于 liquid-dsp 的 Wi-Fi 4 风格 OFDM 收发系统。业务数据不是常规的 BPSK/QAM
数据，也不做业务信道编码；每个 burst 前面放置一个标准的 802.11a/g legacy
6 Mbps PHY anchor，用它完成同步、CFO、CSI 和 FCS 验证，随后发送项目自己的
复数 I/Q OFDM block。

当前主路径是 `ofdm-workbench-service + Tauri GUI`。`ofdm-tx-service`、
`ofdm-rx-service` 和 `ofdm-device-probe` 用于命令行诊断、文件回放和 SDR 设备管理。

## 1. 当前空口格式

| 参数 | 当前值 |
| --- | --- |
| 采样率 | 20 Msps |
| FFT / IFFT | 64 点，liquid-dsp |
| 循环前缀 | 16 个采样 |
| Wi-Fi 数据子载波 | 48 个，±1…±26，排除 ±7、±21 |
| 项目数据子载波 | 52 个，±1…±28，排除 ±7、±21 |
| Pilot | ±7、±21，802.11 127 周期极性序列 |
| Anchor | legacy 6 Mbps，BPSK，卷积码 rate 1/2 |
| 业务 I/Q | CF32 频域符号直传，不再做 QAM/FEC |

每个 burst 的时域布局为：

```text
L-STF/L-LTF (320) | SIGNAL (80) + 6 Mbps anchor data (1600)
                  | guard (32) | project HT-LTF (80)
                  | raw-I/Q OFDM data (N × 80) | tail guard (16)
```

因此 anchor 长度是 2000 个采样。Anchor 是合法的 broadcast data MPDU，PSDU
长度固定为 56 字节：24 字节 MAC header、8 字节 LLC/SNAP、20 字节控制 payload
和 4 字节 FCS。控制 payload 为：

```text
magic='OFDM' | version=2 | format=CF32 | FFT=64 | CP=16
burst_id(u32 LE) | complex_length(u32 LE) | ofdm_blocks(u16 LE)
data_bins=52 | pilot_scheme=legacy
```

业务输入长度为 `L` 个复数值时，项目数据符号数为 `ceil(L / 52)`，最后一个符号
补零。每个项目 OFDM 符号的输入顺序为有符号子载波 `-28 … -1, +1 … +28`（去除
pilot），TX 插入 pilot、IFFT 和 CP；RX 去 CP、FFT、CSI 均衡、pilot 相位跟踪后
恢复原始 `L` 个复数值。

RX 使用 anchor 的 STF/L-LTF 做包检测、定时和粗/细 CFO，并使用紧邻扩展数据的
project HT-LTF 估计全部 56 个 occupied-bin CSI。CSI 是 64 个 FFT bin，顺序
为 `[0, +1 … +31, -32 … -1]`；没有可靠解码的 burst 不会发布恢复数据。

Anchor 本身是完整的 Wi-Fi MAC 帧，使用实验 EtherType `0x88b5`；后面的 raw-I/Q
区域是项目自定义扩展格式，不能作为标准 Wi-Fi DATA 字段交给普通网卡解析。

## 2. 依赖和系统配置

原生后端需要：

- Linux、CMake >= 3.20、C++17 编译器、pthread；
- 系统安装的 liquid-dsp（头文件和 `libliquid` 可被 CMake 找到）；
- libzmq（优先提供 `libzmq.pc`，否则 CMake 会查找系统运行库和头文件）；
- SoapySDR 及对应设备插件。真实 SDR 还需要 UHD、Lime、bladeRF 等硬件驱动和
  设备的 udev 规则；
- nlohmann/json 3.11.3。首次 CMake 配置可能通过 FetchContent 下载它，因此首次
  配置需要网络，或提前准备 CMake 的下载缓存。

GUI 还需要 Node.js/npm、Rust/Cargo 和 Tauri 2；Python 收发和 DeepJSCC
重构脚本的依赖列在 `requirements-python.txt`。服务器现有的 `jscc` Conda
环境已经包含 PyTorch 时，无需重复安装。

### 一键安装依赖（Debian/Ubuntu）

```bash
sudo apt update && sudo apt install -y \
  build-essential \
  cmake \
  pkg-config \
  libzmq3-dev \
  libsoapysdr-dev \
  libglib2.0-dev \
  libpango1.0-dev \
  libcairo2-dev \
  libatk1.0-dev \
  libgdk-pixbuf-2.0-dev \
  libgtk-3-dev \
  libsoup-3.0-dev \
  libwebkit2gtk-4.1-dev
```

先检查系统是否能看到依赖和设备：

```bash
pkg-config --modversion SoapySDR
SoapySDRUtil --info
SoapySDRUtil --find
ldconfig -p | grep liquid
```

`SoapySDRUtil --find` 能发现设备不代表程序一定有权限打开它。若出现
`USB open failed: insufficient permissions`，安装/重新加载对应 udev 规则、重新插拔
设备，并以普通用户运行 GUI 和服务，不要用 `sudo` 启动应用。

## 3. 编译原生程序

在仓库根目录执行：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

生成的程序位于 `build/`：

```text
ofdm-workbench-service   # GUI 和统一收发服务使用
ofdm-device-probe        # SoapySDR 设备发现/能力探测
ofdm-tx-service          # TX 命令行诊断
ofdm-rx-service          # RX 命令行诊断
ofdm_tests               # 原生测试
```

## 4. GUI 构建和使用

原生程序必须先构建，GUI 的 `prepare-native` 会从 `../build/` 复制
`ofdm-workbench-service` 和 `ofdm-device-probe` 到 Tauri resource 目录。

```bash
cd gui
npm ci
npm run build:web
npm run prepare-native

# 生成 Debian 包；一键命令 npm run build 也会执行前面的两个步骤
npm run build

# 开发模式
npm run tauri dev

# 可选 AppImage
npm run build:appimage
```

GUI 的工作流：

1. 选择 `Simulation loopback` 或 `SoapySDR hardware`。
2. 硬件模式点击“重新扫描”，从设备列表选择设备，再点击 `CONNECT & PROBE`。
3. 在 TX/RX 文件夹页中选择当前方向；每次只显示一套面板。
4. 点击 `START TX` 或 `START RX`，停止时点击 `STOP`。

Simulation 模式下 GUI 默认附加 `--demo-source`，每 250 ms 生成确定性的 CF32
频域数据，因此不需要外部发送脚本就会持续显示 TX I/Q、RX loopback I/Q 和 CSI。
如果希望由网络输入驱动仿真，可以直接启动服务而不加 `--demo-source`，再运行
`TX.py`。

硬件模式会从 SoapySDR 设备能力中填充通道、天线和增益选项。GUI 的设备目录缓存
120 秒；`CONNECT & PROBE` 最长等待 300 秒，以容纳 B200/B210 在 USB 2.0 下首次
加载 UHD/FPGA 镜像。初始化期间不要拔出设备。GUI 的 TX/RX 页面分别启动单方向
stream；命令行 `--role duplex` 才会在同一个 SoapySDR handle 上同时启用 RX/TX。

## 5. 统一 Workbench 服务

`ofdm-workbench-service` 的 TX 入口默认绑定 `tcp://127.0.0.1:5555`，RX 结果
默认绑定 `tcp://127.0.0.1:5556`。

### Simulation

```bash
build/ofdm-workbench-service --simulate --demo-source \
  --tx-endpoint tcp://127.0.0.1:5555 \
  --rx-endpoint tcp://127.0.0.1:5556
```

不加 `--demo-source` 时，服务仍会运行，但 TX 只等待外部 ZeroMQ 输入：

```bash
build/ofdm-workbench-service --simulate \
  --tx-endpoint tcp://127.0.0.1:5555 \
  --rx-endpoint tcp://127.0.0.1:5556
```

### Hardware

`--device` 必须使用 SoapySDR 的 kwargs，建议直接复制
`ofdm-device-probe --list` 返回的 `device_args`，例如
`driver=uhd,serial=32FCCE4`：

```bash
# RX 和 TX 共用一个设备句柄
build/ofdm-workbench-service \
  --device 'driver=uhd,serial=32FCCE4' \
  --frequency 2437000000 --tx-gain 0 --rx-gain 20 --role duplex \
  --tx-endpoint tcp://127.0.0.1:5555 \
  --rx-endpoint tcp://127.0.0.1:5556

# 只接收或只发射
build/ofdm-workbench-service --device 'driver=uhd,serial=32FCCE4' \
  --frequency 2437000000 --rx-gain 20 --role rx
build/ofdm-workbench-service --device 'driver=uhd,serial=32FCCE4' \
  --frequency 2437000000 --tx-gain 0 --role tx
```

服务固定配置 20 Msps 采样率和 20 MHz 带宽；可配置中心频率、增益、通道和天线。
TX 数据到达后立即组帧并写入 SDR，不存在时间授权窗口。

## 6. ZeroMQ 数据接口

### TX 输入

服务绑定 `PULL` socket，发送端使用 `PUSH + connect`。每条消息必须是两段
multipart：

1. UTF-8 JSON。`burst_id` 可选；当前推荐格式：

   ```json
   {"version":1,"burst_id":1,"format":"cf32le","num_samples":1024}
   ```

2. little-endian CF32 二进制，按 `I0,Q0,I1,Q1,...` 排列，每个复数 8 字节。

服务检查长度、有限数和消息上限（8,388,608 个复数值）。JSON 里的其他字段不会
参与 PHY 解码。

### RX 输出

服务绑定 `PUSH` socket，接收端使用 `PULL + connect`。每个成功 burst 是三段
multipart：

1. UTF-8 JSON，例如：

   ```json
   {
     "version":1, "burst_id":1, "format":"cf32le",
     "original_length":1024, "num_data_symbols":20,
     "cfo_hz":12.3, "rssi":0.1, "snr_db":30.0,
     "noise_var":0.001, "csi_quality_db":20.0,
     "equalizer":"mmse", "crc_ok":true, "sync_offset":160
   }
   ```

2. 64 个 CF32 CSI（512 字节）；
3. 恢复后的 CF32 数据，长度为 `original_length`。

跨机器使用时，服务端不能绑定 `127.0.0.1`，应将 endpoint 改成例如
`tcp://*:5555`/`tcp://*:5556`，发送端和接收端再连接服务机 IP。网络和防火墙必须
允许相应端口。

## 7. Python 发送/接收脚本

```bash
python3 -m pip install -r requirements-python.txt

# 终端 A：启动仿真服务
build/ofdm-workbench-service --simulate \
  --tx-endpoint tcp://127.0.0.1:5555 \
  --rx-endpoint tcp://127.0.0.1:5556

# 终端 B：接收 CSI 和恢复数据；默认自动保存到 ./captures
python3 RX.py --endpoint tcp://127.0.0.1:5556

# 终端 C：发送 10 个、每个 1024 个复数值的确定性测试 burst
python3 TX.py --endpoint tcp://127.0.0.1:5555 --count 10 --interval 0.25
```

`TX.py` 发送的是项目 TX pipeline 的频域 CF32 输入，不是已经生成的时域 OFDM
波形。常用参数：`--samples`、`--amplitude`、`--interval`、`--count`、
`--start-id`。`RX.py` 校验三段 multipart 和 64 个 CSI bin，默认在 `./captures`
保存 JSON、`.csi.cf32le` 和 `.data.cf32le`。恢复的业务数据还会按 100 个 burst
自动批量写入 `./captures/npy`；收到 `Ctrl-C` 或达到
`--max-bursts` 时，不足 100 个的最后一批也会保存。等长 burst 生成形状为
`(records, complex_samples)` 的 `complex64` 数组；变长 burst 生成 object array，
读取时需要 `numpy.load(path, allow_pickle=True)`。需要改位置时仍可显式传入
`--save-dir` 和 `--npy-dir`。

### Decoder-only 图像重构

`decode_rx_npy.py` 只实例化原 DeepJSCC 的 Decoder，不导入 Encoder 或仿真信道。
它从 checkpoint 中提取 `decoder.*` 权重，自动推断通道数 `c` 和方形 latent
尺寸，并把 RX 均衡后保存的 complex64 NPY 批量重构为 RGB PNG：

```bash
/home/xwt/miniconda3/envs/jscc/bin/python decode_rx_npy.py
```

默认输入为 `captures/npy`，checkpoint 为
`models/target_free_c16_capped_psnr_drop_20260902_80ksteps_retry1/best.pt`，输出为
`captures/reconstructed`。也可指定单个文件做 smoke test：

```bash
/home/xwt/miniconda3/envs/jscc/bin/python decode_rx_npy.py \
  captures/npy/data_example_batch_000001.npy \
  --output-dir captures/reconstructed_smoke --max-records 1
```

输出目录包含每条记录对应的 PNG 和 `manifest.json`；manifest 记录 checkpoint
SHA-256、输入文件快照、`c/h/w`、设备和输出清单。已有同名 PNG 时脚本默认拒绝
覆盖，需要明确传入 `--overwrite`。

## 8. 命令行文件回放和 SDR 诊断

### TX 文件输出

```bash
# 只验证组帧，不输出到 SDR
build/ofdm-tx-service --endpoint tcp://127.0.0.1:5555 --dry-run

# 将收到的一个 burst 写成时域 CF32 文件
build/ofdm-tx-service --endpoint tcp://127.0.0.1:5555 \
  --file-out /tmp/burst.cf32 --once
python3 TX.py --endpoint tcp://127.0.0.1:5555 --count 1
```

`ofdm-tx-service` 的 `--device`、`--file-out`、`--dry-run` 三者必须且只能选择
一个。真实发射示例：

```bash
build/ofdm-tx-service \
  --device 'driver=uhd,serial=32FCCE4' \
  --frequency 2437000000 --gain 0 --antenna TX/RX --channel 0
```

### RX 文件输入

```bash
build/ofdm-rx-service --file-in /tmp/burst.cf32 --no-zmq --once
```

也可以发布到 RX ZeroMQ 端口：

```bash
build/ofdm-rx-service --file-in /tmp/burst.cf32 \
  --endpoint tcp://127.0.0.1:5556 --equalizer mmse --once
python3 RX.py --endpoint tcp://127.0.0.1:5556 --max-bursts 1
```

真实接收示例：

```bash
build/ofdm-rx-service \
  --device 'driver=uhd,serial=32FCCE4' \
  --frequency 2437000000 --gain 20 --antenna RX2 --channel 0 \
  --endpoint tcp://127.0.0.1:5556 --equalizer mmse
```

`--equalizer zf` 可用于诊断；默认使用 MMSE。文件是裸 little-endian CF32，没有
文件头，必须由本项目的 TX 服务生成或提供相同格式的 IQ 捕获。

## 9. 设备发现和权限排查

```bash
build/ofdm-device-probe --list
build/ofdm-device-probe --factories
build/ofdm-device-probe --driver uhd
build/ofdm-device-probe --device 'driver=uhd,serial=32FCCE4'
```

`--list` 是快速枚举，不打开设备；GUI 的 `CONNECT & PROBE` 会进一步读取通道、
天线、采样率、频率、带宽、增益和 full-duplex 能力。设备列表中的 `device_args`
是可以直接传给服务 `--device` 的 SoapySDR 参数。

常见错误：

- `native SDR device probe not found`：先执行根目录 CMake 构建，再执行
  `cd gui && npm run prepare-native`；打包安装时确认 `resources/bin/` 中包含两个
  native sidecar。
- `SoapySDR::Device::make() no match`：不要把 GUI 标签/name 当成设备参数；使用
  `ofdm-device-probe --list` 返回的完整 `device_args`，并确认 SoapySDR 硬件插件已
  安装且 `SoapySDRUtil --find` 能看到同一设备。
- `USB open failed: insufficient permissions`：修复 udev 规则并重新插拔设备，
  不要用 sudo 运行 GUI。
- B200/B210 在 USB 2.0 上首次打开可能需要较长时间，并可能出现 RX overflow；
  首次加载镜像时等待设备完成枚举，持续高速采集建议使用 USB 3.0。
- `TX send timeout`：确认服务已经启动、脚本 endpoint 与服务 TX endpoint 完全一致，
  且服务端口没有被其他进程占用。
- `telemetry header read failed`：GUI sidecar 已退出或被杀死。先检查 sidecar
  stderr，再确认 `ofdm-workbench-service` 与 GUI 版本来自同一次构建。

## 10. 测试和限制

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

测试覆盖 FFT/IFFT、子载波和 pilot、前导、CSI、CFO、FCS、理想环回、AWGN、多径/延迟、
流式接收、多 burst 和数据长度裁剪。

当前限制：

- 后置 raw-I/Q 不是标准 802.11n MAC/PHY，不能由普通 Wi-Fi 网卡解码；
- CSI 由 burst 开头的训练序列获得，长 burst 中不会周期性插入 LTF；
- 业务 CF32 不做调制、信道编码、交织或扰码；
- 20 Msps 对 USB、主机调度和 SDR 缓冲有要求；OTA/线缆直连必须使用合适衰减器，
  避免损坏接收机。

## 11. 目录

```text
src/dsp/       FFT、前导、子载波、pilot、旧兼容控制头
src/wifi/      标准 6 Mbps BPSK 1/2 legacy control anchor
src/tx/        anchor + raw-I/Q burst 组装
src/rx/        同步、CFO、CSI、均衡和 raw-I/Q 恢复
src/radio/    SoapySDR 设备发现和 RX/TX stream
src/interface/ZeroMQ 和 GUI telemetry
src/loopback/ 软件环回
gui/           React/Vite/Tauri 前端
TX.py/RX.py   ZeroMQ 测试收发脚本
tests/         C++ 原生测试
```
